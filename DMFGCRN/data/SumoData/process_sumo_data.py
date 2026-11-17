import os

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "forecast_baseline_node_level.csv")
OUTPUT_NPZ = os.path.join(BASE_DIR, "SUMO.npz")
NODE_IDS_CSV = os.path.join(BASE_DIR, "SUMO_node_ids.csv")

FEATURE_COLUMNS = [
    "flow_count_5m",
    "e1_occupancy",
    "speed",
    "is_missing_speed",
]


def validate_source(df):
    required = {"begin", "end", "node_id", "flow_count_5m", "e1_occupancy", "speed"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    duplicate_count = df.duplicated(["begin", "node_id"]).sum()
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate (begin, node_id) rows")

    interval = df["end"] - df["begin"]
    if interval.nunique() != 1 or int(interval.iloc[0]) != 300:
        raise ValueError(f"Expected 300-second intervals, got {interval.unique()[:10]}")

    begins = np.sort(df["begin"].unique())
    begin_steps = np.diff(begins)
    if len(begin_steps) and (begin_steps != 300).any():
        raise ValueError("Begin timestamps are not spaced by 300 seconds")

    node_ids = sorted(df["node_id"].unique())
    expected_rows = len(begins) * len(node_ids)
    if len(df) != expected_rows:
        raise ValueError(f"Expected full grid with {expected_rows} rows, got {len(df)}")

    timestep_counts = df.groupby("begin")["node_id"].nunique()
    if not (timestep_counts == len(node_ids)).all():
        raise ValueError("At least one timestep does not contain every node")

    node_counts = df.groupby("node_id")["begin"].nunique()
    if not (node_counts == len(begins)).all():
        raise ValueError("At least one node does not contain every timestep")

    return begins, node_ids


def process():
    df = pd.read_csv(INPUT_CSV)
    begins, node_ids = validate_source(df)

    for column in ["flow_count_5m", "e1_occupancy"]:
        if df[column].isna().any():
            raise ValueError(f"{column} contains missing values")

    df["is_missing_speed"] = df["speed"].isna().astype(np.float32)
    positive_flow_missing_speed = df["speed"].isna() & (df["flow_count_5m"] > 0)
    if positive_flow_missing_speed.any():
        raise ValueError("Found missing speed rows with positive flow_count_5m")

    df["speed"] = df["speed"].fillna(0.0)

    arrays = []
    for feature in FEATURE_COLUMNS:
        pivot = df.pivot(index="begin", columns="node_id", values=feature)
        pivot = pivot.reindex(index=begins, columns=node_ids)
        values = pivot.to_numpy(dtype=np.float32)
        arrays.append(values)

    data = np.stack(arrays, axis=-1)

    if not np.isfinite(data).all():
        raise ValueError("Processed data contains NaN or infinite values")

    np.savez_compressed(
        OUTPUT_NPZ,
        data=data,
        feature_names=np.array(FEATURE_COLUMNS),
        begin=np.array(begins, dtype=np.int64),
        node_ids=np.array(node_ids),
    )
    pd.DataFrame({"node_index": range(len(node_ids)), "node_id": node_ids}).to_csv(
        NODE_IDS_CSV, index=False
    )

    print(f"Saved {OUTPUT_NPZ}")
    print(f"Saved {NODE_IDS_CSV}")
    print(f"data shape: {data.shape}")
    print(f"is_missing_speed count: {int(data[..., 3].sum())}")


if __name__ == "__main__":
    process()
