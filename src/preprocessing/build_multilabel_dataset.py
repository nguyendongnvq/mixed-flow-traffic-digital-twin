from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.data import clean_df, df_to_tensor


CLASS_NAMES = ("accident", "congestion_spike", "zigzag", "sudden_stop")
DEFAULT_CONFIG = Path("configs/dmfgcrn_multilabel.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DMFGCRN multi-label window dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-windows-per-source", type=int)
    return parser.parse_args()


def add_time_features(
    windows: np.ndarray,
    window_begins: np.ndarray,
    steps_per_day: int,
) -> np.ndarray:
    lag, num_nodes = windows.shape[1], windows.shape[2]
    starts = window_begins[:, None] - np.arange(lag - 1, -1, -1)[None, :] * 300
    time_of_day = ((starts % 86400) / 86400).astype(np.float32)
    day_index = np.floor(starts / 86400).astype(np.int64)
    day_of_week = (((day_index + 2) % 7) / 7.0).astype(np.float32)
    time_of_day = np.repeat(time_of_day[:, :, None, None], num_nodes, axis=2)
    day_of_week = np.repeat(day_of_week[:, :, None, None], num_nodes, axis=2)
    return np.concatenate([windows, time_of_day, day_of_week], axis=-1)


def split_ranges(length: int, train_ratio: float, val_ratio: float) -> dict[str, tuple[int, int]]:
    train_end = int(length * train_ratio)
    val_end = int(length * (train_ratio + val_ratio))
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, length),
    }


def load_source(
    source: dict[str, Any],
    *,
    nodes: list[str],
    feature_cols: list[str],
    class_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = clean_df(pd.read_csv(source["path"]))
    tensor, timesteps = df_to_tensor(
        df,
        feature_cols,
        nodes,
        time_col="begin",
        node_col="node_id",
    )
    labels = np.zeros((len(timesteps), len(nodes), len(class_names)), dtype=np.float32)
    time_to_idx = {value: idx for idx, value in enumerate(timesteps)}
    node_to_idx = {value: idx for idx, value in enumerate(nodes)}

    if source.get("class_name"):
        class_idx = class_names.index(str(source["class_name"]))
        for row in df.loc[df["is_anomaly"] == 1, ["begin", "node_id"]].itertuples(index=False):
            labels[time_to_idx[row.begin], node_to_idx[row.node_id], class_idx] = 1.0
    elif bool(source.get("multilabel", False)):
        for class_idx, class_name in enumerate(class_names):
            if class_name not in df.columns:
                raise ValueError(f"Mixed source {source['path']} is missing label column {class_name}.")
            positive = df.loc[df[class_name] == 1, ["begin", "node_id"]]
            for row in positive.itertuples(index=False):
                labels[time_to_idx[row.begin], node_to_idx[row.node_id], class_idx] = 1.0

    return tensor, labels, np.asarray(timesteps, dtype=np.float64)


def build_windows_for_split(
    data: np.ndarray,
    labels: np.ndarray,
    timesteps: np.ndarray,
    *,
    start: int,
    end: int,
    lag: int,
    input_indices: list[int],
    output_indices: list[int],
    steps_per_day: int,
    max_windows: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_data = data[start:end]
    split_labels = labels[start:end]
    split_times = timesteps[start:end]
    sample_count = len(split_data) - lag
    if sample_count <= 0:
        empty_x = np.empty((0, lag, data.shape[1], len(input_indices) + 2), dtype=np.float32)
        empty_node_y = np.empty((0, data.shape[1], labels.shape[2]), dtype=np.float32)
        empty_global_y = np.empty((0, labels.shape[2]), dtype=np.float32)
        empty_target = np.empty((0, data.shape[1], len(output_indices)), dtype=np.float32)
        return empty_x, empty_node_y, empty_global_y, empty_target, np.empty(0)

    indices = np.arange(sample_count)
    if max_windows is not None and len(indices) > max_windows:
        indices = np.linspace(0, len(indices) - 1, max_windows, dtype=int)

    x = np.stack([split_data[i : i + lag, :, input_indices] for i in indices]).astype(np.float32)
    target_indices = indices + lag
    node_y = split_labels[target_indices].astype(np.float32)
    global_y = node_y.max(axis=1)
    targets = split_data[target_indices][:, :, output_indices].astype(np.float32)
    begins = split_times[target_indices]
    x = add_time_features(x, begins, steps_per_day)
    return x, node_y, global_y, targets, begins


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_path = args.output or Path(config["dataset_output"])
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_path}")

    nodes = list(config["nodes"])
    feature_cols = list(config["feature_cols"])
    input_indices = [feature_cols.index(name) for name in config["input_features"]]
    output_indices = [feature_cols.index(name) for name in config["output_features"]]
    lag = int(config["lag"])
    train_ratio = float(config["train_ratio"])
    val_ratio = float(config["val_ratio"])
    steps_per_day = int(config["steps_per_day"])

    collected: dict[str, dict[str, list[np.ndarray]]] = {
        split: {
            key: []
            for key in (
                "x",
                "node_y",
                "global_y",
                "target",
                "begin",
                "source",
                "traffic_scale",
            )
        }
        for split in ("train", "val", "test")
    }

    for source_idx, source in enumerate(config["sources"]):
        data, labels, timesteps = load_source(
            source,
            nodes=nodes,
            feature_cols=feature_cols,
            class_names=CLASS_NAMES,
        )
        for split, (start, end) in split_ranges(len(data), train_ratio, val_ratio).items():
            x, node_y, global_y, target, begins = build_windows_for_split(
                data,
                labels,
                timesteps,
                start=start,
                end=end,
                lag=lag,
                input_indices=input_indices,
                output_indices=output_indices,
                steps_per_day=steps_per_day,
                max_windows=args.max_windows_per_source,
            )
            if len(x) == 0:
                continue
            collected[split]["x"].append(x)
            collected[split]["node_y"].append(node_y)
            collected[split]["global_y"].append(global_y)
            collected[split]["target"].append(target)
            collected[split]["begin"].append(begins)
            collected[split]["source"].append(
                np.full(len(x), source_idx, dtype=np.int32)
            )
            collected[split]["traffic_scale"].append(
                np.full(
                    len(x),
                    float(source.get("traffic_scale", 1.0)),
                    dtype=np.float32,
                )
            )

    payload: dict[str, np.ndarray] = {
        "class_names": np.asarray(CLASS_NAMES),
        "nodes": np.asarray(nodes),
        "feature_cols": np.asarray(feature_cols),
        "input_features": np.asarray(config["input_features"]),
        "output_features": np.asarray(config["output_features"]),
        "source_names": np.asarray(
            [str(source.get("name", index)) for index, source in enumerate(config["sources"])]
        ),
    }
    for split, split_data in collected.items():
        for key, arrays in split_data.items():
            if not arrays:
                raise ValueError(f"No arrays collected for split={split}, key={key}.")
            payload[f"{split}_{key}"] = np.concatenate(arrays, axis=0)

    normal_mask = payload["train_global_y"].sum(axis=1) == 0
    continuous = payload["train_x"][normal_mask, ..., : len(input_indices)]
    scaler_mean = continuous.mean(axis=(0, 1, 2))
    scaler_scale = continuous.std(axis=(0, 1, 2))
    scaler_scale[scaler_scale == 0] = 1
    payload["scaler_mean"] = scaler_mean.astype(np.float32)
    payload["scaler_scale"] = scaler_scale.astype(np.float32)

    for split in ("train", "val", "test"):
        x = payload[f"{split}_x"]
        x[..., : len(input_indices)] = (
            x[..., : len(input_indices)] - scaler_mean
        ) / scaler_scale
        target = payload[f"{split}_target"]
        target_mean = scaler_mean[: len(output_indices)]
        target_scale = scaler_scale[: len(output_indices)]
        payload[f"{split}_target"] = (target - target_mean) / target_scale

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    print(f"dataset={output_path}")
    for split in ("train", "val", "test"):
        print(
            f"{split}_samples={len(payload[f'{split}_x'])} "
            f"positives={payload[f'{split}_global_y'].sum(axis=0).astype(int).tolist()}"
        )


if __name__ == "__main__":
    main()
