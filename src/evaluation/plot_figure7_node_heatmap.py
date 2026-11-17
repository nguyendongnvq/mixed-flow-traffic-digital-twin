from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.evaluation.scoring import compute_weighted_node_max_scores, predict_reconstruction
from src.models.model import LSTMAutoencoder
from src.preprocessing.data import clean_df, df_to_tensor
from src.preprocessing.preprocess import flatten_node_feature, make_loader, make_windows


DEFAULT_CHECKPOINT = Path(
    "models/outputs_weighted_p85_accident/lstm_autoencoder_weighted_node_max_p85.pt"
)
DEFAULT_ACCIDENT_DATA = Path(
    "datasets/processed/accident/forecast_accident_node_level_accident_only.csv"
)
DEFAULT_WINDOW_LABELS = Path(
    "datasets/processed/accident/anomaly_window_labels_accident.csv"
)
DEFAULT_SENSOR_MAPPING = Path("datasets/processed/references/sensor_node_mapping.csv")
DEFAULT_RESULT_CSV = Path("models/outputs_weighted_p85_accident/accident_window_result.csv")
DEFAULT_OUTPUT_PNG = Path(
    "models/outputs_weighted_p85_accident/figure7_node_level_anomaly_heatmap.png"
)
DEFAULT_OUTPUT_CSV = Path(
    "models/outputs_weighted_p85_accident/figure7_node_level_anomaly_heatmap.csv"
)

NETWORK_LAYOUT = {
    "30_4_htp": (0.0, 2.0),
    "pdl_htp": (1.0, 2.0),
    "xoviet_htp": (2.0, 2.0),
    "30_4_lnh": (0.0, 1.0),
    "pdl_lnh": (1.0, 1.0),
    "xoviet_lnh": (2.0, 1.0),
    "30_4_ltn": (0.0, 0.0),
    "pdl_ltn": (1.0, 0.0),
    "xoviet_ltn": (2.0, 0.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Figure 7: intersection-level anomaly heatmap for a representative accident."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--accident-data", type=Path, default=DEFAULT_ACCIDENT_DATA)
    parser.add_argument("--window-labels", type=Path, default=DEFAULT_WINDOW_LABELS)
    parser.add_argument("--sensor-mapping", type=Path, default=DEFAULT_SENSOR_MAPPING)
    parser.add_argument("--result-csv", type=Path, default=DEFAULT_RESULT_CSV)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def intersection_id(node_id: str) -> str:
    marker = "_lane_"
    if marker not in node_id:
        raise ValueError(f"Cannot derive intersection from node_id={node_id!r}.")
    return node_id.split(marker, maxsplit=1)[0]


def split_active_edges(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [edge.strip() for edge in str(value).split(";") if edge.strip()]


def load_node_scores(
    checkpoint_path: Path,
    accident_data_path: Path,
) -> tuple[dict[str, object], list[float], np.ndarray, np.ndarray]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    nodes = list(payload["nodes"])
    feature_cols = list(payload["feature_cols"])
    seq_size = int(payload["seq_size"])
    n_nodes = int(payload["N"])
    n_features = int(payload["F"])

    accident_df = clean_df(pd.read_csv(accident_data_path))
    x_accident, accident_timesteps = df_to_tensor(
        accident_df,
        feature_cols,
        nodes,
        time_col="begin",
        node_col="node_id",
    )

    scaler_mean = np.asarray(payload["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(payload["scaler_scale"], dtype=np.float32)
    x_scaled = ((x_accident - scaler_mean) / scaler_scale).astype(np.float32)
    accident_windows = make_windows(flatten_node_feature(x_scaled), seq_size)
    accident_loader = make_loader(accident_windows, batch_size=64, shuffle=False)

    model = LSTMAutoencoder(
        input_dim=int(payload["input_dim"]),
        hidden_dim=128,
        latent_dim=64,
        num_layers=1,
        dropout=0.2,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    batch_x, reconstruction = predict_reconstruction(
        model,
        accident_loader,
        torch.device("cpu"),
    )
    window_scores, node_scores = compute_weighted_node_max_scores(
        batch_x,
        reconstruction,
        n_nodes=n_nodes,
        n_features=n_features,
        seq_size=seq_size,
        feature_weights=np.asarray(payload["feature_weights"], dtype=np.float32),
    )
    window_begins = accident_timesteps[seq_size - 1 :]
    return payload, window_begins, window_scores, node_scores


def build_intersection_scores(nodes: list[str], node_scores: np.ndarray) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "node_id": nodes,
            "intersection_id": [intersection_id(node) for node in nodes],
            "node_score": node_scores,
        }
    )
    intersection_scores = (
        rows.groupby("intersection_id", as_index=False)["node_score"]
        .max()
        .rename(columns={"node_score": "intersection_score"})
    )
    return intersection_scores


def select_representative_window(
    *,
    nodes: list[str],
    window_begins: list[float],
    window_scores: np.ndarray,
    node_scores: np.ndarray,
    threshold: float,
    window_labels_path: Path,
    sensor_mapping_path: Path,
) -> tuple[int, str, str, str]:
    labels = pd.read_csv(window_labels_path)
    labels["window_start"] = pd.to_numeric(labels["window_start"])
    label_by_begin = labels.set_index("window_start")

    mapping = pd.read_csv(sensor_mapping_path).drop_duplicates(["edge_id", "node_id"])
    edge_to_node = dict(zip(mapping["edge_id"], mapping["node_id"]))

    localized_candidates: list[tuple[float, int, str, str]] = []
    fallback_candidates: list[tuple[float, float, int, str, str]] = []

    for window_idx, begin in enumerate(window_begins):
        if window_scores[window_idx] <= threshold or begin not in label_by_begin.index:
            continue

        label_row = label_by_begin.loc[begin]
        if isinstance(label_row, pd.DataFrame):
            label_row = label_row.iloc[0]
        if int(label_row["is_anomaly_window"]) != 1:
            continue

        active_edges = split_active_edges(label_row["active_edge_ids"])
        affected_nodes = [edge_to_node[edge] for edge in active_edges if edge in edge_to_node]
        affected_intersections = sorted({intersection_id(node) for node in affected_nodes})
        if len(affected_intersections) != 1:
            continue

        affected_intersection = affected_intersections[0]
        intersection_scores = build_intersection_scores(nodes, node_scores[window_idx])
        score_map = dict(
            zip(
                intersection_scores["intersection_id"],
                intersection_scores["intersection_score"],
            )
        )
        if affected_intersection not in score_map:
            continue

        affected_score = float(score_map[affected_intersection])
        other_scores = [
            score for name, score in score_map.items() if name != affected_intersection
        ]
        contrast = affected_score / max(float(np.median(other_scores)), np.finfo(float).eps)
        top_intersection = max(score_map, key=score_map.get)
        event_id = str(label_row["active_event_ids"])

        fallback_candidates.append(
            (contrast, float(window_scores[window_idx]), window_idx, affected_intersection, event_id)
        )
        if top_intersection == affected_intersection:
            localized_candidates.append(
                (float(window_scores[window_idx]), window_idx, affected_intersection, event_id)
            )

    if localized_candidates:
        _, window_idx, affected_intersection, event_id = max(localized_candidates)
        return window_idx, affected_intersection, event_id, "highest localized detected score"
    if fallback_candidates:
        _, _, window_idx, affected_intersection, event_id = max(fallback_candidates)
        return window_idx, affected_intersection, event_id, "highest affected-node contrast"
    raise ValueError("No detected accident window could be mapped to one affected intersection.")


def make_export_table(
    *,
    nodes: list[str],
    selected_node_scores: np.ndarray,
    affected_intersection: str,
    begin: float,
    event_id: str,
    selection_rule: str,
) -> pd.DataFrame:
    intersection_scores = build_intersection_scores(nodes, selected_node_scores)
    missing_layout = sorted(set(intersection_scores["intersection_id"]) - set(NETWORK_LAYOUT))
    if missing_layout:
        raise ValueError(f"Missing network layout for intersections: {missing_layout}")

    max_score = float(intersection_scores["intersection_score"].max())
    intersection_scores["normalized_score"] = (
        intersection_scores["intersection_score"] / max_score
    )
    intersection_scores["x"] = intersection_scores["intersection_id"].map(
        lambda name: NETWORK_LAYOUT[name][0]
    )
    intersection_scores["y"] = intersection_scores["intersection_id"].map(
        lambda name: NETWORK_LAYOUT[name][1]
    )
    intersection_scores["is_accident_intersection"] = (
        intersection_scores["intersection_id"] == affected_intersection
    )
    intersection_scores["window_begin"] = begin
    intersection_scores["event_id"] = event_id
    intersection_scores["selection_rule"] = selection_rule
    return intersection_scores.sort_values(["y", "x"], ascending=[False, True])


def plot_heatmap(export_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 7.2))

    for y in (0.0, 1.0, 2.0):
        ax.plot([0.0, 2.0], [y, y], color="#b8b8b8", linewidth=3, zorder=0)
    for x in (0.0, 1.0, 2.0):
        ax.plot([x, x], [0.0, 2.0], color="#b8b8b8", linewidth=3, zorder=0)

    scatter = ax.scatter(
        export_df["x"],
        export_df["y"],
        c=export_df["normalized_score"],
        s=900,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        edgecolors="#4a4a4a",
        linewidths=1.2,
        zorder=2,
    )

    accident_row = export_df[export_df["is_accident_intersection"]].iloc[0]
    ax.scatter(
        [accident_row["x"] + 0.16],
        [accident_row["y"] + 0.16],
        marker="*",
        s=300,
        facecolors="white",
        edgecolors="black",
        linewidths=1.8,
        zorder=3,
        label="Accident intersection",
    )

    for row in export_df.itertuples(index=False):
        label = f"{row.intersection_id.replace('_', ' ').upper()} INTERSECTION"
        ax.text(
            row.x,
            row.y - 0.18,
            label,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            row.x,
            row.y + 0.02,
            f"{row.normalized_score:.2f}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_aspect("equal")
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(-0.45, 2.45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", frameon=True, prop={"weight": "bold"})
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_label("Normalized weighted node-max score", fontweight="bold")
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def ensure_writable(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing output(s): {joined}")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_writable([args.output_png, args.output_csv], args.overwrite)

    payload, window_begins, window_scores, node_scores = load_node_scores(
        args.checkpoint,
        args.accident_data,
    )
    nodes = list(payload["nodes"])
    threshold = float(payload["threshold"])

    result_df = pd.read_csv(args.result_csv)
    if len(result_df) != len(window_scores):
        raise ValueError("Recomputed window count does not match accident_window_result.csv.")
    if not np.allclose(result_df["begin"], window_begins):
        raise ValueError("Recomputed window timestamps do not match accident_window_result.csv.")
    if not np.allclose(result_df["anomaly_score"], window_scores, rtol=1e-5, atol=1e-7):
        raise ValueError("Recomputed anomaly scores do not match accident_window_result.csv.")

    window_idx, affected_intersection, event_id, selection_rule = select_representative_window(
        nodes=nodes,
        window_begins=window_begins,
        window_scores=window_scores,
        node_scores=node_scores,
        threshold=threshold,
        window_labels_path=args.window_labels,
        sensor_mapping_path=args.sensor_mapping,
    )
    export_df = make_export_table(
        nodes=nodes,
        selected_node_scores=node_scores[window_idx],
        affected_intersection=affected_intersection,
        begin=float(window_begins[window_idx]),
        event_id=event_id,
        selection_rule=selection_rule,
    )

    intersection_max = float(export_df["intersection_score"].max())
    if not np.isclose(intersection_max, window_scores[window_idx], rtol=1e-6, atol=1e-8):
        raise ValueError("Intersection maximum does not match the weighted node-max score.")
    if not bool(
        export_df.loc[
            export_df["intersection_score"].idxmax(),
            "is_accident_intersection",
        ]
    ):
        raise ValueError("Selected accident intersection is not the spatial score maximum.")

    export_df.to_csv(args.output_csv, index=False)
    plot_heatmap(export_df, args.output_png)

    affected_score = float(
        export_df.loc[
            export_df["is_accident_intersection"],
            "intersection_score",
        ].iloc[0]
    )
    other_median = float(
        export_df.loc[
            ~export_df["is_accident_intersection"],
            "intersection_score",
        ].median()
    )
    print(f"output_png={args.output_png}")
    print(f"output_csv={args.output_csv}")
    print(f"window_idx={window_idx}")
    print(f"window_begin={window_begins[window_idx]}")
    print(f"event_id={event_id}")
    print(f"accident_intersection={affected_intersection}")
    print(f"selection_rule={selection_rule}")
    print(f"window_score={window_scores[window_idx]:.9f}")
    print(f"affected_score={affected_score:.9f}")
    print(f"other_intersection_median={other_median:.9f}")


if __name__ == "__main__":
    main()
