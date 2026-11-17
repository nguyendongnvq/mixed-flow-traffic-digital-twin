from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ("accident", "congestion_spike", "zigzag", "sudden_stop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach per-node multi-label anomaly targets to mixed forecast data."
    )
    parser.add_argument(
        "--forecast",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "processed"
        / "mixed"
        / "forecast_mixed_node_level_rebuild.csv",
    )
    parser.add_argument(
        "--window-labels",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "raw"
        / "sumo_data"
        / "anomaly_window_labels_mixed.csv",
    )
    parser.add_argument(
        "--sensor-mapping",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "processed"
        / "references"
        / "sensor_node_mapping.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "processed"
        / "mixed"
        / "forecast_mixed_node_level_with_labels.csv",
    )
    return parser.parse_args()


def split_ids(raw: object) -> set[str]:
    if pd.isna(raw):
        return set()
    return {item.strip() for item in str(raw).split(";") if item.strip()}


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    forecast = pd.read_csv(args.forecast)
    labels = pd.read_csv(args.window_labels)
    mapping = pd.read_csv(args.sensor_mapping).drop_duplicates()

    required_forecast = {"begin", "node_id"}
    required_labels = {"window_start"} | {
        f"active_edge_ids_{name}" for name in CLASS_NAMES
    }
    required_mapping = {"edge_id", "node_id"}
    for path, frame, required in (
        (args.forecast, forecast, required_forecast),
        (args.window_labels, labels, required_labels),
        (args.sensor_mapping, mapping, required_mapping),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

    edge_to_nodes = (
        mapping.groupby("edge_id")["node_id"]
        .apply(lambda values: set(values.astype(str)))
        .to_dict()
    )
    class_nodes_by_window: dict[str, dict[float, set[str]]] = {
        name: {} for name in CLASS_NAMES
    }
    for row in labels.itertuples(index=False):
        window_start = float(row.window_start)
        for class_name in CLASS_NAMES:
            nodes: set[str] = set()
            for edge_id in split_ids(
                getattr(row, f"active_edge_ids_{class_name}")
            ):
                nodes.update(edge_to_nodes.get(edge_id, set()))
            class_nodes_by_window[class_name][window_start] = nodes

    output = forecast.copy()
    output["begin"] = pd.to_numeric(output["begin"], errors="raise")
    output["node_id"] = output["node_id"].astype(str)
    for class_name in CLASS_NAMES:
        nodes_by_window = class_nodes_by_window[class_name]
        output[class_name] = [
            int(node_id in nodes_by_window.get(float(begin), set()))
            for begin, node_id in zip(output["begin"], output["node_id"])
        ]
    output["is_anomaly"] = output[list(CLASS_NAMES)].max(axis=1).astype(int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    positive_counts = {
        name: int(output[name].sum()) for name in CLASS_NAMES
    }
    print(f"output={args.output}")
    print(f"rows={len(output)} nodes={output['node_id'].nunique()}")
    print(f"positive_node_windows={positive_counts}")


if __name__ == "__main__":
    main()
