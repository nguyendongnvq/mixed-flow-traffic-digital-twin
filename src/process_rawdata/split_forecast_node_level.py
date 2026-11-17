"""Split forecast node-level datasets into normal-only and anomaly-only variants.

Design goals:
- Flexible for accident/congestion_spike/zigzag (and baseline if needed).
- Never overwrite existing artifacts by default (uses suffix `_v2`).
- Validate split integrity and optionally compare with existing legacy split files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(LOCAL_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PROJECT_ROOT))


DEFAULT_MODES = ["accident", "congestion_spike", "zigzag", "sudden_stop"]

MODE_TO_DIR: dict[str, str] = {
    "baseline": "normal",
    "accident": "accident",
    "congestion_spike": "congestion",
    "zigzag": "zigzag",
    "sudden_stop": "sudden_stop",
}

MODE_TO_BASE_FORECAST: dict[str, str] = {
    "baseline": "forecast_baseline_node_level.csv",
    "accident": "forecast_accident_node_level.csv",
    "congestion_spike": "forecast_congestion_spike_node_level.csv",
    "zigzag": "forecast_zigzag_node_level.csv",
    "sudden_stop": "forecast_sudden_stop_node_level.csv",
}

MODE_TO_ANOMALY_TAG: dict[str, str] = {
    "baseline": "anomaly",
    "accident": "accident",
    "congestion_spike": "congestion_spike",
    "zigzag": "zigzag",
    "sudden_stop": "sudden_stop",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split forecast node-level CSV into normal_only and anomaly_only files."
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(DEFAULT_MODES),
        help="Comma-separated modes (default: accident,congestion_spike,zigzag).",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=LOCAL_PROJECT_ROOT / "datasets" / "processed",
        help="Processed datasets root folder.",
    )
    parser.add_argument(
        "--sensor-node-mapping",
        type=Path,
        default=LOCAL_PROJECT_ROOT / "datasets" / "processed" / "references" / "sensor_node_mapping.csv",
        help="Path to sensor_node_mapping.csv used to map active_edge_ids to node_id for node-level labeling.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_v2",
        help="Suffix appended before .csv for new output files (default: _v2).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing inputs or ambiguous labeling sources.",
    )
    return parser.parse_args()


def parse_modes(raw: str) -> list[str]:
    modes = [m.strip() for m in raw.split(",") if m.strip()]
    if not modes:
        raise ValueError("No mode specified in --modes.")
    return modes


def build_mode_paths(processed_root: Path, mode: str, suffix: str) -> dict[str, Path]:
    if mode not in MODE_TO_DIR or mode not in MODE_TO_BASE_FORECAST:
        raise ValueError(f"Unsupported mode: {mode}")
    mode_dir = processed_root / MODE_TO_DIR[mode]
    base_name = MODE_TO_BASE_FORECAST[mode]
    base_path = mode_dir / base_name
    with_labels = mode_dir / base_name.replace(".csv", "_with_labels.csv")
    window_labels = mode_dir / f"anomaly_window_labels_{mode}.csv"
    anomaly_tag = MODE_TO_ANOMALY_TAG.get(mode, mode)
    normal_out = mode_dir / base_name.replace(".csv", f"_normal_only{suffix}.csv")
    anomaly_out = mode_dir / base_name.replace(".csv", f"_{anomaly_tag}_only{suffix}.csv")

    legacy_normal = mode_dir / base_name.replace(".csv", "_normal_only.csv")
    legacy_anomaly = mode_dir / base_name.replace(".csv", f"_{anomaly_tag}_only.csv")
    return {
        "mode_dir": mode_dir,
        "base": base_path,
        "with_labels": with_labels,
        "window_labels": window_labels,
        "normal_out": normal_out,
        "anomaly_out": anomaly_out,
        "legacy_normal": legacy_normal,
        "legacy_anomaly": legacy_anomaly,
    }


def fingerprint_df(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "EMPTY"
    sort_cols = [c for c in ["begin", "end", "node_id"] if c in df.columns]
    if sort_cols:
        ordered = df.sort_values(sort_cols).reset_index(drop=True)
    else:
        ordered = df.reset_index(drop=True)
    hash_series = pd.util.hash_pandas_object(ordered, index=False).to_numpy(dtype=np.uint64)
    digest = hashlib.md5(hash_series.tobytes()).hexdigest()
    return digest


def begin_set_from_with_labels(with_labels_df: pd.DataFrame) -> set[float]:
    required = {"begin", "is_anomaly"}
    missing = required - set(with_labels_df.columns)
    if missing:
        raise ValueError(f"with_labels missing columns: {sorted(missing)}")

    grouped = with_labels_df.groupby("begin", as_index=False)["is_anomaly"].max()
    starts = grouped[grouped["is_anomaly"] > 0]["begin"].tolist()
    return set(float(x) for x in starts)


def begin_set_from_window_labels(window_labels_df: pd.DataFrame) -> set[float]:
    required = {"window_end", "is_anomaly_window"}
    missing = required - set(window_labels_df.columns)
    if missing:
        raise ValueError(f"window_labels missing columns: {sorted(missing)}")

    marked = window_labels_df[window_labels_df["is_anomaly_window"] > 0]
    return set(float(x) for x in marked["window_end"].tolist())


def split_joined_ids(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    out = []
    for item in str(raw).split(";"):
        t = item.strip()
        if t:
            out.append(t)
    return out


def load_edge_to_nodes_map(sensor_node_mapping_path: Path) -> dict[str, set[str]]:
    if not sensor_node_mapping_path.exists():
        return {}
    df = pd.read_csv(sensor_node_mapping_path)
    required = {"edge_id", "node_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sensor_node_mapping missing columns: {sorted(missing)}")
    out: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        edge = str(row["edge_id"]).strip()
        node = str(row["node_id"]).strip()
        if not edge or edge == "nan" or not node or node == "nan":
            continue
        out.setdefault(edge, set()).add(node)
    return out


def event_ranges_from_window_labels(window_labels_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    required = {"window_start", "window_end", "is_anomaly_window", "active_event_ids"}
    missing = required - set(window_labels_df.columns)
    if missing:
        raise ValueError(f"window_labels missing columns for event ranges: {sorted(missing)}")

    ranges: dict[str, tuple[float, float]] = {}
    anomaly_rows = window_labels_df[window_labels_df["is_anomaly_window"] > 0]
    for _, row in anomaly_rows.iterrows():
        w_start = float(row["window_start"])
        w_end = float(row["window_end"])
        ids = split_joined_ids(row.get("active_event_ids"))
        for event_id in ids:
            if event_id not in ranges:
                ranges[event_id] = (w_start, w_end)
            else:
                prev_start, prev_end = ranges[event_id]
                ranges[event_id] = (min(prev_start, w_start), max(prev_end, w_end))
    return ranges


def event_ranges_and_edges_from_window_labels(window_labels_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    required = {"window_start", "window_end", "is_anomaly_window", "active_event_ids", "active_edge_ids"}
    missing = required - set(window_labels_df.columns)
    if missing:
        raise ValueError(f"window_labels missing columns for event ranges: {sorted(missing)}")

    events: dict[str, dict[str, Any]] = {}
    anomaly_rows = window_labels_df[window_labels_df["is_anomaly_window"] > 0]
    for _, row in anomaly_rows.iterrows():
        w_start = float(row["window_start"])
        w_end = float(row["window_end"])
        ids = split_joined_ids(row.get("active_event_ids"))
        edges = split_joined_ids(row.get("active_edge_ids"))
        if not ids:
            continue

        if len(edges) == len(ids) and len(ids) > 0:
            pairs = list(zip(ids, edges))
        else:
            pairs = []
            for event_id in ids:
                if edges:
                    for edge in edges:
                        pairs.append((event_id, edge))
                else:
                    pairs.append((event_id, ""))

        for event_id, edge in pairs:
            if event_id not in events:
                events[event_id] = {"start": w_start, "end": w_end, "edges": set()}
            else:
                events[event_id]["start"] = min(float(events[event_id]["start"]), w_start)
                events[event_id]["end"] = max(float(events[event_id]["end"]), w_end)
            if edge:
                events[event_id]["edges"].add(edge)
    return events


def begin_to_nodes_from_window_labels_simple(
    window_labels_df: pd.DataFrame,
    edge_to_nodes: dict[str, set[str]],
) -> tuple[set[float], dict[float, set[str]], dict[str, Any]]:
    required = {"window_end", "is_anomaly_window", "active_edge_ids"}
    missing = required - set(window_labels_df.columns)
    if missing:
        raise ValueError(f"window_labels missing columns for node labels: {sorted(missing)}")

    begin_to_nodes: dict[float, set[str]] = {}
    anomaly_rows = window_labels_df[window_labels_df["is_anomaly_window"] > 0]
    for _, row in anomaly_rows.iterrows():
        begin = float(row["window_end"])
        edges = split_joined_ids(row.get("active_edge_ids"))
        nodes: set[str] = set()
        for edge in edges:
            nodes.update(edge_to_nodes.get(edge, set()))
        begin_to_nodes.setdefault(begin, set()).update(nodes)

    begin_set = set(begin_to_nodes.keys())
    meta = {
        "anomaly_begin_count_from_window_end": int(len(begin_set)),
        "begin_with_mapped_nodes_count": int(sum(1 for _, v in begin_to_nodes.items() if len(v) > 0)),
    }
    return begin_set, begin_to_nodes, meta


def begin_set_from_congestion_event_ranges(
    window_labels_df: pd.DataFrame,
    base_begin_values: pd.Series,
    edge_to_nodes: dict[str, set[str]],
) -> tuple[set[float], dict[float, set[str]], dict[str, Any]]:
    ranges = event_ranges_and_edges_from_window_labels(window_labels_df)
    begin_set: set[float] = set()
    begin_to_nodes: dict[float, set[str]] = {}
    begins = pd.to_numeric(base_begin_values, errors="coerce").dropna().astype(float).unique().tolist()
    for begin in begins:
        nodes: set[str] = set()
        for event_info in ranges.values():
            start = float(event_info["start"])
            end = float(event_info["end"])
            if start < begin <= end:
                begin_set.add(float(begin))
                for edge in event_info["edges"]:
                    nodes.update(edge_to_nodes.get(edge, set()))
        if begin in begin_set:
            begin_to_nodes[float(begin)] = nodes
    meta = {
        "num_event_ids": int(len(ranges)),
        "num_merged_event_ranges": int(len(ranges)),
        "anomaly_begin_count_from_ranges": int(len(begin_set)),
        "begin_with_mapped_nodes_count": int(sum(1 for _, v in begin_to_nodes.items() if len(v) > 0)),
    }
    return begin_set, begin_to_nodes, meta


def compare_to_legacy(new_df: pd.DataFrame, legacy_path: Path) -> dict[str, Any] | None:
    if not legacy_path.exists():
        return None
    legacy_df = pd.read_csv(legacy_path)
    common_cols = [c for c in legacy_df.columns if c in new_df.columns]
    legacy_common = legacy_df[common_cols].copy()
    new_common = new_df[common_cols].copy()

    out = {
        "legacy_path": str(legacy_path),
        "legacy_rows": int(len(legacy_df)),
        "new_rows": int(len(new_df)),
        "legacy_columns": list(legacy_df.columns),
        "new_columns": list(new_df.columns),
        "common_columns": common_cols,
        "legacy_unique_begin": int(legacy_df["begin"].nunique()) if "begin" in legacy_df.columns else None,
        "new_unique_begin": int(new_df["begin"].nunique()) if "begin" in new_df.columns else None,
        "rows_equal": bool(len(legacy_df) == len(new_df)),
        "begin_set_equal": bool(set(legacy_df["begin"].tolist()) == set(new_df["begin"].tolist()))
        if "begin" in legacy_df.columns and "begin" in new_df.columns
        else None,
        "fingerprint_equal_common_columns": bool(fingerprint_df(legacy_common) == fingerprint_df(new_common)),
        "legacy_fingerprint_common_columns": fingerprint_df(legacy_common),
        "new_fingerprint_common_columns": fingerprint_df(new_common),
    }
    return out


def split_mode(mode: str, paths: dict[str, Path], strict: bool, edge_to_nodes: dict[str, set[str]]) -> dict[str, Any]:
    base_path = paths["base"]
    if not base_path.exists():
        raise FileNotFoundError(f"Missing base forecast file for mode '{mode}': {base_path}")

    base_df = pd.read_csv(base_path)
    if "begin" not in base_df.columns:
        raise ValueError(f"Base forecast file missing 'begin': {base_path}")

    with_labels_path = paths["with_labels"]
    window_labels_path = paths["window_labels"]
    begin_anomaly: set[float]
    begin_to_nodes: dict[float, set[str]] = {}
    label_source: str
    label_meta: dict[str, Any] = {}
    with_labels_df: pd.DataFrame | None = None

    if with_labels_path.exists():
        with_labels_df = pd.read_csv(with_labels_path)
        begin_anomaly = begin_set_from_with_labels(with_labels_df)
        label_source = str(with_labels_path)
        label_meta["source_type"] = "with_labels"
    elif window_labels_path.exists():
        window_labels_df = pd.read_csv(window_labels_path)
        if mode == "congestion_spike":
            begin_anomaly, begin_to_nodes, label_meta = begin_set_from_congestion_event_ranges(
                window_labels_df=window_labels_df,
                base_begin_values=base_df["begin"],
                edge_to_nodes=edge_to_nodes,
            )
            label_meta["source_type"] = "window_labels_event_ranges"
        else:
            begin_anomaly, begin_to_nodes, label_meta = begin_to_nodes_from_window_labels_simple(
                window_labels_df=window_labels_df,
                edge_to_nodes=edge_to_nodes,
            )
            label_meta["source_type"] = "window_labels"
        label_source = str(window_labels_path)
    else:
        message = (
            f"No label source found for mode '{mode}'. "
            f"Expected one of: {with_labels_path}, {window_labels_path}"
        )
        if strict:
            raise FileNotFoundError(message)
        print(f"[WARN] {message} -> split will produce only normal rows.")
        begin_anomaly = set()
        label_source = "none"
        label_meta["source_type"] = "none"

    begin_vals = pd.to_numeric(base_df["begin"], errors="coerce").astype(float)
    is_anomaly_begin = begin_vals.isin(begin_anomaly)

    anomaly_df = base_df[is_anomaly_begin].copy()
    normal_df = base_df[~is_anomaly_begin].copy()

    # Keep normal_only schema unchanged; add label only for anomaly_only output.
    if with_labels_df is not None:
        cols = ["begin", "end", "node_id", "is_anomaly"]
        for c in cols:
            if c not in with_labels_df.columns:
                raise ValueError(f"with_labels missing required column '{c}' for anomaly label merge.")
        merge_df = with_labels_df[cols].drop_duplicates(subset=["begin", "end", "node_id"], keep="last")
        anomaly_df = anomaly_df.merge(merge_df, on=["begin", "end", "node_id"], how="left")
        anomaly_df["is_anomaly"] = pd.to_numeric(anomaly_df["is_anomaly"], errors="coerce").fillna(0).astype(int)
    else:
        # Node-level fallback from window labels + edge->node mapping.
        if begin_to_nodes:
            rows = []
            for begin, nodes in begin_to_nodes.items():
                for node in nodes:
                    rows.append((float(begin), str(node), 1))
            if rows:
                node_label_df = pd.DataFrame(rows, columns=["begin", "node_id", "is_anomaly"]).drop_duplicates()
                anomaly_df = anomaly_df.merge(node_label_df, on=["begin", "node_id"], how="left")
                anomaly_df["is_anomaly"] = pd.to_numeric(anomaly_df["is_anomaly"], errors="coerce").fillna(0).astype(int)
            else:
                anomaly_df["is_anomaly"] = 0
        else:
            anomaly_df["is_anomaly"] = 0

    paths["normal_out"].parent.mkdir(parents=True, exist_ok=True)
    normal_df.to_csv(paths["normal_out"], index=False)
    anomaly_df.to_csv(paths["anomaly_out"], index=False)

    begin_normal = set(float(x) for x in pd.to_numeric(normal_df["begin"], errors="coerce").dropna().tolist())
    begin_anomaly_out = set(float(x) for x in pd.to_numeric(anomaly_df["begin"], errors="coerce").dropna().tolist())
    begin_base = set(float(x) for x in begin_vals.dropna().tolist())

    split_integrity = {
        "base_rows": int(len(base_df)),
        "normal_rows": int(len(normal_df)),
        "anomaly_rows": int(len(anomaly_df)),
        "row_sum_ok": bool(len(normal_df) + len(anomaly_df) == len(base_df)),
        "begin_overlap_count": int(len(begin_normal & begin_anomaly_out)),
        "begin_partition_ok": bool((begin_normal | begin_anomaly_out) == begin_base),
        "base_unique_begin": int(base_df["begin"].nunique()),
        "normal_unique_begin": int(normal_df["begin"].nunique()),
        "anomaly_unique_begin": int(anomaly_df["begin"].nunique()),
        "anomaly_only_is_anomaly_sum": int(anomaly_df["is_anomaly"].sum()) if "is_anomaly" in anomaly_df.columns else None,
        "anomaly_only_has_label": bool("is_anomaly" in anomaly_df.columns),
    }

    normal_compare = compare_to_legacy(normal_df, paths["legacy_normal"])
    anomaly_compare = compare_to_legacy(anomaly_df, paths["legacy_anomaly"])

    print(
        f"[INFO] mode={mode} base={len(base_df)} normal={len(normal_df)} anomaly={len(anomaly_df)} "
        f"labels={Path(label_source).name if label_source != 'none' else 'none'}"
    )
    return {
        "mode": mode,
        "label_source": label_source,
        "base_path": str(base_path),
        "normal_output": str(paths["normal_out"]),
        "anomaly_output": str(paths["anomaly_out"]),
        "label_meta": label_meta,
        "split_integrity": split_integrity,
        "compare_legacy_normal": normal_compare,
        "compare_legacy_anomaly": anomaly_compare,
    }


def main() -> None:
    args = parse_args()
    modes = parse_modes(args.modes)
    edge_to_nodes = load_edge_to_nodes_map(args.sensor_node_mapping)

    results: dict[str, Any] = {}
    for mode in modes:
        paths = build_mode_paths(args.processed_root, mode, suffix=args.suffix)
        results[mode] = split_mode(mode=mode, paths=paths, strict=args.strict, edge_to_nodes=edge_to_nodes)

    references_dir = args.processed_root / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "processed_root": str(args.processed_root),
        "modes": modes,
        "suffix": args.suffix,
        "results": results,
    }
    summary_path = references_dir / f"split_forecast_node_level_summary{args.suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] Saved split summary: {summary_path}")


if __name__ == "__main__":
    main()
