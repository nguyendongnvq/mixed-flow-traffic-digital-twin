"""Timeseries-only exporter from SUMO detector XML into datasets/processed/<mode>/.

This entrypoint intentionally excludes npz sequence/graph generation and only
produces long-form node-level timeseries CSV per mode plus export summary.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(LOCAL_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PROJECT_ROOT))

from src.process_rawdata.export_sumo_timeseries_and_dataset import (
    DEFAULT_MODES,
    aggregate_to_node,
    densify_timeseries,
    parse_e1_xml,
    parse_e2_xml,
    parse_modes,
    resolve_mode_paths,
)


MODE_TO_PROCESSED_SUBDIR: dict[str, str] = {
    "baseline": "normal",
    "accident": "accident",
    "congestion_spike": "congestion",
    "zigzag": "zigzag",
    "sudden_stop": "sudden_stop",
    "mixed": "mixed",
}

MODE_TO_OUTPUT_FILENAME: dict[str, str] = {
    "baseline": "baseline_traffic_timeseries_long.csv",
    "accident": "accident_traffic_timeseries_long.csv",
    "congestion_spike": "congestion_spike_traffic_timeseries_long.csv",
    "zigzag": "zigzag_traffic_timeseries_long.csv",
    "sudden_stop": "sudden_stop_traffic_timeseries_long.csv",
    "mixed": "mixed_traffic_timeseries_long.csv",
}

ZERO_FILL_COLS = [
    "nVehContrib_e1",
    "flow_veh_h_e1",
    "nVehEntered_e1",
    "sampledSeconds",
    "nVehEntered_e2",
    "nVehLeft_e2",
    "nVehSeen_e2",
    "meanMaxJamLengthInVehicles",
    "maxJamLengthInVehicles",
    "meanHaltingDuration",
    "startedHalts",
    "meanVehicleNumber",
    "maxVehicleNumber",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export node-level traffic timeseries CSV per mode into datasets/processed/<mode>."
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(DEFAULT_MODES),
        help="Comma-separated modes (default: baseline,accident,congestion_spike,zigzag).",
    )
    parser.add_argument(
        "--detector-dir",
        type=Path,
        default=LOCAL_PROJECT_ROOT / "datasets" / "raw" / "sumo_data" / "detectors",
        help="Directory containing detector XML files.",
    )
    parser.add_argument(
        "--scenarios-dir",
        type=Path,
        default=LOCAL_PROJECT_ROOT / "datasets" / "raw" / "sumo_data",
        help="Directory containing anomaly label/record files.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=LOCAL_PROJECT_ROOT / "datasets" / "processed",
        help="Processed dataset root containing normal/accident/congestion/zigzag folders.",
    )
    parser.add_argument(
        "--window-sec",
        type=int,
        default=300,
        help="Expected detector interval seconds (default: 300).",
    )
    parser.add_argument(
        "--node-mode",
        choices=["lane", "road_dir", "road"],
        default="lane",
        help="Node aggregation mode (default: lane).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any expected input artifact is missing or inconsistent.",
    )
    parser.add_argument(
        "--skip-copy-labels",
        action="store_true",
        help="Skip copying mode label/spawn files into processed mode folders.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        help="Optional summary JSON path. Default writes references/timeseries_export_summary.json.",
    )
    return parser.parse_args()


def output_path_for_mode(processed_root: Path, mode: str) -> tuple[Path, Path]:
    subdir = MODE_TO_PROCESSED_SUBDIR.get(mode, mode)
    filename = MODE_TO_OUTPUT_FILENAME.get(mode, f"{mode}_traffic_timeseries_long.csv")
    mode_dir = processed_root / subdir
    return mode_dir, mode_dir / filename


def validate_timeseries_output_path(path: Path) -> None:
    # Protect training/preprocessed artifacts from accidental overwrite.
    if path.name.startswith("forecast_") and path.name.endswith("_node_level.csv"):
        raise ValueError(
            "Timeseries exporter must not write to forecast_*_node_level.csv. "
            f"Refusing output path: {path}"
        )


def copy_mode_labels(paths: Any, mode_dir: Path) -> dict[str, str | None]:
    copied: dict[str, str | None] = {
        "step_labels": None,
        "window_labels": None,
        "spawn_records": None,
    }
    for src in [paths.step_labels, paths.window_labels]:
        dst = mode_dir / src.name
        shutil.copy2(src, dst)
        key = "step_labels" if "step_labels" in src.name else "window_labels"
        copied[key] = str(dst)
    if paths.spawn_records is not None:
        dst = mode_dir / paths.spawn_records.name
        shutil.copy2(paths.spawn_records, dst)
        copied["spawn_records"] = str(dst)
    return copied


def process_mode(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    paths = resolve_mode_paths(
        mode=mode,
        detector_dir=args.detector_dir,
        scenarios_dir=args.scenarios_dir,
        strict=args.strict,
        window_sec=args.window_sec,
    )
    print(f"[INFO] Exporting timeseries for mode '{mode}'...")
    e1_df = parse_e1_xml(paths.e1_xml)
    e2_df = parse_e2_xml(paths.e2_xml)
    merged = pd.merge(
        e1_df,
        e2_df,
        on=["begin", "end", "base_id", "road_key", "lane_idx", "direction"],
        how="outer",
        validate="one_to_one",
    )
    for col in ZERO_FILL_COLS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    node_ts = aggregate_to_node(merged, node_mode=args.node_mode)
    node_ts_full, median_step = densify_timeseries(node_ts)
    if median_step != args.window_sec:
        message = (
            f"Mode '{mode}' has median step {median_step}s, expected {args.window_sec}s. "
            "Continuing with detected median step."
        )
        if args.strict:
            raise ValueError(message)
        print(f"[WARN] {message}")

    mode_dir, out_csv = output_path_for_mode(args.processed_root, mode)
    mode_dir.mkdir(parents=True, exist_ok=True)
    validate_timeseries_output_path(out_csv)
    node_ts_full.to_csv(out_csv, index=False)

    copied = {"step_labels": None, "window_labels": None, "spawn_records": None}
    if not args.skip_copy_labels:
        copied = copy_mode_labels(paths, mode_dir)

    print(
        f"[INFO] Mode '{mode}' done: csv={out_csv}, "
        f"timesteps={node_ts_full['begin'].nunique()}, nodes={node_ts_full['node_id'].nunique()}, rows={len(node_ts_full)}"
    )
    return {
        "mode": mode,
        "output_dir": str(mode_dir),
        "timeseries_csv": str(out_csv),
        "e1_xml": str(paths.e1_xml),
        "e2_xml": str(paths.e2_xml),
        "copied_files": copied,
        "time_steps": int(node_ts_full["begin"].nunique()),
        "nodes": int(node_ts_full["node_id"].nunique()),
        "rows": int(len(node_ts_full)),
        "median_step_sec": int(median_step),
    }


def main() -> None:
    args = parse_args()
    modes = parse_modes(args.modes)
    if args.window_sec <= 0:
        raise ValueError("--window-sec must be > 0.")

    summaries: dict[str, Any] = {}
    for mode in modes:
        summaries[mode] = process_mode(args, mode)

    references_dir = args.processed_root / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "processed_root": str(args.processed_root),
        "detector_dir": str(args.detector_dir),
        "scenarios_dir": str(args.scenarios_dir),
        "modes": summaries,
        "node_mode": args.node_mode,
        "window_sec": int(args.window_sec),
    }
    summary_path = args.summary_path or references_dir / "timeseries_export_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] Timeseries export completed. Summary: {summary_path}")


if __name__ == "__main__":
    main()
