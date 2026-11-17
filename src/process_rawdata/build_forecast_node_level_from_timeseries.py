"""Build forecast node-level CSV from timeseries exports for baseline/anomaly modes.

This script converts:
  <mode>_traffic_timeseries_long.csv
into:
  forecast_<mode>_node_level<suffix>.csv

Core conversion logic follows existing forecast node-level artifacts:
  - collapse node_id from lane-index level to direction level
  - keep enabled nodes only from detector_mapping.csv
  - aggregate by begin/end/node_id with fixed reductions
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(LOCAL_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PROJECT_ROOT))


DEFAULT_MODES = ["baseline", "accident", "congestion_spike", "zigzag", "sudden_stop"]
NUMERIC_COLS = ["flow_count_5m", "speed", "e1_occupancy", "e2_meanOccupancy", "jam_veh"]
OUTPUT_COLS = ["begin", "end", "node_id", *NUMERIC_COLS]

MODE_TO_DIR: dict[str, str] = {
    "baseline": "normal",
    "accident": "accident",
    "congestion_spike": "congestion",
    "zigzag": "zigzag",
    "sudden_stop": "sudden_stop",
    "mixed": "mixed",
}

MODE_TO_TIMESERIES: dict[str, str] = {
    "baseline": "baseline_traffic_timeseries_long.csv",
    "accident": "accident_traffic_timeseries_long.csv",
    "congestion_spike": "congestion_spike_traffic_timeseries_long.csv",
    "zigzag": "zigzag_traffic_timeseries_long.csv",
    "sudden_stop": "sudden_stop_traffic_timeseries_long.csv",
    "mixed": "mixed_traffic_timeseries_long.csv",
}

MODE_TO_FORECAST_BASE: dict[str, str] = {
    "baseline": "forecast_baseline_node_level.csv",
    "accident": "forecast_accident_node_level.csv",
    "congestion_spike": "forecast_congestion_spike_node_level.csv",
    "zigzag": "forecast_zigzag_node_level.csv",
    "sudden_stop": "forecast_sudden_stop_node_level.csv",
    "mixed": "forecast_mixed_node_level.csv",
}

NODE_COLLAPSE_PATTERN = re.compile(r"^(?P<prefix>.*_lane)_\d+_(?P<dir>[NSEW])$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert timeseries_long CSV to forecast node-level CSV for all supported modes."
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(DEFAULT_MODES),
        help="Comma-separated modes (default: baseline,accident,congestion_spike,zigzag).",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=LOCAL_PROJECT_ROOT / "datasets" / "processed",
        help="Processed root containing normal/accident/congestion/zigzag/references.",
    )
    parser.add_argument(
        "--detector-mapping",
        type=Path,
        default=LOCAL_PROJECT_ROOT / "datasets" / "processed" / "references" / "detector_mapping.csv",
        help="Path to detector_mapping.csv used to filter enabled nodes.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_rebuild",
        help="Suffix appended before .csv in output forecast file (default: _rebuild).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any mode input is missing. Default skips missing modes with warning.",
    )
    parser.add_argument(
        "--min-begin",
        type=float,
        default=3600.0,
        help="Drop rows with begin < min-begin after aggregation (default: 3600).",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        help="Optional summary JSON path. Default writes references/forecast_node_level_rebuild_summary.json.",
    )
    return parser.parse_args()


def parse_modes(raw: str) -> list[str]:
    modes = [m.strip() for m in raw.split(",") if m.strip()]
    if not modes:
        raise ValueError("No mode specified in --modes.")
    return modes


def collapse_node_id(node_id: str) -> str:
    m = NODE_COLLAPSE_PATTERN.match(str(node_id))
    if not m:
        return str(node_id)
    return f"{m.group('prefix')}_{m.group('dir')}"


def load_enabled_nodes(detector_mapping_path: Path) -> set[str]:
    if not detector_mapping_path.exists():
        raise FileNotFoundError(f"Missing detector mapping file: {detector_mapping_path}")
    mapping_df = pd.read_csv(detector_mapping_path)
    required = {"node_id", "enabled"}
    missing = required - set(mapping_df.columns)
    if missing:
        raise ValueError(f"detector_mapping missing columns: {sorted(missing)}")
    enabled_df = mapping_df[mapping_df["enabled"] == 1]
    return set(enabled_df["node_id"].astype(str).unique().tolist())


def mode_paths(processed_root: Path, mode: str, suffix: str) -> tuple[Path, Path, Path]:
    mode_dir = processed_root / MODE_TO_DIR.get(mode, mode)
    ts_path = mode_dir / MODE_TO_TIMESERIES.get(
        mode,
        f"{mode}_traffic_timeseries_long.csv",
    )
    base_forecast = mode_dir / MODE_TO_FORECAST_BASE.get(
        mode,
        f"forecast_{mode}_node_level.csv",
    )
    out_name = f"{base_forecast.stem}{suffix}{base_forecast.suffix}"
    out_path = mode_dir / out_name
    return ts_path, base_forecast, out_path


def convert_timeseries_to_forecast(
    ts_df: pd.DataFrame,
    enabled_nodes: set[str],
    min_begin: float | None,
) -> pd.DataFrame:
    required = {"begin", "end", "node_id", *NUMERIC_COLS}
    missing = required - set(ts_df.columns)
    if missing:
        raise ValueError(f"Timeseries missing required columns: {sorted(missing)}")

    df = ts_df[["begin", "end", "node_id", *NUMERIC_COLS]].copy()
    df["node_id"] = df["node_id"].astype(str).map(collapse_node_id)
    df = df[df["node_id"].isin(enabled_nodes)].copy()

    agg = (
        df.groupby(["begin", "end", "node_id"], as_index=False)
        .agg(
            flow_count_5m=("flow_count_5m", "sum"),
            speed=("speed", "mean"),
            e1_occupancy=("e1_occupancy", "mean"),
            e2_meanOccupancy=("e2_meanOccupancy", "mean"),
            jam_veh=("jam_veh", "max"),
        )
        .sort_values(["begin", "node_id"])
        .reset_index(drop=True)
    )

    for col in NUMERIC_COLS:
        agg[col] = pd.to_numeric(agg[col], errors="coerce")

    if min_begin is not None:
        agg = agg[agg["begin"] >= float(min_begin)].copy()

    return agg[OUTPUT_COLS]


def compare_with_existing(existing_df: pd.DataFrame, rebuilt_df: pd.DataFrame) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "existing_rows": int(len(existing_df)),
        "rebuilt_rows": int(len(rebuilt_df)),
    }
    merged = existing_df.merge(
        rebuilt_df,
        on=["begin", "end", "node_id"],
        how="inner",
        suffixes=("_old", "_new"),
    )
    comparison["matched_rows"] = int(len(merged))
    comparison["existing_nodes"] = int(existing_df["node_id"].nunique())
    comparison["rebuilt_nodes"] = int(rebuilt_df["node_id"].nunique())

    diff_stats: dict[str, Any] = {}
    for col in NUMERIC_COLS:
        old_col = f"{col}_old"
        new_col = f"{col}_new"
        if old_col not in merged.columns or new_col not in merged.columns:
            continue
        abs_diff = (merged[old_col] - merged[new_col]).abs()
        diff_stats[col] = {
            "mae": float(abs_diff.mean()) if len(abs_diff) else float("nan"),
            "max_abs_diff": float(abs_diff.max()) if len(abs_diff) else float("nan"),
        }
    comparison["diff"] = diff_stats
    return comparison


def process_mode(
    mode: str,
    processed_root: Path,
    enabled_nodes: set[str],
    suffix: str,
    strict: bool,
    min_begin: float | None,
) -> dict[str, Any]:
    ts_path, base_forecast_path, out_path = mode_paths(processed_root, mode, suffix)
    if not ts_path.exists():
        message = f"Missing timeseries input for mode '{mode}': {ts_path}"
        if strict:
            raise FileNotFoundError(message)
        print(f"[WARN] {message}")
        return {"mode": mode, "skipped": True, "reason": message}

    ts_df = pd.read_csv(ts_path)
    rebuilt_df = convert_timeseries_to_forecast(
        ts_df,
        enabled_nodes=enabled_nodes,
        min_begin=min_begin,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rebuilt_df.to_csv(out_path, index=False)

    summary: dict[str, Any] = {
        "mode": mode,
        "skipped": False,
        "timeseries_input": str(ts_path),
        "forecast_output": str(out_path),
        "rows": int(len(rebuilt_df)),
        "nodes": int(rebuilt_df["node_id"].nunique()),
        "begin_min": float(rebuilt_df["begin"].min()) if len(rebuilt_df) else None,
        "begin_max": float(rebuilt_df["begin"].max()) if len(rebuilt_df) else None,
    }

    if base_forecast_path.exists():
        existing_df = pd.read_csv(base_forecast_path)
        summary["existing_forecast"] = str(base_forecast_path)
        summary["compare"] = compare_with_existing(existing_df, rebuilt_df)
    else:
        summary["existing_forecast"] = None
        summary["compare"] = None

    print(
        f"[INFO] mode={mode} output={out_path.name} "
        f"rows={summary['rows']} nodes={summary['nodes']}"
    )
    return summary


def main() -> None:
    args = parse_args()
    modes = parse_modes(args.modes)
    enabled_nodes = load_enabled_nodes(args.detector_mapping)

    mode_summaries: dict[str, Any] = {}
    for mode in modes:
        mode_summaries[mode] = process_mode(
        mode=mode,
        processed_root=args.processed_root,
        enabled_nodes=enabled_nodes,
        suffix=args.suffix,
        strict=args.strict,
        min_begin=args.min_begin,
    )

    references_dir = args.processed_root / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "processed_root": str(args.processed_root),
        "detector_mapping": str(args.detector_mapping),
        "modes": modes,
        "suffix": args.suffix,
        "enabled_node_count": int(len(enabled_nodes)),
        "min_begin": args.min_begin,
        "results": mode_summaries,
    }
    summary_path = (
        args.summary_path
        or references_dir / "forecast_node_level_rebuild_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
