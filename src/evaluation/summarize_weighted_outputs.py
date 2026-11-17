from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize outputs_weighted* model runs into one comparison table."
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Root directory containing outputs_weighted* folders.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="outputs_weighted*",
        help="Directory glob under models root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "summary_weighted_outputs",
        help="Directory to save summary CSV and JSON (timestamped).",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def compute_precision_proxy(detected_events: int | None, normal_alert_count: int | None) -> float | None:
    if detected_events is None or normal_alert_count is None:
        return None
    denom = detected_events + normal_alert_count
    if denom <= 0:
        return None
    return float(detected_events / denom)


def compute_f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    denom = precision + recall
    if denom <= 0:
        return None
    return float(2.0 * precision * recall / denom)


def summarize_one(model_dir: Path) -> dict[str, Any]:
    pilot_path = model_dir / "pilot_summary.json"
    event_path = model_dir / "event_eval_alert.csv"
    config_path = model_dir / "config.json"

    if not pilot_path.exists():
        raise FileNotFoundError(f"Missing pilot_summary.json: {pilot_path}")
    if not event_path.exists():
        raise FileNotFoundError(f"Missing event_eval_alert.csv: {event_path}")

    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    config: dict[str, Any] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    checkpoint_files = sorted(model_dir.glob("*.pt"))
    checkpoint_name = checkpoint_files[0].name if checkpoint_files else None

    detected_events = safe_int(pilot.get("detected_events"))
    total_events = safe_int(pilot.get("total_events"))
    normal_alert_count = safe_int(pilot.get("normal_alert_count"))
    recall = safe_float(pilot.get("event_recall"))
    precision_proxy = compute_precision_proxy(detected_events, normal_alert_count)
    f1_proxy = compute_f1(precision_proxy, recall)

    return {
        "model_dir": model_dir.name,
        "anomaly": infer_anomaly_name(model_dir.name),
        "checkpoint_file": checkpoint_name,
        "score_name": pilot.get("score_name"),
        "threshold_percentile": safe_float(pilot.get("threshold_percentile")),
        "threshold": safe_float(pilot.get("threshold")),
        "min_consecutive": safe_int(pilot.get("min_consecutive")),
        "cooldown_steps": safe_int(pilot.get("cooldown_steps")),
        "event_gap_seconds": safe_float(config.get("event_gap_seconds")),
        "event_detected_events": detected_events,
        "event_total_events": total_events,
        "event_missed_events": safe_int(pilot.get("missed_events")),
        "event_precision_proxy": precision_proxy,
        "event_recall": recall,
        "event_f1_proxy": f1_proxy,
        "detection_delay_mean_seconds": safe_float(pilot.get("mean_delay_seconds")),
        "detection_delay_median_seconds": safe_float(pilot.get("median_delay_seconds")),
        "detection_delay_mean_minutes": safe_float(pilot.get("mean_delay_minutes")),
        "detection_delay_median_minutes": safe_float(pilot.get("median_delay_minutes")),
        "false_positive_rate": safe_float(pilot.get("normal_fp_rate")),
        "normal_fp_windows": safe_int(pilot.get("normal_fp_windows")),
        "normal_total_windows": safe_int(pilot.get("normal_total_windows")),
        "normal_alert_count": normal_alert_count,
        "false_alerts_per_hour": safe_float(pilot.get("false_alerts_per_hour")),
        "false_alerts_per_day": safe_float(pilot.get("false_alerts_per_day")),
        "window_detect_ratio": safe_float(pilot.get("window_detect_ratio")),
        "pilot_summary_path": str(pilot_path),
        "event_eval_alert_path": str(event_path),
        "config_path": str(config_path) if config_path.exists() else None,
    }


def infer_anomaly_name(model_dir_name: str) -> str:
    lower = model_dir_name.lower()
    if "sudden_stop" in lower:
        return "sudden_stop"
    if "congestion" in lower:
        return "congestion_spike"
    if "zigzag" in lower:
        return "zigzag"
    if "accident" in lower:
        return "accident"
    return "unknown"


def main() -> None:
    args = parse_args()
    model_dirs = sorted([p for p in args.models_root.glob(args.glob) if p.is_dir()])

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for model_dir in model_dirs:
        try:
            rows.append(summarize_one(model_dir))
        except Exception as exc:
            skipped.append({"model_dir": model_dir.name, "reason": str(exc)})

    if not rows:
        raise RuntimeError("No valid outputs_weighted* model directories found for summary.")

    df = pd.DataFrame(rows).sort_values("model_dir").reset_index(drop=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Split into parts instead of one all-in-one file.
    overview_cols = [
        "anomaly",
        "model_dir",
        "checkpoint_file",
        "score_name",
        "threshold_percentile",
        "threshold",
        "min_consecutive",
        "cooldown_steps",
        "event_gap_seconds",
        "window_detect_ratio",
    ]
    event_cols = [
        "anomaly",
        "model_dir",
        "event_detected_events",
        "event_total_events",
        "event_missed_events",
        "event_precision_proxy",
        "event_recall",
        "event_f1_proxy",
    ]
    delay_cols = [
        "anomaly",
        "model_dir",
        "detection_delay_mean_seconds",
        "detection_delay_median_seconds",
        "detection_delay_mean_minutes",
        "detection_delay_median_minutes",
    ]
    fp_cols = [
        "anomaly",
        "model_dir",
        "false_positive_rate",
        "normal_fp_windows",
        "normal_total_windows",
        "normal_alert_count",
        "false_alerts_per_hour",
        "false_alerts_per_day",
    ]

    overview_csv = args.output_dir / f"weighted_outputs_overview_{stamp}.csv"
    event_csv = args.output_dir / f"weighted_outputs_event_metrics_{stamp}.csv"
    delay_csv = args.output_dir / f"weighted_outputs_delay_metrics_{stamp}.csv"
    fp_csv = args.output_dir / f"weighted_outputs_false_positive_metrics_{stamp}.csv"
    json_path = args.output_dir / f"weighted_outputs_summary_{stamp}.json"

    df[overview_cols].to_csv(overview_csv, index=False)
    df[event_cols].to_csv(event_csv, index=False)
    df[delay_cols].to_csv(delay_csv, index=False)
    df[fp_cols].to_csv(fp_csv, index=False)

    # Per-anomaly outputs (requested format): one row per anomaly with full key metrics.
    per_anomaly_dir = args.output_dir / f"weighted_outputs_by_anomaly_{stamp}"
    per_anomaly_dir.mkdir(parents=True, exist_ok=False)
    anomaly_cols = [
        "anomaly",
        "model_dir",
        "checkpoint_file",
        "score_name",
        "threshold_percentile",
        "threshold",
        "min_consecutive",
        "cooldown_steps",
        "event_gap_seconds",
        "event_detected_events",
        "event_total_events",
        "event_missed_events",
        "event_precision_proxy",
        "event_recall",
        "event_f1_proxy",
        "detection_delay_mean_seconds",
        "detection_delay_median_seconds",
        "false_positive_rate",
        "normal_fp_windows",
        "normal_total_windows",
        "normal_alert_count",
        "window_detect_ratio",
    ]
    for anomaly, gdf in df.groupby("anomaly", dropna=False):
        anomaly_name = str(anomaly)
        out_csv = per_anomaly_dir / f"{anomaly_name}_summary_{stamp}.csv"
        out_json = per_anomaly_dir / f"{anomaly_name}_summary_{stamp}.json"
        gdf[anomaly_cols].to_csv(out_csv, index=False)
        out_json.write_text(
            json.dumps(gdf.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )

    # Also write per-model JSON files for easier drill-down.
    by_model_dir = args.output_dir / f"weighted_outputs_by_model_{stamp}"
    by_model_dir.mkdir(parents=True, exist_ok=False)
    for row in rows:
        model_json = by_model_dir / f"{row['model_dir']}.json"
        model_json.write_text(json.dumps(row, indent=2), encoding="utf-8")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models_root": str(args.models_root),
        "glob": args.glob,
        "included_models": [r["model_dir"] for r in rows],
        "skipped_models": skipped,
        "precision_proxy_formula": "detected_events / (detected_events + normal_alert_count)",
        "f1_proxy_formula": "2 * precision_proxy * recall / (precision_proxy + recall)",
        "false_positive_rate_field": "normal_fp_rate from pilot_summary.json",
        "summary_parts": {
            "overview_csv": str(overview_csv),
            "event_metrics_csv": str(event_csv),
            "delay_metrics_csv": str(delay_csv),
            "false_positive_metrics_csv": str(fp_csv),
            "by_model_dir": str(by_model_dir),
            "by_anomaly_dir": str(per_anomaly_dir),
        },
        "rows": len(df),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[DONE] Overview CSV: {overview_csv}")
    print(f"[DONE] Event CSV: {event_csv}")
    print(f"[DONE] Delay CSV: {delay_csv}")
    print(f"[DONE] FP CSV: {fp_csv}")
    print(f"[DONE] By-model JSON dir: {by_model_dir}")
    print(f"[DONE] By-anomaly dir: {per_anomaly_dir}")
    print(f"[DONE] Summary JSON: {json_path}")
    print(f"[DONE] Included models: {len(rows)}")
    if skipped:
        print(f"[WARN] Skipped models: {len(skipped)}")


if __name__ == "__main__":
    main()
