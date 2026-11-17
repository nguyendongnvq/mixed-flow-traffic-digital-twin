from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import load_config
from src.evaluation.event_eval import assign_event_groups_from_begin, infer_step_seconds
from src.evaluation.scoring import (
    apply_alert_logic,
    compute_weighted_node_max_scores_from_error,
)
from src.models.dmfgcrn_adapter import DMFGCRNForecaster, build_forecast_windows
from src.preprocessing.data import (
    df_to_tensor,
    ensure_same_nodes,
    load_dataframes,
    make_time_labels,
    to_feature_weights_array,
)
from src.preprocessing.preprocess import fit_scaler, scale_tensor, split_normal_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep DMFGCRN anomaly parameters to satisfy FP and event-recall constraints."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dmfgcrn_accident.json",
        help="Path to DMFGCRN runtime config.",
    )
    parser.add_argument(
        "--target-fp-max",
        type=float,
        default=0.15,
        help="Maximum allowed normal_fp_rate.",
    )
    parser.add_argument(
        "--target-recall-min",
        type=float,
        default=0.75,
        help="Minimum required event_recall.",
    )
    parser.add_argument(
        "--horizon-values",
        type=str,
        default="0,1,2,3,5,7,11",
        help="Comma-separated dmfgcrn_horizon_index values.",
    )
    parser.add_argument(
        "--percentiles",
        type=str,
        default="70,72,74,76,78,80,82,84,86,88,90,92,94,96",
        help="Comma-separated threshold percentiles to test.",
    )
    parser.add_argument(
        "--min-consecutive-values",
        type=str,
        default="1,2,3",
        help="Comma-separated min_consecutive values.",
    )
    parser.add_argument(
        "--cooldown-values",
        type=str,
        default="0,1,2,3,4,5",
        help="Comma-separated cooldown_steps values.",
    )
    parser.add_argument(
        "--event-gap-values",
        type=str,
        default="auto,300,600",
        help="Comma-separated event gap seconds ('auto' supported).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="models/outputs_dmfgcrn_weighted_p80/tuning_dmfgcrn_fp_recall_sweep.csv",
        help="Where to save sweep table CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="models/outputs_dmfgcrn_weighted_p80/tuning_dmfgcrn_fp_recall_best.json",
        help="Where to save best candidate JSON.",
    )
    parser.add_argument(
        "--apply-config",
        action="store_true",
        help="Update config file with best parameters.",
    )
    return parser.parse_args()


def parse_float_values(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty float values.")
    return vals


def parse_int_values(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty integer values.")
    return vals


def parse_gap_values(raw: str) -> list[float | None]:
    vals: list[float | None] = []
    for token in [x.strip() for x in raw.split(",") if x.strip()]:
        lower = token.lower()
        if lower in {"auto", "none", "null"}:
            vals.append(None)
        else:
            vals.append(float(token))
    if not vals:
        raise ValueError("Empty event-gap values.")
    return vals


def compute_event_metrics(
    alerts: np.ndarray,
    begins: np.ndarray,
    event_groups: np.ndarray,
) -> tuple[float, int, int, float]:
    if len(event_groups) == 0:
        return float("nan"), 0, 0, float("nan")
    total_events = int(event_groups.max()) + 1
    detected_events = 0
    delays: list[float] = []
    for gid in range(total_events):
        mask = event_groups == gid
        event_begins = begins[mask]
        event_alerts = alerts[mask]
        if event_alerts.any():
            detected_events += 1
            first_alert_idx = np.where(event_alerts)[0][0]
            delays.append(float(event_begins[first_alert_idx] - event_begins[0]))
    recall = float(detected_events / total_events)
    mean_delay = float(np.mean(delays)) if delays else float("nan")
    return recall, detected_events, total_events, mean_delay


def _load_scaled_splits(config: dict[str, Any]) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float], list[float]]:
    label_col = config["label_col"]
    time_col = config["time_col"]
    node_col = config["node_col"]
    feature_cols = config["feature_cols"]

    df_normal, df_anomaly = load_dataframes(
        config["normal_dataset"],
        config["accident_dataset"],
        label_col=label_col,
    )
    nodes = ensure_same_nodes(df_normal, df_anomaly, node_col)
    x_normal, normal_timesteps = df_to_tensor(
        df_normal,
        feature_cols,
        nodes,
        time_col=time_col,
        node_col=node_col,
    )
    x_anomaly, anomaly_timesteps = df_to_tensor(
        df_anomaly,
        feature_cols,
        nodes,
        time_col=time_col,
        node_col=node_col,
    )
    anomaly_time_labels = make_time_labels(
        df_anomaly,
        anomaly_timesteps,
        time_col=time_col,
        label_col=label_col,
    )

    split = split_normal_data(
        x_normal,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
    )
    scaler = fit_scaler(split.x_train)
    x_val_scaled = scale_tensor(split.x_val, scaler)
    x_test_normal_scaled = scale_tensor(split.x_test_normal, scaler)
    x_anomaly_scaled = scale_tensor(x_anomaly, scaler)

    val_timesteps = normal_timesteps[split.train_end : split.val_end]
    normal_test_timesteps = normal_timesteps[split.val_end :]
    return (
        split,
        x_val_scaled,
        x_test_normal_scaled,
        x_anomaly_scaled,
        anomaly_time_labels,
        val_timesteps,
        normal_test_timesteps,
        anomaly_timesteps,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    backend = str(config.get("model_backend", "")).lower()
    if backend != "dmfgcrn":
        raise ValueError(f"Expected model_backend='dmfgcrn', got: {backend}")

    percentiles = parse_float_values(args.percentiles)
    horizon_values_raw = parse_int_values(args.horizon_values)
    min_consecutive_values = parse_int_values(args.min_consecutive_values)
    cooldown_values = parse_int_values(args.cooldown_values)
    event_gap_values = parse_gap_values(args.event_gap_values)

    (
        split,
        x_val_scaled,
        x_test_normal_scaled,
        x_anomaly_scaled,
        anomaly_time_labels,
        val_timesteps,
        normal_test_timesteps,
        anomaly_timesteps,
    ) = _load_scaled_splits(config)
    n_nodes = split.x_train.shape[1]

    feature_cols = list(config["feature_cols"])
    feature_to_idx = {name: idx for idx, name in enumerate(feature_cols)}
    input_features = list(config["dmfgcrn_input_features"])
    output_features = list(config["dmfgcrn_output_features"])
    input_indices = [feature_to_idx[name] for name in input_features]
    output_indices = [feature_to_idx[name] for name in output_features]
    output_weights = to_feature_weights_array(output_features, config["feature_weights"])

    dmfgcrn_device_raw = str(config.get("dmfgcrn_device", "cpu")).lower()
    if dmfgcrn_device_raw.startswith("cuda") and not torch.cuda.is_available():
        dmfgcrn_device = torch.device("cpu")
    else:
        dmfgcrn_device = torch.device(dmfgcrn_device_raw)

    lag = int(config["seq_size"])
    steps_per_day = int(config.get("dmfgcrn_steps_per_day", 288))

    forecaster = DMFGCRNForecaster(
        project_root=PROJECT_ROOT,
        checkpoint_path=Path(config["dmfgcrn_checkpoint_path"]),
        conf_path=Path(config["dmfgcrn_conf_path"]),
        num_nodes=n_nodes,
        lag=lag,
        device=dmfgcrn_device,
    )
    max_horizon = int(forecaster.args.horizon)
    horizon_values = [h for h in horizon_values_raw if 0 <= h < max_horizon]
    if not horizon_values:
        raise ValueError(
            f"No valid horizon indices in {horizon_values_raw}; model horizon is {max_horizon}."
        )

    rows: list[dict[str, Any]] = []
    for horizon_index in horizon_values:
        val_x, val_y, _, _ = build_forecast_windows(
            x_scaled=x_val_scaled,
            timesteps=val_timesteps,
            step_labels=np.zeros(len(val_timesteps), dtype=np.int32),
            lag=lag,
            horizon_index=horizon_index,
            input_indices=input_indices,
            output_indices=output_indices,
            use_day=bool(forecaster.args.use_day),
            use_week=bool(forecaster.args.use_week),
            steps_per_day=steps_per_day,
        )
        normal_x, normal_y, _, _ = build_forecast_windows(
            x_scaled=x_test_normal_scaled,
            timesteps=normal_test_timesteps,
            step_labels=np.zeros(len(normal_test_timesteps), dtype=np.int32),
            lag=lag,
            horizon_index=horizon_index,
            input_indices=input_indices,
            output_indices=output_indices,
            use_day=bool(forecaster.args.use_day),
            use_week=bool(forecaster.args.use_week),
            steps_per_day=steps_per_day,
        )
        anomaly_x, anomaly_y, _, anomaly_window_end_begin = build_forecast_windows(
            x_scaled=x_anomaly_scaled,
            timesteps=anomaly_timesteps,
            step_labels=anomaly_time_labels,
            lag=lag,
            horizon_index=horizon_index,
            input_indices=input_indices,
            output_indices=output_indices,
            use_day=bool(forecaster.args.use_day),
            use_week=bool(forecaster.args.use_week),
            steps_per_day=steps_per_day,
        )

        pred_val = forecaster.predict(val_x, int(config["batch_size"]), horizon_index)
        pred_normal = forecaster.predict(normal_x, int(config["batch_size"]), horizon_index)
        pred_anomaly = forecaster.predict(anomaly_x, int(config["batch_size"]), horizon_index)

        err_val = np.abs(pred_val - val_y)
        err_normal = np.abs(pred_normal - normal_y)
        err_anomaly = np.abs(pred_anomaly - anomaly_y)

        val_scores, _ = compute_weighted_node_max_scores_from_error(err_val, output_weights)
        normal_scores, _ = compute_weighted_node_max_scores_from_error(err_normal, output_weights)
        anomaly_scores, _ = compute_weighted_node_max_scores_from_error(err_anomaly, output_weights)

        event_groups_by_gap: dict[str, np.ndarray] = {}
        for gap in event_gap_values:
            gap_key = "auto" if gap is None else str(float(gap))
            if gap is None:
                inferred_step = infer_step_seconds(anomaly_window_end_begin)
                gap_threshold = None if np.isnan(inferred_step) else float(inferred_step)
            else:
                gap_threshold = float(gap)
            event_groups_by_gap[gap_key] = assign_event_groups_from_begin(
                begins=anomaly_window_end_begin,
                gap_threshold=gap_threshold,
            )

        thresholds = np.percentile(val_scores, percentiles)
        for p, threshold in zip(percentiles, thresholds):
            normal_fp_rate = float((normal_scores > threshold).mean())
            window_detect_ratio = float((anomaly_scores > threshold).mean())
            for min_consecutive in min_consecutive_values:
                for cooldown_steps in cooldown_values:
                    alerts = apply_alert_logic(
                        anomaly_scores,
                        threshold=float(threshold),
                        min_consecutive=int(min_consecutive),
                        cooldown_steps=int(cooldown_steps),
                    )
                    for gap in event_gap_values:
                        gap_key = "auto" if gap is None else str(float(gap))
                        event_groups = event_groups_by_gap[gap_key]
                        event_recall, detected_events, total_events, mean_delay_seconds = (
                            compute_event_metrics(
                                alerts=alerts,
                                begins=anomaly_window_end_begin,
                                event_groups=event_groups,
                            )
                        )
                        rows.append(
                            {
                                "horizon_index": int(horizon_index),
                                "threshold_percentile": float(p),
                                "threshold": float(threshold),
                                "min_consecutive": int(min_consecutive),
                                "cooldown_steps": int(cooldown_steps),
                                "event_gap_seconds": gap,
                                "normal_fp_rate": normal_fp_rate,
                                "window_detect_ratio": window_detect_ratio,
                                "event_recall": event_recall,
                                "detected_events": detected_events,
                                "total_events": total_events,
                                "mean_delay_seconds": mean_delay_seconds,
                            }
                        )

    df = pd.DataFrame(rows)
    df["meets_constraints"] = (
        (df["normal_fp_rate"] <= float(args.target_fp_max))
        & (df["event_recall"] >= float(args.target_recall_min))
    )
    feasible = df[df["meets_constraints"]].copy()
    found_feasible = len(feasible) > 0

    if found_feasible:
        best = feasible.sort_values(
            ["event_recall", "normal_fp_rate", "mean_delay_seconds"],
            ascending=[False, True, True],
        ).iloc[0]
    else:
        df["recall_gap"] = np.maximum(0.0, float(args.target_recall_min) - df["event_recall"])
        df["fp_gap"] = np.maximum(0.0, df["normal_fp_rate"] - float(args.target_fp_max))
        df["constraint_gap"] = df["recall_gap"] + df["fp_gap"]
        best = df.sort_values(
            ["constraint_gap", "event_recall", "normal_fp_rate", "mean_delay_seconds"],
            ascending=[True, False, True, True],
        ).iloc[0]

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    best_payload = {
        "found_feasible": bool(found_feasible),
        "target_fp_max": float(args.target_fp_max),
        "target_recall_min": float(args.target_recall_min),
        "best_horizon_index": int(best["horizon_index"]),
        "best_threshold_percentile": float(best["threshold_percentile"]),
        "best_threshold": float(best["threshold"]),
        "best_min_consecutive": int(best["min_consecutive"]),
        "best_cooldown_steps": int(best["cooldown_steps"]),
        "best_event_gap_seconds": None
        if pd.isna(best["event_gap_seconds"])
        else float(best["event_gap_seconds"]),
        "best_normal_fp_rate": float(best["normal_fp_rate"]),
        "best_event_recall": float(best["event_recall"]),
        "best_window_detect_ratio": float(best["window_detect_ratio"]),
        "best_mean_delay_seconds": float(best["mean_delay_seconds"])
        if pd.notna(best["mean_delay_seconds"])
        else None,
        "best_detected_events": int(best["detected_events"]),
        "best_total_events": int(best["total_events"]),
        "config_path": str(Path(args.config)),
        "checkpoint_path": str(Path(config["dmfgcrn_checkpoint_path"])),
        "rows": int(len(df)),
        "feasible_rows": int(len(feasible)),
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)

    if args.apply_config:
        cfg_path = Path(args.config)
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        raw["threshold_percentile"] = int(round(best_payload["best_threshold_percentile"]))
        raw["min_consecutive"] = int(best_payload["best_min_consecutive"])
        raw["cooldown_steps"] = int(best_payload["best_cooldown_steps"])
        raw["dmfgcrn_horizon_index"] = int(best_payload["best_horizon_index"])
        if best_payload["best_event_gap_seconds"] is not None:
            raw["event_gap_seconds"] = float(best_payload["best_event_gap_seconds"])
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

    print("Sweep complete")
    print("rows:", len(df))
    print("found_feasible:", found_feasible)
    print("best_horizon_index:", best_payload["best_horizon_index"])
    print("best_threshold_percentile:", best_payload["best_threshold_percentile"])
    print("best_min_consecutive:", best_payload["best_min_consecutive"])
    print("best_cooldown_steps:", best_payload["best_cooldown_steps"])
    print("best_event_gap_seconds:", best_payload["best_event_gap_seconds"])
    print("best_normal_fp_rate:", best_payload["best_normal_fp_rate"])
    print("best_event_recall:", best_payload["best_event_recall"])
    print("best_window_detect_ratio:", best_payload["best_window_detect_ratio"])
    print("best_mean_delay_seconds:", best_payload["best_mean_delay_seconds"])
    print("saved_csv:", output_csv)
    print("saved_json:", output_json)
    if args.apply_config:
        print("updated_config:", args.config)


if __name__ == "__main__":
    main()
