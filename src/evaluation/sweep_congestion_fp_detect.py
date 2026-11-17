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
from src.models.model import LSTMAutoencoder
from src.preprocessing.data import (
    df_to_tensor,
    ensure_same_nodes,
    load_dataframes,
    make_time_labels,
)
from src.preprocessing.preprocess import (
    fit_scaler,
    flatten_node_feature,
    make_loader,
    make_windows,
    make_windows_with_labels,
    scale_tensor,
    split_normal_data,
)
from src.evaluation.event_eval import assign_event_groups_from_begin, infer_step_seconds
from src.evaluation.scoring import apply_alert_logic, predict_reconstruction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep feature weights + threshold percentile to maximize detect under FP constraint."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lstm_autoencoder_congestion.json",
        help="Path to runtime config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/outputs_weighted_p85_congestion/lstm_autoencoder_weighted_node_max_p85_congestion.pt",
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--target-fp-max",
        type=float,
        default=0.25,
        help="Maximum allowed normal_fp_rate.",
    )
    parser.add_argument(
        "--target-detect-min",
        type=float,
        default=0.6,
        help="Minimum desired detect metric value for reporting.",
    )
    parser.add_argument(
        "--detect-metric",
        choices=["window_detect_ratio", "event_recall"],
        default="window_detect_ratio",
        help="Detection metric to optimize under FP constraint.",
    )
    parser.add_argument(
        "--event-gap-seconds",
        type=float,
        default=None,
        help="Optional event grouping gap seconds for event_recall. Default infers from timesteps.",
    )
    parser.add_argument(
        "--min-consecutive-values",
        type=str,
        default=None,
        help="Comma-separated min_consecutive values to sweep. Default uses config value only.",
    )
    parser.add_argument(
        "--cooldown-values",
        type=str,
        default=None,
        help="Comma-separated cooldown_steps values to sweep. Default uses config value only.",
    )
    parser.add_argument(
        "--event-gap-values",
        type=str,
        default=None,
        help="Comma-separated event gap seconds to sweep (e.g. 300,600,900). Default uses --event-gap-seconds.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=12000,
        help="Number of random Dirichlet weight trials.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for sweep.",
    )
    parser.add_argument(
        "--percentiles",
        type=str,
        default="55,60,65,70,75,80,85,90",
        help="Comma-separated threshold percentiles to test.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="models/outputs_weighted_p85_congestion/tuning_fp_detect_sweep.csv",
        help="Where to save sweep table CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="models/outputs_weighted_p85_congestion/tuning_fp_detect_best.json",
        help="Where to save best candidate JSON.",
    )
    parser.add_argument(
        "--apply-config",
        action="store_true",
        help="Update config file with the best candidate weights/threshold percentile.",
    )
    return parser.parse_args()


def parse_percentiles(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("No percentiles provided.")
    return vals


def parse_int_values(raw: str | None, fallback: int) -> list[int]:
    if raw is None:
        return [int(fallback)]
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty integer sweep values.")
    return vals


def parse_gap_values(raw: str | None, fallback: float | None) -> list[float | None]:
    if raw is None:
        return [fallback]
    vals: list[float | None] = []
    for token in [x.strip() for x in raw.split(",") if x.strip()]:
        lower = token.lower()
        if lower in {"auto", "none", "null"}:
            vals.append(None)
        else:
            vals.append(float(token))
    if not vals:
        raise ValueError("Empty event gap sweep values.")
    return vals


def load_cached_errors(
    config: dict[str, Any],
    checkpoint_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[float], int]:
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

    seq_size = config["seq_size"]
    val_x = make_windows(flatten_node_feature(x_val_scaled), seq_size)
    test_normal_x = make_windows(flatten_node_feature(x_test_normal_scaled), seq_size)
    anomaly_x, _ = make_windows_with_labels(
        flatten_node_feature(x_anomaly_scaled),
        anomaly_time_labels,
        seq_size,
    )

    val_loader = make_loader(val_x, config["batch_size"], shuffle=False)
    normal_loader = make_loader(test_normal_x, config["batch_size"], shuffle=False)
    anomaly_loader = make_loader(anomaly_x, config["batch_size"], shuffle=False)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LSTMAutoencoder(
        input_dim=payload["input_dim"],
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    val_batch_x, val_recon = predict_reconstruction(model, val_loader, torch.device("cpu"))
    normal_batch_x, normal_recon = predict_reconstruction(model, normal_loader, torch.device("cpu"))
    anomaly_batch_x, anomaly_recon = predict_reconstruction(model, anomaly_loader, torch.device("cpu"))

    n_nodes = split.x_train.shape[1]
    n_features = split.x_train.shape[2]

    # Mean over sequence first -> [windows, nodes, features]
    err_val = np.abs(val_recon - val_batch_x).reshape(val_batch_x.shape[0], seq_size, n_nodes, n_features).mean(axis=1)
    err_normal = np.abs(normal_recon - normal_batch_x).reshape(normal_batch_x.shape[0], seq_size, n_nodes, n_features).mean(axis=1)
    err_anomaly = np.abs(anomaly_recon - anomaly_batch_x).reshape(anomaly_batch_x.shape[0], seq_size, n_nodes, n_features).mean(axis=1)
    return err_val, err_normal, err_anomaly, feature_cols, anomaly_timesteps, seq_size


def compute_weighted_node_and_scores(err_node_feature: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # err_node_feature: [windows, nodes, features]
    weighted_node = (err_node_feature * weights.reshape(1, 1, -1)).sum(axis=2)
    weighted_node = weighted_node / float(len(weights))
    return weighted_node, weighted_node.max(axis=1)


def build_event_groups(
    *,
    anomaly_timesteps: list[float],
    seq_size: int,
    event_gap_seconds: float | None,
) -> np.ndarray:
    begins = np.array(anomaly_timesteps[seq_size - 1 :], dtype=float)
    gap_threshold = event_gap_seconds
    if gap_threshold is None:
        inferred = infer_step_seconds(begins)
        gap_threshold = None if np.isnan(inferred) else float(inferred)
    return assign_event_groups_from_begin(
        begins=begins,
        gap_threshold=gap_threshold,
    )


def compute_event_recall_from_alerts(alerts: np.ndarray, event_groups: np.ndarray) -> tuple[float, int, int]:
    if len(event_groups) == 0:
        return float("nan"), 0, 0
    total_events = int(event_groups.max()) + 1
    detected_events = 0
    for gid in range(total_events):
        if bool(np.any(alerts[event_groups == gid])):
            detected_events += 1
    return float(detected_events / total_events), detected_events, total_events


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    percentiles = parse_percentiles(args.percentiles)
    err_val, err_normal, err_anomaly, feature_cols, anomaly_timesteps, seq_size = load_cached_errors(config, checkpoint_path)
    n_features = len(feature_cols)
    detect_col = args.detect_metric
    min_consecutive_values = parse_int_values(args.min_consecutive_values, int(config["min_consecutive"]))
    cooldown_values = parse_int_values(args.cooldown_values, int(config["cooldown_steps"]))
    event_gap_values = parse_gap_values(args.event_gap_values, args.event_gap_seconds)

    event_groups_by_gap: dict[str, np.ndarray] = {}
    for gap in event_gap_values:
        key = "auto" if gap is None else str(float(gap))
        event_groups_by_gap[key] = build_event_groups(
            anomaly_timesteps=anomaly_timesteps,
            seq_size=seq_size,
            event_gap_seconds=gap,
        )

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []

    for trial in range(args.n_trials):
        weights = rng.dirichlet(np.ones(n_features))

        _, val_scores = compute_weighted_node_and_scores(err_val, weights)
        _, normal_scores = compute_weighted_node_and_scores(err_normal, weights)
        _, anomaly_scores = compute_weighted_node_and_scores(err_anomaly, weights)

        thresholds = np.percentile(val_scores, percentiles)
        for p, thr in zip(percentiles, thresholds):
            normal_fp_rate = float((normal_scores > thr).mean())
            window_detect_ratio = float((anomaly_scores > thr).mean())
            for min_consecutive in min_consecutive_values:
                for cooldown_steps in cooldown_values:
                    alerts = apply_alert_logic(
                        anomaly_scores,
                        threshold=float(thr),
                        min_consecutive=int(min_consecutive),
                        cooldown_steps=int(cooldown_steps),
                    )
                    for gap in event_gap_values:
                        gap_key = "auto" if gap is None else str(float(gap))
                        event_groups = event_groups_by_gap[gap_key]
                        event_recall, detected_events, total_events = compute_event_recall_from_alerts(
                            alerts=alerts,
                            event_groups=event_groups,
                        )
                        rows.append(
                            {
                                "trial": trial,
                                "threshold_percentile": float(p),
                                "threshold": float(thr),
                                "min_consecutive": int(min_consecutive),
                                "cooldown_steps": int(cooldown_steps),
                                "event_gap_seconds": gap,
                                "normal_fp_rate": normal_fp_rate,
                                "window_detect_ratio": window_detect_ratio,
                                "event_recall": event_recall,
                                "detected_events": detected_events,
                                "total_events": total_events,
                                **{f"w_{f}": float(w) for f, w in zip(feature_cols, weights)},
                            }
                        )

    df = pd.DataFrame(rows)
    feasible = df[df[detect_col] >= args.target_detect_min].copy()
    df["meets_detect_target"] = df[detect_col] >= args.target_detect_min
    if len(feasible) == 0:
        best = df.sort_values(
            [detect_col, "normal_fp_rate"],
            ascending=[False, True],
        ).iloc[0]
    else:
        best = feasible.sort_values(
            ["normal_fp_rate", detect_col],
            ascending=[True, False],
        ).iloc[0]

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    best_payload = {
        "target_fp_max": args.target_fp_max,
        "target_detect_min": args.target_detect_min,
        "detect_metric": detect_col,
        "best_min_consecutive": int(best["min_consecutive"]),
        "best_cooldown_steps": int(best["cooldown_steps"]),
        "best_event_gap_seconds": None if pd.isna(best["event_gap_seconds"]) else float(best["event_gap_seconds"]),
        "best_threshold_percentile": float(best["threshold_percentile"]),
        "best_threshold": float(best["threshold"]),
        "best_normal_fp_rate": float(best["normal_fp_rate"]),
        "best_window_detect_ratio": float(best["window_detect_ratio"]),
        "best_event_recall": float(best["event_recall"]) if pd.notna(best["event_recall"]) else None,
        "best_detect_metric_value": float(best[detect_col]) if pd.notna(best[detect_col]) else None,
        "best_detected_events": int(best["detected_events"]) if pd.notna(best["detected_events"]) else None,
        "best_total_events": int(best["total_events"]) if pd.notna(best["total_events"]) else None,
        "feature_weights": {
            f: float(best[f"w_{f}"]) for f in feature_cols
        },
        "config_path": str(Path(args.config)),
        "checkpoint_path": str(checkpoint_path),
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)

    if args.apply_config:
        config_path = Path(args.config)
        with config_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        raw["threshold_percentile"] = int(round(best_payload["best_threshold_percentile"]))
        raw["min_consecutive"] = int(best_payload["best_min_consecutive"])
        raw["cooldown_steps"] = int(best_payload["best_cooldown_steps"])
        if best_payload["best_event_gap_seconds"] is not None:
            raw["event_gap_seconds"] = float(best_payload["best_event_gap_seconds"])
        raw["feature_weights"] = {
            k: round(v, 6) for k, v in best_payload["feature_weights"].items()
        }
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

    print("Sweep complete")
    print("rows:", len(df))
    print("detect_metric:", detect_col)
    print("best_min_consecutive:", best_payload["best_min_consecutive"])
    print("best_cooldown_steps:", best_payload["best_cooldown_steps"])
    print("best_event_gap_seconds:", best_payload["best_event_gap_seconds"])
    print("best_threshold_percentile:", best_payload["best_threshold_percentile"])
    print("best_normal_fp_rate:", best_payload["best_normal_fp_rate"])
    print("best_window_detect_ratio:", best_payload["best_window_detect_ratio"])
    print("best_event_recall:", best_payload["best_event_recall"])
    print("best_detect_metric_value:", best_payload["best_detect_metric_value"])
    print("best_feature_weights:")
    for f in feature_cols:
        print(f"  {f}: {best_payload['feature_weights'][f]:.6f}")
    print("saved_csv:", output_csv)
    print("saved_json:", output_json)
    if args.apply_config:
        print("updated_config:", args.config)


if __name__ == "__main__":
    main()
