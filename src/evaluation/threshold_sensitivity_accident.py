from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.configs.config import load_config
from src.evaluation.event_eval import build_accident_result_df, evaluate_event_level, infer_step_seconds, summarize_metrics
from src.evaluation.scoring import apply_alert_logic, compute_weighted_node_max_scores, predict_reconstruction, select_threshold
from src.models.model import LSTMAutoencoder
from src.preprocessing.data import df_to_tensor, ensure_same_nodes, load_dataframes, make_time_labels, to_feature_weights_array
from src.preprocessing.preprocess import (
    fit_scaler,
    flatten_node_feature,
    make_loader,
    make_windows,
    make_windows_with_labels,
    scale_tensor,
    split_normal_data,
)
from src.train.trainer import train_model
from src.utils.io_utils import ensure_output_dir
from src.utils.plotting import plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep accident detection threshold percentile and export Figure 8 sensitivity curve."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lstm_autoencoder_accident.json",
        help="Path to accident config JSON.",
    )
    parser.add_argument(
        "--percentiles",
        type=str,
        default="80,85,90,95",
        help="Comma-separated threshold percentiles to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output folder. Default uses config['output_dir'].",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="threshold_sensitivity_accident.csv",
        help="Output CSV file name.",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="threshold_sensitivity_curve_figure8.png",
        help="Output Figure 8 file name.",
    )
    parser.add_argument(
        "--summary-json-name",
        type=str,
        default="threshold_sensitivity_accident_summary.json",
        help="Output JSON summary file name.",
    )
    return parser.parse_args()


def parse_percentiles(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("No percentile provided.")
    if any(v < 0 or v > 100 for v in vals):
        raise ValueError(f"Percentile must be in [0,100], got: {vals}")
    # Preserve order but drop duplicates.
    out: list[float] = []
    seen: set[float] = set()
    for v in vals:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fmt_percentile_label(p: float) -> str:
    rounded = int(round(p))
    if abs(p - rounded) < 1e-9:
        return f"P{rounded}"
    return f"P{p:g}"


def plot_threshold_sensitivity(
    *,
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    x = np.arange(len(df), dtype=float)
    labels = [_fmt_percentile_label(float(p)) for p in df["threshold_percentile"].tolist()]

    fig, ax1 = plt.subplots(figsize=(10.5, 5.6))
    ax1.set_xlabel("Threshold percentile")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Event recall", color="#1f77b4")
    line1 = ax1.plot(
        x,
        df["event_recall"].to_numpy(dtype=float),
        marker="o",
        linewidth=2.0,
        color="#1f77b4",
        label="Event recall",
    )
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.set_ylabel("FP/day", color="#ff7f0e")
    line2 = ax2.plot(
        x,
        df["false_alerts_per_day"].to_numpy(dtype=float),
        marker="s",
        linewidth=2.0,
        color="#ff7f0e",
        label="FP/day",
    )
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.14))
    ax3.set_ylabel("Mean delay (seconds)", color="#2ca02c")
    line3 = ax3.plot(
        x,
        df["mean_delay_seconds"].to_numpy(dtype=float),
        marker="^",
        linewidth=2.0,
        color="#2ca02c",
        label="Mean delay (seconds)",
    )
    ax3.tick_params(axis="y", labelcolor="#2ca02c")

    lines = line1 + line2 + line3
    labels_all = [l.get_label() for l in lines]
    ax1.legend(lines, labels_all, loc="best")
    ax1.set_title("Threshold sensitivity curve", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    percentiles = parse_percentiles(args.percentiles)

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ensure_output_dir(args.output_dir if args.output_dir else config["output_dir"])

    print("Device:", device)
    print("Output dir:", output_dir)
    print("Percentiles:", percentiles)

    time_col = config["time_col"]
    node_col = config["node_col"]
    label_col = config["label_col"]
    feature_cols = config["feature_cols"]
    feature_weights = to_feature_weights_array(feature_cols, config["feature_weights"])

    df_normal_clean, df_accident_clean = load_dataframes(
        config["normal_dataset"],
        config["accident_dataset"],
        label_col=label_col,
    )
    nodes = ensure_same_nodes(df_normal_clean, df_accident_clean, node_col)
    idx_to_node = {i: node for i, node in enumerate(nodes)}

    x_normal, normal_timesteps = df_to_tensor(
        df_normal_clean,
        feature_cols,
        nodes,
        time_col=time_col,
        node_col=node_col,
    )
    x_accident, accident_timesteps = df_to_tensor(
        df_accident_clean,
        feature_cols,
        nodes,
        time_col=time_col,
        node_col=node_col,
    )
    accident_time_labels = make_time_labels(
        df_accident_clean,
        accident_timesteps,
        time_col=time_col,
        label_col=label_col,
    )

    split = split_normal_data(
        x_normal,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
    )
    scaler = fit_scaler(split.x_train)
    x_train_scaled = scale_tensor(split.x_train, scaler)
    x_val_scaled = scale_tensor(split.x_val, scaler)
    x_test_normal_scaled = scale_tensor(split.x_test_normal, scaler)
    x_accident_scaled = scale_tensor(x_accident, scaler)

    _, n_nodes, n_features = split.x_train.shape
    seq_size = int(config["seq_size"])
    x_train_flat = flatten_node_feature(x_train_scaled)
    x_val_flat = flatten_node_feature(x_val_scaled)
    x_test_normal_flat = flatten_node_feature(x_test_normal_scaled)
    x_accident_flat = flatten_node_feature(x_accident_scaled)

    train_x = make_windows(x_train_flat, seq_size)
    val_x = make_windows(x_val_flat, seq_size)
    test_normal_x = make_windows(x_test_normal_flat, seq_size)
    accident_x, accident_window_labels = make_windows_with_labels(
        x_accident_flat,
        accident_time_labels,
        seq_size,
    )

    train_loader = make_loader(train_x, config["batch_size"], shuffle=True)
    val_loader = make_loader(val_x, config["batch_size"], shuffle=False)
    test_normal_loader = make_loader(test_normal_x, config["batch_size"], shuffle=False)
    accident_loader = make_loader(accident_x, config["batch_size"], shuffle=False)

    input_dim = train_x.shape[2]
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    model, history, _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["epochs"],
        lr=config["lr"],
        patience=config["patience"],
        device=device,
    )
    plot_training_history(history, output_dir)

    val_batch_x, val_recon = predict_reconstruction(model, val_loader, device)
    test_normal_batch_x, test_normal_recon = predict_reconstruction(model, test_normal_loader, device)
    accident_batch_x, accident_recon = predict_reconstruction(model, accident_loader, device)

    val_weighted, _ = compute_weighted_node_max_scores(
        val_batch_x,
        val_recon,
        n_nodes,
        n_features,
        seq_size,
        feature_weights,
    )
    test_normal_weighted, _ = compute_weighted_node_max_scores(
        test_normal_batch_x,
        test_normal_recon,
        n_nodes,
        n_features,
        seq_size,
        feature_weights,
    )
    accident_weighted, accident_weighted_node_scores = compute_weighted_node_max_scores(
        accident_batch_x,
        accident_recon,
        n_nodes,
        n_features,
        seq_size,
        feature_weights,
    )

    min_consecutive = int(config["min_consecutive"])
    cooldown_steps = int(config["cooldown_steps"])
    step_seconds = infer_step_seconds(normal_timesteps)
    event_alert_col = str(config.get("event_alert_col", "alert"))
    raw_event_gap_seconds = config.get("event_gap_seconds")
    if raw_event_gap_seconds is not None:
        event_gap_seconds = float(raw_event_gap_seconds)
    else:
        event_gap_multiplier = float(config.get("event_gap_multiplier", 1.0))
        event_gap_seconds = float(step_seconds * event_gap_multiplier) if not np.isnan(step_seconds) else np.nan

    test_normal_timesteps = normal_timesteps[split.val_end :]
    test_normal_window_end_begin = np.array(test_normal_timesteps[seq_size - 1 :])

    rows: list[dict[str, Any]] = []
    for p in percentiles:
        threshold = select_threshold(val_weighted, p)
        normal_pred_window = test_normal_weighted > threshold
        accident_pred_window = accident_weighted > threshold
        normal_alerts = apply_alert_logic(
            test_normal_weighted,
            threshold,
            min_consecutive=min_consecutive,
            cooldown_steps=cooldown_steps,
        )
        accident_alerts = apply_alert_logic(
            accident_weighted,
            threshold,
            min_consecutive=min_consecutive,
            cooldown_steps=cooldown_steps,
        )

        result_df = build_accident_result_df(
            accident_weighted=accident_weighted,
            selected_threshold=threshold,
            accident_pred_window=accident_pred_window,
            accident_alerts=accident_alerts,
            accident_window_labels=accident_window_labels,
            accident_weighted_node_scores=accident_weighted_node_scores,
            accident_timesteps=accident_timesteps,
            seq_size=seq_size,
            idx_to_node=idx_to_node,
        )
        if event_alert_col not in result_df.columns:
            raise ValueError(f"Missing event alert column '{event_alert_col}' in result_df.")

        event_eval_alert = evaluate_event_level(
            result_df,
            alert_col=event_alert_col,
            gap_threshold=event_gap_seconds,
        )
        summary, _ = summarize_metrics(
            score_name=config["score_name"],
            threshold_percentile=p,
            selected_threshold=threshold,
            min_consecutive=min_consecutive,
            cooldown_steps=cooldown_steps,
            step_seconds=step_seconds,
            normal_pred_window=normal_pred_window,
            accident_pred_window=accident_pred_window,
            normal_alerts=normal_alerts,
            test_normal_window_end_begin=test_normal_window_end_begin,
            event_eval_alert=event_eval_alert,
        )
        row = {
            "threshold_percentile": float(p),
            "threshold_label": _fmt_percentile_label(float(p)),
            "selected_threshold": float(summary["threshold"]),
            "event_recall": float(summary["event_recall"]),
            "false_alerts_per_day": float(summary["false_alerts_per_day"]),
            "mean_delay_seconds": float(summary["mean_delay_seconds"]),
            "median_delay_seconds": float(summary["median_delay_seconds"]),
            "normal_fp_rate": float(summary["normal_fp_rate"]),
            "false_alerts_per_hour": float(summary["false_alerts_per_hour"]),
            "detected_events": int(summary["detected_events"]),
            "total_events": int(summary["total_events"]),
        }
        rows.append(row)
        print(
            f"P{p:g}: recall={row['event_recall']:.4f}, "
            f"FP/day={row['false_alerts_per_day']:.4f}, "
            f"mean_delay_s={row['mean_delay_seconds']:.3f}"
        )

    sensitivity_df = pd.DataFrame(rows).sort_values("threshold_percentile").reset_index(drop=True)
    csv_path = output_dir / args.csv_name
    plot_path = output_dir / args.plot_name
    summary_json_path = output_dir / args.summary_json_name

    sensitivity_df.to_csv(csv_path, index=False)
    plot_threshold_sensitivity(df=sensitivity_df, output_path=plot_path)

    payload = {
        "config_path": args.config,
        "output_dir": str(output_dir),
        "percentiles": percentiles,
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
        "training_epochs": int(config["epochs"]),
        "device": str(device),
        "rows": sensitivity_df.to_dict(orient="records"),
    }
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Saved CSV:", csv_path)
    print("Saved Figure 8 plot:", plot_path)
    print("Saved summary:", summary_json_path)


if __name__ == "__main__":
    main()
