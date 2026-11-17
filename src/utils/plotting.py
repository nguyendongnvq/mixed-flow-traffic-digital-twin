from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _savefig(output_dir: Path, filename: str) -> None:
    output_path = output_dir / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_training_history(history: dict[str, list[float]], output_dir: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MAE Loss")
    plt.grid(alpha=0.3)
    _savefig(output_dir, "training_validation_loss.png")


def plot_score_distribution(
    test_normal_weighted: np.ndarray,
    accident_weighted: np.ndarray,
    threshold: float,
    threshold_percentile: float,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(test_normal_weighted, bins=50, alpha=0.6, density=True, label="normal test")
    plt.hist(accident_weighted, bins=50, alpha=0.6, density=True, label="accident")
    plt.axvline(threshold, linestyle="--", label=f"P{threshold_percentile} threshold")
    plt.title("Weighted node-max score distribution")
    plt.xlabel("Anomaly score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    _savefig(output_dir, "score_distribution.png")


def plot_score_timeline(
    scores: np.ndarray,
    threshold: float,
    alerts: np.ndarray,
    title: str,
    output_name: str,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(14, 4))
    plt.plot(scores, label="score")
    plt.axhline(threshold, linestyle="--", label="threshold")
    alert_idx = np.where(alerts)[0]
    plt.scatter(alert_idx, scores[alert_idx], s=35, label="alerts")
    plt.title(title)
    plt.xlabel("Window index")
    plt.ylabel("Weighted node-max score")
    plt.legend()
    plt.grid(alpha=0.3)
    _savefig(output_dir, output_name)


def plot_summary_rates(
    window_detect_ratio: float,
    event_recall: float,
    normal_fp_rate: float,
    output_dir: Path,
) -> None:
    rate_names = ["Window detect ratio", "Event recall", "Normal FP rate"]
    rate_values = [window_detect_ratio * 100, event_recall * 100, normal_fp_rate * 100]

    plt.figure(figsize=(8, 5))
    plt.bar(rate_names, rate_values)
    plt.ylabel("Percent (%)")
    plt.title("Detector summary rates")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)
    for i, value in enumerate(rate_values):
        plt.text(i, value + 1, f"{value:.2f}%", ha="center")
    _savefig(output_dir, "summary_rates.png")


def plot_delay_charts(
    mean_delay_seconds: float,
    median_delay_seconds: float,
    event_eval_alert: pd.DataFrame,
    output_dir: Path,
) -> None:
    delay_names = ["Mean delay", "Median delay"]
    delay_values = [mean_delay_seconds / 60, median_delay_seconds / 60]

    plt.figure(figsize=(6, 4))
    plt.bar(delay_names, delay_values)
    plt.ylabel("Minutes")
    plt.title("Detection delay")
    plt.grid(axis="y", alpha=0.3)
    for i, value in enumerate(delay_values):
        plt.text(i, value + 0.5, f"{value:.1f} min", ha="center")
    _savefig(output_dir, "delay_summary.png")

    delay_values_seconds = event_eval_alert["delay"].dropna().values
    plt.figure(figsize=(9, 4))
    plt.hist(delay_values_seconds / 60, bins=30)
    plt.xlabel("Delay minutes")
    plt.ylabel("Event count")
    plt.title("Detection delay distribution")
    plt.grid(alpha=0.3)
    _savefig(output_dir, "delay_distribution.png")


def plot_top_alert_nodes(result_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    alert_node_counts = (
        result_df[result_df["alert"]]
        .groupby("top_node_id")
        .size()
        .sort_values(ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10, 5))
    plt.barh(alert_node_counts.index[::-1], alert_node_counts.values[::-1])
    plt.xlabel("Alert count")
    plt.title("Top nodes causing alerts")
    plt.grid(axis="x", alpha=0.3)
    _savefig(output_dir, "top_alert_nodes.png")

    return alert_node_counts.reset_index(name="alert_count")
