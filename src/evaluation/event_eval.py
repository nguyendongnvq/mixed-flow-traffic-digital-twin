from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def infer_step_seconds(timesteps: Sequence[float]) -> float:
    ts = np.array(sorted(timesteps), dtype=float)
    if len(ts) < 2:
        return np.nan
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return np.nan
    return float(np.median(diffs))


def assign_event_groups_from_begin(
    begins: Sequence[float],
    gap_threshold: float | None = None,
) -> np.ndarray:
    begins_arr = np.array(begins)
    if len(begins_arr) == 0:
        return np.array([], dtype=int)

    if gap_threshold is None:
        inferred_step = infer_step_seconds(begins_arr)
        gap_threshold = inferred_step if not np.isnan(inferred_step) else 300

    groups = []
    current_group = 0
    prev_t = None

    for t in begins_arr:
        if prev_t is not None and (t - prev_t) > gap_threshold:
            current_group += 1
        groups.append(current_group)
        prev_t = t

    return np.array(groups, dtype=int)


def evaluate_event_level(
    result_df: pd.DataFrame,
    alert_col: str = "alert",
    gap_threshold: float | None = None,
) -> pd.DataFrame:
    df = result_df.sort_values("begin").copy()
    df["event_group"] = assign_event_groups_from_begin(
        df["begin"].values,
        gap_threshold=gap_threshold,
    )

    rows = []
    for event_group, group_df in df.groupby("event_group"):
        event_start = group_df["begin"].min()
        event_end = group_df["begin"].max()
        detected_rows = group_df[group_df[alert_col] == True]  # noqa: E712

        if len(detected_rows) > 0:
            first_detect = detected_rows["begin"].min()
            delay = first_detect - event_start
            detected = True
        else:
            first_detect = np.nan
            delay = np.nan
            detected = False

        rows.append(
            {
                "event_group": event_group,
                "event_start": event_start,
                "event_end": event_end,
                "event_duration": event_end - event_start,
                "detected": detected,
                "first_detect": first_detect,
                "delay": delay,
                "num_windows": len(group_df),
                "num_detected_windows": int(group_df[alert_col].sum()),
            }
        )

    return pd.DataFrame(rows)


def build_accident_result_df(
    accident_weighted: np.ndarray,
    selected_threshold: float,
    accident_pred_window: np.ndarray,
    accident_alerts: np.ndarray,
    accident_window_labels: np.ndarray,
    accident_weighted_node_scores: np.ndarray,
    accident_timesteps: Sequence[float],
    seq_size: int,
    idx_to_node: dict[int, str],
    window_end_begin: np.ndarray | None = None,
) -> pd.DataFrame:
    top_node_idx = accident_weighted_node_scores.argmax(axis=1)
    top_node_id = [idx_to_node[i] for i in top_node_idx]
    if window_end_begin is None:
        accident_window_end_begin = np.array(accident_timesteps[seq_size - 1 :])
    else:
        accident_window_end_begin = np.asarray(window_end_begin)

    return pd.DataFrame(
        {
            "window_idx": np.arange(len(accident_weighted)),
            "begin": accident_window_end_begin,
            "anomaly_score": accident_weighted,
            "threshold": selected_threshold,
            "pred_window": accident_pred_window,
            "alert": accident_alerts,
            "true_window_label": accident_window_labels,
            "top_node_id": top_node_id,
        }
    )


def summarize_metrics(
    *,
    score_name: str,
    threshold_percentile: float,
    selected_threshold: float,
    min_consecutive: int,
    cooldown_steps: int,
    step_seconds: float,
    normal_pred_window: np.ndarray,
    accident_pred_window: np.ndarray,
    normal_alerts: np.ndarray,
    test_normal_window_end_begin: np.ndarray,
    event_eval_alert: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if len(test_normal_window_end_begin) > 1 and not np.isnan(step_seconds):
        normal_test_duration_seconds = (
            test_normal_window_end_begin.max()
            - test_normal_window_end_begin.min()
            + step_seconds
        )
    else:
        normal_test_duration_seconds = len(normal_alerts) * step_seconds

    normal_test_duration_hours = normal_test_duration_seconds / 3600
    normal_test_duration_days = normal_test_duration_hours / 24

    false_alerts = int(normal_alerts.sum())
    false_alerts_per_hour = false_alerts / normal_test_duration_hours
    false_alerts_per_day = false_alerts / normal_test_duration_days

    window_detect_ratio = float(accident_pred_window.mean())
    normal_fp_rate = float(normal_pred_window.mean())

    event_recall = float(event_eval_alert["detected"].mean())
    detected_events = int(event_eval_alert["detected"].sum())
    total_events = int(len(event_eval_alert))

    mean_delay_seconds = float(event_eval_alert["delay"].dropna().mean())
    median_delay_seconds = float(event_eval_alert["delay"].dropna().median())

    summary = {
        "score_name": score_name,
        "threshold_percentile": threshold_percentile,
        "threshold": selected_threshold,
        "min_consecutive": min_consecutive,
        "cooldown_steps": cooldown_steps,
        "step_seconds": step_seconds,
        "normal_fp_windows": int(normal_pred_window.sum()),
        "normal_total_windows": int(len(normal_pred_window)),
        "normal_fp_rate": normal_fp_rate,
        "accident_detected_windows": int(accident_pred_window.sum()),
        "accident_total_windows": int(len(accident_pred_window)),
        "window_detect_ratio": window_detect_ratio,
        "normal_alert_count": false_alerts,
        "normal_test_duration_hours": normal_test_duration_hours,
        "false_alerts_per_hour": false_alerts_per_hour,
        "false_alerts_per_day": false_alerts_per_day,
        "detected_events": detected_events,
        "total_events": total_events,
        "missed_events": total_events - detected_events,
        "event_recall": event_recall,
        "mean_delay_seconds": mean_delay_seconds,
        "median_delay_seconds": median_delay_seconds,
        "mean_delay_minutes": mean_delay_seconds / 60,
        "median_delay_minutes": median_delay_seconds / 60,
    }

    summary_df = pd.DataFrame([summary]).T.reset_index()
    summary_df.columns = ["metric", "value"]
    return summary, summary_df
