from __future__ import annotations

from typing import Any

import numpy as np


def build_events(
    labels: np.ndarray,
    begins: np.ndarray,
    *,
    step_seconds: float = 300.0,
) -> list[tuple[float, float]]:
    order = np.argsort(begins)
    events: list[tuple[float, float]] = []
    start: float | None = None
    end: float | None = None
    for index in order:
        begin = float(begins[index])
        if bool(labels[index]):
            if start is None or end is None or begin - end > step_seconds:
                if start is not None and end is not None:
                    events.append((start, end))
                start = begin
            end = begin
        elif start is not None and end is not None:
            events.append((start, end))
            start = None
            end = None
    if start is not None and end is not None:
        events.append((start, end))
    return events


def apply_temporal_policy(
    probabilities: np.ndarray,
    begins: np.ndarray,
    policy: dict[str, Any],
) -> np.ndarray:
    order = np.argsort(begins)
    alerts = np.zeros(len(probabilities), dtype=bool)
    threshold = float(policy["threshold"])
    emergency_threshold = float(policy["emergency_threshold"])
    confirmation_windows = int(policy.get("confirmation_windows", 2))
    rolling_windows = int(policy.get("rolling_windows", 2))
    cooldown_seconds = float(policy.get("cooldown_seconds", 1800.0))
    step_seconds = float(policy.get("step_seconds", 300.0))

    rolling_scores: list[float] = []
    consecutive = 0
    previous_begin: float | None = None
    last_alert = -np.inf
    for index in order:
        begin = float(begins[index])
        if previous_begin is None or begin - previous_begin > step_seconds:
            rolling_scores = []
            consecutive = 0
        rolling_scores.append(float(probabilities[index]))
        rolling_scores = rolling_scores[-rolling_windows:]
        rolling_score = max(rolling_scores)
        consecutive = consecutive + 1 if rolling_score >= threshold else 0
        confirmed = consecutive >= confirmation_windows
        emergency = float(probabilities[index]) >= emergency_threshold
        if (confirmed or emergency) and begin - last_alert >= cooldown_seconds:
            alerts[index] = True
            last_alert = begin
            consecutive = 0
        previous_begin = begin
    return alerts


def apply_active_state_policy(
    probabilities: np.ndarray,
    begins: np.ndarray,
    policy: dict[str, Any],
) -> np.ndarray:
    order = np.argsort(begins)
    active_windows = np.zeros(len(probabilities), dtype=bool)
    enter_threshold = float(policy["enter_threshold"])
    exit_threshold = float(policy["exit_threshold"])
    enter_confirmation = int(policy.get("enter_confirmation_windows", 1))
    release_confirmation = int(policy.get("release_confirmation_windows", 2))
    rolling_windows = int(policy.get("rolling_windows", 1))
    step_seconds = float(policy.get("step_seconds", 300.0))

    rolling_scores: list[float] = []
    enter_count = 0
    release_count = 0
    is_active = False
    previous_begin: float | None = None
    for index in order:
        begin = float(begins[index])
        if previous_begin is None or begin - previous_begin > step_seconds:
            rolling_scores = []
            enter_count = 0
            release_count = 0
            is_active = False

        rolling_scores.append(float(probabilities[index]))
        rolling_scores = rolling_scores[-rolling_windows:]
        rolling_score = max(rolling_scores)
        if not is_active:
            enter_count = enter_count + 1 if rolling_score >= enter_threshold else 0
            if enter_count >= enter_confirmation:
                is_active = True
                release_count = 0
        else:
            release_count = (
                release_count + 1
                if rolling_score < exit_threshold
                else 0
            )
            if release_count >= release_confirmation:
                is_active = False
                enter_count = 0

        active_windows[index] = is_active
        previous_begin = begin
    return active_windows


def binary_window_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    labels_bool = labels.astype(bool)
    predictions_bool = predictions.astype(bool)
    true_positive = int(np.sum(labels_bool & predictions_bool))
    false_positive = int(np.sum(~labels_bool & predictions_bool))
    false_negative = int(np.sum(labels_bool & ~predictions_bool))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def calibrate_active_state_policy(
    labels: np.ndarray,
    probabilities: np.ndarray,
    begins: np.ndarray,
    base_policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float | int]]:
    candidates = []
    for rolling_windows in base_policy.get("rolling_windows_options", [1, 2]):
        for enter_confirmation in base_policy.get(
            "enter_confirmation_windows_options",
            [1, 2],
        ):
            for release_confirmation in base_policy.get(
                "release_confirmation_windows_options",
                [1, 2, 3],
            ):
                for enter_threshold in base_policy.get(
                    "enter_threshold_values",
                    np.linspace(0.40, 0.75, 36),
                ):
                    for exit_threshold in base_policy.get(
                        "exit_threshold_values",
                        np.linspace(0.30, 0.65, 36),
                    ):
                        if exit_threshold > enter_threshold:
                            continue
                        policy = {
                            **base_policy,
                            "rolling_windows": int(rolling_windows),
                            "enter_confirmation_windows": int(enter_confirmation),
                            "release_confirmation_windows": int(release_confirmation),
                            "enter_threshold": float(enter_threshold),
                            "exit_threshold": float(exit_threshold),
                        }
                        active = apply_active_state_policy(
                            probabilities,
                            begins,
                            policy,
                        )
                        metrics = binary_window_metrics(labels, active)
                        rank = (
                            float(metrics["f1"]),
                            float(metrics["recall"]),
                            float(metrics["precision"]),
                            -float(enter_threshold),
                        )
                        candidates.append((rank, policy, metrics))
    _, selected_policy, selected_metrics = max(
        candidates,
        key=lambda item: item[0],
    )
    return selected_policy, selected_metrics


def event_metrics(
    labels: np.ndarray,
    alerts: np.ndarray,
    begins: np.ndarray,
    *,
    max_latency_seconds: float = 600.0,
    step_seconds: float = 300.0,
) -> dict[str, float | int]:
    events = build_events(labels, begins, step_seconds=step_seconds)
    alert_times = np.sort(begins[alerts].astype(float))
    matched_alerts: set[int] = set()
    latencies: list[float] = []
    detected = 0
    for start, end in events:
        candidates = np.where(
            (alert_times >= start)
            & (alert_times <= end)
            & (alert_times - start <= max_latency_seconds)
        )[0]
        candidates = [index for index in candidates if index not in matched_alerts]
        if candidates:
            alert_index = candidates[0]
            matched_alerts.add(alert_index)
            detected += 1
            latencies.append(float(alert_times[alert_index] - start))

    false_alerts = len(alert_times) - len(matched_alerts)
    day_count = max(1, len(np.unique((begins.astype(float) // 86400).astype(int))))
    precision = detected / len(alert_times) if len(alert_times) else 0.0
    recall = detected / len(events) if events else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "event_count": len(events),
        "detected_events": detected,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "false_alerts": false_alerts,
        "false_alerts_per_day": false_alerts / day_count,
        "median_latency_seconds": float(np.median(latencies)) if latencies else -1.0,
        "p90_latency_seconds": float(np.percentile(latencies, 90)) if latencies else -1.0,
    }


def daily_event_metrics(
    labels: np.ndarray,
    alerts: np.ndarray,
    begins: np.ndarray,
    *,
    max_latency_seconds: float = 600.0,
    step_seconds: float = 300.0,
) -> list[dict[str, float | int]]:
    rows = []
    day_indices = (begins.astype(float) // 86400).astype(int)
    for day_index in np.unique(day_indices):
        day_mask = day_indices == day_index
        metrics = event_metrics(
            labels[day_mask],
            alerts[day_mask],
            begins[day_mask],
            max_latency_seconds=max_latency_seconds,
            step_seconds=step_seconds,
        )
        rows.append({"day_index": int(day_index), **metrics})
    return rows


def calibrate_temporal_policy(
    labels: np.ndarray,
    probabilities: np.ndarray,
    begins: np.ndarray,
    base_policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float | int]]:
    max_false_alerts = float(base_policy.get("max_false_alerts_per_day", 2.0))
    candidates: list[tuple[tuple[float, ...], dict[str, Any], dict[str, float | int]]] = []
    rolling_options = base_policy.get("rolling_windows_options", [1, 2, 3])
    confirmation_options = base_policy.get("confirmation_windows_options", [1, 2])
    cooldown_options = base_policy.get("cooldown_seconds_options", [900, 1800])
    threshold_values = base_policy.get(
        "threshold_values",
        np.linspace(0.05, 0.95, 91),
    )
    emergency_deltas = base_policy.get("emergency_deltas", [0.10, 0.20, 0.30])
    for rolling_windows in rolling_options:
        for confirmation_windows in confirmation_options:
            for cooldown_seconds in cooldown_options:
                for threshold in threshold_values:
                    for emergency_delta in emergency_deltas:
                        emergency_threshold = min(0.99, threshold + emergency_delta)
                        policy = {
                            **base_policy,
                            "rolling_windows": int(rolling_windows),
                            "confirmation_windows": int(confirmation_windows),
                            "cooldown_seconds": float(cooldown_seconds),
                            "threshold": float(threshold),
                            "emergency_threshold": float(emergency_threshold),
                        }
                        alerts = apply_temporal_policy(probabilities, begins, policy)
                        metrics = event_metrics(
                            labels,
                            alerts,
                            begins,
                            max_latency_seconds=float(
                                policy.get("max_latency_seconds", 600.0)
                            ),
                            step_seconds=float(policy.get("step_seconds", 300.0)),
                        )
                        daily = daily_event_metrics(
                            labels,
                            alerts,
                            begins,
                            max_latency_seconds=float(
                                policy.get("max_latency_seconds", 600.0)
                            ),
                            step_seconds=float(policy.get("step_seconds", 300.0)),
                        )
                        max_daily_false_alerts = max(
                            (float(row["false_alerts"]) for row in daily),
                            default=0.0,
                        )
                        metrics["max_daily_false_alerts"] = max_daily_false_alerts
                        feasible = max_daily_false_alerts <= max_false_alerts
                        rank = (
                            float(feasible),
                            float(metrics["event_recall"]),
                            float(metrics["event_f1"]),
                            float(metrics["event_precision"]),
                            -max_daily_false_alerts,
                            -float(threshold),
                        )
                        candidates.append((rank, policy, metrics))
    _, policy, metrics = max(candidates, key=lambda item: item[0])
    return policy, metrics
