from __future__ import annotations

from typing import Any
from itertools import product

import numpy as np


DEFAULT_SINGLE_LABEL_POLICY = {
    "priority_label": "accident",
    "selection": "threshold_ratio",
    "tie_break_order": [
        "congestion_spike",
        "zigzag",
        "sudden_stop",
    ],
    "class_weights": {},
}


def normalize_single_label_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(DEFAULT_SINGLE_LABEL_POLICY)
    if policy:
        normalized.update(policy)
    if normalized["selection"] != "threshold_ratio":
        raise ValueError(
            f"Unsupported single-label selection: {normalized['selection']}"
        )
    normalized["tie_break_order"] = list(normalized["tie_break_order"])
    normalized["class_weights"] = {
        str(class_name): float(weight)
        for class_name, weight in normalized.get("class_weights", {}).items()
    }
    return normalized


def apply_unknown_temporal_policy(
    residuals: np.ndarray,
    begins: np.ndarray,
    policy: dict[str, Any],
) -> np.ndarray:
    order = np.argsort(begins)
    predictions = np.zeros(len(residuals), dtype=bool)
    threshold = float(policy["threshold"])
    rolling_windows = int(policy.get("rolling_windows", 1))
    confirmation_windows = int(policy.get("confirmation_windows", 1))
    step_seconds = float(policy.get("step_seconds", 300.0))
    rolling_scores: list[float] = []
    consecutive = 0
    previous_begin: float | None = None
    for index in order:
        begin = float(begins[index])
        if previous_begin is None or begin - previous_begin > step_seconds:
            rolling_scores = []
            consecutive = 0
        rolling_scores.append(float(residuals[index]))
        rolling_scores = rolling_scores[-rolling_windows:]
        rolling_score = float(np.mean(rolling_scores))
        consecutive = consecutive + 1 if rolling_score >= threshold else 0
        predictions[index] = consecutive >= confirmation_windows
        previous_begin = begin
    return predictions


def calibrate_unknown_temporal_policy(
    residuals: np.ndarray,
    begins: np.ndarray,
    normal_mask: np.ndarray,
    base_policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    target_fp_rate = float(base_policy.get("target_normal_fp_rate", 0.05))
    normal_residuals = residuals[normal_mask]
    percentile_values = base_policy.get(
        "threshold_percentiles",
        [90.0, 92.5, 95.0, 97.5, 99.0],
    )
    candidates = []
    for rolling_windows in base_policy.get("rolling_windows_options", [1, 2, 3]):
        for confirmation_windows in base_policy.get(
            "confirmation_windows_options",
            [1, 2],
        ):
            for percentile in percentile_values:
                threshold = float(np.percentile(normal_residuals, percentile))
                policy = {
                    **base_policy,
                    "rolling_windows": int(rolling_windows),
                    "confirmation_windows": int(confirmation_windows),
                    "threshold_percentile": float(percentile),
                    "threshold": threshold,
                }
                predictions = apply_unknown_temporal_policy(
                    residuals,
                    begins,
                    policy,
                )
                fp_rate = float(predictions[normal_mask].mean())
                feasible = fp_rate <= target_fp_rate
                rank = (
                    float(feasible),
                    -threshold if feasible else -fp_rate,
                    -float(confirmation_windows),
                    -float(rolling_windows),
                )
                candidates.append(
                    (
                        rank,
                        policy,
                        {"normal_false_positive_rate": fp_rate},
                    )
                )
    _, selected_policy, selected_metrics = max(
        candidates,
        key=lambda item: item[0],
    )
    return selected_policy, selected_metrics


def select_primary_label(
    class_names: list[str],
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    predictions: np.ndarray,
    *,
    residual: float,
    generic_threshold: float,
    unknown_prediction: bool | None = None,
    policy: dict[str, Any] | None = None,
) -> str:
    normalized = normalize_single_label_policy(policy)
    detected_indices = np.flatnonzero(predictions)
    if detected_indices.size == 0:
        is_unknown = (
            bool(unknown_prediction)
            if unknown_prediction is not None
            else residual > generic_threshold
        )
        return "unknown_anomaly" if is_unknown else "normal"

    priority_label = str(normalized["priority_label"])
    if priority_label in class_names:
        priority_index = class_names.index(priority_label)
        if bool(predictions[priority_index]):
            return priority_label

    tie_break_order = {
        class_name: index
        for index, class_name in enumerate(normalized["tie_break_order"])
    }
    ranked = sorted(
        detected_indices.tolist(),
        key=lambda class_index: (
            -float(
                probabilities[class_index]
                / thresholds[class_index]
                * normalized["class_weights"].get(
                    class_names[class_index],
                    1.0,
                )
            ),
            -float(probabilities[class_index]),
            tie_break_order.get(class_names[class_index], len(tie_break_order)),
            class_names[class_index],
        ),
    )
    return class_names[ranked[0]]


def select_primary_labels(
    class_names: list[str],
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    predictions: np.ndarray,
    residuals: np.ndarray,
    *,
    generic_threshold: float,
    unknown_predictions: np.ndarray | None = None,
    policy: dict[str, Any] | None = None,
) -> np.ndarray:
    return np.asarray(
        [
            select_primary_label(
                class_names,
                probabilities[sample_index],
                thresholds,
                predictions[sample_index],
                residual=float(residuals[sample_index]),
                generic_threshold=generic_threshold,
                unknown_prediction=(
                    bool(unknown_predictions[sample_index])
                    if unknown_predictions is not None
                    else None
                ),
                policy=policy,
            )
            for sample_index in range(len(probabilities))
        ],
        dtype=object,
    )


def primary_label_metrics(
    labels: np.ndarray,
    primary_labels: np.ndarray,
    class_names: list[str],
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_single_label_policy(policy)
    priority_order = [
        str(normalized["priority_label"]),
        *[
            str(class_name)
            for class_name in normalized["tie_break_order"]
            if class_name != normalized["priority_label"]
        ],
    ]
    priority_order.extend(
        class_name for class_name in class_names if class_name not in priority_order
    )

    expected = []
    hits = []
    for sample_labels, primary_label in zip(labels, primary_labels, strict=True):
        active = {
            class_names[class_index]
            for class_index in np.flatnonzero(sample_labels)
        }
        if active:
            expected.append(next(label for label in priority_order if label in active))
            hits.append(str(primary_label) in active)
        else:
            expected.append("normal")
            hits.append(str(primary_label) == "normal")

    output_labels = [
        "normal",
        "unknown_anomaly",
        *priority_order,
    ]
    confusion_matrix = {
        expected_label: {
            predicted_label: 0
            for predicted_label in output_labels
        }
        for expected_label in ["normal", *priority_order]
    }
    for expected_label, predicted_label in zip(expected, primary_labels, strict=True):
        confusion_matrix[expected_label][str(predicted_label)] += 1

    return {
        "primary_label_hit_rate": float(np.mean(hits)),
        "primary_label_confusion_matrix": confusion_matrix,
    }


def primary_label_class_metrics(
    labels: np.ndarray,
    primary_labels: np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, float | int]]:
    rows = {}
    for class_index, class_name in enumerate(class_names):
        truth = labels[:, class_index].astype(bool)
        prediction = primary_labels == class_name
        true_positive = int(np.sum(truth & prediction))
        false_positive = int(np.sum(~truth & prediction))
        false_negative = int(np.sum(truth & ~prediction))
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
        rows[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(truth.sum()),
            "selected_count": int(prediction.sum()),
        }
    return rows


def calibrate_single_label_policy(
    labels: np.ndarray,
    class_names: list[str],
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    predictions: np.ndarray,
    residuals: np.ndarray,
    unknown_predictions: np.ndarray,
    base_policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = normalize_single_label_policy(base_policy)
    priority_label = str(normalized["priority_label"])
    weighted_classes = [
        class_name
        for class_name in class_names
        if class_name != priority_label
    ]
    weight_values = [
        float(value)
        for value in base_policy.get(
            "weight_values",
            [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        )
    ]
    candidates = []
    for values in product(weight_values, repeat=len(weighted_classes)):
        policy = {
            **normalized,
            "class_weights": dict(zip(weighted_classes, values, strict=True)),
        }
        primary_labels = select_primary_labels(
            class_names,
            probabilities,
            thresholds,
            predictions,
            residuals,
            generic_threshold=float("inf"),
            unknown_predictions=unknown_predictions,
            policy=policy,
        )
        metrics = primary_label_metrics(
            labels,
            primary_labels,
            class_names,
            policy=policy,
        )
        class_metrics = primary_label_class_metrics(
            labels,
            primary_labels,
            class_names,
        )
        non_starved = all(
            row["support"] == 0 or row["selected_count"] > 0
            for row in class_metrics.values()
        )
        supported_recalls = [
            float(row["recall"])
            for row in class_metrics.values()
            if row["support"] > 0
        ]
        rank = (
            float(non_starved),
            float(metrics["primary_label_hit_rate"]),
            min(supported_recalls, default=0.0),
            -sum(abs(weight - 1.0) for weight in values),
        )
        candidates.append(
            (
                rank,
                policy,
                {
                    **metrics,
                    "class_metrics": class_metrics,
                    "starvation_free": non_starved,
                },
            )
        )
    _, selected_policy, selected_metrics = max(
        candidates,
        key=lambda item: item[0],
    )
    return selected_policy, selected_metrics


def calibrate_secondary_thresholds(
    labels: np.ndarray,
    class_names: list[str],
    probabilities: np.ndarray,
    base_thresholds: np.ndarray,
    accident_predictions: np.ndarray,
    multiplier_values: list[float] | None = None,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    accident_index = class_names.index("accident")
    secondary_indices = [
        class_index
        for class_index in range(len(class_names))
        if class_index != accident_index
    ]
    values = multiplier_values or [1.0, 1.1, 1.2, 1.3, 1.5]
    true_known = labels.astype(bool).any(axis=1)
    candidates = []
    for multipliers in product(values, repeat=len(secondary_indices)):
        effective_thresholds = base_thresholds.astype(float).copy()
        for class_index, multiplier in zip(
            secondary_indices,
            multipliers,
            strict=True,
        ):
            effective_thresholds[class_index] *= float(multiplier)
        predictions = probabilities >= effective_thresholds.reshape(1, -1)
        predictions[:, accident_index] = accident_predictions
        predicted_known = predictions.any(axis=1)
        true_positive = int(np.sum(true_known & predicted_known))
        false_positive = int(np.sum(~true_known & predicted_known))
        false_negative = int(np.sum(true_known & ~predicted_known))
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
        secondary_detected = all(
            np.any(
                predictions[:, class_index]
                & labels[:, class_index].astype(bool)
            )
            for class_index in secondary_indices
            if labels[:, class_index].sum() > 0
        )
        rank = (
            float(secondary_detected),
            f1,
            precision,
            recall,
            -sum(abs(float(value) - 1.0) for value in multipliers),
        )
        candidates.append(
            (
                rank,
                effective_thresholds,
                {
                    class_names[class_index]: float(multiplier)
                    for class_index, multiplier in zip(
                        secondary_indices,
                        multipliers,
                        strict=True,
                    )
                },
                {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                },
            )
        )
    _, thresholds, multipliers, metrics = max(
        candidates,
        key=lambda item: item[0],
    )
    return thresholds, multipliers, metrics
