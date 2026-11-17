from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def extract_domain_features(x: np.ndarray) -> np.ndarray:
    continuous = x[..., :4].astype(np.float64)
    history_mean = continuous.mean(axis=1)
    last_deviation = continuous[:, -1] - history_mean
    history_std = continuous.std(axis=1)
    if continuous.shape[1] > 1:
        time = np.linspace(-1.0, 1.0, continuous.shape[1], dtype=np.float64)
        denominator = np.square(time).sum()
        slope = ((continuous - history_mean[:, None]) * time[None, :, None, None]).sum(
            axis=1,
        ) / max(denominator, 1e-6)
        max_change = np.abs(np.diff(continuous, axis=1)).max(axis=1)
    else:
        slope = np.zeros_like(history_mean)
        max_change = np.zeros_like(history_mean)
    statistics = np.stack([history_mean, last_deviation, history_std, slope, max_change], axis=-1)
    return np.concatenate(
        [
            statistics.mean(axis=1).reshape(len(x), -1),
            statistics.max(axis=1).reshape(len(x), -1),
        ],
        axis=1,
    ).astype(np.float32)


@dataclass
class DomainGate:
    model: LogisticRegression
    threshold: float

    def predict_mixed_probability(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(extract_domain_features(x))[:, 1]

    def predict_is_mixed(self, x: np.ndarray) -> np.ndarray:
        return self.predict_mixed_probability(x) >= self.threshold

    def to_manifest(self) -> dict[str, Any]:
        return {
            "model_type": "logistic_regression",
            "threshold": float(self.threshold),
            "classes": ["source_like", "mixed_like"],
            "feature_set": "window_mean_lastdev_std_slope_maxchange_meanmax",
        }


def train_domain_gate(
    train_x: np.ndarray,
    train_is_mixed: np.ndarray,
    val_x: np.ndarray,
    val_is_mixed: np.ndarray,
    *,
    minimum_mixed_recall: float = 0.98,
) -> tuple[DomainGate, dict[str, Any]]:
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(extract_domain_features(train_x), train_is_mixed.astype(int))
    val_probability = model.predict_proba(extract_domain_features(val_x))[:, 1]
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 91):
        prediction = val_probability >= threshold
        mixed_recall = float(
            (prediction[val_is_mixed].mean()) if val_is_mixed.any() else 1.0
        )
        source_recall = float(
            ((~prediction[~val_is_mixed]).mean()) if (~val_is_mixed).any() else 1.0
        )
        feasible = mixed_recall >= minimum_mixed_recall
        rank = (float(feasible), source_recall, mixed_recall, -abs(float(threshold) - 0.5))
        candidates.append((rank, float(threshold), prediction))
    _, threshold, prediction = max(candidates, key=lambda item: item[0])
    gate = DomainGate(model=model, threshold=threshold)
    metrics = {
        "accuracy": float(accuracy_score(val_is_mixed, prediction)),
        "mixed_recall": float(prediction[val_is_mixed].mean()) if val_is_mixed.any() else 1.0,
        "source_recall": float((~prediction[~val_is_mixed]).mean()) if (~val_is_mixed).any() else 1.0,
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(
            val_is_mixed.astype(int),
            prediction.astype(int),
            labels=[0, 1],
        ).tolist(),
    }
    return gate, metrics
