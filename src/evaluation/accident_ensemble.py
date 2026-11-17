from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.accident_event_detection import apply_temporal_policy


def aggregate_ensemble_probabilities(
    probabilities: np.ndarray,
    method: str,
    *,
    vote_thresholds: np.ndarray | None = None,
) -> np.ndarray:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [models, samples]")
    if method == "mean":
        return probabilities.mean(axis=0)
    if method == "max":
        return probabilities.max(axis=0)
    if method == "vote":
        if vote_thresholds is None or len(vote_thresholds) != probabilities.shape[0]:
            raise ValueError("vote requires one threshold per model")
        return (
            probabilities >= vote_thresholds.reshape(-1, 1)
        ).mean(axis=0)
    raise ValueError(f"Unsupported ensemble method: {method}")


def apply_vote_temporal_policy(
    probabilities: np.ndarray,
    begins: np.ndarray,
    policy: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.asarray(policy["model_thresholds"], dtype=np.float32)
    vote_fraction = aggregate_ensemble_probabilities(
        probabilities,
        "vote",
        vote_thresholds=thresholds,
    )
    alerts = apply_temporal_policy(vote_fraction, begins, policy)
    return vote_fraction, alerts
