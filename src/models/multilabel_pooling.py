from __future__ import annotations

from typing import Any

import numpy as np


def aggregate_node_probabilities(
    node_probabilities: np.ndarray,
    class_names: list[str],
    pooling_config: dict[str, Any] | None,
) -> np.ndarray:
    global_probabilities = node_probabilities.max(axis=1)
    if not pooling_config or not bool(pooling_config.get("enabled", False)):
        return global_probabilities

    class_name = str(pooling_config.get("class_name", "accident"))
    if class_name not in class_names:
        return global_probabilities

    class_idx = class_names.index(class_name)
    top_k = min(int(pooling_config.get("top_k", 3)), node_probabilities.shape[1])
    max_weight = float(pooling_config.get("max_weight", 0.7))
    top_k_weight = float(pooling_config.get("top_k_weight", 0.3))
    if top_k <= 0:
        raise ValueError("pooling top_k must be positive")
    if not np.isclose(max_weight + top_k_weight, 1.0):
        raise ValueError("pooling weights must sum to 1.0")

    class_node_probabilities = node_probabilities[:, :, class_idx]
    top_k_probabilities = np.partition(
        class_node_probabilities,
        kth=class_node_probabilities.shape[1] - top_k,
        axis=1,
    )[:, -top_k:]
    global_probabilities[:, class_idx] = (
        max_weight * class_node_probabilities.max(axis=1)
        + top_k_weight * top_k_probabilities.mean(axis=1)
    )
    return global_probabilities
