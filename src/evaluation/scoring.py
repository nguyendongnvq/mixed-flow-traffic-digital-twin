from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def predict_reconstruction(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_x = []
    all_recon = []
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            x_recon = model(x_batch)
            all_x.append(x_batch.cpu().numpy())
            all_recon.append(x_recon.cpu().numpy())
    return np.concatenate(all_x, axis=0), np.concatenate(all_recon, axis=0)


def compute_weighted_node_max_scores(
    x: np.ndarray,
    x_recon: np.ndarray,
    n_nodes: int,
    n_features: int,
    seq_size: int,
    feature_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    error = np.abs(x_recon - x)
    error_node_feature = error.reshape(error.shape[0], seq_size, n_nodes, n_features)
    weights = feature_weights.reshape(1, 1, 1, n_features)
    weighted_error = error_node_feature * weights
    node_scores = weighted_error.mean(axis=(1, 3))
    window_scores = node_scores.max(axis=1)
    return window_scores, node_scores


def compute_weighted_node_max_scores_from_error(
    error_node_feature: np.ndarray,
    feature_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # error_node_feature: [samples, nodes, features]
    weights = feature_weights.reshape(1, 1, -1)
    weighted_error = error_node_feature * weights
    node_scores = weighted_error.mean(axis=2)
    window_scores = node_scores.max(axis=1)
    return window_scores, node_scores


def select_threshold(scores: np.ndarray, percentile: float) -> float:
    return float(np.percentile(scores, percentile))


def apply_alert_logic(
    scores: np.ndarray,
    threshold: float,
    min_consecutive: int = 1,
    cooldown_steps: int = 3,
) -> np.ndarray:
    alerts = []
    consecutive = 0
    cooldown = 0

    for score in scores:
        if cooldown > 0:
            alerts.append(False)
            cooldown -= 1
            continue

        if score > threshold:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= min_consecutive:
            alerts.append(True)
            cooldown = cooldown_steps
            consecutive = 0
        else:
            alerts.append(False)

    return np.array(alerts, dtype=bool)
