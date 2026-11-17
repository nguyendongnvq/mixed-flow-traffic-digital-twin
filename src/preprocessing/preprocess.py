from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SplitResult:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test_normal: np.ndarray
    train_end: int
    val_end: int


def split_normal_data(
    x_normal: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> SplitResult:
    total = x_normal.shape[0]
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    return SplitResult(
        x_train=x_normal[:train_end],
        x_val=x_normal[train_end:val_end],
        x_test_normal=x_normal[val_end:],
        train_end=train_end,
        val_end=val_end,
    )


def fit_scaler(x_train: np.ndarray) -> StandardScaler:
    _, _, num_features = x_train.shape
    scaler = StandardScaler()
    scaler.fit(x_train.reshape(-1, num_features))
    return scaler


def scale_tensor(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    t, n, f = x.shape
    x_scaled = scaler.transform(x.reshape(-1, f))
    return x_scaled.reshape(t, n, f).astype(np.float32)


def flatten_node_feature(x: np.ndarray) -> np.ndarray:
    t, n, f = x.shape
    return x.reshape(t, n * f).astype(np.float32)


def make_windows(x_flat: np.ndarray, seq_size: int) -> np.ndarray:
    if len(x_flat) < seq_size:
        raise ValueError(f"Not enough timesteps ({len(x_flat)}) for seq_size={seq_size}.")
    windows = [x_flat[i : i + seq_size] for i in range(len(x_flat) - seq_size + 1)]
    return np.array(windows, dtype=np.float32)


def make_windows_with_labels(
    x_flat: np.ndarray,
    time_labels: np.ndarray,
    seq_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(x_flat) < seq_size:
        raise ValueError(f"Not enough timesteps ({len(x_flat)}) for seq_size={seq_size}.")
    windows = []
    labels = []
    for i in range(len(x_flat) - seq_size + 1):
        windows.append(x_flat[i : i + seq_size])
        labels.append(time_labels[i : i + seq_size].max())
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int32)


def make_loader(
    x_windows: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x_windows)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )

