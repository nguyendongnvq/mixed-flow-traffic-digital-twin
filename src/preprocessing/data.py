from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["speed_missing"] = cleaned["speed"].isna().astype(np.float32)
    cleaned["speed"] = cleaned["speed"].fillna(0)
    return cleaned


def ensure_label_column(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    labeled = df.copy()
    if label_col not in labeled.columns:
        labeled[label_col] = 0
    labeled[label_col] = pd.to_numeric(labeled[label_col], errors="coerce").fillna(0).astype(np.int32)
    return labeled


def load_dataframes(
    normal_path: str,
    accident_path: str,
    label_col: str = "is_anomaly",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_normal = pd.read_csv(normal_path)
    df_accident = pd.read_csv(accident_path)
    df_normal = ensure_label_column(df_normal, label_col=label_col)
    df_accident = ensure_label_column(df_accident, label_col=label_col)
    return clean_df(df_normal), clean_df(df_accident)


def ensure_same_nodes(
    df_normal: pd.DataFrame,
    df_accident: pd.DataFrame,
    node_col: str,
) -> list[str]:
    normal_nodes = sorted(df_normal[node_col].unique())
    accident_nodes = sorted(df_accident[node_col].unique())
    if normal_nodes != accident_nodes:
        raise ValueError("Node list of normal and accident datasets is not identical.")
    return normal_nodes


def to_feature_weights_array(feature_cols: Sequence[str], feature_weights: dict[str, float]) -> np.ndarray:
    return np.array([feature_weights[c] for c in feature_cols], dtype=np.float32)


def df_to_tensor(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    nodes: Sequence[str],
    time_col: str,
    node_col: str,
) -> tuple[np.ndarray, list[float]]:
    sorted_df = df.copy().sort_values([time_col, node_col])
    timesteps = sorted(sorted_df[time_col].unique())
    feature_mats = []

    for feat in feature_cols:
        mat = sorted_df.pivot_table(
            index=time_col,
            columns=node_col,
            values=feat,
            aggfunc="mean",
        )
        mat = mat.reindex(index=timesteps, columns=nodes)
        mat = mat.ffill().bfill().fillna(0)
        feature_mats.append(mat.values)

    tensor = np.stack(feature_mats, axis=-1).astype(np.float32)
    return tensor, timesteps


def make_time_labels(
    df: pd.DataFrame,
    timesteps: Sequence[float],
    time_col: str,
    label_col: str,
) -> np.ndarray:
    labels = (
        df.groupby(time_col)[label_col]
        .max()
        .reindex(timesteps)
        .fillna(0)
        .astype(np.int32)
        .values
    )
    return labels
