from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "normal_dataset": r"datasets\processed\accident\forecast_accident_node_level_normal_only.csv",
    "accident_dataset": r"datasets\processed\accident\forecast_accident_node_level_accident_only.csv",
    "time_col": "begin",
    "end_col": "end",
    "node_col": "node_id",
    "label_col": "is_anomaly",
    "feature_cols": [
        "flow_count_5m",
        "speed",
        "speed_missing",
        "e1_occupancy",
        "e2_meanOccupancy",
        "jam_veh",
    ],
    "seq_size": 12,
    "batch_size": 64,
    "hidden_dim": 128,
    "latent_dim": 64,
    "num_layers": 1,
    "dropout": 0.2,
    "epochs": 50,
    "lr": 1e-3,
    "patience": 7,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "score_name": "weighted_node_max",
    "threshold_percentile": 85,
    "min_consecutive": 2,
    "cooldown_steps": 3,
    "feature_weights": {
        "flow_count_5m": 0.3,
        "speed": 0.3,
        "speed_missing": 0.1,
        "e1_occupancy": 0.3,
        "e2_meanOccupancy": 2.5,
        "jam_veh": 1.5,
    },
    "output_dir": "models/outputs_weighted_p85",
    "model_output_name": "lstm_autoencoder_weighted_node_max_p85.pt",
    "model_backend": "lstm_autoencoder",
    "dmfgcrn_checkpoint_path": r"DMFGCRN\experiments\SUMO\20260512041404\10\best_model.pth",
    "dmfgcrn_conf_path": r"DMFGCRN\config_file\SUMO_DMFGCRN.conf",
    "dmfgcrn_device": "cpu",
    "dmfgcrn_horizon_index": 0,
    "dmfgcrn_input_features": [
        "flow_count_5m",
        "e1_occupancy",
        "speed",
        "speed_missing",
    ],
    "dmfgcrn_output_features": [
        "flow_count_5m",
        "e1_occupancy",
        "speed",
    ],
    "seed": 42,
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        user_config = json.load(f)
    config = _deep_merge(DEFAULT_CONFIG, user_config)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_keys = [
        "normal_dataset",
        "accident_dataset",
        "time_col",
        "node_col",
        "label_col",
        "feature_cols",
        "feature_weights",
        "seq_size",
        "batch_size",
        "epochs",
        "output_dir",
    ]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    feature_cols = config["feature_cols"]
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError("config['feature_cols'] must be a non-empty list.")

    feature_weights = config["feature_weights"]
    if not isinstance(feature_weights, dict):
        raise ValueError("config['feature_weights'] must be a dict.")

    missing_weights = [c for c in feature_cols if c not in feature_weights]
    if missing_weights:
        raise ValueError(f"Missing feature weights for: {missing_weights}")

    if config["seq_size"] <= 0:
        raise ValueError("config['seq_size'] must be > 0.")
    if config["batch_size"] <= 0:
        raise ValueError("config['batch_size'] must be > 0.")

    backend = str(config.get("model_backend", "lstm_autoencoder")).lower()
    if backend not in {"lstm_autoencoder", "dmfgcrn"}:
        raise ValueError("config['model_backend'] must be 'lstm_autoencoder' or 'dmfgcrn'.")

    if backend == "dmfgcrn":
        required_dmfgcrn = [
            "dmfgcrn_checkpoint_path",
            "dmfgcrn_conf_path",
            "dmfgcrn_horizon_index",
            "dmfgcrn_input_features",
            "dmfgcrn_output_features",
        ]
        missing_dmfgcrn = [key for key in required_dmfgcrn if key not in config]
        if missing_dmfgcrn:
            raise ValueError(f"Missing DMFGCRN config keys: {missing_dmfgcrn}")

        if int(config["dmfgcrn_horizon_index"]) < 0:
            raise ValueError("config['dmfgcrn_horizon_index'] must be >= 0.")

        input_features = config["dmfgcrn_input_features"]
        output_features = config["dmfgcrn_output_features"]
        if not isinstance(input_features, list) or not input_features:
            raise ValueError("config['dmfgcrn_input_features'] must be a non-empty list.")
        if not isinstance(output_features, list) or not output_features:
            raise ValueError("config['dmfgcrn_output_features'] must be a non-empty list.")

        missing_output_weights = [c for c in output_features if c not in feature_weights]
        if missing_output_weights:
            raise ValueError(
                f"Missing feature weights for dmfgcrn output features: {missing_output_weights}"
            )
