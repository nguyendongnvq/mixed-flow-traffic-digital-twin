from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluate_accident_ensemble import predict_accident
from src.evaluation.accident_ensemble import aggregate_ensemble_probabilities
from src.evaluation.accident_event_detection import (
    apply_active_state_policy,
    apply_temporal_policy,
)
from src.evaluation.single_label import (
    apply_unknown_temporal_policy,
    select_primary_labels,
)
from src.models.dmfgcrn_multilabel import DMFGCRNMultiLabel
from src.models.multilabel_pooling import aggregate_node_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ensemble accident inference.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def predict_fallback(
    checkpoint_path: Path,
    x: np.ndarray,
    target: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = list(payload["class_names"])
    model = DMFGCRNMultiLabel(
        project_root=Path.cwd(),
        conf_path=Path(payload["dmfgcrn_conf_path"]),
        forecast_checkpoint_path=Path(payload["dmfgcrn_checkpoint_path"]),
        num_nodes=len(payload["nodes"]),
        lag=int(payload["lag"]),
        num_classes=len(class_names),
        device=torch.device("cpu"),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x).float()),
        batch_size=batch_size,
        shuffle=False,
    )
    node_probabilities = []
    forecasts = []
    with torch.no_grad():
        for (x_batch,) in loader:
            node_logits, forecast = model(x_batch)
            node_probabilities.append(torch.sigmoid(node_logits).numpy())
            forecasts.append(forecast[:, 0, :, :].numpy())
    node_prob = np.concatenate(node_probabilities)
    global_prob = aggregate_node_probabilities(
        node_prob,
        class_names,
        payload.get("pooling_config"),
    )
    forecast = np.concatenate(forecasts)
    residual = np.abs(forecast - target).mean(axis=2).max(axis=1)
    return node_prob, global_prob, residual, payload


def main() -> None:
    args = parse_args()
    if args.output_csv.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_csv}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    data = np.load(args.dataset, allow_pickle=False)
    x = data[f"{args.split}_x"]
    target = data[f"{args.split}_target"]
    begins = data[f"{args.split}_begin"]
    sources = data[f"{args.split}_source"]
    node_prob, global_prob, residual, fallback = predict_fallback(
        Path(manifest["fallback_checkpoint"]),
        x,
        target,
        batch_size=args.batch_size,
    )
    class_names = list(fallback["class_names"])
    accident_idx = class_names.index("accident")
    thresholds = np.asarray(fallback["class_thresholds"], dtype=float).copy()
    effective_thresholds = thresholds.copy()
    for class_index, class_name in enumerate(class_names):
        effective_thresholds[class_index] *= float(
            manifest.get("class_threshold_multipliers", {}).get(
                class_name,
                1.0,
            )
        )
    predictions = global_prob >= thresholds.reshape(1, -1)

    mixed_source_id = int(manifest.get("mixed_source_id", 5))
    mixed_mask = sources == mixed_source_id
    predictions[mixed_mask] = (
        global_prob[mixed_mask]
        >= effective_thresholds.reshape(1, -1)
    )
    ensemble_probabilities = []
    for checkpoint_path in manifest["checkpoint_paths"]:
        probability, _ = predict_accident(
            Path(checkpoint_path),
            x[mixed_mask],
            batch_size=args.batch_size,
            model_type=str(manifest.get("model_type", "multilabel")),
        )
        ensemble_probabilities.append(probability)
    ensemble_probabilities_array = np.stack(ensemble_probabilities)
    ensemble_probability = aggregate_ensemble_probabilities(
        ensemble_probabilities_array,
        str(manifest["aggregation"]),
        vote_thresholds=np.asarray(manifest["model_thresholds"]),
    )
    accident_alerts = apply_temporal_policy(
        ensemble_probability,
        begins[mixed_mask],
        dict(manifest["temporal_policy"]),
    )
    accident_active = apply_active_state_policy(
        ensemble_probability,
        begins[mixed_mask],
        dict(manifest["accident_active_policy"]),
    )
    predictions[mixed_mask, accident_idx] = accident_active

    mean_probability = ensemble_probabilities_array.mean(axis=0)
    max_probability = ensemble_probabilities_array.max(axis=0)
    votes = (
        ensemble_probabilities_array
        >= np.asarray(manifest["model_thresholds"]).reshape(-1, 1)
    ).sum(axis=0)
    mixed_position = {sample_index: position for position, sample_index in enumerate(np.where(mixed_mask)[0])}
    generic_threshold = float(fallback["generic_threshold"])
    unknown_predictions = residual > generic_threshold
    unknown_predictions[mixed_mask] = apply_unknown_temporal_policy(
        residual[mixed_mask],
        begins[mixed_mask],
        dict(manifest["unknown_temporal_policy"]),
    )
    primary_labels = select_primary_labels(
        class_names,
        global_prob,
        thresholds,
        predictions,
        residual,
        generic_threshold=generic_threshold,
        unknown_predictions=unknown_predictions,
        policy=fallback.get("single_label_policy"),
    )
    primary_labels[mixed_mask] = select_primary_labels(
        class_names,
        global_prob[mixed_mask],
        effective_thresholds,
        predictions[mixed_mask],
        residual[mixed_mask],
        generic_threshold=float(manifest["unknown_temporal_policy"]["threshold"]),
        unknown_predictions=unknown_predictions[mixed_mask],
        policy=manifest.get("single_label_policy"),
    )
    rows = []
    for sample_index in range(len(x)):
        primary_label = str(primary_labels[sample_index])
        detected = primary_label if primary_label in class_names else ""
        status = (
            "known_anomaly"
            if detected
            else "unknown_anomaly"
            if primary_label == "unknown_anomaly"
            else "normal"
        )
        position = mixed_position.get(sample_index)
        row: dict[str, object] = {
            "begin": float(begins[sample_index]),
            "status": status,
            "primary_label": primary_label,
            "labels": detected,
            "generic_anomaly_score": float(residual[sample_index]),
            "accident_prob_mean": (
                float(mean_probability[position]) if position is not None else np.nan
            ),
            "accident_prob_max": (
                float(max_probability[position]) if position is not None else np.nan
            ),
            "accident_votes": int(votes[position]) if position is not None else 0,
            "accident_event_alert": (
                bool(accident_alerts[position]) if position is not None else False
            ),
            "accident_active": (
                bool(accident_active[position]) if position is not None else False
            ),
        }
        for class_index, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = float(global_prob[sample_index, class_index])
            top_node_index = int(node_prob[sample_index, :, class_index].argmax())
            row[f"top_node_{class_name}"] = fallback["nodes"][top_node_index]
        rows.append(row)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"output={args.output_csv}")


if __name__ == "__main__":
    main()
