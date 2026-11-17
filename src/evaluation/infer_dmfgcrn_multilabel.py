from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

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
    parser = argparse.ArgumentParser(description="Run unified DMFGCRN multi-label inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--global-threshold-json",
        type=Path,
        help="Optional global_threshold_summary.json from analyze_global_threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_csv.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_csv}")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data = np.load(args.dataset, allow_pickle=False)
    class_names = list(payload["class_names"])
    nodes = list(payload["nodes"])
    model = DMFGCRNMultiLabel(
        project_root=Path.cwd(),
        conf_path=Path(payload["dmfgcrn_conf_path"]),
        forecast_checkpoint_path=Path(payload["dmfgcrn_checkpoint_path"]),
        num_nodes=len(nodes),
        lag=int(payload["lag"]),
        num_classes=len(class_names),
        device=torch.device("cpu"),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    x = data[f"{args.split}_x"]
    target = data[f"{args.split}_target"]
    begins = data[f"{args.split}_begin"]
    sources = data[f"{args.split}_source"]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(target).float()),
        batch_size=args.batch_size,
        shuffle=False,
    )

    node_probabilities = []
    forecasts = []
    with torch.no_grad():
        for x_batch, _ in loader:
            node_logits, forecast = model(x_batch)
            node_probabilities.append(torch.sigmoid(node_logits).numpy())
            forecasts.append(forecast[:, 0, :, :].numpy())

    node_prob = np.concatenate(node_probabilities, axis=0)
    global_prob = aggregate_node_probabilities(
        node_prob,
        class_names,
        payload.get("pooling_config"),
    )
    forecast = np.concatenate(forecasts, axis=0)
    residual = np.abs(forecast - target).mean(axis=2).max(axis=1)
    thresholds = np.asarray(payload["class_thresholds"])
    predictions = global_prob >= thresholds.reshape(1, -1)
    accident_alerts = np.zeros(len(x), dtype=bool)
    accident_active = np.zeros(len(x), dtype=bool)
    temporal_policy = payload.get("accident_temporal_policy")
    accident_idx = class_names.index("accident")
    if temporal_policy:
        mixed_source_id = int(temporal_policy.get("mixed_source_id", 5))
        mixed_mask = sources == mixed_source_id
        accident_alerts[mixed_mask] = apply_temporal_policy(
            global_prob[mixed_mask, accident_idx],
            begins[mixed_mask],
            temporal_policy,
        )
        active_policy = payload.get("accident_active_policy")
        if active_policy:
            accident_active[mixed_mask] = apply_active_state_policy(
                global_prob[mixed_mask, accident_idx],
                begins[mixed_mask],
                active_policy,
            )
        else:
            accident_active[mixed_mask] = accident_alerts[mixed_mask]
        predictions[mixed_mask, accident_idx] = accident_active[mixed_mask]

    generic_threshold = float(payload.get("global_threshold", payload["generic_threshold"]))
    if args.global_threshold_json is not None:
        threshold_payload = json.loads(args.global_threshold_json.read_text(encoding="utf-8"))
        generic_threshold = float(threshold_payload["global_threshold"])

    use_global_gate = (
        args.global_threshold_json is not None
        or "global_threshold" in payload
    )
    global_anomaly_predictions = residual >= generic_threshold
    if use_global_gate:
        predictions = predictions & global_anomaly_predictions.reshape(-1, 1)
    unknown_predictions = global_anomaly_predictions & ~predictions.any(axis=1)
    unknown_policy = payload.get("unknown_temporal_policy")
    if temporal_policy and unknown_policy and not use_global_gate:
        unknown_predictions[mixed_mask] = apply_unknown_temporal_policy(
            residual[mixed_mask],
            begins[mixed_mask],
            unknown_policy,
        )
    primary_labels = select_primary_labels(
        class_names,
        global_prob,
        thresholds,
        predictions,
        residual,
        generic_threshold=generic_threshold,
        unknown_predictions=unknown_predictions,
        policy=payload.get("single_label_policy"),
    )
    rows = []
    for sample_idx in range(len(x)):
        primary_label = str(primary_labels[sample_idx])
        detected = primary_label if primary_label in class_names else ""
        if detected:
            status = "known_anomaly"
        elif primary_label == "unknown_anomaly":
            status = "unknown_anomaly"
        else:
            status = "normal"

        row: dict[str, object] = {
            "begin": float(begins[sample_idx]),
            "status": status,
            "is_anomaly_pred": bool(primary_label != "normal"),
            "primary_label": primary_label,
            "labels": detected,
            "generic_anomaly_score": float(residual[sample_idx]),
            "accident_event_alert": bool(accident_alerts[sample_idx]),
            "accident_active": bool(accident_active[sample_idx]),
        }
        for class_idx, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = float(global_prob[sample_idx, class_idx])
            top_node_idx = int(node_prob[sample_idx, :, class_idx].argmax())
            row[f"top_node_{class_name}"] = nodes[top_node_idx]
        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"output={args.output_csv}")
    print(pd.Series([row["status"] for row in rows]).value_counts().to_string())


if __name__ == "__main__":
    main()
