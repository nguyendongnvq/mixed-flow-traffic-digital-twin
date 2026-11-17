from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_accident_ensemble import predict_accident, predict_fallback
from scripts.evaluate_two_stage_ensemble import (
    accident_override_mask,
    final_predictions,
    grouped_temporal_policy,
    predict_two_stage,
)
from src.evaluation.accident_ensemble import aggregate_ensemble_probabilities
from src.evaluation.accident_event_detection import (
    apply_active_state_policy,
    apply_temporal_policy,
)
from src.evaluation.single_label import apply_unknown_temporal_policy
from src.train.train_dmfgcrn_two_stage import coherent_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-stage ensemble inference.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_csv.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_csv}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    accident_manifest = json.loads(
        Path(manifest["accident_manifest"]).read_text(encoding="utf-8")
    )
    data = np.load(args.dataset, allow_pickle=False)
    x = data[f"{args.split}_x"]
    target = data[f"{args.split}_target"]
    begins = data[f"{args.split}_begin"]
    sources = data[f"{args.split}_source"]
    source_names = (
        data["source_names"].astype(str).tolist()
        if "source_names" in data.files
        else [
            "normal",
            "accident",
            "congestion_spike",
            "zigzag",
            "sudden_stop",
            "mixed_s02",
        ]
    )

    detector_probabilities = []
    router_probabilities = []
    for checkpoint_path in manifest["checkpoint_paths"]:
        detector_probability, router_probability, _ = predict_two_stage(
            Path(checkpoint_path),
            x,
            batch_size=args.batch_size,
        )
        detector_probabilities.append(detector_probability)
        router_probabilities.append(router_probability)
    detector_probability = np.stack(detector_probabilities).mean(axis=0)
    router_probability = np.stack(router_probabilities).mean(axis=0)
    class_probability = coherent_probabilities(
        detector_probability,
        router_probability,
    )

    _, residual, fallback_payload = predict_fallback(
        Path(manifest["fallback_checkpoint"]),
        x,
        target,
        batch_size=args.batch_size,
    )
    mixed_source_ids = [
        index for index, name in enumerate(source_names) if name.startswith("mixed")
    ]
    mixed_mask = np.isin(sources, mixed_source_ids)
    accident_probabilities = []
    for checkpoint_path in accident_manifest["checkpoint_paths"]:
        probability, _ = predict_accident(
            Path(checkpoint_path),
            x[mixed_mask],
            batch_size=args.batch_size,
            model_type=str(accident_manifest["model_type"]),
        )
        accident_probabilities.append(probability)
    mixed_accident_probability = aggregate_ensemble_probabilities(
        np.stack(accident_probabilities),
        str(accident_manifest["aggregation"]),
        vote_thresholds=np.asarray(accident_manifest["model_thresholds"]),
    )
    accident_probability = np.zeros(len(x), dtype=np.float32)
    accident_probability[mixed_mask] = mixed_accident_probability
    accident_active = np.zeros(len(x), dtype=bool)
    accident_alert = np.zeros(len(x), dtype=bool)
    accident_active[mixed_mask] = grouped_temporal_policy(
        accident_probability[mixed_mask],
        begins[mixed_mask],
        sources[mixed_mask],
        accident_manifest["accident_active_policy"],
        apply_active_state_policy,
    )
    accident_alert[mixed_mask] = grouped_temporal_policy(
        accident_probability[mixed_mask],
        begins[mixed_mask],
        sources[mixed_mask],
        accident_manifest["temporal_policy"],
        apply_temporal_policy,
    )
    accident_override = np.zeros(len(x), dtype=bool)
    accident_override[mixed_mask] = accident_override_mask(
        accident_active[mixed_mask],
        accident_probability[mixed_mask],
        router_probability[mixed_mask],
        float(accident_manifest["temporal_policy"]["emergency_threshold"]),
    )
    unknown = residual > float(fallback_payload["generic_threshold"])
    unknown[mixed_mask] = grouped_temporal_policy(
        residual[mixed_mask],
        begins[mixed_mask],
        sources[mixed_mask],
        accident_manifest["unknown_temporal_policy"],
        apply_unknown_temporal_policy,
    )
    predictions = final_predictions(
        detector_probability,
        router_probability,
        float(manifest["detector_threshold"]),
        accident_override,
        unknown,
    )
    output_classes = list(manifest["output_classes"])
    primary_labels = np.asarray(
        [
            "unknown_anomaly" if prediction == -1 else output_classes[prediction]
            for prediction in predictions
        ],
        dtype=object,
    )
    status = np.where(
        primary_labels == "normal",
        "normal",
        np.where(primary_labels == "unknown_anomaly", "unknown_anomaly", "known_anomaly"),
    )
    confidence = class_probability.max(axis=1)
    rows = {
        "begin": begins,
        "source": sources,
        "traffic_scale": (
            data[f"{args.split}_traffic_scale"]
            if f"{args.split}_traffic_scale" in data.files
            else np.where(sources == 5, 0.2, 1.0)
        ),
        "status": status,
        "primary_label": primary_labels,
        "confidence": confidence,
        "detector_probability": detector_probability,
        "generic_anomaly_score": residual,
        "accident_active": accident_active,
        "accident_override": accident_override,
        "accident_event_alert": accident_alert,
    }
    for class_index, class_name in enumerate(output_classes):
        rows[f"prob_{class_name}"] = class_probability[:, class_index]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"output={args.output_csv}")


if __name__ == "__main__":
    main()
