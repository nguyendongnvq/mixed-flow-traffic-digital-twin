from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluate_accident_ensemble import predict_accident, predict_fallback
from src.evaluation.accident_ensemble import aggregate_ensemble_probabilities
from src.evaluation.accident_event_detection import (
    apply_active_state_policy,
    apply_temporal_policy,
    binary_window_metrics,
    event_metrics,
)
from src.evaluation.single_label import apply_unknown_temporal_policy
from src.models.dmfgcrn_two_stage import DMFGCRNTwoStage
from src.train.train_dmfgcrn_two_stage import (
    ANOMALY_CLASSES,
    OUTPUT_CLASSES,
    apply_router_multipliers,
    build_primary_targets,
    coherent_probabilities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate two-stage DMFGCRN ensemble.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "datasets/processed/multilabel/"
            "dmfgcrn_multilabel_windows_multiscale_v2.npz"
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("models/outputs_dmfgcrn_two_stage_v2"),
    )
    parser.add_argument(
        "--accident-manifest",
        type=Path,
        default=Path(
            "models/outputs_dmfgcrn_accident_binary_ensemble_single_label_v3/"
            "ensemble_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/outputs_dmfgcrn_two_stage_ensemble_v2"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def predict_two_stage(
    checkpoint_path: Path,
    x: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    router_classes = payload.get("router_classes", ANOMALY_CLASSES)
    model = DMFGCRNTwoStage(
        project_root=Path.cwd(),
        conf_path=Path(payload["dmfgcrn_conf_path"]),
        forecast_checkpoint_path=Path(payload["dmfgcrn_checkpoint_path"]),
        num_nodes=len(payload["nodes"]),
        lag=int(payload["lag"]),
        num_output_classes=len(router_classes),
        device=torch.device("cpu"),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x).float()),
        batch_size=batch_size,
        shuffle=False,
    )
    detector_probabilities = []
    router_probabilities = []
    with torch.no_grad():
        for (x_batch,) in loader:
            detector_logit, router_logits = model.classify(x_batch)
            detector_probabilities.append(torch.sigmoid(detector_logit).numpy())
            router_probabilities.append(torch.softmax(router_logits, dim=-1).numpy())
    router_probability = np.concatenate(router_probabilities)
    multipliers = np.asarray(
        payload.get("router_multipliers", np.ones(router_probability.shape[1])),
        dtype=np.float64,
    )
    return (
        np.concatenate(detector_probabilities),
        apply_router_multipliers(router_probability, multipliers),
        payload,
    )


def grouped_temporal_policy(
    values: np.ndarray,
    begins: np.ndarray,
    groups: np.ndarray,
    policy: dict[str, Any],
    apply: Callable[[np.ndarray, np.ndarray, dict[str, Any]], np.ndarray],
) -> np.ndarray:
    result = np.zeros(len(values), dtype=bool)
    for group in np.unique(groups):
        mask = groups == group
        result[mask] = apply(values[mask], begins[mask], policy)
    return result


def accident_override_mask(
    accident_active: np.ndarray,
    accident_probability: np.ndarray,
    router_probability: np.ndarray,
    emergency_threshold: float,
) -> np.ndarray:
    router_accident = router_probability.argmax(axis=1) == 0
    return accident_active & (
        router_accident | (accident_probability >= emergency_threshold)
    )


def final_predictions(
    detector_probability: np.ndarray,
    router_probability: np.ndarray,
    detector_threshold: float,
    accident_override: np.ndarray,
    unknown_predictions: np.ndarray,
) -> np.ndarray:
    routed = router_probability.argmax(axis=1) + 1
    predictions = np.where(detector_probability >= detector_threshold, routed, 0)
    predictions = np.where(unknown_predictions & (predictions == 0), -1, predictions)
    return np.where(accident_override, 1, predictions)


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    labels = np.arange(len(OUTPUT_CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                targets,
                predictions,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "unknown_count": int((predictions == -1).sum()),
        "by_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(OUTPUT_CLASSES)
        },
        "confusion_matrix": confusion_matrix(
            targets,
            predictions,
            labels=labels,
        ).tolist(),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    data = np.load(args.dataset, allow_pickle=False)
    class_names = data["class_names"].astype(str).tolist()
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
    mixed_source_ids = np.asarray(
        [index for index, name in enumerate(source_names) if name.startswith("mixed")],
        dtype=np.int32,
    )
    accident_index = class_names.index("accident")
    accident_manifest = json.loads(args.accident_manifest.read_text(encoding="utf-8"))
    checkpoints = [
        args.checkpoint_root / f"seed_{seed}" / "dmfgcrn_two_stage.pt"
        for seed in (42, 43, 44)
    ]

    split_payload: dict[str, dict[str, Any]] = {}
    for split in ("val", "test"):
        detector_probabilities = []
        router_probabilities = []
        for checkpoint in checkpoints:
            detector_probability, router_probability, _ = predict_two_stage(
                checkpoint,
                data[f"{split}_x"],
                batch_size=args.batch_size,
            )
            detector_probabilities.append(detector_probability)
            router_probabilities.append(router_probability)
        detector_probability = np.stack(detector_probabilities).mean(axis=0)
        router_probability = np.stack(router_probabilities).mean(axis=0)

        _, residual, fallback_payload = predict_fallback(
            Path(accident_manifest["fallback_checkpoint"]),
            data[f"{split}_x"],
            data[f"{split}_target"],
            batch_size=args.batch_size,
        )
        sources = data[f"{split}_source"]
        begins = data[f"{split}_begin"]
        mixed_mask = np.isin(sources, mixed_source_ids)
        accident_probability = np.zeros(len(sources), dtype=np.float32)
        if mixed_mask.any():
            specialist_probabilities = []
            for checkpoint in accident_manifest["checkpoint_paths"]:
                probability, _ = predict_accident(
                    Path(checkpoint),
                    data[f"{split}_x"][mixed_mask],
                    batch_size=args.batch_size,
                    model_type=str(accident_manifest["model_type"]),
                )
                specialist_probabilities.append(probability)
            accident_probability[mixed_mask] = aggregate_ensemble_probabilities(
                np.stack(specialist_probabilities),
                str(accident_manifest["aggregation"]),
                vote_thresholds=np.asarray(accident_manifest["model_thresholds"]),
            )
        accident_active = np.zeros(len(sources), dtype=bool)
        accident_alert = np.zeros(len(sources), dtype=bool)
        if mixed_mask.any():
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
        emergency_threshold = float(
            accident_manifest["temporal_policy"]["emergency_threshold"]
        )
        accident_override = np.zeros(len(sources), dtype=bool)
        accident_override[mixed_mask] = accident_override_mask(
            accident_active[mixed_mask],
            accident_probability[mixed_mask],
            router_probability[mixed_mask],
            emergency_threshold,
        )
        unknown = residual > float(fallback_payload["generic_threshold"])
        if mixed_mask.any():
            unknown[mixed_mask] = grouped_temporal_policy(
                residual[mixed_mask],
                begins[mixed_mask],
                sources[mixed_mask],
                accident_manifest["unknown_temporal_policy"],
                apply_unknown_temporal_policy,
            )
        split_payload[split] = {
            "detector_probability": detector_probability,
            "router_probability": router_probability,
            "class_probability": coherent_probabilities(
                detector_probability,
                router_probability,
            ),
            "targets": build_primary_targets(data[f"{split}_global_y"], class_names),
            "accident_probability": accident_probability,
            "accident_active": accident_active,
            "accident_alert": accident_alert,
            "accident_override": accident_override,
            "unknown": unknown,
            "residual": residual,
            "mixed_mask": mixed_mask,
        }

    val = split_payload["val"]
    candidates = []
    for threshold in np.linspace(0.10, 0.90, 81):
        predictions = final_predictions(
            val["detector_probability"],
            val["router_probability"],
            float(threshold),
            val["accident_override"],
            val["unknown"],
        )
        result = classification_metrics(val["targets"], predictions)
        rank = (
            result["macro_f1"],
            result["by_class"]["accident"]["recall"],
            result["by_class"]["normal"]["recall"],
        )
        candidates.append((rank, float(threshold), result))
    _, detector_threshold, validation_metrics = max(
        candidates,
        key=lambda item: item[0],
    )

    test = split_payload["test"]
    test_predictions = final_predictions(
        test["detector_probability"],
        test["router_probability"],
        detector_threshold,
        test["accident_override"],
        test["unknown"],
    )
    test_metrics = classification_metrics(test["targets"], test_predictions)
    source_metrics = {}
    for source_id, source_name in enumerate(source_names):
        source_mask = data["test_source"] == source_id
        source_metrics[source_name] = classification_metrics(
            test["targets"][source_mask],
            test_predictions[source_mask],
        )

    mixed_labels = data["test_global_y"][test["mixed_mask"], accident_index]
    mixed_groups = data["test_source"][test["mixed_mask"]]
    event_rows = []
    active_rows = []
    for source_id in np.unique(mixed_groups):
        mask = mixed_groups == source_id
        event_rows.append(
            event_metrics(
                mixed_labels[mask],
                test["accident_alert"][test["mixed_mask"]][mask],
                data["test_begin"][test["mixed_mask"]][mask],
                max_latency_seconds=600,
            )
        )
        active_rows.append(
            binary_window_metrics(
                mixed_labels[mask],
                test["accident_active"][test["mixed_mask"]][mask],
            )
        )
    accident_event_f1 = min(
        (float(row["event_f1"]) for row in event_rows),
        default=0.0,
    )
    accident_active_f1 = min(
        (float(row["f1"]) for row in active_rows),
        default=0.0,
    )
    normal_mask = test["targets"] == 0
    normal_specificity = float((test_predictions[normal_mask] == 0).mean())
    unknown_fp_count = int((test_predictions[normal_mask] == -1).sum())
    accident_only_recall = source_metrics["accident"]["by_class"]["accident"]["recall"]
    mixed_accuracies = [
        source_metrics[name]["accuracy"]
        for name in source_names
        if name.startswith("mixed")
    ]
    acceptance = {
        "macro_f1": test_metrics["macro_f1"] >= 0.75,
        "normal_specificity": normal_specificity >= 0.90,
        "accident_recall_overall": (
            test_metrics["by_class"]["accident"]["recall"] >= 0.70
        ),
        "accident_recall_accident_only": accident_only_recall >= 0.65,
        "congestion_recall": (
            test_metrics["by_class"]["congestion_spike"]["recall"] >= 0.65
        ),
        "zigzag_recall": test_metrics["by_class"]["zigzag"]["recall"] >= 0.70,
        "sudden_stop_recall": (
            test_metrics["by_class"]["sudden_stop"]["recall"] >= 0.95
        ),
        "mixed_accuracy": min(mixed_accuracies, default=0.0) >= 0.95,
        "accident_event_f1": accident_event_f1 >= 0.9375,
        "accident_active_f1": accident_active_f1 >= 0.90,
        "unknown_false_positives": unknown_fp_count <= 5,
    }
    promote = all(acceptance.values())
    summary = {
        "validation_metrics": validation_metrics,
        "detector_threshold": detector_threshold,
        "test_metrics": test_metrics,
        "source_metrics": source_metrics,
        "normal_specificity": normal_specificity,
        "unknown_false_positive_count": unknown_fp_count,
        "accident_event_metrics_by_scale": event_rows,
        "accident_active_metrics_by_scale": active_rows,
        "acceptance": acceptance,
        "promote": promote,
    }
    manifest = {
        "model_type": "dmfgcrn_two_stage_ensemble_v2",
        "checkpoint_paths": [str(path.resolve()) for path in checkpoints],
        "aggregation": "mean",
        "detector_threshold": detector_threshold,
        "output_classes": OUTPUT_CLASSES,
        "router_classes": ANOMALY_CLASSES,
        "accident_manifest": str(args.accident_manifest.resolve()),
        "fallback_checkpoint": accident_manifest["fallback_checkpoint"],
        "promoted": promote,
    }
    (args.output_dir / "ensemble_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "ensemble_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    output_rows: dict[str, Any] = {
        "begin": data["test_begin"],
        "source": data["test_source"],
        "traffic_scale": (
            data["test_traffic_scale"]
            if "test_traffic_scale" in data.files
            else np.where(data["test_source"] == 5, 0.2, 1.0)
        ),
        "target": [OUTPUT_CLASSES[target] for target in test["targets"]],
        "primary_label": [
            "unknown_anomaly" if prediction == -1 else OUTPUT_CLASSES[prediction]
            for prediction in test_predictions
        ],
        "detector_probability": test["detector_probability"],
        "accident_probability": test["accident_probability"],
        "accident_override": test["accident_override"],
        "accident_event_alert": test["accident_alert"],
        "generic_anomaly_score": test["residual"],
    }
    for class_index, class_name in enumerate(OUTPUT_CLASSES):
        output_rows[f"prob_{class_name}"] = test["class_probability"][:, class_index]
    pd.DataFrame(output_rows).to_csv(
        args.output_dir / "test_inference.csv",
        index=False,
    )
    print(json.dumps({"promote": promote, **test_metrics}, indent=2))


if __name__ == "__main__":
    main()
