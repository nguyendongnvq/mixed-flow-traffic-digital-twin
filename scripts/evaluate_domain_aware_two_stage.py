from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.evaluate_two_stage_ensemble import (
    classification_metrics,
    final_predictions,
    predict_two_stage,
)
from src.evaluation.domain_gate import train_domain_gate
from src.train.train_dmfgcrn_two_stage import OUTPUT_CLASSES, build_primary_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate domain-aware two-stage ensemble.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-checkpoint-root", type=Path, required=True)
    parser.add_argument("--mixed-checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gate-minimum-mixed-recall", type=float, default=0.98)
    parser.add_argument("--fallback-margin", type=float, default=0.10)
    return parser.parse_args()


def ensemble_probabilities(
    checkpoint_root: Path,
    x: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    detector_probabilities = []
    router_probabilities = []
    for seed in (42, 43, 44):
        detector_probability, router_probability, _ = predict_two_stage(
            checkpoint_root / f"seed_{seed}" / "dmfgcrn_two_stage.pt",
            x,
            batch_size=batch_size,
        )
        detector_probabilities.append(detector_probability)
        router_probabilities.append(router_probability)
    return (
        np.stack(detector_probabilities).mean(axis=0),
        np.stack(router_probabilities).mean(axis=0),
    )


def choose_threshold(
    detector_probability: np.ndarray,
    router_probability: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    candidates = []
    for threshold in np.linspace(0.10, 0.90, 81):
        predictions = final_predictions(
            detector_probability[mask],
            router_probability[mask],
            float(threshold),
            np.zeros(int(mask.sum()), dtype=bool),
            np.zeros(int(mask.sum()), dtype=bool),
        )
        result = classification_metrics(targets[mask], predictions)
        rank = (
            result["macro_f1"],
            result["by_class"]["normal"]["recall"],
            result["by_class"]["accident"]["recall"],
        )
        candidates.append((rank, float(threshold), result))
    _, threshold, result = max(candidates, key=lambda item: item[0])
    return threshold, result


def path_predictions(
    detector_probability: np.ndarray,
    router_probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return final_predictions(
        detector_probability,
        router_probability,
        threshold,
        np.zeros(len(detector_probability), dtype=bool),
        np.zeros(len(detector_probability), dtype=bool),
    )


def combine_predictions(
    source_predictions: np.ndarray,
    mixed_predictions: np.ndarray,
    mixed_probability: np.ndarray,
    gate_threshold: float,
    fallback_margin: float,
) -> np.ndarray:
    choose_mixed = mixed_probability >= gate_threshold
    output = np.where(choose_mixed, mixed_predictions, source_predictions)
    uncertain = np.abs(mixed_probability - gate_threshold) <= fallback_margin
    both_normal = (source_predictions == 0) & (mixed_predictions == 0)
    output[uncertain & both_normal] = 0
    source_known = source_predictions > 0
    mixed_known = mixed_predictions > 0
    output[uncertain & source_known & ~mixed_known] = source_predictions[
        uncertain & source_known & ~mixed_known
    ]
    output[uncertain & mixed_known & ~source_known] = mixed_predictions[
        uncertain & mixed_known & ~source_known
    ]
    disagreement = uncertain & source_known & mixed_known & (source_predictions != mixed_predictions)
    output[disagreement] = -1
    return output


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    data = np.load(args.dataset, allow_pickle=False)
    source_names = data["source_names"].astype(str).tolist()
    class_names = data["class_names"].astype(str).tolist()
    mixed_source_ids = np.asarray(
        [index for index, name in enumerate(source_names) if name.startswith("mixed")],
        dtype=np.int32,
    )
    split_targets = {
        split: build_primary_targets(data[f"{split}_global_y"], class_names)
        for split in ("train", "val", "test")
    }
    train_is_mixed = np.isin(data["train_source"], mixed_source_ids)
    val_is_mixed = np.isin(data["val_source"], mixed_source_ids)
    test_is_mixed = np.isin(data["test_source"], mixed_source_ids)
    gate, gate_metrics = train_domain_gate(
        data["train_x"],
        train_is_mixed,
        data["val_x"],
        val_is_mixed,
        minimum_mixed_recall=args.gate_minimum_mixed_recall,
    )

    probabilities: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for split in ("val", "test"):
        probabilities[split] = {
            "source": ensemble_probabilities(
                args.source_checkpoint_root,
                data[f"{split}_x"],
                batch_size=args.batch_size,
            ),
            "mixed": ensemble_probabilities(
                args.mixed_checkpoint_root,
                data[f"{split}_x"],
                batch_size=args.batch_size,
            ),
        }

    source_threshold, source_validation = choose_threshold(
        probabilities["val"]["source"][0],
        probabilities["val"]["source"][1],
        split_targets["val"],
        ~val_is_mixed,
    )
    mixed_threshold, mixed_validation = choose_threshold(
        probabilities["val"]["mixed"][0],
        probabilities["val"]["mixed"][1],
        split_targets["val"],
        val_is_mixed,
    )

    test_source_predictions = path_predictions(
        probabilities["test"]["source"][0],
        probabilities["test"]["source"][1],
        source_threshold,
    )
    test_mixed_predictions = path_predictions(
        probabilities["test"]["mixed"][0],
        probabilities["test"]["mixed"][1],
        mixed_threshold,
    )
    test_mixed_probability = gate.predict_mixed_probability(data["test_x"])
    test_predictions = combine_predictions(
        test_source_predictions,
        test_mixed_predictions,
        test_mixed_probability,
        gate.threshold,
        args.fallback_margin,
    )
    test_metrics = classification_metrics(split_targets["test"], test_predictions)

    source_metrics = {}
    for source_id, source_name in enumerate(source_names):
        source_mask = data["test_source"] == source_id
        source_metrics[source_name] = classification_metrics(
            split_targets["test"][source_mask],
            test_predictions[source_mask],
        )
    normal_mask = split_targets["test"] == 0
    mixed_accuracies = [
        source_metrics[name]["accuracy"]
        for name in source_names
        if name.startswith("mixed")
    ]
    summary = {
        "gate_metrics": gate_metrics,
        "gate_manifest": gate.to_manifest(),
        "source_threshold": source_threshold,
        "mixed_threshold": mixed_threshold,
        "source_validation_metrics": source_validation,
        "mixed_validation_metrics": mixed_validation,
        "test_metrics": test_metrics,
        "source_metrics": source_metrics,
        "normal_specificity": float((test_predictions[normal_mask] == 0).mean()),
        "unknown_false_positive_count": int((test_predictions[normal_mask] == -1).sum()),
        "mixed_min_accuracy": float(min(mixed_accuracies, default=0.0)),
        "acceptance": {
            "macro_f1": test_metrics["macro_f1"] >= 0.75,
            "normal_specificity": float((test_predictions[normal_mask] == 0).mean()) >= 0.90,
            "mixed_accuracy": min(mixed_accuracies, default=0.0) >= 0.95,
            "unknown_false_positives": int((test_predictions[normal_mask] == -1).sum()) <= 5,
        },
    }
    summary["promote"] = all(summary["acceptance"].values())
    (args.output_dir / "domain_aware_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "model_type": "domain_aware_two_stage",
        "source_checkpoint_root": str(args.source_checkpoint_root.resolve()),
        "mixed_checkpoint_root": str(args.mixed_checkpoint_root.resolve()),
        "source_threshold": source_threshold,
        "mixed_threshold": mixed_threshold,
        "domain_gate": gate.to_manifest(),
        "promoted": bool(summary["promote"]),
    }
    (args.output_dir / "domain_aware_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    rows = {
        "begin": data["test_begin"],
        "source": data["test_source"],
        "traffic_scale": data["test_traffic_scale"],
        "target": [OUTPUT_CLASSES[target] for target in split_targets["test"]],
        "primary_label": [
            "unknown_anomaly" if prediction == -1 else OUTPUT_CLASSES[prediction]
            for prediction in test_predictions
        ],
        "domain_mixed_probability": test_mixed_probability,
        "source_path_prediction": [
            "unknown_anomaly" if prediction == -1 else OUTPUT_CLASSES[prediction]
            for prediction in test_source_predictions
        ],
        "mixed_path_prediction": [
            "unknown_anomaly" if prediction == -1 else OUTPUT_CLASSES[prediction]
            for prediction in test_mixed_predictions
        ],
    }
    pd.DataFrame(rows).to_csv(args.output_dir / "domain_aware_test_inference.csv", index=False)
    print(json.dumps({"promote": summary["promote"], **test_metrics}, indent=2))


if __name__ == "__main__":
    main()
