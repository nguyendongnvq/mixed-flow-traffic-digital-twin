from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.models.dmfgcrn_two_stage import DMFGCRNTwoStage


OUTPUT_CLASSES = [
    "normal",
    "accident",
    "congestion_spike",
    "zigzag",
    "sudden_stop",
]
ANOMALY_CLASSES = OUTPUT_CLASSES[1:]
ANOMALY_PRIORITY = ANOMALY_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train two-stage DMFGCRN classifier.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-sources",
        type=str,
        help="Comma-separated source names or ids to keep for this training run.",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_primary_targets(
    global_labels: np.ndarray,
    class_names: list[str],
) -> np.ndarray:
    targets = np.zeros(len(global_labels), dtype=np.int64)
    for output_index, class_name in enumerate(ANOMALY_PRIORITY, start=1):
        class_index = class_names.index(class_name)
        unresolved = targets == 0
        targets[unresolved & (global_labels[:, class_index] == 1)] = output_index
    return targets


def build_group_sampling_weights(
    targets: np.ndarray,
    sources: np.ndarray,
    traffic_scales: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    scale_keys = np.round(traffic_scales.astype(np.float64), 3)
    groups = list(zip(targets.tolist(), sources.tolist(), scale_keys.tolist(), strict=True))
    group_counts: dict[tuple[int, int, float], int] = {}
    for group in groups:
        group_counts[group] = group_counts.get(group, 0) + 1
    class_weights = config.get("sampling_weights", {})
    weights = np.asarray(
        [
            float(class_weights.get(OUTPUT_CLASSES[target], 1.0))
            / group_counts[(target, int(source), float(scale))]
            for target, source, scale in groups
        ],
        dtype=np.float64,
    )
    weights *= len(weights) / weights.sum()
    return weights


def make_loader(
    x: np.ndarray,
    detector_targets: np.ndarray,
    class_targets: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    sampling_weights: np.ndarray | None = None,
    seed: int = 42,
) -> DataLoader:
    sampler = None
    if sampling_weights is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = WeightedRandomSampler(
            torch.from_numpy(sampling_weights).double(),
            num_samples=len(sampling_weights),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        TensorDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(detector_targets).float(),
            torch.from_numpy(class_targets).long(),
        ),
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        drop_last=False,
    )


def select_balanced_indices(targets: np.ndarray, limit: int | None) -> np.ndarray:
    if limit is None or len(targets) <= limit:
        return np.arange(len(targets))
    groups = [
        np.where(targets == class_index)[0]
        for class_index in range(len(OUTPUT_CLASSES))
    ]
    per_group = max(1, limit // len(groups))
    selected: list[int] = []
    for group in groups:
        if not len(group):
            continue
        positions = np.linspace(0, len(group) - 1, min(per_group, len(group)), dtype=int)
        selected.extend(group[positions].tolist())
    if len(selected) < limit:
        remaining = np.setdiff1d(np.arange(len(targets)), np.asarray(selected))
        selected.extend(remaining[: limit - len(selected)].tolist())
    return np.asarray(sorted(set(selected))[:limit], dtype=int)


def source_filter_mask(
    sources: np.ndarray,
    source_names: list[str],
    include_sources: str | None,
) -> np.ndarray:
    if not include_sources:
        return np.ones(len(sources), dtype=bool)
    requested = {item.strip() for item in include_sources.split(",") if item.strip()}
    include_ids: set[int] = set()
    for item in requested:
        if item.isdigit():
            include_ids.add(int(item))
        elif item in source_names:
            include_ids.add(source_names.index(item))
        else:
            raise ValueError(f"Unknown source in --include-sources: {item}")
    return np.isin(sources, np.asarray(sorted(include_ids), dtype=np.int32))


def focal_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float,
    alpha: float,
    pos_weight: float,
) -> torch.Tensor:
    base = functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=torch.tensor(pos_weight, device=logits.device),
        reduction="none",
    )
    probabilities = torch.sigmoid(logits)
    target_probabilities = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha_weights = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_weights * (1 - target_probabilities).pow(gamma) * base).mean()


def focal_router_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    base = functional.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        reduction="none",
    )
    target_probabilities = torch.softmax(logits, dim=-1).gather(
        1,
        targets[:, None],
    ).squeeze(1)
    return ((1 - target_probabilities).pow(gamma) * base).mean()


def coherent_probabilities(
    detector_probabilities: np.ndarray,
    router_probabilities: np.ndarray,
) -> np.ndarray:
    known = detector_probabilities[:, None]
    return np.concatenate([1 - known, known * router_probabilities], axis=1)


def apply_router_multipliers(
    router_probabilities: np.ndarray,
    multipliers: np.ndarray,
) -> np.ndarray:
    calibrated = router_probabilities * multipliers[None, :]
    denominator = calibrated.sum(axis=1, keepdims=True)
    return calibrated / np.maximum(denominator, 1e-12)


def predict(
    model: DMFGCRNTwoStage,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    detector_probabilities = []
    router_probabilities = []
    model.eval()
    with torch.no_grad():
        for x_batch, _, _ in loader:
            detector_logit, router_logits = model.classify(x_batch.to(device))
            detector_probabilities.append(torch.sigmoid(detector_logit).cpu().numpy())
            router_probabilities.append(torch.softmax(router_logits, dim=-1).cpu().numpy())
    return np.concatenate(detector_probabilities), np.concatenate(router_probabilities)


def metrics(
    detector_targets: np.ndarray,
    class_targets: np.ndarray,
    detector_probabilities: np.ndarray,
    router_probabilities: np.ndarray,
    detector_threshold: float,
) -> dict[str, Any]:
    detector_predictions = detector_probabilities >= detector_threshold
    routed_predictions = router_probabilities.argmax(axis=1) + 1
    final_predictions = np.where(detector_predictions, routed_predictions, 0)
    precision, recall, f1, support = precision_recall_fscore_support(
        class_targets,
        final_predictions,
        labels=np.arange(len(OUTPUT_CLASSES)),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(class_targets, final_predictions)),
        "macro_f1": float(
            f1_score(class_targets, final_predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(class_targets, final_predictions, average="weighted", zero_division=0)
        ),
        "detector_f1": float(
            f1_score(detector_targets, detector_predictions, zero_division=0)
        ),
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
            class_targets,
            final_predictions,
            labels=np.arange(len(OUTPUT_CLASSES)),
        ).tolist(),
    }


def group_detector_metrics(
    detector_targets: np.ndarray,
    detector_probabilities: np.ndarray,
    sources: np.ndarray,
    traffic_scales: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    predictions = detector_probabilities >= threshold
    for source, scale in sorted(
        set(zip(sources.tolist(), np.round(traffic_scales, 3).tolist(), strict=True))
    ):
        mask = (sources == source) & np.isclose(traffic_scales, scale)
        if detector_targets[mask].min() == detector_targets[mask].max():
            continue
        key = f"source_{int(source)}_scale_{float(scale):g}"
        results[key] = {
            "samples": int(mask.sum()),
            "f1": float(
                f1_score(detector_targets[mask], predictions[mask], zero_division=0)
            ),
        }
    return results


def calibrate_validation(
    detector_targets: np.ndarray,
    class_targets: np.ndarray,
    detector_probabilities: np.ndarray,
    router_probabilities: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    multipliers = np.ones(len(ANOMALY_CLASSES), dtype=np.float64)
    best_threshold = 0.5
    best_result: dict[str, Any] | None = None
    for _ in range(2):
        for class_index in range(len(ANOMALY_CLASSES)):
            candidates = []
            for value in (0.75, 1.0, 1.25, 1.5):
                candidate_multipliers = multipliers.copy()
                candidate_multipliers[class_index] = value
                calibrated = apply_router_multipliers(
                    router_probabilities,
                    candidate_multipliers,
                )
                for threshold in np.linspace(0.10, 0.90, 81):
                    result = metrics(
                        detector_targets,
                        class_targets,
                        detector_probabilities,
                        calibrated,
                        float(threshold),
                    )
                    rank = (
                        result["macro_f1"],
                        result["by_class"]["accident"]["recall"],
                        result["by_class"]["normal"]["recall"],
                    )
                    candidates.append(
                        (
                            rank,
                            float(threshold),
                            candidate_multipliers,
                            result,
                        )
                    )
            _, best_threshold, multipliers, best_result = max(
                candidates,
                key=lambda item: item[0],
            )
    if best_result is None:
        raise RuntimeError("Validation calibration produced no candidate")
    return best_threshold, multipliers, best_result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device(config.get("device", "cpu"))
    data = np.load(args.dataset, allow_pickle=False)
    class_names = data["class_names"].astype(str).tolist()
    source_names = (
        data["source_names"].astype(str).tolist()
        if "source_names" in data.files
        else [str(index) for index in range(int(data["train_source"].max()) + 1)]
    )
    split_scales = {
        split: (
            data[f"{split}_traffic_scale"]
            if f"{split}_traffic_scale" in data.files
            else np.where(data[f"{split}_source"] == 5, 0.2, 1.0).astype(np.float32)
        )
        for split in ("train", "val", "test")
    }
    split_targets = {
        split: build_primary_targets(data[f"{split}_global_y"], class_names)
        for split in ("train", "val", "test")
    }
    indices = {}
    for split in ("train", "val", "test"):
        allowed = np.flatnonzero(
            source_filter_mask(
                data[f"{split}_source"],
                source_names,
                args.include_sources,
            )
        )
        selected = select_balanced_indices(
            split_targets[split][allowed],
            args.max_train_samples if split == "train" else args.max_eval_samples,
        )
        indices[split] = allowed[selected]
    train_targets = split_targets["train"][indices["train"]]
    train_scales = split_scales["train"][indices["train"]]
    sampling_weights = build_group_sampling_weights(
        train_targets,
        data["train_source"][indices["train"]],
        train_scales,
        config,
    )
    batch_size = int(config.get("batch_size", 64))
    loaders = {
        split: make_loader(
            data[f"{split}_x"][indices[split]],
            split_targets[split][indices[split]] != 0,
            split_targets[split][indices[split]],
            batch_size=batch_size,
            shuffle=split == "train",
            sampling_weights=sampling_weights if split == "train" else None,
            seed=seed,
        )
        for split in ("train", "val", "test")
    }

    base_payload = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    model = DMFGCRNTwoStage(
        project_root=Path.cwd(),
        conf_path=Path(base_payload["dmfgcrn_conf_path"]),
        forecast_checkpoint_path=Path(base_payload["dmfgcrn_checkpoint_path"]),
        num_nodes=len(base_payload["nodes"]),
        lag=int(base_payload["lag"]),
        num_output_classes=len(ANOMALY_CLASSES),
        device=device,
    )
    model.load_finetuned_backbone(base_payload["model_state_dict"])
    router_class_weights = torch.tensor(
        [
            float(config.get("class_loss_weights", {}).get(class_name, 1.0))
            for class_name in ANOMALY_CLASSES
        ],
        dtype=torch.float32,
        device=device,
    )
    best: dict[str, Any] = {"rank": (-np.inf,) * 4}
    history = []
    stages = [
        ("head", int(config.get("head_epochs", 20)), False),
        ("fine_tune", int(config.get("fine_tune_epochs", 8)), True),
    ]
    for stage_name, epochs, fine_tune in stages:
        if fine_tune:
            model.enable_encoder_finetuning()
        else:
            model.set_backbone_trainable(False)
        parameters = [
            {
                "params": model.detector.parameters(),
                "lr": float(config.get("head_lr", 0.001)),
            },
            {
                "params": model.router.parameters(),
                "lr": float(config.get("head_lr", 0.001)),
            },
        ]
        encoder_parameters = [
            parameter for parameter in model.backbone.parameters() if parameter.requires_grad
        ]
        if encoder_parameters:
            parameters.append(
                {
                    "params": encoder_parameters,
                    "lr": float(config.get("backbone_lr", 0.0001)),
                }
            )
        optimizer = torch.optim.Adam(parameters)
        epochs_without_improvement = 0
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_samples = 0
            for x_batch, detector_batch, class_batch in loaders["train"]:
                x_batch = x_batch.to(device)
                detector_batch = detector_batch.to(device)
                class_batch = class_batch.to(device)
                optimizer.zero_grad()
                detector_logit, router_logits = model.classify(x_batch)
                detector_loss = focal_binary_loss(
                    detector_logit,
                    detector_batch,
                    gamma=float(config.get("detector_focal_gamma", 2.0)),
                    alpha=float(config.get("detector_focal_alpha", 0.55)),
                    pos_weight=float(config.get("detector_pos_weight", 1.0)),
                )
                anomaly_mask = class_batch > 0
                if anomaly_mask.any():
                    router_loss = focal_router_loss(
                        router_logits[anomaly_mask],
                        class_batch[anomaly_mask] - 1,
                        gamma=float(config.get("router_focal_gamma", 2.0)),
                        class_weights=router_class_weights,
                    )
                else:
                    router_loss = router_logits.sum() * 0
                loss = (
                    float(config.get("detector_loss_weight", 1.0)) * detector_loss
                    + float(config.get("router_loss_weight", 1.0)) * router_loss
                )
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(x_batch)
                total_samples += len(x_batch)

            val_detector_prob, val_router_prob = predict(model, loaders["val"], device)
            threshold, multipliers, val_metrics = calibrate_validation(
                split_targets["val"][indices["val"]] != 0,
                split_targets["val"][indices["val"]],
                val_detector_prob,
                val_router_prob,
            )
            groups = group_detector_metrics(
                split_targets["val"][indices["val"]] != 0,
                val_detector_prob,
                data["val_source"][indices["val"]],
                split_scales["val"][indices["val"]],
                threshold,
            )
            worst_group_f1 = min(
                (float(group["f1"]) for group in groups.values()),
                default=val_metrics["detector_f1"],
            )
            rank = (
                worst_group_f1,
                val_metrics["macro_f1"],
                val_metrics["by_class"]["accident"]["recall"],
                val_metrics["by_class"]["normal"]["recall"],
            )
            improved = rank > best["rank"]
            if improved:
                best = {
                    "rank": rank,
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "threshold": threshold,
                    "router_multipliers": multipliers.copy(),
                    "validation_metrics": val_metrics,
                    "validation_group_metrics": groups,
                    "stage": stage_name,
                    "epoch": epoch,
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            history.append(
                {
                    "stage": stage_name,
                    "epoch": epoch,
                    "train_loss": total_loss / max(total_samples, 1),
                    "detector_threshold": threshold,
                    "validation_macro_f1": val_metrics["macro_f1"],
                    "validation_worst_group_f1": worst_group_f1,
                    "is_best": int(improved),
                }
            )
            print(
                f"stage={stage_name} epoch={epoch} "
                f"loss={history[-1]['train_loss']:.6f} "
                f"macro_f1={val_metrics['macro_f1']:.6f} "
                f"worst_group_f1={worst_group_f1:.6f}"
            )
            if epochs_without_improvement >= int(config.get("patience", 5)):
                break

    model.load_state_dict(best["state_dict"])
    test_detector_prob, test_router_prob = predict(model, loaders["test"], device)
    calibrated_test_router = apply_router_multipliers(
        test_router_prob,
        best["router_multipliers"],
    )
    test_metrics = metrics(
        split_targets["test"][indices["test"]] != 0,
        split_targets["test"][indices["test"]],
        test_detector_prob,
        calibrated_test_router,
        float(best["threshold"]),
    )
    test_metrics["group_detector_metrics"] = group_detector_metrics(
        split_targets["test"][indices["test"]] != 0,
        test_detector_prob,
        data["test_source"][indices["test"]],
        split_scales["test"][indices["test"]],
        float(best["threshold"]),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "nodes": base_payload["nodes"],
            "lag": base_payload["lag"],
            "dmfgcrn_conf_path": base_payload["dmfgcrn_conf_path"],
            "dmfgcrn_checkpoint_path": base_payload["dmfgcrn_checkpoint_path"],
            "output_classes": OUTPUT_CLASSES,
            "router_classes": ANOMALY_CLASSES,
            "anomaly_priority": ANOMALY_PRIORITY,
            "detector_threshold": best["threshold"],
            "router_multipliers": best["router_multipliers"].tolist(),
            "validation_metrics": best["validation_metrics"],
            "validation_group_metrics": best["validation_group_metrics"],
            "best_stage": best["stage"],
            "best_epoch": best["epoch"],
        },
        args.output_dir / "dmfgcrn_two_stage.pt",
    )
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    (args.output_dir / "test_summary.json").write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
