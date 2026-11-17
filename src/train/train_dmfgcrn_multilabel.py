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
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.evaluation.accident_event_detection import (
    apply_temporal_policy,
    calibrate_temporal_policy,
    event_metrics,
)
from src.evaluation.single_label import primary_label_metrics, select_primary_labels
from src.models.dmfgcrn_multilabel import DMFGCRNMultiLabel
from src.models.multilabel_pooling import aggregate_node_probabilities


DEFAULT_CONFIG = Path("configs/dmfgcrn_multilabel.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train unified DMFGCRN multi-label detector.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--head-epochs", type=int)
    parser.add_argument("--fine-tune-epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(
    x: np.ndarray,
    node_y: np.ndarray,
    target: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    sample_weights: np.ndarray | None = None,
    loss_sample_weights: np.ndarray | None = None,
    seed: int = 42,
) -> DataLoader:
    sampler = None
    if sample_weights is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = WeightedRandomSampler(
            torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
    if loss_sample_weights is None:
        loss_sample_weights = np.ones(len(x), dtype=np.float32)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(node_y).float(),
            torch.from_numpy(target).float(),
            torch.from_numpy(loss_sample_weights).float(),
        ),
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        drop_last=False,
    )


def build_sample_weights(
    global_labels: np.ndarray,
    class_names: list[str],
    sampling_config: dict[str, float],
) -> np.ndarray:
    normal_weight = float(sampling_config.get("normal", 1.0))
    weights = np.full(len(global_labels), normal_weight, dtype=np.float64)
    for class_idx, class_name in enumerate(class_names):
        class_weight = float(sampling_config.get(class_name, 1.0))
        weights = np.maximum(
            weights,
            np.where(global_labels[:, class_idx] == 1, class_weight, normal_weight),
        )
    if np.any(weights <= 0):
        raise ValueError("sampling weights must be positive")
    return weights


def build_loss_sample_weights(
    global_labels: np.ndarray,
    sources: np.ndarray,
    begins: np.ndarray,
    class_names: list[str],
    config: dict[str, Any],
) -> np.ndarray:
    weights = np.ones(len(global_labels), dtype=np.float32)
    if not config:
        return weights

    accident_idx = class_names.index("accident")
    mixed_source = int(config.get("mixed_source_id", 5))
    accident_only_source = int(config.get("accident_only_source_id", 1))
    mixed_accident = (sources == mixed_source) & (global_labels[:, accident_idx] == 1)
    weights[mixed_accident] = float(config.get("mixed_accident", 3.0))
    weights[
        (sources == accident_only_source) & (global_labels[:, accident_idx] == 1)
    ] = float(config.get("accident_only", 1.0))

    hard_normal_radius = float(config.get("hard_normal_radius_seconds", 1800.0))
    mixed_normal = (sources == mixed_source) & (global_labels.sum(axis=1) == 0)
    accident_times = begins[mixed_accident].astype(float)
    if accident_times.size:
        distances = np.min(
            np.abs(begins[mixed_normal, None].astype(float) - accident_times[None, :]),
            axis=1,
        )
        hard_normal_indices = np.where(mixed_normal)[0][distances <= hard_normal_radius]
        weights[hard_normal_indices] = float(config.get("mixed_hard_normal", 1.5))
    return weights


class ClassWeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, class_weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.register_buffer("class_weight", class_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        loss = loss * self.class_weight
        per_sample = loss.mean(dim=(1, 2))
        if sample_weights is not None:
            per_sample = per_sample * sample_weights
        return per_sample.mean()


def balanced_indices(global_labels: np.ndarray, limit: int | None) -> np.ndarray:
    if limit is None or len(global_labels) <= limit:
        return np.arange(len(global_labels))
    groups = [np.where(global_labels.sum(axis=1) == 0)[0]]
    groups.extend(np.where(global_labels[:, idx] == 1)[0] for idx in range(global_labels.shape[1]))
    per_group = max(1, limit // len(groups))
    selected: list[int] = []
    for group in groups:
        if len(group) == 0:
            continue
        take = min(per_group, len(group))
        positions = np.linspace(0, len(group) - 1, take, dtype=int)
        selected.extend(group[positions].tolist())
    if len(selected) < limit:
        remaining = np.setdiff1d(np.arange(len(global_labels)), np.asarray(selected))
        selected.extend(remaining[: limit - len(selected)].tolist())
    return np.asarray(sorted(set(selected))[:limit], dtype=int)


def train_epoch(
    model: DMFGCRNMultiLabel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for x_batch, node_y_batch, _, sample_weight_batch in loader:
        x_batch = x_batch.to(device)
        node_y_batch = node_y_batch.to(device)
        sample_weight_batch = sample_weight_batch.to(device)
        optimizer.zero_grad()
        node_logits = model.classify(x_batch)
        loss = criterion(node_logits, node_y_batch, sample_weight_batch)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(x_batch)
        total_samples += len(x_batch)
    return total_loss / max(total_samples, 1)


def predict(
    model: DMFGCRNMultiLabel,
    loader: DataLoader,
    device: torch.device,
    *,
    include_forecast: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    forecasts = []
    targets = []
    with torch.no_grad():
        for x_batch, _, target_batch, _ in loader:
            x_batch = x_batch.to(device)
            if include_forecast:
                node_logits, forecast = model(x_batch)
                forecasts.append(forecast[:, 0, :, :].cpu().numpy())
            else:
                node_logits = model.classify(x_batch)
            probabilities.append(torch.sigmoid(node_logits).cpu().numpy())
            targets.append(target_batch.numpy())
    return (
        np.concatenate(probabilities, axis=0),
        np.concatenate(forecasts, axis=0) if forecasts else np.empty((0,), dtype=np.float32),
        np.concatenate(targets, axis=0),
    )


def choose_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    target_fp_max: float,
) -> np.ndarray:
    thresholds = []
    candidates = np.concatenate([np.linspace(0.01, 0.99, 197), np.array([1.0])])
    for class_idx in range(labels.shape[1]):
        y_true = labels[:, class_idx].astype(int)
        y_prob = probabilities[:, class_idx]
        rows: list[tuple[float, float, float]] = []
        for threshold in candidates:
            y_pred = y_prob >= threshold
            negatives = y_true == 0
            fp_rate = float(y_pred[negatives].mean()) if negatives.any() else 0.0
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            rows.append((f1, -fp_rate, float(threshold)))
        feasible = [
            row for row in rows if -row[1] <= target_fp_max
        ]
        selected = max(feasible if feasible else rows)
        thresholds.append(selected[2])
    return np.asarray(thresholds, dtype=np.float32)


def metrics_table(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    class_names: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    predictions = probabilities >= thresholds.reshape(1, -1)
    rows = []
    for class_idx, class_name in enumerate(class_names):
        y_true = labels[:, class_idx].astype(int)
        y_pred = predictions[:, class_idx].astype(int)
        negatives = y_true == 0
        rows.append(
            {
                "class_name": class_name,
                "threshold": float(thresholds[class_idx]),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "pr_auc": average_precision_score(y_true, probabilities[:, class_idx]),
                "fp_rate": float(y_pred[negatives].mean()) if negatives.any() else 0.0,
                "support": int(y_true.sum()),
            }
        )
    summary = {
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "exact_match_ratio": float(np.all(labels == predictions, axis=1).mean()),
        "hamming_loss": float(hamming_loss(labels, predictions)),
    }
    return pd.DataFrame(rows), summary


def checkpoint_rank(
    metrics: pd.DataFrame,
    *,
    target_fp_max: float,
    class_f1_floors: dict[str, float],
) -> tuple[float, ...]:
    by_class = metrics.set_index("class_name")
    accident = by_class.loc["accident"]
    floors_pass = all(
        class_name in by_class.index
        and float(by_class.loc[class_name, "f1"]) >= float(floor)
        for class_name, floor in class_f1_floors.items()
    )
    fp_pass = float(accident["fp_rate"]) <= target_fp_max
    return (
        float(floors_pass and fp_pass),
        float(fp_pass),
        float(accident["f1"]),
        float(accident["recall"]),
        -float(accident["fp_rate"]),
        float(metrics["f1"].mean()),
    )


def event_checkpoint_rank(
    metrics: pd.DataFrame,
    event_summary: dict[str, float | int],
    *,
    target_fp_max: float,
    class_f1_floors: dict[str, float],
    max_false_alerts_per_day: float,
) -> tuple[float, ...]:
    by_class = metrics.set_index("class_name")
    accident = by_class.loc["accident"]
    floors_pass = all(
        class_name in by_class.index
        and float(by_class.loc[class_name, "f1"]) >= float(floor)
        for class_name, floor in class_f1_floors.items()
    )
    fp_pass = float(accident["fp_rate"]) <= target_fp_max
    alert_budget_pass = (
        float(event_summary["false_alerts_per_day"]) <= max_false_alerts_per_day
    )
    return (
        float(floors_pass and fp_pass and alert_budget_pass),
        float(alert_budget_pass),
        float(event_summary["event_f1"]),
        float(event_summary["event_recall"]),
        -float(event_summary["false_alerts_per_day"]),
        float(accident["f1"]),
    )


def accident_event_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    begins: np.ndarray,
    sources: np.ndarray,
    class_names: list[str],
    *,
    max_gap_seconds: float = 300.0,
) -> dict[str, float | int]:
    accident_idx = class_names.index("accident")
    event_count = 0
    detected_count = 0
    for source in np.unique(sources):
        source_indices = np.where(sources == source)[0]
        order = source_indices[np.argsort(begins[source_indices])]
        active_event: list[int] = []
        previous_begin: float | None = None
        for sample_idx in order:
            is_positive = bool(labels[sample_idx, accident_idx])
            begin = float(begins[sample_idx])
            continues = (
                is_positive
                and active_event
                and previous_begin is not None
                and begin - previous_begin <= max_gap_seconds
            )
            if is_positive and not continues:
                if active_event:
                    event_count += 1
                    detected_count += int(predictions[active_event, accident_idx].any())
                active_event = [int(sample_idx)]
            elif continues:
                active_event.append(int(sample_idx))
            elif active_event:
                event_count += 1
                detected_count += int(predictions[active_event, accident_idx].any())
                active_event = []
            previous_begin = begin
        if active_event:
            event_count += 1
            detected_count += int(predictions[active_event, accident_idx].any())
    return {
        "accident_event_count": event_count,
        "accident_event_detected": detected_count,
        "accident_event_recall": detected_count / event_count if event_count else 0.0,
    }


def run_stage(
    *,
    name: str,
    epochs: int,
    model: DMFGCRNMultiLabel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    val_labels: np.ndarray,
    class_names: list[str],
    pooling_config: dict[str, Any],
    target_fp_max: float,
    class_f1_floors: dict[str, float],
    val_begins: np.ndarray,
    val_sources: np.ndarray,
    temporal_policy_config: dict[str, Any],
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    best_state: dict[str, Any],
) -> list[dict[str, float | int | str]]:
    history = []
    epochs_without_improvement = 0
    stage_best_event_f1 = -np.inf
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_prob_node, _, _ = predict(
            model,
            val_loader,
            device,
            include_forecast=False,
        )
        val_prob = aggregate_node_probabilities(
            val_prob_node,
            class_names,
            pooling_config,
        )
        thresholds = choose_thresholds(
            val_labels,
            val_prob,
            target_fp_max=target_fp_max,
        )
        val_metrics, val_summary = metrics_table(
            val_labels,
            val_prob,
            thresholds,
            class_names,
        )
        accident_idx = class_names.index("accident")
        mixed_source_id = int(temporal_policy_config.get("mixed_source_id", 5))
        mixed_mask = val_sources == mixed_source_id
        temporal_policy, temporal_metrics = calibrate_temporal_policy(
            val_labels[mixed_mask, accident_idx],
            val_prob[mixed_mask, accident_idx],
            val_begins[mixed_mask],
            temporal_policy_config,
        )
        rank = event_checkpoint_rank(
            val_metrics,
            temporal_metrics,
            target_fp_max=target_fp_max,
            class_f1_floors=class_f1_floors,
            max_false_alerts_per_day=float(
                temporal_policy_config.get("max_false_alerts_per_day", 2.0)
            ),
        )
        accident_metrics = val_metrics.set_index("class_name").loc["accident"]
        scheduler.step(float(temporal_metrics["event_f1"]))
        improved = rank > best_state["rank"]
        meaningful_improvement = (
            float(temporal_metrics["event_f1"])
            >= stage_best_event_f1 + early_stopping_min_delta
        )
        if improved:
            best_state.update(
                {
                    "rank": rank,
                    "accident_f1": float(accident_metrics["f1"]),
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "thresholds": thresholds.copy(),
                    "metrics": val_metrics.copy(),
                    "summary": dict(val_summary),
                    "temporal_policy": temporal_policy,
                    "temporal_metrics": dict(temporal_metrics),
                    "stage": name,
                    "epoch": epoch,
                }
            )
        if meaningful_improvement:
            stage_best_event_f1 = float(temporal_metrics["event_f1"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row: dict[str, float | int | str] = {
            "stage": name,
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "accident_val_f1": float(accident_metrics["f1"]),
            "accident_val_recall": float(accident_metrics["recall"]),
            "accident_val_fp_rate": float(accident_metrics["fp_rate"]),
            "macro_val_f1": float(val_summary["macro_f1"]),
            "accident_event_val_f1": float(temporal_metrics["event_f1"]),
            "accident_event_val_recall": float(temporal_metrics["event_recall"]),
            "accident_false_alerts_per_day": float(
                temporal_metrics["false_alerts_per_day"]
            ),
            "is_best": int(improved),
        }
        history.append(row)
        print(
            f"stage={name} epoch={epoch} train_loss={train_loss:.6f} "
            f"accident_f1={float(accident_metrics['f1']):.6f} "
            f"accident_recall={float(accident_metrics['recall']):.6f} "
            f"accident_fp={float(accident_metrics['fp_rate']):.6f} "
            f"event_f1={float(temporal_metrics['event_f1']):.6f} "
            f"event_recall={float(temporal_metrics['event_recall']):.6f} "
            f"false_alerts_day={float(temporal_metrics['false_alerts_per_day']):.6f}"
        )
        if epochs_without_improvement >= early_stopping_patience:
            print(f"stage={name} early_stopping epoch={epoch}")
            break
    return history


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = json.loads(args.config.read_text(encoding="utf-8"))
    dataset_path = args.dataset or Path(config["dataset_output"])
    output_dir = args.output_dir or Path(config["training_output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device(config.get("device", "cpu"))
    data = np.load(dataset_path, allow_pickle=False)
    class_names = data["class_names"].astype(str).tolist()
    nodes = data["nodes"].astype(str).tolist()

    train_x = data["train_x"]
    train_node_y = data["train_node_y"]
    train_target = data["train_target"]
    train_global_y = data["train_global_y"]
    train_source = data["train_source"]
    train_begin = data["train_begin"]
    if args.max_train_samples and len(train_x) > args.max_train_samples:
        selected = balanced_indices(train_global_y, args.max_train_samples)
        train_x = train_x[selected]
        train_node_y = train_node_y[selected]
        train_target = train_target[selected]
        train_global_y = train_global_y[selected]
        train_source = train_source[selected]
        train_begin = train_begin[selected]

    batch_size = int(config["batch_size"])
    sampling_config = {
        str(key): float(value)
        for key, value in config.get("sampling_weights", {}).items()
    }
    sample_weights = (
        build_sample_weights(train_global_y, class_names, sampling_config)
        if sampling_config
        else None
    )
    loss_sample_weights = build_loss_sample_weights(
        train_global_y,
        train_source,
        train_begin,
        class_names,
        dict(config.get("source_loss_weights", {})),
    )
    train_loader = make_loader(
        train_x,
        train_node_y,
        train_target,
        batch_size=batch_size,
        shuffle=True,
        sample_weights=sample_weights,
        loss_sample_weights=loss_sample_weights,
        seed=seed,
    )
    val_indices = balanced_indices(data["val_global_y"], args.max_eval_samples)
    test_indices = balanced_indices(data["test_global_y"], args.max_eval_samples)
    val_loader = make_loader(
        data["val_x"][val_indices],
        data["val_node_y"][val_indices],
        data["val_target"][val_indices],
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = make_loader(
        data["test_x"][test_indices],
        data["test_node_y"][test_indices],
        data["test_target"][test_indices],
        batch_size=batch_size,
        shuffle=False,
    )

    model = DMFGCRNMultiLabel(
        project_root=Path.cwd(),
        conf_path=Path(config["dmfgcrn_conf_path"]),
        forecast_checkpoint_path=Path(config["dmfgcrn_checkpoint_path"]),
        num_nodes=len(nodes),
        lag=int(config["lag"]),
        num_classes=len(class_names),
        device=device,
    )

    positive = train_node_y.sum(axis=(0, 1))
    total = train_node_y.shape[0] * train_node_y.shape[1]
    negative = total - positive
    pos_weight = np.divide(
        negative,
        np.maximum(positive, 1),
    ).astype(np.float32)
    pos_weight = np.minimum(pos_weight, float(config.get("pos_weight_cap", np.inf)))
    configured_class_weights = config.get("class_loss_weights", {})
    class_weight = np.asarray(
        [float(configured_class_weights.get(name, 1.0)) for name in class_names],
        dtype=np.float32,
    )
    criterion = ClassWeightedBCEWithLogitsLoss(
        pos_weight=torch.from_numpy(pos_weight).to(device).reshape(1, 1, -1),
        class_weight=torch.from_numpy(class_weight).to(device).reshape(1, 1, -1),
    )

    history: list[dict[str, float | int | str]] = []
    pooling_config = dict(config.get("accident_pooling", {}))
    class_f1_floors = {
        str(key): float(value)
        for key, value in config.get("class_f1_floors", {}).items()
    }
    early_stopping = dict(config.get("early_stopping", {}))
    temporal_policy_config = dict(config.get("accident_temporal_policy", {}))
    best_state: dict[str, Any] = {
        "rank": (-np.inf,) * 6,
        "accident_f1": -np.inf,
    }
    model.set_backbone_trainable(False)
    head_epochs = args.head_epochs if args.head_epochs is not None else int(config["head_epochs"])
    head_optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=float(config["head_lr"]),
    )
    history.extend(
        run_stage(
            name="head",
            epochs=head_epochs,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=head_optimizer,
            device=device,
            val_labels=data["val_global_y"][val_indices],
            class_names=class_names,
            pooling_config=pooling_config,
            target_fp_max=float(config["target_fp_max"]),
            class_f1_floors=class_f1_floors,
            val_begins=data["val_begin"][val_indices],
            val_sources=data["val_source"][val_indices],
            temporal_policy_config=temporal_policy_config,
            early_stopping_patience=int(early_stopping.get("patience", 5)),
            early_stopping_min_delta=float(early_stopping.get("min_delta", 0.002)),
            best_state=best_state,
        )
    )
    model.load_state_dict(best_state["model_state_dict"])

    fine_tune_epochs = (
        args.fine_tune_epochs
        if args.fine_tune_epochs is not None
        else int(config["fine_tune_epochs"])
    )
    if fine_tune_epochs > 0:
        model.enable_encoder_finetuning()
        fine_tune_optimizer = torch.optim.Adam(
            [
                {"params": model.classifier.parameters(), "lr": float(config["head_lr"])},
                {
                    "params": [
                        parameter
                        for parameter in model.backbone.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": float(config["backbone_lr"]),
                },
            ]
        )
        history.extend(
            run_stage(
                name="fine_tune",
                epochs=fine_tune_epochs,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=fine_tune_optimizer,
                device=device,
                val_labels=data["val_global_y"][val_indices],
                class_names=class_names,
                pooling_config=pooling_config,
                target_fp_max=float(config["target_fp_max"]),
                class_f1_floors=class_f1_floors,
                val_begins=data["val_begin"][val_indices],
                val_sources=data["val_source"][val_indices],
                temporal_policy_config=temporal_policy_config,
                early_stopping_patience=int(early_stopping.get("patience", 5)),
                early_stopping_min_delta=float(early_stopping.get("min_delta", 0.002)),
                best_state=best_state,
            )
        )

    model.load_state_dict(best_state["model_state_dict"])
    val_node_prob, val_forecast, val_target = predict(model, val_loader, device)
    val_prob = aggregate_node_probabilities(
        val_node_prob,
        class_names,
        pooling_config,
    )
    thresholds = np.asarray(best_state["thresholds"], dtype=np.float32)
    normal_val = data["val_global_y"][val_indices].sum(axis=1) == 0
    val_residual = np.abs(val_forecast - val_target).mean(axis=2).max(axis=1)
    generic_threshold = float(
        np.percentile(
            val_residual[normal_val],
            float(config["generic_threshold_percentile"]),
        )
    )

    test_node_prob, test_forecast, test_target = predict(model, test_loader, device)
    test_prob = aggregate_node_probabilities(
        test_node_prob,
        class_names,
        pooling_config,
    )
    metrics_df, summary = metrics_table(
        data["test_global_y"][test_indices],
        test_prob,
        thresholds,
        class_names,
    )
    test_residual = np.abs(test_forecast - test_target).mean(axis=2).max(axis=1)
    normal_test = data["test_global_y"][test_indices].sum(axis=1) == 0
    summary["generic_normal_fp_rate"] = float(
        (test_residual[normal_test] > generic_threshold).mean()
    )
    summary["generic_threshold"] = generic_threshold
    test_predictions = test_prob >= thresholds.reshape(1, -1)
    summary.update(
        accident_event_metrics(
            data["test_global_y"][test_indices],
            test_predictions,
            data["test_begin"][test_indices],
            data["test_source"][test_indices],
            class_names,
        )
    )
    accident_idx = class_names.index("accident")
    mixed_source_id = int(temporal_policy_config.get("mixed_source_id", 5))
    mixed_test = data["test_source"][test_indices] == mixed_source_id
    test_accident_alerts = apply_temporal_policy(
        test_prob[mixed_test, accident_idx],
        data["test_begin"][test_indices][mixed_test],
        best_state["temporal_policy"],
    )
    output_predictions = test_predictions.copy()
    output_predictions[mixed_test, accident_idx] = test_accident_alerts
    single_label_policy = dict(config.get("single_label_policy", {}))
    primary_labels = select_primary_labels(
        class_names,
        test_prob,
        thresholds,
        output_predictions,
        test_residual,
        generic_threshold=generic_threshold,
        policy=single_label_policy,
    )
    summary.update(
        primary_label_metrics(
            data["test_global_y"][test_indices],
            primary_labels,
            class_names,
            policy=single_label_policy,
        )
    )
    mixed_event_summary = event_metrics(
        data["test_global_y"][test_indices][mixed_test, accident_idx],
        test_accident_alerts,
        data["test_begin"][test_indices][mixed_test],
        max_latency_seconds=float(
            best_state["temporal_policy"].get("max_latency_seconds", 600.0)
        ),
        step_seconds=float(best_state["temporal_policy"].get("step_seconds", 300.0)),
    )
    summary["mixed_accident_event"] = mixed_event_summary

    checkpoint_path = output_dir / "dmfgcrn_multilabel.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "class_thresholds": thresholds,
            "generic_threshold": generic_threshold,
            "nodes": nodes,
            "lag": int(config["lag"]),
            "dmfgcrn_conf_path": config["dmfgcrn_conf_path"],
            "dmfgcrn_checkpoint_path": config["dmfgcrn_checkpoint_path"],
            "scaler_mean": data["scaler_mean"],
            "scaler_scale": data["scaler_scale"],
            "input_features": data["input_features"].astype(str).tolist(),
            "output_features": data["output_features"].astype(str).tolist(),
            "pooling_config": pooling_config,
            "single_label_policy": single_label_policy,
            "best_stage": best_state["stage"],
            "best_epoch": best_state["epoch"],
            "validation_metrics": best_state["metrics"].to_dict(orient="records"),
            "validation_summary": best_state["summary"],
            "accident_temporal_policy": best_state["temporal_policy"],
            "validation_accident_event_metrics": best_state["temporal_metrics"],
        },
        checkpoint_path,
    )
    metrics_df.to_csv(output_dir / "test_metrics_by_class.csv", index=False)
    best_state["metrics"].to_csv(
        output_dir / "validation_metrics_by_class.csv",
        index=False,
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "test_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    print(metrics_df.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
