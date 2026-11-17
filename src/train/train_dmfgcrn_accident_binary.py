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
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.accident_event_detection import (
    apply_temporal_policy,
    calibrate_temporal_policy,
    daily_event_metrics,
    event_metrics,
)
from src.models.dmfgcrn_accident_binary import DMFGCRNAccidentBinary
from src.train.train_dmfgcrn_multilabel import build_loss_sample_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dedicated DMFGCRN accident head.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    sample_weights: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(node_y).float(),
            torch.from_numpy(sample_weights).float(),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def stratified_indices(
    global_labels: np.ndarray,
    sources: np.ndarray,
    accident_idx: int,
    limit: int | None,
) -> np.ndarray:
    if limit is None or len(global_labels) <= limit:
        return np.arange(len(global_labels))
    groups = []
    for source in np.unique(sources):
        source_mask = sources == source
        groups.append(np.where(source_mask & (global_labels[:, accident_idx] == 0))[0])
        groups.append(np.where(source_mask & (global_labels[:, accident_idx] == 1))[0])
    groups = [group for group in groups if len(group)]
    per_group = max(1, limit // len(groups))
    selected = []
    for group in groups:
        take = min(per_group, len(group))
        positions = np.linspace(0, len(group) - 1, take, dtype=int)
        selected.extend(group[positions].tolist())
    if len(selected) < limit:
        remaining = np.setdiff1d(np.arange(len(global_labels)), np.asarray(selected))
        selected.extend(remaining[: limit - len(selected)].tolist())
    return np.asarray(sorted(set(selected))[:limit], dtype=int)


def evaluation_indices(
    global_labels: np.ndarray,
    sources: np.ndarray,
    accident_idx: int,
    limit: int | None,
    *,
    mixed_source_id: int,
) -> np.ndarray:
    if limit is None or len(global_labels) <= limit:
        return np.arange(len(global_labels))
    mixed_indices = np.where(sources == mixed_source_id)[0]
    remaining_limit = max(0, limit - len(mixed_indices))
    other_indices = np.where(sources != mixed_source_id)[0]
    selected_other = stratified_indices(
        global_labels[other_indices],
        sources[other_indices],
        accident_idx,
        remaining_limit,
    )
    return np.sort(
        np.concatenate([mixed_indices, other_indices[selected_other]])
    )


def predict(
    model: DMFGCRNAccidentBinary,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    probabilities = []
    model.eval()
    with torch.no_grad():
        for x_batch, _, _ in loader:
            node_logits = model.classify(x_batch.to(device))
            probabilities.append(torch.sigmoid(node_logits).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def train_epoch(
    model: DMFGCRNAccidentBinary,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for x_batch, node_y_batch, sample_weight_batch in loader:
        x_batch = x_batch.to(device)
        node_y_batch = node_y_batch.to(device)
        sample_weight_batch = sample_weight_batch.to(device)
        optimizer.zero_grad()
        logits = model.classify(x_batch)
        loss = criterion(logits, node_y_batch).mean(dim=(1, 2))
        loss = (loss * sample_weight_batch).mean()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(x_batch)
        total_samples += len(x_batch)
    return total_loss / max(total_samples, 1)


def window_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    negatives = labels == 0
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "fp_rate": float(predictions[negatives].mean()) if negatives.any() else 0.0,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    config: dict[str, Any] = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device(config.get("device", "cpu"))
    data = np.load(args.dataset, allow_pickle=False)
    class_names = data["class_names"].astype(str).tolist()
    accident_idx = class_names.index("accident")
    mixed_source_id = 5

    train_indices = stratified_indices(
        data["train_global_y"],
        data["train_source"],
        accident_idx,
        args.max_train_samples,
    )
    val_indices = evaluation_indices(
        data["val_global_y"],
        data["val_source"],
        accident_idx,
        args.max_eval_samples,
        mixed_source_id=mixed_source_id,
    )
    test_indices = evaluation_indices(
        data["test_global_y"],
        data["test_source"],
        accident_idx,
        args.max_eval_samples,
        mixed_source_id=mixed_source_id,
    )

    source_weight_config = {
        "mixed_source_id": mixed_source_id,
        "accident_only_source_id": 1,
        **config.get(
            "accident_binary_source_weights",
            {
                "mixed_accident": 3.0,
                "accident_only": 1.0,
                "mixed_hard_normal": 1.5,
                "hard_normal_radius_seconds": 1800,
            },
        ),
    }
    train_weights = build_loss_sample_weights(
        data["train_global_y"][train_indices],
        data["train_source"][train_indices],
        data["train_begin"][train_indices],
        class_names,
        source_weight_config,
    )
    hard_negative_weights = {
        int(source_id): float(weight)
        for source_id, weight in config.get(
            "accident_binary_hard_negative_weights",
            {"0": 2.0, "2": 4.0, "3": 5.0, "4": 1.5},
        ).items()
    }
    train_sources = data["train_source"][train_indices]
    train_accident_labels = data["train_global_y"][train_indices, accident_idx]
    for source_id, weight in hard_negative_weights.items():
        hard_negative_mask = (
            (train_sources == source_id)
            & (train_accident_labels == 0)
        )
        train_weights[hard_negative_mask] *= weight
    train_loader = make_loader(
        data["train_x"][train_indices],
        data["train_node_y"][train_indices, :, accident_idx : accident_idx + 1],
        train_weights,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )
    val_loader = make_loader(
        data["val_x"][val_indices],
        data["val_node_y"][val_indices, :, accident_idx : accident_idx + 1],
        np.ones(len(val_indices), dtype=np.float32),
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )
    test_loader = make_loader(
        data["test_x"][test_indices],
        data["test_node_y"][test_indices, :, accident_idx : accident_idx + 1],
        np.ones(len(test_indices), dtype=np.float32),
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    base_payload = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    model = DMFGCRNAccidentBinary(
        project_root=Path.cwd(),
        conf_path=Path(base_payload["dmfgcrn_conf_path"]),
        forecast_checkpoint_path=Path(base_payload["dmfgcrn_checkpoint_path"]),
        num_nodes=len(base_payload["nodes"]),
        lag=int(base_payload["lag"]),
        device=device,
    )
    model.load_finetuned_backbone(base_payload["model_state_dict"])
    positive = data["train_node_y"][
        train_indices, :, accident_idx : accident_idx + 1
    ].sum()
    total = (
        len(train_indices)
        * data["train_node_y"].shape[1]
    )
    pos_weight = min((total - positive) / max(float(positive), 1.0), 20.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([[[pos_weight]]], dtype=torch.float32, device=device),
        reduction="none",
    )

    temporal_config = {
        "mixed_source_id": mixed_source_id,
        "rolling_windows_options": [1, 2],
        "confirmation_windows_options": [1, 2],
        "cooldown_seconds_options": [900, 1800],
        "threshold_values": np.linspace(0.10, 0.90, 17).tolist(),
        "emergency_deltas": [0.20],
        "step_seconds": 300,
        "max_latency_seconds": 600,
        "max_false_alerts_per_day": 2.0,
    }
    best: dict[str, Any] = {"rank": (-np.inf,) * 6}
    history = []
    stages = [
        ("head", int(config.get("accident_head_epochs", 20)), False),
        ("fine_tune", int(config.get("accident_fine_tune_epochs", 12)), True),
    ]
    for stage_name, epochs, fine_tune in stages:
        if fine_tune:
            model.enable_encoder_finetuning()
        else:
            model.set_backbone_trainable(False)
        optimizer = torch.optim.Adam(
            [
                {"params": model.classifier.parameters(), "lr": 0.001},
                *(
                    [
                        {
                            "params": [
                                parameter
                                for parameter in model.backbone.parameters()
                                if parameter.requires_grad
                            ],
                            "lr": 0.0001,
                        }
                    ]
                    if fine_tune
                    else []
                ),
            ]
        )
        epochs_without_improvement = 0
        for epoch in range(1, epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_node_prob = predict(model, val_loader, device)
            val_prob = val_node_prob.max(axis=1).reshape(-1)
            val_labels = data["val_global_y"][val_indices, accident_idx]
            val_sources = data["val_source"][val_indices]
            mixed_mask = val_sources == mixed_source_id
            policy, event_summary = calibrate_temporal_policy(
                val_labels[mixed_mask],
                val_prob[mixed_mask],
                data["val_begin"][val_indices][mixed_mask],
                temporal_config,
            )
            daily = daily_event_metrics(
                val_labels[mixed_mask],
                apply_temporal_policy(
                    val_prob[mixed_mask],
                    data["val_begin"][val_indices][mixed_mask],
                    policy,
                ),
                data["val_begin"][val_indices][mixed_mask],
                max_latency_seconds=600,
            )
            max_daily_false = max(
                (float(row["false_alerts"]) for row in daily),
                default=0.0,
            )
            metrics = window_metrics(val_labels, val_prob, float(policy["threshold"]))
            cross_negative_mask = np.isin(val_sources, [0, 2, 3, 4])
            cross_anomaly_fp = float(
                (
                    val_prob[cross_negative_mask]
                    >= float(policy["threshold"])
                ).mean()
            )
            feasible = (
                max_daily_false <= 2
                and metrics["fp_rate"] <= 0.15
                and cross_anomaly_fp
                <= float(config.get("accident_cross_anomaly_fp_max", 0.10))
            )
            rank = (
                float(feasible),
                float(event_summary["event_recall"]),
                float(event_summary["event_f1"]),
                -max_daily_false,
                -cross_anomaly_fp,
                float(metrics["f1"]),
            )
            improved = rank > best["rank"]
            if improved:
                best = {
                    "rank": rank,
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "policy": policy,
                    "event_metrics": event_summary,
                    "window_metrics": metrics,
                    "cross_anomaly_fp": cross_anomaly_fp,
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
                    "train_loss": loss,
                    "event_recall": event_summary["event_recall"],
                    "event_f1": event_summary["event_f1"],
                    "false_alerts_per_day": event_summary["false_alerts_per_day"],
                    "window_f1": metrics["f1"],
                    "window_fp": metrics["fp_rate"],
                    "cross_anomaly_fp": cross_anomaly_fp,
                    "is_best": int(improved),
                }
            )
            print(
                f"stage={stage_name} epoch={epoch} loss={loss:.6f} "
                f"event_recall={event_summary['event_recall']:.6f} "
                f"event_f1={event_summary['event_f1']:.6f}"
            )
            if epochs_without_improvement >= 5:
                break

    model.load_state_dict(best["state_dict"])
    test_node_prob = predict(model, test_loader, device)
    test_prob = test_node_prob.max(axis=1).reshape(-1)
    test_labels = data["test_global_y"][test_indices, accident_idx]
    test_sources = data["test_source"][test_indices]
    mixed_test = test_sources == mixed_source_id
    test_alerts = apply_temporal_policy(
        test_prob[mixed_test],
        data["test_begin"][test_indices][mixed_test],
        best["policy"],
    )
    test_event_summary = event_metrics(
        test_labels[mixed_test],
        test_alerts,
        data["test_begin"][test_indices][mixed_test],
        max_latency_seconds=600,
    )
    test_window_metrics = window_metrics(
        test_labels,
        test_prob,
        float(best["policy"]["threshold"]),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "nodes": base_payload["nodes"],
            "lag": base_payload["lag"],
            "dmfgcrn_conf_path": base_payload["dmfgcrn_conf_path"],
            "dmfgcrn_checkpoint_path": base_payload["dmfgcrn_checkpoint_path"],
            "temporal_policy": best["policy"],
            "best_stage": best["stage"],
            "best_epoch": best["epoch"],
            "validation_event_metrics": best["event_metrics"],
            "validation_window_metrics": best["window_metrics"],
        },
        args.output_dir / "dmfgcrn_accident_binary.pt",
    )
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    summary = {
        "test_event_metrics": test_event_summary,
        "test_window_metrics": test_window_metrics,
    }
    (args.output_dir / "test_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
