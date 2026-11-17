from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        optimizer.zero_grad()
        x_recon = model(x_batch)
        loss = criterion(x_recon, x_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            x_recon = model(x_batch)
            loss = criterion(x_recon, x_batch)
            losses.append(loss.item())
    return float(np.mean(losses))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, Any]]:
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
            }
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")

    model.load_state_dict(best_state["model_state"])
    print("Best epoch:", best_state["epoch"])
    print("Best val loss:", best_state["val_loss"])
    return model, history, best_state

