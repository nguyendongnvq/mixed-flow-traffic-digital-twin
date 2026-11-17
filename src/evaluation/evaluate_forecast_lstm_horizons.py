from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EvalConfig:
    dataset_csv: Path
    time_col: str
    node_col: str
    target_col: str
    input_steps: int
    output_steps: int
    horizons: tuple[int, ...]
    train_ratio: float
    val_ratio: float
    hidden_dim: int
    num_layers: int
    dropout: float
    lr: float
    batch_size: int
    epochs: int
    seed: int
    device: str
    output_root: Path


class ForecastLSTM(nn.Module):
    def __init__(
        self,
        *,
        n_nodes: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        output_steps: int,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.output_steps = output_steps
        self.lstm = nn.LSTM(
            input_size=n_nodes,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, output_steps * n_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        pred = self.head(last)
        return pred.reshape(x.shape[0], self.output_steps, self.n_nodes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate multi-horizon LSTM traffic forecast with horizon-wise metrics."
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed" / "normal" / "forecast_baseline_node_level_normal_only.csv",
    )
    parser.add_argument("--time-col", type=str, default="begin")
    parser.add_argument("--node-col", type=str, default="node_id")
    parser.add_argument("--target-col", type=str, default="flow_count_5m")
    parser.add_argument("--input-steps", type=int, default=24)
    parser.add_argument("--output-steps", type=int, default=72)
    parser.add_argument("--horizons", type=str, default="1,3,6,12,24,36,48,72")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--figure4-horizon",
        type=int,
        default=1,
        help="Forecast horizon index (1-based) used for Figure 4 actual-vs-predicted plot.",
    )
    parser.add_argument(
        "--disable-figure4",
        action="store_true",
        help="Disable Figure 4 artifact export.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "models" / "forecast_eval_runs",
    )
    return parser.parse_args()


def parse_horizons(raw: str, output_steps: int) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not vals:
        raise ValueError("No horizons provided.")
    if any(v <= 0 for v in vals):
        raise ValueError(f"Horizons must be > 0, got: {vals}")
    if any(v > output_steps for v in vals):
        raise ValueError(f"Horizon exceeds output_steps={output_steps}: {vals}")
    return vals


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_matrix(
    *,
    dataset_csv: Path,
    time_col: str,
    node_col: str,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(dataset_csv)
    required = {time_col, node_col, target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    pivot = (
        df.pivot_table(
            index=time_col,
            columns=node_col,
            values=target_col,
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    if pivot.isna().any().any():
        pivot = pivot.fillna(0.0)

    timesteps = pivot.index.to_numpy(dtype=float)
    nodes = [str(c) for c in pivot.columns.tolist()]
    values = pivot.to_numpy(dtype=np.float32)
    return values, timesteps, nodes


def make_windows(
    arr: np.ndarray,
    input_steps: int,
    output_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = arr.shape[0]
    n_nodes = arr.shape[1]
    n = total - input_steps - output_steps + 1
    if n <= 0:
        raise ValueError(
            f"Not enough timesteps={total} for input_steps={input_steps}, output_steps={output_steps}"
        )
    x = np.zeros((n, input_steps, n_nodes), dtype=np.float32)
    y = np.zeros((n, output_steps, n_nodes), dtype=np.float32)
    for i in range(n):
        x[i] = arr[i : i + input_steps]
        y[i] = arr[i + input_steps : i + input_steps + output_steps]
    return x, y


def split_by_time(arr: np.ndarray, train_ratio: float, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = arr.shape[0]
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    train = arr[:train_end]
    val = arr[train_end:val_end]
    test = arr[val_end:]
    return train, val, test


def zscore_fit(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def zscore_apply(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((arr - mean) / std).astype(np.float32)


def build_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[nn.Module, dict[str, list[float]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * xb.shape[0]
            train_count += xb.shape[0]
        train_loss = train_loss_sum / max(train_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss_sum += float(loss.item()) * xb.shape[0]
                val_count += xb.shape[0]
        val_loss = val_loss_sum / max(val_count, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"[Epoch {epoch:03d}] train_mae={train_loss:.6f} val_mae={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training failed: no best state")
    model.load_state_dict(best_state)
    return model, history


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred.astype(np.float32))
            trues.append(yb.numpy().astype(np.float32))
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def metric_rows(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: Iterable[int],
    step_minutes: int = 5,
) -> list[dict[str, float | int]]:
    eps = 1e-6
    rows: list[dict[str, float | int]] = []
    for h in horizons:
        t = y_true[:, h - 1, :].reshape(-1)
        p = y_pred[:, h - 1, :].reshape(-1)
        err = p - t
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        mape = float(np.mean(np.abs(err) / np.maximum(np.abs(t), eps)) * 100.0)
        smape = float(np.mean(200.0 * np.abs(err) / np.maximum(np.abs(p) + np.abs(t), eps)))
        rows.append(
            {
                "Horizon": int(h),
                "Time Ahead": int(h * step_minutes),
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "sMAPE": smape,
            }
        )
    return rows


def infer_step_seconds(timesteps: np.ndarray) -> float:
    if len(timesteps) < 2:
        return float("nan")
    diffs = np.diff(np.sort(timesteps.astype(float)))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return float("nan")
    return float(np.median(diffs))


def _nearest_median_index(values: np.ndarray) -> int:
    arr = np.asarray(values, dtype=float)
    med = float(np.median(arr))
    return int(np.argmin(np.abs(arr - med)))


def export_figure4_artifacts(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_timesteps: np.ndarray,
    nodes: list[str],
    horizon: int,
    target_col: str,
    output_dir: Path,
) -> dict[str, str | int | float]:
    if horizon < 1 or horizon > y_true.shape[1]:
        raise ValueError(f"figure4 horizon out of range: {horizon} (output_steps={y_true.shape[1]})")

    horizon_idx = horizon - 1
    n_samples, _, n_nodes = y_true.shape

    offset = horizon_idx
    target_timestamps = test_timesteps[offset : offset + n_samples].astype(float)
    if len(target_timestamps) != n_samples:
        raise ValueError(
            f"Invalid timestamp mapping for Figure 4: got {len(target_timestamps)} timestamps for {n_samples} samples."
        )

    true_h = y_true[:, horizon_idx, :]
    pred_h = y_pred[:, horizon_idx, :]
    node_mae = np.mean(np.abs(pred_h - true_h), axis=0)
    rep_node_idx = _nearest_median_index(node_mae)
    rep_node_id = str(nodes[rep_node_idx])
    rep_node_mae = float(node_mae[rep_node_idx])

    df_node = pd.DataFrame(
        {
            "timestamp": target_timestamps,
            "actual_flow_count_5m": true_h[:, rep_node_idx].astype(float),
            "predicted_flow_count_5m": pred_h[:, rep_node_idx].astype(float),
        }
    )
    df_node["abs_error"] = (df_node["predicted_flow_count_5m"] - df_node["actual_flow_count_5m"]).abs()
    df_node["day_id"] = np.floor(df_node["timestamp"] / 86400.0).astype(int)

    day_stats = (
        df_node.groupby("day_id", as_index=False)
        .agg(
            day_mae=("abs_error", "mean"),
            points=("timestamp", "size"),
        )
        .sort_values("day_id")
        .reset_index(drop=True)
    )
    if day_stats.empty:
        raise ValueError("No day-level data available for Figure 4.")

    max_points = int(day_stats["points"].max())
    dense_days = day_stats[day_stats["points"] == max_points].copy()
    rep_day_row = dense_days.iloc[_nearest_median_index(dense_days["day_mae"].to_numpy())]
    rep_day_id = int(rep_day_row["day_id"])
    rep_day_mae = float(rep_day_row["day_mae"])
    rep_day_points = int(rep_day_row["points"])

    day_df = df_node[df_node["day_id"] == rep_day_id].copy().sort_values("timestamp").reset_index(drop=True)
    day_df["seconds_in_day"] = day_df["timestamp"] - rep_day_id * 86400.0
    day_df["node_id"] = rep_node_id
    day_df["horizon"] = int(horizon)
    day_df["target_col"] = target_col

    series_path = output_dir / "figure4_series.csv"
    day_df.to_csv(series_path, index=False)

    step_seconds = infer_step_seconds(test_timesteps)
    if np.isnan(step_seconds):
        step_seconds = 300.0
    fig_minutes = float(horizon * step_seconds / 60.0)

    fig_path = output_dir / "figure4_actual_vs_predicted.png"
    plt.figure(figsize=(12, 4.8))
    plt.plot(
        day_df["seconds_in_day"] / 3600.0,
        day_df["actual_flow_count_5m"],
        label=f"Actual {target_col}",
        linewidth=1.8,
    )
    plt.plot(
        day_df["seconds_in_day"] / 3600.0,
        day_df["predicted_flow_count_5m"],
        label=f"Predicted {target_col}",
        linewidth=1.8,
    )
    plt.xlabel("Hour of day")
    plt.ylabel(target_col)
    plt.title(
        f"Figure 4 - Actual vs Predicted {target_col} "
        f"(node={rep_node_id}, day_id={rep_day_id}, horizon={horizon}, ahead={fig_minutes:.1f}m)"
    )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    info = {
        "selection_rule_node": "node_mae_nearest_median",
        "selection_rule_day": "within_max_points_days_day_mae_nearest_median",
        "horizon": int(horizon),
        "time_ahead_minutes": fig_minutes,
        "representative_node_id": rep_node_id,
        "representative_node_mae": rep_node_mae,
        "representative_day_id": rep_day_id,
        "representative_day_mae": rep_day_mae,
        "representative_day_points": rep_day_points,
        "available_days": int(len(day_stats)),
        "series_csv": str(series_path),
        "figure_png": str(fig_path),
    }
    info_path = output_dir / "figure4_representative_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    return {
        "figure_png": str(fig_path),
        "series_csv": str(series_path),
        "representative_info_json": str(info_path),
        "representative_node_id": rep_node_id,
        "representative_day_id": rep_day_id,
        "horizon": int(horizon),
    }


def main() -> None:
    args = parse_args()
    horizons = parse_horizons(args.horizons, args.output_steps)
    if args.figure4_horizon < 1 or args.figure4_horizon > args.output_steps:
        raise ValueError(
            f"--figure4-horizon must be in [1, {args.output_steps}], got {args.figure4_horizon}"
        )

    cfg = EvalConfig(
        dataset_csv=args.dataset_csv,
        time_col=args.time_col,
        node_col=args.node_col,
        target_col=args.target_col,
        input_steps=args.input_steps,
        output_steps=args.output_steps,
        horizons=horizons,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        output_root=args.output_root,
    )

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    values, timesteps, nodes = load_matrix(
        dataset_csv=cfg.dataset_csv,
        time_col=cfg.time_col,
        node_col=cfg.node_col,
        target_col=cfg.target_col,
    )
    train_raw, val_raw, test_raw = split_by_time(values, cfg.train_ratio, cfg.val_ratio)
    mean, std = zscore_fit(train_raw)
    train_z = zscore_apply(train_raw, mean, std)
    val_z = zscore_apply(val_raw, mean, std)
    test_z = zscore_apply(test_raw, mean, std)

    train_x, train_y = make_windows(train_z, cfg.input_steps, cfg.output_steps)
    val_x, val_y = make_windows(val_z, cfg.input_steps, cfg.output_steps)
    test_x, test_y = make_windows(test_z, cfg.input_steps, cfg.output_steps)

    train_loader = build_loader(train_x, train_y, cfg.batch_size, shuffle=True)
    val_loader = build_loader(val_x, val_y, cfg.batch_size, shuffle=False)
    test_loader = build_loader(test_x, test_y, cfg.batch_size, shuffle=False)

    model = ForecastLSTM(
        n_nodes=len(nodes),
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        output_steps=cfg.output_steps,
    ).to(device)

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
        device=device,
    )

    pred_z, true_z = predict(model, test_loader, device)
    pred = pred_z * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    true = true_z * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    rows = metric_rows(y_true=true, y_pred=pred, horizons=cfg.horizons)
    metrics_df = pd.DataFrame(rows)

    run_id = datetime.now().strftime("forecast_eval_%Y%m%d_%H%M%S")
    out_dir = cfg.output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    metrics_path = out_dir / "horizon_metrics_table7.csv"
    metrics_df.to_csv(metrics_path, index=False)

    history_path = out_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    model_path = out_dir / "forecast_lstm.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_nodes": len(nodes),
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout,
            "input_steps": cfg.input_steps,
            "output_steps": cfg.output_steps,
            "target_col": cfg.target_col,
            "time_col": cfg.time_col,
            "node_col": cfg.node_col,
            "nodes": nodes,
            "scaler_mean": mean,
            "scaler_std": std,
            "horizons": list(cfg.horizons),
        },
        model_path,
    )

    figure4_artifacts: dict[str, str | int | float] = {}
    if not args.disable_figure4:
        test_start_idx = train_raw.shape[0] + val_raw.shape[0]
        test_timesteps = timesteps[test_start_idx:]
        figure4_artifacts = export_figure4_artifacts(
            y_true=true,
            y_pred=pred,
            test_timesteps=test_timesteps,
            nodes=nodes,
            horizon=args.figure4_horizon,
            target_col=cfg.target_col,
            output_dir=out_dir,
        )

    summary = {
        "run_id": run_id,
        "output_dir": str(out_dir),
        "dataset_csv": str(cfg.dataset_csv),
        "timesteps": int(values.shape[0]),
        "nodes": int(values.shape[1]),
        "train_timesteps": int(train_raw.shape[0]),
        "val_timesteps": int(val_raw.shape[0]),
        "test_timesteps": int(test_raw.shape[0]),
        "train_samples": int(train_x.shape[0]),
        "val_samples": int(val_x.shape[0]),
        "test_samples": int(test_x.shape[0]),
        "config": asdict(cfg) | {"dataset_csv": str(cfg.dataset_csv), "output_root": str(cfg.output_root)},
        "files": {
            "metrics_csv": str(metrics_path),
            "training_history_json": str(history_path),
            "model_checkpoint": str(model_path),
            "figure4_plot_png": figure4_artifacts.get("figure_png"),
            "figure4_series_csv": figure4_artifacts.get("series_csv"),
            "figure4_info_json": figure4_artifacts.get("representative_info_json"),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] Forecast evaluation run: {run_id}")
    print(f"[DONE] Output dir: {out_dir}")
    print(f"[DONE] Metrics CSV: {metrics_path}")


if __name__ == "__main__":
    main()
