from __future__ import annotations

import configparser
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch


def _load_dmfgcrn_network(project_root: Path):
    dmfgcrn_root = project_root / "DMFGCRN"
    if str(dmfgcrn_root) not in sys.path:
        sys.path.insert(0, str(dmfgcrn_root))
    from model.DMFGCRN import DMFGCRN as Network  # type: ignore

    return Network


def _load_dmfgcrn_conf(conf_path: Path) -> configparser.ConfigParser:
    conf = configparser.ConfigParser()
    read_ok = conf.read(conf_path)
    if not read_ok:
        raise FileNotFoundError(f"DMFGCRN config file not found: {conf_path}")
    return conf


def _build_args_from_conf(
    conf: configparser.ConfigParser,
    *,
    num_nodes: int,
    lag: int,
    device: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_nodes=num_nodes,
        lag=lag,
        horizon=conf.getint("data", "horizon"),
        input_dim=conf.getint("model", "input_dim"),
        output_dim=conf.getint("model", "output_dim"),
        embed_dim=conf.getint("model", "embed_dim"),
        rnn_units=conf.getint("model", "rnn_units"),
        num_layers=conf.getint("model", "num_layers"),
        cheb_k=conf.getint("model", "cheb_order"),
        use_day=conf.getboolean("model", "use_day"),
        use_week=conf.getboolean("model", "use_week"),
        default_graph=conf.getboolean("data", "default_graph"),
        device=device,
    )


def build_forecast_windows(
    *,
    x_scaled: np.ndarray,
    timesteps: Sequence[float],
    step_labels: np.ndarray,
    lag: int,
    horizon_index: int,
    input_indices: Sequence[int],
    output_indices: Sequence[int],
    use_day: bool,
    use_week: bool,
    steps_per_day: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start_weekday = 2  # 2020-01-01
    total_steps = x_scaled.shape[0]
    target_offset = lag + horizon_index
    num_samples = total_steps - target_offset
    if num_samples <= 0:
        raise ValueError(
            f"Not enough timesteps ({total_steps}) for lag={lag}, horizon_index={horizon_index}."
        )

    x_main = x_scaled[:, :, input_indices].astype(np.float32)
    y_main = x_scaled[:, :, output_indices].astype(np.float32)
    step_labels = np.asarray(step_labels, dtype=np.int32)
    timestep_arr = np.asarray(timesteps, dtype=np.float64)

    x_list = []
    y_list = []
    label_list = []
    begin_list = []

    for i in range(num_samples):
        x_window = x_main[i : i + lag]
        target_idx = i + target_offset
        target = y_main[target_idx]
        label = int(step_labels[target_idx])
        begin = float(timestep_arr[target_idx])

        if use_day or use_week:
            begin_window = timestep_arr[i : i + lag]
            time_feats = []
            if use_day:
                day_feat = ((begin_window % (24 * 60 * 60)) / (24 * 60 * 60)).reshape(lag, 1, 1)
                day_feat = np.repeat(day_feat, x_window.shape[1], axis=1).astype(np.float32)
                time_feats.append(day_feat)
            if use_week:
                day_index = np.floor(begin_window / (24 * 60 * 60)).astype(np.int64)
                week_feat = ((day_index + start_weekday) % 7) / 7.0
                week_feat = week_feat.reshape(lag, 1, 1)
                week_feat = np.repeat(week_feat, x_window.shape[1], axis=1).astype(np.float32)
                time_feats.append(week_feat)
            x_window = np.concatenate([x_window, *time_feats], axis=-1)

        x_list.append(x_window)
        y_list.append(target)
        label_list.append(label)
        begin_list.append(begin)

    return (
        np.asarray(x_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(label_list, dtype=np.int32),
        np.asarray(begin_list, dtype=np.float64),
    )


class DMFGCRNForecaster:
    def __init__(
        self,
        *,
        project_root: Path,
        checkpoint_path: Path,
        conf_path: Path,
        num_nodes: int,
        lag: int,
        device: torch.device,
    ) -> None:
        conf = _load_dmfgcrn_conf(conf_path)
        args = _build_args_from_conf(conf, num_nodes=num_nodes, lag=lag, device=str(device))
        network_cls = _load_dmfgcrn_network(project_root)
        model = network_cls(args).to(device)

        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()

        self.model = model
        self.args = args
        self.device = device

    def predict(self, x_windows: np.ndarray, batch_size: int, horizon_index: int) -> np.ndarray:
        if horizon_index >= int(self.args.horizon):
            raise ValueError(
                f"dmfgcrn_horizon_index={horizon_index} out of range for horizon={self.args.horizon}."
            )

        preds = []
        with torch.no_grad():
            for i in range(0, len(x_windows), batch_size):
                batch = torch.from_numpy(x_windows[i : i + batch_size]).to(self.device)
                out = self.model(batch)
                preds.append(out[:, horizon_index, :, :].cpu().numpy())
        return np.concatenate(preds, axis=0)
