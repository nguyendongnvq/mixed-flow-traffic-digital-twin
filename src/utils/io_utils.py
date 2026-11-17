from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_summary_files(
    output_dir: Path,
    summary: dict[str, Any],
    summary_df: pd.DataFrame,
    result_df: pd.DataFrame,
    event_eval_alert: pd.DataFrame,
) -> None:
    summary_df.to_csv(output_dir / "pilot_summary.csv", index=False)
    with (output_dir / "pilot_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    result_df.to_csv(output_dir / "accident_window_result.csv", index=False)
    event_eval_alert.to_csv(output_dir / "event_eval_alert.csv", index=False)


def save_model_artifact(
    output_dir: Path,
    model_output_name: str,
    payload: dict[str, Any],
) -> Path:
    output_path = output_dir / model_output_name
    torch.save(payload, output_path)
    return output_path


def save_runtime_config(output_dir: Path, config: dict[str, Any]) -> Path:
    output_path = output_dir / "config.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return output_path

