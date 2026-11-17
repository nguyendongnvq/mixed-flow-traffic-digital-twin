from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


VARIANTS = {
    "temporal_calibration": {
        "source_loss_weights": {},
    },
    "source_weighted": {
        "source_loss_weights": {
            "mixed_source_id": 5,
            "accident_only_source_id": 1,
            "mixed_accident": 3.0,
            "accident_only": 1.0,
            "mixed_hard_normal": 1.0,
            "hard_normal_radius_seconds": 1800,
        },
    },
    "source_weighted_hard_normal": {
        "source_loss_weights": {
            "mixed_source_id": 5,
            "accident_only_source_id": 1,
            "mixed_accident": 3.0,
            "accident_only": 1.0,
            "mixed_hard_normal": 1.5,
            "hard_normal_radius_seconds": 1800,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run accident event-level ablations.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dmfgcrn_multilabel.json"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "datasets/processed/multilabel/"
            "dmfgcrn_multilabel_windows_with_mixed_14d.npz"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("models/outputs_dmfgcrn_accident_event_ablation"),
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def run(command: list[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}); see {log_path}")


def summarize(output_root: Path, seeds: list[int]) -> None:
    rows = []
    for variant in VARIANTS:
        for seed in seeds:
            output_dir = output_root / f"{variant}_seed_{seed}"
            metrics_path = output_dir / "test_metrics_by_class.csv"
            summary_path = output_dir / "test_summary.json"
            if not metrics_path.exists() or not summary_path.exists():
                continue
            metrics = pd.read_csv(metrics_path).set_index("class_name")
            event = json.loads(summary_path.read_text(encoding="utf-8"))[
                "mixed_accident_event"
            ]
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    **event,
                    "accident_window_f1": metrics.loc["accident", "f1"],
                    "accident_window_fp": metrics.loc["accident", "fp_rate"],
                    "congestion_spike_f1": metrics.loc["congestion_spike", "f1"],
                    "zigzag_f1": metrics.loc["zigzag", "f1"],
                    "sudden_stop_f1": metrics.loc["sudden_stop", "f1"],
                }
            )
    runs = pd.DataFrame(rows)
    runs.to_csv(output_root / "ablation_runs.csv", index=False)
    if runs.empty:
        return
    summary = (
        runs.groupby("variant", as_index=False)
        .mean(numeric_only=True)
        .drop(columns=["seed"])
    )
    summary["eligible"] = (
        (summary["event_recall"] >= 0.85)
        & (summary["p90_latency_seconds"] <= 600)
        & (summary["false_alerts_per_day"] <= 2.0)
        & (summary["accident_window_fp"] <= 0.15)
        & (summary["congestion_spike_f1"] >= 0.20)
        & (summary["zigzag_f1"] >= 0.20)
        & (summary["sudden_stop_f1"] >= 0.65)
    )
    summary = summary.sort_values(
        ["eligible", "event_f1", "event_recall", "false_alerts_per_day"],
        ascending=[False, False, False, True],
    )
    summary.to_csv(output_root / "ablation_summary.csv", index=False)
    eligible = summary.loc[summary["eligible"]]
    selected = eligible.iloc[0].to_dict() if not eligible.empty else None
    payload = {
        "promote": selected is not None,
        "selected": selected,
        "best_candidate": summary.iloc[0].to_dict(),
        "fallback": None if selected else "best_checkpoint_seed_43",
    }
    (output_root / "selection.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(
        f"[done] promote={payload['promote']} "
        f"candidate={payload['best_candidate']['variant']}"
    )


def main() -> None:
    args = parse_args()
    base_config: dict[str, Any] = json.loads(
        args.config.read_text(encoding="utf-8")
    )
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)
    for variant, overrides in VARIANTS.items():
        for seed in seeds:
            output_dir = args.output_root / f"{variant}_seed_{seed}"
            if (output_dir / "test_metrics_by_class.csv").exists():
                print(f"[skip] {variant} seed={seed}")
                continue
            config = {**base_config, **overrides, "seed": seed}
            if args.smoke:
                config["head_epochs"] = 1
                config["fine_tune_epochs"] = 0
                config["early_stopping"] = {"patience": 1, "min_delta": 0.002}
            config_path = args.output_root / f"{variant}_seed_{seed}.config.json"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            log_path = args.output_root / f"{variant}_seed_{seed}.log"
            command = [
                sys.executable,
                "-m",
                "src.train.train_dmfgcrn_multilabel",
                "--config",
                str(config_path),
                "--dataset",
                str(args.dataset),
                "--output-dir",
                str(output_dir),
            ]
            if args.smoke:
                command.extend(
                    ["--max-train-samples", "256", "--max-eval-samples", "512"]
                )
            print(f"[run] {variant} seed={seed}")
            run(command, log_path)
    summarize(args.output_root, seeds)


if __name__ == "__main__":
    main()
