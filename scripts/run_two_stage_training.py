from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train three two-stage DMFGCRN models.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "datasets/processed/multilabel/"
            "dmfgcrn_multilabel_windows_with_mixed_14d.npz"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dmfgcrn_two_stage_v2.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("models/outputs_dmfgcrn_two_stage_v2"),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--include-sources",
        type=str,
        help="Comma-separated source names or ids passed to each seed training run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    base_config = json.loads(args.config.read_text(encoding="utf-8"))
    for seed in (42, 43, 44):
        output_dir = args.output_root / f"seed_{seed}"
        if (output_dir / "dmfgcrn_two_stage.pt").exists():
            print(f"[skip] seed={seed}")
            continue
        config = {
            **base_config,
            "seed": seed,
            "head_epochs": 1 if args.smoke else int(base_config["head_epochs"]),
            "fine_tune_epochs": 0 if args.smoke else int(base_config["fine_tune_epochs"]),
        }
        config_path = args.output_root / f"seed_{seed}.config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        base_checkpoint = Path(
            "models/outputs_dmfgcrn_accident_event_ablation/"
            f"source_weighted_hard_normal_seed_{seed}/dmfgcrn_multilabel.pt"
        )
        command = [
            sys.executable,
            "-m",
            "src.train.train_dmfgcrn_two_stage",
            "--config",
            str(config_path),
            "--dataset",
            str(args.dataset),
            "--base-checkpoint",
            str(base_checkpoint),
            "--output-dir",
            str(output_dir),
        ]
        if args.smoke:
            command.extend(["--max-train-samples", "512", "--max-eval-samples", "512"])
        if args.include_sources:
            command.extend(["--include-sources", args.include_sources])
        log_path = args.output_root / f"seed_{seed}.log"
        print(f"[run] seed={seed}")
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if process.returncode:
            raise RuntimeError(f"Training failed for seed={seed}; see {log_path}")


if __name__ == "__main__":
    main()
