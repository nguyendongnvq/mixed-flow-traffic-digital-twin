from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "models" / "outputs_dmfgcrn_two_stage_pipeline_v2"
STATUS_PATH = RUN_ROOT / "pipeline_status.json"
LOG_PATH = RUN_ROOT / "pipeline.log"
TWO_STAGE_ROOT = PROJECT_ROOT / "models" / "outputs_dmfgcrn_two_stage_v2"
ACCIDENT_ENSEMBLE = PROJECT_ROOT / (
    "models/outputs_dmfgcrn_accident_binary_ensemble_single_label_v3"
)
TWO_STAGE_ENSEMBLE = PROJECT_ROOT / "models" / "outputs_dmfgcrn_two_stage_ensemble_v2"


def write_status(stage: str, state: str, **extra: object) -> None:
    STATUS_PATH.write_text(
        json.dumps(
            {
                "updated_at": datetime.now().isoformat(),
                "stage": stage,
                "state": state,
                **extra,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_stage(stage: str, command: list[str]) -> None:
    write_status(stage, "running", command=command)
    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now().isoformat()}] stage={stage}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        write_status(stage, "failed", returncode=result.returncode)
        raise RuntimeError(f"Stage failed: {stage}; see {LOG_PATH}")
    write_status(stage, "completed")


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    dataset = (
        "datasets/processed/multilabel/"
        "dmfgcrn_multilabel_windows_multiscale_v2.npz"
    )
    try:
        run_stage(
            "train_two_stage",
            [
                python,
                "-m",
                "scripts.run_two_stage_training",
                "--dataset",
                dataset,
                "--output-root",
                str(TWO_STAGE_ROOT),
            ],
        )
        run_stage(
            "evaluate_two_stage_ensemble",
            [
                python,
                "-m",
                "scripts.evaluate_two_stage_ensemble",
                "--dataset",
                dataset,
                "--checkpoint-root",
                str(TWO_STAGE_ROOT),
                "--accident-manifest",
                str(ACCIDENT_ENSEMBLE / "ensemble_manifest.json"),
                "--output-dir",
                str(TWO_STAGE_ENSEMBLE),
            ],
        )
    except Exception as error:
        if not STATUS_PATH.exists():
            write_status("pipeline", "failed", error=str(error))
        raise
    write_status(
        "pipeline",
        "completed",
        two_stage_summary=str(TWO_STAGE_ENSEMBLE / "ensemble_summary.json"),
    )


if __name__ == "__main__":
    main()
