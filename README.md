# NewAnomalyDetector

Traffic simulation with SUMO and detection of four anomaly types:

- `accident`: traffic accidents.
- `congestion_spike`: sudden increases in congestion.
- `zigzag`: vehicles moving in a zigzag pattern.
- `sudden_stop`: vehicles stopping abruptly.

The system has three main stages:

1. Run SUMO simulations and generate anomaly labels.
2. Convert SUMO detector data into node-level datasets and time windows.
3. Train, evaluate, and run inference using an LSTM Autoencoder or DMFGCRN.

Use the multi-label DMFGCRN workflow to detect all anomaly types with one model.
The LSTM workflow provides single-label baselines.

> Run all commands in this guide from the project root using PowerShell.
> Example project directory: `E:\Projects\NewAnomalyDetector`.

## 1. Project structure

```text
NewAnomalyDetector/
|-- configs/                 Training and dataset configurations
|-- datasets/
|   |-- raw/sumo_data/        Raw SUMO detector XML and labels
|   `-- processed/           Processed CSV/NPZ datasets
|-- DMFGCRN/                 DMFGCRN backbone source and checkpoint
|-- models/                  Checkpoints, metrics, plots, and inference outputs
|-- scripts/                 Automated pipelines and evaluation tools
|-- src/
|   |-- evaluation/          Inference, threshold calibration, and evaluation
|   |-- models/              Model architectures
|   |-- preprocessing/       Time-window construction
|   |-- process_rawdata/     XML/CSV conversion for training
|   |-- simulation/          SUMO simulation and anomaly injection
|   `-- train/               Model training
|-- sumo/                    Networks, routes, scenarios, and additional files
|-- main.py                  Single-label LSTM/DMFGCRN baseline entry point
|-- requirements.txt         Pinned Python dependencies
`-- SUMO-Simulate.xlsx        Traffic input workbook
```

## 2. System requirements

### 2.1 Software

- Windows 10/11 and PowerShell.
- Python 3.13 for the pinned dependencies. The project environment has been
  tested with Python 3.13.9.
- SUMO with `sumo.exe`, `sumo-gui.exe`, and the `tools` directory.
- Enough disk space for datasets and checkpoints in `datasets/` and `models/`.

The current DMFGCRN training configurations use the CPU by default. To set
`device` to `cuda`, install a PyTorch build compatible with your GPU and CUDA setup.

### 2.2 Check Python

```powershell
python --version
python -m pip --version
```

If multiple Python versions are installed, inspect them with Python Launcher:

```powershell
py -0p
```

Use a command such as `py -3.13` only if that version appears in `py -0p`.

### 2.3 Install and configure SUMO

After installing SUMO, point `SUMO_HOME` to its installation directory. For example:

```powershell
$env:SUMO_HOME = "E:\Projects\Sumo"
$env:Path = "$env:SUMO_HOME\bin;$env:Path"
$env:PYTHONPATH = "$env:SUMO_HOME\tools;$env:PYTHONPATH"
```

These three environment variables apply only to the current PowerShell window.
To save them for your Windows user account and future terminals:

```powershell
[Environment]::SetEnvironmentVariable(
  "SUMO_HOME",
  "E:\Projects\Sumo",
  "User"
)

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
[Environment]::SetEnvironmentVariable(
  "Path",
  "E:\Projects\Sumo\bin;$userPath",
  "User"
)

$userPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "User")
[Environment]::SetEnvironmentVariable(
  "PYTHONPATH",
  "E:\Projects\Sumo\tools;$userPythonPath",
  "User"
)
```

Close and reopen PowerShell, then check:

```powershell
sumo --version
sumo-gui --version
python -c "import traci; print(traci.__file__)"
```

`traci` is provided by SUMO's `tools` directory and is not included in
`requirements.txt`. If the SUMO commands work but Python cannot import `traci`,
check `SUMO_HOME` and `PYTHONPATH`.

## 3. Project setup

### 3.1 Open the project directory

```powershell
Set-Location E:\Projects\NewAnomalyDetector
```

### 3.2 Create a virtual environment

```powershell
python -m venv .venv
```

If Python Launcher lists Python 3.13, you can select it explicitly:

```powershell
py -3.13 -m venv .venv
```

### 3.3 Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell reports `running scripts is disabled`, change the execution policy
for the current process only, then activate the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

After activation, `python` should point to `.venv`:

```powershell
python -c "import sys; print(sys.executable)"
```

### 3.4 Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins the CPU build of PyTorch. Check the main libraries:

```powershell
python -c "import numpy, pandas, sklearn, torch; print(torch.__version__)"
python -c "import traci; print('TraCI: OK')"
```

Check the entry points without starting training:

```powershell
python main.py --help
python -m src.preprocessing.build_multilabel_dataset --help
python -m src.train.train_dmfgcrn_multilabel --help
python -m src.evaluation.infer_dmfgcrn_multilabel --help
```

## 4. Required data and checkpoints

The GitHub repository includes source code, configurations, SUMO networks/routes/
detectors, the input workbook, reference mapping tables, and an approximately
9 MB backbone checkpoint. Generated simulation datasets, new training checkpoints,
benchmark results, generated reports, and archives are excluded from Git.
Removing these files from Git tracking keeps the local copies on disk.

### 4.1 Use existing data

Pre-generated datasets and related images are available in the shared
[Google Drive folder](https://drive.google.com/drive/folders/1ylqXFEVje3tHbs622pYU-0lq_bVigWzw?usp=sharing).
Download the datasets you need and place them at the local paths specified in
the corresponding JSON configurations under `configs/`.

To run without repeating the SUMO simulation, check these files:

```powershell
Test-Path datasets\processed\multilabel\dmfgcrn_multilabel_windows_flat_7d_s02.npz
Test-Path DMFGCRN\experiments\SUMO\20260512041404\10\best_model.pth
```

After cloning, the backbone checkpoint should be available. Regenerate the dataset
as described in section 8 or download it from the Google Drive folder above.
Once data is prepared, both commands should return `True`:

- The `.npz` file contains train/validation/test windows.
- The `.pth` file is the DMFGCRN backbone checkpoint used by the multi-label model.

Dataset paths, source CSV files, nodes, features, and split ratios are defined in
JSON files under `configs/`. When changing datasets, use the configuration that
was used to build the selected dataset.

### 4.2 Regenerate data from scratch

You need:

- `SUMO-Simulate.xlsx`.
- Network, route, scenario, and additional files under `sumo/`.
- SUMO and TraCI configured as described in section 2.3.

Simulation can take several hours depending on the number of days and the
configuration. Pipelines generate large files, and some scripts refuse to
overwrite existing datasets. Choose a new output name or move older artifacts
before rerunning a pipeline.

## 5. Multi-label DMFGCRN quick start

This workflow predicts `accident`, `congestion_spike`, `zigzag`, and `sudden_stop`
together. A time window can have multiple labels.

### 5.1 Build an NPZ dataset from processed CSV files

Skip this step if the NPZ file already exists and does not need rebuilding:

```powershell
python -m src.preprocessing.build_multilabel_dataset `
  --config configs\dmfgcrn_multilabel_flat_7d_s02.json `
  --output datasets\processed\multilabel\dmfgcrn_multilabel_windows_flat_7d_s02.npz
```

For a smoke test, limit the number of windows per source:

```powershell
python -m src.preprocessing.build_multilabel_dataset `
  --config configs\dmfgcrn_multilabel_flat_7d_s02.json `
  --output datasets\processed\multilabel\dmfgcrn_multilabel_smoke.npz `
  --max-windows-per-source 100
```

### 5.2 Train

The `dmfgcrn_multilabel_flat_7d_s02.json` configuration above only defines dataset
construction. For training, use `dmfgcrn_multilabel.json`, which also specifies
the backbone checkpoint, hyperparameters, and threshold policy. Override its
dataset path with `--dataset`.

Full training:

```powershell
python -m src.train.train_dmfgcrn_multilabel `
  --config configs\dmfgcrn_multilabel.json `
  --dataset datasets\processed\multilabel\dmfgcrn_multilabel_windows_flat_7d_s02.npz `
  --output-dir models\outputs_dmfgcrn_multilabel_flat_7d_s02
```

Run a short smoke test to check the workflow before a longer training run:

```powershell
python -m src.train.train_dmfgcrn_multilabel `
  --config configs\dmfgcrn_multilabel.json `
  --dataset datasets\processed\multilabel\dmfgcrn_multilabel_smoke.npz `
  --output-dir models\outputs_dmfgcrn_multilabel_smoke `
  --head-epochs 1 `
  --fine-tune-epochs 1 `
  --max-train-samples 100 `
  --max-eval-samples 100
```

Important output artifacts:

- `dmfgcrn_multilabel.pt`: checkpoint for inference.
- `test_metrics_by_class.csv`: precision, recall, and F1 for each class.
- `test_summary.json`: test-set results summary.
- Threshold and prediction files generated according to the training configuration.

### 5.3 Run inference

```powershell
python -m src.evaluation.infer_dmfgcrn_multilabel `
  --checkpoint models\outputs_dmfgcrn_multilabel_flat_7d_s02\dmfgcrn_multilabel.pt `
  --dataset datasets\processed\multilabel\dmfgcrn_multilabel_windows_flat_7d_s02.npz `
  --split test `
  --output-csv models\outputs_dmfgcrn_multilabel_flat_7d_s02\test_predictions.csv
```

The output `status` column means:

- `known_anomaly`: at least one class exceeds its class-specific threshold.
- `unknown_anomaly`: no class exceeds its threshold, but the aggregate residual is high.
- `normal`: class probabilities and the residual are below their thresholds.

## 6. Two-stage DMFGCRN

The two-stage model combines an anomaly detector with a router that classifies
the anomaly type. The pipeline trains three seeds and evaluates an ensemble:

```powershell
python -m scripts.run_two_stage_training `
  --dataset datasets\processed\multilabel\dmfgcrn_multilabel_windows_multiscale_s02_s06_v2.npz `
  --config configs\dmfgcrn_two_stage_v2.json `
  --output-root models\outputs_dmfgcrn_two_stage_v2_s02_s06
```

Evaluate the ensemble:

```powershell
python -m scripts.evaluate_two_stage_ensemble `
  --dataset datasets\processed\multilabel\dmfgcrn_multilabel_windows_multiscale_s02_s06_v2.npz `
  --checkpoint-root models\outputs_dmfgcrn_two_stage_v2_s02_s06 `
  --output-dir models\outputs_dmfgcrn_two_stage_ensemble_v2_s02_s06
```

Use `--smoke` for a short training run:

```powershell
python -m scripts.run_two_stage_training `
  --dataset datasets\processed\multilabel\dmfgcrn_multilabel_windows_multiscale_s02_s06_v2.npz `
  --config configs\dmfgcrn_two_stage_v2.json `
  --output-root models\outputs_dmfgcrn_two_stage_smoke `
  --smoke
```

## 7. Single-label LSTM/DMFGCRN baselines

`main.py` runs either the LSTM Autoencoder baseline or the single-label DMFGCRN
backend, depending on `model_backend` in the configuration.

| Anomaly type | LSTM configuration |
|---|---|
| Accident | `configs/lstm_autoencoder_accident.json` |
| Congestion spike | `configs/lstm_autoencoder_congestion.json` |
| Zigzag | `configs/lstm_autoencoder_zigzag.json` |
| Sudden stop | `configs/lstm_autoencoder_sudden_stop_target075.json` |

Run one configuration:

```powershell
python main.py --config configs\lstm_autoencoder_accident.json
```

Run all four configurations in sequence:

```powershell
$configs = @(
  "configs\lstm_autoencoder_accident.json",
  "configs\lstm_autoencoder_congestion.json",
  "configs\lstm_autoencoder_zigzag.json",
  "configs\lstm_autoencoder_sudden_stop_target075.json"
)

foreach ($config in $configs) {
  Write-Host "Running $config"
  python main.py --config $config
  if ($LASTEXITCODE -ne 0) {
    throw "Pipeline failed for config: $config"
  }
}
```

Single-label DMFGCRN backend:

```powershell
python main.py --config configs\dmfgcrn_accident.json
```

Typical artifacts in the configuration's `output_dir`:

- `pilot_summary.json` and `pilot_summary.csv`.
- `event_eval_alert.csv`.
- `accident_window_result.csv`.
- A `*.pt` checkpoint.
- Score distribution, detection rate, and delay plots.

## 8. Regenerate data with SUMO

### 8.1 Inspect the simulation with the GUI

The included `mixed*` configurations already have matching scenarios and
detectors. To create a new variant, the generator uses a template from this
repository:

```powershell
python -m src.simulation.build_mixed_simulation_config --variant mixed_smoke --days 1 --force
```

`--force` overwrites an existing variant's configuration and scenario files;
use it only when you intend to regenerate them. To check SUMO startup without
running a full day, add `--max-seconds 600 --no-gui` to the runner command.
A 600-second dataset is not sufficient for training.

The legacy scenarios `named_roads_peak_hours.sumocfg` and
`named_roads_peak_hours_wide_coverage.sumocfg` are missing their route inputs.
They are excluded from GitHub and retained only locally. The old route
`named_roads.rou.xml` is also untracked because it is no longer referenced.
The `named_roads_updated.*` runner workflows in this guide do not depend on
those three files. The one-day preset `named_roads.sumocfg` is retained because
it has distinct parameters and uses the updated route. JSON presets and SUMO
variants for different weeks/seeds are retained for reproducibility.

Use the GUI to inspect the network, routes, and anomaly behavior:

```powershell
python -m src.simulation.xlsx_traci_flow_runner `
  --config src\simulation\configs\xlsx_traci_flow_config.mixed_smoke.json
```

Run without the GUI for faster processing:

```powershell
python -m src.simulation.xlsx_traci_flow_runner `
  --config src\simulation\configs\xlsx_traci_flow_config.mixed_smoke.json `
  --no-gui
```

Run the simulation with `python -m ...` because the module uses relative imports.

### 8.2 Seven-day mixed pipeline

The following script automatically:

1. Generates SUMO configurations and scenarios.
2. Runs the simulation.
3. Exports detector XML to time series.
4. Builds node-level forecast data.
5. Assigns node-level labels.
6. Packages the multi-label DMFGCRN dataset.

```powershell
.\scripts\run_mixed_flat_7d_pipeline.ps1 `
  -Variant mixed_flat_7d_s06 `
  -Days 7 `
  -TrafficScale 0.6
```

Useful options:

- `-SkipSimulation`: skip simulation if XML files and labels already exist.
- `-ForceConfig`: overwrite configurations/scenarios generated by the script.
- `-MaxWindowsPerSource 100`: limit data for a smoke test.

Logs are written to `.tmp\<variant>_run.log` and `.tmp\<variant>_pipeline.log`.

### 8.3 Run each step manually

Example using the `mixed_flat_7d_s02` variant:

```powershell
python -m src.simulation.xlsx_traci_flow_runner `
  --config src\simulation\configs\xlsx_traci_flow_config.mixed_flat_7d_s02.json `
  --no-gui
```

Export detector XML to time series:

```powershell
python -m src.process_rawdata.export_timeseries `
  --modes mixed_flat_7d_s02 `
  --detector-dir datasets\raw\sumo_data\detectors `
  --scenarios-dir datasets\raw\sumo_data `
  --processed-root datasets\processed `
  --summary-path datasets\processed\mixed_flat_7d_s02\timeseries_export_summary.json `
  --strict
```

Build the node-level forecast CSV:

```powershell
python -m src.process_rawdata.build_forecast_node_level_from_timeseries `
  --modes mixed_flat_7d_s02 `
  --processed-root datasets\processed `
  --suffix "" `
  --summary-path datasets\processed\mixed_flat_7d_s02\forecast_summary.json `
  --strict
```

Assign multiple anomaly labels to each node:

```powershell
python -m src.process_rawdata.build_mixed_node_labels `
  --forecast datasets\processed\mixed_flat_7d_s02\forecast_mixed_flat_7d_s02_node_level.csv `
  --window-labels datasets\raw\sumo_data\anomaly_window_labels_mixed_flat_7d_s02.csv `
  --sensor-mapping datasets\processed\references\sensor_node_mapping.csv `
  --output datasets\processed\mixed_flat_7d_s02\forecast_mixed_flat_7d_s02_node_level_with_labels.csv
```

Then build the NPZ dataset as described in section 5.1 with the matching configuration.

## 9. Interpret evaluation results

Main metrics:

- `precision`: correct alerts divided by all alerts.
- `recall`: the proportion of actual anomalies detected.
- `f1`: the harmonic mean of precision and recall.
- `normal_fp_rate`: the proportion of normal windows incorrectly flagged.
- `event_recall`: the proportion of anomaly events detected.
- `window_detect_ratio`: the proportion of anomalous windows detected.
- `mean_delay_seconds`: average detection delay.
- `false_alerts_per_hour`: number of false alerts per hour.

Do not evaluate the model using accuracy alone: anomaly data is imbalanced.
When comparing checkpoints, consider recall, F1, and the false-positive rate
together, using the same dataset and threshold calibration procedure.

## 10. Troubleshooting

### Cannot activate `.venv`

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Alternatively, call Python directly without activating the environment:

```powershell
.\.venv\Scripts\python.exe main.py --help
```

### `ModuleNotFoundError: No module named 'traci'`

```powershell
$env:SUMO_HOME = "E:\Projects\Sumo"
$env:PYTHONPATH = "$env:SUMO_HOME\tools;$env:PYTHONPATH"
python -c "import traci; print(traci.__file__)"
```

### `sumo` or `sumo-gui` not found

```powershell
$env:Path = "$env:SUMO_HOME\bin;$env:Path"
Get-Command sumo
Get-Command sumo-gui
```

### Relative import errors when running simulations

Correct:

```powershell
python -m src.simulation.xlsx_traci_flow_runner --config <config.json>
```

Incorrect:

```powershell
python src\simulation\xlsx_traci_flow_runner.py --config <config.json>
```

### Missing detector XML

Check the files in `datasets\raw\sumo_data\detectors`:

```powershell
Get-ChildItem datasets\raw\sumo_data\detectors\*.xml
```

If files are missing, check the `sumocfg`, additional files, detector output
paths, and simulation logs in `.tmp`.

### Missing label files

A mixed variant requires at least:

```text
anomaly_step_labels_<variant>.csv
anomaly_window_labels_<variant>.csv
anomaly_spawn_records_<variant>.json
```

These files must be in `datasets\raw\sumo_data` or the directory passed to
`--scenarios-dir`.

### DMFGCRN checkpoint not found

Check the checkpoint path in the configuration:

```powershell
Test-Path DMFGCRN\experiments\SUMO\20260512041404\10\best_model.pth
```

If using a different checkpoint, update the corresponding JSON field and keep
the architecture, node count, and input features consistent with that checkpoint.

### Out of memory while building datasets or training

- Use `--max-windows-per-source` when building NPZ files.
- Use `--max-train-samples` and `--max-eval-samples` for smoke tests.
- Reduce `batch_size` in the configuration.
- Avoid loading multiple large `.npz` datasets at the same time.

## 11. Rerunning experiments

- Do not overwrite checkpoints or results that are still in use; choose a new `output-dir`.
- Save the configuration alongside outputs for reproducibility.
- Keep the `seed`, train/validation/test split, and dataset fixed when comparing runs.
- Do not select thresholds directly on the test set.
- Check logs and exit codes before using artifacts in the next pipeline stage.

## 12. Checks before publishing to GitHub

```powershell
python -m pip check
python -m unittest discover -s tests -v
git status --short
git diff --cached --stat
```

`test_repository_inputs.py` checks inputs for every runner configuration,
network/route/additional references in all published scenarios, checkpoints
specified by training configurations, and reference mapping tables. Unit tests
do not replace testing the complete simulation → processing → training → inference workflow.

The two-stage/ensemble workflows require base checkpoints and trained seeds from
the corresponding scripts. Experiment outputs are not distributed through Git.
`.gitignore` does not remove artifacts from existing commits. Your local remote
may still point to GitLab; check `git remote -v` before configuring GitHub and pushing.

## 13. References and acknowledgments

This project's DMFGCRN backbone uses and adapts code from
[OvOYu/DMFGCRN](https://github.com/OvOYu/DMFGCRN). We thank the authors for sharing
their implementation, which provided a foundation for developing and evaluating
this traffic anomaly detection system.
