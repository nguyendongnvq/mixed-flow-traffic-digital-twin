param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$Variant = "mixed_flat_7d_s06",
    [int]$Days = 7,
    [double]$TrafficScale = 0.6,
    [switch]$ForceConfig,
    [switch]$SkipSimulation,
    [int]$MaxWindowsPerSource
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $ProjectRoot

$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$logDir = Join-Path $ProjectRoot ".tmp"
$runLog = Join-Path $logDir "${Variant}_run.log"
$pipelineLog = Join-Path $logDir "${Variant}_pipeline.log"
$datasetOutput = Join-Path $ProjectRoot "datasets\processed\multilabel\dmfgcrn_multilabel_windows_flat_7d_s06.npz"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
if (Test-Path -LiteralPath $datasetOutput) {
    throw "Refusing to overwrite existing dataset: $datasetOutput"
}

"[$(Get-Date -Format o)] Building $Variant config." | Set-Content -LiteralPath $pipelineLog
$configArgs = @(
    "-m", "src.simulation.build_mixed_simulation_config",
    "--variant", $Variant,
    "--days", "$Days",
    "--traffic-scale", "$TrafficScale",
    "--flow-mode", "constant_scaled"
)
if ($ForceConfig) {
    $configArgs += "--force"
}
& $python @configArgs *>> $pipelineLog
if ($LASTEXITCODE -ne 0) {
    throw "Config generation failed. See $pipelineLog"
}

if (-not $SkipSimulation) {
    "[$(Get-Date -Format o)] Starting SUMO run for $Variant." | Add-Content -LiteralPath $pipelineLog
    & $python -m src.simulation.xlsx_traci_flow_runner `
        --config "src/simulation/configs/xlsx_traci_flow_config.$Variant.json" `
        --no-gui *> $runLog
    if ($LASTEXITCODE -ne 0) {
        throw "SUMO run failed. See $runLog"
    }

    $failCount = (Select-String -Path $runLog -Pattern "\[anomaly\]\[fail\]" -ErrorAction SilentlyContinue).Count
    if ($failCount -ne 0) {
        throw "Unexpected anomaly failures: fail=$failCount. See $runLog"
    }
}

$processedDir = "datasets/processed/$Variant"
"[$(Get-Date -Format o)] Exporting detector timeseries." | Add-Content -LiteralPath $pipelineLog
& $python -m src.process_rawdata.export_timeseries `
    --modes $Variant `
    --detector-dir datasets/raw/sumo_data/detectors `
    --scenarios-dir datasets/raw/sumo_data `
    --processed-root datasets/processed `
    --summary-path "$processedDir/timeseries_export_summary.json" `
    --strict *>> $pipelineLog
if ($LASTEXITCODE -ne 0) {
    throw "Timeseries export failed. See $pipelineLog"
}

"[$(Get-Date -Format o)] Building forecast node-level data." | Add-Content -LiteralPath $pipelineLog
& $python -m src.process_rawdata.build_forecast_node_level_from_timeseries `
    --modes $Variant `
    --suffix "" `
    --summary-path "$processedDir/forecast_summary.json" `
    --strict *>> $pipelineLog
if ($LASTEXITCODE -ne 0) {
    throw "Forecast rebuild failed. See $pipelineLog"
}

"[$(Get-Date -Format o)] Attaching per-node multi-label targets." | Add-Content -LiteralPath $pipelineLog
& $python -m src.process_rawdata.build_mixed_node_labels `
    --forecast "$processedDir/forecast_${Variant}_node_level.csv" `
    --window-labels "datasets/raw/sumo_data/anomaly_window_labels_$Variant.csv" `
    --output "$processedDir/forecast_${Variant}_node_level_with_labels.csv" *>> $pipelineLog
if ($LASTEXITCODE -ne 0) {
    throw "Node label build failed. See $pipelineLog"
}

"[$(Get-Date -Format o)] Building DMFGCRN multi-label NPZ." | Add-Content -LiteralPath $pipelineLog
$datasetArgs = @(
    "-m", "src.preprocessing.build_multilabel_dataset",
    "--config", "configs/dmfgcrn_multilabel_flat_7d_s06.json",
    "--output", "datasets/processed/multilabel/dmfgcrn_multilabel_windows_flat_7d_s06.npz"
)
if ($MaxWindowsPerSource -gt 0) {
    $datasetArgs += @("--max-windows-per-source", "$MaxWindowsPerSource")
}
& $python @datasetArgs *>> $pipelineLog
if ($LASTEXITCODE -ne 0) {
    throw "NPZ build failed. See $pipelineLog"
}

"[$(Get-Date -Format o)] $Variant pipeline completed." | Add-Content -LiteralPath $pipelineLog
Write-Host "dataset=$datasetOutput"
Write-Host "pipeline_log=$pipelineLog"
if (-not $SkipSimulation) {
    Write-Host "run_log=$runLog"
}
