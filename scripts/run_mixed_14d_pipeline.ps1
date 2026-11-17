param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $ProjectRoot

$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$logDir = Join-Path $ProjectRoot ".tmp"
$runLog = Join-Path $logDir "mixed_14d_run.log"
$pipelineLog = Join-Path $logDir "mixed_14d_pipeline.log"
$datasetOutput = Join-Path $ProjectRoot "datasets\processed\multilabel\dmfgcrn_multilabel_windows_with_mixed_14d.npz"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
if (Test-Path -LiteralPath $datasetOutput) {
    throw "Refusing to overwrite existing dataset: $datasetOutput"
}

"[$(Get-Date -Format o)] Starting mixed 14-day SUMO run." | Set-Content -LiteralPath $pipelineLog
& $python -m src.simulation.xlsx_traci_flow_runner `
    --config src/simulation/configs/xlsx_traci_flow_config.mixed.json `
    --no-gui *> $runLog
if ($LASTEXITCODE -ne 0) {
    throw "SUMO mixed run failed. See $runLog"
}

$okCount = (Select-String -Path $runLog -Pattern "\[anomaly\]\[ok\]").Count
$failCount = (Select-String -Path $runLog -Pattern "\[anomaly\]\[fail\]").Count
if ($okCount -ne 448 -or $failCount -ne 0) {
    throw "Unexpected anomaly result: ok=$okCount fail=$failCount"
}

"[$(Get-Date -Format o)] SUMO complete; exporting detector timeseries." | Add-Content -LiteralPath $pipelineLog
& $python -m src.process_rawdata.export_timeseries `
    --modes mixed `
    --detector-dir datasets/raw/sumo_data/detectors `
    --scenarios-dir datasets/raw/sumo_data `
    --processed-root datasets/processed `
    --summary-path datasets/processed/mixed/timeseries_export_summary_mixed_14d.json `
    --strict *>> $pipelineLog
if ($LASTEXITCODE -ne 0) {
    throw "Mixed timeseries export failed. See $pipelineLog"
}

"[$(Get-Date -Format o)] Building forecast node-level data." | Add-Content -LiteralPath $pipelineLog
& $python -m src.process_rawdata.build_forecast_node_level_from_timeseries `
    --modes mixed `
    --suffix _rebuild `
    --summary-path datasets/processed/mixed/forecast_node_level_rebuild_summary_mixed_14d.json `
    --strict *>> $pipelineLog
if ($LASTEXITCODE -ne 0) {
    throw "Mixed forecast rebuild failed. See $pipelineLog"
}

"[$(Get-Date -Format o)] Attaching per-node multi-label targets." | Add-Content -LiteralPath $pipelineLog
& $python -m src.process_rawdata.build_mixed_node_labels *>> $pipelineLog
if ($LASTEXITCODE -ne 0) {
    throw "Mixed node label build failed. See $pipelineLog"
}

"[$(Get-Date -Format o)] Building additive DMFGCRN multi-label NPZ." | Add-Content -LiteralPath $pipelineLog
& $python -m src.preprocessing.build_multilabel_dataset `
    --config configs/dmfgcrn_multilabel.json `
    --output datasets/processed/multilabel/dmfgcrn_multilabel_windows_with_mixed_14d.npz *>> $pipelineLog
if ($LASTEXITCODE -ne 0) {
    throw "Mixed NPZ build failed. See $pipelineLog"
}

"[$(Get-Date -Format o)] Mixed 14-day pipeline completed." | Add-Content -LiteralPath $pipelineLog
