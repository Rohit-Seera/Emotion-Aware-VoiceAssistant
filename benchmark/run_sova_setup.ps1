param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host ""
Write-Host "=== SOVA-Bench evaluation setup (offline preflight) ===" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
    Write-Host "Project .venv Python not found; using python from PATH." -ForegroundColor Yellow
} else {
    Write-Host "Using project venv: $Python"
}

if (-not $SkipTests) {
    Write-Host "`n[1/4] Running benchmark unit tests..." -ForegroundColor Cyan
    & $Python -m pytest tests/benchmark -q --import-mode=importlib
    if ($LASTEXITCODE -ne 0) { throw "Benchmark tests failed. Fix failures before running model inference." }
} else {
    Write-Host "[1/4] Benchmark unit tests skipped by request." -ForegroundColor Yellow
}

Write-Host "`n[2/4] Listing registered model adapters..." -ForegroundColor Cyan
& $Python -m benchmark.run_sova_model --model all --list-models
if ($LASTEXITCODE -ne 0) { throw "Could not load benchmark model registry." }

Write-Host "`n[3/4] Checking packaged data and saved outputs..." -ForegroundColor Cyan
& $Python -m benchmark.check_sova_setup --output benchmark\audit\setup_check.json
if ($LASTEXITCODE -ne 0) { throw "SOVA setup audit failed." }

Write-Host "`n[4/4] Building reference Table-4-shaped CSVs..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path benchmark\runs | Out-Null
if (-not (Test-Path benchmark\runs\README.md)) {
    @"
# New SOVA model runs

Each adapter's active run is written to `<model_id>/results.json`.
Use `benchmark\run_sova_smoke.ps1` to run and score a five-case pilot first.
Do not copy historical pilot results here unless they were produced by the same model/protocol and include case-level outputs.
"@ | Set-Content -Encoding UTF8 benchmark\runs\README.md
}
& $Python -m benchmark.compare_sova_table4 --runs-root benchmark\runs --output benchmark\Table4_comparison.csv
if ($LASTEXITCODE -ne 0) { throw "Could not create Table4_comparison.csv." }
& $Python -m benchmark.compare_sova_reference --output benchmark\sova_reference_comparison.csv
if ($LASTEXITCODE -ne 0) { throw "Could not create sova_reference_comparison.csv." }

Write-Host ""
Write-Host "Setup checks complete." -ForegroundColor Green
Write-Host "Main comparison: benchmark\Table4_comparison.csv"
Write-Host "Reference crosswalk: benchmark\sova_reference_comparison.csv"
Write-Host "Data/setup audit: benchmark\audit\setup_check.json"
Write-Host "Historical results inventory: benchmark\audit\existing_results_inventory.csv"
Write-Host "Next: prepare/validate a manifest with real audio; see benchmark\RESULTS_AND_CALCULATIONS_GUIDE.md"
Write-Host "No model inference, training, or paid API call was performed by this setup script."
