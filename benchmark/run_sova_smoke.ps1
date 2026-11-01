param(
    [Parameter(Mandatory = $true)]
    [string]$Model,

    [Parameter(Mandatory = $true)]
    [string]$Manifest,

    [ValidateRange(1, 10000000)]
    [int]$Limit = 5,

    [switch]$AllCases,

    [switch]$Judge
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

if (-not (Test-Path $Manifest)) {
    # Also allow the manifest path to be project-root-relative when called from another directory.
    $candidate = Join-Path $ProjectRoot $Manifest
    if (Test-Path $candidate) { $Manifest = $candidate }
}
if (-not (Test-Path $Manifest)) { throw "Manifest not found: $Manifest" }

Write-Host "Validating audio manifest before inference..." -ForegroundColor Cyan
& $Python -m benchmark.check_sova_setup --manifest $Manifest --strict-manifest --output benchmark\audit\latest_manifest_check.json
if ($LASTEXITCODE -ne 0) {
    throw "Manifest check failed. Fix missing audio_path values before model inference."
}

$RunDir = Join-Path $ProjectRoot ("benchmark\runs\" + $Model)
$RawResults = Join-Path $RunDir "results.json"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

if (Test-Path $RawResults) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    Copy-Item $RawResults (Join-Path $RunDir ("results_backup_" + $Stamp + ".json"))
    Write-Host "Previous raw results backed up; new run will replace active results.json." -ForegroundColor Yellow
}

$RunArgs = @("-m", "benchmark.run_sova_model", "--model", $Model, "--input", $Manifest, "--output-dir", "benchmark\runs")
if (-not $AllCases) { $RunArgs += @("--limit", "$Limit") }

Write-Host "`nRunning model '$Model'..." -ForegroundColor Cyan
& $Python @RunArgs
if ($LASTEXITCODE -ne 0) { throw "Model runner exited with code $LASTEXITCODE. Check runtime/server requirements." }
if (-not (Test-Path $RawResults)) { throw "Runner finished but did not create $RawResults" }

if ($Judge) {
    Write-Host "`nRunning declared GPT-OSS judge step..." -ForegroundColor Cyan
    Write-Host "This step uses your configured judge API and may consume rate limits." -ForegroundColor Yellow
    & $Python -m benchmark.judge_sova_outputs --input $RawResults --output $RawResults
    if ($LASTEXITCODE -ne 0) { throw "Judge step failed. Raw result backup was preserved." }
}

Write-Host "`nScoring per-case outputs..." -ForegroundColor Cyan
& $Python -m benchmark.score_sova_results --input $RawResults --output (Join-Path $RunDir "scored")
if ($LASTEXITCODE -ne 0) { throw "Metric scoring failed." }

Write-Host "`nRebuilding SOVA Table-4-shaped comparison..." -ForegroundColor Cyan
& $Python -m benchmark.compare_sova_table4 --runs-root benchmark\runs --output benchmark\Table4_comparison.csv
if ($LASTEXITCODE -ne 0) { throw "Table 4 export failed." }

Write-Host ""
Write-Host "Run finished. Review:" -ForegroundColor Green
Write-Host "  Raw case outputs: $RawResults"
Write-Host "  Scored summary:   $(Join-Path $RunDir 'scored\summary.json')"
Write-Host "  Task CSV:         $(Join-Path $RunDir 'scored\task_summary.csv')"
Write-Host "  Table 4 CSV:      benchmark\Table4_comparison.csv"
Write-Host "Reminder: this is a subset if -AllCases was not supplied; report n_cases with every metric."
