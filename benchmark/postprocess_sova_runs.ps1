[CmdletBinding()]
param(
    [string]$RunsRoot = "benchmark\runs",
    [string]$TableOutput = "benchmark\Table4_comparison.csv",
    [switch]$SkipJudge,
    [switch]$SkipConsistency,
    [switch]$SkipAcoustic
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $RunsRoot -PathType Container)) {
    throw "Runs root was not found: $RunsRoot. Run benchmark.run_sova_model first."
}

$python = (Get-Command python -ErrorAction Stop).Source
$modelDirs = Get-ChildItem -LiteralPath $RunsRoot -Directory
if (-not $modelDirs) {
    throw "No model run directories found under $RunsRoot."
}

foreach ($modelDir in $modelDirs) {
    $results = Join-Path $modelDir.FullName "results.json"
    if (-not (Test-Path -LiteralPath $results -PathType Leaf)) { continue }
    Write-Host "`n=== Post-processing $($modelDir.Name) ===" -ForegroundColor Cyan

    if (-not $SkipJudge) {
        if ($env:GROQ_API_KEY) {
            & $python -m benchmark.judge_sova_outputs --input $results
            if ($LASTEXITCODE -ne 0) { Write-Warning "GPT judgement failed/skipped for $($modelDir.Name) (exit $LASTEXITCODE)." }
        } else {
            Write-Warning "GROQ_API_KEY is not set; skipping GPT-OSS 120B evaluation for $($modelDir.Name)."
        }
    }

    if (-not $SkipConsistency) {
        & $python -m benchmark.transcribe_sova_outputs_local --input $results --model large-v3 --device cpu --compute-type int8
        if ($LASTEXITCODE -ne 0) { Write-Warning "Local Whisper-large-v3 transcription failed/skipped for $($modelDir.Name) (exit $LASTEXITCODE)." }
    }

    if (-not $SkipAcoustic) {
        $acousticOut = Join-Path $modelDir.FullName "acoustic"
        & $python -m benchmark.acoustic_eval --audio-dir $modelDir.FullName --input $results --generation-only --output $acousticOut
        if ($LASTEXITCODE -eq 0) {
            & $python -m benchmark.attach_sova_metrics --results $results --acoustic-results (Join-Path $acousticOut "acoustic_results.json")
            if ($LASTEXITCODE -ne 0) { Write-Warning "Acoustic metric attachment failed for $($modelDir.Name)." }
        } else {
            Write-Warning "UTMOSv2 could not score generation audio for $($modelDir.Name) (exit $LASTEXITCODE)."
        }
    }
}

& $python -m benchmark.compare_sova_table4 --runs-root $RunsRoot --output $TableOutput
if ($LASTEXITCODE -ne 0) { throw "Table-4 comparison export failed (exit $LASTEXITCODE)." }
Write-Host "`nFinished. Review measured rows, n_cases, metric_method_notes, and blanks in $TableOutput before reporting results."
