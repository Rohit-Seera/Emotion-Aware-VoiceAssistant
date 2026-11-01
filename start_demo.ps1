param(
  [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "=============================================" -ForegroundColor Magenta
Write-Host "        Emotion Aware Voice Assistant - AIKO DEMO LAUNCHER       " -ForegroundColor Magenta
Write-Host "=============================================" -ForegroundColor Magenta
Write-Host ""

if (!(Test-Path ".env")) {
    Write-Host "[!] .env not found. Copying .env.example -> .env" -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
}

# The current JWT implementation intentionally allowlists HS256.
$env:SECURITY_JWT_ALGORITHM = "HS256"

# If the local env still contains the placeholder, provide a per-process secret.
if (-not $env:SECURITY_JWT_SECRET_KEY -or $env:SECURITY_JWT_SECRET_KEY.Length -lt 64) {
    $bytes = New-Object byte[] 48
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $env:SECURITY_JWT_SECRET_KEY = [Convert]::ToBase64String($bytes)
}

$env:APP_ENVIRONMENT = "development"
$env:APP_DEBUG = "true"

Write-Host "[1/3] Starting FastAPI..." -ForegroundColor Cyan
$api = Start-Process powershell `
    -ArgumentList "-NoExit", "-Command", "uv run uvicorn app.main:app --host 127.0.0.1 --port 8000" `
    -PassThru

Start-Sleep -Seconds 3

Write-Host "[2/3] Checking API docs endpoint..." -ForegroundColor Cyan
try {
    $health = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8000/docs"
    Write-Host "      API docs: $($health.StatusCode)" -ForegroundColor Green
} catch {
    Write-Host "      API health check failed. Open the backend terminal for the traceback." -ForegroundColor Yellow
}

Write-Host "[3/3] Opening Aiko demo..." -ForegroundColor Cyan
if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:8000/api/v1/demo"
}

Write-Host ""
Write-Host "Demo URL: http://127.0.0.1:8000/api/v1/demo" -ForegroundColor Green
Write-Host "Keep the API terminal open during the presentation." -ForegroundColor Gray
Write-Host ""
