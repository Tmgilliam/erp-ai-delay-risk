#Requires -Version 5.1
<#
.SYNOPSIS
    Start local ERP AI stack with environment-variable auth (portfolio default).
#>
param(
    [int]$ApiPort = 8001,
    [int]$DashboardPort = 8501
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

& (Join-Path $PSScriptRoot "preflight.ps1") | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Preflight failed. Fix issues above." -ForegroundColor Red
    exit 1
}

$env:AUTH_MODE = "env"
$env:ENTRA_AUTH_ENABLED = "false"
$env:DASH_USER = if ($env:DASH_USER) { $env:DASH_USER } else { "demo@erp-ai.local" }
$env:DASH_PASS = if ($env:DASH_PASS) { $env:DASH_PASS } else { "demo1234" }
$env:DASH_ROLE = "EXEC"
$env:API_URL = "http://127.0.0.1:$ApiPort"

Write-Host "Starting API on port $ApiPort ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$ProjectRoot'; `$env:ENTRA_AUTH_ENABLED='false'; python -m uvicorn app.main:app --host 127.0.0.1 --port $ApiPort --reload"
)

Start-Sleep -Seconds 3

Write-Host "Starting Dashboard on port $DashboardPort ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$ProjectRoot'; `$env:AUTH_MODE='env'; `$env:DASH_USER='$($env:DASH_USER)'; `$env:DASH_PASS='$($env:DASH_PASS)'; `$env:API_URL='http://127.0.0.1:$ApiPort'; python -m streamlit run dashboard/app.py --server.port $DashboardPort"
)

Write-Host ""
Write-Host "API:       http://127.0.0.1:$ApiPort/docs" -ForegroundColor Green
Write-Host "Dashboard: http://127.0.0.1:$DashboardPort" -ForegroundColor Green
Write-Host "Login:     $($env:DASH_USER) / $($env:DASH_PASS)"
