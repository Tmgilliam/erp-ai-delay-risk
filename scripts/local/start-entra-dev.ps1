#Requires -Version 5.1
<#
.SYNOPSIS
    Start local ERP AI stack with Entra ID auth (requires generated.env).
#>
param(
    [int]$ApiPort = 8001,
    [int]$DashboardPort = 8501
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

. (Join-Path $PSScriptRoot "load-entra-env.ps1")
if ($LASTEXITCODE -ne 0) { exit 1 }

$env:ENTRA_AUTH_ENABLED = "true"
$env:AUTH_MODE = "entra"
$env:API_URL = "http://127.0.0.1:$ApiPort"

Write-Host "Starting API with Entra JWT enforcement on port $ApiPort ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$ProjectRoot'; . .\scripts\local\load-entra-env.ps1; `$env:ENTRA_AUTH_ENABLED='true'; python -m uvicorn app.main:app --host 127.0.0.1 --port $ApiPort --reload"
)

Start-Sleep -Seconds 3

Write-Host "Starting Dashboard with Entra device code login on port $DashboardPort ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$ProjectRoot'; . .\scripts\local\load-entra-env.ps1; `$env:AUTH_MODE='entra'; `$env:API_URL='http://127.0.0.1:$ApiPort'; python -m streamlit run dashboard/app.py --server.port $DashboardPort"
)

Write-Host ""
Write-Host "API:       http://127.0.0.1:$ApiPort/docs (Bearer token required)" -ForegroundColor Green
Write-Host "Dashboard: http://127.0.0.1:$DashboardPort (Sign in with Microsoft)" -ForegroundColor Green
