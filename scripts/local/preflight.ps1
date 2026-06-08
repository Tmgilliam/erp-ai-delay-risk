#Requires -Version 5.1
<#
.SYNOPSIS
    Check local prerequisites for ERP AI development and Entra/APIM readiness.
#>
param(
    [switch]$RequireEntra
)

$ErrorActionPreference = "Continue"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

function Test-Tool([string]$Name, [scriptblock]$Check) {
    try {
        $ok = & $Check
        if ($ok) {
            Write-Host "[OK]   $Name" -ForegroundColor Green
            return $true
        }
        Write-Host "[MISS] $Name" -ForegroundColor Yellow
        return $false
    }
    catch {
        Write-Host "[MISS] $Name" -ForegroundColor Yellow
        return $false
    }
}

Write-Host "ERP AI - Local Preflight" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"
Write-Host ""

$results = @{}
$results["Python"] = Test-Tool "Python" { (Get-Command python -ErrorAction Stop).Source }
$results["Docker"] = Test-Tool "Docker" { docker version 2>$null; $LASTEXITCODE -eq 0 }
$results["AzureCLI"] = Test-Tool "Azure CLI (az)" {
    $az = Get-Command az -ErrorAction SilentlyContinue
    if ($az) { az account show 2>$null; $LASTEXITCODE -eq 0 } else { $false }
}
$results["Model"] = Test-Tool "Model artifact" { Test-Path (Join-Path $ProjectRoot "models\delay_model.pkl") }
$results["EntraEnv"] = Test-Tool "Entra generated.env" { Test-Path (Join-Path $ProjectRoot "config\entra\generated.env") }

Write-Host ""
if (-not $results["AzureCLI"]) {
    Write-Host "Azure CLI not available. Options:" -ForegroundColor Yellow
    Write-Host "  1. Install: winget install Microsoft.AzureCLI"
    Write-Host "  2. Manual Entra setup: docs/entra-portal-setup.md"
    Write-Host "  3. Continue with AUTH_MODE=env (no Entra required)"
}

if ($RequireEntra -and -not $results["EntraEnv"]) {
    Write-Host ""
    Write-Host "Entra config missing. Run register-apps.ps1 or follow entra-portal-setup.md" -ForegroundColor Red
    exit 1
}

if ($results["Python"] -and $results["Model"]) {
    Write-Host "Ready for local dev (env auth)." -ForegroundColor Green
    exit 0
}

exit 1
