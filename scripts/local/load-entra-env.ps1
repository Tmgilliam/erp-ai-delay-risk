#Requires -Version 5.1
<#
.SYNOPSIS
    Load config/entra/generated.env into the current PowerShell session.
#>
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$EnvFile = Join-Path $ProjectRoot "config\entra\generated.env"

if (-not (Test-Path $EnvFile)) {
    Write-Host "Missing $EnvFile" -ForegroundColor Red
    Write-Host "Run: .\scripts\entra\register-apps.ps1"
    Write-Host "Or:  docs/entra-portal-setup.md (manual portal steps)"
    exit 1
}

Get-Content $EnvFile | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -notmatch '=') { return }
    $name, $value = $_ -split '=', 2
    Set-Item -Path "env:$name" -Value $value
    Write-Host "Set $name"
}

Write-Host "Entra environment loaded." -ForegroundColor Green
