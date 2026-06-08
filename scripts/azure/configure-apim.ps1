#Requires -Version 5.1
<#
.SYNOPSIS
    Configure APIM for ERP AI — validate now, apply when subscription is active.

.PARAMETER ResourceGroup
    Resource group containing APIM instance.

.PARAMETER ApimName
    API Management service name.

.PARAMETER ApiBackendUrl
    Container Apps API URL (https://...).

.PARAMETER ValidateOnly
    Validate policy files and OpenAPI spec without touching Azure (default when APIM not deployed).

.PARAMETER Apply
    Apply policies and import API to live APIM instance (requires paid subscription).
#>
param(
    [string]$ResourceGroup = "",
    [string]$ApimName = "",
    [string]$ApiBackendUrl = "",
    [switch]$ValidateOnly,
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$PolicyDir = Join-Path $ProjectRoot "config\apim\policies"
$OpenApiPath = Join-Path $ProjectRoot "config\apim\openapi.json"
$EntraEnvPath = Join-Path $ProjectRoot "config\entra\generated.env"

function Read-EntraVar([string]$Name) {
    if (-not (Test-Path $EntraEnvPath)) { return "" }
    $line = Get-Content $EntraEnvPath | Where-Object { $_ -match "^$Name=" } | Select-Object -First 1
    if ($line) { return ($line -split "=", 2)[1] }
    return ""
}

function Substitute-Policy([string]$Content) {
    $tenantId = Read-EntraVar "ENTRA_TENANT_ID"
    $audience = Read-EntraVar "ENTRA_API_AUDIENCE"
    $gateway = if ($ApimName) { "https://$ApimName.azure-api.net" } else { "{{APIM_GATEWAY_URL}}" }
    $backend = if ($ApiBackendUrl) { $ApiBackendUrl.TrimEnd("/") } else { "{{API_BACKEND_URL}}" }

    return $Content `
        -replace '\{\{ENTRA_TENANT_ID\}\}', $tenantId `
        -replace '\{\{ENTRA_API_AUDIENCE\}\}', $audience `
        -replace '\{\{APIM_GATEWAY_URL\}\}', $gateway `
        -replace '\{\{API_BACKEND_URL\}\}', $backend
}

Write-Host "==> Validating APIM configuration artifacts" -ForegroundColor Cyan

$requiredFiles = @(
    "global-policy.xml",
    "api-inbound-policy.xml",
    "predict-operation-policy.xml"
)
foreach ($file in $requiredFiles) {
    $path = Join-Path $PolicyDir $file
    if (-not (Test-Path $path)) {
        throw "Missing policy file: $path"
    }
    [xml]$null = (Get-Content $path -Raw)
    Write-Host "  OK  $file"
}

if (-not (Test-Path $OpenApiPath)) {
    throw "Missing OpenAPI spec: $OpenApiPath"
}
$null = Get-Content $OpenApiPath -Raw | ConvertFrom-Json
Write-Host "  OK  openapi.json"

if ($ValidateOnly -or -not $Apply) {
    Write-Host ""
    Write-Host "Validation passed. APIM not modified (paid service not required)." -ForegroundColor Green
    Write-Host ""
    Write-Host "When subscription is active, run:"
    Write-Host '  .\scripts\azure\configure-apim.ps1 -Apply -ResourceGroup rg-erp-ai-delay-risk -ApimName <apim-name> -ApiBackendUrl https://<api-fqdn>'
    exit 0
}

if (-not $ResourceGroup -or -not $ApimName -or -not $ApiBackendUrl) {
    throw "Apply requires -ResourceGroup, -ApimName, and -ApiBackendUrl"
}

Write-Host "==> Applying APIM configuration to $ApimName" -ForegroundColor Cyan

# Import OpenAPI as new API revision
$resolvedOpenApi = Substitute-Policy (Get-Content $OpenApiPath -Raw)
$tempOpenApi = Join-Path $env:TEMP "erp-ai-openapi.json"
Set-Content -Path $tempOpenApi -Value $resolvedOpenApi

az apim api import `
    --resource-group $ResourceGroup `
    --service-name $ApimName `
    --path "v1" `
    --api-id "erp-delay-risk" `
    --specification-format OpenApiJson `
    --specification-path $tempOpenApi `
    --service-url $ApiBackendUrl | Out-Null

# Apply API policy
$apiPolicy = Substitute-Policy (Get-Content (Join-Path $PolicyDir "api-inbound-policy.xml") -Raw)
$tempPolicy = Join-Path $env:TEMP "erp-ai-api-policy.xml"
Set-Content -Path $tempPolicy -Value $apiPolicy

az apim api policy create `
    --resource-group $ResourceGroup `
    --service-name $ApimName `
    --api-id "erp-delay-risk" `
    --xml-file $tempPolicy | Out-Null

Write-Host "APIM configured. Gateway: https://$ApimName.azure-api.net/v1" -ForegroundColor Green
