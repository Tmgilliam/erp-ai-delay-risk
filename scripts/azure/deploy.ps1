#Requires -Version 5.1
<#
.SYNOPSIS
    Deploy ERP AI Delay Risk to Azure Container Apps.

.DESCRIPTION
    1. Creates resource group (if missing)
    2. Deploys Bicep infrastructure
    3. Builds and pushes container images to ACR
    4. Uploads model artifact to Blob Storage
    5. Updates Container Apps with real images

.PARAMETER ResourceGroup
    Azure resource group name.

.PARAMETER Location
    Azure region (default: eastus2).

.PARAMETER DeployApim
    Enable Azure API Management (Developer tier — additional cost).
#>
param(
    [string]$ResourceGroup = "rg-erp-ai-delay-risk",
    [string]$Location = "eastus2",
    [switch]$DeployApim
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$InfraPath = Join-Path $ProjectRoot "infra\azure\main.bicep"

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

Write-Step "Checking Azure CLI login"
az account show | Out-Null

Write-Step "Creating resource group: $ResourceGroup"
az group create --name $ResourceGroup --location $Location | Out-Null

$dashboardPassword = if ($env:DASHBOARD_PASSWORD) { $env:DASHBOARD_PASSWORD } else { "ChangeMe-Demo1234!" }

Write-Step "Deploying Bicep infrastructure"
$deployment = az deployment group create `
    --resource-group $ResourceGroup `
    --template-file $InfraPath `
    --parameters environmentName=erpai location=$Location deployApim=$DeployApim dashboardPassword=$dashboardPassword `
    --query "properties.outputs" -o json | ConvertFrom-Json

$acrName = $deployment.acrName.value
$acrServer = $deployment.acrLoginServer.value
$storageAccount = $deployment.storageAccountName.value
$apiAppFull = $deployment.apiAppName.value
$dashAppFull = $deployment.dashboardAppName.value

Write-Step "Waiting 30s for RBAC role assignments to propagate"
Start-Sleep -Seconds 30

Write-Step "Building and pushing API image"
az acr login --name $acrName
docker build -f (Join-Path $ProjectRoot "Dockerfile.azure.api") -t "${acrServer}/erp-ai-api:latest" $ProjectRoot
docker push "${acrServer}/erp-ai-api:latest"

Write-Step "Building and pushing Dashboard image"
docker build -f (Join-Path $ProjectRoot "Dockerfile.mtp.dashboard") -t "${acrServer}/erp-ai-dashboard:latest" $ProjectRoot
docker push "${acrServer}/erp-ai-dashboard:latest"

Write-Step "Configuring ACR pull for Container Apps"
az containerapp registry set -g $ResourceGroup -n $apiAppFull --server $acrServer --identity system
az containerapp registry set -g $ResourceGroup -n $dashAppFull --server $acrServer --identity system

Write-Step "Uploading model to Blob Storage"
az storage blob upload `
    --account-name $storageAccount `
    --container-name models `
    --name delay_model.pkl `
    --file (Join-Path $ProjectRoot "models\delay_model.pkl") `
    --auth-mode login `
    --overwrite

Write-Step "Updating Container Apps with production images"
az containerapp update -g $ResourceGroup -n $apiAppFull --image "${acrServer}/erp-ai-api:latest"
az containerapp update -g $ResourceGroup -n $dashAppFull --image "${acrServer}/erp-ai-dashboard:latest"

$apiUrl = az containerapp show -g $ResourceGroup -n $apiAppFull --query "properties.configuration.ingress.fqdn" -o tsv
$dashUrl = az containerapp show -g $ResourceGroup -n $dashAppFull --query "properties.configuration.ingress.fqdn" -o tsv

Write-Step "Deployment complete"
Write-Host "API URL:       https://$apiUrl" -ForegroundColor Green
Write-Host "Dashboard URL: https://$dashUrl" -ForegroundColor Green
Write-Host "Health check:  https://$apiUrl/health"
Write-Host "API docs:      https://$apiUrl/docs"
Write-Host ""
Write-Host "Dashboard login: demo@erp-ai.local / (password from DASHBOARD_PASSWORD or default)"
Write-Host "GCP Cloud Run remains live until you validate and cut over DNS."
