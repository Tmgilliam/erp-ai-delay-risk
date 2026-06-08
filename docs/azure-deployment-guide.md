# Azure Deployment Guide — ERP AI Delay Risk

**Author:** Dr. Tatianna Gilliam  
**Prerequisites:** Azure CLI 2.53+, Docker, Azure subscription with Contributor access

---

## Overview

This guide deploys the ERP AI Delay Risk system to Azure Container Apps following the architecture in [azure-migration-architecture.md](azure-migration-architecture.md). Deployment is infrastructure substitution — no application re-architecture required.

**Estimated time:** 30–45 minutes  
**Estimated cost:** ~$15–30/month (without APIM)

---

## Architecture Deployed

| Azure Service | Purpose |
|---------------|---------|
| Container Apps (API) | FastAPI inference, Blob model loading, managed identity |
| Container Apps (Dashboard) | Streamlit executive UI |
| Azure Container Registry | Container images |
| Blob Storage | Versioned model artifact (`models/delay_model.pkl`) |
| Azure Files | Persisted scoring history volume mount |
| Key Vault | Dashboard credentials (RBAC-enabled) |
| Log Analytics | Container logs and monitoring foundation |
| Monitor Alerts | API replica count alert |
| APIM (optional) | API governance layer |

---

## Quick Deploy (PowerShell)

```powershell
cd C:\Users\teemm\MTP-Projects\erp-ai-delay-risk

# Optional: set a secure dashboard password
$env:DASHBOARD_PASSWORD = "YourSecurePassword123!"

# Deploy everything
.\scripts\azure\deploy.ps1 -ResourceGroup "rg-erp-ai-delay-risk" -Location "eastus2"

# With APIM (Developer tier — ~$50/month additional)
.\scripts\azure\deploy.ps1 -ResourceGroup "rg-erp-ai-delay-risk" -DeployApim
```

### What the Script Does

1. Creates resource group
2. Deploys `infra/azure/main.bicep`
3. Waits for RBAC propagation
4. Builds and pushes `erp-ai-api` and `erp-ai-dashboard` images to ACR
5. Configures ACR managed identity pull on Container Apps
6. Uploads `models/delay_model.pkl` to Blob Storage
7. Updates Container Apps with production images
8. Prints API and dashboard URLs

---

## Manual Deploy (Step by Step)

### 1. Infrastructure

```powershell
az group create --name rg-erp-ai-delay-risk --location eastus2

az deployment group create `
  --resource-group rg-erp-ai-delay-risk `
  --template-file infra/azure/main.bicep `
  --parameters environmentName=erpai location=eastus2 dashboardPassword='ChangeMe-Demo1234!'
```

Capture outputs:

```powershell
az deployment group show `
  --resource-group rg-erp-ai-delay-risk `
  --name main `
  --query properties.outputs
```

### 2. Build and Push Images

```powershell
$acrName = "<from outputs>"
$acrServer = "<from outputs>"

az acr login --name $acrName
docker build -f Dockerfile.azure.api -t "${acrServer}/erp-ai-api:latest" .
docker build -f Dockerfile.mtp.dashboard -t "${acrServer}/erp-ai-dashboard:latest" .
docker push "${acrServer}/erp-ai-api:latest"
docker push "${acrServer}/erp-ai-dashboard:latest"
```

### 3. Configure Registry and Update Apps

```powershell
$rg = "rg-erp-ai-delay-risk"
$apiApp = "<apiAppName from outputs>"
$dashApp = "<dashboardAppName from outputs>"

az containerapp registry set -g $rg -n $apiApp --server $acrServer --identity system
az containerapp registry set -g $rg -n $dashApp --server $acrServer --identity system

az containerapp update -g $rg -n $apiApp --image "${acrServer}/erp-ai-api:latest"
az containerapp update -g $rg -n $dashApp --image "${acrServer}/erp-ai-dashboard:latest"
```

### 4. Upload Model to Blob

```powershell
$storage = "<storageAccountName from outputs>"

az storage blob upload `
  --account-name $storage `
  --container-name models `
  --name delay_model.pkl `
  --file models/delay_model.pkl `
  --auth-mode login `
  --overwrite
```

Your account needs **Storage Blob Data Contributor** on the storage account for this step.

### 5. Validate

```powershell
$apiFqdn = az containerapp show -g $rg -n $apiApp --query "properties.configuration.ingress.fqdn" -o tsv

curl "https://$apiFqdn/health"
curl "https://$apiFqdn/monitoring/drift-report"
curl "https://$apiFqdn/monitoring/scoring-history"
```

Open dashboard URL in browser. Login: `demo@erp-ai.local` / password from deployment.

---

## Post-Deploy: Drift Metric Function (Optional)

Deploy `azure/functions/drift_metric/` as a timer-triggered Function App:

1. Create Function App (Python 3.11, consumption plan)
2. Set `ERP_AI_API_URL` = your API Container App URL
3. Enable Application Insights
4. Deploy function code
5. Create alert rule on custom log search: `DRIFT_METRIC` where `metric_value > 0`

---

## Post-Deploy: APIM Configuration (If Enabled)

After APIM provisions (~30–45 minutes):

1. Azure Portal → API Management → APIs → Add API → OpenAPI
2. Import from `https://<api-fqdn>/openapi.json`
3. Set backend URL to Container App FQDN
4. Add inbound policy: rate-limit (100 calls/minute)
5. Add OAuth 2.0 validation with Entra ID app registration (enterprise)

---

## Post-Deploy: Entra ID SSO (Enterprise)

1. Register app: **ERP AI Dashboard**
2. Redirect URI: `https://<dashboard-fqdn>/oauth2/callback` (if using auth proxy)
3. Assign users to EXEC / OPS / ANALYST groups
4. For portfolio demo, env-var auth (`DASH_USER`/`DASH_PASS`) remains active

---

## GCP → Azure Bridge Framing

This deployment proves the architecture is cloud-agnostic. Phase 1 on GCP Cloud Run demonstrated the ML system works. Phase 4 on Azure demonstrates the architect knows how to land it with enterprise governance — managed identity, Blob model versioning, monitoring, optional APIM.

When asked why GCP first: *"Deliberate choice. The containers and API contracts migrate with infrastructure substitution, not re-architecture."*

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Image pull failure | Run `az containerapp registry set --identity system` before image update |
| Model load failure | Verify Blob upload and Storage Blob Data Reader role on API managed identity |
| RBAC propagation delay | Wait 30–60 seconds after Bicep deploy before pushing images |
| Dashboard can't reach API | Confirm `API_URL` env var matches API FQDN with `https://` |
| Scoring history not persisting | Check Azure Files mount at `/app/monitoring` on API container |

---

## Rollback

```powershell
az group delete --name rg-erp-ai-delay-risk --yes --no-wait
```

GCP Cloud Run remains live until DNS cutover. No data loss on GCP side.

---

## Files Reference

| Path | Purpose |
|------|---------|
| `.azure/deployment-plan.md` | Deployment plan artifact |
| `infra/azure/main.bicep` | Root Bicep template |
| `scripts/azure/deploy.ps1` | One-command deployment |
| `Dockerfile.azure.api` | API image with Azure SDK |
| `app/azure_model_loader.py` | Blob model download via managed identity |
| `azure/functions/drift_metric/` | Optional drift metric Function |
