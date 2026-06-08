# Phase 4 Changelog — Azure Deployment

**Project:** ERP AI — Delay Risk Prediction System  
**Owner:** Dr. Tatianna Gilliam  
**Date:** June 2026

## Overview

Phase 4 delivers deployable Azure infrastructure and application changes for the CAF-aligned migration documented in Phase 2. GCP Cloud Run remains the production deployment until validation and DNS cutover.

## Added

### Infrastructure (`infra/azure/`)

- `main.bicep` — root template orchestrating full Azure stack
- Modules: Log Analytics, ACR, Storage, Key Vault, Container Apps Environment, Container Apps, RBAC, Monitor alerts
- Optional APIM module (`deployApim=true`)
- `main.bicepparam` — default parameters

### Azure Application Support

- `app/azure_model_loader.py` — downloads model from Blob Storage via managed identity
- `requirements-azure.txt` — `azure-identity`, `azure-storage-blob`
- `Dockerfile.azure.api` — API image with Azure SDK dependencies

### Deployment Automation

- `.azure/deployment-plan.md` — deployment plan artifact
- `scripts/azure/deploy.ps1` — one-command deploy script
- `docs/azure-deployment-guide.md` — step-by-step runbook

### Optional Monitoring Function

- `azure/functions/drift_metric/` — timer-triggered drift metric emitter for Azure Monitor

## Changed

- `app/model.py` — uses `resolve_model_path()` for local or Blob model loading
- `README.md` — Phase 4 section and Azure deploy instructions

## Azure Resources Provisioned

| Resource | SKU / Tier |
|----------|-----------|
| Container Apps Environment | Consumption |
| Container Apps (API + Dashboard) | Scale 0→N |
| ACR | Basic |
| Storage Account | Standard LRS |
| Key Vault | Standard |
| Log Analytics | PerGB2018 |
| Monitor Alert | Metric alert |
| APIM (optional) | Developer |

## Unchanged

- GCP Cloud Run deployment (`src/`, `docker-compose.yml`)
- Local MTP stack (`docker-compose.mtp.yml`)
- Portfolio documentation

## Deploy Command

```powershell
.\scripts\azure\deploy.ps1 -ResourceGroup "rg-erp-ai-delay-risk" -Location "eastus2"
```

## Next Steps

Phase 5 completed — see [phase5-changelog.md](phase5-changelog.md) and [entra-apim-readiness.md](entra-apim-readiness.md).

Remaining (when subscription funded):

- Execute `deploy.ps1` and validate endpoints
- `configure-apim.ps1 -Apply` after APIM provisions
- Deploy drift metric Function App
- DNS cutover from GCP after validation window
