# Azure Deployment Plan — ERP AI Delay Risk

**Owner:** Dr. Tatianna Gilliam  
**Status:** Ready for deployment  
**Target:** Azure Container Apps (rehost from GCP Cloud Run)

---

## Workload Summary

| Component | Source | Azure Target |
|-----------|--------|--------------|
| FastAPI API | `Dockerfile.mtp.api` | Container App (external ingress, port 8000) |
| Streamlit Dashboard | `Dockerfile.mtp.dashboard` | Container App (external ingress, port 8501) |
| Model artifact | `models/delay_model.pkl` | Blob Storage `models/delay_model.pkl` |
| Scoring history | `monitoring/scoring_history.csv` | Azure Files volume mount (API) |
| Drift baseline | `monitoring/reference_baseline.csv` | Baked in image |

## Architecture Decisions

1. **Container Apps** over AKS — scale-to-zero, consumption plan, portfolio-scale workload
2. **Bicep IaC** — `infra/azure/` modular templates, no azd dependency required
3. **Managed identity** — AcrPull, Storage Blob Data Reader, Key Vault Secrets User
4. **APIM optional** — `deployApim=false` by default (cost control); enable for enterprise demo
5. **Entra ID** — documented manual steps; env-var auth retained for portfolio dashboard
6. **Two-phase ACR** — placeholder image in Bicep; real images pushed post-provision

## Deployment Phases

### Phase A — Infrastructure (Bicep)
- [x] Resource group scoped deployment
- [x] Log Analytics workspace
- [x] Azure Container Registry (Basic)
- [x] Storage account (models + scoring history file share)
- [x] Key Vault (dashboard credentials)
- [x] Container Apps Environment
- [x] Container Apps (API + Dashboard)
- [x] RBAC role assignments (ACR pull, Storage, Key Vault)
- [x] Monitor alert rules (API health, drift detected)
- [ ] APIM (optional, `deployApim=true`)

### Phase B — Application
- [x] Azure Blob model loader (`app/azure_model_loader.py`)
- [x] `requirements-azure.txt` for Azure SDK dependencies
- [x] `Dockerfile.azure.api` with Azure deps

### Phase C — Deploy Scripts
- [x] `scripts/azure/deploy.ps1` — full deployment orchestration
- [x] `scripts/azure/build-push.ps1` — ACR image build and push
- [x] `docs/azure-deployment-guide.md` — step-by-step runbook

### Phase D — Entra + APIM Readiness (FREE — no subscription)
- [x] Entra app registration script (`scripts/entra/register-apps.ps1`)
- [x] API JWT validation module (`app/auth/entra_jwt.py`, disabled by default)
- [x] Dashboard auth modes (`dashboard/auth.py`: env / entra / dual)
- [x] APIM policy artifacts (`config/apim/`)
- [x] APIM configure script with `-ValidateOnly` mode
- [x] Readiness guide (`docs/entra-apim-readiness.md`)

### Phase E — Paid Enablement (When Subscription Active)
- [ ] Run `deploy.ps1` for Container Apps stack
- [ ] Run `register-apps.ps1` (if not done) + assign app roles
- [ ] Provision APIM (`deployApim=true`) + `configure-apim.ps1 -Apply`
- [ ] Set `ENTRA_AUTH_ENABLED=true` and `AUTH_MODE=entra` on Container Apps
- [ ] Azure Function for drift → custom metric (optional)
- [ ] DNS cutover from GCP Cloud Run

## Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `environmentName` | `erpai` | Resource naming prefix |
| `location` | `eastus2` | Azure region |
| `deployApim` | `false` | Set true for APIM Developer tier |
| `dashboardUser` | `demo@erp-ai.local` | Stored in Key Vault |
| `dashboardPassword` | generated | Stored in Key Vault |

## Validation Checklist

- [ ] `az deployment group create` succeeds
- [ ] API `/health` returns 200
- [ ] `POST /predict` returns scoring result
- [ ] `GET /monitoring/drift-report` returns JSON
- [ ] `GET /monitoring/scoring-history` returns trend
- [ ] Dashboard loads Executive Risk Summary
- [ ] Model loaded from Blob (check API logs for blob download)

## Cost Estimate (Portfolio / Dev)

| Service | Tier | Est. Monthly |
|---------|------|-------------|
| Container Apps | Consumption, scale-to-zero | $5–20 |
| ACR | Basic | ~$5 |
| Storage | Standard LRS | ~$1 |
| Log Analytics | Pay-per-GB | ~$2–5 |
| Key Vault | Standard | ~$1 |
| APIM (optional) | Developer | ~$50 |

**Total without APIM:** ~$15–30/month  
**Total with APIM:** ~$65–80/month

## Rollback

- GCP Cloud Run remains live during validation window
- Azure deployment is additive until DNS cutover
- `az group delete` removes all Azure resources if needed
