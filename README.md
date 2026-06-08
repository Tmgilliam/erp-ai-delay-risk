# ERP AI — Delay Risk Prediction System

**MTP Working Copy** | Dr. Tatianna Gilliam, Cloud & AI Architect (AZ-305 | AI-102 | AZ-104)

> **Public repo:** [github.com/Tmgilliam/erp-ai-delay-risk](https://github.com/Tmgilliam/erp-ai-delay-risk)  
> This folder is the local working and portfolio layer. GitHub is the public-facing, deployed version.

---

## Phase 1 Summary

Phase 1 delivered a production-deployed AI system that predicts shipment delay risk for open ERP orders. A FastAPI inference service (real-time + batch scoring), Streamlit executive dashboard, scikit-learn classifier with domain-informed features, and Docker containers deployed to Google Cloud Run. See [phase1-summary.md](phase1-summary.md) for full inventory.

## Phase 2 Additions

| Addition | Location | What It Proves |
|----------|----------|----------------|
| Azure migration architecture | `docs/azure-migration-architecture.md` | Platform-agnostic design, CAF-aligned Azure landing zone |
| Drift monitoring | `monitoring/drift_monitor.py` | Production ML governance, KS-test detection |
| Drift API endpoint | `GET /monitoring/drift-report` in `app/main.py` | Operational monitoring integration point |
| Executive dashboard upgrade | `dashboard/app.py` | Business-user design, not engineer-facing |
| Architecture Decision Records | `docs/design-decisions.md` | Enterprise trade-off reasoning |
| Portfolio packaging | `portfolio/` | Interview-ready case study, talk tracks, resume bullets |

## Phase 3 Additions

| Addition | Location | What It Proves |
|----------|----------|----------------|
| Scoring history persistence | `monitoring/scoring_history.py` | Real executive trend data, not simulated |
| Scoring history API | `GET /monitoring/scoring-history` | Ops telemetry integration point |
| MTP Docker stack | `docker-compose.mtp.yml` | Runnable local portfolio environment |
| Configurable model path | `MODEL_PATH` env var in `app/model.py` | Azure Blob externalization readiness |

See [docs/phase3-changelog.md](docs/phase3-changelog.md).

## Phase 4 — Azure Deployment

| Addition | Location | What It Proves |
|----------|----------|----------------|
| Bicep IaC | `infra/azure/` | CAF-aligned Azure landing zone |
| Blob model loader | `app/azure_model_loader.py` | MLOps externalized artifacts |
| Deploy script | `scripts/azure/deploy.ps1` | One-command Azure migration |
| Deployment guide | `docs/azure-deployment-guide.md` | Enterprise deploy runbook |

```powershell
.\scripts\azure\deploy.ps1 -ResourceGroup "rg-erp-ai-delay-risk" -Location "eastus2"
```

See [docs/phase4-changelog.md](docs/phase4-changelog.md) and [docs/azure-deployment-guide.md](docs/azure-deployment-guide.md).

## Phase 5 — Entra ID + APIM Readiness (No Subscription Required)

| Addition | Location | Cost |
|----------|----------|------|
| Entra app registration | `scripts/entra/register-apps.ps1` | Free |
| API JWT auth (optional) | `app/auth/entra_jwt.py` | Free |
| Dashboard Entra login | `dashboard/auth.py` | Free |
| APIM policies + OpenAPI | `config/apim/` | Free (artifacts) |
| APIM validate script | `scripts/azure/configure-apim.ps1 -ValidateOnly` | Free |

```powershell
# Local dev (no Entra required)
.\scripts\local\start-dev.ps1

# Free Entra/APIM prep
.\scripts\local\preflight.ps1
.\scripts\entra\register-apps.ps1          # needs: az login
.\scripts\azure\configure-apim.ps1 -ValidateOnly

# No Azure CLI? Manual Entra: docs/entra-portal-setup.md
```

Defaults unchanged: `ENTRA_AUTH_ENABLED=false`, `AUTH_MODE=env`. See [docs/entra-apim-readiness.md](docs/entra-apim-readiness.md).

## Phase 6 — GCP Platform Layer (Companion Projects)

| Project | GitHub | What It Proves |
|---------|--------|----------------|
| GCP Enterprise Data Pipeline | [gcp-enterprise-data-pipeline](https://github.com/Tmgilliam/gcp-enterprise-data-pipeline) | BigQuery medallion, Vertex AI Feature Store, ML training pipelines |
| GCP Microservices + Apigee | [gcp-microservices-api-gateway](https://github.com/Tmgilliam/gcp-microservices-api-gateway) | Service decomposition, API gateway governance, circuit breaker |

See [docs/gcp-platform-roadmap.md](docs/gcp-platform-roadmap.md) for how these projects feed the same delay risk model at enterprise scale. Phase 1 trained locally; Phase 6+ is the data platform and microservices infrastructure underneath production inference.

## Project Structure

```
erp-ai-delay-risk/
├── README.md                      # This file
├── phase1-summary.md              # Phase 1 inventory and MTP context
├── app/                           # Refactored FastAPI module (MTP local dev)
│   ├── main.py                    # /predict, /predict/batch, /health, /monitoring/drift-report
│   ├── model.py                   # Model loading and inference
│   └── schemas.py                 # Pydantic request/response schemas
├── dashboard/
│   └── app.py                     # Streamlit dashboard with Executive Risk Summary
├── pipelines/
│   └── batch_score.py             # CLI batch scoring workflow
├── monitoring/                    # Phase 2 — drift detection
│   ├── drift_monitor.py
│   ├── reference_baseline.csv
│   └── README.md
├── infra/azure/                   # Phase 4 — Bicep IaC for Azure deployment
│   ├── main.bicep
│   └── modules/
├── scripts/azure/                 # Phase 4 — deployment automation
│   └── deploy.ps1
├── .azure/                        # Deployment plan artifact
│   └── deployment-plan.md
├── docs/                          # Architecture and deployment documentation
│   ├── azure-migration-architecture.md
│   ├── azure-deployment-guide.md
│   ├── design-decisions.md
│   ├── phase2-changelog.md
│   ├── phase3-changelog.md
│   ├── phase4-changelog.md
│   ├── phase5-changelog.md
│   ├── phase6-changelog.md
│   ├── gcp-platform-roadmap.md
│   └── entra-apim-readiness.md
├── config/
│   ├── apim/                      # APIM policies + OpenAPI (apply when paid)
│   ├── entra/                     # App role definitions
│   └── feature-flags.env.example
├── scripts/entra/
│   └── register-apps.ps1          # Free Entra registration
├── portfolio/                     # Phase 2 — interview assets
│   ├── case-study.md
│   ├── interview-talk-track.md
│   └── resume-bullets.md
├── src/                           # Original GitHub layout (Cloud Run deployment)
├── models/                        # Trained model artifact
├── data/                          # Training and scoring sample data
└── requirements.txt
```

## Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Seed executive trend history (first run)
python monitoring/seed_scoring_history.py

# Option A — Docker (recommended)
docker compose -f docker-compose.mtp.yml up --build

# Option B — Manual
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
# separate terminal:
set DASH_USER=demo@erp-ai.local
set DASH_PASS=demo1234
set API_URL=http://127.0.0.1:8001
streamlit run dashboard/app.py

# Batch score (auto-records to scoring history via API)
python pipelines/batch_score.py --input data/open_orders_scoring_sample.csv

# Monitoring
curl http://127.0.0.1:8001/monitoring/drift-report
curl http://127.0.0.1:8001/monitoring/scoring-history
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Real-time single-order scoring |
| `POST` | `/predict/batch` | Batch scoring |
| `GET` | `/monitoring/drift-report` | Feature drift detection report |
| `GET` | `/monitoring/scoring-history` | Persisted scoring trend for executive dashboard |
| `GET` | `/docs` | OpenAPI documentation |
| `POST` | `/score_order` | Legacy alias for `/predict` |
| `POST` | `/batch_score` | Legacy alias for `/predict/batch` |

## MTP Note

This is the **local working and portfolio layer**. It mirrors the GitHub codebase and adds Phase 2 artifacts (Azure architecture, drift monitoring, executive dashboard upgrade, portfolio packaging) that the public repo does not carry. GitHub remains the deployed, public-facing version.

**Owner positioning:** *"I didn't model delay risk from the outside — I spent years inside the system that generates it."*
