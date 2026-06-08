# Phase 3 Changelog

**Project:** ERP AI — Delay Risk Prediction System  
**Owner:** Dr. Tatianna Gilliam  
**Date:** June 2026

## Overview

Phase 3 makes the MTP stack runnable end-to-end and replaces simulated executive trend data with persisted scoring history — the first production-ML ops loop before Azure deployment.

## Added

### Scoring History (`monitoring/scoring_history.py`)

- Append-only CSV store for scoring run snapshots
- Records `scored_at`, order counts, high-risk %, avg risk score, and source
- Auto-recorded on every `POST /predict` and `POST /predict/batch` call

### API Endpoint

- `GET /monitoring/scoring-history?trend_days=30` — daily high-risk trend for executive dashboard

### Seed Script

- `monitoring/seed_scoring_history.py` — bootstraps 30 days of trend data from bundled sample orders

### MTP Docker Stack

- `Dockerfile.mtp.api` — runs `app.main:app` (Phase 2 layout)
- `Dockerfile.mtp.dashboard` — runs `dashboard/app.py`
- `docker-compose.mtp.yml` — local stack with volume mounts for `monitoring/` and `data/`

### Configuration

- `MODEL_PATH` environment variable in `app/model.py` — prep for Azure Blob externalized artifacts
- `.gitignore` — excludes runtime `scoring_history.csv` and batch output

## Changed

- **Executive dashboard** — 30-day trend reads persisted history from API; simulated data only when history is empty
- **Last Scored metric** — pulls from latest persisted scoring run when available

## Unchanged

- Phase 1 Cloud Run deployment (`docker-compose.yml`, `src/`, `Dockerfile.api`)
- Drift monitoring module and `/monitoring/drift-report`
- Portfolio and Azure architecture documentation

## Run the MTP Stack

```bash
# Seed trend history (first time)
python monitoring/seed_scoring_history.py

# Docker
docker compose -f docker-compose.mtp.yml up --build

# Dashboard: http://localhost:8501 (demo@erp-ai.local / demo1234)
# API: http://localhost:8001/docs
```

## Next Steps

Phase 4 completed — see [phase4-changelog.md](phase4-changelog.md).

Remaining:

- Execute Azure deployment and validate
- APIM + Entra ID integration (post-deploy configuration)
- Deploy drift metric Function App
- DNS cutover from GCP Cloud Run
