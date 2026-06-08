# Phase 2 Changelog

**Project:** ERP AI — Delay Risk Prediction System  
**Owner:** Dr. Tatianna Gilliam  
**Date:** June 2026

## Overview

Phase 2 extends the deployed Phase 1 system with enterprise ML governance, Azure migration architecture, executive dashboard upgrades, and interview-ready portfolio packaging — without breaking existing inference functionality.

## Added

### Application Layer (`app/`)

- Refactored FastAPI module: `app/main.py`, `app/model.py`, `app/schemas.py`
- Standardized endpoints: `POST /predict`, `POST /predict/batch`, `GET /health`
- Legacy route aliases preserved: `/score_order`, `/batch_score`
- New monitoring endpoint: `GET /monitoring/drift-report`

### Model Monitoring (`monitoring/`)

- `drift_monitor.py` — KS-test drift detection via scipy
- `reference_baseline.csv` — training distribution baseline (5,000 records)
- `README.md` — enterprise drift monitoring rationale and Azure integration path

### Executive Dashboard (`dashboard/app.py`)

- **Executive Risk Summary** panel (first view):
  - Metric cards: High Risk Count, High Risk %, Avg Risk Score, Last Scored
  - Top 3 delay risk drivers in plain English with bar indicators
  - 30-day high-risk trend (simulated until scoring history is persisted)
  - Recommended Action callout based on current risk level

### Batch Pipeline (`pipelines/batch_score.py`)

- CLI batch scorer calling `/predict/batch` with CSV input/output

### Documentation (`docs/`)

- `azure-migration-architecture.md` — CAF-aligned GCP → Azure migration plan
- `design-decisions.md` — Architecture Decision Records (6 decisions)
- `phase2-changelog.md` — this file

### Portfolio (`portfolio/`)

- `case-study.md` — business problem, differentiators, architecture, talk track
- `interview-talk-track.md` — 30s, 90s, and 5-minute versions
- `resume-bullets.md` — 6 VP/ML-engineer bullets

### Root

- `README.md` — MTP working copy with GitHub link and Phase 2 summary
- `phase1-summary.md` — honest Phase 1 inventory and MTP context

## Changed

- `requirements.txt` — added `scipy` for drift detection (documented; evidently not added to keep footprint light)

## Unchanged (Backward Compatible)

- `src/` directory — original GitHub layout preserved
- `models/delay_model.pkl` — same model artifact
- Dockerfiles — still reference `src/` for deployed Cloud Run services
- Phase 1 Cloud Run deployment — no infrastructure changes in Phase 2

## Next Steps

Phase 3 completed — see [phase3-changelog.md](phase3-changelog.md).

Remaining (Phase 4):

- Azure Container Apps deployment per migration architecture doc
- Externalize model artifacts to Azure Blob Storage
- APIM + Entra ID integration for enterprise auth
- Evidently report visualizations (optional, additive)
