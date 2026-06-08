# Phase 1 Summary — ERP AI Delay Risk Prediction System

## What Phase 1 Built

Phase 1 delivered a production-deployed AI system that scores open ERP orders for shipment delay risk before the delay happens. The stack includes:

- **FastAPI inference service** — real-time (`POST /predict`) and batch (`POST /predict/batch`) scoring with Pydantic schema validation
- **scikit-learn RandomForest classifier** — trained on domain-informed ERP features (lead time, ATP, supplier reliability, customer backlog, AR risk)
- **Streamlit executive dashboard** — auth-protected, role-ready UI for single-order and batch scoring with ERP-style KPIs
- **Batch scoring pipeline** — CSV-driven workflow for nightly open-order book scoring
- **Docker containers** — separate API and dashboard images deployed to **Google Cloud Run**
- **Model artifact** — `models/delay_model.pkl` loaded at container startup

Live deployment: [github.com/Tmgilliam/erp-ai-delay-risk](https://github.com/Tmgilliam/erp-ai-delay-risk)

## What Phase 1 Proves

Phase 1 is not a notebook exercise. It demonstrates:

1. **Domain-informed feature engineering** — features reflect operational drivers Dr. Gilliam managed in healthcare manufacturing (lead time variance, inventory coverage, supplier reliability), not generic tabular columns
2. **Production API design** — schema validation, health checks, OpenAPI docs, dual scoring modes
3. **Executive-facing delivery** — a dashboard operations leadership can open before a planning call, not a Jupyter output
4. **Cloud-native deployment** — containerized, stateless, horizontally scalable on Cloud Run

The model evaluation focused on operational reliability: consistent feature alignment between training and inference, not just leaderboard metrics.

## MTP Context

This folder (`MTP-Projects/erp-ai-delay-risk`) is the **local working and portfolio layer**. It mirrors the GitHub codebase for development and adds Phase 2 artifacts the public repo does not carry:

- Azure migration architecture documentation
- Model drift monitoring module
- Executive dashboard upgrade
- Interview-ready portfolio packaging (case study, talk tracks, resume bullets)
- Architecture Decision Records

GitHub remains the public-facing, deployed version. MTP-Projects is where portfolio depth and Phase 2 evolution live.

## Phase 1 Technical Inventory

| Component | Location | Status |
|-----------|----------|--------|
| Inference API | `app/main.py` (MTP) / `src/api.py` (GitHub) | Deployed |
| Model loader | `app/model.py` | Deployed |
| Schemas | `app/schemas.py` | Deployed |
| Dashboard | `dashboard/app.py` (MTP) / `src/dashboard.py` (GitHub) | Deployed |
| Batch pipeline | `pipelines/batch_score.py` | Deployed |
| Training data | `data/open_orders_train.csv` | 5,000 records |
| Model | `models/delay_model.pkl` | RandomForest, 300 estimators |
| Containers | `Dockerfile.api`, `Dockerfile.dashboard` | Cloud Run |
