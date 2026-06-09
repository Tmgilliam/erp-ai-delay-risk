# GCP Platform Roadmap — ERP AI Delay Risk Phase 6+

**Dr. Tatianna Gilliam** | Cloud & AI Architect | AZ-305 | AI-102 | AZ-104

Phase 1 deployed a monolithic FastAPI inference service to Cloud Run. Phases 2–5 added Azure migration architecture, drift monitoring, Entra ID readiness, and APIM policy artifacts. **Phase 6+ extends the GCP story** with two companion projects that prove enterprise architecture depth on the same platform.

---

## Portfolio Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ERP AI DELAY RISK (this repo) — Application & inference layer                │
│  Cloud Run FastAPI · Streamlit dashboard · drift monitoring                 │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
┌──────────────────────────────┐      ┌──────────────────────────────────────┐
│  GCP Enterprise Data Pipeline │      │  GCP Microservices + Apigee Gateway  │
│  BigQuery · Dataflow ·        │      │  3 Cloud Run services · API gateway  │
│  Vertex AI Feature Store      │      │  governance · circuit breaker        │
└──────────────────────────────┘      └──────────────────────────────────────┘
   Data platform layer                    Service decomposition layer
```

| Project | Repo | What It Proves |
|---------|------|----------------|
| **GCP ERP AI Platform (umbrella)** | [github.com/Tmgilliam/gcp-erp-ai-platform](https://github.com/Tmgilliam/gcp-erp-ai-platform) | All three layers in one standalone clone |
| **ERP AI Delay Risk** | [github.com/Tmgilliam/erp-ai-delay-risk](https://github.com/Tmgilliam/erp-ai-delay-risk) | Production ML deployment on Cloud Run |
| **GCP Enterprise Data Pipeline** | [github.com/Tmgilliam/gcp-enterprise-data-pipeline](https://github.com/Tmgilliam/gcp-enterprise-data-pipeline) | Data lake, warehouse, Feature Store, ML pipelines |
| **GCP Microservices + Apigee** | [github.com/Tmgilliam/gcp-microservices-api-gateway](https://github.com/Tmgilliam/gcp-microservices-api-gateway) | Microservices decomposition, API gateway governance |

---

## Phase Evolution

| Phase | Focus | Platform |
|-------|-------|----------|
| Phase 1 | Monolith inference + Cloud Run deploy | GCP |
| Phase 2 | Drift monitoring, Azure migration docs, portfolio | GCP + Azure design |
| Phase 3 | Scoring history, local Docker stack | GCP |
| Phase 4 | Bicep IaC, Azure Container Apps path | Azure |
| Phase 5 | Entra ID + APIM readiness (free tier) | Azure |
| **Phase 6** | **Data platform layer** | **GCP** |
| **Phase 7** | **Microservices + API gateway** | **GCP** |

---

## Phase 6 — Enterprise Data Platform

**Project:** [gcp-enterprise-data-pipeline](../gcp-enterprise-data-pipeline/) (sibling repo in MTP-Projects)

### Problem solved

Phase 1 trained on CSV exports and computed features at request time. At enterprise scale, that creates training-serving skew, stale features, and no artifact lineage.

### What the data pipeline provides

| Capability | GCP Service | Feeds ERP AI Delay Risk |
|------------|-------------|-------------------------|
| Batch ingestion | GCS landing zones | Nightly Sage 100 / WMS exports |
| Streaming ingestion | Pub/Sub | Real-time inventory and order events |
| Transformation | Dataflow + BigQuery SQL | `ml_features.sku_risk_features` table |
| Feature serving | Vertex AI Feature Store | `/predict` online feature lookup |
| Model training | Vertex AI Pipelines | Same sklearn classifier, gated promotion |
| Infrastructure | Terraform | Reproducible dev/prod environments |

### Feature alignment

The `ml_features.sku_risk_features` table computes the same domain signals as `src/features.py`:

- Demand windows (7d, 30d, 90d)
- Lead time percentiles (p50, p90)
- Stockout flags and days of supply
- `delay_risk_score` composite

### Key documents

- [architecture-overview.md](../gcp-enterprise-data-pipeline/docs/architecture-overview.md)
- [multi-cloud-comparison.md](../gcp-enterprise-data-pipeline/docs/multi-cloud-comparison.md) — GCP / Azure / AWS mapping

---

## Phase 7 — Microservices + API Gateway

**Project:** [gcp-microservices-api-gateway](../gcp-microservices-api-gateway/) (sibling repo in MTP-Projects)

### Problem solved

The Phase 1 monolith couples risk scoring, feature retrieval, and notifications in one deployable unit. Different components have different scaling profiles and team ownership boundaries.

### Decomposition

| Service | Responsibility | Scaling |
|---------|----------------|---------|
| Risk Scoring | `POST /v1/score`, `/v1/score/batch` | min 0, max 10 |
| Feature Service | `GET /v1/features/{id}` | min 1 (always warm) |
| Notification | `POST /v1/notify` via Pub/Sub | min 0, event-driven |

### API governance (Apigee)

- API key authentication at gateway edge
- Rate limiting (100 req/min standard tier)
- Request ID injection and standardized error format
- Route `/v1/score` → risk-scoring, `/v1/features` → feature-service

### Azure equivalent

This is the GCP implementation of the Phase 5 APIM + Container Apps pattern documented in `docs/azure-migration-architecture.md` and `config/apim/`.

| GCP | Azure |
|-----|-------|
| Apigee | API Management |
| Cloud Run | Container Apps |
| Pub/Sub | Service Bus |
| Cloud Run IAM | Managed Identity |

### Key documents

- [microservices-architecture.md](../gcp-microservices-api-gateway/docs/microservices-architecture.md)
- [api-governance-design.md](../gcp-microservices-api-gateway/docs/api-governance-design.md)
- [gateway/openapi-spec.yaml](../gcp-microservices-api-gateway/gateway/openapi-spec.yaml)

---

## Multi-Cloud Positioning

| Platform | Evidence in Portfolio |
|----------|------------------------|
| **GCP** | Production Cloud Run deploy + data pipeline + microservices |
| **Azure** | AZ-305 / AI-102 / AZ-104 certs + Bicep IaC + APIM policies |
| **AWS** | Documented equivalents in data pipeline multi-cloud comparison (SAA-C03 path) |

The application (this repo) stays stable. The infrastructure projects demonstrate that the same delay risk model can be fed by an enterprise data platform and exposed through governed microservices — on GCP today, on Azure via the documented migration path.

---

## Interview Framing

> "Phase 1 proved the model works in production on Cloud Run. The data pipeline project is how it trains and serves features at enterprise scale. The microservices project is how the API layer decomposes when scoring volume, feature latency, and notification logic need independent scaling. Azure APIM and Container Apps are the same pattern on the other side of the portfolio."

---

*Dr. Tatianna Gilliam — My production deployment is on GCP. My certifications are on Azure. My architecture thinking is platform-agnostic.*
