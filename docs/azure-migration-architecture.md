# Azure Migration Architecture — ERP AI Delay Risk Prediction System

**Author:** Dr. Tatianna Gilliam, Cloud & AI Architect  
**Document Type:** Architecture Specification  
**Version:** 1.0 | June 2026

---

## Executive Summary

This document defines the target-state Azure architecture for the ERP AI Delay Risk Prediction System — a containerized ML inference platform currently deployed on Google Cloud Run. The migration follows Microsoft Cloud Adoption Framework (CAF) methodology and requires infrastructure substitution, not application re-architecture. The containerized, API-first, stateless design makes this a rehost path with clear optimization opportunities post-migration.

---

## Current State (GCP)

| Component | Implementation | Notes |
|-----------|---------------|-------|
| Inference API | Cloud Run (FastAPI) | Stateless, scale-to-zero, scikit-learn model loaded at startup |
| Executive Dashboard | Cloud Run (Streamlit) | Separate service, calls API via REST |
| Container Registry | Google Artifact Registry | API + dashboard images |
| Model Artifact | Baked into container (`models/delay_model.pkl`) | No versioning, no external storage |
| Authentication | Environment-variable dashboard auth | No API gateway, no identity provider |
| Monitoring | GCP built-in logging only | No custom metrics, no drift alerts |
| Batch Scoring | CLI pipeline + API batch endpoint | No scheduled orchestration |

**Operational context:** The system scores open ERP orders for shipment delay risk using domain-informed features — lead time variance, inventory coverage (ATP), supplier reliability, customer backlog, and AR risk signals. These are the same operational drivers managed in healthcare manufacturing environments where inventory accuracy, fill rate, and shipping accuracy are measured business outcomes.

---

## Target State (Azure)

| Component | Azure Service | Rationale |
|-----------|--------------|-----------|
| Inference API | **Azure Container Apps** | Scale-to-zero, consumption plan, no AKS overhead for this workload size |
| Executive Dashboard | **Azure Container Apps** | Same platform, independent scaling |
| Container Registry | **Azure Container Registry (ACR)** | Geo-replicated image storage, managed identity pull |
| API Governance | **Azure API Management (APIM)** | Rate limiting, auth, versioning, developer portal |
| Authentication | **Microsoft Entra ID** | Managed identity for service-to-service; OAuth for dashboard users |
| Secrets | **Azure Key Vault** | API keys, connection strings, model storage credentials |
| Model Artifacts | **Azure Blob Storage** | Versioned model storage; enables MLOps swap without rebuild |
| Monitoring | **Azure Monitor + Log Analytics** | Custom metrics, alert rules, operational dashboards |
| Drift Detection | **Existing `/monitoring/drift-report` → Azure Function → custom metric** | Governance signal, not just logs |

### Target Architecture Flow

```
[ERP System / Batch Pipeline]
        │
        ▼
[Azure API Management]
  ├── Rate limiting
  ├── Entra ID auth (OAuth 2.0)
  ├── API versioning (/v1/predict)
  └── Developer portal
        │
        ▼
[Azure Container Apps — API]
  ├── FastAPI inference
  ├── Managed identity → Key Vault, Blob Storage
  └── Model loaded from Blob (versioned)
        │
        ├──► [Azure Monitor] ← custom metrics, drift alerts
        │
        └──► [Log Analytics] ← structured inference + drift logs

[Azure Container Apps — Dashboard]
  ├── Streamlit executive UI
  ├── Entra ID SSO
  └── Calls API via APIM (managed identity)

[Azure Blob Storage]
  └── models/delay_model_v{version}.pkl

[Azure Key Vault]
  ├── Blob connection string
  ├── APIM subscription keys (if hybrid auth)
  └── Dashboard secrets
```

---

## Migration Approach — CAF Methodology

### Assess

| Activity | Deliverable |
|----------|-------------|
| Dependency discovery | Container images, env vars, model artifact, API contracts |
| Workload classification | **Rehost candidate** — containerized, stateless, no managed service lock-in |
| Cost modeling | Container Apps consumption vs. current Cloud Run spend; APIM Developer tier for portfolio/enterprise |
| Identity mapping | Dashboard users → Entra ID groups (EXEC, OPS, ANALYST) |
| Data classification | ERP order payloads = operational data; no PII in current schema |

### Migrate

**Path: Rehost (lift-and-shift containers)**

This workload is correctly classified as rehost because:

- Application is already containerized (Dockerfile.api, Dockerfile.dashboard)
- No database migration required (stateless API, CSV-based batch input)
- No code rewrite needed — infrastructure substitution only
- Feature alignment logic is platform-independent

Migration sequence:

1. Provision ACR, push existing images
2. Deploy Container Apps (API + dashboard) with managed identity
3. Externalize model to Blob Storage; update startup to load from Blob
4. Place APIM in front of API Container App
5. Configure Entra ID app registration for dashboard + APIM OAuth
6. Wire Key Vault for secrets
7. Configure Azure Monitor alerts (health, latency, drift)
8. DNS cutover; decommission Cloud Run after validation window

### Optimize

| Optimization | Trigger | Action |
|-------------|---------|--------|
| Autoscaling rules | Post-migration baseline | Scale on HTTP concurrent requests; min replicas = 0 for cost |
| Azure Advisor | 30 days post-migration | Right-size Container Apps CPU/memory |
| Reserved capacity | Sustained production traffic | Evaluate only if min replicas > 0 justified by SLA |
| APIM caching | Repeated identical payloads | Semantic caching for batch pre-checks (optional) |
| Blob lifecycle | Model version accumulation | Tier older model versions to Cool/Archive |

---

## Key Design Decisions

### 1. Container Apps vs. AKS

**Decision:** Azure Container Apps

| Factor | Container Apps | AKS |
|--------|---------------|-----|
| Operational overhead | Managed platform, no node management | Cluster ops, node pools, upgrades |
| Scale-to-zero | Native | Possible but complex (KEDA) |
| Cost at this scale | Consumption-based, minimal idle cost | Control plane + node cost even at low traffic |
| Workload fit | 2 containerized services, HTTP-triggered | Over-engineered for current footprint |

**Trade-off accepted:** Less Kubernetes-native flexibility. If the portfolio evolves into a multi-model MLOps platform with GPU training nodes, AKS becomes the right next step. For inference + dashboard at current scale, Container Apps is the correct enterprise decision.

### 2. API Management vs. Direct Exposure

**Decision:** APIM in front of Container Apps API

Even for a single API, APIM provides:

- **Rate limiting** — protects inference from runaway batch jobs
- **Authentication** — Entra ID integration without modifying FastAPI auth middleware
- **Versioning** — `/v1/predict` today, `/v2/predict` tomorrow without breaking consumers
- **Developer portal** — self-service API docs for ERP integration teams
- **Observability** — APIM analytics complement Container Apps metrics

**Trade-off accepted:** APIM adds cost and latency (~5–15ms). For an enterprise deployment where the API integrates with ERP workflows, governance value exceeds the overhead. For local dev, direct Container Apps access remains available.

### 3. Entra ID Managed Identity vs. API Key Auth

**Decision:** Managed identity for service-to-service; Entra ID OAuth for human users

| Auth Model | Security Posture | Enterprise Fit |
|-----------|-----------------|----------------|
| API keys in env vars | Shared secret, rotation burden | Demo/portfolio acceptable |
| Managed identity | No secrets in code, automatic rotation | Zero Trust aligned |
| Entra ID OAuth | Per-user audit trail, group-based RBAC | EXEC/OPS/ANALYST role model |

**Trade-off accepted:** More setup complexity (app registrations, RBAC assignments). Eliminates secret sprawl and provides auditability required in healthcare manufacturing compliance contexts.

### 4. Blob Storage for Model Artifacts vs. Baked-in Pickle

**Decision:** Externalize to Azure Blob Storage with versioned paths

| Approach | MLOps Capability | Deployment Impact |
|----------|-----------------|-------------------|
| Baked-in pickle | Rebuild image to update model | Simple, current Phase 1 state |
| Blob Storage | Swap model version without rebuild | Enables CI/CD model promotion |

**Trade-off accepted:** Startup dependency on Blob availability. Mitigated by health check failure → no traffic routing, and optional local cache on container start.

---

## Architecture Diagram

```
GCP (Current)                    Azure (Target)
─────────────────────            ─────────────────────────────────
Cloud Run (API)                  Azure API Management
Cloud Run (Dashboard)     →      Azure Container Apps (API)
GCR / Artifact Registry          Azure Container Apps (Dashboard)
GCP Logging                      Azure Container Registry
                                 Azure Monitor + Log Analytics
                                 Azure Key Vault
                                 Azure Blob Storage (models)
                                 Entra ID (auth)
```

### Detailed Azure Topology

```
                    ┌─────────────────────┐
                    │   Microsoft Entra   │
                    │   ID (OAuth/MI)     │
                    └─────────┬───────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│  APIM Gateway  │  │ Container Apps  │  │ Container Apps   │
│  /v1/predict   │  │ API (FastAPI)   │  │ Dashboard (ST)   │
│  rate limit    │──│ scale 0→N       │  │ scale 0→N        │
│  auth          │  └────────┬────────┘  └────────┬─────────┘
└────────────────┘           │                     │
                             │                     │
                    ┌────────┼─────────────────────┤
                    │        │                     │
                    ▼        ▼                     ▼
            ┌──────────┐ ┌──────────┐      ┌──────────────┐
            │Key Vault │ │Blob Store│      │Log Analytics │
            │ secrets  │ │ models/  │      │+ Monitor     │
            └──────────┘ └──────────┘      └──────────────┘
                    ▲
                    │
            ┌───────┴────────┐
            │ Azure Container│
            │ Registry (ACR) │
            └────────────────┘
```

---

## Drift Monitoring Integration

Phase 2 adds `GET /monitoring/drift-report` with KS-test drift detection. In Azure:

1. **Scheduled trigger** (Logic App or Container Apps Job) calls drift endpoint daily
2. **Azure Function** parses `drift_detected_count` and per-feature flags
3. **Custom metric** `ERP.DelayRisk.DriftDetected` emitted to Azure Monitor
4. **Alert rule** fires when count > 0 → Action Group → Teams notification
5. **Log Analytics** retains full JSON for model governance audit

This closes the loop between ML deployment and operational awareness — the same discipline applied to inventory accuracy and fill rate monitoring on the manufacturing floor.

---

## GCP → Azure Bridge Framing

This project was deliberately deployed on GCP to demonstrate cloud-agnostic architecture thinking. The containerized, API-first design means the Azure migration path is straightforward — no re-architecture required, only infrastructure substitution. The design decisions that matter (separation of concerns, externalized configuration, stateless API design) are platform-independent.

When asked why production is on GCP while holding Azure certifications: the answer is architectural intent, not platform preference. Phase 1 proves the ML system works. Phase 2 proves the architect knows how to land it on Azure with enterprise governance. Both are portfolio assets.

---

## Appendix: Service Mapping

| GCP | Azure | Migration Effort |
|-----|-------|-----------------|
| Cloud Run | Container Apps | Low — redeploy same image |
| Artifact Registry | ACR | Low — retag and push |
| Cloud Logging | Log Analytics | Medium — structured log format |
| (none) | APIM | Medium — policy configuration |
| (none) | Entra ID | Medium — app registration, RBAC |
| (none) | Key Vault | Low — secret migration |
| (none) | Blob Storage | Low — upload model, update startup |
| (none) | Azure Monitor | Medium — custom metrics, alerts |

**Estimated migration effort:** 2–3 days for infrastructure provisioning and validation; 1 day for APIM + Entra ID configuration; ongoing optimization per CAF Optimize phase.
