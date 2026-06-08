# Case Study: ERP AI — Delay Risk Prediction System

**Dr. Tatianna Gilliam** | Cloud & AI Architect | AZ-305 | AI-102 | AZ-104

---

## Business Problem

ERP environments generate operational data that contains delay risk signals — in purchasing patterns, lead time variance, inventory positioning — that go unread until the delay has already happened.

In healthcare manufacturing, a late shipment is not an abstract metric. It triggers expedite costs, customer escalations, and downstream production replanning. The question this project answers: **can we surface that risk earlier, in a form operations stakeholders can act on?**

Not a data science experiment. A planning tool.

---

## What Makes This Different From a Generic ML Project

I didn't model delay risk from the outside — I spent years inside the system that generates it.

This project is built on 10 years of ERP and manufacturing operations leadership at SSCOR, where I managed environments with:

- **98% inventory accuracy**
- **97% cycle count accuracy**
- **100% shipping accuracy** (zero errors)
- **95% fill rate**
- Full Scanco/Sage 100 WMS implementation (2024)
- Cleaned and structured 40 years of ERP transaction history

The feature engineering is not generic tabular column selection. Lead time variance, ATP coverage, supplier reliability, customer backlog, and AR risk signals are operational drivers I managed on the floor. The model reflects what actually predicts delays in ERP workflows because the person who built it has shipped against those KPIs.

A generic ML project asks: "what columns are available?" This project asks: "what would I check on Monday morning before the planning call?"

---

## Phase 1 Architecture

### Inference API (FastAPI)
- `POST /predict` — real-time single-order scoring for CSR/planner ad-hoc decisions
- `POST /predict/batch` — batch scoring for open-order book review
- Pydantic schema validation ensures ERP payloads match training feature alignment
- Stateless design: model loaded at startup, no session state, horizontally scalable

**Why FastAPI:** OpenAPI docs for ERP integration teams, async-ready, schema validation prevents silent feature misalignment — the most common production ML failure mode.

### Model (scikit-learn RandomForest)
- Trained on 5,000 ERP-style open orders with domain-informed features
- Feature importance maps to plain-English operational drivers
- Evaluation focused on operational reliability (feature alignment), not just F1/AUC

**Why scikit-learn, not deep learning:** 16 features, 5K records, interpretability requirement. Operations leadership needs to trust the signal, not admire the architecture.

### Executive Dashboard (Streamlit)
- Auth-protected, role-ready (EXEC / OPS / ANALYST)
- Single-order scoring, batch CSV upload, ERP-style KPIs
- Client-side latency instrumentation

**Why Streamlit:** Speed-to-proof. An operations VP needs a decision view, not a React portfolio piece.

### Batch Pipeline
- CLI workflow: CSV in → API batch score → scored CSV out
- Designed for nightly open-order book scoring before planning cycles

### Deployment (GCP Cloud Run)
- Docker containers for API and dashboard
- Scale-to-zero, consumption pricing
- Live endpoints: `/predict`, `/predict/batch`, `/health`, `/docs`

**Why dual scoring + dual services:** ERP operations have two scoring moments — ad-hoc during the day, systematic before planning. One mode serves one persona.

---

## Phase 2 Additions

### Azure Migration Architecture
Documents the CAF-aligned path from GCP Cloud Run to Azure Container Apps, APIM, Entra ID, Key Vault, Blob Storage, and Azure Monitor. Proves platform-agnostic design thinking — the containerized, API-first architecture requires infrastructure substitution, not re-architecture. This is the answer to "why GCP when you hold Azure certs?"

### Drift Monitoring
`DriftMonitor` class with KS-test detection, `/monitoring/drift-report` API endpoint, and Azure Monitor integration design. Proves production ML governance understanding — deploying inference without drift detection is an enterprise anti-pattern. In manufacturing, process drift has real operational cost. I understand this from the floor up.

### Executive Dashboard Upgrade
"Executive Risk Summary" panel as the first view:
- High Risk Count / % / Avg Score / Last Scored
- Top 3 delay risk drivers in plain English
- 30-day trend line
- Recommended Action callout

Proves designing for business users, not engineers. Every element answers a question an operations VP would ask before a Monday planning call.

---

## Interview Talk Track — 90-Second Version

*"In healthcare manufacturing, I managed ERP operations where shipping accuracy had to be 100% and fill rate was measured at 95%. Late shipments weren't abstract — they triggered expedite costs and customer escalations. The data to predict those delays was already in the ERP system, but nobody was reading it until the delay happened.*

*So I built a production AI system that scores open orders for delay risk before shipment. I didn't model this from the outside — I spent ten years inside the system that generates it. The features are operational drivers I actually managed: lead time variance, inventory coverage, supplier reliability, customer backlog.*

*Phase one is live on Cloud Run: FastAPI inference API, batch scoring pipeline, executive Streamlit dashboard, Docker containers. Dual scoring because ERP has two moments — ad-hoc during the day and systematic before Monday planning.*

*Phase two adds what separates a demo from enterprise AI: drift monitoring with a governance API endpoint, an Azure migration architecture using Container Apps, APIM, Entra ID, and Blob Storage for model versioning, and an executive dashboard upgrade that tells operations leadership what to do, not just what the model scored.*

*I deployed on GCP deliberately — to prove the architecture is cloud-agnostic. The containers, API contracts, and stateless design migrate to Azure with infrastructure substitution, not re-architecture. That's the point: I build AI systems the way enterprises actually need them to work."*

---

## Key Takeaway

This project connects ERP domain expertise directly to production AI deployment. It is proof that Dr. Gilliam builds AI systems the way enterprises actually need them to work — with operational credibility, production governance, and platform-agnostic architecture thinking.

**Power phrase:** *"I didn't model delay risk from the outside — I spent years inside the system that generates it."*
