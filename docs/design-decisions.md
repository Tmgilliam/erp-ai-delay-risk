# Architecture Decision Records — ERP AI Delay Risk

**Project:** ERP AI — Delay Risk Prediction System  
**Author:** Dr. Tatianna Gilliam

Each record follows: **Decision → Alternatives → Rationale → Trade-offs Accepted**

---

## ADR-001: GCP for Initial Deployment (Not Azure)

### Decision
Deploy Phase 1 to Google Cloud Run on GCP, with a documented Azure migration path as Phase 2.

### Alternatives Considered
- **Azure Container Apps from day one** — aligns with certification portfolio
- **AWS Lambda + API Gateway** — serverless alternative
- **Local-only deployment** — no cloud proof

### Rationale
The primary Phase 1 goal was proving the ML system works in production: domain-informed features, dual scoring, executive dashboard, containerized deployment. Cloud Run offered the fastest path to a live URL with scale-to-zero and minimal infrastructure overhead. Deliberately choosing GCP demonstrates cloud-agnostic thinking — the architecture (containerized API, stateless inference, externalized config) is platform-independent. Phase 2 documents the Azure landing zone, answering the inevitable interviewer question with a prepared migration architecture, not an apology.

### Trade-offs Accepted
- Portfolio viewers may question Azure certs + GCP deployment (mitigated by bridge framing)
- No Azure-native services in Phase 1 (APIM, Entra ID, Key Vault deferred to Phase 2 design)
- Dual cloud knowledge required for maintenance during migration window

---

## ADR-002: Azure Container Apps (Not AKS) for Target State

### Decision
Target Azure deployment uses Container Apps for both API and dashboard services.

### Alternatives Considered
- **AKS** — full Kubernetes control, GPU node pools for future ML training
- **Azure App Service** — PaaS for web apps, less container-native
- **Azure Functions** — serverless, but poor fit for long-running Streamlit + model-in-memory

### Rationale
Current workload: 2 containerized HTTP services, CPU-only inference, intermittent traffic (executive dashboard + batch scoring). Container Apps provides scale-to-zero, managed ingress, and consumption pricing without cluster operations. AKS would add node management, upgrade cycles, and baseline cost for a portfolio-scale workload. If the system evolves to multi-model MLOps with GPU training, AKS becomes the right escalation — not the starting point.

### Trade-offs Accepted
- Less fine-grained Kubernetes control (no custom CRDs, limited sidecar patterns)
- Container Apps networking model requires learning vs. familiar AKS service mesh options
- Migration to AKS later requires redeployment effort if scale demands it

---

## ADR-003: scikit-learn RandomForest (Not Deep Learning)

### Decision
Use scikit-learn RandomForestClassifier for delay risk classification.

### Alternatives Considered
- **Gradient boosting (XGBoost/LightGBM)** — potentially higher accuracy
- **Neural network (PyTorch/TensorFlow)** — deep learning for tabular data
- **Rule-based scoring** — no ML, pure business logic thresholds

### Rationale
Problem size: ~5,000 training records, 16 features, binary classification. Interpretability requirement: operations leadership needs to understand *why* an order is flagged — feature importance maps to plain-English drivers (lead time variance, supplier reliability, inventory coverage). RandomForest delivers adequate accuracy with built-in feature importance, fast inference (<10ms), and no GPU dependency. In manufacturing operations, a model nobody trusts is a model nobody uses — interpretability is a business requirement, not a nice-to-have.

### Trade-offs Accepted
- May leave accuracy on the table vs. tuned gradient boosting
- One-hot encoded categoricals expand feature space (manageable at this scale)
- Not suitable if feature space grows to hundreds of dimensions or unstructured data

---

## ADR-004: Dual Scoring — Real-time API + Batch Pipeline

### Decision
Expose both real-time single-order scoring (`POST /predict`) and batch scoring (`POST /predict/batch` + CLI pipeline).

### Alternatives Considered
- **Real-time only** — API for all scoring, including nightly batch
- **Batch only** — nightly scoring sufficient for planning workflows
- **Streaming (Event Hubs)** — event-driven scoring on order creation

### Rationale
ERP operations have two distinct scoring moments:
1. **Real-time** — CSR or planner scores a single order during a customer call or expedite decision
2. **Batch** — operations manager scores the full open-order book before Monday planning

Both happen in production manufacturing environments. Dual mode mirrors how Dr. Gilliam used ERP data: ad-hoc lookups during the day, systematic review before planning cycles. A single mode would serve one persona and ignore the other.

### Trade-offs Accepted
- Two code paths to maintain (mitigated by shared `DelayRiskModel` class)
- Batch via API loads all records in one request (size limits apply; large books need chunked batch)
- No event-driven scoring yet (Phase 3 candidate with Service Bus/Event Hubs)

---

## ADR-005: Streamlit Dashboard (Not Custom Frontend)

### Decision
Use Streamlit for the executive dashboard.

### Alternatives Considered
- **React/Next.js custom frontend** — full design control, production UX
- **Power BI embedded** — Microsoft ecosystem alignment
- **Grafana** — metrics-focused, less form-based interaction

### Rationale
Phase 1 goal was speed-to-proof: an operations VP opens a dashboard before a Monday planning call and sees risk, drivers, and recommended actions. Streamlit delivers this in days, not weeks. Auth, batch upload, KPI charts, and API integration are functional without frontend engineering overhead. The dashboard is a decision-support tool, not a customer-facing product — polish yields diminishing returns vs. getting operational signal in front of stakeholders.

### Trade-offs Accepted
- Limited UI customization vs. custom React
- Streamlit session model constrains multi-user concurrency patterns
- Not suitable as a long-term customer-facing portal (migrate to custom frontend if productized)

---

## ADR-006: Externalized Model Artifacts (Phase 2) vs. Baked-in Pickle (Phase 1)

### Decision
Phase 1: model baked into container image. Phase 2 target: Azure Blob Storage with versioned model paths.

### Alternatives Considered
- **MLflow Model Registry** — full MLOps lifecycle
- **Azure Machine Learning managed endpoint** — Azure-native model hosting
- **Continue baked-in** — simplest, no external dependency

### Rationale
Phase 1 baked-in pickle was correct for speed-to-deployment: one artifact, one container, one `docker push`. Phase 2 externalization enables model versioning and swap-without-rebuild — the minimum viable MLOps pattern for enterprise governance. When a drift alert fires (ADR-002 monitoring integration), the response is retrain → upload v2 → update Blob pointer → no image rebuild. This is how production ML systems actually operate.

### Trade-offs Accepted
- Phase 1: model updates require image rebuild and redeploy
- Phase 2: startup dependency on Blob availability (mitigated by health checks + local cache)
- No full MLflow integration yet (sufficient for current scale; add when model versions exceed ~5)

---

## Decision Summary Matrix

| ADR | Decision | Business Driver |
|-----|----------|----------------|
| 001 | GCP first, Azure documented | Cloud-agnostic proof + cert alignment |
| 002 | Container Apps over AKS | Cost and ops overhead at current scale |
| 003 | scikit-learn over deep learning | Interpretability for operations trust |
| 004 | Dual scoring modes | Real-time + planning-cycle workflows |
| 005 | Streamlit over custom UI | Speed-to-proof for executive users |
| 006 | Externalize models in Phase 2 | MLOps governance and drift response |
