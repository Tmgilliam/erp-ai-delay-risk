# Interview Talk Track — ERP AI Delay Risk Prediction System

**Dr. Tatianna Gilliam** | Three versions for different interview stages.

---

## Version 1 — 30 Seconds (Recruiter Screen)

*"I built a production AI system that predicts ERP shipment delay risk before orders go late. It's deployed live — FastAPI inference API, batch scoring, executive dashboard, all containerized on Cloud Run. I designed it from ten years inside healthcare manufacturing ERP operations, not from the outside. Phase two adds drift monitoring and a full Azure migration architecture. Happy to go deeper on architecture or the business problem."*

---

## Version 2 — 90 Seconds (Hiring Manager)

*"In healthcare manufacturing, I managed ERP operations where we held 98% inventory accuracy and 95% fill rate. Late shipments triggered real costs — expedites, escalations, replanning. The delay risk signals were already in the ERP data, but nobody surfaced them until the delay happened.*

*I built a production system that scores open orders for delay risk before shipment. Features come from operational drivers I actually managed: lead time variance, inventory coverage, supplier reliability, customer backlog. Not generic ML columns — operational signals.*

*It's live today: FastAPI API with real-time and batch scoring, Streamlit executive dashboard, Docker on Cloud Run. I deployed on GCP deliberately to prove cloud-agnostic design — the architecture migrates to Azure Container Apps, APIM, and Entra ID with infrastructure substitution, not re-architecture.*

*Phase two adds drift monitoring, an Azure migration architecture doc, and an executive dashboard that tells operations leadership what to do — not just what the model scored. I didn't model delay risk from the outside. I spent years inside the system that generates it."*

---

## Version 3 — Deep Dive (Technical Panel, 5 Minutes)

### Opening — Business Problem (45 seconds)

*"Let me start with why this exists, not what stack I used.*

*In healthcare manufacturing at SSCOR, I led ERP and warehouse operations where shipping accuracy was 100%, fill rate was 95%, and I implemented a full Scanco/Sage 100 WMS in 2024. Late shipments weren't a dashboard metric — they were Monday morning crises. The data to predict them was in purchasing patterns, lead time variance, and inventory positioning. But ERP systems don't surface risk — they surface transactions.*

*This project asks: can we score open orders for delay risk early enough that operations can act? And can we deliver that score in a form a VP opens before a planning call — not a Jupyter notebook?"*

### Architecture Walkthrough (2 minutes)

*"Phase one is a containerized ML inference platform with three components:*

*1. **FastAPI inference API** — `POST /predict` for real-time single-order scoring, `POST /predict/batch` for open-order book review. Pydantic schemas enforce feature alignment with training columns. This is the most common production ML failure — silent schema drift between training and inference. The schema layer prevents it.*

*2. **scikit-learn RandomForest** — 16 domain-informed features, 5,000 training records. I chose classic ML over deep learning because interpretability is a business requirement. Operations leadership needs feature importance they can map to actions — lead time variance, supplier reliability, inventory coverage. A model nobody trusts is a model nobody uses.*

*3. **Streamlit executive dashboard** — auth-protected, role-ready. Single-order scoring, batch CSV upload, ERP-style KPIs. Speed-to-proof tradeoff: Streamlit in days, not a React frontend in weeks.*

*Dual scoring is an operational decision, not a technical one. ERP has two scoring moments: ad-hoc during a customer call, and systematic before Monday planning. One mode serves one persona.*

*Deployed on GCP Cloud Run — two containers, scale-to-zero, consumption pricing."*

### Phase 2 — Production ML Thinking (1.5 minutes)

*"Phase two is what separates a demo from enterprise AI:*

***Drift monitoring** — `DriftMonitor` class using Kolmogorov-Smirnov tests compares current scoring distributions against training baseline. Exposed via `GET /monitoring/drift-report`. In Azure, this flows to a Function, custom metric, Monitor alert, ops notification. In manufacturing, process drift has real cost — I understand this from managing 98% inventory accuracy, not from MLOps literature.*

***Azure migration architecture** — CAF-aligned path to Container Apps, APIM, Entra ID, Key Vault, Blob Storage for versioned models, Azure Monitor. Documented as a first-class portfolio artifact.*

***Executive dashboard upgrade** — 'Executive Risk Summary' panel first: high risk count, top 3 drivers in plain English, 30-day trend, recommended action callout. Designed for an operations VP, not an engineer."*

### Hard Questions — Prepared Answers

#### "Why GCP and not Azure?"

*"Deliberate choice. Phase one goal was proving the ML system works in production — fastest path was Cloud Run. The architecture is cloud-agnostic: containerized, stateless API, externalized config, no managed-service lock-in. Phase two documents the Azure landing zone — Container Apps, APIM, Entra ID, Blob Storage. Migration is infrastructure substitution, not re-architecture. I hold Azure certs because I know how to land workloads on Azure, not because every portfolio piece must run there on day one. The bridge framing is: Phase one proves the system works. Phase two proves I know where it goes in enterprise Azure."*

#### "What would you do differently?"

*"Three things, in priority order:*

*1. **Externalize model artifacts from day one** — Blob Storage with versioning instead of baked-in pickle. Enables model swap without image rebuild. I accepted the baked-in tradeoff for speed-to-deployment but would change this for a true enterprise rollout.*

*2. **Persist scoring history** — the 30-day trend in the executive dashboard uses simulated data until scoring runs are stored. In production, every batch score should write to Blob or Log Analytics for real trend analysis.*

*3. **APIM + Entra ID from the start** if deploying for an enterprise customer, not a portfolio proof. For portfolio speed, env-var auth was acceptable. For SSCOR-scale deployment, Zero Trust from day one."*

#### "How would this scale to enterprise?"

*"The architecture already scales horizontally — stateless API, containerized, no session affinity. Enterprise scale adds four layers:*

*1. **APIM** — rate limiting, auth, versioning for ERP integration teams consuming the API*
*2. **Event-driven scoring** — Service Bus trigger on order creation, not just batch/real-time pull*
*3. **Model registry** — MLflow or Azure ML for model versioning, A/B testing, champion/challenger*
*4. **Multi-tenant isolation** — per-plant or per-business-unit model variants if operational patterns differ*

*Container Apps handles the current scale. AKS becomes right when you add GPU training pipelines or multi-model serving. That's an escalation, not a day-one decision."*

#### "How does drift monitoring connect to enterprise ML governance?"

*"`/monitoring/drift-report` returns per-feature KS-test results with drift flags. In Azure: scheduled trigger calls the endpoint daily, Azure Function parses `drift_detected_count`, emits custom metric to Monitor, alert fires to ops team. Full JSON retained in Log Analytics for audit.*

*This closes the governance loop: deploy model → monitor distributions → alert on drift → investigate operational cause → retrain → promote new version from Blob Storage. Without this loop, you're running a static snapshot of last quarter's operations. In manufacturing, that's how fill rate erodes before anyone notices."*

### Close (15 seconds)

*"The through-line: I didn't model delay risk from the outside — I spent years inside the system that generates it. This project proves I build AI systems the way enterprises actually need them to work — operational credibility, production governance, and architecture that migrates across clouds without rewrites."*

**Power phrase:** *"I didn't model delay risk from the outside — I spent years inside the system that generates it."*
