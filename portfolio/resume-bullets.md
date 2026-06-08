# Resume Bullets — ERP AI Delay Risk Prediction System

**Dr. Tatianna Gilliam** | Use 4 VP/Architect + 2 ML Engineer bullets as appropriate per role.

---

## VP Cloud Architect / Solutions Architect (4 bullets)

- **Architected** a production ML inference platform for ERP shipment delay risk prediction, deploying containerized FastAPI and Streamlit services with dual real-time and batch scoring — designed cloud-agnostically on GCP Cloud Run with a documented CAF-aligned migration path to **Azure Container Apps**, **APIM**, **Entra ID**, and **Azure Monitor**.

- **Designed** enterprise API governance layer using **Azure API Management** for rate limiting, OAuth authentication, and API versioning — demonstrating that even single-API workloads benefit from governance when integrating with ERP operational workflows.

- **Defined** Zero Trust authentication architecture replacing environment-variable secrets with **Entra ID managed identity** for service-to-service communication and **Azure Key Vault** for secrets management — aligning ML deployment with enterprise security posture.

- **Established** MLOps governance patterns including KS-test drift monitoring, versioned model artifacts in **Azure Blob Storage**, and custom metric alerting via **Azure Monitor** — closing the loop between ML deployment and operational awareness in manufacturing-grade SLA environments.

---

## AI Engineer / ML Engineer (2 bullets)

- **Engineered** domain-informed feature pipeline for ERP delay risk classification (lead time variance, ATP coverage, supplier reliability, customer backlog) using scikit-learn RandomForest — achieving production-ready inference with feature alignment guarantees between training and real-time/batch scoring endpoints.

- **Built** end-to-end ML scoring workflow: Pydantic-validated FastAPI inference API, CLI batch scoring pipeline, drift detection module (Kolmogorov-Smirnov tests via scipy), and executive dashboard with plain-English risk drivers — evaluated on operational reliability and interpretability, not just F1/AUC metrics.

---

## Usage Notes

- VP bullets lead with architecture verbs (Architected, Designed, Defined, Established)
- ML bullets lead with engineering verbs (Engineered, Built)
- GCP → Azure bridge framing is embedded in bullet 1 — preempts the platform question
- Operational metrics (98% inventory accuracy, 95% fill rate) belong in cover letter or case study, not resume bullets — keep bullets outcome-oriented and scannable
