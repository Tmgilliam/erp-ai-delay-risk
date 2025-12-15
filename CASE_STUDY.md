# Case Study: ERP AI Delay Risk Microservice

## Business Context
Late shipments are a persistent operational problem in manufacturing and supply chain environments.
They drive customer dissatisfaction, increase expediting costs, and create downstream planning risk.

This project demonstrates how machine learning can be embedded into an ERP-style workflow to
predict shipment delay risk early, using real-time order data and a production-ready API.

---

## Business Problem
ERP systems contain rich operational data, but delay risk is often identified too late:
- After the order is already past due
- After production or procurement options are limited

The goal was to build a system that:
- Scores open ERP orders for delay risk *before* shipment
- Integrates cleanly with ERP-style payloads
- Is deployable as a standalone service

---

## Solution
I built a containerized machine learning microservice that:
- Accepts ERP order payloads as JSON
- Applies a trained ML model to predict delay risk
- Returns both a binary risk flag and a probability score
- Exposes OpenAPI/Swagger documentation for testing and integration

---

## Technical Architecture

ERP Order Payload (JSON)
→ FastAPI (validation + routing)
→ Feature engineering & alignment
→ Trained ML model (RandomForest)
→ Delay probability + classification
→ REST API response

The entire service is packaged using Docker to ensure portability and consistent execution.

---

## Machine Learning Approach
- Synthetic ERP-style data generated for training and testing
- Supervised classification using scikit-learn
- Focus on feature consistency between training and inference
- Model artifacts persisted and loaded at runtime

The emphasis was not just model accuracy, but **operational reliability**.

---

## Engineering & Deployment
- FastAPI for inference and schema validation
- Swagger/OpenAPI for self-documenting endpoints
- Dockerized microservice for portable deployment
- Designed to run locally or in cloud container platforms

---

## Outcomes
- Fully functional ERP-style AI scoring service
- Real-time prediction via REST API
- Containerized, cloud-ready artifact
- Foundation for dashboards, alerts, and ERP integration

---

## Next Enhancements
- Streamlit executive dashboard for visual risk review
- Batch scoring for open order books
- Cloud deployment (Cloud Run / Azure Container Apps)
- Explainability (feature importance / SHAP)

---

## Key Takeaway
This project demonstrates how machine learning can be applied pragmatically to ERP operations,
bridging business risk, data science, and production engineering.
