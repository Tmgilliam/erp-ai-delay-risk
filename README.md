# ERP AI – Delay Risk Prediction System

Production-ready AI system for predicting shipment delay risk in ERP workflows.

## 🚀 Overview
This project implements a full-stack, cloud-deployed AI solution consisting of:
- A **FastAPI machine learning inference service**
- A **Streamlit executive dashboard**
- **Application-level authentication**
- **Batch and real-time scoring**
- Deployed on **Google Cloud Run**

Designed to mirror real-world ERP and supply chain decision systems.

---

## 🏗 Architecture

- **Model API**
  - FastAPI + Pydantic
  - Scikit-learn model
  - Schema validation
  - Deployed on Cloud Run

- **Dashboard**
  - Streamlit UI
  - Auth-protected access
  - ERP-style KPIs
  - Real-time scoring via REST API
  - Deployed on Cloud Run

---

## 🔐 Security
- Environment-variable based authentication
- Constant-time credential comparison
- Role-ready design (EXEC / OPS / ANALYST)

---

## ☁️ Cloud Stack
- Google Cloud Run
- Artifact Registry
- Docker
- REST APIs

---

## 📊 Features
- Single order scoring
- Batch CSV scoring
- Delay probability prediction
- ERP operational drivers
- Executive dashboard UX

---

## 🖼 Screenshots
![Dashboard](screenshots/dashboard.png)
![Login](screenshots/login.png)

---

## 🧠 Skills Demonstrated
- Machine Learning deployment
- API design
- Cloud-native architecture
- Secure dashboard development
- ERP / supply chain analytics
