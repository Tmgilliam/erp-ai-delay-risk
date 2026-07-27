# Architecture — ERP AI Delay Risk

## Overview
A machine-learning microservice that predicts the probability of a purchase order arriving late. It consists of three loosely coupled layers: a data / training pipeline, a FastAPI scoring API, and a Streamlit operations dashboard.

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit Dashboard  (src/dashboard.py)                │
│  - Single-order scoring form                            │
│  - Batch CSV upload + scored export                     │
│  - KPI charts and latency metrics                       │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP (REST)
┌────────────────────▼────────────────────────────────────┐
│  FastAPI Scoring API  (src/api.py)                      │
│  POST /score_order    — score one order                 │
│  POST /batch_score    — score many orders at once       │
│  GET  /               — health check                    │
└────────────────────┬────────────────────────────────────┘
                     │ joblib.load()
┌────────────────────▼────────────────────────────────────┐
│  Model Bundle  (models/delay_model.pkl)                 │
│  {"model": RandomForestClassifier,                      │
│   "columns": [...training feature columns...]}          │
└────────────────────┬────────────────────────────────────┘
                     │ produced by
┌────────────────────▼────────────────────────────────────┐
│  Training Pipeline                                      │
│  src/generate_data.py  — synthetic order data (5 000)   │
│  src/train_model.py    — trains RF, saves bundle to     │
│                          models/delay_model.pkl         │
└─────────────────────────────────────────────────────────┘
```

## Components

### `src/generate_data.py`
Generates a synthetic dataset of 5 000 ERP orders with a logistic-regression-based label (`late_flag`). Risk drivers are: low order priority, insufficient stock, low supplier reliability, past-due invoices, and short requested lead time. Writes:
- `data/open_orders_train.csv` — labelled training set
- `data/open_orders_scoring_sample.csv` — unlabelled sample for manual testing

### `src/train_model.py`
Reads the training CSV, drops non-generalisable columns (`order_id`, raw date strings), one-hot encodes categoricals, trains a `RandomForestClassifier(n_estimators=300)`, evaluates on a held-out 20 % split, and saves the model bundle.

### `src/api.py`
FastAPI application. Both endpoints share a `_prepare()` helper that:
1. Drops identifier / date string columns.
2. One-hot encodes remaining categoricals via `pd.get_dummies()`.
3. Aligns the resulting frame to the training column set (adds missing columns as 0).

### `src/inference.py`
Standalone Python helper for use outside the HTTP API (scripts, notebooks). Loads the same bundle and exposes `score_order(dict) -> dict`.

### `src/dashboard.py`
Streamlit UI with simple email/password login (credentials from `DASH_USER` / `DASH_PASS` env vars). Three tabs: single-order scoring form, batch CSV upload with export, and KPI + latency charts.

### `src/erp_client.py`
Thin requests wrapper around the API for downstream service integration.

## Data Flow (scoring)
```
Client JSON payload
      │
      ▼
OrderPayload (Pydantic validation)
      │
      ▼
_prepare(): drop IDs + date strings -> get_dummies -> align to model_cols
      │
      ▼
RandomForestClassifier.predict_proba()
      │
      ▼
{order_id, late_flag_pred, late_probability}
```

## Deployment
Local development:
```bash
uvicorn src.api:app --reload --port 8001
streamlit run src/dashboard.py
```

Docker Compose:
```bash
docker-compose up --build
```

## Feature Columns
After dropping `order_id`, `order_date`, `requested_ship_date`, `promised_ship_date`:

| Column | Type | Notes |
|--------|------|-------|
| customer_id | categorical | one-hot encoded |
| item_id | categorical | one-hot encoded |
| plant | categorical | one-hot encoded |
| order_priority | int | 1 = expedite, 3 = low |
| order_qty | int | |
| current_available_qty | int | |
| historical_lead_time_days | float | |
| supplier_reliability_score | float | 0–1 |
| num_open_orders_customer | int | |
| past_due_invoices_flag | int | 0/1 |
| weekday_ordered | int | 0 = Mon, 6 = Sun |
| month_ordered | int | 1–12 |
