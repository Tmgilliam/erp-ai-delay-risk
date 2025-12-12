from pathlib import Path
from typing import List
from typing import List


import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ERP AI Delay Risk API")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "delay_model.pkl"

# load model + training columns
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
model_cols = bundle["columns"]


class OrderPayload(BaseModel):
    order_id: str
    customer_id: str
    item_id: str
    plant: str
    order_date: str
    requested_ship_date: str
    promised_ship_date: str
    order_priority: int
    order_qty: int
    current_available_qty: int
    historical_lead_time_days: float
    supplier_reliability_score: float
    num_open_orders_customer: int
    past_due_invoices_flag: int
    weekday_ordered: int
    month_ordered: int


@app.get("/")
def root():
    return {"status": "ok", "message": "ERP Delay Risk API is running"}


@app.post("/score_order")
def score_order(order: OrderPayload):
    # 1) Turn payload into a one-row DataFrame
    df = pd.DataFrame([order.dict()])

    # 2) One-hot encode like we did at training time
    df = pd.get_dummies(df)

    # 3) Add any missing training columns and align order
    for col in model_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[model_cols]

    # 4) Predict with the trained model
    proba = model.predict_proba(df)[0][1]   # probability of late (class 1)
    pred = int(model.predict(df)[0])

    # 5) Return a JSON response
    return {
        "order_id": order.order_id,
        "late_flag_pred": pred,
        "late_probability": round(float(proba), 4)
    }
@app.post("/batch_score")
def batch_score(orders: List[OrderPayload]):
    """
    Score multiple orders in one call.
    Accepts a JSON array of OrderPayload objects.
    Returns per-order predictions plus summary stats.
    """
    if not orders:
        return {"n_orders": 0, "late_count": 0, "results": []}

    # 1) Turn list of payloads into DataFrame
    df = pd.DataFrame([o.dict() for o in orders])

    # 2) One-hot encode like training
    df = pd.get_dummies(df)

    # 3) Add any missing training columns and align order
    for col in model_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[model_cols]

    # 4) Predict with the trained model
    probs = model.predict_proba(df)[:, 1]
    preds = model.predict(df)

    results = []
    for order_obj, pred, prob in zip(orders, preds, probs):
        results.append(
            {
                "order_id": order_obj.order_id,
                "late_flag_pred": int(pred),
                "late_probability": round(float(prob), 4),
            }
        )

    late_count = int((preds == 1).sum())

    return {
        "n_orders": len(orders),
        "late_count": late_count,
        "results": results,
    }
@app.post("/batch_score")
def batch_score(orders: List[OrderPayload]):
    """
    Score multiple orders in one call.
    Accepts a JSON array of OrderPayload objects.
    Returns per-order predictions plus summary stats.
    """
    if not orders:
        return {"n_orders": 0, "late_count": 0, "results": []}

    # Payloads -> DataFrame
    df = pd.DataFrame([o.dict() for o in orders])

    # One-hot encode
    df = pd.get_dummies(df)

    # Align with training columns
    for col in model_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[model_cols]

    # Predict
    probs = model.predict_proba(df)[:, 1]
    preds = model.predict(df)

    results = []
    for order_obj, pred, prob in zip(orders, preds, probs):
        results.append(
            {
                "order_id": order_obj.order_id,
                "late_flag_pred": int(pred),
                "late_probability": round(float(prob), 4),
            }
        )

    late_count = int((preds == 1).sum())

    return {
        "n_orders": len(orders),
        "late_count": late_count,
        "results": results,
    }
