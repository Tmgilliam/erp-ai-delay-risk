from pathlib import Path
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

# Columns that are identifiers or raw date strings — dropped before encoding
# because they cannot generalise to unseen values at inference time.
# Time-based signals are already captured by weekday_ordered and month_ordered.
_DROP_BEFORE_ENCODE = {"order_id", "order_date", "requested_ship_date", "promised_ship_date"}


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


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Shared preprocessing: drop non-generalisable columns, one-hot encode,
    align to training column set."""
    df = df.drop(columns=[c for c in _DROP_BEFORE_ENCODE if c in df.columns])
    df = pd.get_dummies(df)
    for col in model_cols:
        if col not in df.columns:
            df[col] = 0
    return df[model_cols]


@app.get("/")
def root():
    return {"status": "ok", "message": "ERP Delay Risk API is running"}


@app.post("/score_order")
def score_order(order: OrderPayload):
    df = _prepare(pd.DataFrame([order.dict()]))

    proba = model.predict_proba(df)[0][1]
    pred = int(model.predict(df)[0])

    return {
        "order_id": order.order_id,
        "late_flag_pred": pred,
        "late_probability": round(float(proba), 4),
    }


@app.post("/batch_score")
def batch_score(orders: List[OrderPayload]):
    """Score multiple orders in one call.
    Accepts a JSON array of OrderPayload objects.
    Returns per-order predictions plus summary stats.
    """
    if not orders:
        return {"n_orders": 0, "late_count": 0, "results": []}

    df = _prepare(pd.DataFrame([o.dict() for o in orders]))

    probs = model.predict_proba(df)[:, 1]
    preds = model.predict(df)

    results = [
        {
            "order_id": o.order_id,
            "late_flag_pred": int(pred),
            "late_probability": round(float(prob), 4),
        }
        for o, pred, prob in zip(orders, preds, probs)
    ]

    return {
        "n_orders": len(orders),
        "late_count": int((preds == 1).sum()),
        "results": results,
    }
