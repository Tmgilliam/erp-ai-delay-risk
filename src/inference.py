from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "delay_model.pkl"

# train_model.py saves a bundle: {"model": clf, "columns": [...]}
_bundle = joblib.load(MODEL_PATH)
model = _bundle["model"]
model_cols = _bundle["columns"]

# Columns that are identifiers or raw date strings — same list as api.py
_DROP_BEFORE_ENCODE = {"order_id", "order_date", "requested_ship_date", "promised_ship_date"}


def score_order(order_payload: dict) -> dict:
    """
    Score a single order dict.

    Parameters
    ----------
    order_payload : dict
        Keys matching OrderPayload fields (order_id, order_date, etc.).

    Returns
    -------
    dict with keys: late_probability (float), late_flag (int)
    """
    df = pd.DataFrame([order_payload])
    df = df.drop(columns=[c for c in _DROP_BEFORE_ENCODE if c in df.columns])
    df = pd.get_dummies(df)

    for col in model_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[model_cols]

    proba = float(model.predict_proba(df)[:, 1][0])
    return {
        "late_probability": proba,
        "late_flag": int(proba >= 0.5),
    }
