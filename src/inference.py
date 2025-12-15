from pathlib import Path
import json

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

MODEL_PATH = MODELS_DIR / "delay_model.pkl"
FEATURE_COLS_PATH = MODELS_DIR / "feature_columns.json"

# Load model & metadata once at import
model = joblib.load(MODEL_PATH)
feature_cols = json.loads(FEATURE_COLS_PATH.read_text())


def score_order(order_payload: dict) -> dict:
    """
    order_payload: dict with same keys as training features.
    Returns: { "late_probability": float, "late_flag": int }
    """
    df = pd.DataFrame([order_payload])

    # Ensure same columns as during training
    for col in feature_cols:
        if col not in df.columns:
            df[col] = None

    df = df[feature_cols]

    proba = model.predict_proba(df)[:, 1][0]
    flag = int(proba >= 0.5)

    return {
        "late_probability": float(proba),
        "late_flag": flag,
    }
