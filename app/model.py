"""Model loading and inference for ERP delay risk prediction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

from app.azure_model_loader import resolve_model_path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "delay_model.pkl"


class DelayRiskModel:
    """Wrapper around the trained scikit-learn delay risk classifier."""

    def __init__(self, model_path: Path | None = None) -> None:
        """Load model bundle from local path or Azure Blob Storage at startup."""
        resolved_path = model_path or resolve_model_path(DEFAULT_MODEL_PATH)
        bundle = joblib.load(resolved_path)
        self.model = bundle["model"]
        self.model_cols: List[str] = bundle["columns"]

    def _prepare_features(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Align incoming ERP payloads with training-time feature columns."""
        df = pd.DataFrame(records)
        df = pd.get_dummies(df)

        missing_cols = [col for col in self.model_cols if col not in df.columns]
        if missing_cols:
            filler = pd.DataFrame(0, index=df.index, columns=missing_cols)
            df = pd.concat([df, filler], axis=1)

        return df[self.model_cols]

    def predict_one(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single order and return probability + binary flag."""
        df = self._prepare_features([order])
        proba = float(self.model.predict_proba(df)[0][1])
        pred = int(self.model.predict(df)[0])

        return {
            "order_id": order["order_id"],
            "late_flag_pred": pred,
            "late_probability": round(proba, 4),
        }

    def predict_batch(self, orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Score multiple orders in one inference pass."""
        if not orders:
            return {"n_orders": 0, "late_count": 0, "results": []}

        df = self._prepare_features(orders)
        probs = self.model.predict_proba(df)[:, 1]
        preds = self.model.predict(df)

        results = []
        for order_obj, pred, prob in zip(orders, preds, probs):
            results.append(
                {
                    "order_id": order_obj["order_id"],
                    "late_flag_pred": int(pred),
                    "late_probability": round(float(prob), 4),
                }
            )

        return {
            "n_orders": len(orders),
            "late_count": int((preds == 1).sum()),
            "results": results,
        }

    def feature_importances(self) -> List[Dict[str, Any]]:
        """Return ranked feature importances from the underlying model."""
        if not hasattr(self.model, "feature_importances_"):
            return []

        pairs = sorted(
            zip(self.model_cols, self.model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            {"feature": name, "importance": float(score)}
            for name, score in pairs
        ]


# Module-level singleton loaded once at import for FastAPI startup.
delay_model = DelayRiskModel()
