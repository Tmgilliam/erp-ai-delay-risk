"""FastAPI application for ERP delay risk prediction and monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException

from app.auth.entra_jwt import require_entra_user
from app.model import delay_model
from app.schemas import BatchPredictionResponse, OrderPayload, PredictionResult
from monitoring.drift_monitor import DriftMonitor
from monitoring.scoring_history import ScoringHistory

app = FastAPI(
    title="ERP AI Delay Risk API",
    description="Production inference service for ERP shipment delay risk prediction.",
    version="2.0.0",
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_BASELINE_PATH = ROOT / "monitoring" / "reference_baseline.csv"
CURRENT_SAMPLE_PATH = ROOT / "data" / "open_orders_scoring_sample.csv"
SCORING_HISTORY_PATH = ROOT / "monitoring" / "scoring_history.csv"

_drift_monitor: DriftMonitor | None = None
_scoring_history: ScoringHistory | None = None


def get_scoring_history() -> ScoringHistory:
    """Lazy-load scoring history store."""
    global _scoring_history
    if _scoring_history is None:
        _scoring_history = ScoringHistory(SCORING_HISTORY_PATH)
    return _scoring_history


def _record_probabilities(probabilities: list[float], source: str) -> None:
    """Persist aggregate scoring metrics without blocking the API response."""
    if probabilities:
        get_scoring_history().record_run(probabilities=probabilities, source=source)


def get_drift_monitor() -> DriftMonitor:
    """Lazy-load drift monitor so API startup is not blocked on missing files."""
    global _drift_monitor
    if _drift_monitor is None:
        if not REFERENCE_BASELINE_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Reference baseline not found: {REFERENCE_BASELINE_PATH}",
            )
        _drift_monitor = DriftMonitor(str(REFERENCE_BASELINE_PATH))
    return _drift_monitor


@app.get("/health")
def health() -> dict:
    """Health check for load balancers and container orchestration."""
    return {"status": "healthy", "service": "erp-delay-risk-api"}


@app.get("/")
def root() -> dict:
    """Root endpoint for quick connectivity checks."""
    return {"status": "ok", "message": "ERP Delay Risk API is running"}


@app.post("/predict", response_model=PredictionResult)
def predict(
    order: OrderPayload,
    _claims: dict | None = Depends(require_entra_user),
) -> dict:
    """Real-time single-record delay risk scoring."""
    result = delay_model.predict_one(order.model_dump())
    _record_probabilities([result["late_probability"]], source="api_single")
    return result


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    orders: List[OrderPayload],
    _claims: dict | None = Depends(require_entra_user),
) -> dict:
    """Batch delay risk scoring for multiple ERP orders."""
    payloads = [order.model_dump() for order in orders]
    result = delay_model.predict_batch(payloads)
    probabilities = [row["late_probability"] for row in result["results"]]
    _record_probabilities(probabilities, source="api_batch")
    return result


@app.get("/monitoring/scoring-history")
def scoring_history(
    trend_days: int = 30,
    _claims: dict | None = Depends(require_entra_user),
) -> dict:
    """Return persisted scoring run trend for executive dashboard."""
    return get_scoring_history().get_summary(trend_days=trend_days)


# Legacy Phase 1 routes — preserved for backward compatibility with deployed services.
@app.post("/score_order", response_model=PredictionResult)
def score_order(order: OrderPayload) -> dict:
    """Legacy alias for /predict."""
    return predict(order)


@app.post("/batch_score", response_model=BatchPredictionResponse)
def batch_score(orders: List[OrderPayload]) -> dict:
    """Legacy alias for /predict/batch."""
    return predict_batch(orders)


@app.get("/monitoring/drift-report")
def drift_report(
    threshold: float = 0.05,
    _claims: dict | None = Depends(require_entra_user),
) -> dict:
    """
    Compare recent scoring sample distributions against training baseline.

    Uses Kolmogorov-Smirnov tests per numeric feature. Returns JSON suitable
    for downstream Azure Monitor custom metric ingestion.
    """
    monitor = get_drift_monitor()

    if not CURRENT_SAMPLE_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Current sample data not found: {CURRENT_SAMPLE_PATH}",
        )

    current_data = pd.read_csv(CURRENT_SAMPLE_PATH)
    drift_results = monitor.compute_drift(current_data)
    flags = monitor.flag_drift(drift_results, threshold=threshold)
    report = monitor.generate_report(drift_results, flags=flags, threshold=threshold)

    return report
