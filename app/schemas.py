"""Pydantic schemas for ERP delay risk prediction API."""

from typing import List

from pydantic import BaseModel, Field


class OrderPayload(BaseModel):
    """ERP order payload matching training feature schema."""

    order_id: str
    customer_id: str
    item_id: str
    plant: str
    order_date: str
    requested_ship_date: str
    promised_ship_date: str
    order_priority: int = Field(ge=1, le=5)
    order_qty: int = Field(ge=1)
    current_available_qty: int = Field(ge=0)
    historical_lead_time_days: float = Field(ge=0.0)
    supplier_reliability_score: float = Field(ge=0.0, le=1.0)
    num_open_orders_customer: int = Field(ge=0)
    past_due_invoices_flag: int = Field(ge=0, le=1)
    weekday_ordered: int = Field(ge=0, le=6)
    month_ordered: int = Field(ge=1, le=12)


class PredictionResult(BaseModel):
    """Single-order prediction response."""

    order_id: str
    late_flag_pred: int
    late_probability: float


class BatchPredictionResponse(BaseModel):
    """Batch scoring response with summary statistics."""

    n_orders: int
    late_count: int
    results: List[PredictionResult]
