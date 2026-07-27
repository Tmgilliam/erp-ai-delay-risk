"""
Tests for src/inference.py

Run with:  pytest tests/test_inference.py -v
"""

import pytest
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "delay_model.pkl"
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="models/delay_model.pkl not found — run src/train_model.py first",
)

from src.inference import score_order

BASE_PAYLOAD = {
    "order_id": "TEST001",
    "customer_id": "C0001",
    "item_id": "ITEM0001",
    "plant": "PLANT_A",
    "order_date": "2024-06-01",
    "requested_ship_date": "2024-06-05",
    "promised_ship_date": "2024-06-08",
    "order_priority": 2,
    "order_qty": 100,
    "current_available_qty": 80,
    "historical_lead_time_days": 5.0,
    "supplier_reliability_score": 0.85,
    "num_open_orders_customer": 10,
    "past_due_invoices_flag": 0,
    "weekday_ordered": 5,
    "month_ordered": 6,
}


def test_score_order_returns_dict():
    result = score_order(BASE_PAYLOAD)
    assert isinstance(result, dict)


def test_score_order_keys():
    result = score_order(BASE_PAYLOAD)
    assert "late_probability" in result
    assert "late_flag" in result


def test_probability_range():
    result = score_order(BASE_PAYLOAD)
    assert 0.0 <= result["late_probability"] <= 1.0


def test_flag_is_binary():
    result = score_order(BASE_PAYLOAD)
    assert result["late_flag"] in (0, 1)


def test_flag_matches_probability():
    result = score_order(BASE_PAYLOAD)
    expected_flag = int(result["late_probability"] >= 0.5)
    assert result["late_flag"] == expected_flag


def test_high_risk_order_scores_higher():
    """A low-priority, stock-out, unreliable-supplier order should score higher
    than the low-risk base order."""
    high_risk = {
        **BASE_PAYLOAD,
        "order_priority": 3,
        "current_available_qty": 0,
        "supplier_reliability_score": 0.45,
        "past_due_invoices_flag": 1,
    }
    low_risk_prob = score_order(BASE_PAYLOAD)["late_probability"]
    high_risk_prob = score_order(high_risk)["late_probability"]
    assert high_risk_prob > low_risk_prob


def test_unseen_customer_does_not_crash():
    """An order from a customer not in training data should still return a result."""
    unseen = {**BASE_PAYLOAD, "customer_id": "C_BRAND_NEW_99999"}
    result = score_order(unseen)
    assert 0.0 <= result["late_probability"] <= 1.0
