"""
Tests for src/api.py

Requires a trained model at models/delay_model.pkl.
Run `python src/train_model.py` first if the model is missing.

Run with:  pytest tests/test_api.py -v
"""

import pytest

# Skip the entire module if the model file is absent (CI without artefacts)
from pathlib import Path
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "delay_model.pkl"
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="models/delay_model.pkl not found — run src/train_model.py first",
)

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

VALID_ORDER = {
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


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_root_health():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /score_order
# ---------------------------------------------------------------------------

def test_score_order_returns_200():
    resp = client.post("/score_order", json=VALID_ORDER)
    assert resp.status_code == 200


def test_score_order_response_shape():
    resp = client.post("/score_order", json=VALID_ORDER)
    body = resp.json()
    assert "order_id" in body
    assert "late_flag_pred" in body
    assert "late_probability" in body


def test_score_order_probability_in_range():
    resp = client.post("/score_order", json=VALID_ORDER)
    prob = resp.json()["late_probability"]
    assert 0.0 <= prob <= 1.0


def test_score_order_flag_is_binary():
    resp = client.post("/score_order", json=VALID_ORDER)
    flag = resp.json()["late_flag_pred"]
    assert flag in (0, 1)


def test_score_order_preserves_order_id():
    resp = client.post("/score_order", json=VALID_ORDER)
    assert resp.json()["order_id"] == "TEST001"


def test_score_order_missing_field_returns_422():
    incomplete = {k: v for k, v in VALID_ORDER.items() if k != "order_qty"}
    resp = client.post("/score_order", json=incomplete)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /batch_score
# ---------------------------------------------------------------------------

def test_batch_score_empty_list():
    resp = client.post("/batch_score", json=[])
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_orders"] == 0
    assert body["late_count"] == 0
    assert body["results"] == []


def test_batch_score_single_order():
    resp = client.post("/batch_score", json=[VALID_ORDER])
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_orders"] == 1
    assert len(body["results"]) == 1


def test_batch_score_multiple_orders():
    order2 = {**VALID_ORDER, "order_id": "TEST002", "past_due_invoices_flag": 1,
               "current_available_qty": 10, "order_priority": 3}
    resp = client.post("/batch_score", json=[VALID_ORDER, order2])
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_orders"] == 2
    assert 0 <= body["late_count"] <= 2


def test_batch_score_late_count_consistent():
    order2 = {**VALID_ORDER, "order_id": "TEST002"}
    resp = client.post("/batch_score", json=[VALID_ORDER, order2])
    body = resp.json()
    computed = sum(r["late_flag_pred"] for r in body["results"])
    assert computed == body["late_count"]


def test_batch_score_probabilities_in_range():
    resp = client.post("/batch_score", json=[VALID_ORDER])
    for r in resp.json()["results"]:
        assert 0.0 <= r["late_probability"] <= 1.0
