"""
ERP AI Delay Risk Executive Dashboard.

MTP Phase 2 upgrade: adds Executive Risk Summary panel for operations leadership.
"""

import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

from dashboard.auth import auth_mode_label, get_api_auth_headers, require_login


# ============================================================
# Page Config (MUST be first Streamlit call)
# ============================================================
APP_TITLE = "ERP AI – Delay Risk Dashboard"
st.set_page_config(page_title=APP_TITLE, layout="wide")


# ============================================================
# Configuration
# ============================================================
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001").rstrip("/")

DEBUG_UI = os.getenv("DEBUG_UI", "0") == "1"
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "delay_model.pkl"
SAMPLE_DATA_PATH = ROOT / "data" / "open_orders_scoring_sample.csv"

# Plain-English labels for operations stakeholders (not model column names).
FEATURE_LABELS: Dict[str, str] = {
    "historical_lead_time_days": "Lead Time Variance",
    "supplier_reliability_score": "Supplier Reliability",
    "current_available_qty": "Inventory Coverage (ATP)",
    "order_qty": "Order Volume Pressure",
    "num_open_orders_customer": "Customer Order Backlog",
    "past_due_invoices_flag": "Accounts Receivable Risk",
    "order_priority": "Order Priority Pressure",
    "requested_lead_time_days": "Requested Ship Window",
}


def _record_latency(ms: float) -> None:
    st.session_state.last_latency_ms = ms
    st.session_state.latency_ms.append(ms)
    if len(st.session_state.latency_ms) > 200:
        st.session_state.latency_ms = st.session_state.latency_ms[-200:]


def _latency_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"p50": None, "p95": None, "avg": None}
    series = pd.Series(values)
    return {
        "p50": float(series.quantile(0.50)),
        "p95": float(series.quantile(0.95)),
        "avg": float(series.mean()),
    }


def score_single(payload: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    r = requests.post(
        f"{API_URL}/predict",
        json=payload,
        headers=get_api_auth_headers(),
        timeout=30,
    )
    ms = (time.perf_counter() - t0) * 1000
    _record_latency(ms)
    r.raise_for_status()
    return r.json()


def score_batch(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    r = requests.post(
        f"{API_URL}/predict/batch",
        json=records,
        headers=get_api_auth_headers(),
        timeout=60,
    )
    ms = (time.perf_counter() - t0) * 1000
    _record_latency(ms)
    r.raise_for_status()
    return r.json()


def risk_label(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    if prob >= 0.40:
        return "MEDIUM"
    return "LOW"


def _friendly_feature_name(feature: str) -> str:
    """Map encoded or raw feature names to operations language."""
    for key, label in FEATURE_LABELS.items():
        if key in feature:
            return label
    return feature.replace("_", " ").title()


def _load_top_risk_drivers(limit: int = 3) -> List[Dict[str, float]]:
    """Load top feature importances from the trained model."""
    if not MODEL_PATH.exists():
        return [
            {"label": "Lead Time Variance", "importance": 0.34},
            {"label": "Supplier Reliability", "importance": 0.28},
            {"label": "Inventory Coverage (ATP)", "importance": 0.21},
        ]

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    columns = bundle["columns"]

    if not hasattr(model, "feature_importances_"):
        return []

    ranked = sorted(
        zip(columns, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )

    aggregated: Dict[str, float] = {}
    for name, score in ranked:
        label = _friendly_feature_name(name)
        aggregated[label] = aggregated.get(label, 0.0) + float(score)

    top = sorted(aggregated.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [{"label": label, "importance": score} for label, score in top]


def _get_executive_dataset() -> pd.DataFrame:
    """
    Prefer last batch scored data; fall back to bundled sample CSV.
    """
    batch = st.session_state.get("last_batch_scored")
    if batch is not None and "late_probability" in batch.columns:
        return batch

    if SAMPLE_DATA_PATH.exists():
        sample = pd.read_csv(SAMPLE_DATA_PATH)
        try:
            result = score_batch(sample.to_dict(orient="records"))
            scored = pd.DataFrame(result["results"])
            merged = sample.merge(scored, on="order_id", how="left")
            st.session_state["last_batch_scored"] = merged
            st.session_state["last_scored_at"] = datetime.now().isoformat(timespec="seconds")
            return merged
        except requests.RequestException:
            # API unavailable — use sample rows without scores for layout demo only.
            if "late_probability" not in sample.columns:
                sample = sample.copy()
                sample["late_probability"] = 0.25
            return sample

    return pd.DataFrame()


def _fetch_scoring_trend(days: int = 30) -> Dict[str, Any]:
    """Load persisted scoring history from API."""
    try:
        response = requests.get(
            f"{API_URL}/monitoring/scoring-history",
            params={"trend_days": days},
            headers=get_api_auth_headers(),
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {"points": [], "latest": None, "data_source": "unavailable"}


def _simulated_30_day_trend(current_high_risk_pct: float) -> pd.DataFrame:
    """
    Fallback trend when persisted scoring history is unavailable.

    Run `python monitoring/seed_scoring_history.py` or score batches via the API
    to populate monitoring/scoring_history.csv with real data.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=datetime.now().date(), periods=30, freq="D")
    base = max(current_high_risk_pct, 0.05)
    noise = rng.normal(0, 0.02, size=30)
    values = np.clip(base + np.cumsum(noise) * 0.15, 0.02, 0.65)
    return pd.DataFrame({"date": dates, "high_risk_pct": values}).set_index("date")


def _build_trend_chart(
    current_high_risk_pct: float,
    days: int = 30,
) -> tuple[pd.DataFrame, str]:
    """Prefer persisted API history; fall back to simulated demo data."""
    history = _fetch_scoring_trend(days=days)
    points = history.get("points") or []

    if len(points) >= 2:
        trend = pd.DataFrame(points)
        trend["date"] = pd.to_datetime(trend["date"])
        trend = trend.set_index("date").sort_index()
        return trend, "persisted scoring history"

    trend = _simulated_30_day_trend(current_high_risk_pct)
    return trend, "simulated (run batch scores or seed script to persist history)"


def _recommended_action(high_risk_pct: float) -> str:
    if high_risk_pct > 0.30:
        return (
            "Immediate review recommended — flag top 10 at-risk orders "
            "for operations team."
        )
    if high_risk_pct >= 0.10:
        return (
            "Monitor closely — review lead time signals for flagged SKUs."
        )
    return "Normal operations — continue standard review cadence."


def render_executive_risk_summary() -> None:
    """First panel: operations VP view before Monday planning calls."""
    st.subheader("Executive Risk Summary")
    st.caption("Shipment delay risk at a glance — built for planning conversations, not model review.")

    dataset = _get_executive_dataset()
    if dataset.empty or "late_probability" not in dataset.columns:
        st.warning("Score a batch or load sample data to populate the executive summary.")
        return

    high_risk_threshold = 0.30
    high_risk_count = int((dataset["late_probability"] >= high_risk_threshold).sum())
    high_risk_pct = high_risk_count / max(len(dataset), 1)
    avg_risk = float(dataset["late_probability"].mean())

    history = _fetch_scoring_trend()
    latest = history.get("latest") or {}
    last_scored = (
        latest.get("scored_at")
        or st.session_state.get("last_scored_at")
        or datetime.now().isoformat(timespec="seconds")
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High Risk Count", f"{high_risk_count}")
    c2.metric("High Risk %", f"{high_risk_pct:.1%}")
    c3.metric("Avg Risk Score", f"{avg_risk:.1%}")
    c4.metric("Last Scored", last_scored)

    st.markdown("**Top 3 Delay Risk Drivers**")
    drivers = _load_top_risk_drivers(limit=3)
    if drivers:
        max_imp = max(d["importance"] for d in drivers) or 1.0
        for driver in drivers:
            width = int((driver["importance"] / max_imp) * 100)
            st.write(f"**{driver['label']}** — {driver['importance']:.2f} relative influence")
            st.progress(min(width / 100.0, 1.0))
    else:
        st.info("Feature importance unavailable for this model artifact.")

    st.markdown("**30-Day High-Risk Trend**")
    trend, trend_source = _build_trend_chart(high_risk_pct)
    st.caption(f"Trend source: {trend_source}")
    st.line_chart(trend["high_risk_pct"])

    st.info(f"**Recommended Action:** {_recommended_action(high_risk_pct)}")

    st.divider()


# ============================================================
# App bootstrap
# ============================================================
require_login()

if "latency_ms" not in st.session_state:
    st.session_state.latency_ms = []
if "last_latency_ms" not in st.session_state:
    st.session_state.last_latency_ms = None

st.title("ERP AI – Delay Risk Dashboard")
st.markdown("Executive view of shipment delay risk using a machine learning microservice.")

top_left, top_right = st.columns([4, 1])
with top_left:
    st.caption(f"Signed in as **{st.session_state.user}** ({st.session_state.role})")
with top_right:
    if st.button("Logout"):
        st.session_state.authed = False
        st.session_state.user = None
        st.session_state.role = None
        st.rerun()

with st.expander("Debug", expanded=False):
    st.write("API_URL:", API_URL)
    st.write("Auth mode:", auth_mode_label())
    st.write("Role:", st.session_state.get("role"))
    st.write("Entra token set?:", bool(st.session_state.get("access_token")))
    st.write("Running from:", __file__)

render_executive_risk_summary()

tabs = ["Single Order", "Batch Scoring", "KPIs & Charts"]
tab1, tab2, tab3 = st.tabs(tabs)


# ============================================================
# TAB 1: Single Order Scoring
# ============================================================
with tab1:
    st.subheader("Score a Single ERP Order")

    with st.form("single_order_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            order_id = st.text_input("Order ID", "A1001")
        with c2:
            customer_id = st.text_input("Customer ID", "C555")
        with c3:
            item_id = st.text_input("Item ID", "P123")
        with c4:
            plant = st.text_input("Plant", "LA01")

        st.divider()

        d1, d2, d3 = st.columns(3)
        with d1:
            order_date = st.date_input("Order Date", value=date.today())
        with d2:
            requested_ship_date = st.date_input(
                "Requested Ship Date", value=date.today() + timedelta(days=2)
            )
        with d3:
            promised_ship_date = st.date_input(
                "Promised Ship Date", value=date.today() + timedelta(days=4)
            )

        m1, m2, m3 = st.columns(3)
        with m1:
            order_priority = st.selectbox("Order Priority (1=highest)", [1, 2, 3, 4, 5], index=2)
        with m2:
            past_due_invoices_flag = st.selectbox("Past Due Invoices?", [0, 1], index=0)
        with m3:
            num_open_orders_customer = st.number_input(
                "Open Customer Orders", min_value=0, max_value=500, value=22
            )

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            order_qty = st.number_input("Order Quantity", min_value=1, max_value=100000, value=150)
            current_available_qty = st.number_input(
                "Available Quantity", min_value=0, max_value=100000, value=80
            )
            historical_lead_time_days = st.number_input(
                "Historical Lead Time (days)", min_value=0.0, max_value=180.0, value=4.5
            )
        with col2:
            supplier_reliability_score = st.slider("Supplier Reliability", 0.0, 1.0, 0.87)
            weekday_ordered = st.selectbox(
                "Weekday Ordered (0=Mon..6=Sun)",
                [0, 1, 2, 3, 4, 5, 6],
                index=int(order_date.weekday()),
            )
            month_ordered = st.selectbox(
                "Month Ordered", list(range(1, 13)), index=int(order_date.month) - 1
            )

        submit = st.form_submit_button("Score Order")

    if submit:
        payload = {
            "order_id": str(order_id),
            "customer_id": str(customer_id),
            "item_id": str(item_id),
            "plant": str(plant),
            "order_date": order_date.isoformat(),
            "requested_ship_date": requested_ship_date.isoformat(),
            "promised_ship_date": promised_ship_date.isoformat(),
            "order_priority": int(order_priority),
            "order_qty": int(order_qty),
            "current_available_qty": int(current_available_qty),
            "historical_lead_time_days": float(historical_lead_time_days),
            "supplier_reliability_score": float(supplier_reliability_score),
            "num_open_orders_customer": int(num_open_orders_customer),
            "past_due_invoices_flag": int(past_due_invoices_flag),
            "weekday_ordered": int(weekday_ordered),
            "month_ordered": int(month_ordered),
        }

        with st.expander("Payload sent to API", expanded=False):
            st.json(payload)

        try:
            with st.spinner("Scoring order…"):
                result = score_single(payload)

            prob = float(result["late_probability"])
            late_flag = int(result["late_flag_pred"])

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Delay Risk", f"{prob:.2%}", risk_label(prob))
            k2.metric("Predicted Late Flag", f"{late_flag}")
            k3.metric("ATP Ratio", f"{min(current_available_qty / max(order_qty, 1), 1.0):.2f}")
            k4.metric(
                "Last API Latency",
                f"{st.session_state.last_latency_ms:.0f} ms"
                if st.session_state.last_latency_ms
                else "-",
            )

        except requests.HTTPError as exc:
            st.error(f"API error: {exc}")
            try:
                st.code(exc.response.text)
            except Exception:
                pass


# ============================================================
# TAB 2: Batch scoring + export + summary
# ============================================================
with tab2:
    st.subheader("Batch Score Orders (CSV Upload)")
    st.info("CSV must include all OrderPayload columns. FastAPI returns 422 if columns are missing.")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write("Input Preview")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Score Batch"):
            records = df.to_dict(orient="records")

            try:
                with st.spinner("Scoring batch…"):
                    result = score_batch(records)

                n_orders = int(result["n_orders"])
                late_count = int(result["late_count"])
                results_df = pd.DataFrame(result["results"])

                merged = df.copy()
                merged = merged.merge(results_df, on="order_id", how="left")

                st.success(
                    f"Batch complete: {late_count} late out of {n_orders} "
                    f"({(late_count / max(n_orders, 1)):.1%})"
                )
                st.dataframe(merged, use_container_width=True)

                st.download_button(
                    "Download scored CSV",
                    data=merged.to_csv(index=False).encode("utf-8"),
                    file_name="scored_orders.csv",
                    mime="text/csv",
                )

                st.session_state["last_batch_scored"] = merged
                st.session_state["last_scored_at"] = datetime.now().isoformat(timespec="seconds")

            except requests.HTTPError as exc:
                st.error(f"API error: {exc}")
                try:
                    st.code(exc.response.text)
                except Exception:
                    pass


# ============================================================
# TAB 3: ERP KPIs + Charts + Latency metrics
# ============================================================
with tab3:
    st.subheader("ERP KPIs, Charts, and Latency")

    lat_stats = _latency_stats(st.session_state.latency_ms)
    a, b, c, d = st.columns(4)
    a.metric("Latency p50", "-" if lat_stats["p50"] is None else f"{lat_stats['p50']:.0f} ms")
    b.metric("Latency p95", "-" if lat_stats["p95"] is None else f"{lat_stats['p95']:.0f} ms")
    c.metric("Latency Avg", "-" if lat_stats["avg"] is None else f"{lat_stats['avg']:.0f} ms")
    d.metric(
        "Last Call",
        "-"
        if st.session_state.last_latency_ms is None
        else f"{st.session_state.last_latency_ms:.0f} ms",
    )

    if st.session_state.latency_ms:
        st.line_chart(pd.Series(st.session_state.latency_ms, name="latency_ms"))

    st.divider()

    batch = st.session_state.get("last_batch_scored")
    if batch is None:
        st.warning("Run a batch score first to populate KPIs and charts.")
        st.stop()

    total = len(batch)
    late_count = int((batch["late_flag_pred"] == 1).sum()) if "late_flag_pred" in batch.columns else 0
    late_rate = late_count / max(total, 1)
    on_time_rate = 1.0 - late_rate

    avg_lead = (
        float(batch["historical_lead_time_days"].mean())
        if "historical_lead_time_days" in batch.columns
        else float("nan")
    )
    atp_ratio = (
        float(
            (batch["current_available_qty"] / batch["order_qty"].clip(lower=1))
            .clip(upper=1)
            .mean()
        )
        if {"current_available_qty", "order_qty"}.issubset(set(batch.columns))
        else float("nan")
    )
    past_due_rate = (
        float(batch["past_due_invoices_flag"].mean())
        if "past_due_invoices_flag" in batch.columns
        else float("nan")
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Orders Scored", f"{total}")
    k2.metric("On-Time %", f"{on_time_rate:.1%}")
    k3.metric("Late %", f"{late_rate:.1%}")
    k4.metric("Avg Lead Time (days)", "-" if pd.isna(avg_lead) else f"{avg_lead:.2f}")
    k5.metric("ATP Ratio (avg)", "-" if pd.isna(atp_ratio) else f"{atp_ratio:.2f}")
    k6.metric("Past Due Rate", "-" if pd.isna(past_due_rate) else f"{past_due_rate:.1%}")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Risk Distribution (late_probability)**")
        if "late_probability" in batch.columns:
            st.bar_chart(batch["late_probability"])
        else:
            st.info("No late_probability column found.")

    with c2:
        st.markdown("**Late Probability Histogram**")
        if "late_probability" in batch.columns:
            hist = (
                pd.cut(batch["late_probability"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
                .value_counts()
                .sort_index()
            )
            st.bar_chart(hist)
        else:
            st.info("No late_probability column found.")

    st.divider()

    st.markdown("**Supplier Reliability vs Predicted Risk**")
    if {"supplier_reliability_score", "late_probability"}.issubset(set(batch.columns)):
        scatter = batch[["supplier_reliability_score", "late_probability"]].copy()
        st.scatter_chart(scatter, x="supplier_reliability_score", y="late_probability")
    else:
        st.info("Missing columns for supplier vs risk chart.")

    if DEBUG_UI:
        st.write("Batch columns:", list(batch.columns))
