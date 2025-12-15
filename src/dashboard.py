import os
import hmac
import time
from datetime import date, timedelta
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
import streamlit as st


# ============================================================
# Page Config (MUST be first Streamlit call)
# ============================================================
APP_TITLE = "📦 ERP AI – Delay Risk Dashboard"
st.set_page_config(page_title=APP_TITLE, layout="wide")


# ============================================================
# Configuration
# ============================================================
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001").rstrip("/")

# Dashboard auth env vars (Cloud Run / local env)
DASH_USER = os.getenv("DASH_USER", "").strip().lower()
DASH_PASS = os.getenv("DASH_PASS", "")
DASH_ROLE = os.getenv("DASH_ROLE", "EXEC").strip().upper()  # EXEC / OPS / ANALYST

# Optional: set to "1" to show extra debug UI
DEBUG_UI = os.getenv("DEBUG_UI", "0") == "1"


# ============================================================
# Auth (Level 1: simple login gate)
# ============================================================
def _constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))

def require_login() -> None:
    """Fail-closed auth gate."""
    if not DASH_USER or not DASH_PASS:
        st.error("Auth is not configured. Set DASH_USER and DASH_PASS as environment variables.")
        st.stop()

    if "authed" not in st.session_state:
        st.session_state.authed = False
        st.session_state.user = None
        st.session_state.role = None

    if st.session_state.authed:
        return

    st.title("🔐 Sign in")
    st.caption("Authorized access only.")

    with st.form("login"):
        email = st.text_input("Email").strip().lower()
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if _constant_time_equals(email, DASH_USER) and _constant_time_equals(password, DASH_PASS):
            st.session_state.authed = True
            st.session_state.user = email
            st.session_state.role = DASH_ROLE
            st.success("Login successful.")
            st.rerun()
        else:
            st.error("Invalid email or password.")
            st.stop()

require_login()


# ============================================================
# UI Header
# ============================================================
st.title("📦 ERP AI – Delay Risk Dashboard")
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

with st.expander("🔧 Debug", expanded=False):
    st.write("API_URL:", API_URL)
    st.write("DASH_USER set?:", bool(DASH_USER))
    st.write("DASH_PASS set?:", bool(DASH_PASS))
    st.write("Role:", DASH_ROLE)
    st.write("Running from:", __file__)


# ============================================================
# Latency instrumentation (client-side)
# ============================================================
if "latency_ms" not in st.session_state:
    st.session_state.latency_ms = []  # list of floats
if "last_latency_ms" not in st.session_state:
    st.session_state.last_latency_ms = None

def _record_latency(ms: float) -> None:
    st.session_state.last_latency_ms = ms
    st.session_state.latency_ms.append(ms)
    # keep last 200 points
    if len(st.session_state.latency_ms) > 200:
        st.session_state.latency_ms = st.session_state.latency_ms[-200:]

def _latency_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"p50": None, "p95": None, "avg": None}
    s = pd.Series(values)
    return {
        "p50": float(s.quantile(0.50)),
        "p95": float(s.quantile(0.95)),
        "avg": float(s.mean()),
    }


# ============================================================
# API helpers
# ============================================================
def score_single(payload: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    r = requests.post(f"{API_URL}/score_order", json=payload, timeout=30)
    ms = (time.perf_counter() - t0) * 1000
    _record_latency(ms)
    r.raise_for_status()
    return r.json()

def score_batch(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    r = requests.post(f"{API_URL}/batch_score", json=records, timeout=60)
    ms = (time.perf_counter() - t0) * 1000
    _record_latency(ms)
    r.raise_for_status()
    return r.json()

def risk_label(prob: float) -> str:
    if prob >= 0.70:
        return "🔴 HIGH"
    if prob >= 0.40:
        return "🟠 MEDIUM"
    return "🟢 LOW"


# ============================================================
# Role-based navigation
# ============================================================
role = (st.session_state.role or "EXEC").upper()
tabs = ["📄 Single Order", "📊 Batch Scoring", "📈 KPIs & Charts"]

# OPS users may not need KPI tab (but we’ll leave it visible unless you want strict RBAC)
tab1, tab2, tab3 = st.tabs(tabs)


# ============================================================
# TAB 1: Single Order Scoring
# ============================================================
with tab1:
    st.subheader("Score a Single ERP Order")

    with st.form("single_order_form"):
        # --- ERP identifiers ---
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

        # --- Dates ---
        d1, d2, d3 = st.columns(3)
        with d1:
            order_date = st.date_input("Order Date", value=date.today())
        with d2:
            requested_ship_date = st.date_input("Requested Ship Date", value=date.today() + timedelta(days=2))
        with d3:
            promised_ship_date = st.date_input("Promised Ship Date", value=date.today() + timedelta(days=4))

        m1, m2, m3 = st.columns(3)
        with m1:
            order_priority = st.selectbox("Order Priority (1=highest)", [1, 2, 3, 4, 5], index=2)
        with m2:
            past_due_invoices_flag = st.selectbox("Past Due Invoices?", [0, 1], index=0)
        with m3:
            num_open_orders_customer = st.number_input("Open Customer Orders", min_value=0, max_value=500, value=22)

        st.divider()

        # --- Operational drivers ---
        col1, col2 = st.columns(2)
        with col1:
            order_qty = st.number_input("Order Quantity", min_value=1, max_value=100000, value=150)
            current_available_qty = st.number_input("Available Quantity", min_value=0, max_value=100000, value=80)
            historical_lead_time_days = st.number_input("Historical Lead Time (days)", min_value=0.0, max_value=180.0, value=4.5)
        with col2:
            supplier_reliability_score = st.slider("Supplier Reliability", 0.0, 1.0, 0.87)
            weekday_ordered = st.selectbox("Weekday Ordered (0=Mon..6=Sun)", [0,1,2,3,4,5,6], index=int(order_date.weekday()))
            month_ordered = st.selectbox("Month Ordered", list(range(1, 13)), index=int(order_date.month)-1)

        submit = st.form_submit_button("🚀 Score Order")

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

        with st.expander("📦 Payload sent to API", expanded=False):
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
            k4.metric("Last API Latency", f"{st.session_state.last_latency_ms:.0f} ms")

        except requests.HTTPError as e:
            st.error(f"API error: {e}")
            try:
                st.code(e.response.text)
            except Exception:
                pass


# ============================================================
# TAB 2: Batch scoring + export + summary
# ============================================================
with tab2:
    st.subheader("Batch Score Orders (CSV Upload)")

    st.info("CSV must include ALL OrderPayload columns. If not, FastAPI will return a 422.")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write("Input Preview")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Score Batch"):
            records = df.to_dict(orient="records")

            try:
                with st.spinner("Scoring batch…"):
                    result = score_batch(records)

                # API returns: { n_orders, late_count, results:[{order_id, late_flag_pred, late_probability}] }
                n_orders = int(result["n_orders"])
                late_count = int(result["late_count"])
                results_df = pd.DataFrame(result["results"])

                # Join predictions back to input for KPI calculations / drill-down
                merged = df.copy()
                merged = merged.merge(results_df, on="order_id", how="left")

                st.success(f"Batch complete: {late_count} late out of {n_orders} ({(late_count/max(n_orders,1)):.1%})")
                st.dataframe(merged, use_container_width=True)

                # Export
                st.download_button(
                    "⬇️ Download scored CSV",
                    data=merged.to_csv(index=False).encode("utf-8"),
                    file_name="scored_orders.csv",
                    mime="text/csv",
                )

                # Store for KPI tab
                st.session_state["last_batch_scored"] = merged

            except requests.HTTPError as e:
                st.error(f"API error: {e}")
                try:
                    st.code(e.response.text)
                except Exception:
                    pass


# ============================================================
# TAB 3: ERP KPIs + Charts + Latency metrics
# ============================================================
with tab3:
    st.subheader("ERP KPIs, Charts, and Latency")

    # Latency block (client-side)
    lat_stats = _latency_stats(st.session_state.latency_ms)
    a, b, c, d = st.columns(4)
    a.metric("Latency p50", "-" if lat_stats["p50"] is None else f"{lat_stats['p50']:.0f} ms")
    b.metric("Latency p95", "-" if lat_stats["p95"] is None else f"{lat_stats['p95']:.0f} ms")
    c.metric("Latency Avg", "-" if lat_stats["avg"] is None else f"{lat_stats['avg']:.0f} ms")
    d.metric("Last Call", "-" if st.session_state.last_latency_ms is None else f"{st.session_state.last_latency_ms:.0f} ms")

    if st.session_state.latency_ms:
        st.line_chart(pd.Series(st.session_state.latency_ms, name="latency_ms"))

    st.divider()

    batch = st.session_state.get("last_batch_scored")
    if batch is None:
        st.warning("Run a batch score first to populate KPIs and charts.")
        st.stop()

    # ---- ERP-style KPIs ----
    # On-time % is approximated as (1 - late_probability thresholded or late_flag_pred)
    total = len(batch)
    late_count = int((batch["late_flag_pred"] == 1).sum()) if "late_flag_pred" in batch.columns else 0
    late_rate = late_count / max(total, 1)
    on_time_rate = 1.0 - late_rate

    avg_lead = float(batch["historical_lead_time_days"].mean()) if "historical_lead_time_days" in batch.columns else float("nan")
    avg_supplier_rel = float(batch["supplier_reliability_score"].mean()) if "supplier_reliability_score" in batch.columns else float("nan")

    # ATP ratio (Available-to-Promise proxy)
    if {"current_available_qty", "order_qty"}.issubset(set(batch.columns)):
        atp_ratio = float((batch["current_available_qty"] / batch["order_qty"].clip(lower=1)).clip(upper=1).mean())
    else:
        atp_ratio = float("nan")

    # Backlog proxy (open customer orders)
    backlog_avg = float(batch["num_open_orders_customer"].mean()) if "num_open_orders_customer" in batch.columns else float("nan")

    # Finance risk proxy (past due invoices rate)
    past_due_rate = float(batch["past_due_invoices_flag"].mean()) if "past_due_invoices_flag" in batch.columns else float("nan")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Orders Scored", f"{total}")
    k2.metric("On-Time %", f"{on_time_rate:.1%}")
    k3.metric("Late %", f"{late_rate:.1%}")
    k4.metric("Avg Lead Time (days)", "-" if pd.isna(avg_lead) else f"{avg_lead:.2f}")
    k5.metric("ATP Ratio (avg)", "-" if pd.isna(atp_ratio) else f"{atp_ratio:.2f}")
    k6.metric("Past Due Rate", "-" if pd.isna(past_due_rate) else f"{past_due_rate:.1%}")

    st.divider()

    # ---- Charts ----
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Risk Distribution (late_probability)**")
        if "late_probability" in batch.columns:
            st.bar_chart(batch["late_probability"])
        else:
            st.info("No late_probability column found. (Batch should include it from API results.)")

    with c2:
        st.markdown("**Late Probability Histogram**")
        if "late_probability" in batch.columns:
            hist = pd.cut(batch["late_probability"], bins=[0, .2, .4, .6, .8, 1.0]).value_counts().sort_index()
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
