import os
from datetime import date, timedelta

import requests
import pandas as pd
import streamlit as st

# =========================
# Configuration
# =========================
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

st.set_page_config(page_title="ERP AI – Delay Risk Dashboard", layout="wide")
st.title("📦 ERP AI – Delay Risk Dashboard")
st.markdown("Executive view of shipment delay risk using a machine learning microservice.")

with st.expander("🔧 Debug"):
    st.write("Running from:", __file__)
    st.write("API_URL:", API_URL)

# =========================
# Helper Functions
# =========================
def score_single(payload: dict):
    r = requests.post(f"{API_URL}/score_order", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def score_batch(records: list[dict]):
    r = requests.post(f"{API_URL}/batch_score", json=records, timeout=60)
    r.raise_for_status()
    return r.json()

def risk_label(prob: float):
    if prob >= 0.70:
        return "🔴 HIGH"
    elif prob >= 0.40:
        return "🟠 MEDIUM"
    return "🟢 LOW"

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["📄 Single Order", "📊 Batch Scoring (CSV)"])

# =========================
# Single Order Tab
# =========================
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

        # --- Dates / priority / finance flags ---
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

        # --- Operational drivers (model features) ---
        col1, col2 = st.columns(2)
        with col1:
            order_qty = st.number_input("Order Quantity", min_value=1, max_value=10000, value=150)
            current_available_qty = st.number_input("Available Quantity", min_value=0, max_value=100000, value=80)
        with col2:
            historical_lead_time_days = st.number_input("Historical Lead Time (days)", min_value=0.0, max_value=60.0, value=4.5)
            supplier_reliability_score = st.slider("Supplier Reliability", 0.0, 1.0, 0.87)

        submit = st.form_submit_button("🚀 Score Order")

    if submit:
        weekday_ordered = int(order_date.weekday())  # 0=Mon .. 6=Sun
        month_ordered = int(order_date.month)

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
            "weekday_ordered": weekday_ordered,
            "month_ordered": month_ordered,
        }

        with st.expander("📦 Payload sent to API"):
            st.json(payload)

        try:
            with st.spinner("Scoring order…"):
                result = score_single(payload)

            # API returns late_probability + late_flag_pred
            prob = float(result["late_probability"])
            late_flag = int(result["late_flag_pred"])

            k1, k2, k3 = st.columns(3)
            k1.metric("Delay Risk", f"{prob:.2%}", risk_label(prob))
            k2.metric("Predicted Late Flag", f"{late_flag}")
            k3.metric("Order Priority", f"{order_priority}")

        except requests.HTTPError as e:
            st.error(f"API error: {e}")
            # show response body if present
            try:
                st.code(e.response.text)
            except Exception:
                pass

# =========================
# Batch Scoring Tab
# =========================
with tab2:
    st.subheader("Batch Score Orders (CSV Upload)")

    st.info(
        "CSV must include all OrderPayload columns (same as Single Order), "
        "or you'll get a 422."
    )

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write("Input Preview")
        st.dataframe(df.head())

        if st.button("🚀 Score Batch"):
            records = df.to_dict(orient="records")

            try:
                with st.spinner("Scoring batch…"):
                    result = score_batch(records)

                # Your API returns {"n_orders":..., "late_count":..., "results":[...]}
                st.success(f"Batch complete: {result['late_count']} late out of {result['n_orders']}")
                results_df = pd.DataFrame(result["results"])
                st.dataframe(results_df)

            except requests.HTTPError as e:
                st.error(f"API error: {e}")
                try:
                    st.code(e.response.text)
                except Exception:
                    pass
