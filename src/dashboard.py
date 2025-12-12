import os
import requests
import pandas as pd
import streamlit as st
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")


# =========================
# Configuration
# =========================
API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(
    page_title="ERP AI – Delay Risk Dashboard",
    layout="wide",
)

st.title("📦 ERP AI – Delay Risk Dashboard")
st.markdown(
    "Executive view of shipment delay risk using a machine learning microservice."
)

# =========================
# Helper Functions
# =========================
def score_single(payload):
    response = requests.post(
        f"{API_URL}/score_order",
        json=payload,
        timeout=5
    )
    return response.json()


def score_batch(df):
    response = requests.post(
        f"{API_URL}/batch_score",
        json=df.to_dict(orient="records")
    )
    response.raise_for_status()
    return response.json()


def risk_label_and_color(prob: float):
    if prob >= 0.70:
        return "HIGH", "🔴"
    elif prob >= 0.40:
        return "MEDIUM", "🟠"
    else:
        return "LOW", "🟢"


# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["Single Order", "Batch Orders (CSV)"])

# =========================
# SINGLE ORDER TAB
# =========================
with tab1:
    st.subheader("Score a Single ERP Order")

    col1, col2 = st.columns(2)

    with col1:
        order_id = st.text_input("Order ID", "A1001")
        customer_id = st.text_input("Customer ID", "C555")
        item_id = st.text_input("Item ID", "P123")
        plant = st.text_input("Plant", "LA01")
        order_qty = st.number_input("Order Quantity", 1, 10000, 150)

    with col2:
        current_available_qty = st.number_input("Available Quantity", 0, 100000, 80)
        historical_lead_time_days = st.number_input("Historical Lead Time (days)", 0.0, 60.0, 4.5)
        supplier_reliability_score = st.number_input("Supplier Reliability", 0.0, 1.0, 0.87)
        num_open_orders_customer = st.number_input("Open Orders (Customer)", 0, 500, 22)
