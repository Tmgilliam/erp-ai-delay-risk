from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def generate_orders(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Core IDs & basic fields ---
    base_date = pd.Timestamp("2024-01-01")
    order_dates = base_date + pd.to_timedelta(
        rng.integers(0, 365, size=n),
        unit="D",
    )

    customer_ids = [f"C{rng.integers(1, 501):04d}" for _ in range(n)]
    item_ids = [f"ITEM{rng.integers(1, 201):04d}" for _ in range(n)]
    plants = rng.choice(["PLANT_A", "PLANT_B", "PLANT_C"], size=n, p=[0.5, 0.3, 0.2])

    # Priority: 1=Expedite, 2=Normal, 3=Low
    order_priority = rng.choice([1, 2, 3], size=n, p=[0.2, 0.5, 0.3])

    # Lead times and dates
    historical_lead_time = rng.normal(loc=7, scale=2, size=n).clip(1, 21)
    requested_lead = historical_lead_time + rng.normal(loc=0, scale=1.5, size=n)
    requested_lead = np.clip(np.round(requested_lead), 1, 30)

    requested_ship_date = order_dates + pd.to_timedelta(requested_lead, unit="D")
    promised_ship_date = order_dates + pd.to_timedelta(
        np.round(historical_lead_time),
        unit="D",
    )

    # Quantities and supply situation
    order_qty = rng.integers(1, 500, size=n)
    current_available_qty = order_qty + rng.integers(-200, 300, size=n)
    current_available_qty = np.clip(current_available_qty, 0, None)

    supplier_reliability_score = rng.normal(0.9, 0.08, size=n).clip(0.4, 1.0)

    num_open_orders_customer = rng.integers(1, 20, size=n)
    past_due_invoices_flag = rng.choice([0, 1], size=n, p=[0.7, 0.3])

    weekday_ordered = order_dates.weekday
    month_ordered = order_dates.month

    # --- Generate late_flag based on risk drivers ---
    base = -1.0
    prob = (
        base
        + 0.8 * (order_priority == 3)  # low priority
        + 0.5 * (current_available_qty < order_qty)  # not enough stock
        + 1.0 * (supplier_reliability_score < 0.8)  # shaky supplier
        + 0.6 * past_due_invoices_flag
        + 0.4 * (requested_lead < historical_lead_time)  # customer asking too fast
    )
    prob = 1 / (1 + np.exp(-prob))  # logistic

    late_flag = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "order_id": [f"O{100000 + i}" for i in range(n)],
            "customer_id": customer_ids,
            "item_id": item_ids,
            "plant": plants,
            "order_date": order_dates,
            "requested_ship_date": requested_ship_date,
            "promised_ship_date": promised_ship_date,
            "order_priority": order_priority,
            "order_qty": order_qty,
            "current_available_qty": current_available_qty,
            "historical_lead_time_days": np.round(historical_lead_time, 1),
            "supplier_reliability_score": np.round(supplier_reliability_score, 2),
            "num_open_orders_customer": num_open_orders_customer,
            "past_due_invoices_flag": past_due_invoices_flag,
            "weekday_ordered": weekday_ordered,
            "month_ordered": month_ordered,
            "late_flag": late_flag,
        }
    )

    return df


if __name__ == "__main__":
    # Training data with label
    train_df = generate_orders(n=5000, seed=42)
    train_path = DATA_DIR / "open_orders_train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"Wrote training data → {train_path}")

    # Scoring sample (no late_flag, like real open orders)
    scoring_df = train_df.sample(200, random_state=123).drop(columns=["late_flag"])
    scoring_path = DATA_DIR / "open_orders_scoring_sample.csv"
    scoring_df.to_csv(scoring_path, index=False)
    print(f"Wrote scoring sample → {scoring_path}")
