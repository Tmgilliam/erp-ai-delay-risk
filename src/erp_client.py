 import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_open_orders(filename: str = "open_orders_train.csv") -> pd.DataFrame:
    """
    Simulate pulling 'open orders' from ERP.

    For now this just loads a CSV from the data/ folder.
    In a real system this would call SAP/NetSuite/Sage/etc. APIs.
    """
    csv_path = DATA_DIR / filename
    df = pd.read_csv(
        csv_path,
        parse_dates=["order_date", "requested_ship_date", "promised_ship_date"],
    )
    return df
