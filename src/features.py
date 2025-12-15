import pandas as pd


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for engineered features.
    You can expand this later.
    """
    if {"order_date", "requested_ship_date"}.issubset(df.columns):
        df["requested_lead_time_days"] = (
            (df["requested_ship_date"] - df["order_date"]).dt.days
        )
    return df
