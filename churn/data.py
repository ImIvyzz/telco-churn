"""Load and clean the IBM Telco Customer Churn dataset."""
from pathlib import Path

import pandas as pd

from churn import config


def load_raw(path: Path = config.DATA_PATH) -> pd.DataFrame:
    """Read the CSV exactly as published."""
    raw = pd.read_csv(path)
    if not raw[config.ID_COL].is_unique:
        raise ValueError("customerID is not unique: check the input file")
    return raw


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return an analysis-ready copy of the data.

    - TotalCharges arrives as text; its 11 blank values all belong to customers
      with 0 months of tenure (signed up, never billed), so they become 0.
    - Churn becomes 0/1 and SeniorCitizen becomes "No"/"Yes" like the other flags.
    - customerID is dropped: an ID carries no predictive signal.
    """
    df = raw.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    blank = df["TotalCharges"].isna()
    if not (df.loc[blank, "tenure"] == 0).all():
        raise ValueError("Blank TotalCharges found for customers with tenure > 0")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df[config.TARGET] = (df[config.TARGET] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df = df.drop(columns=config.ID_COL)
    if df.isna().any().any():
        raise ValueError("Unexpected missing values after cleaning")
    return df


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket tenure into the four groups used in the segment analysis."""
    out = df.copy()
    out["tenure_group"] = pd.cut(out["tenure"], bins=[-1, 12, 24, 48, 72],
                                 labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"])
    return out
