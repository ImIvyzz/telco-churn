"""Business impact: a risk score for every customer, risk bands and a retention scenario."""
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from churn import config

EXPORT_COLS = ["Contract", "tenure", "InternetService", "PaymentMethod", "TechSupport",
               "SeniorCitizen", "MonthlyCharges", config.TARGET]


def score_customers(model: Pipeline, raw: pd.DataFrame, df: pd.DataFrame,
                    features: pd.DataFrame, target: pd.Series, cv) -> pd.DataFrame:
    """Out-of-fold churn risk for every customer.

    Each customer's score comes from a model that never saw that customer, so the
    scores are honest even though they cover the whole base.
    """
    risk = cross_val_predict(clone(model), features, target, cv=cv, method="predict_proba")[:, 1]
    scored = raw[[config.ID_COL]].join(df[EXPORT_COLS])
    scored["churn_risk"] = risk.round(4)
    scored["risk_band"] = pd.cut(risk, bins=config.RISK_BINS, labels=config.RISK_LABELS,
                                 include_lowest=True)
    return scored


def risk_bands(scored: pd.DataFrame) -> pd.DataFrame:
    """Customers, real churn and revenue in each risk band."""
    bands = (scored.groupby("risk_band", observed=True)
                   .agg(customers=(config.ID_COL, "size"),
                        actual_churn_rate=(config.TARGET, "mean"),
                        churners=(config.TARGET, "sum"),
                        monthly_revenue=("MonthlyCharges", "sum")))
    bands["share_of_all_churners"] = bands["churners"] / scored[config.TARGET].sum()
    return bands


def retention_scenario(scored: pd.DataFrame, save_rate: float = config.SAVE_RATE,
                       band: str = config.TARGET_BAND) -> dict[str, float]:
    """Revenue protected if a retention offer to `band` keeps `save_rate` of its churners.

    The result is before the cost of the offer: it is a ceiling, not a profit figure.
    """
    if not 0 <= save_rate <= 1:
        raise ValueError("save_rate must be between 0 and 1")
    if band not in config.RISK_LABELS:
        raise ValueError(f"band must be one of {config.RISK_LABELS}")
    targeted = scored[scored["risk_band"] == band]
    caught = targeted[targeted[config.TARGET] == 1]
    saved_monthly = save_rate * caught["MonthlyCharges"].sum()
    return {
        "customers_contacted": len(targeted),
        "share_of_base": len(targeted) / len(scored),
        "churners_reached": len(caught),
        "share_of_churners": len(caught) / scored[config.TARGET].sum(),
        "saved_monthly": saved_monthly,
        "saved_yearly": saved_monthly * 12,
    }
