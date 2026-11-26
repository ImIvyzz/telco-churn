"""Descriptive analysis: churn by segment, headline figures and chi-square tests."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from churn import config

SEGMENTS = ["Contract", "tenure_group", "InternetService", "PaymentMethod", "TechSupport", "SeniorCitizen"]


def churn_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Customers and churn rate for each value of `col`."""
    return (df.groupby(col, observed=True)[config.TARGET]
              .agg(customers="size", churn_rate="mean")
              .reset_index())


def headline_figures(df: pd.DataFrame) -> dict[str, float]:
    """The numbers quoted in the README and the presentation."""
    churned = df[config.TARGET] == 1

    def rate(mask: pd.Series) -> float:
        return df.loc[mask, config.TARGET].mean()

    return {
        "customers": len(df),
        "churn_rate": df[config.TARGET].mean(),
        "monthly_revenue": df["MonthlyCharges"].sum(),
        "lost_monthly_revenue": df.loc[churned, "MonthlyCharges"].sum(),
        "churn_month_to_month": rate(df["Contract"] == "Month-to-month"),
        "churn_one_year": rate(df["Contract"] == "One year"),
        "churn_two_year": rate(df["Contract"] == "Two year"),
        "churn_first_12_months": rate(df["tenure"] <= 12),
        "churn_after_24_months": rate(df["tenure"] > 24),
        "churn_fiber": rate(df["InternetService"] == "Fiber optic"),
        "churn_electronic_check": rate(df["PaymentMethod"] == "Electronic check"),
        "churn_no_tech_support": rate(df["TechSupport"] == "No"),
        "churn_seniors": rate(df["SeniorCitizen"] == "Yes"),
        "month_to_month_share_of_churners": (df.loc[churned, "Contract"] == "Month-to-month").mean(),
    }


def chi_square_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-square test of independence between each categorical variable and churn."""
    cat_vars = [c for c in df.columns
                if c not in (config.TARGET, "tenure_group") and not pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for col in cat_vars:
        chi2, p, _, _ = chi2_contingency(pd.crosstab(df[col], df[config.TARGET]))
        rows.append({"variable": col, "chi2": chi2, "p_value": p})
    tests = pd.DataFrame(rows).sort_values("chi2", ascending=False).reset_index(drop=True)
    tests["related_to_churn"] = tests["p_value"] < 0.05
    return tests


def plot_segments(df: pd.DataFrame, path: Path) -> None:
    """Six bar charts of churn rate by segment, with the overall rate as a dashed line."""
    overall = df[config.TARGET].mean()
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    for ax, col in zip(axes.flat, SEGMENTS):
        t = churn_by(df, col)
        t[col] = t[col].astype(str)
        sns.barplot(data=t, x="churn_rate", y=col, ax=ax, color=config.ACCENT, order=list(t[col]))
        ax.axvline(overall, ls="--", c="grey", lw=1)
        ax.set(xlabel="Churn rate", ylabel="", title=col, xlim=(0, 0.62))
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        for patch, rate in zip(ax.patches, t["churn_rate"]):
            ax.text(patch.get_width() + 0.006, patch.get_y() + patch.get_height() / 2,
                    f"{rate:.1%}", va="center", fontsize=9)
    fig.suptitle("Churn rate by customer segment", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
