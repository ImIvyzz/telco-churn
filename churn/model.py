"""Predictive model: leakage-free pipelines, evaluation, selection and interpretation."""
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config

BASELINE = "Baseline (always 'stays')"
LOGIT = "Logistic regression"
FOREST = "Random forest"


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Model inputs: every cleaned column except the target and helper columns."""
    features = df.drop(columns=[c for c in (config.TARGET, "tenure_group") if c in df.columns])
    return features, df[config.TARGET]


def split(features: pd.DataFrame, target: pd.Series) -> Split:
    """Stratified hold-out split, so train and test keep the same churn share."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=config.TEST_SIZE, stratify=target, random_state=config.RANDOM_STATE)
    return Split(X_train, X_test, y_train, y_test)


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def build_pipelines(features: pd.DataFrame) -> dict[str, Pipeline]:
    """Scaling and encoding live inside each Pipeline, so they are fitted on training data only."""
    cat_cols = [c for c in features.columns if c not in config.NUM_COLS]
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), config.NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
    ])
    models = {
        BASELINE: DummyClassifier(strategy="most_frequent"),
        LOGIT: LogisticRegression(max_iter=2000, class_weight="balanced"),
        FOREST: RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced",
                                       random_state=config.RANDOM_STATE, n_jobs=-1),
    }
    return {name: Pipeline([("prep", clone(preprocess)), ("model", m)]) for name, m in models.items()}


def evaluate(pipes: dict[str, Pipeline], s: Split) -> pd.DataFrame:
    """Cross-validate on the training set, then fit and score each model on the test set.

    The pipelines are fitted in place, so they are ready to use afterwards.
    """
    rows = []
    for name, pipe in pipes.items():
        cv_auc = cross_val_score(pipe, s.X_train, s.y_train, cv=cv_splitter(), scoring="roc_auc")
        pipe.fit(s.X_train, s.y_train)
        proba = pipe.predict_proba(s.X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "model": name,
            "CV ROC-AUC (train, 5-fold)": cv_auc.mean(),
            "accuracy": accuracy_score(s.y_test, pred),
            "precision": precision_score(s.y_test, pred, zero_division=0),
            "recall": recall_score(s.y_test, pred),
            "F1": f1_score(s.y_test, pred),
            "ROC-AUC": roc_auc_score(s.y_test, proba),
        })
    return pd.DataFrame(rows).set_index("model")


def select_model(results: pd.DataFrame) -> str:
    """Highest cross-validated ROC-AUC; a practical tie goes to logistic regression."""
    cv_auc = results.drop(index=BASELINE)["CV ROC-AUC (train, 5-fold)"]
    if cv_auc.max() - cv_auc[LOGIT] < config.TIE_TOLERANCE:
        return LOGIT
    return str(cv_auc.idxmax())


def threshold_table(pipe: Pipeline, s: Split, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)) -> pd.DataFrame:
    """Recall vs. precision trade-off at different decision thresholds."""
    proba = pipe.predict_proba(s.X_test)[:, 1]
    return pd.DataFrame([{
        "threshold": t,
        "customers_flagged": int((proba >= t).sum()),
        "recall": recall_score(s.y_test, proba >= t),
        "precision": precision_score(s.y_test, proba >= t),
    } for t in thresholds]).set_index("threshold")


def odds_ratios(logit: Pipeline) -> pd.Series:
    """Odds ratio per feature (numeric features are per one standard deviation)."""
    names = [n.split("__", 1)[1] for n in logit.named_steps["prep"].get_feature_names_out()]
    odds = pd.Series(np.exp(logit.named_steps["model"].coef_[0]), index=names).sort_values()
    # "No internet service" repeats the same information in six add-on columns
    return odds[~odds.index.str.endswith("No internet service")]


def importance(pipe: Pipeline, s: Split, top: int = 10) -> pd.DataFrame:
    """Permutation importance on the test set: how much ROC-AUC drops when a column is shuffled."""
    imp = permutation_importance(pipe, s.X_test, s.y_test, scoring="roc_auc", n_repeats=10,
                                 random_state=config.RANDOM_STATE, n_jobs=-1)
    return (pd.DataFrame({"feature": s.X_test.columns, "auc_drop": imp.importances_mean})
              .sort_values("auc_drop", ascending=False).head(top).reset_index(drop=True))


def plot_evaluation(pipes: dict[str, Pipeline], best_name: str, s: Split, path: Path) -> None:
    """ROC curves for both models and the confusion matrix of the selected one."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for name in (LOGIT, FOREST):
        RocCurveDisplay.from_estimator(pipes[name], s.X_test, s.y_test, ax=axes[0], name=name)
    axes[0].plot([0, 1], [0, 1], "--", c="grey", label="Random guess")
    axes[0].set_title("ROC curve (test set)")
    axes[0].legend(loc="lower right")
    ConfusionMatrixDisplay.from_estimator(pipes[best_name], s.X_test, s.y_test,
                                          display_labels=["Stays", "Churns"], cmap="Greens",
                                          colorbar=False, ax=axes[1])
    axes[1].set_title(f"Confusion matrix: {best_name} (threshold 0.5)")
    axes[1].grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drivers(odds: pd.Series, path: Path) -> None:
    """The six strongest protective and seven strongest risk factors."""
    top = pd.concat([odds.head(6), odds.tail(7)])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top.values, color=[config.ACCENT if v > 1 else "#9ca3af" for v in top.values])
    ax.axvline(1, c="black", lw=1)
    ax.set_xscale("log")
    ticks = [0.25, 0.5, 1, 1.5, 2]
    ax.set_xticks(ticks, [f"{t:g}x" for t in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Odds ratio (log scale): > 1 raises churn risk, < 1 lowers it",
           title="Strongest churn drivers (logistic regression)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
