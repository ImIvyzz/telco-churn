"""Checks that guard the analysis against the mistakes fixed in v2.

Run with:  pytest
"""
import numpy as np
import pandas as pd
import pytest

from churn import analysis, config, data, impact, model


@pytest.fixture(scope="module")
def raw():
    return data.load_raw()


@pytest.fixture(scope="module")
def df(raw):
    return data.clean(raw)


@pytest.fixture(scope="module")
def fitted(df):
    features, target = model.features_and_target(df)
    s = model.split(features, target)
    pipes = model.build_pipelines(features)
    pipes[model.LOGIT].fit(s.X_train, s.y_train)
    return pipes[model.LOGIT], s, features, target


# --- data cleaning -------------------------------------------------------------

def test_blank_total_charges_are_new_customers_set_to_zero(raw, df):
    blank = pd.to_numeric(raw["TotalCharges"].astype(str).str.strip(), errors="coerce").isna()
    assert blank.sum() == 11
    assert (df.loc[blank, "tenure"] == 0).all()
    assert (df.loc[blank, "TotalCharges"] == 0).all()


def test_clean_data_is_complete_and_encoded(df):
    assert len(df) == 7043
    assert not df.isna().any().any()
    assert set(df[config.TARGET].unique()) == {0, 1}
    assert config.ID_COL not in df.columns


def test_headline_figures(df):
    f = analysis.headline_figures(df)
    assert f["churn_rate"] == pytest.approx(0.265, abs=0.001)
    assert f["churn_month_to_month"] == pytest.approx(0.427, abs=0.001)
    assert f["churn_two_year"] == pytest.approx(0.028, abs=0.001)


def test_gender_is_not_a_churn_driver(df):
    tests = analysis.chi_square_tests(df).set_index("variable")
    assert not tests.loc["gender", "related_to_churn"]
    assert tests.loc["Contract", "related_to_churn"]


# --- modelling -----------------------------------------------------------------

def test_split_is_stratified(fitted):
    _, s, _, target = fitted
    assert s.y_test.mean() == pytest.approx(target.mean(), abs=0.005)


def test_scaler_is_fitted_on_training_data_only(fitted):
    """v1 re-fitted the scaler on the test set. Here it must only see training rows."""
    logit, s, _, _ = fitted
    scaler = logit.named_steps["prep"].named_transformers_["num"]
    assert np.allclose(scaler.mean_, s.X_train[config.NUM_COLS].mean())
    assert not np.allclose(scaler.mean_, s.X_test[config.NUM_COLS].mean())


def test_model_beats_baseline_on_recall_and_auc(fitted):
    from sklearn.metrics import recall_score, roc_auc_score
    logit, s, _, _ = fitted
    proba = logit.predict_proba(s.X_test)[:, 1]
    assert roc_auc_score(s.y_test, proba) > 0.80
    assert recall_score(s.y_test, proba >= 0.5) > 0.70   # the baseline catches 0%


def test_tie_goes_to_logistic_regression():
    results = pd.DataFrame({"CV ROC-AUC (train, 5-fold)": [0.5, 0.844, 0.846]},
                           index=[model.BASELINE, model.LOGIT, model.FOREST])
    assert model.select_model(results) == model.LOGIT
    results.loc[model.FOREST] = 0.90
    assert model.select_model(results) == model.FOREST


# --- business impact -----------------------------------------------------------

def test_retention_scenario_math():
    scored = pd.DataFrame({
        "risk_band": ["High", "High", "High", "Low"],
        config.TARGET: [1, 1, 0, 1],
        "MonthlyCharges": [100.0, 50.0, 80.0, 70.0],
    })
    sc = impact.retention_scenario(scored, save_rate=0.2, band="High")
    assert sc["customers_contacted"] == 3
    assert sc["churners_reached"] == 2
    assert sc["saved_monthly"] == pytest.approx(30.0)       # 20% of (100 + 50)
    assert sc["saved_yearly"] == pytest.approx(360.0)


def test_retention_scenario_rejects_bad_assumptions():
    scored = pd.DataFrame({"risk_band": ["High"], config.TARGET: [1], "MonthlyCharges": [10.0]})
    with pytest.raises(ValueError):
        impact.retention_scenario(scored, save_rate=1.5)
    with pytest.raises(ValueError):
        impact.retention_scenario(scored, band="Extreme")
