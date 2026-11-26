"""Paths, constants and the business assumptions behind the analysis."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = ROOT / "outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.30          # stratified hold-out share
CV_FOLDS = 5

TARGET = "Churn"
ID_COL = "customerID"
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Model selection: if random forest beats logistic regression by less than this
# cross-validated ROC-AUC, keep logistic regression (simpler and explainable).
TIE_TOLERANCE = 0.005

# Risk bands for the customer scores
RISK_BINS = [0, 0.3, 0.6, 1]
RISK_LABELS = ["Low", "Medium", "High"]

# Retention scenario. These are ASSUMPTIONS, not results: override them from the
# command line (python run_analysis.py --save-rate 0.3) to test other cases.
SAVE_RATE = 0.20          # share of contacted churners a retention offer keeps
TARGET_BAND = "High"

ACCENT = "#0f766e"        # chart colour
