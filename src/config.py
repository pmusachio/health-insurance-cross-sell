"""Central configuration: paths, dataset identity, modeling constants and the
Dracula palette shared by the pipeline, the serving layer and the dashboard.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SAMPLE_DIR: Path = DATA_DIR / "sample"
MODELS_DIR: Path = BASE_DIR / "models"

PIPELINE_PATH: Path = MODELS_DIR / "pipeline.joblib"
MODEL_CARD_PATH: Path = MODELS_DIR / "model_card.json"
PROCESSED_PATH: Path = PROCESSED_DIR / "train.parquet"

SAMPLE_FILENAME: str = "health_insurance_sample.csv"
SAMPLE_PATH: Path = SAMPLE_DIR / SAMPLE_FILENAME

# --------------------------------------------------------------------------- #
# Dataset identity
# --------------------------------------------------------------------------- #
KAGGLE_DATASET: str = "anmolkumar/health-insurance-cross-sell-prediction"
RAW_TRAIN_FILENAME: str = "train.csv"

# --------------------------------------------------------------------------- #
# Schema / business framing
# --------------------------------------------------------------------------- #
TARGET: str = "Response"
ID_COL: str = "id"
POSITIVE_LABEL: int = 1

NUMERIC_FEATURES: tuple[str, ...] = (
    "Age",
    "Annual_Premium",
    "Vintage",
    "Region_Code",
    "Policy_Sales_Channel",
)
BINARY_FEATURES: tuple[str, ...] = (
    "Driving_License",
    "Previously_Insured",
    "Gender",          # mapped Male=1 / Female=0
    "Vehicle_Damage",  # mapped Yes=1 / No=0
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("Vehicle_Age",)

VEHICLE_AGE_ORDER: tuple[str, ...] = ("< 1 Year", "1-2 Year", "> 2 Years")

# No target leakage: every feature is known at scoring time, before the customer
# is contacted. Response is the only field derived from the outcome and is the target.

# --------------------------------------------------------------------------- #
# Modeling
# --------------------------------------------------------------------------- #
TEST_SIZE: float = 0.2
SEED: int = 42
CV_FOLDS: int = 4
TUNING_ITERS: int = 12
SCORING: str = "roc_auc"

# Business capacity points (number of customers contacted) for lift / gains.
CONTACT_CAPACITIES: tuple[int, ...] = (5_000, 10_000, 20_000)
DEFAULT_CAPACITY_PCT: int = 20

# --------------------------------------------------------------------------- #
# Visual identity (Dracula)
# --------------------------------------------------------------------------- #
DRACULA = {
    "background": "#282a36",
    "current_line": "#44475a",
    "foreground": "#f8f8f2",
    "comment": "#6272a4",
    "cyan": "#8be9fd",
    "green": "#50fa7b",
    "orange": "#ffb86c",
    "pink": "#ff79c6",
    "purple": "#bd93f9",
    "red": "#ff5555",
    "yellow": "#f1fa8c",
}
