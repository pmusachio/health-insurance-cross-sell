"""Transformation layer.

The feature cleaning (binary mapping of Gender/Vehicle_Damage, normalization of
Vehicle_Age) lives in a custom transformer that is the first step of the model
Pipeline, so the exact same transform runs at training and serving time and no
training-serving skew can arise.
"""
from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

logger = logging.getLogger(__name__)

FEATURE_SOURCE_COLUMNS = (
    list(config.NUMERIC_FEATURES)
    + ["Driving_License", "Previously_Insured", "Gender", "Vehicle_Damage", "Vehicle_Age"]
)


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Maps categorical strings to model-ready values. Vectorized and idempotent."""
    out = df.copy()
    if "Gender" in out:
        out["Gender"] = out["Gender"].map({"Male": 1, "Female": 0}).fillna(out["Gender"]).astype(float)
    if "Vehicle_Damage" in out:
        out["Vehicle_Damage"] = (
            out["Vehicle_Damage"].map({"Yes": 1, "No": 0}).fillna(out["Vehicle_Damage"]).astype(float)
        )
    if "Vehicle_Age" in out:
        out["Vehicle_Age"] = (
            out["Vehicle_Age"].astype(str).str.strip().replace({"nan": "1-2 Year"})
        )
    return out


class FeaturePrep(BaseEstimator, TransformerMixin):
    """First pipeline step: clean raw rows into the modeling feature frame."""

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> pd.DataFrame:
        df = pd.DataFrame(X).copy()
        df = clean_features(df)
        for col in FEATURE_SOURCE_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df[FEATURE_SOURCE_COLUMNS]


def build_column_transformer() -> ColumnTransformer:
    numeric = list(config.NUMERIC_FEATURES) + ["Gender", "Vehicle_Damage", "Driving_License", "Previously_Insured"]
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")),
         ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric),
         ("cat", categorical_pipe, ["Vehicle_Age"])],
        remainder="drop",
    )


class Preprocessor:
    """Splits target from features and persists a processed reference frame."""

    def __init__(self, processed_path=config.PROCESSED_PATH) -> None:
        self.processed_path = processed_path

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if config.TARGET not in df.columns:
            raise ValueError(f"Target '{config.TARGET}' missing from training data")
        y = df[config.TARGET].astype(int)
        X = df[[c for c in FEATURE_SOURCE_COLUMNS if c in df.columns]].copy()

        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned = clean_features(df).copy()
        cleaned[config.TARGET] = y.values
        cleaned.to_parquet(self.processed_path, index=False)
        logger.info("Processed frame (%d rows) written to %s", len(cleaned), self.processed_path)
        return X, y
