"""Smoke tests for the data contract, leakage guarantees and the serving surface.
Run offline against the versioned sample; no Kaggle access required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.predict import Predictor  # noqa: E402
from src.preprocessing import FeaturePrep, Preprocessor, clean_features  # noqa: E402


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(config.SAMPLE_PATH)


def test_target_present_and_imbalanced(sample):
    assert config.TARGET in sample.columns
    rate = sample[config.TARGET].mean()
    assert 0.05 < rate < 0.30  # known ~12% positive class


def test_preprocessing_excludes_id_and_target_from_features(sample):
    X, y = Preprocessor().run(sample)
    assert config.ID_COL not in X.columns
    assert config.TARGET not in X.columns
    assert set(y.unique()) <= {0, 1}


def test_clean_features_maps_binaries(sample):
    cleaned = clean_features(sample)
    assert set(pd.unique(cleaned["Gender"])) <= {0.0, 1.0}
    assert set(pd.unique(cleaned["Vehicle_Damage"])) <= {0.0, 1.0}


def test_feature_prep_yields_fixed_columns(sample):
    prepped = FeaturePrep().fit_transform(sample.head(20))
    assert list(prepped.columns) == list(FeaturePrep().transform(sample.head(5)).columns)


def test_predictor_contract():
    pred = Predictor()
    record = {
        "Age": 40, "Gender": "Male", "Driving_License": 1, "Region_Code": 28.0,
        "Previously_Insured": 0, "Vehicle_Age": "1-2 Year", "Vehicle_Damage": "Yes",
        "Annual_Premium": 30000.0, "Policy_Sales_Channel": 26.0, "Vintage": 150,
    }
    score = pred.score_one(record)
    assert 0.0 <= score <= 1.0
    assert 0.0 < pred.base_rate < 1.0
    assert len(pred.top_features(5)) >= 1


def test_predictor_ranks_higher_risk_above_lower_risk():
    pred = Predictor()
    likely = {"Age": 45, "Gender": "Male", "Driving_License": 1, "Region_Code": 28.0,
              "Previously_Insured": 0, "Vehicle_Age": "> 2 Years", "Vehicle_Damage": "Yes",
              "Annual_Premium": 40000.0, "Policy_Sales_Channel": 26.0, "Vintage": 200}
    unlikely = {**likely, "Previously_Insured": 1, "Vehicle_Damage": "No", "Vehicle_Age": "< 1 Year"}
    assert pred.score_one(likely) > pred.score_one(unlikely)
