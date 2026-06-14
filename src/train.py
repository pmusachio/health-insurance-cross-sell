"""Modeling layer: baseline, cross-validated model selection and tuning, holdout
evaluation with ranking metrics and slices, business translation (lift / gains),
and serialization of a self-contained pipeline plus a model card.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.preprocessing import FeaturePrep, build_column_transformer

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
N_SELECT = 80_000  # cap rows used for CV model selection / tuning, for bounded runtime
# Fold-level parallelism via loky stalls badly on macOS when nested with the
# estimators' own threading. Run meta-estimators single-process and let each fit
# use the estimator's internal parallelism instead.
N_JOBS = 1


@dataclass
class TrainingResult:
    baseline: Dict[str, float] = field(default_factory=dict)
    cv_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_model: str = ""
    best_params: Dict[str, Any] = field(default_factory=dict)
    holdout: Dict[str, Any] = field(default_factory=dict)
    business: Dict[str, Any] = field(default_factory=dict)
    importances: list = field(default_factory=list)


def _model(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.SEED)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      n_jobs=2, random_state=config.SEED)
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=config.SEED)


def _pipeline(name: str) -> Pipeline:
    return Pipeline([("prep", FeaturePrep()), ("ct", build_column_transformer()), ("clf", _model(name))])


def _param_distributions(name: str) -> Dict[str, Any]:
    if name == "logreg":
        return {"clf__C": np.logspace(-3, 2, 30)}
    if name == "random_forest":
        return {
            "clf__n_estimators": [200, 300, 500],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_leaf": [1, 2, 4, 8],
            "clf__max_features": ["sqrt", "log2", 0.5],
        }
    return {
        "clf__learning_rate": np.logspace(-2, -0.3, 12),
        "clf__max_leaf_nodes": [15, 31, 63, 127],
        "clf__max_depth": [None, 4, 8, 16],
        "clf__l2_regularization": [0.0, 0.1, 1.0, 10.0],
        "clf__max_iter": [200, 400, 600],
    }


class ModelTrainer:
    def __init__(self, X: pd.DataFrame, y: pd.Series, data_source: Path | None = None) -> None:
        self.data_source = data_source
        self.X_train, self.X_holdout, self.y_train, self.y_holdout = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
        )
        self.base_rate = float(y.mean())
        self.result = TrainingResult()

    def _select_subsample(self):
        if len(self.X_train) <= N_SELECT:
            return self.X_train, self.y_train
        idx = (
            pd.Series(range(len(self.y_train)))
            .groupby(self.y_train.reset_index(drop=True), group_keys=False)
            .apply(lambda s: s.sample(frac=N_SELECT / len(self.y_train), random_state=config.SEED))
        )
        return self.X_train.iloc[idx.values], self.y_train.iloc[idx.values]

    def fit_baseline(self) -> Dict[str, float]:
        pipe = _pipeline("logreg").fit(self.X_train, self.y_train)
        proba = pipe.predict_proba(self.X_holdout)[:, 1]
        self.result.baseline = {
            "model": "LogisticRegression (balanced)",
            "roc_auc": float(roc_auc_score(self.y_holdout, proba)),
            "average_precision": float(average_precision_score(self.y_holdout, proba)),
        }
        logger.info("Baseline ROC AUC=%.4f AP=%.4f", self.result.baseline["roc_auc"],
                    self.result.baseline["average_precision"])
        return self.result.baseline

    def fit(self) -> Pipeline:
        Xs, ys = self._select_subsample()
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
        rows = []
        for name in ("logreg", "random_forest", "hist_gb"):
            scores = cross_val_score(_pipeline(name), Xs, ys, cv=cv, scoring=config.SCORING, n_jobs=N_JOBS)
            rows.append({"model": name, "roc_auc_mean": scores.mean(), "roc_auc_std": scores.std()})
            logger.info("CV %-14s ROC AUC=%.4f +/- %.4f", name, scores.mean(), scores.std())
        self.result.cv_table = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
        best = self.result.cv_table.iloc[0]["model"]
        self.result.best_model = best

        search = RandomizedSearchCV(
            _pipeline(best), _param_distributions(best), n_iter=config.TUNING_ITERS,
            scoring=config.SCORING, cv=cv, n_jobs=N_JOBS, random_state=config.SEED, refit=True,
        ).fit(Xs, ys)
        self.result.best_params = {k: _jsonable(v) for k, v in search.best_params_.items()}
        logger.info("Tuned %s best CV ROC AUC=%.4f params=%s", best, search.best_score_, self.result.best_params)

        # Refit the tuned configuration on the full training split.
        self.final_pipeline = _pipeline(best).set_params(**search.best_params_).fit(self.X_train, self.y_train)
        return self.final_pipeline

    def evaluate(self) -> Dict[str, Any]:
        proba = self.final_pipeline.predict_proba(self.X_holdout)[:, 1]
        y = self.y_holdout.reset_index(drop=True)
        order = np.argsort(proba)[::-1]
        y_sorted = y.iloc[order].to_numpy()

        capacities = {}
        for k in config.CONTACT_CAPACITIES:
            kk = min(k, len(y_sorted))
            prec = float(y_sorted[:kk].mean())
            capacities[str(k)] = {
                "precision_at_k": round(prec, 4),
                "lift_at_k": round(prec / self.base_rate, 3) if self.base_rate else None,
                "buyers_captured_pct": round(100 * y_sorted[:kk].sum() / y_sorted.sum(), 1),
            }

        slices = {}
        for col, mapper in (("Previously_Insured", None), ("Vehicle_Damage", {"Yes": 1, "No": 0})):
            s = self.X_holdout[col]
            if mapper:
                s = s.map(mapper).fillna(s)
            for val in sorted(pd.Series(s).dropna().unique()):
                mask = (s == val).to_numpy()
                if mask.sum() > 50 and len(np.unique(y[mask])) == 2:
                    slices[f"{col}={val}"] = round(float(roc_auc_score(y[mask], proba[mask])), 4)

        self.result.holdout = {
            "roc_auc": float(roc_auc_score(y, proba)),
            "average_precision": float(average_precision_score(y, proba)),
            "base_rate": round(self.base_rate, 4),
            "capacities": capacities,
            "roc_auc_by_slice": slices,
        }
        logger.info("Holdout ROC AUC=%.4f AP=%.4f", self.result.holdout["roc_auc"],
                    self.result.holdout["average_precision"])
        return self.result.holdout

    def to_business_metrics(self) -> Dict[str, Any]:
        cap20 = self.result.holdout["capacities"].get("20000", {})
        lift = cap20.get("lift_at_k")
        self.result.business = {
            "headline": (
                f"Contacting the top 20,000 ranked customers reaches {cap20.get('buyers_captured_pct')}% "
                f"of all interested buyers, {lift}x more efficient than random outreach."
            ),
            "lift_at_20000": lift,
            "buyers_captured_pct_at_20000": cap20.get("buyers_captured_pct"),
        }
        return self.result.business

    def compute_importances(self) -> list:
        n = min(10_000, len(self.X_holdout))
        Xs = self.X_holdout.iloc[:n]
        ys = self.y_holdout.iloc[:n]
        r = permutation_importance(self.final_pipeline, Xs, ys, n_repeats=3,
                                   random_state=config.SEED, scoring="roc_auc", n_jobs=N_JOBS)
        cols = list(Xs.columns)
        self.result.importances = sorted(
            [{"feature": cols[i], "importance": round(float(r.importances_mean[i]), 5)} for i in range(len(cols))],
            key=lambda d: d["importance"], reverse=True,
        )
        return self.result.importances

    def save(self) -> None:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "pipeline": self.final_pipeline,
                "best_model": self.result.best_model,
                "base_rate": self.base_rate,
                "importances": self.result.importances,
                "feature_columns": list(self.X_train.columns),
            },
            config.PIPELINE_PATH,
        )
        logger.info("Pipeline artifact written to %s", config.PIPELINE_PATH)
        card = {
            "schema_version": SCHEMA_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": config.KAGGLE_DATASET,
            "data_sha256": self._data_hash(),
            "target": config.TARGET,
            "problem": "propensity ranking (imbalanced binary classification)",
            "best_model": self.result.best_model,
            "best_params": self.result.best_params,
            "cv_selection": self.result.cv_table.to_dict(orient="records"),
            "baseline": self.result.baseline,
            "holdout": self.result.holdout,
            "business": self.result.business,
            "top_features": self.result.importances[:8],
        }
        config.MODEL_CARD_PATH.write_text(json.dumps(card, indent=2))
        logger.info("Model card written to %s", config.MODEL_CARD_PATH)

    def _data_hash(self) -> str:
        src = self.data_source or config.SAMPLE_PATH
        if src and Path(src).exists():
            return hashlib.sha256(Path(src).read_bytes()).hexdigest()
        return "unknown"


def _jsonable(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v
