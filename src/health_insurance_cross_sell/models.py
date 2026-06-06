"""Training, evaluation and batch prediction entrypoints."""

from __future__ import annotations

import json
from pathlib import Path

from .config import load_config, resolve_project_path
from .data import load_training_frame, read_csv
from .features import model_matrix, prepare_features


def _one_hot_encoder():
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _column_types(X):
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    return numeric, categorical


def _preprocessor(X):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    numeric, categorical = _column_types(X)
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def _classifier(config: dict):
    from sklearn.linear_model import LogisticRegression

    modeling = config.get("modeling", {})
    class_weight = modeling.get("class_weight", "balanced")
    return LogisticRegression(
        max_iter=int(modeling.get("max_iter", 1000)),
        class_weight=class_weight,
        n_jobs=-1,
        random_state=int(modeling.get("random_state", 42)),
    )


def _regressor(config: dict):
    from sklearn.ensemble import RandomForestRegressor

    modeling = config.get("modeling", {})
    return RandomForestRegressor(
        n_estimators=int(modeling.get("n_estimators", 160)),
        min_samples_leaf=int(modeling.get("min_samples_leaf", 2)),
        n_jobs=-1,
        random_state=int(modeling.get("random_state", 42)),
    )


def _pipeline(X, estimator):
    from sklearn.pipeline import Pipeline

    return Pipeline(steps=[("preprocess", _preprocessor(X)), ("model", estimator)])


def _classification_metrics(y_true, y_pred, y_proba, config: dict):
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels = sorted(set(y_true.dropna().tolist()))
    average = "binary" if len(labels) == 2 else "macro"
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
    }
    if y_proba is not None and len(labels) == 2:
        positive = config.get("data", {}).get("positive_label", labels[-1])
        classes = list(getattr(y_proba, "classes_", []))
        scores = y_proba
        if hasattr(y_proba, "shape") and len(y_proba.shape) == 2:
            pos_idx = 1
            scores = y_proba[:, pos_idx]
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
            metrics["average_precision"] = float(average_precision_score(y_true, scores))
        except ValueError:
            pass
        for k in config.get("evaluation", {}).get("top_k", []):
            k = min(int(k), len(scores))
            if k <= 0:
                continue
            order = np.argsort(scores)[::-1][:k]
            positive_mask = (y_true.reset_index(drop=True).iloc[order] == positive).astype(int)
            base_rate = float((y_true == positive).mean())
            precision_at_k = float(positive_mask.mean())
            metrics[f"precision_at_{k}"] = precision_at_k
            metrics[f"lift_at_{k}"] = precision_at_k / base_rate if base_rate else None
    return metrics


def _regression_metrics(y_true, y_pred):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = np.where(y_true == 0, np.nan, y_true)
    rmspe = np.sqrt(np.nanmean(((y_true - y_pred) / denominator) ** 2))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmspe": float(rmspe),
    }


def _try_log_mlflow(config: dict, model, metrics: dict, model_path) -> str | None:
    if not config.get("modeling", {}).get("use_mlflow", False):
        return None
    try:
        import mlflow
    except ImportError:
        return "MLflow is not installed. Install requirements-mlflow.txt to enable tracking."

    experiment = config.get("modeling", {}).get("mlflow_experiment", config.get("project", {}).get("slug", "kaggle_project"))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=config.get("project", {}).get("slug", "training")):
        mlflow.log_params(
            {
                "problem_type": config.get("project", {}).get("problem_type"),
                "family": config.get("project", {}).get("family"),
                "target": config.get("data", {}).get("target", ""),
            }
        )
        mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, (int, float))})
        mlflow.log_artifact(str(model_path))
        try:
            mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception:
            pass
    return None


def _build_candidate_estimators(config: dict) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    random_state = int(config.get("modeling", {}).get("random_state", 42))
    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
    }
    try:
        import lightgbm as lgb

        candidates["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )
    except ImportError:
        pass
    return candidates


def cross_validate_models(config: dict):
    """Evaluate multiple classifiers with StratifiedKFold cross-validation.

    Returns a pandas DataFrame with columns: model, roc_auc_mean, roc_auc_std,
    avg_precision_mean, avg_precision_std.
    """
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline

    df = load_training_frame(config)
    X, y, _ = model_matrix(df, config, training=True)
    if y is None:
        raise ValueError("Target column is not available in the training data.")

    tuning = config.get("tuning", {})
    n_splits = int(tuning.get("cv_folds", 5))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(config.get("modeling", {}).get("random_state", 42)))
    candidates = _build_candidate_estimators(config)
    rows = []
    for name, estimator in candidates.items():
        pipe = _pipeline(X, estimator)
        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=["roc_auc", "average_precision"],
            n_jobs=-1,
            return_train_score=False,
        )
        rows.append(
            {
                "model": name,
                "roc_auc_mean": scores["test_roc_auc"].mean(),
                "roc_auc_std": scores["test_roc_auc"].std(),
                "avg_precision_mean": scores["test_average_precision"].mean(),
                "avg_precision_std": scores["test_average_precision"].std(),
            }
        )
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


def tune_best_model(config: dict, model_name: str):
    """Run RandomizedSearchCV on the named model.

    model_name must be one of: 'LogisticRegression', 'RandomForest', 'LightGBM'.
    Returns the fitted RandomizedSearchCV object.
    """
    import numpy as np
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    df = load_training_frame(config)
    X, y, _ = model_matrix(df, config, training=True)
    if y is None:
        raise ValueError("Target column is not available.")

    tuning = config.get("tuning", {})
    n_iter = int(tuning.get("n_iter", 20))
    n_splits = int(tuning.get("cv_folds", 5))
    scoring = tuning.get("scoring", "roc_auc")
    random_state = int(config.get("modeling", {}).get("random_state", 42))

    candidates = _build_candidate_estimators(config)
    if model_name not in candidates:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(candidates)}")

    param_distributions: dict = {}
    if model_name == "LogisticRegression":
        param_distributions = {
            "model__C": np.logspace(-3, 2, 50),
            "model__solver": ["lbfgs", "saga"],
        }
    elif model_name == "RandomForest":
        param_distributions = {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", 0.5],
        }
    elif model_name == "LightGBM":
        param_distributions = {
            "model__n_estimators": [200, 300, 500, 800],
            "model__learning_rate": np.logspace(-2, -0.3, 20),
            "model__num_leaves": [31, 63, 127, 255],
            "model__min_child_samples": [10, 20, 50, 100],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        }

    pipe = _pipeline(X, candidates[model_name])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
        refit=True,
    )
    search.fit(X, y)
    return search


def train_supervised(config: dict):
    import joblib
    from sklearn.model_selection import train_test_split

    df = load_training_frame(config)
    X, y, _ = model_matrix(df, config, training=True)
    if y is None:
        raise ValueError("Target column is not available in the training data.")

    modeling = config.get("modeling", {})
    problem_type = config.get("project", {}).get("problem_type", "binary_classification")
    stratify = y if problem_type in {"binary_classification", "multiclass_classification", "ranking"} else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=float(modeling.get("test_size", 0.2)),
        random_state=int(modeling.get("random_state", 42)),
        stratify=stratify,
    )

    estimator = _regressor(config) if problem_type == "regression" else _classifier(config)
    model = _pipeline(X_train, estimator)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid) if hasattr(model, "predict_proba") else None
    metrics = (
        _regression_metrics(y_valid, y_pred)
        if problem_type == "regression"
        else _classification_metrics(y_valid, y_pred, y_proba, config)
    )

    model_path = resolve_project_path(config, config.get("modeling", {}).get("model_file", "models/model.joblib"))
    metrics_path = resolve_project_path(config, "reports/metrics.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    mlflow_note = _try_log_mlflow(config, model, metrics, model_path)
    if mlflow_note:
        metrics["mlflow_note"] = mlflow_note
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"model_path": str(model_path), "metrics_path": str(metrics_path), "metrics": metrics}


def train_clustering(config: dict):
    import json
    import joblib
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    from sklearn.pipeline import Pipeline

    df = load_training_frame(config)
    X, _, prepared = model_matrix(df, config, training=True)
    preprocessor = _preprocessor(X)
    X_matrix = preprocessor.fit_transform(X)
    cluster_range = config.get("modeling", {}).get("cluster_range", [3, 4, 5, 6, 7])
    sample_size = int(config.get("modeling", {}).get("silhouette_sample_size", 10000))
    results = []
    best = None
    for k in cluster_range:
        model = KMeans(n_clusters=int(k), n_init="auto", random_state=int(config.get("modeling", {}).get("random_state", 42)))
        labels = model.fit_predict(X_matrix)
        if len(labels) > sample_size:
            rng = np.random.default_rng(int(config.get("modeling", {}).get("random_state", 42)))
            idx = rng.choice(len(labels), size=sample_size, replace=False)
            sil = silhouette_score(X_matrix[idx], labels[idx])
        else:
            sil = silhouette_score(X_matrix, labels)
        metrics = {
            "k": int(k),
            "silhouette": float(sil),
            "calinski_harabasz": float(calinski_harabasz_score(X_matrix, labels)),
            "davies_bouldin": float(davies_bouldin_score(X_matrix, labels)),
        }
        results.append(metrics)
        if best is None or metrics["silhouette"] > best["silhouette"]:
            best = metrics

    selected_model = KMeans(
        n_clusters=best["k"],
        n_init="auto",
        random_state=int(config.get("modeling", {}).get("random_state", 42)),
    )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", selected_model)])
    labels = pipeline.fit_predict(X)
    prepared = prepared.copy()
    prepared["cluster"] = labels

    model_path = resolve_project_path(config, config.get("modeling", {}).get("model_file", "models/model.joblib"))
    metrics_path = resolve_project_path(config, "reports/metrics.json")
    clusters_path = resolve_project_path(config, "reports/cluster_assignments.csv")
    for path in [model_path, metrics_path, clusters_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps({"selected": best, "trials": results}, indent=2), encoding="utf-8")
    prepared.to_csv(clusters_path, index=False)
    return {"model_path": str(model_path), "metrics_path": str(metrics_path), "clusters_path": str(clusters_path), "metrics": best}


def train(config_path: str | Path | None = None):
    config = load_config(config_path)
    problem_type = config.get("project", {}).get("problem_type")
    if problem_type == "clustering":
        return train_clustering(config)
    if problem_type in {"ab_testing", "price_elasticity"}:
        from .analysis import run_analysis

        return run_analysis(config_path)
    return train_supervised(config)


def predict(config_path: str | Path | None, input_path: str | Path, output_path: str | Path | None = None):
    import joblib
    import pandas as pd

    config = load_config(config_path)
    model_path = resolve_project_path(config, config.get("modeling", {}).get("model_file", "models/model.joblib"))
    model = joblib.load(model_path)
    raw = read_csv(input_path)
    prepared = prepare_features(raw, config, training=False)
    X, _, _ = model_matrix(prepared, config, training=False)
    result = raw.copy()
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            result["score"] = proba[:, 1]
        else:
            for idx, klass in enumerate(model.classes_):
                result[f"score_{klass}"] = proba[:, idx]
    result["prediction"] = model.predict(X)
    output = Path(output_path) if output_path else resolve_project_path(config, "data/processed/predictions.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return output
