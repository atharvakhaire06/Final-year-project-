"""Leakage-safe spatial splitting, model fitting, and evaluation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def make_spatial_splits(frame: pd.DataFrame, groups: pd.Series, *, seed: int, block_size_degrees: float = 5.0,
                        validation_fraction: float = .15, test_fraction: float = .15) -> tuple[pd.Series, pd.DataFrame]:
    """Assign whole profile groups to geographic grid blocks, never row-wise splits."""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation/test fractions must be positive and sum to less than one.")
    locations = pd.DataFrame({"group": groups.astype(str), "lat": frame["latitude_decimal_degrees"], "lon": frame["longitude_decimal_degrees"]})
    centroids = locations.groupby("group", sort=False)[["lat", "lon"]].median()
    centroids["lat_block"] = np.floor((centroids["lat"] + 90) / block_size_degrees).astype(int)
    centroids["lon_block"] = np.floor((centroids["lon"] + 180) / block_size_degrees).astype(int)
    centroids["spatial_block"] = centroids["lat_block"].astype(str) + "_" + centroids["lon_block"].astype(str)
    block_counts = centroids.groupby("spatial_block").size().sort_values(ascending=False)
    rng = np.random.default_rng(seed)
    ordered_blocks = rng.permutation(block_counts.index.to_numpy())
    target_test, target_val = len(centroids) * test_fraction, len(centroids) * validation_fraction
    accumulated = {"test": 0, "validation": 0}; mapping = {}
    for block in ordered_blocks:
        n = int(block_counts[block])
        if accumulated["test"] < target_test:
            split = "test"; accumulated[split] += n
        elif accumulated["validation"] < target_val:
            split = "validation"; accumulated[split] += n
        else:
            split = "train"
        mapping[block] = split
    centroids["split"] = centroids["spatial_block"].map(mapping)
    split_by_group = centroids["split"]
    split = groups.astype(str).map(split_by_group)
    assignments = frame[["source_row"]].copy(); assignments["profile_group"] = groups.astype(str); assignments["split"] = split.values
    assignments["spatial_block"] = groups.astype(str).map(centroids["spatial_block"]).values
    return split, assignments


def build_rf_pipeline(numeric_features: list[str], categorical_features: list[str], *, seed: int, n_estimators: int) -> Pipeline:
    transforms = []
    if numeric_features:
        transforms.append(("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric_features))
    if categorical_features:
        transforms.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), categorical_features))
    return Pipeline([
        ("preprocess", ColumnTransformer(transforms, remainder="drop")),
        ("model", RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=2, max_features=.8, n_jobs=-1, random_state=seed)),
    ])


def metrics(y_true: pd.Series, prediction: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)), "r2": float(r2_score(y_true, prediction)),
        # sqrt() works with both current sklearn and releases predating the
        # `squared` keyword on mean_squared_error.
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "mean_prediction_bias": float(np.mean(prediction - np.asarray(y_true))),
    }


def _encode_for_boosting(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, numeric: list[str], categorical: list[str]):
    """Ordinal-encode categories from training only; preserve numeric NaNs for XGBoost."""
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    train_x, val_x, test_x = train[numeric + categorical].copy(), validation[numeric + categorical].copy(), test[numeric + categorical].copy()
    if categorical:
        train_categories = train_x[categorical].fillna("<missing>").astype(str)
        validation_categories = val_x[categorical].fillna("<missing>").astype(str)
        test_categories = test_x[categorical].fillna("<missing>").astype(str)
        train_x.loc[:, categorical] = encoder.fit_transform(train_categories)
        val_x.loc[:, categorical] = encoder.transform(validation_categories)
        test_x.loc[:, categorical] = encoder.transform(test_categories)
    # Assignment into a mixed pandas frame can retain object dtype; XGBoost
    # requires an explicitly numeric matrix even for ordinal-coded categories.
    train_x = train_x.apply(pd.to_numeric, errors="coerce")
    val_x = val_x.apply(pd.to_numeric, errors="coerce")
    test_x = test_x.apply(pd.to_numeric, errors="coerce")
    return train_x, val_x, test_x, encoder


def fit_available_models(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, *, target: str,
                         numeric_features: list[str], categorical_features: list[str], seed: int,
                         rf_estimators: int, boost_iterations: int = 500,
                         output_dir: str | Path = ".") -> tuple[dict, dict]:
    """Train RF and optional XGBoost/CatBoost. Never disguise absent dependencies as results."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    y_train, y_val, y_test = train[target], validation[target], test[target]
    outcomes: dict = {}; fitted: dict = {}
    rf = build_rf_pipeline(numeric_features, categorical_features, seed=seed, n_estimators=rf_estimators)
    rf.fit(train, y_train); fitted["random_forest"] = rf
    outcomes["random_forest"] = {"status": "trained", "train": metrics(y_train, rf.predict(train)), "validation": metrics(y_val, rf.predict(validation)), "test": metrics(y_test, rf.predict(test))}
    joblib.dump(rf, out / "random_forest.joblib")
    if importlib.util.find_spec("xgboost"):
        x_train, x_val, x_test, encoder = _encode_for_boosting(train, validation, test, numeric_features, categorical_features)
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=boost_iterations, max_depth=8, learning_rate=.05, subsample=.8, colsample_bytree=.8, objective="reg:squarederror", n_jobs=-1, random_state=seed)
        xgb.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        fitted["xgboost"] = (xgb, encoder)
        outcomes["xgboost"] = {"status": "trained", "train": metrics(y_train, xgb.predict(x_train)), "validation": metrics(y_val, xgb.predict(x_val)), "test": metrics(y_test, xgb.predict(x_test))}
        joblib.dump({"model": xgb, "category_encoder": encoder, "features": numeric_features + categorical_features}, out / "xgboost.joblib")
    else:
        outcomes["xgboost"] = {"status": "skipped", "reason": "xgboost is not installed"}
    if importlib.util.find_spec("catboost"):
        from catboost import CatBoostRegressor
        features = numeric_features + categorical_features
        c_train, c_val, c_test = train[features].copy(), validation[features].copy(), test[features].copy()
        for c in categorical_features:
            c_train[c] = c_train[c].astype("string").fillna("<missing>"); c_val[c] = c_val[c].astype("string").fillna("<missing>"); c_test[c] = c_test[c].astype("string").fillna("<missing>")
        cat_indices = [features.index(c) for c in categorical_features]
        cat = CatBoostRegressor(iterations=boost_iterations, depth=8, learning_rate=.05, loss_function="RMSE", verbose=False, random_seed=seed, thread_count=-1)
        cat.fit(c_train, y_train, cat_features=cat_indices, eval_set=(c_val, y_val), early_stopping_rounds=50, verbose=False)
        fitted["catboost"] = cat
        outcomes["catboost"] = {"status": "trained", "train": metrics(y_train, cat.predict(c_train)), "validation": metrics(y_val, cat.predict(c_val)), "test": metrics(y_test, cat.predict(c_test))}
        cat.save_model(str(out / "catboost.cbm")); joblib.dump({"features": features, "categorical_features": categorical_features}, out / "catboost_metadata.joblib")
    else:
        outcomes["catboost"] = {"status": "skipped", "reason": "catboost is not installed"}
    (out / "model_outcomes.json").write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
    return outcomes, fitted
