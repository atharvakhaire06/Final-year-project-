"""Small Streamlit inference UI for the selected saved model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from soc_pipeline.stock import calculate_horizon_stock

st.set_page_config(page_title="SOC assessment", layout="wide")
st.title("Soil Organic Carbon assessment")
st.caption("Predictions are estimates, not substitutes for laboratory measurements.")
run_dir = st.sidebar.text_input("Artifact run directory", "artifacts/trained-soc-percent-enhanced")
path = Path(run_dir)
if not (path / "metrics.json").exists():
    st.info("Train a run first, then enter its artifacts directory.")
    st.stop()
metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
features = (json.loads((path / "feature_set.json").read_text(encoding="utf-8"))
            if (path / "feature_set.json").exists() else metrics["features"])
numeric_features = features.get("numeric_features", features.get("numeric", []))
categorical_features = features.get("categorical_features", features.get("categorical", []))
# Keep inference intentionally simple. The selected model uses native handling
# for the remaining, optional soil-property fields.
dashboard_numeric_inputs = [
    "clay_total", "silt_total", "sand_total", "ph_h2o", "hzn_top",
    "layer_sequence", "latitude_decimal_degrees", "longitude_decimal_degrees",
]
dashboard_categorical_inputs = ["hzn_desgn"]
model_name = metrics["selected_by_validation_rmse"]
if model_name == "random_forest":
    model = joblib.load(path / "models" / "random_forest.joblib")
elif model_name == "xgboost":
    model = joblib.load(path / "models" / "xgboost.joblib")
elif model_name == "catboost":
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(); model.load_model(str(path / "models" / "catboost.cbm"))
else:
    st.error(f"No dashboard loader is available for selected model: {model_name}")
    st.stop()
st.subheader("Validated model performance")
comparison_sources = [
    ("random_forest", Path("artifacts/trained-soc-percent/metrics.json"), "random_forest"),
    ("xgboost", Path("artifacts/trained-soc-percent-enhanced/metrics.json"), "xgboost_extended"),
    ("catboost", Path("artifacts/trained-soc-percent-catboost-improved/metrics.json"), "catboost_extended"),
]
comparison = {}
for display_name, metrics_path, stored_name in comparison_sources:
    if metrics_path.exists():
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        result = saved.get("models", {}).get(stored_name)
        if result and result.get("status") == "trained":
            comparison[display_name] = result["test"]
st.dataframe(pd.DataFrame(comparison).T)
with st.form("prediction"):
    values = {}
    left, right = st.columns(2)
    for i, feature in enumerate(dashboard_numeric_inputs):
        with (left if i % 2 == 0 else right):
            raw = st.text_input(feature, "")
            values[feature] = float(raw) if raw.strip() else None
    for feature in dashboard_categorical_inputs:
        values[feature] = st.text_input(feature, "") or None
    density = st.text_input("bulk_density_oven_dry (g/cm³)", "")
    submit = st.form_submit_button("Predict SOC")
if submit:
    row = pd.DataFrame([values])
    # Supply intentionally omitted model features as missing. XGBoost learned
    # missing-value branches from the training partition.
    for feature in numeric_features:
        if feature not in row:
            row[feature] = np.nan
    for feature in categorical_features:
        if feature not in row:
            row[feature] = None
    top = row.at[0, "hzn_top"]
    # The simplified dashboard standardises every estimate to the primary
    # 0–30 cm assessment interval. The entered top is retained and the lower
    # boundary is explicitly set to 30 cm for the uncorrected stock formula.
    row["hzn_bot"] = 30.0 if pd.notna(top) and top < 30 else np.nan
    if pd.notna(top) and top < 30:
        row["horizon_midpoint_cm"] = (top + 30.0) / 2
        row["horizon_thickness_cm"] = 30.0 - top
        midpoint = row.at[0, "horizon_midpoint_cm"]
        row["depth_category"] = "0-10" if midpoint <= 10 else "10-30" if midpoint <= 30 else "30-60" if midpoint <= 60 else "60-100" if midpoint <= 100 else "100+"
    row["surface_subsurface"] = "surface" if pd.notna(top) and top < 10 else "subsurface" if pd.notna(top) else None
    latitude, longitude = row.at[0, "latitude_decimal_degrees"], row.at[0, "longitude_decimal_degrees"]
    row["lat_sin"] = np.sin(np.deg2rad(latitude)) if pd.notna(latitude) else None
    row["lat_cos"] = np.cos(np.deg2rad(latitude)) if pd.notna(latitude) else None
    row["lon_sin"] = np.sin(np.deg2rad(longitude)) if pd.notna(longitude) else None
    row["lon_cos"] = np.cos(np.deg2rad(longitude)) if pd.notna(longitude) else None
    if model_name == "random_forest":
        prediction = float(model.predict(row)[0])
    elif model_name == "xgboost":
        x_row = row[model["features"]].copy()
        categories = categorical_features
        if categories:
            x_row.loc[:, categories] = model["category_encoder"].transform(x_row[categories].fillna("<missing>").astype(str))
        x_row = x_row.apply(pd.to_numeric, errors="coerce")
        prediction = float(model["model"].predict(x_row)[0])
    else:
        c_row = row[numeric_features + categorical_features].copy()
        for column in categorical_features:
            c_row[column] = c_row[column].fillna("<missing>").astype(str)
        prediction = float(model.predict(c_row)[0])
    target_info = metrics.get("target", {})
    output_unit = target_info.get("model_output_unit", "unverified")
    st.metric("Predicted SOC", f"{prediction:.3f}%", help="Model output is SOC concentration in percentage.")
    # Retain only valid supplied depth features; model uses hzn_top/hzn_bot.
    if density.strip():
        stock_row = row.copy(); stock_row["predicted_soc_percent"] = prediction; stock_row["bulk_density_oven_dry"] = float(density)
        stock = calculate_horizon_stock(stock_row, "predicted_soc_percent", soc_unit="percent" if output_unit == "percent" else None)
        value = stock.loc[0, "soc_stock_uncorrected_t_c_ha"]
        if pd.notna(value):
            st.metric("Uncorrected SOC stock (to 30 cm)", f"{value:.2f} t C/ha")
            st.caption(f"Assumes the entered soil properties represent the layer from {top:.1f} to 30 cm.")
        else:
            st.warning("SOC stock unavailable: enter an upper depth (`hzn_top`) below 30 cm and a valid bulk density.")
    elif output_unit != "percent":
        st.warning("SOC stock is unavailable because this artifact does not specify percentage SOC output.")
