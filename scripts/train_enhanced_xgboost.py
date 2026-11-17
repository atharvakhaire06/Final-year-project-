"""Train the selected extended-feature XGBoost configuration reproducibly.

Configuration selection is based on spatial validation only.  The test partition
is scored only after fitting the selected configuration.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soc_pipeline.constants import CORE_NUMERIC, EXTENDED_NUMERIC, TARGET
from soc_pipeline.modeling import metrics
from soc_pipeline.stock import calculate_horizon_stock


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["soc_percent"] = data[TARGET] / 10.0
    data["horizon_thickness_cm"] = data["hzn_bot"] - data["hzn_top"]
    data["lat_sin"] = np.sin(np.deg2rad(data["latitude_decimal_degrees"]))
    data["lat_cos"] = np.cos(np.deg2rad(data["latitude_decimal_degrees"]))
    data["lon_sin"] = np.sin(np.deg2rad(data["longitude_decimal_degrees"]))
    data["lon_cos"] = np.cos(np.deg2rad(data["longitude_decimal_degrees"]))
    return data


def transform(frame: pd.DataFrame, numeric: list[str], categorical: list[str], encoder: OrdinalEncoder | None, fit: bool) -> tuple[pd.DataFrame, OrdinalEncoder]:
    number_frame = frame[numeric].apply(pd.to_numeric, errors="coerce")
    category_frame = frame[categorical].astype("string").fillna("<missing>")
    if fit:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
        encoded = encoder.fit_transform(category_frame)
    else:
        if encoder is None:
            raise ValueError("A fitted categorical encoder is required for prediction.")
        encoded = encoder.transform(category_frame)
    coded_frame = pd.DataFrame(encoded, columns=categorical, index=frame.index)
    return pd.concat([number_frame, coded_frame], axis=1), encoder


parser = argparse.ArgumentParser(description="Fit selected extended XGBoost SOC-% model.")
parser.add_argument("--prepared", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, default=20261115)
args = parser.parse_args()

prepared, output = Path(args.prepared), Path(args.output)
output.mkdir(parents=True, exist_ok=False)
data = add_features(pd.read_pickle(prepared / "prepared_spatial_sample.pkl"))
numeric = [c for c in CORE_NUMERIC + EXTENDED_NUMERIC + ["horizon_thickness_cm", "lat_sin", "lat_cos", "lon_sin", "lon_cos"] if c in data]
categorical = [c for c in ["hzn_desgn", "depth_category", "surface_subsurface"] if c in data]
train, validation, test = (data.loc[data["split"] == split].copy() for split in ("train", "validation", "test"))
x_train, encoder = transform(train, numeric, categorical, None, fit=True)
x_validation, _ = transform(validation, numeric, categorical, encoder, fit=False)
x_test, _ = transform(test, numeric, categorical, encoder, fit=False)

model = XGBRegressor(
    n_estimators=500, max_depth=8, learning_rate=0.035, subsample=0.85,
    colsample_bytree=0.9, min_child_weight=10, reg_lambda=10,
    objective="reg:squarederror", n_jobs=-1, random_state=args.seed,
)
model.fit(x_train, train["soc_percent"], eval_set=[(x_validation, validation["soc_percent"])], verbose=False)
predictions = {"train": model.predict(x_train), "validation": model.predict(x_validation), "test": model.predict(x_test)}
outcomes = {"xgboost_extended": {"status": "trained", **{name: metrics(partition["soc_percent"], predictions[name]) for name, partition in {"train": train, "validation": validation, "test": test}.items()}}}

models = output / "models"; models.mkdir()
joblib.dump({"model": model, "category_encoder": encoder, "features": numeric + categorical}, models / "xgboost.joblib")
pd.DataFrame({"feature": numeric + categorical, "importance": model.feature_importances_}).sort_values("importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
stock = test.copy(); stock["predicted_soc_percent"] = predictions["test"]
stock = calculate_horizon_stock(stock, "predicted_soc_percent", soc_unit="percent")
stock.to_csv(output / "test_predicted_horizon_stock.csv", index=False)
summary = {
    "prepared_input": str(prepared), "sample_rows": len(data), "split_rows": data["split"].value_counts().to_dict(),
    "target": {"source_column": TARGET, "source_unit": "g/kg", "conversion": "soc_percent = organic_carbon / 10", "model_output_unit": "percent"},
    "features": {"numeric": numeric, "categorical": categorical, "configuration": "extended + target-independent depth and coordinate transforms"},
    "selection": {"criterion": "spatial validation R²", "validation_r2": outcomes["xgboost_extended"]["validation"]["r2"], "test_not_used_for_selection": True},
    "models": outcomes, "selected_by_validation_rmse": "xgboost", "stock": {"eligible_uncorrected_test_horizons": int(stock["soc_stock_uncorrected_t_c_ha"].notna().sum()), "corrected_stock_available": False, "label": "uncorrected; coarse-fragment weight percent not applied"},
}
(output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(output / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
(models / "model_outcomes.json").write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
