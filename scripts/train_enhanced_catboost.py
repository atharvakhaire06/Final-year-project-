"""Train an extended-feature CatBoost SOC-% model with spatial validation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soc_pipeline.constants import CORE_NUMERIC, EXTENDED_NUMERIC, TARGET
from soc_pipeline.modeling import metrics
from soc_pipeline.stock import calculate_horizon_stock


def engineer(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["soc_percent"] = data[TARGET] / 10.0
    data["horizon_thickness_cm"] = data["hzn_bot"] - data["hzn_top"]
    for prefix, source in (("lat", "latitude_decimal_degrees"), ("lon", "longitude_decimal_degrees")):
        data[f"{prefix}_sin"] = np.sin(np.deg2rad(data[source]))
        data[f"{prefix}_cos"] = np.cos(np.deg2rad(data[source]))
    return data


parser = argparse.ArgumentParser()
parser.add_argument("--prepared", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, default=20261115)
parser.add_argument("--train-rows", type=int, default=None, help="Optional reproducible subset of the already spatially separated training partition.")
parser.add_argument("--iterations", type=int, default=800)
args = parser.parse_args()
output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
data = engineer(pd.read_pickle(Path(args.prepared) / "prepared_spatial_sample.pkl"))
numeric = [col for col in CORE_NUMERIC + EXTENDED_NUMERIC + ["horizon_thickness_cm", "lat_sin", "lat_cos", "lon_sin", "lon_cos"] if col in data]
categorical = [col for col in ["hzn_desgn", "depth_category", "surface_subsurface"] if col in data]
features = numeric + categorical
train, validation, test = (data.loc[data["split"] == split].copy() for split in ("train", "validation", "test"))
if args.train_rows is not None:
    train = train.sample(n=min(args.train_rows, len(train)), random_state=args.seed).copy()
for frame in (train, validation, test):
    for column in categorical:
        frame[column] = frame[column].astype("string").fillna("<missing>").astype(str)

model = CatBoostRegressor(
    iterations=args.iterations, depth=7, learning_rate=.05, l2_leaf_reg=10,
    random_strength=.5, bagging_temperature=1, loss_function="RMSE",
    verbose=100, random_seed=args.seed, thread_count=-1,
)
model.fit(train[features], train["soc_percent"], cat_features=categorical,
          eval_set=(validation[features], validation["soc_percent"]), early_stopping_rounds=100,
          verbose=100)
predictions = {"train": model.predict(train[features]), "validation": model.predict(validation[features]), "test": model.predict(test[features])}
outcomes = {"catboost_extended": {"status": "trained", **{name: metrics(frame["soc_percent"], predictions[name]) for name, frame in {"train": train, "validation": validation, "test": test}.items()}}}
models = output / "models"; models.mkdir()
model.save_model(str(models / "catboost.cbm"))
(models / "catboost_metadata.json").write_text(json.dumps({"features": features, "categorical": categorical}, indent=2), encoding="utf-8")
pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
stock = test.copy(); stock["predicted_soc_percent"] = predictions["test"]
calculate_horizon_stock(stock, "predicted_soc_percent", soc_unit="percent").to_csv(output / "test_predicted_horizon_stock.csv", index=False)
summary = {"prepared_input": args.prepared, "sample_rows": len(data), "train_rows_used": len(train), "split_rows": data["split"].value_counts().to_dict(), "target": {"source_column": TARGET, "source_unit": "g/kg", "conversion": "soc_percent = organic_carbon / 10", "model_output_unit": "percent"}, "features": {"numeric": numeric, "categorical": categorical, "configuration": "extended + target-independent depth and coordinate transforms"}, "selection": {"criterion": "spatial validation R²", "validation_r2": outcomes["catboost_extended"]["validation"]["r2"], "test_not_used_for_selection": True}, "models": outcomes, "selected_by_validation_rmse": "catboost", "stock": {"eligible_uncorrected_test_horizons": int(stock.bulk_density_oven_dry.notna().sum()), "corrected_stock_available": False}}
(output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(output / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
