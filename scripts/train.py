from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soc_pipeline.constants import CORE_CATEGORICAL, CORE_NUMERIC, EXTENDED_NUMERIC, TARGET
from soc_pipeline.data import choose_profile_group, inspect_dataset, list_columns, prepare_primary_frame, read_input
from soc_pipeline.modeling import fit_available_models, make_spatial_splits
from soc_pipeline.stock import calculate_horizon_stock

parser = argparse.ArgumentParser(description="Train spatially validated SOC models with saved audit artefacts.")
parser.add_argument("--input", required=True)
parser.add_argument("--output", default=None, help="Default: artifacts/run-UTC timestamp")
parser.add_argument("--seed", type=int, default=20261115)
parser.add_argument("--block-size-degrees", type=float, default=5.0)
parser.add_argument("--max-rows", type=int, default=None, help="Deterministic first-N development run; omit for all rows.")
parser.add_argument("--rf-estimators", type=int, default=200)
parser.add_argument("--extended", action="store_true", help="Priority 2 comparison; core is always trained first.")
parser.add_argument("--soc-unit", choices=["percent"], default=None, help="Set only after source documentation verifies target values are percent.")
args = parser.parse_args()
run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
out = Path(args.output or Path("artifacts") / run_id); out.mkdir(parents=True, exist_ok=False)
columns = list_columns(args.input)
needed = set(CORE_NUMERIC + CORE_CATEGORICAL + EXTENDED_NUMERIC + [TARGET, "hzn_depth", "bulk_density_oven_dry", "total_frag_wt_pct_gt_2_mm_ws", "site_key", "upedonid", "dsiteid", "olc_id"])
available = [c for c in columns if c in needed]
data = read_input(args.input, usecols=available, nrows=args.max_rows)
clean, audit = prepare_primary_frame(data)
audit.to_csv(out / "preprocessing_audit.csv", index=False)
group_column, groups, group_info = choose_profile_group(clean)
split, assignments = make_spatial_splits(clean, groups, seed=args.seed, block_size_degrees=args.block_size_degrees)
assignments.to_csv(out / "split_indices.csv", index=False)
clean["split"] = split.values
numeric = [c for c in CORE_NUMERIC if c in clean.columns]
categorical = [c for c in CORE_CATEGORICAL if c in clean.columns]
if args.extended:
    numeric += [c for c in EXTENDED_NUMERIC if c in clean.columns]
feature_set = {"configuration": "extended" if args.extended else "core", "numeric_features": numeric, "categorical_features": categorical, "unavailable_requested_core_features": [c for c in CORE_NUMERIC + CORE_CATEGORICAL if c not in clean.columns]}
(out / "feature_set.json").write_text(json.dumps(feature_set, indent=2), encoding="utf-8")
train, validation, test = (clean.loc[clean.split == name].copy() for name in ("train", "validation", "test"))
if min(len(train), len(validation), len(test)) == 0: raise RuntimeError("A spatial split was empty; choose a smaller block size or inspect coordinates.")
outcomes, fitted = fit_available_models(train, validation, test, target=TARGET, numeric_features=numeric, categorical_features=categorical, seed=args.seed, rf_estimators=args.rf_estimators, output_dir=out / "models")
trained = {name: value for name, value in outcomes.items() if value["status"] == "trained"}
best = min(trained, key=lambda name: trained[name]["validation"]["rmse"])
stock = calculate_horizon_stock(test, TARGET, soc_unit=args.soc_unit)
stock.to_csv(out / "test_horizon_stock.csv", index=False)
summary = {"run_id": run_id, "input": str(args.input), "input_rows_loaded": len(data), "rows_after_primary_filter": len(clean), "split_rows": clean.split.value_counts().to_dict(), "profile_group": group_info, "spatial_split": {"block_size_degrees": args.block_size_degrees, "seed": args.seed}, "target": {"column": TARGET, "unit_status": args.soc_unit or "unverified; stock withheld"}, "models": outcomes, "selected_by_validation_rmse": best, "stock": {"eligible_uncorrected_test_horizons": int(stock.soc_stock_uncorrected_t_c_ha.notna().sum()), "corrected_stock_available": False, "reason": "coarse-fragment data are weight percent, not validated volume percent"}}
(out / "metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
(out / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
print(json.dumps(summary, indent=2, default=str))
