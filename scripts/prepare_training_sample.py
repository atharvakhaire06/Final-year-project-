"""Create a reproducible spatially stratified development sample from all raw rows."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soc_pipeline.constants import CORE_CATEGORICAL, CORE_NUMERIC, EXTENDED_NUMERIC, TARGET
from soc_pipeline.data import choose_profile_group, list_columns, prepare_primary_frame, read_input
from soc_pipeline.modeling import make_spatial_splits

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True); parser.add_argument("--output", required=True)
parser.add_argument("--rows", type=int, default=120000); parser.add_argument("--seed", type=int, default=20261115)
parser.add_argument("--block-size-degrees", type=float, default=5.0)
args = parser.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
columns = list_columns(args.input)
needed = set(CORE_NUMERIC + CORE_CATEGORICAL + EXTENDED_NUMERIC + [TARGET, "hzn_depth", "bulk_density_oven_dry", "total_frag_wt_pct_gt_2_mm_ws", "site_key", "upedonid", "dsiteid", "olc_id"])
raw = read_input(args.input, usecols=[c for c in columns if c in needed])
clean, audit = prepare_primary_frame(raw); group_col, groups, group_info = choose_profile_group(clean)
split, assignments = make_spatial_splits(clean, groups, seed=args.seed, block_size_degrees=args.block_size_degrees)
clean["split"] = split.values; clean["profile_group"] = groups.values; clean["spatial_block"] = assignments["spatial_block"].values
# Keep split proportions while sampling rows deterministically. Profile grouping and split membership remain intact.
pieces = []
for name, subset in clean.groupby("split", sort=False):
    n = max(1, round(args.rows * len(subset) / len(clean)))
    pieces.append(subset.sample(n=min(n, len(subset)), random_state=args.seed))
sample = pd.concat(pieces, ignore_index=True)
sample.to_pickle(out / "prepared_spatial_sample.pkl")
sample[["source_row", "profile_group", "spatial_block", "split"]].to_csv(out / "sample_split_indices.csv", index=False)
audit.to_csv(out / "preprocessing_audit.csv", index=False)
(out / "preparation.json").write_text(json.dumps({"input_rows": len(raw), "primary_rows": len(clean), "sample_rows": len(sample), "seed": args.seed, "block_size_degrees": args.block_size_degrees, "profile_group": group_info, "split_rows_full": clean.split.value_counts().to_dict(), "split_rows_sample": sample.split.value_counts().to_dict()}, indent=2), encoding="utf-8")
print((out / "preparation.json").read_text())
