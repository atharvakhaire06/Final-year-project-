# Soil Organic Carbon (SOC) prediction

This project implements the Priority 1 SOC concentration pipeline for the supplied OpenLandMap horizon dataset. It deliberately does **not** add rasters, predict bulk density, or convert coarse-fragment weight percent to volume percent.

## Reproducible workflow

Use the Python interpreter that has the listed packages installed.

```powershell
python scripts/inspect_dataset.py --input dataset/sol_chem_pnts_horizons_3.csv
python scripts/train.py --input dataset/sol_chem_pnts_horizons_3.csv --soc-unit percent
streamlit run app.py
```

For this supplied OpenLandMap-soildb dataset, `organic_carbon` is documented in g/kg. Convert it with `SOC (%) = organic_carbon / 10`; stocks must remain labelled **uncorrected** because the coarse-fragment field is weight %, not volume %.

Outputs are saved under `artifacts/<run-id>/`: configuration, inspection report, row-removal audit, saved spatial split assignments, metrics, feature list, model artefacts, and eligible uncorrected stock records.

## Input formats

CSV is supported directly. RDS is supported when `pyreadr` is installed. The current supplied file is a CSV, despite the PRD naming an RDS input.

## Guardrails

- Profile groups, not individual horizons, are assigned to spatial blocks.
- `oc_d`, total carbon, identifiers, source database, and metadata are never predictors.
- Only horizons overlapping 0–30 cm are eligible for the primary experiment; overlap is retained for stock calculations.
- Imputation is fit only on training rows for Random Forest. XGBoost and CatBoost use their own missing-value mechanisms when installed.
- Coarse-fragment weight percent is never applied as a volumetric correction. Eligible stock is explicitly **uncorrected**.
