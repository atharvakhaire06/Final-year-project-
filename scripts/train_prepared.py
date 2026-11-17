"""Fit the model comparison from a saved spatial sample."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soc_pipeline.constants import CORE_CATEGORICAL, CORE_NUMERIC, TARGET
from soc_pipeline.modeling import fit_available_models
from soc_pipeline.stock import calculate_horizon_stock

parser=argparse.ArgumentParser(); parser.add_argument("--prepared", required=True); parser.add_argument("--output", required=True)
parser.add_argument("--seed",type=int,default=20261115); parser.add_argument("--rf-estimators",type=int,default=30); parser.add_argument("--boost-iterations",type=int,default=150)
args=parser.parse_args(); prepared=Path(args.prepared); out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
data=__import__('pandas').read_pickle(prepared / 'prepared_spatial_sample.pkl')
# OpenLandMap-soildb documents organic_carbon in g/kg.  Percentage is a
# deterministic unit conversion, retained alongside the source value.
data['soc_percent'] = data[TARGET] / 10.0
numeric=[c for c in CORE_NUMERIC if c in data]; categorical=[c for c in CORE_CATEGORICAL if c in data]
train, validation, test=(data.loc[data.split==x].copy() for x in ('train','validation','test'))
outcomes,_=fit_available_models(train,validation,test,target='soc_percent',numeric_features=numeric,categorical_features=categorical,seed=args.seed,rf_estimators=args.rf_estimators,boost_iterations=args.boost_iterations,output_dir=out/'models')
best=min((n for n,v in outcomes.items() if v['status']=='trained'),key=lambda n:outcomes[n]['validation']['rmse'])
stock=calculate_horizon_stock(test,'soc_percent',soc_unit='percent'); stock.to_csv(out/'test_horizon_stock.csv',index=False)
summary={'prepared_input':str(prepared),'sample_rows':len(data),'split_rows':data.split.value_counts().to_dict(),'target':{'source_column':TARGET,'source_unit':'g/kg','conversion':'soc_percent = organic_carbon / 10','model_output_unit':'percent'},'features':{'numeric':numeric,'categorical':categorical},'models':outcomes,'selected_by_validation_rmse':best,'stock':{'eligible_uncorrected_test_horizons':int(stock.soc_stock_uncorrected_t_c_ha.notna().sum()),'corrected_stock_available':False,'label':'uncorrected; coarse-fragment weight percent not applied'}}
(out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); (out/'run_config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
