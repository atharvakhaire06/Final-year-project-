from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soc_pipeline.data import inspect_dataset

parser = argparse.ArgumentParser(description="Stream-inspect supplied SOC data without assuming its schema.")
parser.add_argument("--input", required=True)
parser.add_argument("--output", default="artifacts/inspection")
args = parser.parse_args()
report = inspect_dataset(args.input, args.output)
print(json.dumps({key: report[key] for key in ("rows", "columns", "source_database_counts", "measurement_metadata_warning") if key in report}, indent=2))
print(f"Full report: {Path(args.output) / 'inspection_report.json'}")
