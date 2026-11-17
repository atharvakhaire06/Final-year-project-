"""Input loading, schema inspection, and transparent primary-experiment filtering."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import COORDINATES, DEPTH_COLUMNS, IDENTIFIER_CANDIDATES, TARGET


def read_input(path: str | Path, usecols: Iterable[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Load CSV/RDS. RDS support is explicit because it needs an optional reader."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=usecols, nrows=nrows, low_memory=False)
    if path.suffix.lower() == ".rds":
        try:
            import pyreadr
        except ImportError as exc:
            raise RuntimeError("RDS input needs pyreadr. Install it, then rerun.") from exc
        objects = pyreadr.read_r(str(path))
        if len(objects) != 1:
            raise ValueError(f"Expected one table in RDS; found {len(objects)} objects.")
        frame = next(iter(objects.values()))
        return frame.loc[:, list(usecols)] if usecols else frame
    raise ValueError(f"Unsupported input format: {path.suffix}")


def list_columns(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=0).columns.tolist()
    return read_input(path, nrows=1).columns.tolist()


def _numeric_summary(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "min": None, "p01": None, "median": None, "p99": None, "max": None}
    return {
        "count": int(values.size), "min": float(values.min()), "p01": float(values.quantile(.01)),
        "median": float(values.median()), "p99": float(values.quantile(.99)), "max": float(values.max()),
    }


def inspect_dataset(path: str | Path, output_dir: str | Path, chunksize: int = 100_000) -> dict:
    """Stream a CSV inspection report so a large raw dataset need not fit RAM."""
    path, output_dir = Path(path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = list_columns(path)
    if path.suffix.lower() != ".csv":
        frame = read_input(path)
        return _inspect_frame(frame, output_dir)
    numeric = [c for c in [*COORDINATES, *DEPTH_COLUMNS, TARGET, "clay_total", "silt_total", "sand_total", "ph_h2o", "bulk_density_oven_dry"] if c in columns]
    missing = Counter(); source_counts = Counter(); rows = 0
    samples: list[pd.DataFrame] = []
    ids = {c: set() for c in IDENTIFIER_CANDIDATES if c in columns}
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        rows += len(chunk)
        missing.update(chunk.isna().sum().astype(int).to_dict())
        if "source_db" in chunk:
            source_counts.update(chunk["source_db"].fillna("<missing>").astype(str).value_counts().to_dict())
        for identifier, values in ids.items():
            ids[identifier].update(chunk[identifier].dropna().astype(str))
        # Deterministic, bounded sample for quantiles and relationships.
        samples.append(chunk.head(max(1, min(2_000, len(chunk)))))
    sample = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(columns=columns)
    report = {
        "input": str(path), "format": "csv", "rows": rows, "columns": len(columns), "column_names": columns,
        "missing_percent": {c: round(100 * missing[c] / rows, 4) for c in columns},
        "source_database_counts": dict(source_counts.most_common()),
        "unique_non_null_identifiers": {c: len(v) for c, v in ids.items()},
        "numeric_sample_summary": {c: _numeric_summary(sample[c]) for c in numeric},
        "measurement_metadata_warning": "No explicit units or laboratory-method columns were identified by name in the supplied CSV. Target units must be verified from source documentation before a stock calculation.",
        "sample_quality_checks": quality_checks(sample),
        "profile_group_diagnostics": profile_group_diagnostics(sample),
    }
    _write_report(report, output_dir)
    return report


def _inspect_frame(frame: pd.DataFrame, output_dir: Path) -> dict:
    report = {
        "rows": len(frame), "columns": len(frame.columns), "column_names": frame.columns.tolist(),
        "missing_percent": (frame.isna().mean().mul(100).round(4)).to_dict(),
        "numeric_sample_summary": {c: _numeric_summary(frame[c]) for c in frame.select_dtypes("number")},
        "sample_quality_checks": quality_checks(frame), "profile_group_diagnostics": profile_group_diagnostics(frame),
    }
    _write_report(report, output_dir)
    return report


def _write_report(report: dict, output_dir: Path) -> None:
    (output_dir / "inspection_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame({"column": list(report["missing_percent"]), "missing_percent": list(report["missing_percent"].values())}).sort_values("missing_percent", ascending=False).to_csv(output_dir / "missingness.csv", index=False)


def quality_checks(frame: pd.DataFrame) -> dict:
    def number(name: str) -> pd.Series:
        return pd.to_numeric(frame[name], errors="coerce") if name in frame else pd.Series(dtype=float)
    top, bottom, soc = number("hzn_top"), number("hzn_bot"), number(TARGET)
    checks = {"rows_sampled": int(len(frame)), "negative_soc": int((soc < 0).sum()), "zero_soc": int((soc == 0).sum())}
    if not top.empty and not bottom.empty:
        checks["invalid_depth_rows"] = int(((top < 0) | (bottom <= top)).sum())
    if {"clay_total", "silt_total", "sand_total"}.issubset(frame):
        texture = frame[["clay_total", "silt_total", "sand_total"]].apply(pd.to_numeric, errors="coerce")
        complete = texture.notna().all(axis=1); sums = texture.sum(axis=1)
        checks.update({"complete_texture_rows": int(complete.sum()), "texture_sum_95_to_105": int((complete & sums.between(95, 105)).sum()), "texture_outside_0_to_100": int(((texture < 0) | (texture > 100)).any(axis=1).sum())})
    return checks


def profile_group_diagnostics(frame: pd.DataFrame) -> dict:
    results = {}
    for name in IDENTIFIER_CANDIDATES:
        if name not in frame:
            continue
        values = frame[name].dropna().astype(str)
        if values.empty:
            continue
        counts = values.value_counts()
        results[name] = {"non_missing_rows": int(values.size), "unique_values": int(counts.size), "rows_in_repeated_values": int(counts[counts > 1].sum()), "max_rows_per_value": int(counts.max())}
    return results


def prepare_primary_frame(frame: pd.DataFrame, *, max_depth_cm: float = 30.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply only domain checks required for the primary 0–30 cm experiment."""
    data = frame.copy(); data["source_row"] = np.arange(len(data), dtype=np.int64)
    audit: list[dict] = [{"stage": "loaded", "rows_remaining": len(data), "rows_removed": 0}]
    numeric_columns = [TARGET, *COORDINATES, "hzn_top", "hzn_bot", "hzn_depth", "clay_total", "silt_total", "sand_total", "ph_h2o", "bulk_density_oven_dry", "layer_sequence"]
    for col in set(numeric_columns).intersection(data.columns): data[col] = pd.to_numeric(data[col], errors="coerce")
    required = [TARGET, *COORDINATES, "hzn_top", "hzn_bot"]
    absent = [c for c in required if c not in data]
    if absent: raise ValueError(f"Primary experiment cannot proceed; missing required columns: {absent}")
    def filter_stage(name: str, mask: pd.Series):
        nonlocal data
        before = len(data); data = data.loc[mask].copy()
        audit.append({"stage": name, "rows_remaining": len(data), "rows_removed": before - len(data)})
    filter_stage("missing_target", data[TARGET].notna())
    filter_stage("negative_target", data[TARGET] >= 0)
    filter_stage("missing_or_invalid_coordinates", data[COORDINATES].notna().all(axis=1) & data["latitude_decimal_degrees"].between(-90, 90) & data["longitude_decimal_degrees"].between(-180, 180))
    filter_stage("invalid_depth", data["hzn_top"].notna() & data["hzn_bot"].notna() & (data["hzn_top"] >= 0) & (data["hzn_bot"] > data["hzn_top"]))
    filter_stage("outside_0_to_30_cm", (data["hzn_top"] < max_depth_cm) & (data["hzn_bot"] > 0))
    data["horizon_thickness_cm"] = data["hzn_bot"] - data["hzn_top"]
    data["horizon_midpoint_cm"] = (data["hzn_top"] + data["hzn_bot"]) / 2
    data["overlap_0_30_cm"] = (np.minimum(data["hzn_bot"], max_depth_cm) - np.maximum(data["hzn_top"], 0)).clip(lower=0)
    data["depth_category"] = pd.cut(data["horizon_midpoint_cm"], [-np.inf, 10, 30, 60, 100, np.inf], labels=["0-10", "10-30", "30-60", "60-100", "100+"])
    data["surface_subsurface"] = np.where(data["hzn_top"] < 10, "surface", "subsurface")
    if "hzn_depth" in data:
        data["reported_midpoint_difference_cm"] = data["hzn_depth"] - data["horizon_midpoint_cm"]
    return data, pd.DataFrame(audit)


def choose_profile_group(frame: pd.DataFrame) -> tuple[str, pd.Series, dict]:
    """Select a real identifier by observed repetition, with traceable fallback."""
    choices = []
    for col in IDENTIFIER_CANDIDATES:
        if col not in frame: continue
        text = frame[col].astype("string")
        counts = text.dropna().value_counts(); repeated = int(counts[counts > 1].sum())
        choices.append((repeated, int(text.notna().sum()), col, text))
    if not choices: raise ValueError("No candidate profile identifier exists in input.")
    repeated, present, name, values = max(choices, key=lambda item: (item[0], item[1]))
    # A missing profile identifier cannot be allowed to join unrelated horizons.
    fallback = "row_" + frame["source_row"].astype(str)
    group = values.fillna(fallback).astype(str)
    return name, group, {"selected_group_column": name, "repeated_rows": repeated, "non_missing_rows": present, "missing_identifier_rows_made_singletons": int(values.isna().sum())}
