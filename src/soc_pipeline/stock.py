"""SOC-stock functions that refuse unsafe assumptions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_horizon_stock(frame: pd.DataFrame, soc_column: str, *, soc_unit: str | None,
                            max_depth_cm: float = 30.0) -> pd.DataFrame:
    """Return layer stocks without converting coarse-fragment weight % to volume %.

    Formula for percent SOC, g/cm3 BD, cm thickness yields t C/ha.  Corrected
    stock is intentionally unavailable because the supplied fragment field is
    weight %, whereas the correction requires volume %.
    """
    result = frame.copy()
    required = [soc_column, "bulk_density_oven_dry", "hzn_top", "hzn_bot"]
    for column in required:
        if column not in result:
            raise KeyError(f"Stock calculation requires {column!r}")
    top = pd.to_numeric(result["hzn_top"], errors="coerce")
    bottom = pd.to_numeric(result["hzn_bot"], errors="coerce")
    bd = pd.to_numeric(result["bulk_density_oven_dry"], errors="coerce")
    soc = pd.to_numeric(result[soc_column], errors="coerce")
    overlap = (np.minimum(bottom, max_depth_cm) - np.maximum(top, 0)).clip(lower=0)
    valid = soc.notna() & bd.notna() & (bd > 0) & top.notna() & bottom.notna() & (bottom > top) & (overlap > 0)
    result["stock_overlap_thickness_cm"] = overlap
    result["stock_eligibility_reason"] = np.select(
        [soc.isna(), bd.isna(), bd <= 0, top.isna() | bottom.isna(), bottom <= top, overlap <= 0],
        ["missing_soc", "missing_bulk_density", "invalid_bulk_density", "missing_depth", "invalid_depth", "outside_target_depth"],
        default="eligible" if soc_unit == "percent" else "soc_unit_unverified",
    )
    eligible = valid & (soc_unit == "percent")
    result["soc_stock_uncorrected_t_c_ha"] = np.where(eligible, soc * bd * overlap, np.nan)
    result["soc_stock_corrected_t_c_ha"] = np.nan
    result["stock_correction_note"] = "Unavailable: supplied coarse-fragment field is weight percent, not validated volume percent."
    return result
