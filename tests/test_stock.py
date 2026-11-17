import pandas as pd
from soc_pipeline.stock import calculate_horizon_stock


def test_percent_formula_and_partial_horizon():
    rows = pd.DataFrame({"soc": [2.0], "bulk_density_oven_dry": [1.3], "hzn_top": [20], "hzn_bot": [50]})
    got = calculate_horizon_stock(rows, "soc", soc_unit="percent")
    assert got.loc[0, "stock_overlap_thickness_cm"] == 10
    assert got.loc[0, "soc_stock_uncorrected_t_c_ha"] == 26.0
    assert pd.isna(got.loc[0, "soc_stock_corrected_t_c_ha"])


def test_stock_withheld_without_unit_verification():
    rows = pd.DataFrame({"soc": [2.0], "bulk_density_oven_dry": [1.3], "hzn_top": [0], "hzn_bot": [10]})
    got = calculate_horizon_stock(rows, "soc", soc_unit=None)
    assert pd.isna(got.loc[0, "soc_stock_uncorrected_t_c_ha"])
    assert got.loc[0, "stock_eligibility_reason"] == "soc_unit_unverified"
