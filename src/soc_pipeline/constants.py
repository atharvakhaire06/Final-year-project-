"""Schema candidates and intentional exclusion rules."""

TARGET = "organic_carbon"
COORDINATES = ["longitude_decimal_degrees", "latitude_decimal_degrees"]
DEPTH_COLUMNS = ["hzn_top", "hzn_bot", "hzn_depth"]
CORE_NUMERIC = [
    "clay_total", "silt_total", "sand_total", "ph_h2o",
    "horizon_midpoint_cm", "hzn_top", "hzn_bot",
    "layer_sequence", "latitude_decimal_degrees", "longitude_decimal_degrees",
]
CORE_CATEGORICAL = ["hzn_desgn"]
EXTENDED_NUMERIC = [
    "total_nitrogen_ncs", "cec_nh4_ph_7", "ca_ext", "mg_ext", "na_ext",
    "k_ext", "ph_kcl", "ph_cacl2", "ec_water_extractable", "ec_predict_one_to_two",
]
IDENTIFIER_CANDIDATES = ["site_key", "upedonid", "dsiteid", "olc_id"]
LEAKAGE_OR_METADATA = {
    "oc_d", "total_carbon_ncs", "site_key", "upedonid", "dsiteid", "olc_id",
    "source_db", "citation_url", "project_url", "uuid", "site_obsdate",
    "confidence_degree", "bulk_density_oven_dry", "total_frag_wt_pct_gt_2_mm_ws",
}
