from pathlib import Path
from datetime import date

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT = Path("reports/SOC_Machine_Learning_Project_Report.docx")


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_border(cell, color="D9D9D9", size="6"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        el = borders.find(tag)
        if el is None:
            el = OxmlElement(f"w:{edge}")
            borders.append(el)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), size)
        el.set(qn("w:color"), color)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_cell_text(cell, text, bold=False, color="000000", size=9.2, align=WD_ALIGN_PARAGRAPH.LEFT):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.05
    r = p.add_run(str(text))
    r.bold = bold
    r.font.name = "Aptos"
    r._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    r._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
    r.font.size = Pt(size)
    r.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    set_cell_margins(cell)
    set_cell_border(cell)


def add_table(doc, headers, rows, widths=None, font_size=9.2):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.style = "Table Grid"
    hdr = table.rows[0]
    set_repeat_table_header(hdr)
    for i, header in enumerate(headers):
        set_cell_text(hdr.cells[i], header, bold=True, color="FFFFFF", size=font_size, align=WD_ALIGN_PARAGRAPH.CENTER)
        set_cell_shading(hdr.cells[i], "17365D")
        if widths:
            hdr.cells[i].width = Inches(widths[i])
    for row_idx, values in enumerate(rows):
        cells = table.add_row().cells
        for i, value in enumerate(values):
            align = WD_ALIGN_PARAGRAPH.CENTER if i == 0 and len(headers) <= 3 else WD_ALIGN_PARAGRAPH.LEFT
            set_cell_text(cells[i], value, size=font_size, align=align)
            if row_idx % 2:
                set_cell_shading(cells[i], "F4F7FA")
            if widths:
                cells[i].width = Inches(widths[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def add_body(doc, text, bold_lead=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.14
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        r.bold = True
        p.add_run(text[len(bold_lead):])
    else:
        p.add_run(text)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.08
    p.add_run(text)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.space_before = Pt(13 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(5)
    p.add_run(text)
    return p


def field_rows():
    return [
        ("site_key", "Source site identifier"),
        ("upedonid", "Source pedon or profile identifier"),
        ("site_obsdate", "Observation date"),
        ("longitude_decimal_degrees", "Longitude in decimal degrees"),
        ("latitude_decimal_degrees", "Latitude in decimal degrees"),
        ("SSL_classification_name", "Soil classification label supplied by the source"),
        ("labsampnum", "Laboratory sample number"),
        ("layer_key", "Source horizon or layer identifier"),
        ("layer_sequence", "Within-profile layer order"),
        ("hzn_top", "Upper horizon depth in cm"),
        ("hzn_bot", "Lower horizon depth in cm"),
        ("hzn_desgn", "Horizon designation or label"),
        ("texture_description", "Reported texture description"),
        ("texture_lab", "Laboratory texture descriptor"),
        ("clay_total", "Clay fraction, percent"),
        ("silt_total", "Silt fraction, percent"),
        ("sand_total", "Sand fraction, percent"),
        ("organic_carbon", "Organic carbon concentration in g/kg; raw modelling target"),
        ("oc_d", "Organic-carbon density field; not used because it is target-derived"),
        ("total_carbon_ncs", "Total carbon measurement"),
        ("total_nitrogen_ncs", "Total nitrogen measurement"),
        ("ph_kcl", "pH measured in potassium chloride"),
        ("ph_h2o", "pH measured in water"),
        ("ph_cacl2", "pH measured in calcium chloride"),
        ("sum_of_cations_cec_pH_8_2", "Cation-exchange measure at pH 8.2"),
        ("cec_nh4_ph_7", "Cation-exchange capacity measured with ammonium at pH 7"),
        ("ecec_base_plus_aluminum", "Effective CEC from base cations and aluminium"),
        ("total_frag_wt_pct_gt_2_mm_ws", "Coarse fragments above 2 mm by weight percent"),
        ("bulk_density_oven_dry", "Oven-dry bulk density"),
        ("ca_ext", "Extractable calcium"),
        ("mg_ext", "Extractable magnesium"),
        ("na_ext", "Extractable sodium"),
        ("k_ext", "Extractable potassium"),
        ("ec_water_extractable", "Water-extractable electrical conductivity"),
        ("ec_predict_one_to_two", "Predicted electrical-conductivity field"),
        ("source_db", "Contributing source database"),
        ("confidence_degree", "Source confidence indicator"),
        ("project_url", "Source project link"),
        ("citation_url", "Source citation link"),
        ("hzn_depth", "Reported horizon depth"),
        ("uuid", "Record UUID"),
        ("olc_id", "OpenLandMap profile identifier used for grouped splitting"),
        ("dsiteid", "Source dataset/site identifier"),
        ("layer_sequence.f", "Formatted layer-sequence field"),
    ]


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(0.72)
    sec.bottom_margin = Inches(0.70)
    sec.left_margin = Inches(0.72)
    sec.right_margin = Inches(0.72)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    for name, size in (("Title", 24), ("Heading 1", 15), ("Heading 2", 11.5)):
        style = styles[name]
        style.font.name = "Aptos Display" if name == "Title" else "Aptos"
        style._element.rPr.rFonts.set(qn("w:ascii"), style.font.name)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), style.font.name)
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = True

    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Soil Organic Carbon Assessment Project Report | 17 November 2026").font.size = Pt(8)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.add_run("Global Geospatial Machine Learning for Soil Organic Carbon Prediction and Stock Assessment")
    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_before = Pt(12)
    subtitle.paragraph_format.space_after = Pt(12)
    run = subtitle.add_run("Dataset, feature engineering, modelling approach and dashboard performance")
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(70, 70, 70)
    meta = doc.add_paragraph()
    meta.paragraph_format.space_after = Pt(18)
    meta.add_run("Prepared from the supplied OpenLandMap horizon dataset and recorded training artifacts\n17 November 2026").italic = True

    add_body(doc, "This report documents the implemented soil organic carbon prediction pipeline. It verifies the supplied source dataset, records each preprocessing decision, distinguishes the 44 source fields from the final model inputs, and reports the dashboard's final held-out test results. The deployed prediction is SOC percent; the dashboard also computes an uncorrected stock estimate to 30 cm when bulk density is entered.")

    add_heading(doc, "1 Project Dataset")
    add_body(doc, "The supplied source is the OpenLandMap soil-chemistry horizon dataset. The project inspected the original RDS and the supplied CSV conversion; the two versions agree for the organic-carbon target. The dataset contains 1,518,934 horizon observations and 44 fields. The original RDS is 75.39 MB and the CSV conversion is 501.76 MB.")
    add_body(doc, "The raw target is organic_carbon. OpenLandMap documentation identifies this field as g/kg. For modelling and dashboard display, it was converted to SOC percent using SOC percent = organic carbon g/kg divided by 10. This is a unit conversion, not a normalization step.")

    add_heading(doc, "Source Fields and Meaning", 2)
    add_body(doc, "All 44 supplied fields are listed below. Field descriptions reflect the supplied names and the source documentation; identifiers and metadata were retained for auditability but not used as predictive inputs.")
    add_table(doc, ["Field", "What it indicates"], field_rows(), widths=[2.45, 4.55], font_size=8.3)

    add_heading(doc, "Data Quality and Availability", 2)
    add_table(doc, ["Check", "Observed result"], [
        ("Target availability", "organic_carbon available for 1,068,424 rows; 29.66% missing in the full source."),
        ("Coordinates", "Latitude and longitude have no missing values in the source."),
        ("Texture", "Clay, silt and sand are each about 37.26% missing; among 16,311 complete sampled texture records, all sums were within 95 to 105%."),
        ("pH in water", "ph_h2o is 48.72% missing."),
        ("Bulk density", "bulk_density_oven_dry is 81.82% missing; it is therefore requested as a dashboard input for stock rather than predicted."),
        ("Coarse fragments", "65.76% missing and recorded as weight percent, so it cannot support a volume-based stock correction."),
        ("Target checks", "No negative SOC values found in the 32,000-row inspection sample; 743 sampled records were zero."),
    ], widths=[2.0, 5.0], font_size=8.8)

    add_heading(doc, "2 Feature Engineering and Preprocessing")
    add_body(doc, "Preprocessing was recorded in a row-level audit so no observations were silently removed. The analysis is deliberately limited to horizons that overlap the 0 to 30 cm layer used by the dashboard.")
    add_table(doc, ["Stage", "Rows remaining", "Rows removed at stage"], [
        ("Loaded source", "1,518,934", "0"),
        ("Removed missing target", "1,068,424", "450,510"),
        ("Removed negative target", "1,068,424", "0"),
        ("Removed missing or invalid coordinates", "1,068,424", "0"),
        ("Removed invalid depth geometry", "1,054,188", "14,236"),
        ("Retained horizons overlapping 0 to 30 cm", "606,886", "447,302"),
    ], widths=[3.2, 1.75, 2.05], font_size=8.8)

    add_heading(doc, "Engineered Fields and Split Design", 2)
    add_bullet(doc, "Derived depth fields: horizon_thickness_cm = hzn_bot minus hzn_top; horizon_midpoint_cm; and the overlap of each horizon with the 0 to 30 cm interval.")
    add_bullet(doc, "Derived categories: depth_category from the calculated midpoint and surface_subsurface from hzn_top. These describe depth without using the target.")
    add_bullet(doc, "Derived geographic transforms: sine and cosine of latitude and longitude. These preserve the cyclic character of longitude for the enhanced models.")
    add_bullet(doc, "Leakage-resistant partitioning: all rows with the same olc_id were kept together, then assigned by 5-degree geographic blocks. The seeded split is 80,853 training, 21,237 validation and 17,909 untouched test rows in the reproducible 119,999-row training sample. Seed: 20261115.")

    add_heading(doc, "Features Considered and Excluded", 2)
    add_table(doc, ["Role", "Fields"], [
        ("Core numeric predictors", "clay_total, silt_total, sand_total, ph_h2o, horizon_midpoint_cm, hzn_top, hzn_bot, layer_sequence, latitude_decimal_degrees, longitude_decimal_degrees"),
        ("Core categorical predictor", "hzn_desgn"),
        ("Enhanced chemical predictors", "total_nitrogen_ncs, cec_nh4_ph_7, ca_ext, mg_ext, na_ext, k_ext, ph_kcl, ph_cacl2, ec_water_extractable, ec_predict_one_to_two"),
        ("Enhanced engineered predictors", "horizon_thickness_cm, lat_sin, lat_cos, lon_sin, lon_cos, depth_category, surface_subsurface"),
        ("Explicitly excluded", "organic_carbon as the target; oc_d and total_carbon_ncs because they are target-derived or target-proximate; source identifiers and metadata; bulk_density_oven_dry and total_frag_wt_pct_gt_2_mm_ws because they are reserved for stock handling, not SOC prediction."),
    ], widths=[1.8, 5.2], font_size=8.6)

    add_heading(doc, "Missing Values and Scaling", 2)
    add_body(doc, "No numeric feature standardization or min-max normalization was applied. The selected algorithms are tree ensembles, which do not require scaled input variables. Texture percentages were checked for plausible bounds and sums but were not renormalized, avoiding unsupported alteration of laboratory measurements. The g/kg-to-percent target conversion is the only numeric conversion applied to SOC.")
    add_bullet(doc, "Random Forest: numeric medians and categorical most-frequent values were learned on training data only, followed by ordinal encoding of categories.")
    add_bullet(doc, "XGBoost: numeric missing values were handled natively; categorical labels were encoded using training data only, with unseen labels mapped to a reserved value.")
    add_bullet(doc, "CatBoost: numeric missing values were handled natively and missing categorical values were represented explicitly as a missing category.")

    add_heading(doc, "3 Models Used and Rationale")
    add_body(doc, "Three tree-based regressors were evaluated on the same spatially separated test set. Tree models suit mixed soil attributes, nonlinear depth effects, geographic variation, and substantial missingness without requiring feature scaling.")
    add_table(doc, ["Model", "Role and reason for use", "Final result"], [
        ("Random Forest", "Robust nonlinear baseline. Its bagged trees provide a stable reference under imputation and heterogeneous soil records.", "Baseline comparator"),
        ("XGBoost", "Gradient-boosted decision trees with native handling of numeric missing values. It was selected for the dashboard after the strongest validation and final test performance.", "Production SOC model"),
        ("CatBoost", "Gradient boosting designed for categorical variables and missing values. It was retained as a second boosted-model comparison after tuning.", "Comparison model"),
    ], widths=[1.25, 4.45, 1.3], font_size=8.8)
    add_body(doc, "Model selection used spatial validation performance, not the untouched test set. The enhanced XGBoost configuration uses the core, enhanced chemical and engineered depth/geographic features. The improved CatBoost used the same feature set but trained on a reproducible 35,000-row subset of the training partition for the available execution budget.")

    add_heading(doc, "4 Dashboard Results")
    add_body(doc, "The dashboard presents the following final held-out test results. Each model was evaluated on the same 17,909-row spatial test partition. R2 is higher-is-better; RMSE and MAE are expressed in SOC percentage points; mean prediction bias is prediction minus observed SOC percentage points.")
    add_table(doc, ["Dashboard model", "Test n", "R2", "RMSE", "MAE", "Mean bias"], [
        ("Random Forest", "17,909", "0.3052", "6.3145", "2.7787", "-0.3311"),
        ("XGBoost", "17,909", "0.3766", "5.9811", "2.2915", "-0.7954"),
        ("CatBoost", "17,909", "0.2416", "6.5974", "3.8750", "1.5850"),
    ], widths=[1.55, 0.75, 0.65, 1.0, 0.9, 1.15], font_size=8.6)
    add_body(doc, "XGBoost is the deployed prediction model because it has the highest held-out R2 (0.3766) and the lowest RMSE (5.9811) and MAE (2.2915) of the final dashboard models.")

    add_heading(doc, "Dashboard Inputs and SOC Stock", 2)
    add_body(doc, "The simplified dashboard requests clay_total, silt_total, sand_total, ph_h2o, hzn_top, layer_sequence, latitude_decimal_degrees, longitude_decimal_degrees, hzn_desgn and bulk_density_oven_dry. It derives the remaining production features internally and sets the lower depth to 30 cm for its 0 to 30 cm display calculation.")
    add_body(doc, "The SOC stock result is intentionally labelled uncorrected. It uses the displayed SOC percent, the entered oven-dry bulk density and the entered upper depth through 30 cm. No coarse-fragment correction is applied because the available field is weight percent rather than validated volume percent. The dashboard does not infer bulk density when it is unavailable.")
    add_table(doc, ["Worked dashboard input", "Displayed result"], [
        ("Clay 8.7%, silt 19.5%, sand 71.8%, pH 8.2, hzn_top 0 cm, layer sequence 11, latitude 36.8106499, longitude -100.6323547, hzn_desgn A, bulk density 1.75 g/cm3", "Predicted SOC: 0.959%\nUncorrected SOC stock to 30 cm: 50.36 t C/ha"),
    ], widths=[4.55, 2.45], font_size=8.8)

    add_heading(doc, "5 Reproducibility Records")
    add_body(doc, "The project preserves the preprocessing audit, grouped split configuration, fitted artifacts and metric files. The documented seed, geographic-block size, split indices and held-out test partition allow the reported dashboard comparison to be reproduced without changing the evaluation set.")

    add_heading(doc, "References", 1)
    add_body(doc, "OpenLandMap SoilDB README. Organic carbon is documented as g/kg. https://github.com/openlandmap/soildb/blob/main/README.md")
    add_body(doc, "OpenLandMap SoilDB chemical-data import documentation. organic_carbon is documented as g/kg. https://soildb.openlandmap.org/025-import_chemical_data.html")

    doc.core_properties.title = "Global Geospatial Machine Learning for Soil Organic Carbon Prediction and Stock Assessment"
    doc.core_properties.subject = "Dataset, feature engineering, model evaluation and dashboard results"
    doc.core_properties.author = "Soil Organic Carbon Assessment Project"
    doc.save(OUT)


if __name__ == "__main__":
    build()
