"""Build notebook/Syria_DroughtLens.ipynb from the module files.

The notebook is generated, not hand-edited, so it always matches the code in
config.py / fetch.py / prepare.py / analysis.py / plots.py.
Run:  python build_notebook.py
"""
import re
from pathlib import Path

import nbformat as nbf

MODULES = ["config", "fetch", "prepare", "analysis", "risk_model", "plots"]
IMPORT_FROM_CONFIG = re.compile(r"^from config import .*\n", re.MULTILINE)


def module_source(name):
    """Module code with cross-module imports removed (everything shares one namespace)."""
    return IMPORT_FROM_CONFIG.sub("", Path(f"{name}.py").read_text()).strip() + "\n"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md("""
# Syria DroughtLens
**FTL Syria AI4Climate: Python for Climate Hackathon**

**Climate challenge:** Water and drought

**Research question:** Have drought conditions in Syria become more frequent or severe between 1981 and 2025, and which regions are most affected?

**Data:** NASA POWER Daily API (MERRA-2 reanalysis): rainfall (`PRECTOTCORR`, mm/day), temperature (`T2M`, C), plus soil wetness and other variables, 1981-2025, 8 Syrian locations.
https://power.larc.nasa.gov/docs/services/api/temporal/daily/

**How to run:** *Runtime > Run all*. The first run downloads the data (a few minutes) and caches it in `data/raw/`.
"""),
    md("""
## Definitions used throughout
- **Hydrological year:** October to September, named after the year it ends in (Oct 1981 - Sep 1982 = 1982). Only complete years 1982-2025 (44 years) are used.
- **Rainfall anomaly (z):** annual rainfall standardised per region: `(x - mean) / std`.
- **Dry year:** z <= -0.5. **Drought year:** z <= -1.
- **Hot year:** temperature z >= 0.5. **Hot-dry year:** dry *and* hot.
- **Periods:** 1982-2003 vs 2004-2025 (22 years each).
"""),
    md("## 1. Setup: configuration and functions"),
    code(module_source("config")),
    code(module_source("fetch")),
    code(module_source("prepare")),
    code(module_source("analysis")),
    code(module_source("risk_model")),
    code(module_source("plots")),
    md("""
## 2. Load and inspect the data
Downloads all 8 locations on the first run and caches them. `check_raw` verifies equal row counts, no duplicate dates, no missing flags, and that no two locations fell in the same NASA grid cell.
"""),
    code("""
from IPython.display import display

for path in (DATA_RAW, DATA_CLEAN, DATA_ANNUAL):
    path.parent.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

if DATA_RAW.exists():
    raw = load_raw(DATA_RAW)
else:
    raw = fetch_all()
    raw.to_csv(DATA_RAW)
    raw = load_raw(DATA_RAW)

check_raw(raw)
display(raw.head())
"""),
    md("## 3. Clean and prepare\nReplace `-999` flags, fill short temperature gaps (never rainfall), add hydrological year, keep complete years only."),
    code("""
df = clean(raw)
df.to_csv(DATA_CLEAN)
print(f"{df.shape[0]:,} daily rows, {df['hydro_year'].nunique()} hydrological years, {df['region'].nunique()} regions")
display(df.head())
display(df.describe().T)
"""),
    md("## 4. Annual indicators and drought flags\nOne row per region and hydrological year: total rainfall, mean temperature, soil wetness, longest wet-season dry spell, and the drought flags."),
    code("""
annual = add_flags(build_annual(df))
annual.to_csv(DATA_ANNUAL, index=False)
display(annual.head(10))
"""),
    md("## 5. Analysis 1: trends (change over time)\nSlope per decade for each region. `p_value < 0.05` counts as significant."),
    code("""
trends = trend_table(annual)
print("Slope per decade:")
display(trends.pivot(index="region", columns="variable", values="slope_per_decade").round(3))
print("p-values:")
display(trends.pivot(index="region", columns="variable", values="p_value").round(3))
"""),
    md("## 6. Analysis 2: comparison between periods\nMeans and counts for 1982-2003 vs 2004-2025, with percentage change in rainfall and temperature change."),
    code("""
comparison = period_comparison(annual)
display(comparison.round(2))
"""),
    md("## 7. Analysis 3: correlation and regional ranking\nRainfall vs temperature/soil wetness, and regions ranked by recent drought frequency and rainfall variability (CV)."),
    code("""
display(correlations(annual).round(2))
display(regional_ranking(annual).round(2))
"""),
    md("## 8. Analysis 4: are dry years getting hotter?\nShare of dry years that were also hot, per region and period, plus a pooled significance test."),
    code("""
share, dry_counts, hot_dry_counts = hot_dry_share(annual)
display(share.round(0))
print(hot_dry_test(annual))
"""),
    md("## 9. Visualisations"),
    md("### 9.1 Rainfall anomaly timeline\nRed bars are drought years (z <= -1)."),
    code("fig_anomaly_timeline(annual)"),
    md("### 9.2 Drought heatmap\nDark red columns mark widespread drought years."),
    code("fig_drought_heatmap(annual)"),
    md("### 9.3 Temperature trends"),
    code("fig_temperature_trends(annual)"),
    md("### 9.4 Headline: dry years and heat"),
    code("fig_hot_dry_share(annual, share)"),
    md("### 9.5 Regional vulnerability"),
    code("fig_vulnerability(annual)"),
    md("""
## 10. Early-warning risk model
**Question:** on 31 December, three months into the hydrological year, can we tell whether the year will end up dry?

- **Inputs (all known by 31 Dec):** Oct-Dec rainfall, Oct-Dec temperature, Oct-Dec soil wetness, previous year's rainfall.
- **Model:** logistic regression, one pooled model for all regions, inputs standardised per region.
- **Honest evaluation:** *walk-forward* validation. To predict year Y the model is trained only on years before Y, so it never sees the future.
- **Baselines:** always predicting the base rate, and a simple rule (alert if early rainfall z <= -0.5).
- **Alert:** issued when predicted probability >= the alert threshold (set in `config.py`).
"""),
    code("""
feats = build_features(df, annual)
preds = {name: walk_forward(feats, name) for name in FEATURE_SETS}
metrics = evaluate(preds)
display(metrics.round(3))

best = preds["rain_climate_model"]
y = best["dry"].astype(int)
alert = best["prob"] >= ALERT_THRESHOLD
print(f"Test years {best['hydro_year'].min()}-{best['hydro_year'].max()}, {len(best)} region-years, "
      f"{y.mean():.0%} of them dry.")
print(f"At the {ALERT_THRESHOLD:.0%} alert threshold the model flagged {alert[y == 1].mean():.0%} of dry years "
      f"with {alert[y == 0].mean():.0%} false alarms (ROC-AUC {metrics.loc['rain_climate_model', 'auc']:.2f}; 0.50 = no skill).")
"""),
    md("### 10.1 Does temperature add anything beyond rainfall?\nCompare `rain_model` with `rain_climate_model` in the table above, and look at the standardised coefficients (positive = raises dry-year risk)."),
    code("""
bundle = fit_final(feats)
display(coefficients(bundle).round(3).to_frame("standardised coefficient"))
"""),
    md("### 10.2 Skill and the early-warning timeline"),
    code("fig_risk_skill(metrics)"),
    code("fig_risk_heatmap(annual, preds['rain_climate_model'])"),
    md("### 10.3 Case studies: would the model have warned us?\nPredicted probability on 31 December for years that were widely dry (where inside the test period)."),
    code("""
case_years = [yr for yr in (2000, 2008, 2014, 2025) if yr in best["hydro_year"].values]
case = best[best["hydro_year"].isin(case_years)].copy()
case["prob_%"] = (case["prob"] * 100).round(0)
display(case.pivot(index="region", columns="hydro_year", values="prob_%"))
display(case.pivot(index="region", columns="hydro_year", values="dry").rename_axis(columns="actually dry?"))
"""),
    md("### 10.4 Decision-support function\nGiven a region and the first three months of data, return a risk level. This is the core of the proposed tool."),
    code("""
example = predict_risk(bundle, "Damascus", rain_early=40, prev_rain=250, temp_early=17.5, soil_early=0.35)
print(example)
"""),
    md("""
## 11. Findings
*Numbers below come from our run. Re-check them against the outputs above if the code or data change.*

1. **Warming is strong and universal:** +0.37 to +0.49 C per decade in all 8 regions (p < 0.001). The second period is 0.74-1.02 C warmer than the first.
2. **No decline in rainfall:** trends are positive in most regions and significant in Raqqa, Homs and Deir ez-Zor; the rest show no significant trend.
3. **Dry years did not become more frequent:** 60 dry region-years in 1982-2003 vs 48 in 2004-2025.
4. **But dry years now coincide with heat:** the share of dry years that were also hot rose from about 17% to 58% (pooled), and increased in every region.
5. **Most exposed regions:** Deir ez-Zor and Raqqa have the lowest rainfall and the highest variability.
6. **Validation:** the flagged drought years match known events (1989-90, 1999-2000, 2008, 2014, 2025).

## 12. Limitations
- NASA POWER is gridded reanalysis (about 50 km), not station data. Rainfall is model/satellite-derived and less reliable in arid areas; use it for trends and anomalies, not exact totals.
- Nearby places can share one grid cell, so results describe broad regional patterns, not individual cities.
- Only 44 years per region: small counts, and simple trend tests that ignore autocorrelation.
- Temperature z-scores use the full period, and all regions warmed, so the rise in hot-dry years partly reflects the warming trend itself. The honest claim is that dry years now occur on a warmer baseline.
- Pooled regions share droughts, so the pooled significance test is indicative only.
- No local water-use, irrigation or crop data.\n- Risk model: about 300 region-years, one pooled model, walk-forward test of roughly 30 years. Early-season rainfall is part of the annual total, so some skill is expected. The dry-year label uses full-period statistics, a mild simplification. Skill estimates are uncertain at this sample size.

## 13. Next steps
- Lagged drought-risk model (previous-year rainfall and temperature as predictors).
- Add soil-moisture and evaporation drought indices; compare with station data where available.
- Dashboard for decision support.
"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
Path("notebook").mkdir(exist_ok=True)
nbf.write(nb, "notebook/Syria_DroughtLens.ipynb")
print("Wrote notebook/Syria_DroughtLens.ipynb with", len(cells), "cells")
