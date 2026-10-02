# Syria DroughtLens

FTL Syria AI4Climate: Python for Climate Hackathon (28 Sep - 2 Oct 2026). Challenge area: **water and drought**.

**Research question:** Have drought conditions in Syria become more frequent or severe between 1981 and 2025, and which regions are most affected?

**Data:** NASA POWER Daily API (MERRA-2), 1981-2025, 8 locations: rainfall (`PRECTOTCORR`), temperature (`T2M`, `T2M_MAX`), soil wetness (`GWETROOT`), evaporation (`EVLAND`). https://power.larc.nasa.gov/

## Project layout

```
config.py            all settings: locations, variables, thresholds, paths
fetch.py             download from NASA POWER
prepare.py           load, validate, clean, hydrological-year features
analysis.py          annual indicators, drought flags, trends, comparisons
risk_model.py        early-warning model: features, walk-forward validation, scoring
plots.py             the seven figures
main.py              runs everything: python main.py
build_notebook.py    generates notebook/Syria_DroughtLens.ipynb from the modules
notebook/            submission notebook (generated)
data/raw/            downloaded data (cached, not committed)
data/processed/      cleaned daily data and annual table
outputs/tables/      result tables (CSV)
outputs/figures/     PNG figures for the slides
```

## Run

```bash
pip install -r requirements.txt
python main.py            # downloads on first run, then reuses data/raw/power_raw.csv
python main.py --fetch    # force a fresh download
```

Or run `notebook/Syria_DroughtLens.ipynb`

After editing any module, regenerate the notebook: `python build_notebook.py`.

## Method in brief

1. **Prepare:** replace `-999`, fill short temperature gaps (never rainfall), hydrological year Oct-Sep, keep complete years 1982-2025.
2. **Indicators:** annual rainfall, mean temperature, soil wetness, longest wet-season dry spell.
3. **Flags:** per-region z-scores. Dry year: rainfall z <= -0.5. Drought: z <= -1. Hot-dry: dry and temperature z >= 0.5.
4. **Analyses:** trends (OLS, per decade), first vs second half (1982-2003 vs 2004-2025), correlation, regional ranking, share of dry years that are also hot (with Fisher test).

5. **Early-warning model:** on 31 December, predict whether the year will end up dry from Oct-Dec rainfall, temperature and soil wetness plus last year's rainfall. Logistic regression, one pooled model, walk-forward validation (train only on earlier years), compared with a base-rate baseline and a simple rainfall rule.

## Limitations

Gridded reanalysis (about 50 km), not station data; rainfall less reliable in arid areas. Only 44 years per region. Temperature z-scores use the full period, so the rise in hot-dry years partly reflects warming itself. Pooled regions share droughts, so pooled p-values are indicative only. The risk model rests on about 300 region-years, so its skill estimates are uncertain; early-season rainfall is part of the annual total, so some skill is expected.
