"""Project configuration for Syria DroughtLens (single source of truth).

Every location, variable name, threshold and file path lives here, so that
changing a setting once changes it everywhere.
"""
from pathlib import Path

# ---- Study locations: name -> (latitude, longitude) ----------------------
# NASA POWER uses a ~0.5 x 0.625 degree grid, so nearby points can share a
# cell (e.g. Damascus city and its countryside). prepare.check_raw() verifies
# that no two locations returned identical rainfall.
LOCATIONS = {
    "Damascus":    (33.51, 36.29),
    "Aleppo":      (36.20, 37.16),
    "Homs":        (34.73, 36.72),
    "Latakia":     (35.52, 35.79),
    "Raqqa":       (35.95, 39.01),
    "Deir ez-Zor": (35.34, 40.14),
    "Hasakah":     (36.50, 40.75),
    "Daraa":       (32.63, 36.10),
}

# ---- NASA POWER request ---------------------------------------------------
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
COMMUNITY = "AG"
CORE_PARAMS = ["PRECTOTCORR", "T2M"]          # rainfall (mm/day), mean temp (C)
EXTRA_PARAMS = ["T2M_MAX", "T2M_MIN", "RH2M", "GWETROOT", "GWETTOP", "EVLAND"]
START_DATE, END_DATE = "19810101", "20251231"
MISSING_VALUE = -999                          # NASA POWER's missing-data flag

# ---- Analysis settings ----------------------------------------------------
# Hydrological year runs Oct-Sep and is named after the year it ends in
# (Oct 1981 - Sep 1982 = 1982). Only complete years 1982-2025 are used.
FIRST_HYDRO_YEAR, LAST_HYDRO_YEAR = 1982, 2025
SPLIT_YEAR = 2003                             # last year of the first period
PERIODS = ("1982-2003", "2004-2025")          # two equal halves, 22 years each
WET_SEASON_MONTHS = (11, 12, 1, 2, 3)         # used for dry-spell length
DRY_DAY_MM = 1.0                              # a day below this is a "dry day"
DRY_Z = -0.5                                  # dry year:     rainfall z <= -0.5
DROUGHT_Z = -1.0                              # drought year: rainfall z <= -1.0
HOT_Z = 0.5                                   # hot year:     temperature z >= 0.5
SIGNIFICANCE = 0.05

# ---- Paths (relative to the folder you run from) --------------------------
DATA_RAW = Path("data/raw/power_raw.csv")
DATA_CLEAN = Path("data/processed/power_clean.csv")
DATA_ANNUAL = Path("data/processed/annual.csv")
TABLES_DIR = Path("outputs/tables")
FIG_DIR = Path("outputs/figures")

# ---- Early-warning risk model ---------------------------------------------
# At the end of the cut-off months (31 Dec) we predict whether the hydrological
# year will end up dry. Only data up to that date is used as input.
CUTOFF_MONTHS = (10, 11, 12)    # Oct-Dec = first three months of the hydrological year
MIN_TRAIN_YEARS = 12            # walk-forward evaluation starts after this many training years
ALERT_THRESHOLD = 0.30          # predicted probability that triggers an alert
RISK_BANDS = (0.30, 0.50)       # low < 0.30 <= moderate < 0.50 <= high
