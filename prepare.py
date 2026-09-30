"""Step 2: load, validate and clean the raw data."""
import numpy as np
import pandas as pd

from config import MISSING_VALUE, FIRST_HYDRO_YEAR, LAST_HYDRO_YEAR


def load_raw(path):
    """Read the raw CSV written by fetch_all()."""
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def check_raw(raw):
    """Sanity-check the raw data. Raises an error on serious problems."""
    counts = raw.groupby("region").size()
    print("Rows per region:")
    print(counts.to_string())
    if counts.nunique() != 1:
        raise ValueError("Regions have different numbers of rows")

    for region, d in raw.groupby("region"):
        if not d.index.is_unique:
            raise ValueError(f"{region}: duplicate dates")
        if not d.index.is_monotonic_increasing:
            raise ValueError(f"{region}: dates are not sorted")

    # Two points in the same grid cell return identical data and would be
    # double-counted as separate regions.
    rain = raw.reset_index().pivot(index="date", columns="region", values="PRECTOTCORR")
    names = list(rain.columns)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if np.allclose(rain[a], rain[b], equal_nan=True):
                raise ValueError(f"{a} and {b} have identical rainfall (same grid cell). Remove one.")

    numeric = raw.drop(columns="region")
    print("\nMissing-value flags (-999) per column:")
    print((numeric == MISSING_VALUE).sum().to_string())
    print("\nEmpty (NaN) cells per column:")
    print(numeric.isna().sum().to_string())


def clean(raw):
    """Clean the raw data and add time features.

    - -999 becomes NaN.
    - Short gaps (up to 3 days) are interpolated for temperature-type
      variables only. Rainfall is never interpolated: it is not a smooth signal.
    - Adds year, month and hydrological year (Oct-Sep, named by the end year).
    - Keeps only complete hydrological years (1982-2025).
    """
    df = raw.replace(MISSING_VALUE, np.nan)
    numeric_cols = [c for c in df.columns if c != "region"]
    fill_cols = [c for c in numeric_cols if c != "PRECTOTCORR"]
    df[fill_cols] = df.groupby("region")[fill_cols].transform(lambda s: s.interpolate(limit=3))

    df["year"] = df.index.year
    df["month"] = df.index.month
    df["hydro_year"] = np.where(df["month"] >= 10, df["year"] + 1, df["year"])
    df = df[(df["hydro_year"] >= FIRST_HYDRO_YEAR) & (df["hydro_year"] <= LAST_HYDRO_YEAR)].copy()

    remaining = df[numeric_cols].isna().sum()
    if remaining.any():
        print("WARNING: missing values remain after cleaning:")
        print(remaining[remaining > 0].to_string())
    return df
