"""Step 3: annual indicators, drought detection, trends and regional comparison."""
import numpy as np
import pandas as pd
from scipy import stats

from config import PERIODS, SPLIT_YEAR, WET_SEASON_MONTHS, DRY_DAY_MM, DRY_Z, DROUGHT_Z, HOT_Z, SIGNIFICANCE


def longest_run(mask):
    """Length of the longest run of consecutive True values."""
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _zscore(series):
    return (series - series.mean()) / series.std()


def build_annual(df):
    """One row per region and hydrological year."""
    agg = {"rain_mm": ("PRECTOTCORR", "sum"), "temp_mean": ("T2M", "mean")}
    optional = {  # only used if the extra variables were downloaded
        "tmax_mean": ("T2M_MAX", "mean"),
        "soil_root": ("GWETROOT", "mean"),
        "evap": ("EVLAND", "mean"),
    }
    for name, (column, func) in optional.items():
        if column in df.columns:
            agg[name] = (column, func)
    annual = df.groupby(["region", "hydro_year"]).agg(**agg).reset_index()

    # Longest run of dry days (<1 mm) inside the wet season (Nov-Mar).
    # Summer is dry every year, so a full-year dry spell would say nothing.
    wet = df[df["month"].isin(WET_SEASON_MONTHS)]
    spell = (wet.groupby(["region", "hydro_year"])["PRECTOTCORR"]
                .apply(lambda s: longest_run((s < DRY_DAY_MM).values))
                .rename("wet_dry_spell")
                .reset_index())
    return annual.merge(spell, on=["region", "hydro_year"])


def add_flags(annual):
    """Add standardised anomalies, period label and drought flags.

    z-scores use each region's own 1982-2025 mean and standard deviation.
    Because every region warmed, temperature z-scores are higher in the
    second period by construction. That is intended (dry years now happen on
    a warmer baseline) but it must not be read as extra dry years.
    """
    a = annual.copy()
    a["rain_z"] = a.groupby("region")["rain_mm"].transform(_zscore)
    a["temp_z"] = a.groupby("region")["temp_mean"].transform(_zscore)
    a["period"] = np.where(a["hydro_year"] <= SPLIT_YEAR, PERIODS[0], PERIODS[1])
    a["dry"] = a["rain_z"] <= DRY_Z
    a["drought"] = a["rain_z"] <= DROUGHT_Z
    a["hot_dry"] = a["dry"] & (a["temp_z"] >= HOT_Z)
    return a


def trend_table(annual, variables=("rain_mm", "temp_mean", "tmax_mean", "soil_root")):
    """Linear trend per region: slope per decade and p-value.

    Note: 44 points per region and simple OLS. Year-to-year autocorrelation
    is ignored, and 8 regions x several variables means several tests, so
    treat p-values near 0.05 as borderline.
    """
    rows = []
    for region, d in annual.groupby("region"):
        for var in variables:
            if var not in d.columns:
                continue
            res = stats.linregress(d["hydro_year"], d[var])
            rows.append({"region": region, "variable": var,
                         "slope_per_decade": res.slope * 10, "p_value": res.pvalue})
    out = pd.DataFrame(rows)
    out["significant"] = out["p_value"] < SIGNIFICANCE
    return out


def period_comparison(annual):
    """First half vs second half: means, counts and changes."""
    first, second = PERIODS
    g = (annual.groupby(["region", "period"])
               .agg(rain=("rain_mm", "mean"), temp=("temp_mean", "mean"),
                    dry_years=("dry", "sum"), droughts=("drought", "sum"),
                    hot_dry_years=("hot_dry", "sum"))
               .unstack("period"))
    out = pd.DataFrame(index=g.index)
    for col in g.columns.levels[0]:
        out[f"{col}_{first}"] = g[(col, first)]
        out[f"{col}_{second}"] = g[(col, second)]
    out["rain_change_pct"] = (out[f"rain_{second}"] / out[f"rain_{first}"] - 1) * 100
    out["temp_change_C"] = out[f"temp_{second}"] - out[f"temp_{first}"]
    return out


def correlations(annual):
    """Correlation between annual rainfall and temperature / soil wetness."""
    rows = []
    for region, d in annual.groupby("region"):
        row = {"region": region, "rain_vs_temp": d["rain_mm"].corr(d["temp_mean"])}
        if "soil_root" in d.columns:
            row["rain_vs_soil"] = d["rain_mm"].corr(d["soil_root"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("region")


def regional_ranking(annual):
    """Rank regions by recent drought frequency, with rainfall variability."""
    second = PERIODS[1]
    g = annual.groupby("region")
    out = pd.DataFrame({
        "mean_rain_mm": g["rain_mm"].mean(),
        "rain_cv": g["rain_mm"].std() / g["rain_mm"].mean(),
        "drought_years": g["drought"].sum(),
        "recent_droughts": annual[annual["period"] == second].groupby("region")["drought"].sum(),
        "hot_dry_years": g["hot_dry"].sum(),
    })
    return out.sort_values("recent_droughts", ascending=False)


def hot_dry_share(annual):
    """Percentage of dry years that were also hot, per region and period.

    Returns (share, dry_counts, hot_dry_counts). The last row of `share`
    pools all regions.
    """
    dry = annual.groupby(["region", "period"])["dry"].sum().unstack("period")
    hot_dry = annual.groupby(["region", "period"])["hot_dry"].sum().unstack("period")
    share = hot_dry / dry * 100
    share.loc["ALL REGIONS"] = hot_dry.sum() / dry.sum() * 100
    return share, dry, hot_dry


def hot_dry_test(annual):
    """Fisher exact test: is a dry year more likely to be hot in period 2?

    Caveat: regions share droughts (one dry year often hits several regions),
    so pooled observations are not independent. The p-value is indicative.
    """
    first, second = PERIODS

    def counts(period):
        d = annual[(annual["period"] == period) & annual["dry"]]
        return int(d["hot_dry"].sum()), int((~d["hot_dry"]).sum())

    hot2, not2 = counts(second)
    hot1, not1 = counts(first)
    odds, p = stats.fisher_exact([[hot2, not2], [hot1, not1]])
    return {f"hot_dry_{first}": hot1, f"dry_{first}": hot1 + not1,
            f"hot_dry_{second}": hot2, f"dry_{second}": hot2 + not2,
            "odds_ratio": odds, "p_value": p}
