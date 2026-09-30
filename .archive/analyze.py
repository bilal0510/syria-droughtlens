import pandas as pd, numpy as np
from scipy import stats

df = pd.read_csv("power_clean.csv", parse_dates=["date"], index_col="date")

# ---------- 1. Annual table (region x hydrological year) ----------
def longest_run(mask):
    """Longest run of consecutive True values."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best

g = df.groupby(["region", "hydro_year"])
annual = pd.DataFrame({
    "rain_mm":   g["PRECTOTCORR"].sum(),
    "temp_mean": g["T2M"].mean(),
    "tmax_mean": g["T2M_MAX"].mean(),
    "soil_root": g["GWETROOT"].mean(),
    "evap":      g["EVLAND"].mean(),
    "hot_days":  g["T2M_MAX"].apply(lambda s: (s >= 35).sum()),
}).reset_index()

# Wet-season (Nov-Mar) longest dry spell, dry day = < 1 mm
wet = df[df["month"].isin([11, 12, 1, 2, 3])]
dry_spell = (wet.groupby(["region", "hydro_year"])["PRECTOTCORR"]
                .apply(lambda s: longest_run((s < 1).values))
                .rename("wet_dry_spell").reset_index())
annual = annual.merge(dry_spell, on=["region", "hydro_year"])

# ---------- 2. Drought detection (standardised rainfall anomaly) ----------
annual["rain_z"] = annual.groupby("region")["rain_mm"].transform(lambda s: (s - s.mean()) / s.std())
annual["drought"] = annual["rain_z"] <= -1          # moderate drought or worse
annual["severe"]  = annual["rain_z"] <= -1.5

# ---------- 3. Trends per region (slope per decade + p-value) ----------
rows = []
for region, d in annual.groupby("region"):
    for var in ["rain_mm", "temp_mean", "tmax_mean", "soil_root"]:
        res = stats.linregress(d["hydro_year"], d[var])
        rows.append({"region": region, "variable": var,
                     "slope_per_decade": res.slope * 10, "p_value": res.pvalue})
trends = pd.DataFrame(rows)
trends["significant"] = trends["p_value"] < 0.05
print(trends.pivot(index="region", columns="variable", values="slope_per_decade").round(3))
print(trends.pivot(index="region", columns="variable", values="p_value").round(3))

# ---------- 4. Period comparison ----------
annual["period"] = np.where(annual["hydro_year"] <= 2003, "1982-2003", "2004-2025")
cmp_ = annual.groupby(["region", "period"]).agg(
    rain=("rain_mm", "mean"), temp=("temp_mean", "mean"),
    droughts=("drought", "sum")).unstack()
cmp_["rain_change_%"] = (cmp_[("rain", "2004-2025")] / cmp_[("rain", "1982-2003")] - 1) * 100
cmp_["temp_change_C"] = cmp_[("temp", "2004-2025")] - cmp_[("temp", "1982-2003")]
print(cmp_.round(2))

# ---------- 5. Correlations (per region) ----------
corr = annual.groupby("region").apply(lambda d: pd.Series({
    "rain_vs_temp": d["rain_mm"].corr(d["temp_mean"]),
    "rain_vs_soil": d["rain_mm"].corr(d["soil_root"]),
})).round(2)
print(corr)

# ---------- 6. Regional ranking ----------
rank = annual.groupby("region").agg(
    mean_rain=("rain_mm", "mean"),
    rain_cv=("rain_mm", lambda s: s.std() / s.mean()),   # variability
    drought_years=("drought", "sum"),
    recent_droughts=("drought", lambda s: s[annual.loc[s.index, "hydro_year"] >= 2004].sum()),
).sort_values("recent_droughts", ascending=False)
print(rank.round(2))

annual.to_csv("annual.csv", index=False)


# 1. Sanity check: which years were flagged as drought?
flagged = annual[annual["drought"]].groupby("hydro_year")["region"].apply(list)
print(flagged)

# 2. Compound hot-dry years: dry AND warmer than the region's long-term mean
annual["temp_z"] = annual.groupby("region")["temp_mean"].transform(lambda s: (s - s.mean()) / s.std())
annual["hot_dry"] = (annual["rain_z"] <= -0.5) & (annual["temp_z"] >= 0.5)
print(annual.groupby(["region", "period"])["hot_dry"].sum().unstack())

# 3. Soil-moisture anomaly as a second drought definition
annual["soil_z"] = annual.groupby("region")["soil_root"].transform(lambda s: (s - s.mean()) / s.std())
print(annual.groupby(["region", "period"])["soil_z"].mean().unstack().round(2))

annual["dry"] = annual["rain_z"] <= -0.5
print(annual.groupby(["region", "period"])["dry"].sum().unstack())


annual["dry"] = annual["rain_z"] <= -0.5
print(annual.groupby(["region", "period"])["dry"].sum().unstack())


dry = annual.groupby(["region", "period"])["dry"].sum().unstack()
hd  = annual.groupby(["region", "period"])["hot_dry"].sum().unstack()
share = (hd / dry * 100).round(0)
print(share)   # % of dry years that were also hot

from scipy.stats import fisher_exact

table = [[28, 48 - 28],   # 2004-2025: hot-dry, not-hot dry
         [10, 60 - 10]]   # 1982-2003
odds, p = fisher_exact(table)
print(f"odds ratio = {odds:.1f}, p = {p:.5f}")