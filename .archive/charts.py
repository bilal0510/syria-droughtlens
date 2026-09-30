import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", font_scale=1.0)
annual = pd.read_csv("annual.csv")

# ---------- Rebuild derived columns ----------
z = lambda s: (s - s.mean()) / s.std()
annual["rain_z"] = annual.groupby("region")["rain_mm"].transform(z)
annual["temp_z"] = annual.groupby("region")["temp_mean"].transform(z)
annual["period"] = np.where(annual["hydro_year"] <= 2003, "1982-2003", "2004-2025")
annual["drought"] = annual["rain_z"] <= -1
annual["dry"]     = annual["rain_z"] <= -0.5
annual["hot_dry"] = annual["dry"] & (annual["temp_z"] >= 0.5)

# Regions ordered wettest -> driest, reused everywhere
order = annual.groupby("region")["rain_mm"].mean().sort_values(ascending=False).index.tolist()

# ---------- 1. Rainfall anomaly timeline ----------
fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey=True)
for ax, region in zip(axes.flat, order):
    d = annual[annual["region"] == region]
    colors = np.where(d["rain_z"] <= -1, "#c0392b", np.where(d["rain_z"] < 0, "#e6b0aa", "#5dade2"))
    ax.bar(d["hydro_year"], d["rain_z"], color=colors)
    ax.axhline(-1, ls="--", lw=1, color="gray")
    ax.set_title(region)
fig.suptitle("Rainfall anomaly by hydrological year (red = drought, z ≤ -1)", fontsize=14)
fig.supylabel("Standardised rainfall anomaly (z)")
plt.tight_layout(); plt.savefig("fig1_anomaly_timeline.png", dpi=200); plt.show()

# ---------- 2. Drought heatmap (region x year) ----------
pivot = annual.pivot(index="region", columns="hydro_year", values="rain_z").loc[order]
plt.figure(figsize=(16, 4.5))
sns.heatmap(pivot, cmap="RdBu", center=0, vmin=-2.5, vmax=2.5,
            cbar_kws={"label": "Rainfall anomaly (z)"})
plt.title("Syria DroughtLens: rainfall anomaly by region and year")
plt.xlabel("Hydrological year"); plt.ylabel("")
plt.tight_layout(); plt.savefig("fig2_drought_heatmap.png", dpi=200); plt.show()

# ---------- 3. Temperature trends ----------
fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey=True)
for ax, region in zip(axes.flat, order):
    d = annual[annual["region"] == region]
    res = stats.linregress(d["hydro_year"], d["temp_mean"])
    ax.scatter(d["hydro_year"], d["temp_mean"], s=18, color="#e67e22")
    ax.plot(d["hydro_year"], res.intercept + res.slope * d["hydro_year"], color="#7b241c")
    ax.set_title(f"{region}  (+{res.slope*10:.2f} °C/decade)")
fig.suptitle("Annual mean temperature and linear trend", fontsize=14)
fig.supylabel("Mean temperature (°C)")
plt.tight_layout(); plt.savefig("fig3_temp_trends.png", dpi=200); plt.show()

# ---------- 4. HEADLINE: share of dry years that were also hot ----------
dry = annual.groupby(["region", "period"])["dry"].sum().unstack()
hd  = annual.groupby(["region", "period"])["hot_dry"].sum().unstack()
share = (hd / dry * 100).loc[order]
share.loc["ALL REGIONS"] = hd.sum() / dry.sum() * 100

ax = share.plot(kind="bar", figsize=(12, 5.5), color=["#5dade2", "#c0392b"], width=0.8)
ax.set_ylabel("% of dry years that were also hot")
ax.set_xlabel("")
ax.set_title("Dry years now coincide with heat far more often")
ax.legend(title="Period")
plt.xticks(rotation=30, ha="right")
plt.tight_layout(); plt.savefig("fig4_hot_dry_share.png", dpi=200); plt.show()
print(share.round(0))

# ---------- 5. Regional vulnerability ----------
v = annual.groupby("region").agg(mean_rain=("rain_mm", "mean"),
                                 cv=("rain_mm", lambda s: s.std() / s.mean()),
                                 hot_dry=("hot_dry", "sum"))
plt.figure(figsize=(8, 6))
plt.scatter(v["mean_rain"], v["cv"], s=v["hot_dry"] * 60 + 40, color="#c0392b", alpha=0.6)
for name, row in v.iterrows():
    plt.annotate(name, (row["mean_rain"], row["cv"]), xytext=(6, 6), textcoords="offset points")
plt.xlabel("Mean annual rainfall (mm)")
plt.ylabel("Rainfall variability (CV)")
plt.title("Regional vulnerability\n(bubble size = number of hot-dry years)")
plt.tight_layout(); plt.savefig("fig5_vulnerability.png", dpi=200); plt.show()