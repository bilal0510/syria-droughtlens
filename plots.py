"""Step 4: figures. Each function saves a PNG and can optionally display it."""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from config import FIG_DIR, DROUGHT_Z

sns.set_theme(style="whitegrid")


def region_order(annual):
    """Regions from wettest to driest (used for consistent ordering)."""
    return annual.groupby("region")["rain_mm"].mean().sort_values(ascending=False).index.tolist()


def _grid(n, ncols=4, **kwargs):
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes[:n]


def _finish(fig, filename, show):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def fig_anomaly_timeline(annual, show=True):
    """Rainfall anomaly per year and region; drought years in red."""
    order = region_order(annual)
    fig, axes = _grid(len(order), figsize=(18, 7), sharey=True)
    for ax, region in zip(axes, order):
        d = annual[annual["region"] == region]
        colors = np.where(d["rain_z"] <= DROUGHT_Z, "#c0392b",
                          np.where(d["rain_z"] < 0, "#e6b0aa", "#5dade2"))
        ax.bar(d["hydro_year"], d["rain_z"], color=colors)
        ax.axhline(DROUGHT_Z, ls="--", lw=1, color="gray")
        ax.set_title(region)
    fig.suptitle(f"Rainfall anomaly by hydrological year (red = drought, z <= {DROUGHT_Z})", fontsize=14)
    fig.supylabel("Standardised rainfall anomaly (z)")
    fig.tight_layout()
    _finish(fig, "fig1_anomaly_timeline.png", show)


def fig_drought_heatmap(annual, show=True):
    """Region x year heatmap of rainfall anomaly."""
    order = region_order(annual)
    pivot = annual.pivot(index="region", columns="hydro_year", values="rain_z").loc[order]
    fig, ax = plt.subplots(figsize=(16, 4.5))
    sns.heatmap(pivot, cmap="RdBu", center=0, vmin=-2.5, vmax=2.5,
                cbar_kws={"label": "Rainfall anomaly (z)"}, ax=ax)
    ax.set_title("Syria DroughtLens: rainfall anomaly by region and year")
    ax.set_xlabel("Hydrological year")
    ax.set_ylabel("")
    fig.tight_layout()
    _finish(fig, "fig2_drought_heatmap.png", show)


def fig_temperature_trends(annual, show=True):
    """Annual mean temperature with a linear trend line per region."""
    order = region_order(annual)
    fig, axes = _grid(len(order), figsize=(18, 7), sharey=True)
    for ax, region in zip(axes, order):
        d = annual[annual["region"] == region]
        res = stats.linregress(d["hydro_year"], d["temp_mean"])
        ax.scatter(d["hydro_year"], d["temp_mean"], s=18, color="#e67e22")
        ax.plot(d["hydro_year"], res.intercept + res.slope * d["hydro_year"], color="#7b241c")
        ax.set_title(f"{region}  ({res.slope * 10:+.2f} C/decade)")
    fig.suptitle("Annual mean temperature and linear trend", fontsize=14)
    fig.supylabel("Mean temperature (C)")
    fig.tight_layout()
    _finish(fig, "fig3_temp_trends.png", show)


def fig_hot_dry_share(annual, share, show=True):
    """Headline chart: share of dry years that were also hot, by period."""
    order = region_order(annual) + ["ALL REGIONS"]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    share.loc[order].plot(kind="bar", color=["#5dade2", "#c0392b"], width=0.8, ax=ax)
    ax.set_ylabel("% of dry years that were also hot")
    ax.set_xlabel("")
    ax.set_title("Dry years now coincide with heat far more often")
    ax.grid(axis="x", visible=False)
    ax.legend(title="Period")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    _finish(fig, "fig4_hot_dry_share.png", show)


def fig_vulnerability(annual, show=True):
    """Mean rainfall vs variability; bubble size = number of hot-dry years."""
    g = annual.groupby("region")
    v = pd.DataFrame({"mean_rain": g["rain_mm"].mean(),
                      "cv": g["rain_mm"].std() / g["rain_mm"].mean(),
                      "hot_dry": g["hot_dry"].sum()})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(v["mean_rain"], v["cv"], s=v["hot_dry"] * 60 + 40, color="#c0392b", alpha=0.6)
    for name, row in v.iterrows():
        ax.annotate(name, (row["mean_rain"], row["cv"]), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Mean annual rainfall (mm)")
    ax.set_ylabel("Rainfall variability (CV)")
    ax.set_title("Regional vulnerability\n(bubble size = number of hot-dry years)")
    fig.tight_layout()
    _finish(fig, "fig5_vulnerability.png", show)


def plot_all(annual, share, show=False):
    fig_anomaly_timeline(annual, show)
    fig_drought_heatmap(annual, show)
    fig_temperature_trends(annual, show)
    fig_hot_dry_share(annual, share, show)
    fig_vulnerability(annual, show)
