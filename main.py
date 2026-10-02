"""Run the full Syria DroughtLens pipeline, start to finish.

Usage (from this folder):
    python main.py            # uses data/raw/power_raw.csv if it exists, else downloads
    python main.py --fetch    # force a fresh download from NASA POWER
"""
import sys

import matplotlib
matplotlib.use("Agg")  # save figures to files, no window needed

from config import DATA_RAW, DATA_CLEAN, DATA_ANNUAL, TABLES_DIR
from fetch import fetch_all
from prepare import load_raw, check_raw, clean
from analysis import (build_annual, add_flags, trend_table, period_comparison, correlations,
                      regional_ranking, hot_dry_share, hot_dry_test)
from plots import plot_all, fig_risk_skill, fig_risk_heatmap
from risk_model import FEATURE_SETS, build_features, walk_forward, evaluate, fit_final, coefficients


def main():
    for path in (DATA_RAW, DATA_CLEAN, DATA_ANNUAL):
        path.parent.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data
    if "--fetch" in sys.argv or not DATA_RAW.exists():
        fetch_all().to_csv(DATA_RAW)
    raw = load_raw(DATA_RAW)
    check_raw(raw)

    # 2. Clean
    df = clean(raw)
    df.to_csv(DATA_CLEAN)
    print(f"\nCleaned data: {df.shape[0]:,} rows, {df['hydro_year'].nunique()} hydrological years")

    # 3. Analysis
    annual = add_flags(build_annual(df))
    annual.to_csv(DATA_ANNUAL, index=False)
    share, dry_counts, hot_dry_counts = hot_dry_share(annual)

    tables = {
        "trends": trend_table(annual),
        "period_comparison": period_comparison(annual),
        "correlations": correlations(annual),
        "regional_ranking": regional_ranking(annual),
        "hot_dry_share": share,
    }
    for name, table in tables.items():
        table.to_csv(TABLES_DIR / f"{name}.csv")
        print(f"\n=== {name} ===")
        print(table.round(3).to_string())
    print("\n=== Fisher test (pooled) ===")
    print(hot_dry_test(annual))

    # 4. Early-warning risk model
    feats = build_features(df, annual)
    preds = {name: walk_forward(feats, name) for name in FEATURE_SETS}
    metrics = evaluate(preds)
    metrics.to_csv(TABLES_DIR / "risk_model_metrics.csv")
    preds["rain_climate_model"].to_csv(TABLES_DIR / "risk_predictions.csv", index=False)
    print("\n=== Early-warning model (walk-forward) ===")
    print(metrics.round(3).to_string())
    print("\nStandardised coefficients (final model):")
    print(coefficients(fit_final(feats)).round(3).to_string())

    # 5. Figures
    plot_all(annual, share, show=False)
    fig_risk_skill(metrics, show=False)
    fig_risk_heatmap(annual, preds["rain_climate_model"], show=False)
    print("\nDone. Tables in outputs/tables, figures in outputs/figures.")


if __name__ == "__main__":
    main()
