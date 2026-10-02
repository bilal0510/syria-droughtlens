"""Step 5: early-warning drought risk model.

Question: at 31 December (three months into the hydrological year), can we
predict whether the year will end up dry (annual rainfall z <= -0.5)?

Design choices that keep the evaluation honest:
- Features use only data available by 31 Dec (no look-ahead).
- Walk-forward validation: to predict year Y, train only on years before Y.
- Per-region standardisation uses training years only.
- One pooled model for all regions (44 years per region is too few for one each).
- The label (dry year) is defined with full-period statistics, as elsewhere in
  the project; this is a mild simplification, noted in the limitations.
Note: early-season rainfall is part of the annual total, so some skill is
expected by construction; that is exactly what an early-warning system uses.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

from config import CUTOFF_MONTHS, MIN_TRAIN_YEARS, ALERT_THRESHOLD, RISK_BANDS

FEATURE_SETS = {
    "rain_model": ["rain_early", "prev_rain"],
    "rain_climate_model": ["rain_early", "prev_rain", "temp_early", "soil_early"],
}


def build_features(df, annual):
    """One row per region and hydrological year: early-season features + label.

    rain_early : rainfall in the cut-off months (Oct-Dec) of that hydrological year
    temp_early : mean temperature over the same months
    soil_early : mean root-zone soil wetness over the same months
    prev_rain  : total rainfall of the previous hydrological year
    dry        : did the year end up dry? (the target)
    """
    early = df[df["month"].isin(CUTOFF_MONTHS)]
    g = early.groupby(["region", "hydro_year"])
    feats = pd.DataFrame({"rain_early": g["PRECTOTCORR"].sum(), "temp_early": g["T2M"].mean()})
    if "GWETROOT" in early.columns:
        feats["soil_early"] = g["GWETROOT"].mean()
    feats = feats.reset_index()

    out = annual[["region", "hydro_year", "rain_mm", "dry", "hot_dry"]].merge(
        feats, on=["region", "hydro_year"])
    out = out.sort_values(["region", "hydro_year"])
    out["prev_rain"] = out.groupby("region")["rain_mm"].shift(1)
    return out.dropna(subset=["prev_rain"]).reset_index(drop=True)


def _scale(frame, reference, cols):
    """Standardise `cols` per region using mean/std of `reference` rows only."""
    out = frame.copy()
    mean = reference.groupby("region")[cols].mean()
    std = reference.groupby("region")[cols].std()
    for c in cols:
        out[c] = (frame[c] - frame["region"].map(mean[c])) / frame["region"].map(std[c])
    return out


def walk_forward(feats, feature_set):
    """Predict each year using only earlier years. Returns one row per region-year."""
    cols = [c for c in FEATURE_SETS[feature_set] if c in feats.columns]
    first_test = feats["hydro_year"].min() + MIN_TRAIN_YEARS
    rows = []
    for year in sorted(feats["hydro_year"].unique()):
        if year < first_test:
            continue
        train = feats[feats["hydro_year"] < year]
        test = feats[feats["hydro_year"] == year]
        train_s, test_s = _scale(train, train, cols), _scale(test, train, cols)
        model = LogisticRegression(max_iter=1000).fit(train_s[cols], train["dry"].astype(int))
        result = test[["region", "hydro_year", "dry", "hot_dry"]].copy()
        result["prob"] = model.predict_proba(test_s[cols])[:, 1]
        result["climatology"] = train["dry"].mean()          # baseline: base rate
        # Simple rule baseline score: lower early rainfall = higher risk
        result["rain_early_z"] = _scale(test, train, ["rain_early"])["rain_early"].values
        rows.append(result)
    return pd.concat(rows).reset_index(drop=True)


def evaluate(preds_by_model):
    """Compare models and baselines on the walk-forward predictions.

    hit_rate         : share of dry years that triggered an alert
    false_alarm_rate : share of non-dry years that triggered an alert
    brier_skill      : improvement over always predicting the base rate (0 = no skill)
    """
    rows = []
    first = next(iter(preds_by_model.values()))
    y = first["dry"].astype(int)
    brier_clim = brier_score_loss(y, first["climatology"])

    def alert_stats(alert):
        return {"hit_rate": alert[y == 1].mean(), "false_alarm_rate": alert[y == 0].mean()}

    rows.append({"model": "climatology (base rate)", "auc": 0.5, "brier_skill": 0.0,
                 "hit_rate": np.nan, "false_alarm_rate": np.nan})
    rule_alert = (first["rain_early_z"] <= -0.5).values
    rows.append({"model": "rule: early rain z <= -0.5", "auc": roc_auc_score(y, -first["rain_early_z"]),
                 "brier_skill": np.nan, **alert_stats(rule_alert)})
    for name, preds in preds_by_model.items():
        rows.append({"model": name, "auc": roc_auc_score(y, preds["prob"]),
                     "brier_skill": 1 - brier_score_loss(y, preds["prob"]) / brier_clim,
                     **alert_stats((preds["prob"] >= ALERT_THRESHOLD).values)})
    return pd.DataFrame(rows).set_index("model")


def fit_final(feats, feature_set="rain_climate_model"):
    """Train on all years. Returns a bundle used for scoring and interpretation."""
    cols = [c for c in FEATURE_SETS[feature_set] if c in feats.columns]
    model = LogisticRegression(max_iter=1000).fit(_scale(feats, feats, cols)[cols], feats["dry"].astype(int))
    return {"model": model, "cols": cols, "reference": feats}


def coefficients(bundle):
    """Standardised coefficients: positive = raises dry-year risk."""
    return pd.Series(bundle["model"].coef_[0], index=bundle["cols"]).sort_values()


def risk_band(prob):
    low, high = RISK_BANDS
    return "High" if prob >= high else "Moderate" if prob >= low else "Low"


def predict_risk(bundle, region, **values):
    """Dry-year probability for a region from its early-season values.

    Example: predict_risk(bundle, "Damascus", rain_early=40, prev_rain=250,
                          temp_early=17.5, soil_early=0.35)
    """
    row = pd.DataFrame([{"region": region, **{c: values[c] for c in bundle["cols"]}}])
    scaled = _scale(row, bundle["reference"], bundle["cols"])
    prob = float(bundle["model"].predict_proba(scaled[bundle["cols"]])[0, 1])
    return {"region": region, "probability": prob, "band": risk_band(prob)}
