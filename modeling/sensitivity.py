"""
Sensitivity runs (step 5): refit the main specification with ONE change at a time and check whether
the conclusions hold.

A variant is a small dict from config.yaml, e.g. {"weights": "impressions"}. Supported keys:
  add_experiment_flag: true          add IN_EXPERIMENT_PERIOD to the features
  weights: impressions               weighted fits (WLS / sample-weighted elastic net)
  target_transform: log1p            model log(1 + rate); effects converted back to pts at the mean
  drop_features: [...]               remove features (e.g. the reach control)
  drop_top_influential: k            refit without the k posts with the largest Cook's D in the main OLS

Held-out scores are always unweighted MAE / R² on the original rate scale, so every variant is
scored the same way as the main model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import statsmodels.api as sm

from modeling.features import POOLED, FeatureSet, build_features
from modeling.models import cv_splits, fit_ols, group_kfold_splits, influence_table, make_regularized

KNOWN_KEYS = {"add_experiment_flag", "weights", "target_transform", "drop_features", "drop_top_influential"}


@dataclass
class VariantData:
    fs: FeatureSet
    weights: np.ndarray | None
    transform: str | None                 # None or "log1p"
    dropped_rows: list                    # master rows removed by this variant


def prepare_variant(prepared, platform: str, variant: dict, config: dict) -> VariantData:
    unknown = set(variant) - KNOWN_KEYS
    if unknown:
        raise ValueError(f"Unknown sensitivity keys: {sorted(unknown)}")
    fs = build_features(prepared, platform, config, sensitivity=variant.get("add_experiment_flag", False))

    dropped = []
    if variant.get("drop_top_influential"):
        main = build_features(prepared, platform, config)
        infl = influence_table(fit_ols(main), main)
        dropped = infl.nlargest(variant["drop_top_influential"], "cooks_d")["master_row"].tolist()
        keep = ~fs.X.index.isin(dropped)
        fs = replace(fs, X=fs.X[keep], y=fs.y[keep], groups=fs.groups[keep])

    if variant.get("drop_features"):
        fs = replace(fs, X=fs.X.drop(columns=variant["drop_features"]))

    weights = None
    if variant.get("weights") == "impressions":
        imp = prepared.set_index("MASTER_ROW").loc[fs.X.index, "IMPRESSIONS"].astype(float).to_numpy()
        weights = imp / imp.mean()                         # normalized; only relative size matters
    elif variant.get("weights"):
        raise ValueError(f"Unknown weights: {variant['weights']}")

    transform = variant.get("target_transform")
    if transform not in (None, "log1p"):
        raise ValueError(f"Unknown target_transform: {transform}")
    return VariantData(fs, weights, transform, dropped)


def _fwd(y, transform):
    return np.log1p(y) if transform == "log1p" else y


def _inv(y, transform):
    return np.expm1(y) if transform == "log1p" else y


def fit_ols_variant(vd: VariantData):
    """OLS / WLS with the same SE choice as the main model (HC3, or content-clustered when pooled)."""
    fs = vd.fs
    X = sm.add_constant(fs.X, has_constant="add")
    y = _fwd(fs.y, vd.transform)
    model = sm.WLS(y, X, weights=vd.weights) if vd.weights is not None else sm.OLS(y, X)
    if fs.platform == POOLED:
        return model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(fs.groups)[0]})
    return model.fit(cov_type="HC3")


def ols_effects(res, vd: VariantData) -> pd.DataFrame:
    """
    Coefficients in pts with 95% CI. For log1p models the coefficient b is on log(1 + rate); the effect
    on the rate at the platform's mean is approximately b * (1 + mean rate), applied to the CI too.
    """
    scale = (1 + vd.fs.y.mean()) if vd.transform == "log1p" else 1.0
    ci = res.conf_int()
    t = pd.DataFrame({"feature": res.params.index, "coef_pts": res.params.values * scale,
                      "ci_lo": ci[0].values * scale, "ci_hi": ci[1].values * scale, "p_value": res.pvalues.values})
    return t[t["feature"] != "const"]


def _fit_regularized_weighted(name, inner, config, X, y, w):
    pipe = make_regularized(name, inner, config)
    step = pipe.steps[-1][0]
    return pipe.fit(X, y, **({f"{step}__sample_weight": w} if w is not None else {}))


def regularized_variant(vd: VariantData, name: str, config: dict):
    """
    Nested-CV score (unweighted MAE / R² on the rate scale) and full-data coefficients in pts for one
    variant. Same outer splits construction and inner tuning as step 3.
    """
    fs = vd.fs
    X, y, groups = fs.X.to_numpy(), fs.y.to_numpy(), fs.groups.to_numpy()
    yt = _fwd(y, vd.transform)
    w = vd.weights
    splits = cv_splits(fs, config["cv"]["n_splits"], config["cv"]["n_repeats"], config["seed"])
    scores = []
    for tr, te in splits:
        inner = group_kfold_splits(groups[tr], config["regularization"]["inner_splits"], 1, config["seed"])
        pipe = _fit_regularized_weighted(name, inner, config, X[tr], yt[tr], None if w is None else w[tr])
        pred = _inv(pipe.predict(X[te]), vd.transform)
        resid = y[te] - pred
        ss_tot = ((y[te] - y[te].mean()) ** 2).sum()
        scores.append({"mae": np.abs(resid).mean(), "r2": 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan})
    pipe = _fit_regularized_weighted(name, splits, config, X, yt, w)
    scale = (1 + y.mean()) if vd.transform == "log1p" else 1.0
    coefs = pd.DataFrame({"feature": fs.X.columns, "coef_pts": pipe[-1].coef_ / pipe[0].scale_ * scale,
                          "nonzero": ~np.isclose(pipe[-1].coef_, 0)})
    return pd.DataFrame(scores), coefs


def robustness_summary(ols: pd.DataFrame, key_features: list[str]) -> pd.DataFrame:
    """
    For each platform × key feature: the main OLS effect, and across the other variants how many keep
    the same sign and how many keep the 95% CI clear of 0 with that sign.
    """
    rows = []
    for (platform, feature), g in ols[ols["feature"].isin(key_features)].groupby(["platform", "feature"], sort=False):
        main = g[g["variant"] == "main"]
        if main.empty:
            continue
        m = main.iloc[0]
        sign = np.sign(m["coef_pts"])
        others = g[g["variant"] != "main"]
        clear = ((others["ci_lo"] > 0) & (sign > 0)) | ((others["ci_hi"] < 0) & (sign < 0))
        rows.append({"platform": platform, "feature": feature, "main_coef_pts": m["coef_pts"],
                     "main_ci_lo": m["ci_lo"], "main_ci_hi": m["ci_hi"],
                     "main_ci_excludes_0": bool(m["ci_lo"] > 0 or m["ci_hi"] < 0),
                     "variants": len(others), "same_sign": int((np.sign(others["coef_pts"]) == sign).sum()),
                     "ci_excludes_0_same_sign": int(clear.sum()),
                     "variants_losing_it": ", ".join(others.loc[~clear, "variant"])})
    return pd.DataFrame(rows)
