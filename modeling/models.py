"""
Model fitting and evaluation for the Stage 1 engagement-rate regression.

Step 2 (this file so far): reference model, baseline OLS, multicollinearity (VIF), influence
diagnostics, and the shared cross-validation splits every later model is scored on.

Conventions
  - Coefficients are in percentage points of engagement rate (the target's own units).
  - Per-platform OLS uses HC3 robust standard errors (each post appears once, so clustering by post
    is the same as robust SEs). The pooled check clusters SEs by content piece, because the same
    content appears on up to three platforms.
  - Everything that involves randomness takes the config seed, so reruns give identical numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from modeling.features import POOLED, FeatureSet


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation splits (shared by every model so scores are comparable)
# ─────────────────────────────────────────────────────────────────────────────
def cv_splits(fs: FeatureSet, n_splits: int, n_repeats: int, seed: int):
    """
    Repeated K-fold as a list of (train_idx, test_idx) position arrays.
    Folds are built over content pieces, so copies of the same content (pooled model) never sit
    on both sides of a split. Per platform each piece appears once, so this is ordinary K-fold.
    """
    rng = np.random.default_rng(seed)
    groups = fs.groups.to_numpy()
    unique = np.unique(groups)
    splits = []
    for _ in range(n_repeats):
        fold_of_group = dict(zip(rng.permutation(unique), np.arange(len(unique)) % n_splits))
        fold = np.array([fold_of_group[g] for g in groups])
        for k in range(n_splits):
            splits.append((np.where(fold != k)[0], np.where(fold == k)[0]))
    return splits


def cv_score(fs: FeatureSet, splits, fit_predict) -> pd.DataFrame:
    """Score a model on the shared splits. fit_predict(X_tr, y_tr, X_te) -> predictions."""
    rows = []
    X, y = fs.X.to_numpy(), fs.y.to_numpy()
    for i, (tr, te) in enumerate(splits):
        pred = fit_predict(X[tr], y[tr], X[te])
        resid = y[te] - pred
        ss_tot = ((y[te] - y[te].mean()) ** 2).sum()
        rows.append({"split": i, "mae": np.abs(resid).mean(), "rmse": np.sqrt((resid ** 2).mean()),
                     "r2": 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan})
    return pd.DataFrame(rows)


def reference_fit_predict(X_tr, y_tr, X_te):
    """Reference model: always predict the training mean."""
    return np.full(len(X_te), y_tr.mean())


def ols_fit_predict(X_tr, y_tr, X_te):
    Xc_tr = np.column_stack([np.ones(len(X_tr)), X_tr])
    Xc_te = np.column_stack([np.ones(len(X_te)), X_te])
    beta, *_ = np.linalg.lstsq(Xc_tr, y_tr, rcond=None)
    return Xc_te @ beta


# ─────────────────────────────────────────────────────────────────────────────
# OLS
# ─────────────────────────────────────────────────────────────────────────────
def fit_ols(fs: FeatureSet):
    """Baseline OLS with robust (HC3) SEs, or content-piece-clustered SEs for the pooled model."""
    X = sm.add_constant(fs.X, has_constant="add")
    model = sm.OLS(fs.y, X)
    if fs.platform == POOLED:
        codes = pd.factorize(fs.groups)[0]
        return model.fit(cov_type="cluster", cov_kwds={"groups": codes})
    return model.fit(cov_type="HC3")


def coef_table(res, fs: FeatureSet) -> pd.DataFrame:
    """Effect sizes: coefficient (pts) with 95% CI, p-value, and how many posts have each binary feature."""
    ci = res.conf_int()
    t = pd.DataFrame({"feature": res.params.index, "coef_pts": res.params.values, "se": res.bse.values,
                      "ci_lo": ci[0].values, "ci_hi": ci[1].values, "p_value": res.pvalues.values})
    binary = {c: int(fs.X[c].sum()) for c in fs.X.columns if set(np.unique(fs.X[c])) <= {0.0, 1.0}}
    t["posts_with_feature"] = t["feature"].map(binary)
    t.insert(0, "platform", fs.platform)
    return t


def fit_stats(res, fs: FeatureSet) -> dict:
    return {"platform": fs.platform, "n_posts": int(res.nobs), "n_features": fs.X.shape[1],
            "r2": res.rsquared, "adj_r2": res.rsquared_adj, "aic": res.aic,
            "resid_sd": np.sqrt(res.scale), "cov_type": res.cov_type,
            "robust_f_pvalue": float(np.squeeze(res.f_pvalue))}


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def vif_table(fs: FeatureSet) -> pd.DataFrame:
    """Variance inflation factor per feature (with intercept). >5 is worth a look, >10 is a problem."""
    X = sm.add_constant(fs.X, has_constant="add").to_numpy()
    rows = [{"platform": fs.platform, "feature": c, "vif": variance_inflation_factor(X, i + 1)}
            for i, c in enumerate(fs.X.columns)]
    return pd.DataFrame(rows)


def influence_table(res, fs: FeatureSet) -> pd.DataFrame:
    """Per-post leverage, studentized residual and Cook's distance (flag: D > 4/n)."""
    infl = res.get_influence()
    n = len(fs.y)
    t = pd.DataFrame({"platform": fs.platform, "master_row": fs.X.index, "y": fs.y.values,
                      "fitted": res.fittedvalues.values, "resid": res.resid.values,
                      "student_resid": infl.resid_studentized_external, "leverage": infl.hat_matrix_diag,
                      "cooks_d": infl.cooks_distance[0]})
    t["influential"] = t["cooks_d"] > 4 / n
    return t
