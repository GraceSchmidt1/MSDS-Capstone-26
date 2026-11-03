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
def group_kfold_splits(groups, n_splits: int, n_repeats: int, seed: int):
    """
    Repeated K-fold over groups, as a list of (train_idx, test_idx) position arrays. All rows of a
    group land in the same fold.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    splits = []
    for _ in range(n_repeats):
        fold_of_group = dict(zip(rng.permutation(unique), np.arange(len(unique)) % n_splits))
        fold = np.array([fold_of_group[g] for g in groups])
        for k in range(n_splits):
            splits.append((np.where(fold != k)[0], np.where(fold == k)[0]))
    return splits


def cv_splits(fs: FeatureSet, n_splits: int, n_repeats: int, seed: int):
    """
    The shared outer CV splits. Folds are built over content pieces, so copies of the same content
    (pooled model) never sit on both sides of a split. Per platform each piece appears once, so this
    is ordinary repeated K-fold.
    """
    return group_kfold_splits(fs.groups.to_numpy(), n_splits, n_repeats, seed)


def cv_score(fs: FeatureSet, splits, fit_predict) -> pd.DataFrame:
    """Score a model on the shared splits. fit_predict(X_tr, y_tr, X_te, groups_tr) -> predictions."""
    rows = []
    X, y, groups = fs.X.to_numpy(), fs.y.to_numpy(), fs.groups.to_numpy()
    for i, (tr, te) in enumerate(splits):
        pred = fit_predict(X[tr], y[tr], X[te], groups[tr])
        resid = y[te] - pred
        ss_tot = ((y[te] - y[te].mean()) ** 2).sum()
        rows.append({"split": i, "mae": np.abs(resid).mean(), "rmse": np.sqrt((resid ** 2).mean()),
                     "r2": 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan})
    return pd.DataFrame(rows)


def reference_fit_predict(X_tr, y_tr, X_te, groups_tr=None):
    """Reference model: always predict the training mean."""
    return np.full(len(X_te), y_tr.mean())


def ols_fit_predict(X_tr, y_tr, X_te, groups_tr=None):
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


# ─────────────────────────────────────────────────────────────────────────────
# Regularized models (step 3): ridge, lasso, elastic net
# ─────────────────────────────────────────────────────────────────────────────
REGULARIZED = ["Ridge", "Lasso", "Elastic net"]


def make_regularized(name: str, inner_cv, config: dict):
    """Standardize -> penalized linear model whose penalty is tuned on `inner_cv` splits."""
    from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    reg = config["regularization"]
    g = reg["ridge_alpha_grid"]
    if name == "Ridge":
        model = RidgeCV(alphas=np.logspace(g["min_exp"], g["max_exp"], g["num"]), cv=inner_cv)
    elif name == "Lasso":
        model = LassoCV(n_alphas=reg["n_alphas"], cv=inner_cv, max_iter=reg["max_iter"],
                        random_state=config["seed"])
    elif name == "Elastic net":
        model = ElasticNetCV(l1_ratio=reg["l1_ratios"], n_alphas=reg["n_alphas"], cv=inner_cv,
                             max_iter=reg["max_iter"], random_state=config["seed"])
    else:
        raise ValueError(name)
    return make_pipeline(StandardScaler(), model)


def regularized_fit_predict(name: str, config: dict):
    """Nested-CV scorer: the penalty is tuned on an inner group K-fold of the training posts only."""
    def fit_predict(X_tr, y_tr, X_te, groups_tr):
        inner = group_kfold_splits(groups_tr, config["regularization"]["inner_splits"], 1, config["seed"])
        return make_regularized(name, inner, config).fit(X_tr, y_tr).predict(X_te)
    return fit_predict


def fit_regularized(fs: FeatureSet, name: str, config: dict, splits):
    """Final model on all posts; penalty tuned on the shared outer splits."""
    return make_regularized(name, splits, config).fit(fs.X.to_numpy(), fs.y.to_numpy())


def regularized_coef_table(pipe, fs: FeatureSet, name: str) -> pd.DataFrame:
    """
    coef_per_sd: effect of a one-standard-deviation change in the feature (pts). This is the
                 importance measure, comparable across features.
    coef_pts:    the same effect in the feature's own units (pts per doubling / per 0→1),
                 comparable with OLS.
    """
    scaler, model = pipe[0], pipe[-1]
    t = pd.DataFrame({"platform": fs.platform, "model": name, "feature": fs.X.columns,
                      "coef_per_sd": model.coef_, "coef_pts": model.coef_ / scaler.scale_})
    t["nonzero"] = ~np.isclose(t["coef_per_sd"], 0)
    t["alpha"] = model.alpha_
    t["l1_ratio"] = getattr(model, "l1_ratio_", 0.0 if name == "Ridge" else 1.0)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Stability and importance (step 4)
# ─────────────────────────────────────────────────────────────────────────────
def _fixed_penalty_model(name: str, fitted_pipe, config: dict):
    """Same model type with the penalty frozen at the value the full-data fit chose."""
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = fitted_pipe[-1]
    reg = config["regularization"]
    if name == "Ridge":
        model = Ridge(alpha=m.alpha_)
    elif name == "Lasso":
        model = Lasso(alpha=m.alpha_, max_iter=reg["max_iter"])
    else:
        model = ElasticNet(alpha=m.alpha_, l1_ratio=m.l1_ratio_, max_iter=reg["max_iter"])
    return make_pipeline(StandardScaler(), model)


def bootstrap_stability(fs: FeatureSet, name: str, fitted_pipe, config: dict) -> pd.DataFrame:
    """
    Refit `name` on n_boot bootstrap samples (penalty fixed) and summarize per feature:
      selection_freq    share of refits where the feature is kept (nonzero); always 1 for ridge
      sign_agreement    share of refits (among kept) whose sign matches the full-data estimate, or the
                        refits' majority sign if the full-data fit dropped the feature
      boot_lo/median/hi 2.5% / 50% / 97.5% of the effect in pts (dropped = 0)
    Resampling is over content pieces, so cross-posted copies move together in the pooled model.
    A binary feature that happens to be all-0 in a resample cannot be estimated and counts as dropped.
    """
    rng = np.random.default_rng(config["seed"])
    X, y, groups = fs.X.to_numpy(), fs.y.to_numpy(), fs.groups.to_numpy()
    unique = np.unique(groups)
    rows_of = {g: np.where(groups == g)[0] for g in unique}
    full = fitted_pipe[-1].coef_ / fitted_pipe[0].scale_
    draws = np.zeros((config["stability"]["n_boot"], X.shape[1]))
    for b in range(len(draws)):
        idx = np.concatenate([rows_of[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        pipe = _fixed_penalty_model(name, fitted_pipe, config).fit(X[idx], y[idx])
        draws[b] = pipe[-1].coef_ / pipe[0].scale_
    kept = ~np.isclose(draws, 0)
    # reference sign: the full-data estimate, or (if the full fit dropped the feature) the refits' majority
    ref_sign = np.where(np.isclose(full, 0), np.sign(draws.sum(axis=0)), np.sign(full))
    same_sign = (np.sign(draws) == ref_sign) & kept
    lo, med, hi = np.percentile(draws, [2.5, 50, 97.5], axis=0)
    return pd.DataFrame({
        "platform": fs.platform, "model": name, "feature": fs.X.columns, "full_data_coef_pts": full,
        "l1_ratio": getattr(fitted_pipe[-1], "l1_ratio_", 0.0 if name == "Ridge" else 1.0),
        "selection_freq": kept.mean(axis=0),
        "sign_agreement": np.where(kept.sum(axis=0) > 0, same_sign.sum(axis=0) / np.maximum(kept.sum(axis=0), 1),
                                   np.nan),
        "boot_lo": lo, "boot_median": med, "boot_hi": hi})


def permutation_importance_cv(fs: FeatureSet, splits, name: str, config: dict, group_of) -> pd.DataFrame:
    """
    Held-out permutation importance on the shared outer splits. For each split the model is tuned and
    fit on the training posts (nested, as in step 3), then each feature, and each feature group
    jointly, is shuffled within the test fold n_repeats times. Importance = rise in test MAE (pts).
    Returns one row per (split, level, unit) with the mean rise over repeats.
    """
    rng = np.random.default_rng(config["seed"])
    X, y, groups = fs.X.to_numpy(), fs.y.to_numpy(), fs.groups.to_numpy()
    cols = list(fs.X.columns)
    units = [("feature", c, [i]) for i, c in enumerate(cols)]
    for g in dict.fromkeys(group_of(c) for c in cols):
        units.append(("group", g, [i for i, c in enumerate(cols) if group_of(c) == g]))
    rows = []
    for s, (tr, te) in enumerate(splits):
        inner = group_kfold_splits(groups[tr], config["regularization"]["inner_splits"], 1, config["seed"])
        pipe = make_regularized(name, inner, config).fit(X[tr], y[tr])
        base = np.abs(y[te] - pipe.predict(X[te])).mean()
        for level, unit, idx in units:
            rises = []
            for _ in range(config["permutation"]["n_repeats"]):
                Xp = X[te].copy()
                Xp[:, idx] = Xp[rng.permutation(len(te))][:, idx]      # shuffle the group's columns together
                rises.append(np.abs(y[te] - pipe.predict(Xp)).mean() - base)
            rows.append({"platform": fs.platform, "split": s, "level": level, "unit": unit,
                         "mae_rise": float(np.mean(rises))})
    return pd.DataFrame(rows)


def summarize_importance(perm: pd.DataFrame, metric: str = "mae_rise") -> pd.DataFrame:
    """Mean score loss (`metric`) across splits, its 2.5-97.5% spread, and the share of splits where it helped (>0)."""
    g = perm.groupby(["platform", "level", "unit"], sort=False)[metric]
    out = pd.DataFrame({f"{metric}_mean": g.mean(), f"{metric}_lo": g.quantile(0.025),
                        f"{metric}_hi": g.quantile(0.975), "share_splits_helpful": g.apply(lambda s: (s > 0).mean())})
    return out.reset_index()
