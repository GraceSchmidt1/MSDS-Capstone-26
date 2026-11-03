"""
Model fitting and evaluation for the Stage 1 high- vs. low-engagement classifier.

Step C2: reference (base rate), plain logistic regression, and a shallow decision tree, all scored on the
same shared CV splits as the regression.
Step C3: penalized logistic regression (ridge, lasso, elastic net; penalty tuned by nested CV), bootstrap
stability of its effects, and held-out permutation importance (drop in AUC).

Conventions
  - Labels come from modeling/targets.py. In CV the label rule is refit on each training fold and applied
    to the held-out fold, so held-out posts never influence their own labels.
  - Scores: ROC AUC (primary: does the model rank high-engagement posts above low ones? 0.5 = chance),
    balanced accuracy at a 0.5 cut, Brier score (mean squared error of the probability; lower is better),
    log loss.
  - Logistic effects are reported two ways: odds ratios (multiplicative, with 95% CI) and average
    marginal effects in probability points (pts): "posts with this feature are X pts more likely to be
    high". Per platform SEs are robust (HC0); the pooled check clusters by content piece, as in OLS.
  - Everything that involves randomness takes the config seed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from modeling.features import POOLED, FeatureSet
from modeling.models import group_kfold_splits
from modeling.targets import make_label_rule


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation scoring (labels learned per training fold)
# ─────────────────────────────────────────────────────────────────────────────
def classification_scores(y_true, prob, threshold: float = 0.5) -> dict:
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
    both = len(np.unique(y_true)) == 2
    return {"auc": roc_auc_score(y_true, prob) if both else np.nan,
            "balanced_accuracy": balanced_accuracy_score(y_true, (prob > threshold).astype(int)) if both else np.nan,
            "brier": brier_score_loss(y_true, prob, pos_label=1),
            "log_loss": log_loss(y_true, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1])}


def cv_score_classifier(fs: FeatureSet, splits, label_name: str, config: dict, fit_predict_proba) -> pd.DataFrame:
    """
    Score a classifier on the shared splits. Per split: fit the label rule on the training posts, label
    both sides, then fit_predict_proba(X_tr, y_tr, X_te, groups_tr) -> P(high) for the test posts.
    `fit_warning` records whether the fit raised a convergence warning (e.g. perfect separation).
    """
    from sklearn.exceptions import ConvergenceWarning
    rows = []
    X, groups = fs.X.to_numpy(), fs.groups.to_numpy()
    for i, (tr, te) in enumerate(splits):
        rule = make_label_rule(label_name, config).fit(fs, tr)
        y_tr, y_te = rule.label(fs, tr), rule.label(fs, te)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            prob = fit_predict_proba(X[tr], y_tr, X[te], groups[tr])
        rows.append({"split": i, **classification_scores(y_te, prob), "test_share_high": y_te.mean(),
                     "fit_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught)})
    return pd.DataFrame(rows)


def base_rate_fit_predict(X_tr, y_tr, X_te, groups_tr=None):
    """Reference model: every post gets the training share of high posts (AUC 0.5 by construction)."""
    return np.full(len(X_te), y_tr.mean())


def logistic_fit_predict(config: dict):
    """Plain (unpenalized) logistic regression. Features are standardized only to help the optimizer."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def fit_predict(X_tr, y_tr, X_te, groups_tr=None):
        model = make_pipeline(StandardScaler(), LogisticRegression(        # tight tol: the exact MLE, as in
            penalty=None, tol=1e-8, max_iter=config["classification"]["logistic"]["max_iter"]))  # fit_logit
        return model.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    return fit_predict


def make_tree(config: dict):
    from sklearn.tree import DecisionTreeClassifier
    t = config["classification"]["tree"]
    return DecisionTreeClassifier(max_depth=t["max_depth"], min_samples_leaf=t["min_samples_leaf"],
                                  random_state=config["seed"])


def tree_fit_predict(config: dict):
    def fit_predict(X_tr, y_tr, X_te, groups_tr=None):
        return make_tree(config).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    return fit_predict


# ─────────────────────────────────────────────────────────────────────────────
# Logistic regression on all posts: odds ratios and probability-point effects
# ─────────────────────────────────────────────────────────────────────────────
def fit_logit(fs: FeatureSet, y):
    """Logistic regression with robust (HC0) SEs, or content-piece-clustered SEs for the pooled model."""
    model = sm.Logit(np.asarray(y), sm.add_constant(fs.X, has_constant="add"))
    if fs.platform == POOLED:
        return model.fit(disp=0, maxiter=200, cov_type="cluster",
                         cov_kwds={"groups": pd.factorize(fs.groups)[0]})
    return model.fit(disp=0, maxiter=200, cov_type="HC0")


def logit_effects(res, fs: FeatureSet, y) -> pd.DataFrame:
    """
    Per feature:
      odds_ratio (+ 95% CI)  multiplicative change in the odds of being high (per 0→1, or per doubling of reach)
      ame_pts (+ 95% CI)     average marginal effect: change in P(high), in percentage points. For 0/1
                             features it is the average change from switching the feature off → on.
      share_high_with        raw share of high posts among posts that have the feature (descriptive)
    """
    y = np.asarray(y)
    ci = res.conf_int()
    t = pd.DataFrame({"feature": res.params.index, "coef_log_odds": res.params.values, "se": res.bse.values,
                      "odds_ratio": np.exp(res.params.values), "or_ci_lo": np.exp(ci[0].values),
                      "or_ci_hi": np.exp(ci[1].values), "p_value": res.pvalues.values})
    me = res.get_margeff(at="overall", method="dydx", dummy=True).summary_frame()
    me = pd.DataFrame({"feature": me.index, "ame_pts": 100 * me["dy/dx"].values,
                       "ame_ci_lo": 100 * me["Conf. Int. Low"].values, "ame_ci_hi": 100 * me["Cont. Int. Hi."].values})
    t = t.merge(me, on="feature", how="left")
    binary = [c for c in fs.X.columns if set(np.unique(fs.X[c])) <= {0.0, 1.0}]
    t["posts_with_feature"] = t["feature"].map({c: int(fs.X[c].sum()) for c in binary})
    t["share_high_with"] = t["feature"].map({c: float(y[fs.X[c].to_numpy() == 1].mean()) for c in binary
                                             if fs.X[c].sum() > 0})
    t.insert(0, "platform", fs.platform)
    return t


def logit_fit_stats(res, fs: FeatureSet, y) -> dict:
    return {"platform": fs.platform, "n_posts": int(res.nobs), "n_features": fs.X.shape[1],
            "share_high": float(np.mean(y)), "pseudo_r2": res.prsquared, "aic": res.aic,
            "llr_pvalue": res.llr_pvalue, "converged": bool(res.mle_retvals["converged"]),
            "cov_type": res.cov_type}


# ─────────────────────────────────────────────────────────────────────────────
# Shallow decision tree on all posts: readable rules
# ─────────────────────────────────────────────────────────────────────────────
def _condition(feature: str, threshold: float, left: bool) -> str:
    """Readable split condition. 0/1 features split at 0.5; reach is shown in impressions, not log2."""
    if feature == "log2_impressions":
        return f"impressions {'≤' if left else '>'} {2 ** threshold:,.0f}"
    if np.isclose(threshold, 0.5):
        return f"{'not ' if left else ''}{feature}"
    return f"{feature} {'≤' if left else '>'} {threshold:.3g}"


def tree_leaves(tree, fs: FeatureSet, y) -> pd.DataFrame:
    """One row per leaf: the rule that reaches it, posts in it, and the share of them that are high."""
    t, cols, y = tree.tree_, list(fs.X.columns), np.asarray(y)
    leaf_of = tree.apply(fs.X.to_numpy())
    rows = []

    def walk(node, conds):
        if t.children_left[node] == -1:
            m = leaf_of == node
            rows.append({"platform": fs.platform, "leaf": int(node), "rule": " AND ".join(conds) or "(all posts)",
                         "posts": int(m.sum()), "high": int(y[m].sum()), "share_high": float(y[m].mean())})
            return
        f, thr = cols[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [_condition(f, thr, True)])
        walk(t.children_right[node], conds + [_condition(f, thr, False)])

    walk(0, [])
    return pd.DataFrame(rows)


def tree_importance(tree, fs: FeatureSet) -> pd.DataFrame:
    """Impurity-based importance (share of the tree's total split gain); 0 = never used."""
    return pd.DataFrame({"platform": fs.platform, "feature": fs.X.columns,
                         "importance": tree.feature_importances_})


# ─────────────────────────────────────────────────────────────────────────────
# Penalized logistic regression (step C3): ridge, lasso, elastic net
# ─────────────────────────────────────────────────────────────────────────────
PENALIZED = ["Ridge logistic", "Lasso logistic", "Elastic net logistic"]


def _penalty_kwargs(name: str, config: dict) -> dict:
    """lbfgs for ridge; saga for the L1 penalties (it, unlike liblinear, leaves the intercept unpenalized)."""
    reg = config["classification"]["regularization"]
    if name == "Ridge logistic":
        return {"penalty": "l2", "solver": "lbfgs"}
    if name == "Lasso logistic":
        return {"penalty": "l1", "solver": "saga", "tol": reg["saga_tol"]}
    if name == "Elastic net logistic":
        return {"penalty": "elasticnet", "solver": "saga", "tol": reg["saga_tol"]}
    raise ValueError(name)


def make_penalized(name: str, inner_cv, config: dict):
    """Standardize -> penalized logistic whose strength C (= 1 / penalty) is tuned on `inner_cv` by log loss."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    reg = config["classification"]["regularization"]
    g = reg["C_grid"]
    extra = {"l1_ratios": reg["l1_ratios"]} if name == "Elastic net logistic" else {}
    model = LogisticRegressionCV(Cs=np.logspace(g["min_exp"], g["max_exp"], g["num"]), cv=inner_cv,
                                 scoring="neg_log_loss", max_iter=reg["max_iter"], random_state=config["seed"],
                                 **_penalty_kwargs(name, config), **extra)
    return make_pipeline(StandardScaler(), model)


def _fit(pipe, X, y):
    """
    Fit, then repair saga's all-zero case: when every coefficient is zero saga stops after one pass with the
    intercept short of its optimum (P(high) off by up to ~2 pts on these data). The exact null fit predicts
    the training share of high posts, so the intercept is set to its log-odds. Nonzero fits are untouched
    (they match an exact solver). Inside LogisticRegressionCV's inner tuning the null fits keep saga's
    intercept, which inflates their log loss by ~1e-4, far below the differences that pick C.
    """
    pipe.fit(X, y)
    m = pipe[-1]
    if np.allclose(m.coef_, 0):
        share = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        m.intercept_[:] = np.log(share / (1 - share))
    return pipe


def _inner_splits(groups_tr, config: dict):
    return group_kfold_splits(groups_tr, config["classification"]["regularization"]["inner_splits"], 1,
                              config["seed"])


def penalized_fit_predict(name: str, config: dict, record: list | None = None):
    """
    Nested-CV scorer: the penalty is tuned on an inner group K-fold of the training posts only.
    `record`, if given, collects each split's fitted model (in split order) for permutation importance.
    """
    def fit_predict(X_tr, y_tr, X_te, groups_tr):
        pipe = _fit(make_penalized(name, _inner_splits(groups_tr, config), config), X_tr, y_tr)
        if record is not None:
            record.append(pipe)
        return pipe.predict_proba(X_te)[:, 1]
    return fit_predict


def fit_penalized(fs: FeatureSet, y, name: str, config: dict, splits):
    """Final model on all posts; penalty tuned on the shared outer splits."""
    return _fit(make_penalized(name, splits, config), fs.X.to_numpy(), np.asarray(y))


def _l1_ratio(model, name: str) -> float:
    return float(model.l1_ratio_[0]) if name == "Elastic net logistic" else (1.0 if name == "Lasso logistic" else 0.0)


def penalized_coef_table(pipe, fs: FeatureSet, name: str, config: dict) -> pd.DataFrame:
    """
    coef_per_sd:  change in log-odds of being high per one-SD change in the feature (comparable across
                  features: the importance measure)
    odds_ratio:   the same effect in the feature's own units (per 0→1, or per doubling of reach)
    C_at_grid_edge  the tuned C is the smallest or largest tried, so the grid may need widening
    """
    scaler, model = pipe[0], pipe[-1]
    g = config["classification"]["regularization"]["C_grid"]
    C = float(model.C_[0])
    t = pd.DataFrame({"platform": fs.platform, "model": name, "feature": fs.X.columns,
                      "coef_per_sd": model.coef_[0], "odds_ratio": np.exp(model.coef_[0] / scaler.scale_)})
    t["nonzero"] = ~np.isclose(t["coef_per_sd"], 0)
    t["C"] = C
    t["C_at_grid_edge"] = bool(np.isclose(np.log10(C), g["min_exp"]) or np.isclose(np.log10(C), g["max_exp"]))
    t["l1_ratio"] = _l1_ratio(model, name)
    return t


def _fixed_penalty_logistic(name: str, fitted_pipe, config: dict):
    """Same model with C (and the elastic-net mix) frozen at the values the full-data fit chose."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = fitted_pipe[-1]
    extra = {"l1_ratio": _l1_ratio(m, name)} if name == "Elastic net logistic" else {}
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=float(m.C_[0]), max_iter=config["classification"]["regularization"]["max_iter"],
        random_state=config["seed"], **_penalty_kwargs(name, config), **extra))


def bootstrap_stability_logistic(fs: FeatureSet, y, name: str, fitted_pipe, config: dict) -> pd.DataFrame:
    """
    Refit `name` on n_boot bootstrap samples (penalty fixed, labels as on all posts) and summarize per feature:
      selection_freq    share of refits where the feature is kept (nonzero); always 1 for ridge
      sign_agreement    share of refits (among kept) whose sign matches the full-data estimate, or the
                        refits' majority sign if the full-data fit dropped the feature
      or_lo/median/hi   2.5% / 50% / 97.5% of the odds ratio (dropped = 1)
    Resampling is over content pieces, so cross-posted copies move together in the pooled model.
    """
    rng = np.random.default_rng(config["seed"])
    X, y, groups = fs.X.to_numpy(), np.asarray(y), fs.groups.to_numpy()
    unique = np.unique(groups)
    rows_of = {g: np.where(groups == g)[0] for g in unique}
    full = fitted_pipe[-1].coef_[0] / fitted_pipe[0].scale_
    draws = np.zeros((config["classification"]["stability"]["n_boot"], X.shape[1]))
    for b in range(len(draws)):
        idx = np.concatenate([rows_of[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        pipe = _fixed_penalty_logistic(name, fitted_pipe, config).fit(X[idx], y[idx])
        # a 0/1 feature that is all-0 in the resample has scale 1 and coef 0: counted as dropped
        draws[b] = pipe[-1].coef_[0] / pipe[0].scale_
    kept = ~np.isclose(draws, 0)
    ref_sign = np.where(np.isclose(full, 0), np.sign(draws.sum(axis=0)), np.sign(full))
    same_sign = (np.sign(draws) == ref_sign) & kept
    lo, med, hi = np.exp(np.percentile(draws, [2.5, 50, 97.5], axis=0))
    return pd.DataFrame({
        "platform": fs.platform, "model": name, "feature": fs.X.columns, "full_data_odds_ratio": np.exp(full),
        "l1_ratio": _l1_ratio(fitted_pipe[-1], name), "selection_freq": kept.mean(axis=0),
        "sign_agreement": np.where(kept.sum(axis=0) > 0, same_sign.sum(axis=0) / np.maximum(kept.sum(axis=0), 1),
                                   np.nan),
        "or_lo": lo, "or_median": med, "or_hi": hi})


def permutation_importance_logistic(fs: FeatureSet, splits, label_name: str, name: str, config: dict,
                                    group_of, fitted: list | None = None) -> pd.DataFrame:
    """
    Held-out permutation importance on the shared outer splits. Per split the label rule is fit on the
    training posts, the model is tuned and fit on them (nested), then each feature, and each feature group
    jointly, is shuffled within the test fold n_repeats times. Importance = drop in test AUC.
    Returns one row per (split, level, unit) with the mean drop over repeats.
    `fitted`: the per-split models already fit while scoring CV (penalized_fit_predict's `record`), so they
    are not refit; the same fits, so the same result.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(config["seed"])
    X, groups = fs.X.to_numpy(), fs.groups.to_numpy()
    cols = list(fs.X.columns)
    units = [("feature", c, [i]) for i, c in enumerate(cols)]
    for g in dict.fromkeys(group_of(c) for c in cols):
        units.append(("group", g, [i for i, c in enumerate(cols) if group_of(c) == g]))
    rows = []
    for s, (tr, te) in enumerate(splits):
        rule = make_label_rule(label_name, config).fit(fs, tr)
        y_tr, y_te = rule.label(fs, tr), rule.label(fs, te)
        if len(np.unique(y_te)) < 2:
            continue                                                   # AUC undefined on a one-class fold
        pipe = fitted[s] if fitted is not None else \
            _fit(make_penalized(name, _inner_splits(groups[tr], config), config), X[tr], y_tr)
        base = roc_auc_score(y_te, pipe.predict_proba(X[te])[:, 1])
        for level, unit, idx in units:
            drops = []
            for _ in range(config["classification"]["permutation"]["n_repeats"]):
                Xp = X[te].copy()
                Xp[:, idx] = Xp[rng.permutation(len(te))][:, idx]      # shuffle the group's columns together
                drops.append(base - roc_auc_score(y_te, pipe.predict_proba(Xp)[:, 1]))
            rows.append({"platform": fs.platform, "split": s, "level": level, "unit": unit,
                         "auc_drop": float(np.mean(drops))})
    return pd.DataFrame(rows)
