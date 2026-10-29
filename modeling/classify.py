"""
Model fitting and evaluation for the Stage 1 high- vs. low-engagement classifier.

Step C2 (this file so far): reference (base rate), plain logistic regression, and a shallow decision
tree, all scored on the same shared CV splits as the regression.

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
