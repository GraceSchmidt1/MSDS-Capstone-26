"""Tests for modeling/classify.py. Run from the repo root:  python -m pytest modeling/tests -q"""
import warnings
from dataclasses import replace

import numpy as np
import pytest

from modeling.classify import (PENALIZED, base_rate_fit_predict, bootstrap_stability_logistic, classification_scores,
                               cv_score_classifier, fit_logit, fit_penalized, forest_fit_predict,
                               logistic_fit_predict, logit_effects, logit_group_tests,
                               make_tree, penalized_coef_table, penalized_fit_predict,
                               permutation_importance_logistic, tree_fit_predict, tree_leaves)
from modeling.features import POOLED, build_features, feature_group, load_config, load_master, prepare_posts
from modeling.models import cv_splits, summarize_importance
from modeling.targets import make_label_rule

CONFIG = load_config()
MAIN = CONFIG["classification"]["main_label"]


@pytest.fixture(scope="module")
def prepared():
    posts, li_detail, _ = load_master(CONFIG)
    return prepare_posts(posts, li_detail, CONFIG)


def _labelled(prepared, platform):
    fs = build_features(prepared, platform, CONFIG)
    return fs, make_label_rule(MAIN, CONFIG).fit(fs).label(fs)


def test_scores_on_known_cases():
    y = np.array([0, 0, 1, 1])
    perfect = classification_scores(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect["auc"] == 1 and perfect["balanced_accuracy"] == 1
    assert perfect["brier"] == pytest.approx(np.mean([0.01, 0.04, 0.04, 0.01]))
    flat = classification_scores(y, np.full(4, 0.5))
    assert flat["auc"] == 0.5 and flat["brier"] == 0.25
    assert flat["precision"] == 0 and flat["recall"] == 0 and flat["accuracy"] == 0.5   # 0.5 is not > 0.5
    mixed = classification_scores(np.array([0, 0, 1, 1, 1]), np.array([0.6, 0.2, 0.9, 0.7, 0.1]))
    assert mixed["accuracy"] == pytest.approx(3 / 5)
    assert mixed["precision"] == pytest.approx(2 / 3) and mixed["recall"] == pytest.approx(2 / 3)
    assert mixed["f1"] == pytest.approx(2 / 3)


def test_models_only_see_training_labels(prepared):
    """fit_predict receives exactly the labels of a rule fit on the training posts."""
    fs = build_features(prepared, "LinkedIn", CONFIG)
    splits = cv_splits(fs, 5, 1, CONFIG["seed"])
    seen = []

    def spy(X_tr, y_tr, X_te, groups_tr):
        seen.append(y_tr)
        return np.full(len(X_te), 0.5)

    cv_score_classifier(fs, splits, MAIN, CONFIG, spy)
    for (tr, _), y_tr in zip(splits, seen):
        np.testing.assert_array_equal(y_tr, make_label_rule(MAIN, CONFIG).fit(fs, tr).label(fs, tr))


@pytest.mark.parametrize("model", ["logistic", "tree", "Lasso logistic", "forest"])
def test_held_out_rates_never_change_predictions(prepared, model):
    """Rewriting test-fold rates changes the test labels (so the scores) but never the predicted probabilities."""
    fs = build_features(prepared, "Instagram", CONFIG)
    tr, te = cv_splits(fs, 5, 1, CONFIG["seed"])[0]
    fp = {"logistic": logistic_fit_predict(CONFIG), "tree": tree_fit_predict(CONFIG),
          "forest": forest_fit_predict(CONFIG)}.get(model) \
        or penalized_fit_predict(model, CONFIG)
    probs = []

    def record(X_tr, y_tr, X_te, groups_tr):
        probs.append(fp(X_tr, y_tr, X_te, groups_tr))
        return probs[-1]

    y = fs.y.copy()
    y.iloc[te] = np.where(np.arange(len(te)) % 2, 500.0, -500.0)
    for f in (fs, replace(fs, y=y)):
        cv_score_classifier(f, [(tr, te)], MAIN, CONFIG, record)
    np.testing.assert_array_equal(probs[0], probs[1])


@pytest.mark.parametrize("platform", CONFIG["platforms"] + [POOLED])
def test_base_rate_is_chance(prepared, platform):
    fs = build_features(prepared, platform, CONFIG)
    s = cv_score_classifier(fs, cv_splits(fs, 5, 2, CONFIG["seed"]), MAIN, CONFIG, base_rate_fit_predict)
    assert (s["auc"] == 0.5).all()


@pytest.mark.parametrize("platform", CONFIG["platforms"] + [POOLED])
def test_logistic_cv_model_matches_statsmodels(prepared, platform):
    """The sklearn model scored in CV is the same logistic regression as the statsmodels one reported."""
    fs, y = _labelled(prepared, platform)
    res = fit_logit(fs, y)
    assert res.mle_retvals["converged"]
    X = fs.X.to_numpy()
    np.testing.assert_allclose(logistic_fit_predict(CONFIG)(X, y, X), res.predict(), atol=1e-4)


def test_logit_effects_definitions(prepared):
    fs, y = _labelled(prepared, "LinkedIn")
    res = fit_logit(fs, y)
    t = logit_effects(res, fs, y).set_index("feature")
    np.testing.assert_allclose(t["odds_ratio"], np.exp(res.params))
    assert (t["or_ci_lo"] <= t["odds_ratio"]).all() and (t["odds_ratio"] <= t["or_ci_hi"]).all()
    f = "Format: Document"                          # AME of a 0/1 feature = average change from off to on
    on, off = fs.X.assign(**{f: 1.0}), fs.X.assign(**{f: 0.0})
    import statsmodels.api as sm
    manual = (res.predict(sm.add_constant(on, has_constant="add"))
              - res.predict(sm.add_constant(off, has_constant="add"))).mean()
    assert t.loc[f, "ame_pts"] == pytest.approx(100 * manual)
    assert t.loc[f, "share_high_with"] == pytest.approx(y[fs.X[f] == 1].mean())
    assert t.loc[f, "posts_with_feature"] == 13


@pytest.mark.parametrize("platform", CONFIG["platforms"] + [POOLED])
def test_reach_gets_no_weight_under_the_reach_adjusted_label(prepared, platform):
    """Label B already removes reach, so reach should show no detectable effect on P(high)."""
    fs, y = _labelled(prepared, platform)
    t = logit_effects(fit_logit(fs, y), fs, y).set_index("feature")
    assert t.loc["log2_impressions", "ame_ci_lo"] < 0 < t.loc["log2_impressions", "ame_ci_hi"]


@pytest.mark.parametrize("platform", CONFIG["platforms"])
def test_tree_leaves_partition_the_posts(prepared, platform):
    fs, y = _labelled(prepared, platform)
    tree = make_tree(CONFIG).fit(fs.X.to_numpy(), y)
    leaves = tree_leaves(tree, fs, y)
    assert leaves["posts"].sum() == len(y) and leaves["high"].sum() == y.sum()
    assert (leaves["posts"] >= CONFIG["classification"]["tree"]["min_samples_leaf"]).all()
    assert leaves["rule"].str.count(" AND ").max() <= CONFIG["classification"]["tree"]["max_depth"] - 1
    assert not leaves["rule"].str.contains("log2_impressions").any()    # reach is shown in impressions


def test_cv_is_deterministic(prepared):
    fs = build_features(prepared, "Facebook", CONFIG)
    splits = cv_splits(fs, 5, 2, CONFIG["seed"])
    a = cv_score_classifier(fs, splits, MAIN, CONFIG, tree_fit_predict(CONFIG))
    b = cv_score_classifier(fs, splits, MAIN, CONFIG, tree_fit_predict(CONFIG))
    assert a.equals(b)


def test_fit_warning_is_recorded(prepared):
    from sklearn.exceptions import ConvergenceWarning
    fs = build_features(prepared, "Facebook", CONFIG)

    def warns(X_tr, y_tr, X_te, groups_tr):
        warnings.warn("did not converge", ConvergenceWarning)
        return np.full(len(X_te), 0.5)

    s = cv_score_classifier(fs, cv_splits(fs, 5, 1, CONFIG["seed"]), MAIN, CONFIG, warns)
    assert s["fit_warning"].all()


# ── penalized logistic, stability and importance (step C3) ───────────────────
CLS = CONFIG["classification"]
SMALL = {**CONFIG, "classification": {**CLS, "stability": {**CLS["stability"], "n_boot": 30},
                                      "permutation": {**CLS["permutation"], "n_repeats": 3}}}


@pytest.mark.parametrize("name", PENALIZED)
def test_penalized_deterministic_and_consistent(prepared, name):
    fs, y = _labelled(prepared, "Instagram")
    splits = cv_splits(fs, 5, 1, CONFIG["seed"])
    a = penalized_coef_table(fit_penalized(fs, y, name, CONFIG, splits), fs, name, CONFIG)
    b = penalized_coef_table(fit_penalized(fs, y, name, CONFIG, splits), fs, name, CONFIG)
    np.testing.assert_array_equal(a["coef_per_sd"], b["coef_per_sd"])
    assert list(a["feature"]) == list(fs.X.columns)
    # per-SD log-odds and per-unit odds ratio describe the same effect
    np.testing.assert_allclose(np.log(a["odds_ratio"]) * fs.X.std(ddof=0).to_numpy(), a["coef_per_sd"],
                               rtol=1e-8, atol=1e-10)
    assert (a["nonzero"] == ~np.isclose(a["coef_per_sd"], 0)).all()
    if name == "Ridge logistic":
        assert a["nonzero"].all()


def test_strongest_lasso_penalty_is_the_base_rate(prepared):
    """With every feature dropped, the lasso predicts the training share of high posts for every post."""
    fs, y = _labelled(prepared, "LinkedIn")
    cfg = {**CONFIG, "classification": {**CLS, "regularization": {**CLS["regularization"],
                                                                   "C_grid": {"min_exp": -4, "max_exp": -4, "num": 1}}}}
    pipe = fit_penalized(fs, y, "Lasso logistic", cfg, cv_splits(fs, 5, 1, CONFIG["seed"]))
    assert np.allclose(pipe[-1].coef_, 0)
    np.testing.assert_allclose(pipe.predict_proba(fs.X.to_numpy())[:, 1], y.mean(), rtol=1e-12)


def test_nested_cv_tunes_on_training_posts_only(prepared, monkeypatch):
    """The inner splits handed to the tuner index only outer-training rows and keep content pieces together."""
    import modeling.classify as c
    fs = build_features(prepared, POOLED, CONFIG)
    seen = []
    real = c.make_penalized

    def spy(name, inner_cv, config):
        seen.append(inner_cv)
        return real(name, inner_cv, config)

    monkeypatch.setattr(c, "make_penalized", spy)
    outer = cv_splits(fs, 5, 1, CONFIG["seed"])[:2]
    cv_score_classifier(fs, outer, MAIN, CONFIG, penalized_fit_predict("Lasso logistic", CONFIG))
    for (tr, _), inner in zip(outer, seen):
        assert max(max(i_tr.max(), i_te.max()) for i_tr, i_te in inner) < len(tr)
        groups_tr = fs.groups.to_numpy()[tr]
        for i_tr, i_te in inner:
            assert not set(groups_tr[i_tr]) & set(groups_tr[i_te])


@pytest.mark.parametrize("name", ["Lasso logistic", "Ridge logistic"])
def test_bootstrap_stability_deterministic_and_anchored(prepared, name):
    fs, y = _labelled(prepared, "Instagram")
    pipe = fit_penalized(fs, y, name, CONFIG, cv_splits(fs, 5, 1, CONFIG["seed"]))
    a = bootstrap_stability_logistic(fs, y, name, pipe, SMALL)
    b = bootstrap_stability_logistic(fs, y, name, pipe, SMALL)
    assert a.drop(columns="sign_agreement").equals(b.drop(columns="sign_agreement"))
    np.testing.assert_allclose(a["full_data_odds_ratio"], penalized_coef_table(pipe, fs, name, CONFIG)["odds_ratio"])
    assert a["selection_freq"].between(0, 1).all()
    assert (a["or_lo"] <= a["or_median"]).all() and (a["or_median"] <= a["or_hi"]).all()


def test_permutation_importance_units_and_determinism(prepared):
    fs = build_features(prepared, "LinkedIn", CONFIG)
    splits = cv_splits(fs, 5, 1, CONFIG["seed"])[:2]
    a = permutation_importance_logistic(fs, splits, MAIN, "Lasso logistic", SMALL, feature_group)
    b = permutation_importance_logistic(fs, splits, MAIN, "Lasso logistic", SMALL, feature_group)
    assert a.equals(b)
    groups = {feature_group(col) for col in fs.X.columns}
    assert set(a.loc[a["level"] == "feature", "unit"]) == set(fs.X.columns)
    assert set(a.loc[a["level"] == "group", "unit"]) == groups
    assert len(a) == len(splits) * (fs.X.shape[1] + len(groups))
    assert summarize_importance(a, metric="auc_drop")["share_splits_helpful"].between(0, 1).all()


def test_permutation_reusing_cv_fits_matches_refitting(prepared):
    """Passing the models recorded during CV scoring gives exactly the importance of refitting them."""
    fs = build_features(prepared, "LinkedIn", CONFIG)
    splits = cv_splits(fs, 5, 1, CONFIG["seed"])[:2]
    fits = []
    cv_score_classifier(fs, splits, MAIN, CONFIG, penalized_fit_predict("Lasso logistic", CONFIG, fits))
    assert len(fits) == len(splits)
    a = permutation_importance_logistic(fs, splits, MAIN, "Lasso logistic", SMALL, feature_group)
    b = permutation_importance_logistic(fs, splits, MAIN, "Lasso logistic", SMALL, feature_group, fitted=fits)
    assert a.equals(b)


def test_permutation_of_unused_feature_changes_nothing(prepared):
    """A feature the lasso gives zero weight must have exactly zero importance."""
    import modeling.classify as c
    fs = build_features(prepared, "Instagram", CONFIG)
    splits = cv_splits(fs, 5, 1, CONFIG["seed"])[:2]
    perm = permutation_importance_logistic(fs, splits, MAIN, "Lasso logistic", SMALL, feature_group)
    checked = 0
    for s, (tr, _) in enumerate(splits):
        y_tr = make_label_rule(MAIN, CONFIG).fit(fs, tr).label(fs, tr)
        pipe = c._fit(c.make_penalized("Lasso logistic", c._inner_splits(fs.groups.to_numpy()[tr], CONFIG), CONFIG),
                      fs.X.to_numpy()[tr], y_tr)
        zero = fs.X.columns[np.isclose(pipe[-1].coef_[0], 0)]
        rows = perm[(perm["split"] == s) & (perm["level"] == "feature") & perm["unit"].isin(zero)]
        assert np.allclose(rows["auc_drop"], 0)
        checked += len(rows)
    assert checked > 0


# ── evaluation framework: Wald tests and the random forest benchmark ────────
@pytest.mark.parametrize("platform", CONFIG["platforms"] + [POOLED])
def test_wald_tests(prepared, platform):
    """Per feature: χ² = z², same p-value. Per group: a one-feature group equals that feature's test, and
    the joint test matches statsmodels' own Wald test."""
    fs, y = _labelled(prepared, platform)
    res = fit_logit(fs, y)
    e = logit_effects(res, fs, y).set_index("feature")
    from scipy import stats
    np.testing.assert_allclose(stats.chi2.sf(e["wald_chi2"], 1), e["p_value"], rtol=1e-6)
    g = logit_group_tests(res, fs, feature_group).set_index("group")
    assert g.loc["Reach (impressions)", "wald_chi2"] == pytest.approx(e.loc["log2_impressions", "wald_chi2"])
    fmt = [c for c in fs.X.columns if feature_group(c) == "Format"]
    restriction = np.eye(len(res.params))[[list(res.params.index).index(c) for c in fmt]]   # one row per format
    sm_test = res.wald_test(restriction, scalar=True)
    assert g.loc["Format", "wald_chi2"] == pytest.approx(float(sm_test.statistic))
    assert g.loc["Format", "df"] == len(fmt)


def test_forest_is_deterministic(prepared):
    fs = build_features(prepared, "Instagram", CONFIG)
    splits = cv_splits(fs, 5, 1, CONFIG["seed"])
    a = cv_score_classifier(fs, splits, MAIN, CONFIG, forest_fit_predict(CONFIG))
    b = cv_score_classifier(fs, splits, MAIN, CONFIG, forest_fit_predict(CONFIG))
    assert a.equals(b)


def test_forest_permutation_reusing_cv_fits_matches_refitting(prepared):
    fs = build_features(prepared, "LinkedIn", CONFIG)
    splits = cv_splits(fs, 5, 1, CONFIG["seed"])[:2]
    fits = []
    cv_score_classifier(fs, splits, MAIN, CONFIG, forest_fit_predict(CONFIG, fits))
    a = permutation_importance_logistic(fs, splits, MAIN, "Random forest", SMALL, feature_group)
    b = permutation_importance_logistic(fs, splits, MAIN, "Random forest", SMALL, feature_group, fitted=fits)
    assert a.equals(b)
