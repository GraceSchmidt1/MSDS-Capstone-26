"""Tests for modeling/classify.py. Run from the repo root:  python -m pytest modeling/tests -q"""
import warnings
from dataclasses import replace

import numpy as np
import pytest

from modeling.classify import (base_rate_fit_predict, classification_scores, cv_score_classifier, fit_logit,
                               logistic_fit_predict, logit_effects, make_tree, tree_fit_predict, tree_leaves)
from modeling.features import POOLED, build_features, load_config, load_master, prepare_posts
from modeling.models import cv_splits
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


@pytest.mark.parametrize("model", ["logistic", "tree"])
def test_held_out_rates_never_change_predictions(prepared, model):
    """Rewriting test-fold rates changes the test labels (so the scores) but never the predicted probabilities."""
    fs = build_features(prepared, "Instagram", CONFIG)
    tr, te = cv_splits(fs, 5, 1, CONFIG["seed"])[0]
    fp = logistic_fit_predict(CONFIG) if model == "logistic" else tree_fit_predict(CONFIG)
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
