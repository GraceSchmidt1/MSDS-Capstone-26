"""Tests for modeling/targets.py. Run from the repo root:  python -m pytest modeling/tests -q"""
from dataclasses import replace

import numpy as np
import pytest

from modeling.features import POOLED, build_features, load_config, load_master, prepare_posts
from modeling.models import cv_splits
from modeling.targets import label_summary, make_label_rule, row_platforms

CONFIG = load_config()
LABELS = list(CONFIG["classification"]["labels"])


@pytest.fixture(scope="module")
def prepared():
    posts, li_detail, _ = load_master(CONFIG)
    return prepare_posts(posts, li_detail, CONFIG)


def test_main_label_is_configured():
    assert CONFIG["classification"]["main_label"] in LABELS


@pytest.mark.parametrize("name", LABELS)
@pytest.mark.parametrize("platform", CONFIG["platforms"])
def test_held_out_posts_never_influence_the_rule(prepared, name, platform):
    """Rewriting the test posts' rates to extremes must not change anything learned on the training posts."""
    fs = build_features(prepared, platform, CONFIG)
    tr, te = cv_splits(fs, 5, 1, CONFIG["seed"])[0]
    rule = make_label_rule(name, CONFIG).fit(fs, tr)
    y = fs.y.copy()
    y.iloc[te] = np.where(np.arange(len(te)) % 2, 500.0, -500.0)
    tampered = replace(fs, y=y)
    rule2 = make_label_rule(name, CONFIG).fit(tampered, tr)
    assert rule.describe().equals(rule2.describe())
    np.testing.assert_array_equal(rule.label(fs, tr), rule2.label(tampered, tr))


@pytest.mark.parametrize("platform", CONFIG["platforms"])
def test_class_balance(prepared, platform):
    fs = build_features(prepared, platform, CONFIG)
    share = {n: make_label_rule(n, CONFIG).fit(fs).label(fs).mean() for n in LABELS}
    assert 0.45 <= share["reach_adjusted"] <= 0.55
    assert 0.45 <= share["above_median"] <= 0.55
    assert 0.25 <= share["top_tercile"] <= 0.40


def test_quantile_rule_matches_definition(prepared):
    fs = build_features(prepared, "LinkedIn", CONFIG)
    rule = make_label_rule("above_median", CONFIG).fit(fs)
    expected = (fs.y.to_numpy() > np.median(fs.y)).astype(int)
    np.testing.assert_array_equal(rule.label(fs), expected)


@pytest.mark.parametrize("platform", CONFIG["platforms"])
def test_reach_adjusted_rule_matches_definition(prepared, platform):
    """The line is a best median fit (no worse than statsmodels' iterative QuantReg), and label = residual > median."""
    import statsmodels.api as sm
    fs = build_features(prepared, platform, CONFIG)
    rule = make_label_rule("reach_adjusted", CONFIG).fit(fs)
    prm = rule.params[platform]
    reach, y = fs.X["log2_impressions"].to_numpy(), fs.y.to_numpy()
    resid = y - (prm["intercept"] + prm["slope_per_doubling"] * reach)
    sm_params = sm.QuantReg(y, sm.add_constant(reach)).fit(q=0.5, max_iter=20000).params
    assert np.abs(resid).sum() <= np.abs(y - sm.add_constant(reach) @ sm_params).sum() + 1e-6
    assert prm["residual_threshold"] == pytest.approx(np.median(resid))
    np.testing.assert_array_equal(rule.label(fs), (resid - np.median(resid) > 1e-9).astype(int))


@pytest.mark.parametrize("platform", CONFIG["platforms"] + [POOLED])
def test_reach_adjusted_labels_do_not_hinge_on_float_noise(prepared, platform, monkeypatch):
    """Across all CV training folds, labels are identical whether ties are judged at 1e-9 or 1e-6."""
    from modeling.targets import ReachAdjustedRule
    fs = build_features(prepared, platform, CONFIG)
    splits = cv_splits(fs, CONFIG["cv"]["n_splits"], CONFIG["cv"]["n_repeats"], CONFIG["seed"])
    rules = [make_label_rule("reach_adjusted", CONFIG).fit(fs, tr) for tr, _ in splits]
    strict = [r.label(fs) for r in rules]
    monkeypatch.setattr(ReachAdjustedRule, "TIE_TOL", 1e-6)
    for r, lab in zip(rules, strict):
        np.testing.assert_array_equal(r.label(fs), lab)


@pytest.mark.parametrize("platform", CONFIG["platforms"])
def test_reach_adjusted_label_is_not_a_reach_proxy(prepared, platform):
    """The main label should be nearly uncorrelated with reach (unlike above-median on Instagram)."""
    fs = build_features(prepared, platform, CONFIG)
    lab = make_label_rule("reach_adjusted", CONFIG).fit(fs).label(fs)
    assert abs(np.corrcoef(lab, fs.X["log2_impressions"])[0, 1]) < 0.15


@pytest.mark.parametrize("name", LABELS)
def test_pooled_labels_equal_per_platform_labels(prepared, name):
    """In the pooled model each platform keeps its own rule, so labels match the per-platform ones."""
    pooled = build_features(prepared, POOLED, CONFIG)
    pooled_lab = make_label_rule(name, CONFIG).fit(pooled).label(pooled)
    plats = row_platforms(pooled, CONFIG)
    for p in CONFIG["platforms"]:
        fs = build_features(prepared, p, CONFIG)
        per = dict(zip(fs.X.index, make_label_rule(name, CONFIG).fit(fs).label(fs)))
        got = dict(zip(pooled.X.index[plats == p], pooled_lab[plats == p]))
        assert got == per


def test_row_platforms_decodes_pooled(prepared):
    pooled = build_features(prepared, POOLED, CONFIG)
    plats = row_platforms(pooled, CONFIG)
    truth = prepared.set_index("MASTER_ROW").loc[pooled.X.index, "PLATFORM"].to_numpy()
    assert (plats == truth).all()


def test_label_summary_counts(prepared):
    s = label_summary(build_features(prepared, POOLED, CONFIG), CONFIG)
    assert set(s["label"]) == set(LABELS)
    assert (s.groupby("label")["posts"].sum() == 390).all()


def test_errors(prepared):
    fs = build_features(prepared, "Facebook", CONFIG)
    with pytest.raises(RuntimeError):
        make_label_rule("above_median", CONFIG).label(fs)                     # not fitted
    with pytest.raises(KeyError):
        make_label_rule("not_a_label", CONFIG)
    bad = {**CONFIG, "classification": {**CONFIG["classification"], "labels": {"x": {"type": "magic"}}}}
    with pytest.raises(ValueError):
        make_label_rule("x", bad)
    no_reach = replace(fs, X=fs.X.drop(columns="log2_impressions"))
    with pytest.raises(ValueError):
        make_label_rule("reach_adjusted", CONFIG).fit(no_reach)
