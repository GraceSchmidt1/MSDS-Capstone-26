"""Tests for modeling/recommend.py. Run from the repo root:  python -m pytest modeling/tests -q"""
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from modeling.evaluate import cohens_d_table, outcomes
from modeling.features import build_features, load_config, load_master, prepare_posts
from modeling.recommend import (DOWNGRADE, VERDICTS, factor_tests, option_masks, option_table, performance,
                                posts_needed, recommendations, verdict)
from modeling.targets import make_label_rule

CONFIG = load_config()


@pytest.fixture(scope="module")
def prepared():
    posts, li_detail, _ = load_master(CONFIG)
    return prepare_posts(posts, li_detail, CONFIG)


@pytest.fixture(scope="module")
def recs(prepared):
    return {p: recommendations(prepared, build_features(prepared, p, CONFIG), CONFIG) for p in CONFIG["platforms"]}


# ── performance and option comparisons ──────────────────────────────────────
@pytest.mark.parametrize("platform", CONFIG["platforms"])
def test_lift_and_beat_are_the_stage1_outcome_and_label(prepared, platform):
    fs = build_features(prepared, platform, CONFIG)
    perf = performance(fs, CONFIG)
    np.testing.assert_allclose(perf["lift"], outcomes(fs, CONFIG)["rate_vs_reach"], atol=1e-12)
    np.testing.assert_array_equal(perf["beat"], make_label_rule(CONFIG["classification"]["main_label"], CONFIG)
                                  .fit(fs).label(fs))
    np.testing.assert_allclose(perf["rate"] - perf["expected_rate"], perf["lift"])


@pytest.mark.parametrize("platform", CONFIG["platforms"])
def test_options_cover_the_platform(prepared, platform):
    fs = build_features(prepared, platform, CONFIG)
    opts = option_masks(prepared, fs, CONFIG)
    fmt = [o for o in opts if o[0] == "Format"]
    assert [o[1] for o in fmt] == CONFIG["formats"][platform]
    assert sum(o[2].sum() for o in fmt) == len(fs.y)                     # every post has exactly one format
    baselines = [o for o in opts if o[3] is None]
    assert {o[1] for o in baselines} >= {CONFIG["formats"]["reference"], "Weekday", "Before 11am"}
    for factor, option, m, cmp, _ in opts:
        if cmp is not None:
            assert not (m & cmp).any()                                    # with and comparison never overlap


@pytest.mark.parametrize("platform", CONFIG["platforms"])
def test_differences_match_stage1_effect_sizes(prepared, platform):
    """diff vs. baseline and Hedges' g reproduce evaluate.cohens_d_table on rate vs. reach."""
    fs = build_features(prepared, platform, CONFIG)
    ot = option_table(prepared, fs, CONFIG).dropna(subset=["hedges_g"])
    d = cohens_d_table(fs, outcomes(fs, CONFIG), CONFIG).query("outcome == 'rate_vs_reach'").set_index("feature")
    prefix = {"Format": "Format: ", "Content pillar": "Pillar: ", "ICP segment": "ICP: "}
    checked = 0
    for _, r in ot[ot["factor"].isin(prefix)].iterrows():
        name = prefix[r["factor"]] + r["option"]
        if name in d.index:
            assert r["diff_vs_baseline"] == pytest.approx(d.loc[name, "diff_pts"])
            assert r["hedges_g"] == pytest.approx(d.loc[name, "hedges_g"])
            assert r["comparison_posts"] == d.loc[name, "n_compared"]
            checked += 1
    assert checked >= 8


def test_option_table_is_deterministic(prepared):
    fs = build_features(prepared, "Instagram", CONFIG)
    pd.testing.assert_frame_equal(option_table(prepared, fs, CONFIG), option_table(prepared, fs, CONFIG))


def test_bh_is_within_factor_and_valid(recs):
    for options, factors in recs.values():
        ok = options["p_value"].notna()
        assert (options.loc[ok, "p_bh"] >= options.loc[ok, "p_value"] - 1e-12).all()
        assert (options.loc[ok, "p_bh"] <= 1).all()
        for _, g in options[ok].groupby("factor"):                        # family = the factor's options
            np.testing.assert_allclose(g["p_bh"], multipletests(g["p_value"], method="fdr_bh")[1])
        assert set(factors["factor"]) == set(options["factor"])


# ── verdicts ────────────────────────────────────────────────────────────────
def _row(**kw):
    base = {"compared_with": "Image", "posts": 30, "comparison_posts": 60, "p_value": 0.5, "p_bh": 0.5,
            "diff_vs_baseline": 1.0, "diff_ci_lo": -1.0, "diff_ci_hi": 3.0}
    return pd.Series({**base, **kw})


@pytest.mark.parametrize("row, factor_p_bh, expected", [
    (_row(compared_with="–"), 0.001, "Baseline"),
    (_row(posts=9), 0.001, "Not enough posts yet"),
    (_row(comparison_posts=9), 0.001, "Not enough posts yet"),
    (_row(p_value=0.001, p_bh=0.01), 0.01, "Do more"),
    (_row(p_value=0.001, p_bh=0.01, diff_vs_baseline=-2.0), 0.01, "Do less"),
    (_row(p_value=0.001, p_bh=0.01), 0.20, "Promising, keep testing"),       # factor not significant
    (_row(p_value=0.03, p_bh=0.09), 0.01, "Promising, keep testing"),        # unadjusted p only
    (_row(p_value=0.08, p_bh=0.2, diff_ci_lo=0.2), 0.01, "Promising, keep testing"),   # factor ok, CI clear
    (_row(p_value=0.08, p_bh=0.2, diff_ci_lo=0.2), 0.20, "No clear difference"),
    (_row(p_value=0.03, p_bh=0.09, diff_vs_baseline=-1.0), 0.5, "Watch, tends to underperform"),
    (_row(), 0.001, "No clear difference"),
])
def test_verdict_rules(row, factor_p_bh, expected):
    assert verdict(row, factor_p_bh, CONFIG) == expected


def test_strong_verdicts_hold_at_every_knot_count(recs):
    """After the knot check, no Do more / Do less differs at any checked knot count; downgrades carry a note."""
    for options, _ in recs.values():
        assert set(options["verdict"]) <= set(VERDICTS)
        cols = [f"verdict_{k}_knots" for k in CONFIG["recommend"]["knot_check"]]
        strong = options[options["verdict"].isin(list(DOWNGRADE))]
        for c in cols:
            assert (strong[c] == strong["verdict"]).all()
        noted = options[options["note"] != ""]
        assert noted["verdict"].isin(list(DOWNGRADE.values())).all()


def test_main_findings(recs):
    """The Stage 1 robust findings come out as firm verdicts; the knot-sensitive ones are downgraded."""
    v = {p: o.set_index(["factor", "option"])["verdict"] for p, (o, _) in recs.items()}
    assert v["LinkedIn"][("Format", "Document")] == "Do more"
    assert v["LinkedIn"][("Format", "Link")] == "Do less"
    assert v["Instagram"][("Format", "Carousel")] == "Promising, keep testing"
    for p in CONFIG["platforms"]:                       # pillar and segment never get a firm verdict
        assert not v[p].loc[["Content pillar", "ICP segment"]].isin(list(DOWNGRADE)).any()


# ── posts needed ────────────────────────────────────────────────────────────
def test_posts_needed():
    assert posts_needed(0.05, 10, 100, CONFIG) == np.inf                  # negligible effect
    assert posts_needed(0.5, 10, 10, CONFIG) == 64 - 10                   # 64 per group for d = 0.5, 80% power
    assert posts_needed(2.0, 13, 77, CONFIG) == 0                         # already enough
    assert posts_needed(0.12, 10, 10, CONFIG) == np.inf                   # beyond max_posts_needed


def test_factor_tests_shape(prepared):
    fs = build_features(prepared, "LinkedIn", CONFIG)
    f = factor_tests(fs, CONFIG)
    assert list(f["factor"]) == ["ICP segment", "Content pillar", "Format", "Timing", "Paid boost"]
    assert (f["p_bh"] >= f["p_value"] - 1e-12).all()
    assert f.set_index("factor").loc["Format", "df"] == len(CONFIG["formats"]["LinkedIn"]) - 1
