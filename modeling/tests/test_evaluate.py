"""Tests for modeling/evaluate.py. Run from the repo root:  python -m pytest modeling/tests -q"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from modeling.evaluate import (adjust, cohens_d_table, cv_mean_sd, driver_ranking, exploded, group_tests,
                               hedges_g, oneway_tests, outcomes, partial_r2_table, pooled_se_comparison, post_tags)
from modeling.features import POOLED, REPO_ROOT, build_features, load_config, load_master, prepare_posts
from modeling.targets import make_label_rule

CONFIG = load_config()
SOURCES = {k: REPO_ROOT / v for k, v in CONFIG["evaluation"]["sources"].items()}


@pytest.fixture(scope="module")
def prepared():
    posts, li_detail, _ = load_master(CONFIG)
    return prepare_posts(posts, li_detail, CONFIG)


# ── outcomes ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("platform", CONFIG["platforms"] + [POOLED])
def test_rate_vs_reach_is_the_label_residual(prepared, platform):
    """rate_vs_reach minus the rule's cut reproduces the main label exactly, and its median is the cut."""
    fs = build_features(prepared, platform, CONFIG)
    rule = make_label_rule(CONFIG["classification"]["main_label"], CONFIG).fit(fs)
    resid = outcomes(fs, CONFIG)["rate_vs_reach"]
    if platform != POOLED:
        cut = rule.params[platform]["residual_threshold"]
        assert np.median(resid) == pytest.approx(cut, abs=1e-9)
        np.testing.assert_array_equal(((resid - cut) > 1e-9).astype(int), rule.label(fs))


# ── item 4 ──────────────────────────────────────────────────────────────────
def test_oneway_tests_match_statsmodels_anova():
    rng = np.random.default_rng(0)
    tag = pd.Series(np.repeat(["a", "b", "c"], [12, 20, 15]))
    y = pd.Series(rng.normal(0, 1, len(tag)) + tag.map({"a": 0, "b": 0.8, "c": -0.3}).to_numpy())
    r = oneway_tests(y, tag)
    table = sm.stats.anova_lm(smf.ols("y ~ C(tag)", pd.DataFrame({"y": y, "tag": tag})).fit())
    assert r["anova_f"] == pytest.approx(table.loc["C(tag)", "F"])
    ss_b, ss_w, df_w = table.loc["C(tag)", "sum_sq"], table.loc["Residual", "sum_sq"], table.loc["Residual", "df"]
    assert r["eta2"] == pytest.approx(ss_b / (ss_b + ss_w))
    assert r["omega2"] == pytest.approx((ss_b - 2 * ss_w / df_w) / (ss_b + ss_w + ss_w / df_w))


@pytest.mark.parametrize("platform", CONFIG["platforms"])
def test_group_tests_shapes_and_rows(prepared, platform):
    """Both factors × both outcomes; Tukey has every pair; post × tag rows = total tag count of kept tags."""
    fs = build_features(prepared, platform, CONFIG)
    outs = outcomes(fs, CONFIG)
    omnibus, pairs, se = group_tests(prepared, fs, outs, 5)
    assert len(omnibus) == 4
    for row, tk in zip(omnibus, pairs):
        k = row["n_groups"]
        assert len(tk) == k * (k - 1) // 2
        col = {"Content pillar": "CONTENT_PILLAR", "ICP segment": "ICP_SEGMENTS"}[row["factor"]]
        tags = post_tags(prepared, fs, col)
        assert row["all_posts_posts"] == len(fs.y)          # every post has at least one kept tag here
        assert row["all_posts_rows"] == len(exploded(fs.y, tags, list(set(t for ts in tags for t in ts))))
    s = pd.DataFrame(se)
    np.testing.assert_allclose(s["ratio_clustered_to_naive"], s["se_clustered"] / s["se_naive"])


def test_clustering_by_post_changes_nothing_without_duplicates():
    """With one row per post, post-clustered SEs equal HC0 up to the cluster small-sample factor."""
    rng = np.random.default_rng(1)
    tags = pd.Series([[t] for t in rng.choice(["a", "b", "c"], 60)], index=range(60))
    y = pd.Series(rng.normal(size=60), index=range(60))
    from modeling.evaluate import tag_ols
    ex = exploded(y, tags, ["a", "b", "c"])
    fits = tag_ols(ex, "a")
    X = fits["naive"].model.exog
    hc0 = fits["naive"].get_robustcov_results("HC0").bse
    n, k = X.shape
    factor = np.sqrt(n / (n - 1) * (n - 1) / (n - k))
    np.testing.assert_allclose(fits["clustered"].bse, hc0 * factor, rtol=1e-8)


# ── item 5 ──────────────────────────────────────────────────────────────────
def test_pooled_clustered_se_matches_regression_model(prepared):
    """The pooled clustered SEs are those of the regression's own pooled OLS (fit_ols)."""
    from modeling.models import fit_ols
    fs = build_features(prepared, POOLED, CONFIG)
    t = pooled_se_comparison(fs).set_index("term")
    np.testing.assert_allclose(t["se_clustered"], fit_ols(fs).bse[fs.X.columns], rtol=1e-10)


# ── item 7 ──────────────────────────────────────────────────────────────────
def test_hedges_g_known_case():
    a, b = np.array([2.0, 4.0, 6.0]), np.array([1.0, 2.0, 3.0])
    r = hedges_g(a, b)
    sp = np.sqrt((2 * 4.0 + 2 * 1.0) / 4)                    # variances 4 and 1, n = 3 each
    assert r["cohens_d"] == pytest.approx(2.0 / sp)
    assert r["hedges_g"] == pytest.approx((1 - 3 / (4 * 6 - 9)) * 2.0 / sp)
    assert r["ci_lo"] < r["hedges_g"] < r["ci_hi"]


def test_formats_are_compared_with_the_reference_format(prepared):
    fs = build_features(prepared, "LinkedIn", CONFIG)
    d = cohens_d_table(fs, outcomes(fs, CONFIG), CONFIG).set_index(["feature", "outcome"])
    n_image = int((prepared.query("PLATFORM == 'LinkedIn' and IMPRESSIONS > 0")["FORMAT"] == "Image").sum())
    assert d.loc[("Format: Document", "rate"), "compared_with"] == CONFIG["formats"]["reference"]
    assert d.loc[("Format: Document", "rate"), "n_compared"] == n_image
    assert d.loc[("WEEKEND", "rate"), "n_with"] + d.loc[("WEEKEND", "rate"), "n_compared"] == len(fs.y)


def test_partial_r2_matches_t_statistic(prepared):
    """For one feature, partial R² = t² / (t² + residual df) with ordinary SEs."""
    fs = build_features(prepared, "Instagram", CONFIG)
    outs = outcomes(fs, CONFIG)
    t = partial_r2_table(fs, outs).query("level == 'feature' and outcome == 'rate'").set_index("unit")
    res = sm.OLS(fs.y.to_numpy(), sm.add_constant(fs.X, has_constant="add")).fit()
    expected = res.tvalues[fs.X.columns] ** 2 / (res.tvalues[fs.X.columns] ** 2 + res.df_resid)
    np.testing.assert_allclose(t.loc[fs.X.columns, "partial_r2"], expected, rtol=1e-8)


# ── items 6 and 3 ───────────────────────────────────────────────────────────
def test_cv_mean_sd_reads_the_source_runs():
    t = cv_mean_sd(SOURCES["regression"], SOURCES["classification"])
    s = pd.read_csv(SOURCES["classification"] / "cv_summary.csv").set_index(["platform", "model"])
    r = t.query("task == 'classification' and platform == 'LinkedIn' and model == 'Ridge logistic' "
                "and metric == 'auc'").iloc[0]
    assert r["mean"] == s.loc[("LinkedIn", "Ridge logistic"), "auc_mean"]
    assert r["report"] == f"{r['mean']:.2f} ± {r['sd']:.2f}"


def test_adjusted_p_values_are_valid():
    df = pd.DataFrame({"fam": ["a"] * 4 + ["b"] * 2, "p": [0.01, 0.04, 0.03, np.nan, 0.5, 0.02]})
    out = adjust(df, ["p"], ["fam"])
    assert np.isnan(out.loc[3, "p_bh"])
    ok = out["p"].notna()
    assert (out.loc[ok, "p_bh"] >= out.loc[ok, "p"] - 1e-12).all() and (out.loc[ok, "p_bh"] <= 1).all()
    assert out.loc[5, "p_bh"] == pytest.approx(0.04)                          # BH within family b: 0.02·2/1


def test_driver_ranking_is_ordered_and_complete(prepared):
    effects = []
    for p in CONFIG["platforms"] + [POOLED]:
        fs = build_features(prepared, p, CONFIG)
        effects.append(cohens_d_table(fs, outcomes(fs, CONFIG), CONFIG))
    st = CONFIG["evaluation"]["stable"]
    t = driver_ranking(SOURCES["regression"], SOURCES["classification"], pd.concat(effects),
                       st["selection_freq"], st["sign_agreement"])
    ols = pd.read_csv(SOURCES["regression"] / "coefficients_ols.csv").query("feature != 'const'")
    assert len(t) == len(ols)                                            # one row per modelled feature
    for _, g in t.groupby("platform"):
        assert list(g["rank"]) == list(range(1, len(g) + 1))
        assert g["best_p_bh"].is_monotonic_increasing
        assert (g["p_reg_holm"] >= g["p_reg_bh"] - 1e-12).all()              # Holm is the stricter
