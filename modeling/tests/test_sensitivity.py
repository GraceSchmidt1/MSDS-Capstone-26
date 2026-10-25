"""Tests for modeling/sensitivity.py. Run from the repo root:  python -m pytest modeling/tests -q"""
import numpy as np
import pandas as pd
import pytest

from modeling.features import build_features, load_config, load_master, prepare_posts
from modeling.models import coef_table, fit_ols, influence_table
from modeling.sensitivity import fit_ols_variant, ols_effects, prepare_variant, robustness_summary

CONFIG = load_config()


@pytest.fixture(scope="module")
def prepared():
    posts, li_detail, _ = load_master(CONFIG)
    return prepare_posts(posts, li_detail, CONFIG)


def test_main_variant_reproduces_main_ols(prepared):
    vd = prepare_variant(prepared, "LinkedIn", {}, CONFIG)
    main = coef_table(fit_ols(build_features(prepared, "LinkedIn", CONFIG)), vd.fs).set_index("feature")
    eff = ols_effects(fit_ols_variant(vd), vd).set_index("feature")
    np.testing.assert_allclose(eff["coef_pts"], main.loc[eff.index, "coef_pts"])
    np.testing.assert_allclose(eff["ci_lo"], main.loc[eff.index, "ci_lo"])


def test_each_variant_changes_only_its_own_thing(prepared):
    base = prepare_variant(prepared, "Instagram", {}, CONFIG)
    exp = prepare_variant(prepared, "Instagram", {"add_experiment_flag": True}, CONFIG)
    assert list(exp.fs.X.columns) == list(base.fs.X.columns) + ["IN_EXPERIMENT_PERIOD"]
    no_reach = prepare_variant(prepared, "Instagram", {"drop_features": ["log2_impressions"]}, CONFIG)
    assert "log2_impressions" not in no_reach.fs.X.columns and len(no_reach.fs.y) == len(base.fs.y)
    w = prepare_variant(prepared, "Instagram", {"weights": "impressions"}, CONFIG)
    assert w.weights.mean() == pytest.approx(1.0) and w.fs.X.equals(base.fs.X)
    lg = prepare_variant(prepared, "Instagram", {"target_transform": "log1p"}, CONFIG)
    assert lg.transform == "log1p" and lg.fs.y.equals(base.fs.y)          # transform applied at fit time


def test_drop_most_influential_removes_largest_cooks_d(prepared):
    fs = build_features(prepared, "LinkedIn", CONFIG)
    top = influence_table(fit_ols(fs), fs).nlargest(1, "cooks_d")["master_row"].iloc[0]
    vd = prepare_variant(prepared, "LinkedIn", {"drop_top_influential": 1}, CONFIG)
    assert vd.dropped_rows == [top]
    assert top not in vd.fs.X.index and len(vd.fs.y) == len(fs.y) - 1


def test_log1p_effects_are_converted_to_pts(prepared):
    vd = prepare_variant(prepared, "Instagram", {"target_transform": "log1p"}, CONFIG)
    res = fit_ols_variant(vd)
    eff = ols_effects(res, vd).set_index("feature")
    expected = res.params.drop("const") * (1 + vd.fs.y.mean())
    np.testing.assert_allclose(eff["coef_pts"], expected.loc[eff.index])


def test_unknown_variant_keys_rejected(prepared):
    with pytest.raises(ValueError):
        prepare_variant(prepared, "LinkedIn", {"drop_outliers": True}, CONFIG)
    with pytest.raises(ValueError):
        prepare_variant(prepared, "LinkedIn", {"weights": "reach"}, CONFIG)


def test_robustness_summary_counts():
    ols = pd.DataFrame([
        {"platform": "LinkedIn", "feature": "f", "variant": "main", "coef_pts": 5, "ci_lo": 1, "ci_hi": 9},
        {"platform": "LinkedIn", "feature": "f", "variant": "a", "coef_pts": 4, "ci_lo": 0.5, "ci_hi": 8},
        {"platform": "LinkedIn", "feature": "f", "variant": "b", "coef_pts": 3, "ci_lo": -1, "ci_hi": 7},
        {"platform": "LinkedIn", "feature": "f", "variant": "c", "coef_pts": -1, "ci_lo": -3, "ci_hi": 1},
    ])
    r = robustness_summary(ols, ["f"]).iloc[0]
    assert (r["variants"], r["same_sign"], r["ci_excludes_0_same_sign"]) == (3, 2, 1)
    assert r["variants_losing_it"] == "b, c"
