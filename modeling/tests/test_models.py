"""Tests for modeling/models.py. Run from the repo root:  python -m pytest modeling/tests -q"""
import numpy as np
import pytest

from modeling.features import POOLED, build_features, load_config, load_master, prepare_posts
from modeling.models import (coef_table, cv_score, cv_splits, fit_ols, influence_table, ols_fit_predict,
                             reference_fit_predict, vif_table)

CONFIG = load_config()
CV = CONFIG["cv"]


@pytest.fixture(scope="module")
def prepared():
    posts, li_detail, _ = load_master(CONFIG)
    return prepare_posts(posts, li_detail, CONFIG)


@pytest.mark.parametrize("platform", CONFIG["platforms"] + [POOLED])
def test_cv_splits_cover_every_post_once_per_repeat(prepared, platform):
    fs = build_features(prepared, platform, CONFIG)
    splits = cv_splits(fs, CV["n_splits"], CV["n_repeats"], CONFIG["seed"])
    assert len(splits) == CV["n_splits"] * CV["n_repeats"]
    for r in range(CV["n_repeats"]):
        tests = np.concatenate([te for _, te in splits[r * CV["n_splits"]:(r + 1) * CV["n_splits"]]])
        assert sorted(tests) == list(range(len(fs.y)))
    for tr, te in splits:
        assert not set(tr) & set(te)


def test_pooled_splits_keep_content_pieces_together(prepared):
    fs = build_features(prepared, POOLED, CONFIG)
    groups = fs.groups.to_numpy()
    assert fs.groups.duplicated().any()               # cross-posted content exists in the pooled data
    for tr, te in cv_splits(fs, CV["n_splits"], CV["n_repeats"], CONFIG["seed"]):
        assert not set(groups[tr]) & set(groups[te])


def test_cv_splits_deterministic(prepared):
    fs = build_features(prepared, "LinkedIn", CONFIG)
    a = cv_splits(fs, 5, 2, CONFIG["seed"])
    b = cv_splits(fs, 5, 2, CONFIG["seed"])
    assert all((x[1] == y[1]).all() for x, y in zip(a, b))


@pytest.mark.parametrize("platform", CONFIG["platforms"])
def test_cv_ols_matches_statsmodels(prepared, platform):
    fs = build_features(prepared, platform, CONFIG)
    res = fit_ols(fs)
    pred = ols_fit_predict(fs.X.to_numpy(), fs.y.to_numpy(), fs.X.to_numpy())
    np.testing.assert_allclose(pred, res.fittedvalues.to_numpy(), rtol=1e-8, atol=1e-8)


def test_standard_errors_type(prepared):
    assert fit_ols(build_features(prepared, "Facebook", CONFIG)).cov_type == "HC3"
    assert fit_ols(build_features(prepared, POOLED, CONFIG)).cov_type == "cluster"


def test_reference_model_predicts_training_mean():
    pred = reference_fit_predict(np.zeros((3, 1)), np.array([1.0, 2.0, 6.0]), np.zeros((2, 1)))
    assert (pred == 3.0).all()


def test_tables_have_one_row_per_feature_and_post(prepared):
    fs = build_features(prepared, "LinkedIn", CONFIG)
    res = fit_ols(fs)
    assert len(coef_table(res, fs)) == fs.X.shape[1] + 1          # + intercept
    assert len(vif_table(fs)) == fs.X.shape[1]
    assert len(influence_table(res, fs)) == len(fs.y)
    scores = cv_score(fs, cv_splits(fs, 5, 1, 0), ols_fit_predict)
    assert len(scores) == 5 and scores["mae"].notna().all()


# ── regularized models (step 3) ──────────────────────────────────────────────
from modeling.models import (REGULARIZED, fit_regularized, regularized_coef_table,  # noqa: E402
                             regularized_fit_predict)


@pytest.mark.parametrize("name", REGULARIZED)
def test_regularized_deterministic_and_consistent(prepared, name):
    fs = build_features(prepared, "Instagram", CONFIG)
    splits = cv_splits(fs, 5, 1, CONFIG["seed"])
    a = regularized_coef_table(fit_regularized(fs, name, CONFIG, splits), fs, name)
    b = regularized_coef_table(fit_regularized(fs, name, CONFIG, splits), fs, name)
    np.testing.assert_array_equal(a["coef_per_sd"], b["coef_per_sd"])
    assert list(a["feature"]) == list(fs.X.columns)
    # per-SD and per-unit coefficients describe the same effect
    np.testing.assert_allclose(a["coef_pts"] * fs.X.std(ddof=0).to_numpy(), a["coef_per_sd"], rtol=1e-8, atol=1e-10)
    assert (a["nonzero"] == ~np.isclose(a["coef_per_sd"], 0)).all()


def test_nested_cv_tunes_on_training_posts_only(prepared, monkeypatch):
    """The inner splits handed to the tuner must index only the outer-training rows."""
    import modeling.models as m
    fs = build_features(prepared, POOLED, CONFIG)
    seen = []
    real = m.make_regularized

    def spy(name, inner_cv, config):
        seen.append(inner_cv)
        return real(name, inner_cv, config)

    monkeypatch.setattr(m, "make_regularized", spy)
    outer = cv_splits(fs, 5, 1, CONFIG["seed"])[:2]
    cv_score(fs, outer, regularized_fit_predict("Lasso", CONFIG))
    for (tr, _), inner in zip(outer, seen):
        assert max(max(i_tr.max(), i_te.max()) for i_tr, i_te in inner) < len(tr)
        groups_tr = fs.groups.to_numpy()[tr]
        for i_tr, i_te in inner:                       # content pieces stay together inside the tuner too
            assert not set(groups_tr[i_tr]) & set(groups_tr[i_te])
