"""
Tests for modeling/features.py. Run from the repo root:  python -m pytest modeling/tests -q

They read the master named in config.yaml (read-only) and check that feature building:
keeps every post accounted for, keeps one row per post, encodes tags/formats faithfully,
is deterministic, and encodes a hypothetical post exactly like the matching real post.
"""
import numpy as np
import pandas as pd
import pytest

from modeling.features import (POOLED, REPO_ROOT, build_features, feature_columns, features_from_spec,
                               file_sha256, load_config, load_master, prepare_posts, split_tags)

CONFIG = load_config()
ALL_SETS = CONFIG["platforms"] + [POOLED]


@pytest.fixture(scope="module")
def master():
    posts, li_detail, sha = load_master(CONFIG)
    return posts, li_detail, sha


@pytest.fixture(scope="module")
def prepared(master):
    posts, li_detail, _ = master
    return prepare_posts(posts, li_detail, CONFIG)


# ── every post accounted for ─────────────────────────────────────────────────
@pytest.mark.parametrize("platform", ALL_SETS)
def test_no_post_silently_dropped(prepared, platform):
    fs = build_features(prepared, platform, CONFIG)
    expected = len(prepared) if platform == POOLED else (prepared["PLATFORM"] == platform).sum()
    assert len(fs.X) + len(fs.excluded) == expected
    assert fs.excluded["REASON"].notna().all()


def test_only_unmodellable_posts_excluded(prepared):
    fs = build_features(prepared, POOLED, CONFIG)
    assert len(fs.excluded) == 1                                 # the Facebook post with 0 impressions
    assert set(fs.excluded["REASON"]) <= {"target is null", "0 impressions"}


def test_prepare_does_not_change_source(master):
    posts, li_detail, _ = master
    before = posts.copy()
    prepare_posts(posts, li_detail, CONFIG)
    pd.testing.assert_frame_equal(posts, before)


# ── grain: one row per post ──────────────────────────────────────────────────
@pytest.mark.parametrize("platform", ALL_SETS)
def test_one_row_per_post(prepared, platform):
    fs = build_features(prepared, platform, CONFIG)
    assert fs.X.index.is_unique
    assert fs.X.index.equals(fs.y.index) and fs.X.index.equals(fs.groups.index)


# ── encoding is faithful ─────────────────────────────────────────────────────
@pytest.mark.parametrize("platform", ALL_SETS)
def test_matrix_is_complete_and_named(prepared, platform):
    fs = build_features(prepared, platform, CONFIG)
    assert not fs.X.isna().any().any()
    assert list(fs.X.columns) == feature_columns(platform, CONFIG)


def test_tag_columns_match_tags(prepared):
    fs = build_features(prepared, POOLED, CONFIG)
    rows = prepared.set_index("MASTER_ROW").loc[fs.X.index]
    for col, vocab, ref in [("ICP_SEGMENTS", CONFIG["icp_segments"], CONFIG["icp_reference"]),
                            ("CONTENT_PILLAR", CONFIG["pillars"], CONFIG["pillar_reference"])]:
        prefix = "ICP: " if col == "ICP_SEGMENTS" else "Pillar: "
        for tag in vocab:
            if tag == ref:
                continue
            expected = rows[col].map(lambda v: int(tag in split_tags(v)))
            assert (fs.X[prefix + tag] == expected).all(), tag


@pytest.mark.parametrize("platform", CONFIG["platforms"])
def test_format_dummies_match_format(prepared, platform):
    fs = build_features(prepared, platform, CONFIG)
    fmt = prepared.set_index("MASTER_ROW").loc[fs.X.index, "FORMAT"]
    fmt_cols = [c for c in fs.X.columns if c.startswith("Format: ")]
    assert (fs.X[fmt_cols].sum(axis=1) == (fmt != CONFIG["formats"]["reference"]).astype(int)).all()
    for c in fmt_cols:
        assert (fs.X[c] == (fmt == c.removeprefix("Format: ")).astype(int)).all()


def test_reach_and_target_values(prepared):
    fs = build_features(prepared, POOLED, CONFIG)
    rows = prepared.set_index("MASTER_ROW").loc[fs.X.index]
    np.testing.assert_allclose(fs.X["log2_impressions"], np.log2(rows["IMPRESSIONS"]))
    np.testing.assert_allclose(fs.y, rows[CONFIG["target"]])


def test_platform_specific_columns(prepared):
    assert "BOOSTED" not in build_features(prepared, "Facebook", CONFIG).X.columns
    li = build_features(prepared, "LinkedIn", CONFIG).X
    assert li["BOOSTED"].sum() == prepared.loc[prepared["PLATFORM"] == "LinkedIn", "BOOSTED"].sum() > 0
    assert "Format: Carousel" not in li.columns


def test_sensitivity_adds_only_experiment_flag(prepared):
    main = build_features(prepared, "LinkedIn", CONFIG).X
    sens = build_features(prepared, "LinkedIn", CONFIG, sensitivity=True).X
    assert list(sens.columns) == list(main.columns) + ["IN_EXPERIMENT_PERIOD"]


# ── reproducibility ──────────────────────────────────────────────────────────
def test_deterministic(master):
    posts, li_detail, _ = master
    a = build_features(prepare_posts(posts, li_detail, CONFIG), POOLED, CONFIG)
    b = build_features(prepare_posts(posts, li_detail, CONFIG), POOLED, CONFIG)
    pd.testing.assert_frame_equal(a.X, b.X)
    pd.testing.assert_series_equal(a.y, b.y)


def test_master_not_modified(master):
    _, _, sha_at_load = master
    assert file_sha256(REPO_ROOT / CONFIG["master"]) == sha_at_load


# ── hypothetical posts (Stage 2) encode like real ones ──────────────────────
@pytest.mark.parametrize("platform", CONFIG["platforms"])
def test_spec_round_trip(prepared, platform):
    fs = build_features(prepared, platform, CONFIG)
    rows = prepared.set_index("MASTER_ROW").loc[fs.X.index]
    for master_row in rows.index[:: max(1, len(rows) // 15)]:      # ~15 posts per platform
        r = rows.loc[master_row]
        spec = {"icp": split_tags(r["ICP_SEGMENTS"]), "pillars": split_tags(r["CONTENT_PILLAR"]),
                "format": r["FORMAT"], "impressions": r["IMPRESSIONS"], "weekend": bool(r["WEEKEND"]),
                "hour": int(r["HOUR"]), "boosted": bool(r["BOOSTED"])}
        from_spec = features_from_spec(spec, platform, CONFIG).iloc[0]
        pd.testing.assert_series_equal(from_spec, fs.X.loc[master_row], check_names=False)


def test_spec_rejects_unknown_values():
    base = {"icp": ["Oncology Operations Leader"], "pillars": ["Events & Partnerships"], "format": "Image",
            "impressions": 100}
    with pytest.raises(ValueError):
        features_from_spec({**base, "format": "Carousel"}, "LinkedIn", CONFIG)     # not a LinkedIn format
    with pytest.raises(ValueError):
        features_from_spec({**base, "icp": ["Oncology CFO"]}, "LinkedIn", CONFIG)  # not a canonical segment
