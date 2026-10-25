"""
Feature building for the Stage 1 engagement-rate models.

One function turns posts into model features, and every later step reuses it:
  - build_features()      -> features for real posts in the master workbook
  - features_from_spec()  -> features for a hypothetical post (Stage 2 recommendations)
Both go through the same `_encode()`, so a candidate post is always encoded exactly like a real one.

Rows are platform posts (one row per post, never exploded). ICP segments and content pillars are
0/1 columns, so a post tagged with two segments has two 1s. Nothing is dropped silently: posts that
can't be modelled (no target / 0 impressions) are returned in `FeatureSet.excluded` with a reason.

All settings (paths, vocabularies, reference categories, formats) come from config.yaml.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
POOLED = "pooled"


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_config(path=DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_master(config: dict):
    """Read the posts sheet and LinkedIn detail sheet (read-only). Returns (posts, li_detail, sha256)."""
    path = REPO_ROOT / config["master"]
    posts = pd.read_excel(path, sheet_name=config["posts_sheet"])
    li_detail = pd.read_excel(path, sheet_name=config["linkedin_detail_sheet"])
    return posts, li_detail, file_sha256(path)


# ─────────────────────────────────────────────────────────────────────────────
# Derived columns (same definitions as the EDA notebook)
# ─────────────────────────────────────────────────────────────────────────────
def split_tags(value) -> list[str]:
    return [] if pd.isna(value) else [t.strip() for t in str(value).split(";") if t.strip()]


def prepare_posts(posts: pd.DataFrame, li_detail: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add the derived columns the encoder needs. Validates tags and formats; returns a new frame."""
    df = posts.copy()
    df.insert(0, "MASTER_ROW", df.index + 2)          # Excel row in the master sheet, for traceability
    df["DATE_TS"] = pd.to_datetime(df["DATE"])
    clock = pd.to_datetime(df["POST_TIME"].str.replace(r"\s+C[DS]T$", "", regex=True),
                           format="%I:%M %p", errors="coerce")
    if clock.isna().any():
        raise ValueError(f"Unparseable POST_TIME at master rows {df.loc[clock.isna(), 'MASTER_ROW'].tolist()}")
    df["HOUR"] = clock.dt.hour
    df["WEEKEND"] = df["DATE_TS"].dt.dayofweek.ge(5).astype(int)
    df["AFTER_11"] = df["HOUR"].ge(config["timing"]["after_hour"]).astype(int)

    boosted_dates = pd.to_datetime(
        li_detail.loc[li_detail["Post type"] == "Sponsored", "Created date"]).dt.normalize().unique()
    df["BOOSTED"] = (df["PLATFORM"].eq("LinkedIn") & df["DATE_TS"].isin(boosted_dates)).astype(int)
    df["IN_EXPERIMENT_PERIOD"] = df["IN_EXPERIMENT_PERIOD"].astype(bool).astype(int)

    # Cross-posted copies of the same content share a key (clustering / grouped CV in pooled models)
    df["CONTENT_KEY"] = (df["POST_TITLE"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
                         + " | " + df["DATE_TS"].dt.strftime("%Y-%m-%d"))

    _validate(df, config)
    return df


def _validate(df: pd.DataFrame, config: dict) -> None:
    for col, vocab in [("ICP_SEGMENTS", config["icp_segments"]), ("CONTENT_PILLAR", config["pillars"])]:
        unknown = {t for v in df[col] for t in split_tags(v)} - set(vocab)
        if unknown:
            raise ValueError(f"{col} has tags not in config: {sorted(unknown)}")
    for platform in config["platforms"]:
        seen = set(df.loc[df["PLATFORM"] == platform, "FORMAT"])
        unknown = seen - set(config["formats"][platform])
        if unknown:
            raise ValueError(f"{platform} has formats not listed in config: {sorted(unknown)}")


# ─────────────────────────────────────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────────────────────────────────────
def _formats_for(platform: str, config: dict) -> list[str]:
    if platform == POOLED:
        seen = []
        for p in config["platforms"]:
            seen += [f for f in config["formats"][p] if f not in seen]
        return seen
    return config["formats"][platform]


def feature_columns(platform: str, config: dict, sensitivity: bool = False) -> list[str]:
    """Ordered, human-readable feature names for a platform model (or 'pooled')."""
    feats = config["features"]
    cols = list(feats["reach"])
    cols += [f"ICP: {t}" for t in config["icp_segments"] if t != config["icp_reference"]]
    cols += [f"Pillar: {t}" for t in config["pillars"] if t != config["pillar_reference"]]
    cols += [f"Format: {f}" for f in _formats_for(platform, config) if f != config["formats"]["reference"]]
    cols += list(feats["timing"])
    if platform in ("LinkedIn", POOLED):
        cols += list(feats["linkedin_only"])
    if sensitivity:
        cols += list(feats["sensitivity_only"])
    if platform == POOLED:
        cols += [f"Platform: {p}" for p in config["platforms"][1:]]   # first platform is the reference
    return cols


def _encode(df: pd.DataFrame, platform: str, config: dict, sensitivity: bool = False) -> pd.DataFrame:
    """Turn prepared rows into the feature matrix. Shared by real posts and hypothetical specs."""
    X = pd.DataFrame(index=df.index)
    X["log2_impressions"] = np.log2(df["IMPRESSIONS"].astype(float))
    icp_sets = df["ICP_SEGMENTS"].map(lambda v: set(split_tags(v)))
    pillar_sets = df["CONTENT_PILLAR"].map(lambda v: set(split_tags(v)))
    for t in config["icp_segments"]:
        X[f"ICP: {t}"] = icp_sets.map(lambda s: int(t in s))
    for t in config["pillars"]:
        X[f"Pillar: {t}"] = pillar_sets.map(lambda s: int(t in s))
    for f in _formats_for(platform, config):
        X[f"Format: {f}"] = df["FORMAT"].eq(f).astype(int)
    for col in ["WEEKEND", "AFTER_11", "BOOSTED", "IN_EXPERIMENT_PERIOD"]:
        X[col] = df[col].astype(int)
    for p in config["platforms"]:
        X[f"Platform: {p}"] = df["PLATFORM"].eq(p).astype(int)
    return X[feature_columns(platform, config, sensitivity)].astype(float)


@dataclass
class FeatureSet:
    platform: str
    X: pd.DataFrame                 # features, index = MASTER_ROW
    y: pd.Series                    # target (rate, percentage points)
    groups: pd.Series               # CONTENT_KEY, for clustering / grouped CV
    excluded: pd.DataFrame = field(default_factory=pd.DataFrame)   # rows not modelled + reason


def build_features(prepared: pd.DataFrame, platform: str, config: dict, sensitivity: bool = False) -> FeatureSet:
    """Feature set for one platform (main analysis) or POOLED (check)."""
    rows = prepared if platform == POOLED else prepared[prepared["PLATFORM"] == platform]
    target = config["target"]
    reason = pd.Series(pd.NA, index=rows.index, dtype="object")
    reason[rows[target].isna()] = "target is null"
    reason[rows["IMPRESSIONS"].fillna(0) <= 0] = "0 impressions"
    keep = reason.isna()

    used = rows[keep]
    X = _encode(used, platform, config, sensitivity).set_axis(used["MASTER_ROW"])
    y = used[target].astype(float).set_axis(used["MASTER_ROW"]).rename(target)
    groups = used["CONTENT_KEY"].set_axis(used["MASTER_ROW"])
    excluded = rows.loc[~keep, ["MASTER_ROW", "PLATFORM", "DATE", "POST_TITLE"]].assign(REASON=reason[~keep])
    return FeatureSet(platform, X, y, groups, excluded.reset_index(drop=True))


def features_from_spec(spec: dict, platform: str, config: dict, sensitivity: bool = False) -> pd.DataFrame:
    """
    One-row feature matrix for a hypothetical post, e.g. for Stage 2:
        {"icp": ["Oncology Operations Leader"], "pillars": ["Events & Partnerships"], "format": "Video",
         "impressions": 250, "weekend": False, "hour": 9, "boosted": False}
    For POOLED, also pass "platform". Unknown tags/formats raise, same as for real posts.
    """
    post_platform = spec.get("platform", platform)
    row = pd.DataFrame([{
        "PLATFORM": post_platform,
        "ICP_SEGMENTS": "; ".join(spec["icp"]),
        "CONTENT_PILLAR": "; ".join(spec["pillars"]),
        "FORMAT": spec["format"],
        "IMPRESSIONS": spec["impressions"],
        "WEEKEND": int(bool(spec.get("weekend", False))),
        "AFTER_11": int(spec.get("hour", 9) >= config["timing"]["after_hour"]),
        "BOOSTED": int(bool(spec.get("boosted", False))),
        "IN_EXPERIMENT_PERIOD": int(bool(spec.get("in_experiment_period", False))),
    }])
    _validate(row, {**config, "platforms": [post_platform]})
    return _encode(row, platform, config, sensitivity)
