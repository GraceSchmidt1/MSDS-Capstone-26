#!/usr/bin/env python3
"""
build_master_data.py
=====================
One-time (v1) consolidation of Hidalga's ~35 raw export files into a single
deduplicated master workbook, per DATA_CLEANING_PLAN.md.

This does NOT replace clean_hidalga_data.py (which stays as the monthly
3-file cleaner). It extends its normalizers/parsers (imported below) to
handle the full historical file inventory, where the job is mostly
"pick the one authoritative source per data family" rather than merging
everything.

Source selection (see DATA_CLEANING_PLAN.md sections 1a-1e for the reasoning):
  - manual-loomly-exporting.xlsx   -> primary post content/labels/format for
                                       FB/IG/LI, and Loomly click data
  - 082026-Social-Media-Data.xlsx  -> follower counts sheet only
  - hidalga-technologies_content_1788892250539.xls
                                    -> LinkedIn native detail + engagement
                                       ground truth
  - hidalga-technologies_visitors_1788892288178.xls
                                    -> LinkedIn visitor metrics + demographics
  - hidalga-technologies_followers_1788892297560.xls
                                    -> LinkedIn daily new-followers
  - Hidalga Content Analytics 2025-10-01 - 2026-09-10.xlsx
                                    -> Instagram Stories (unique content type)
  - Hidalga Technologies, Inc competitors.xlsx
                                    -> competitor reference table

Everything else in data/ is a superseded/duplicate/subset export. This
script does not touch those files; run move_superseded.py afterward (or see
its output) to relocate them to data/data_prev/ without deleting anything.

Outputs (written to data/):
  Hidalga_Master_Data_v1.xlsx
  merge_decisions.csv
  dedup_discrepancies.csv

Requires: pandas, openpyxl, xlrd
"""
from __future__ import annotations
import datetime as dt
import difflib
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import openpyxl

warnings.filterwarnings("ignore")  # noisy but harmless openpyxl style warnings on old .xls-derived files

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from clean_hidalga_data import (  # noqa: E402
    clean_text, norm_format, is_datetime_cell, is_time_cell, parse_labels,
    read_offset_sheet, sheet_names, build_li_content, build_visitors,
    build_followers, build_loomly, write_workbook, ICP_SEGMENTS, PILLAR_MAP,
)

P = lambda name: os.path.join(HERE, name)

# ─────────────────────────────────────────────────────────────────────────────
# Winning source files (see DATA_CLEANING_PLAN.md §1a/§1c)
# ─────────────────────────────────────────────────────────────────────────────
SRC = {
    "manual_loomly": P("manual-loomly-exporting.xlsx"),
    "follower_counts": P("082026-Social-Media-Data.xlsx"),
    "li_content": P("hidalga-technologies_content_1788892250539.xls"),
    "li_visitors": P("hidalga-technologies_visitors_1788892288178.xls"),
    "li_followers": P("hidalga-technologies_followers_1788892297560.xls"),
    "content_analytics_wide": P("Hidalga Content Analytics 2025-10-01 - 2026-09-10.xlsx"),
    "competitors": P("Hidalga Technologies, Inc competitors.xlsx"),
}

# Superseded/subset files kept around only to run "before you discard it" diffs
SUPERSEDED = {
    "follower_counts_subsets": [
        P("062026-Social-Media-Data.xlsx"),
        P("072026-Social-Media-Data.xlsx"),
    ],
    "li_content_subsets": [
        P("hidalga-012026-082026.xls"),
        P("hidalga-technologies_content_062026.xls"),
        P("hidalga-technologies_content_1776889133559.xls"),
        P("hidalga-technologies_content_1785352077908.xls"),
        P("hidalga-technologies_content_1786998188809.xls"),
    ],
    "li_visitors_subsets": [
        P("hidalga-012026082026.xls"),
        P("hidalga-technologies_visitors_062026.xls"),
        P("hidalga-technologies_visitors_1776889282185 copy.xls"),
        P("hidalga-technologies_visitors_1776889282185.xls"),
        P("hidalga-technologies_visitors_1779310462161.xls"),
        P("hidalga-technologies_visitors_1785352089815.xls"),
        P("hidalga-technologies_visitors_1786998157043.xls"),
        P("052026-hidalga-technologies_visitors.xls"),
    ],
}

OUT_WORKBOOK = P("Hidalga_Master_Data_v1.xlsx")
OUT_MERGE_LOG = P("merge_decisions.csv")
OUT_DEDUP_LOG = P("dedup_discrepancies.csv")

merge_decisions: list[dict] = []
dedup_discrepancies: list[dict] = []


def log_merge(category, key, decision, detail=""):
    merge_decisions.append({"CATEGORY": category, "KEY": key, "DECISION": decision, "DETAIL": detail})


def log_dedup(family, key, field, old_value, new_value, old_file, new_file):
    dedup_discrepancies.append({
        "FAMILY": family, "KEY": key, "FIELD": field,
        "OLD_VALUE": old_value, "NEW_VALUE": new_value,
        "OLD_FILE": os.path.basename(old_file), "NEW_FILE": os.path.basename(new_file),
    })


def norm_header(s):
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())


# ─────────────────────────────────────────────────────────────────────────────
# 1. Social Posts (FB/IG/LI) from manual-loomly-exporting.xlsx
# ─────────────────────────────────────────────────────────────────────────────
PLATFORM_SHEET = {"facebook": "Facebook", "instagram": "Instagram", "linkedin": "LinkedIn"}

# canonical column -> set of normalized header aliases seen across exports
COLUMN_ALIASES = {
    "DATE": {"DATE"},
    "FORMAT": {"FORMAT"},
    "POST": {"POST"},
    "LABELS": {"LABELS"},
    "IMPRESSIONS": {"IMPRESSIONS"},
    "REACTIONS": {"REACTIONS"},
    "COMMENTS": {"COMMENTS"},
    "SHARES": {"SHARES"},
    "CLICKS": {"CLICKS"},
    "VIDEO_VIEWS": {"VIDEOVIEWS"},
    "LIKES": {"LIKES"},
    "REACH": {"REACH"},
    "ENGAGEMENT": {"ENGAGEMENT", "ENGAGEMENTS"},
    "ENGAGEMENT_RATE": {"ENGAGEMENTRATE"},
    "SAVES": {"SAVES"},
    "REELS_PLAYS": {"REELSPLAYS"},
}


def build_index(header_row):
    normed = [norm_header(h) for h in header_row]
    idx = {}
    for canon, aliases in COLUMN_ALIASES.items():
        found = None
        for i, h in enumerate(normed):
            if h in aliases:
                found = i
                break
        idx[canon] = found
    return idx


def parse_manual_loomly_platform(path, sheet, plat):
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb[sheet]
    header = [clean_text(c) for c in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
    idx = build_index(header)

    posts, cur = [], None
    for r in ws.iter_rows(min_row=2, values_only=True):
        c0 = r[0] if len(r) else None
        if is_datetime_cell(c0):
            if cur is not None:
                posts.append(cur)
            cur = {"row": list(r), "time": None, "title": None, "body": []}
        else:
            if cur is None:
                continue
            if is_time_cell(c0):
                cur["time"] = c0.strftime("%I:%M %p").lstrip("0")
            elif isinstance(c0, str) and re.search(r"\d", c0):
                cur["time"] = clean_text(c0)
            post_i = idx.get("POST")
            if post_i is not None and post_i < len(r) and r[post_i] is not None:
                txt = clean_text(r[post_i])
                if txt:
                    if cur["title"] is None:
                        cur["title"] = txt
                    else:
                        cur["body"].append(txt)
    if cur is not None:
        posts.append(cur)

    out = []
    for p in posts:
        row = p["row"]

        def get(col):
            i = idx.get(col)
            return row[i] if i is not None and i < len(row) else None

        date = get("DATE")
        labels_raw = get("LABELS")
        segs, pillar, link_tags = parse_labels(labels_raw)

        if plat == "Facebook":
            imp = get("IMPRESSIONS") or 0
            num = sum((get(k) or 0) for k in ["REACTIONS", "COMMENTS", "SHARES", "CLICKS"])
            er = round(num / imp * 100, 4) if imp else None
        else:
            raw_er = get("ENGAGEMENT_RATE")
            er = round(float(raw_er) * 100, 4) if raw_er not in (None, "") else None

        out.append({
            "SOURCE_FILE": os.path.basename(path) + f" ({sheet})",
            "PLATFORM": plat,
            "DATE": pd.to_datetime(date).date() if date is not None else None,
            "POST_TIME": p["time"],
            "FORMAT": norm_format(get("FORMAT")),
            "POST_TITLE": p["title"],
            "POST_BODY": "\n".join(p["body"]) or None,
            "LABELS_RAW": clean_text(labels_raw),
            "ICP_SEGMENTS": "; ".join(segs) if segs else None,
            "CONTENT_PILLAR": pillar,
            "LINK_TAGS": link_tags,
            "IMPRESSIONS": get("IMPRESSIONS"),
            "REACH": get("REACH"),
            "LIKES": get("LIKES"),
            "REACTIONS": get("REACTIONS"),
            "COMMENTS": get("COMMENTS"),
            "SHARES": get("SHARES"),
            "CLICKS": get("CLICKS"),
            "SAVES": get("SAVES"),
            "VIDEO_VIEWS": get("VIDEO_VIEWS"),
            "REELS_PLAYS": get("REELS_PLAYS"),
            "ENGAGEMENT_RAW": get("ENGAGEMENT"),
            "ENGAGEMENT_RATE_PCT": er,
            "LOOMLY_ENGAGEMENT_RATE_PCT_RAW": er if plat == "LinkedIn" else None,
            "ENGAGEMENT_SOURCE": "Loomly",
        })
    return out


def build_social_posts(manual_loomly_path):
    records = []
    for sheet, plat in PLATFORM_SHEET.items():
        records.extend(parse_manual_loomly_platform(manual_loomly_path, sheet, plat))
    df = pd.DataFrame(records)
    for c in ["IMPRESSIONS", "REACH", "LIKES", "REACTIONS", "COMMENTS", "SHARES",
              "CLICKS", "SAVES", "VIDEO_VIEWS", "REELS_PLAYS", "ENGAGEMENT_RAW"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    log_merge("social_posts_source", "all platforms", "manual-loomly-exporting.xlsx used as sole primary source",
               "Family A/B dropped per plan §1a: manual-loomly's date coverage is a superset of both.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. LinkedIn engagement ground-truth override (plan §1b)
# ─────────────────────────────────────────────────────────────────────────────
def title_sim(a, b):
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a[:120].lower(), b[:120].lower()).ratio()


def apply_linkedin_engagement_override(social_posts, li_posts):
    li = li_posts.copy()
    li["Created date"] = pd.to_datetime(li["Created date"], errors="coerce").dt.date
    by_date = {}
    for i, row in li.iterrows():
        by_date.setdefault(row["Created date"], []).append(i)

    li_mask = social_posts["PLATFORM"] == "LinkedIn"
    matched_li_idx = set()
    gaps = []

    for si, srow in social_posts[li_mask].iterrows():
        candidates = by_date.get(srow["DATE"], [])
        if not candidates:
            gaps.append({"TYPE": "social_post_no_li_detail_match", "DATE": srow["DATE"],
                         "POST_TITLE": srow["POST_TITLE"]})
            continue
        if len(candidates) == 1:
            best = candidates[0]
        else:
            best = max(candidates, key=lambda ci: title_sim(srow["POST_TITLE"] or "", li.loc[ci, "Post title"] or ""))
            if title_sim(srow["POST_TITLE"] or "", li.loc[best, "Post title"] or "") < 0.15:
                gaps.append({"TYPE": "ambiguous_same_day_li_posts", "DATE": srow["DATE"],
                             "POST_TITLE": srow["POST_TITLE"]})
                continue

        matched_li_idx.add(best)
        lrow = li.loc[best]
        social_posts.at[si, "IMPRESSIONS"] = pd.to_numeric(pd.Series([lrow.get("Impressions")]), errors="coerce").iloc[0]
        social_posts.at[si, "REACTIONS"] = pd.to_numeric(pd.Series([lrow.get("Likes")]), errors="coerce").iloc[0]
        social_posts.at[si, "COMMENTS"] = pd.to_numeric(pd.Series([lrow.get("Comments")]), errors="coerce").iloc[0]
        social_posts.at[si, "SHARES"] = pd.to_numeric(pd.Series([lrow.get("Reposts")]), errors="coerce").iloc[0]
        social_posts.at[si, "CLICKS"] = pd.to_numeric(pd.Series([lrow.get("Clicks")]), errors="coerce").iloc[0]
        er = pd.to_numeric(pd.Series([lrow.get("Engagement rate (%)")]), errors="coerce").iloc[0]
        social_posts.at[si, "ENGAGEMENT_RATE_PCT"] = round(er, 4) if pd.notna(er) else None
        social_posts.at[si, "ENGAGEMENT_SOURCE"] = "LinkedIn native (Family C)"
        log_merge("li_engagement_override", f"{srow['DATE']} :: {(srow['POST_TITLE'] or '')[:60]}",
                  "overwrote Loomly metrics with LinkedIn-native detail row",
                  f"loomly_ER_raw={srow['LOOMLY_ENGAGEMENT_RATE_PCT_RAW']} -> native_ER={social_posts.at[si, 'ENGAGEMENT_RATE_PCT']}")

    unmatched_li = [i for i in li.index if i not in matched_li_idx]
    for i in unmatched_li:
        gaps.append({"TYPE": "li_detail_row_no_social_post_match", "DATE": li.loc[i, "Created date"],
                     "POST_TITLE": li.loc[i, "Post title"][:80] if isinstance(li.loc[i, "Post title"], str) else None})

    return social_posts, pd.DataFrame(gaps)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Instagram Stories (Family F, unique content type)
# ─────────────────────────────────────────────────────────────────────────────
def build_instagram_stories(path):
    if "Instagram - Stories" not in sheet_names(path):
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name="Instagram - Stories")
    df.columns = [clean_text(c) for c in df.columns]
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df.insert(0, "SOURCE_FILE", os.path.basename(path) + " (Instagram - Stories)")
    return df.dropna(how="all", subset=[c for c in df.columns if c != "SOURCE_FILE"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Competitors reference table (Family I)
# ─────────────────────────────────────────────────────────────────────────────
def build_competitors(path):
    raw = pd.read_excel(path, sheet_name="COMPETITORS", header=None)
    snap_start = clean_text(raw.iloc[0, 0])
    snap_end = clean_text(raw.iloc[0, 1])
    df = raw.iloc[1:].copy()
    df.columns = [clean_text(c) for c in raw.iloc[1]]
    df = df.iloc[1:].reset_index(drop=True)
    df.insert(0, "SOURCE_FILE", os.path.basename(path))
    df.insert(1, "SNAPSHOT_START_DATE", snap_start)
    df.insert(2, "SNAPSHOT_END_DATE", snap_end)
    for c in df.columns[3:]:
        if c == "Page":
            df[c] = df[c].map(clean_text)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. LinkedIn new followers (Family E, new sheet)
# ─────────────────────────────────────────────────────────────────────────────
def build_new_followers(path):
    df = read_offset_sheet(path, "New followers", 0)
    df.insert(0, "SOURCE_FILE", os.path.basename(path) + " (New followers)")
    df = df.rename(columns={"Date": "DATE"})
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.date
    return df.dropna(subset=["DATE"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Cross-checks against superseded/subset files (plan §2 step 2)
# ─────────────────────────────────────────────────────────────────────────────
def diff_follower_counts():
    winner = pd.read_excel(SRC["follower_counts"], sheet_name="follower counts", header=0)
    winner["MONTH_N"] = winner["MONTH"].str.lower()
    for f in SUPERSEDED["follower_counts_subsets"]:
        if not os.path.exists(f):
            continue
        try:
            sheet = next(s for s in sheet_names(f) if norm_header(s) == "FOLLOWERCOUNTS")
        except StopIteration:
            log_dedup("follower_counts", "(whole file)", "sheet_missing", "follower counts", None, f, SRC["follower_counts"])
            continue
        old = pd.read_excel(f, sheet_name=sheet, header=0)
        old["MONTH_N"] = old["MONTH"].str.lower()
        merged = old.merge(winner, on=["CHANNEL", "MONTH_N"], suffixes=("_old", "_new"))
        for _, r in merged.iterrows():
            for metric in ["POSTS", "FOLLOWERS", "ENGAGEMENT", "IMPRESSIONS"]:
                ov, nv = r[f"{metric}_old"], r[f"{metric}_new"]
                if ov != nv:
                    log_dedup("follower_counts", f"{r['CHANNEL']} / {r['MONTH_N']}", metric, ov, nv, f, SRC["follower_counts"])


def diff_li_content_metrics():
    winner = pd.read_excel(SRC["li_content"], sheet_name="Metrics", header=1)
    winner["Date"] = pd.to_datetime(winner["Date"], errors="coerce").dt.date
    for f in SUPERSEDED["li_content_subsets"]:
        if not os.path.exists(f):
            continue
        old = pd.read_excel(f, sheet_name="Metrics", header=1)
        old["Date"] = pd.to_datetime(old["Date"], errors="coerce").dt.date
        merged = old.merge(winner, on="Date", suffixes=("_old", "_new"))
        for metric in ["Impressions (total)", "Clicks (total)", "Reactions (total)",
                       "Comments (total)", "Reposts (total)"]:
            oc, nc = f"{metric}_old", f"{metric}_new"
            if oc not in merged.columns or nc not in merged.columns:
                continue
            mism = merged[merged[oc] != merged[nc]]
            for _, r in mism.iterrows():
                log_dedup("li_content_metrics", str(r["Date"]), metric, r[oc], r[nc], f, SRC["li_content"])


def diff_li_visitor_metrics():
    winner = pd.read_excel(SRC["li_visitors"], sheet_name="Visitor metrics", header=0)
    winner["Date"] = pd.to_datetime(winner["Date"], errors="coerce").dt.date
    for f in SUPERSEDED["li_visitors_subsets"]:
        if not os.path.exists(f):
            continue
        try:
            old = pd.read_excel(f, sheet_name="Visitor metrics", header=0)
        except Exception as e:
            log_dedup("li_visitor_metrics", "(whole file)", "sheet_read_error", str(e), None, f, SRC["li_visitors"])
            continue
        old["Date"] = pd.to_datetime(old["Date"], errors="coerce").dt.date
        common_cols = [c for c in ["Total page views (total)", "Total unique visitors (total)"] if c in old.columns and c in winner.columns]
        merged = old.merge(winner, on="Date", suffixes=("_old", "_new"))
        for metric in common_cols:
            oc, nc = f"{metric}_old", f"{metric}_new"
            mism = merged[merged[oc] != merged[nc]]
            for _, r in mism.iterrows():
                log_dedup("li_visitor_metrics", str(r["Date"]), metric, r[oc], r[nc], f, SRC["li_visitors"])


# ─────────────────────────────────────────────────────────────────────────────
# 7. QA
# ─────────────────────────────────────────────────────────────────────────────
def qa_report(social_posts):
    lines = []
    lines.append("=== Row counts by platform ===")
    counts = social_posts["PLATFORM"].value_counts().to_dict()
    for k, v in counts.items():
        lines.append(f"  {k}: {v}")
    lines.append(f"  TOTAL: {len(social_posts)}")

    prior_path = P("Hidalga_Cleaned_Data_2026-08.xlsx")
    if os.path.exists(prior_path):
        try:
            prior = pd.read_excel(prior_path, sheet_name="Social Posts (unified)")
            window = social_posts[(social_posts["DATE"] >= dt.date(2026, 1, 1)) & (social_posts["DATE"] <= dt.date(2026, 8, 24))]
            lines.append("\n=== Regression check vs Hidalga_Cleaned_Data_2026-08.xlsx (Jan1-Aug24 2026 window) ===")
            lines.append(f"  prior total: {len(prior)}  |  new pipeline in same window: {len(window)}")
            for plat in ["Facebook", "Instagram", "LinkedIn"]:
                p_n = len(prior[prior["PLATFORM"] == plat]) if "PLATFORM" in prior.columns else None
                n_n = len(window[window["PLATFORM"] == plat])
                lines.append(f"  {plat}: prior={p_n}  new={n_n}")
        except Exception as e:
            lines.append(f"\n(regression check skipped: {e})")

    lines.append("\n=== Duplicate scan on (PLATFORM, DATE, POST_TITLE) ===")
    titled = social_posts[social_posts["POST_TITLE"].notna()]
    dupe_mask = titled.duplicated(subset=["PLATFORM", "DATE", "POST_TITLE"], keep=False)
    lines.append(f"  duplicate rows (titled posts only): {dupe_mask.sum()}")
    untitled = social_posts[social_posts["POST_TITLE"].isna()]
    untitled_dupe = untitled.duplicated(subset=["PLATFORM", "DATE", "POST_TIME"], keep=False)
    lines.append(f"  untitled posts: {len(untitled)}  (of which same PLATFORM/DATE/POST_TIME collide: {untitled_dupe.sum()} -- distinct posts with blank text, not true dupes)")

    lines.append("\n=== Month-by-month post count & mean engagement rate ===")
    sp = social_posts.copy()
    sp["MONTH"] = pd.to_datetime(sp["DATE"]).dt.to_period("M")
    monthly = sp.groupby(["MONTH", "PLATFORM"]).agg(
        posts=("POST_TITLE", "count"), mean_er=("ENGAGEMENT_RATE_PCT", "mean")).reset_index()
    lines.append(monthly.to_string(index=False))

    lines.append("\n=== Thin cells: ICP segment x platform with < 3 posts ===")
    exploded = sp.assign(ICP_SEGMENTS=sp["ICP_SEGMENTS"].fillna("")).copy()
    exploded["ICP_SEGMENTS"] = exploded["ICP_SEGMENTS"].str.split("; ")
    exploded = exploded.explode("ICP_SEGMENTS")
    exploded = exploded[exploded["ICP_SEGMENTS"] != ""]
    thin = exploded.groupby(["ICP_SEGMENTS", "PLATFORM"]).size().reset_index(name="n")
    thin = thin[thin["n"] < 3]
    lines.append(thin.to_string(index=False) if not thin.empty else "  (none)")

    lines.append("\n=== Label/pillar coverage (% tagged) ===")
    cov = sp.groupby("PLATFORM").apply(lambda g: pd.Series({
        "n": len(g),
        "pct_icp_tagged": round((g["ICP_SEGMENTS"].notna().sum() / len(g)) * 100, 1) if len(g) else 0,
        "pct_pillar_tagged": round((g["CONTENT_PILLAR"].notna().sum() / len(g)) * 100, 1) if len(g) else 0,
    }), include_groups=False)
    lines.append(cov.to_string())

    return "\n".join(lines), monthly, thin, cov.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Building Social Posts (unified) from manual-loomly-exporting.xlsx ...")
    social_posts = build_social_posts(SRC["manual_loomly"])

    print("Parsing LinkedIn native content export (Family C) ...")
    li_posts, li_metrics = build_li_content(SRC["li_content"])

    print("Applying LinkedIn engagement ground-truth override (plan §1b) ...")
    social_posts, li_join_gaps = apply_linkedin_engagement_override(social_posts, li_posts)

    print("Parsing LinkedIn visitor export (Family D) ...")
    vis_daily, vis_demo = build_visitors(SRC["li_visitors"])

    print("Parsing follower counts (Family A, 082026 workbook) ...")
    follower_counts = build_followers(SRC["follower_counts"])

    print("Parsing Loomly click data (from manual-loomly-exporting.xlsx) ...")
    loomly_clicks = build_loomly(SRC["manual_loomly"])

    print("Parsing LinkedIn new-followers export (Family E) ...")
    new_followers = build_new_followers(SRC["li_followers"])

    print("Extracting Instagram Stories (Family F) ...")
    ig_stories = build_instagram_stories(SRC["content_analytics_wide"])

    print("Parsing competitor reference table (Family I) ...")
    competitors = build_competitors(SRC["competitors"])

    print("Adding SOURCE_PULL_DATE / IN_EXPERIMENT_PERIOD covariates ...")
    pull_date = dt.date.fromtimestamp(os.path.getmtime(SRC["manual_loomly"]))
    social_posts["SOURCE_PULL_DATE"] = pull_date
    social_posts["IN_EXPERIMENT_PERIOD"] = False  # date range TBD by Grace/Joshua; column exists, populate later

    print("Running pre-discard diffs on superseded/subset files (plan §2 step 2) ...")
    diff_follower_counts()
    diff_li_content_metrics()
    diff_li_visitor_metrics()

    print("Running QA pass ...")
    qa_text, monthly, thin, coverage = qa_report(social_posts)
    print("\n" + qa_text + "\n")

    social_posts = social_posts.sort_values(["DATE", "PLATFORM"]).reset_index(drop=True)

    sheets = {
        "Social Posts (unified)": social_posts,
        "LinkedIn Posts (detail)": li_posts,
        "LinkedIn Daily Metrics": li_metrics,
        "Visitor Daily Metrics": vis_daily,
        "Visitor Demographics": vis_demo,
        "Follower Counts": follower_counts,
        "Loomly Clicks": loomly_clicks,
        "Instagram Stories": ig_stories,
        "LinkedIn New Followers (daily)": new_followers,
        "Competitors": competitors,
        "LI Join Gaps (QA)": li_join_gaps,
        "QA Monthly Summary": monthly,
        "QA Thin Cells": thin,
        "QA Label Coverage": coverage,
    }

    write_workbook(OUT_WORKBOOK, sheets)
    pd.DataFrame(merge_decisions).to_csv(OUT_MERGE_LOG, index=False)
    pd.DataFrame(dedup_discrepancies).to_csv(OUT_DEDUP_LOG, index=False)

    print(f"\nWrote {OUT_WORKBOOK}")
    print(f"Wrote {OUT_MERGE_LOG} ({len(merge_decisions)} decisions)")
    print(f"Wrote {OUT_DEDUP_LOG} ({len(dedup_discrepancies)} discrepancies)")


if __name__ == "__main__":
    main()
