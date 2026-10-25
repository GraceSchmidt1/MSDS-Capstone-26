#!/usr/bin/env python3
"""
apply_master_corrections.py
============================
Applies reviewed, one-off data corrections to a master workbook and writes the
next version. Like merge_into_master.py, it never writes to the input master,
refuses to overwrite an existing output, and verifies afterward that the only
cells that changed outside the QA sheets are the ones listed in
`Corrections (log)`.

Corrections (decided 2026-10-25):
  1. CONTENT_PILLAR blank but LABELS_RAW holds only canonical pillar name(s)
     (manual tag fixes landed in the wrong column) -> copy LABELS_RAW across.
  2. PILLAR_OVERRIDES: posts whose raw labels are garbled get a reviewed pillar.
  3. IN_EXPERIMENT_PERIOD = True for posts dated within EXPERIMENT_PERIOD
     (Q2-Q3 2026). The rows stay in the modeling data; the flag is a covariate.
  4. QA sheets (monthly summary, thin cells, label coverage) are regenerated so
     they describe the corrected data.

Usage (from data/):
    python3 apply_master_corrections.py --master Hidalga_Master_Data_v2.xlsx
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import os
import shutil
import sys

import pandas as pd
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from merge_into_master import norm, next_version_path, sha256  # noqa: E402

POSTS = "Social Posts (unified)"
LOG_SHEET = "Corrections (log)"
LOG_COLS = ["CORRECTED_AT", "SHEET", "ROW", "PLATFORM", "DATE", "POST_TITLE", "COLUMN",
            "OLD_VALUE", "NEW_VALUE", "REASON"]
QA_SHEETS = ["QA Monthly Summary", "QA Thin Cells", "QA Label Coverage"]

CANONICAL_PILLARS = {
    "Thought Leadership & Industry Insights",
    "Product Development & Innovations",
    "Events & Partnerships",
    "Patient-Centered Storytelling",
    "Team Spotlights & Company Culture",
}
# (PLATFORM or None for all platforms, POST_TITLE) -> pillar
PILLAR_OVERRIDES = {
    (None, "Cut Denial Rework Before It Hits the Queue"): "Thought Leadership & Industry Insights",
}
EXPERIMENT_PERIOD = (dt.date(2026, 4, 1), dt.date(2026, 9, 30))  # Q2-Q3 2026, inclusive


def plan_corrections(sp):
    """Return [(row_idx, column, old, new, reason)] without touching anything."""
    changes = []
    for i, r in sp.iterrows():
        pillar, labels = r["CONTENT_PILLAR"], r["LABELS_RAW"]
        override = (PILLAR_OVERRIDES.get((r["PLATFORM"], r["POST_TITLE"]))
                    or PILLAR_OVERRIDES.get((None, r["POST_TITLE"])))
        if override and pillar != override:
            changes.append((i, "CONTENT_PILLAR", pillar, override, "reviewed pillar for post with garbled labels"))
        elif pd.isna(pillar) and isinstance(labels, str):
            parts = [p.strip() for p in labels.split(";") if p.strip()]
            if parts and all(p in CANONICAL_PILLARS for p in parts):
                changes.append((i, "CONTENT_PILLAR", pillar, "; ".join(parts),
                                "pillar was in LABELS_RAW, CONTENT_PILLAR blank"))

        in_period = EXPERIMENT_PERIOD[0] <= pd.Timestamp(r["DATE"]).date() <= EXPERIMENT_PERIOD[1]
        if bool(r["IN_EXPERIMENT_PERIOD"]) != in_period:
            changes.append((i, "IN_EXPERIMENT_PERIOD", r["IN_EXPERIMENT_PERIOD"], in_period,
                            f"experiment period {EXPERIMENT_PERIOD[0]}..{EXPERIMENT_PERIOD[1]} (Q2-Q3 2026)"))
    return changes


def qa_tables(sp):
    sp = sp.copy()
    sp["MONTH"] = pd.to_datetime(sp["DATE"]).dt.to_period("M").astype(str)
    monthly = sp.groupby(["MONTH", "PLATFORM"]).agg(
        posts=("PLATFORM", "size"),
        mean_er_unified=("ENGAGEMENT_RATE_UNIFIED_PCT", "mean"),
        mean_er_mixed=("ENGAGEMENT_RATE_PCT", "mean"),
        in_experiment=("IN_EXPERIMENT_PERIOD", "sum")).round(3).reset_index()

    def explode(col):
        e = sp.assign(VALUE=sp[col].fillna("").str.split("; ")).explode("VALUE")
        return e[e["VALUE"] != ""]
    thin = pd.concat([
        explode(c).groupby(["VALUE", "PLATFORM"]).size().reset_index(name="n").assign(DIMENSION=c)
        for c in ["ICP_SEGMENTS", "CONTENT_PILLAR"]])
    thin = thin[thin["n"] < 3][["DIMENSION", "VALUE", "PLATFORM", "n"]]

    coverage = sp.groupby("PLATFORM").agg(
        n=("PLATFORM", "size"),
        pct_icp_tagged=("ICP_SEGMENTS", lambda s: round(s.notna().mean() * 100, 1)),
        pct_pillar_tagged=("CONTENT_PILLAR", lambda s: round(s.notna().mean() * 100, 1))).reset_index()
    return {"QA Monthly Summary": monthly, "QA Thin Cells": thin, "QA Label Coverage": coverage}


def replace_sheet(wb, name, df, style_from):
    idx = wb.sheetnames.index(name) if name in wb.sheetnames else len(wb.sheetnames)
    if name in wb.sheetnames:
        del wb[name]
    ws = wb.create_sheet(name, idx)
    rows = [list(df.columns)] + df.astype(object).where(df.notna(), None).values.tolist()
    for r in rows:
        ws.append(r)
    for c in ws[1]:
        c.font, c.fill, c.alignment = (copy.copy(style_from.font), copy.copy(style_from.fill),
                                       copy.copy(style_from.alignment))
    ws.freeze_panes = "A2"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Apply reviewed corrections to a master workbook (writes next version).")
    ap.add_argument("--master", required=True)
    ap.add_argument("--output", help="Defaults to the next version (_vN+1) next to the master")
    args = ap.parse_args(argv)

    master_path = os.path.abspath(args.master)
    out_path = os.path.abspath(args.output or next_version_path(master_path))
    if out_path == master_path or os.path.exists(out_path):
        sys.exit(f"Refusing to write {out_path}: it is the master or already exists.")
    master_hash = sha256(master_path)

    master = pd.read_excel(master_path, sheet_name=None)
    sp = master[POSTS].copy()
    changes = plan_corrections(sp)

    shutil.copy2(master_path, out_path)
    wb = openpyxl.load_workbook(out_path)
    ws = wb[POSTS]
    header = [c.value for c in ws[1]]
    now = dt.datetime.now().replace(microsecond=0)
    log = []
    for i, col, old, new, reason in changes:
        ws.cell(row=i + 2, column=header.index(col) + 1, value=new)
        sp.at[i, col] = new
        r = sp.loc[i]
        log.append([now, POSTS, i + 2, r["PLATFORM"], pd.Timestamp(r["DATE"]).to_pydatetime(), r["POST_TITLE"],
                    col, None if pd.isna(old) else old, new, reason])

    style = ws.cell(row=1, column=1)
    for name, df in qa_tables(sp).items():
        replace_sheet(wb, name, df, style)
    if LOG_SHEET not in wb.sheetnames:
        replace_sheet(wb, LOG_SHEET, pd.DataFrame(columns=LOG_COLS), style)
    for row in log:
        wb[LOG_SHEET].append(row)
    wb.save(out_path)

    # Verify: outside QA/log sheets, exactly the planned cells changed.
    out = pd.read_excel(out_path, sheet_name=None)
    expected = {(i + 2, col) for i, col, *_ in changes}
    problems = []
    for name, mdf in master.items():
        if name in QA_SHEETS or name == LOG_SHEET:
            continue
        odf = out[name]
        if list(odf.columns) != list(mdf.columns) or len(odf) != len(mdf):
            problems.append(f"{name}: shape or columns changed")
            continue
        before, after = mdf.map(norm), odf.map(norm)
        diff = {(r + 2, c) for c in mdf.columns for r in range(len(mdf)) if before.at[r, c] != after.at[r, c]}
        if name == POSTS:
            if diff != expected:
                problems.append(f"{name}: unexpected cell changes {sorted(diff ^ expected)[:5]}")
        elif diff:
            problems.append(f"{name}: {len(diff)} cell(s) changed")
    if sha256(master_path) != master_hash:
        problems.append("master file changed on disk")
    if problems:
        os.remove(out_path)
        sys.exit("VERIFY FAILED, output removed:\n  " + "\n  ".join(problems))

    by = pd.DataFrame(changes, columns=["row", "column", "old", "new", "reason"]).groupby(["column", "reason"]).size()
    print(by.to_string())
    print(f"\n{len(changes)} cells corrected, logged to '{LOG_SHEET}'. QA sheets regenerated.")
    print(f"Verified: only the logged cells changed; master untouched.\nWrote {out_path}")


if __name__ == "__main__":
    main()
