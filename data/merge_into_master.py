#!/usr/bin/env python3
"""
merge_into_master.py
=====================
Append-only update of the master workbook (DATA_CLEANING_PLAN.md §1e).

Takes the current master (Hidalga_Master_Data_vN.xlsx) plus one workbook
cleaned by clean_hidalga_data.py from a fresh export, and writes the next
version (Hidalga_Master_Data_vN+1.xlsx). Guarantees:

  - The input master file is never written to.
  - The output starts as a byte copy of the master; rows are only appended
    with openpyxl, so every existing cell (including manual tag fixes) is
    left exactly as it was.
  - Incoming rows whose key is not in the master are appended.
  - Incoming rows whose key IS already in the master never overwrite it.
    Any value that differs is recorded in the `Unapplied Updates (log)`
    sheet (also append-only), so newer numbers are kept, not lost.
  - After writing, the output is re-read and every original row of every
    sheet is checked cell-for-cell against the master. If anything changed,
    the output is deleted and the script exits non-zero.

Incoming sheets the merge doesn't know how to key, and incoming columns the
master doesn't have, stop the run instead of being guessed at (plan §1e.5).

Usage (from data/):
    python3 clean_hidalga_data.py <new exports...> -o /tmp/cleaned.xlsx
    python3 merge_into_master.py --master Hidalga_Master_Data_v1.xlsx \
        --incoming /tmp/cleaned.xlsx --pull-date 2026-10-01
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_master_data import (  # noqa: E402
    add_unified_engagement_rate, apply_linkedin_engagement_override,
)

# Dedup key per sheet: an incoming row matching an existing key is not appended.
SHEET_KEYS = {
    "Social Posts (unified)": ["PLATFORM", "DATE", "POST_TITLE"],
    "LinkedIn Posts (detail)": ["Post link"],
    "LinkedIn Daily Metrics": ["DATE"],
    "Visitor Daily Metrics": ["DATE"],
    "Visitor Demographics": ["DIMENSION", "DIMENSION_VALUE"],
    "Follower Counts": ["CHANNEL", "PROFILE", "MONTH"],
    "Loomly Clicks": ["SHORT_LINK"],
    "Instagram Stories": ["DATE", "STORY"],
    "LinkedIn New Followers (daily)": ["DATE"],
    "Competitors": ["SNAPSHOT_END_DATE", "Page"],
}
LOG_SHEET = "Unapplied Updates (log)"
LOG_COLS = ["MERGED_AT", "SOURCE_PULL_DATE", "INCOMING_FILE", "SHEET", "KEY", "COLUMN",
            "MASTER_VALUE", "INCOMING_VALUE"]
# Columns that legitimately differ between master and incoming and aren't data updates
IGNORE_IN_DIFF = {"SOURCE_FILE"}


class MergeError(Exception):
    """The merge can't go ahead safely (unknown sheet or column) or failed its verification."""


def norm(v):
    """Comparable string form of a cell value (keys and change detection)."""
    if v is None or v is pd.NaT or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (pd.Timestamp, dt.datetime)):
        return v.date().isoformat() if v.time() == dt.time(0) else v.isoformat()
    if isinstance(v, dt.date):
        return v.isoformat()
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, float, np.number)):
        return f"{float(v):.4f}"
    return " ".join(str(v).split()).casefold()


def row_key(row, cols):
    parts = [norm(row.get(c)) for c in cols]
    # untitled posts: fall back to the body so same-day untitled posts don't collide
    if cols == SHEET_KEYS["Social Posts (unified)"] and not parts[-1]:
        parts[-1] = norm(str(row.get("POST_BODY") or "")[:120])
    return tuple(parts)


def to_cell(v):
    if v is None or v is pd.NaT or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if isinstance(v, np.generic):
        return v.item()
    return v


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def next_version_path(master):
    m = re.search(r"_v(\d+)\.xlsx$", master)
    if not m:
        sys.exit(f"Can't infer version from {master!r}; pass --output.")
    return master[:m.start()] + f"_v{int(m.group(1)) + 1}.xlsx"


def prepare_new_social_posts(new, incoming_sheets, pull_date):
    """Bring freshly-cleaned posts up to the master's Social Posts schema."""
    new = new.copy()
    new["DATE"] = pd.to_datetime(new["DATE"], errors="coerce").dt.date
    new["LOOMLY_ENGAGEMENT_RATE_PCT_RAW"] = np.where(new["PLATFORM"] == "LinkedIn", new["ENGAGEMENT_RATE_PCT"], None)
    new["ENGAGEMENT_SOURCE"] = "Loomly"
    li_detail = incoming_sheets.get("LinkedIn Posts (detail)")
    if li_detail is not None and not li_detail.empty:
        new, _ = apply_linkedin_engagement_override(new, li_detail)  # plan §1b, new rows only
    new = add_unified_engagement_rate(new.drop(columns=["ENGAGEMENT_RATE_UNIFIED_PCT"], errors="ignore"))
    new["SOURCE_PULL_DATE"] = pull_date
    new["IN_EXPERIMENT_PERIOD"] = False  # date range TBD, same as build_master_data.py
    new["DATE"] = pd.to_datetime(new["DATE"])
    return new


def merge(master_path, incoming_path, out_path, pull_date):
    master = pd.read_excel(master_path, sheet_name=None)
    incoming = {k: v for k, v in pd.read_excel(incoming_path, sheet_name=None).items()
                if list(v.columns) != ["(no data)"]}

    unknown = [s for s in incoming if s not in SHEET_KEYS]
    if unknown:
        raise MergeError(f"Incoming sheets with no merge key defined: {unknown}. Add them to SHEET_KEYS first.")

    merged_at = dt.datetime.now().replace(microsecond=0)
    appends, log_rows, summary = {}, [], []
    for sheet, inc in incoming.items():
        if sheet == "Social Posts (unified)":
            inc = prepare_new_social_posts(inc, incoming, pull_date)
        base = master.get(sheet, pd.DataFrame(columns=inc.columns))
        extra = [c for c in inc.columns if c not in base.columns]
        if extra:
            raise MergeError(f"{sheet}: incoming columns not in master: {extra}. Refusing to guess a mapping.")

        keys = SHEET_KEYS[sheet]
        existing = {row_key(r, keys): r for _, r in base.iterrows()}
        new_rows, seen = [], set()
        for _, r in inc.iterrows():
            k = row_key(r, keys)
            if k in existing:
                old = existing[k]
                for c in inc.columns:
                    if c not in IGNORE_IN_DIFF and norm(old.get(c)) != norm(r.get(c)):
                        log_rows.append([merged_at, pull_date, os.path.basename(incoming_path), sheet,
                                         " | ".join(k), c, to_cell(old.get(c)), to_cell(r.get(c))])
            elif k not in seen:
                seen.add(k)
                new_rows.append(r)
        appends[sheet] = pd.DataFrame(new_rows, columns=inc.columns)
        summary.append((sheet, len(base), len(inc), len(new_rows), len(inc) - len(new_rows)))

    # Write: copy master byte-for-byte, then only append.
    shutil.copy2(master_path, out_path)
    wb = openpyxl.load_workbook(out_path)
    for sheet, rows in appends.items():
        if sheet not in wb.sheetnames:
            wb.create_sheet(sheet).append(list(rows.columns))
        ws = wb[sheet]
        header = [c.value for c in ws[1]]
        for _, r in rows.iterrows():
            ws.append([to_cell(r[c]) if c in rows.columns else None for c in header])
    if log_rows:
        if LOG_SHEET not in wb.sheetnames:
            wb.create_sheet(LOG_SHEET).append(LOG_COLS)
            src = wb.worksheets[0][1][0]
            for c in wb[LOG_SHEET][1]:
                c.font, c.fill, c.alignment = copy.copy(src.font), copy.copy(src.fill), copy.copy(src.alignment)
            wb[LOG_SHEET].freeze_panes = "A2"
        for lr in log_rows:
            wb[LOG_SHEET].append(lr)
    wb.save(out_path)
    return master, appends, log_rows, summary


def verify(master, out_path, appends, n_logged):
    """Every original row unchanged, and row counts = original + appended."""
    out = pd.read_excel(out_path, sheet_name=None)
    problems = []
    for sheet, mdf in master.items():
        odf = out.get(sheet)
        if odf is None:
            problems.append(f"{sheet}: sheet missing from output")
            continue
        expected = len(mdf) + len(appends.get(sheet, []))
        if sheet == LOG_SHEET:
            expected = len(mdf) + n_logged
        if len(odf) != expected:
            problems.append(f"{sheet}: {len(odf)} rows, expected {expected}")
        if list(odf.columns[:len(mdf.columns)]) != list(mdf.columns):
            problems.append(f"{sheet}: column order/names changed")
            continue
        before = mdf.map(norm).to_numpy()
        after = odf[mdf.columns].iloc[:len(mdf)].map(norm).to_numpy()
        if before.shape != after.shape or (before != after).any():
            n = int((before != after).sum()) if before.shape == after.shape else "?"
            problems.append(f"{sheet}: {n} existing cell(s) changed")
    return problems


def update_master(master_path, incoming_path, out_path, pull_date):
    """merge() + verify(); the output is removed and MergeError raised if any original row changed or the master
    was touched. Returns (summary rows: sheet, master rows, incoming rows, appended, kept; number of values logged)."""
    master_hash = sha256(master_path)
    master, appends, log_rows, summary = merge(master_path, incoming_path, out_path, pull_date)
    problems = verify(master, out_path, appends, len(log_rows))
    if sha256(master_path) != master_hash:
        problems.append("master file changed on disk during the merge")
    if problems:
        os.remove(out_path)
        raise MergeError("VERIFY FAILED, output removed:\n  " + "\n  ".join(problems))
    return summary, len(log_rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Append-only merge of a cleaned export into the master workbook.")
    ap.add_argument("--master", required=True, help="Current master, e.g. Hidalga_Master_Data_v1.xlsx")
    ap.add_argument("--incoming", required=True, help="Workbook produced by clean_hidalga_data.py")
    ap.add_argument("--pull-date", required=True, type=dt.date.fromisoformat,
                    help="Date the raw export was pulled (YYYY-MM-DD), stored as SOURCE_PULL_DATE")
    ap.add_argument("--output", help="Defaults to the next version (_vN+1) next to the master")
    args = ap.parse_args(argv)

    master_path = os.path.abspath(args.master)
    out_path = os.path.abspath(args.output or next_version_path(master_path))
    if out_path == master_path:
        sys.exit("Output path is the master itself; refusing to write over it.")
    if os.path.exists(out_path):
        sys.exit(f"{out_path} already exists; refusing to overwrite it.")

    try:
        summary, n_logged = update_master(master_path, args.incoming, out_path, args.pull_date)
    except MergeError as e:
        sys.exit(str(e))

    log_csv = os.path.join(os.path.dirname(master_path), "merge_decisions.csv")
    pd.DataFrame([{"CATEGORY": "append_only_merge", "KEY": sheet,
                   "DECISION": f"appended {n_new} of {n_in} incoming rows",
                   "DETAIL": f"{n_dup} already in master, not overwritten; "
                             f"{os.path.basename(args.incoming)} -> {os.path.basename(out_path)}"}
                  for sheet, _, n_in, n_new, n_dup in summary]).to_csv(
        log_csv, mode="a", header=not os.path.exists(log_csv), index=False)

    print(f"{'sheet':32} {'master':>7} {'incoming':>9} {'appended':>9} {'kept':>6}")
    for sheet, n_base, n_in, n_new, n_dup in summary:
        print(f"{sheet:32} {n_base:7} {n_in:9} {n_new:9} {n_dup:6}")
    print(f"\n{n_logged} differing values logged to '{LOG_SHEET}' (master values kept).")
    print(f"Verified: all original rows unchanged; master untouched.\nWrote {out_path}")


if __name__ == "__main__":
    main()
