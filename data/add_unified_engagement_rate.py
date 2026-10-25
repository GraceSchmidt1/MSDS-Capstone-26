#!/usr/bin/env python3
"""
add_unified_engagement_rate.py
===============================
Adds ENGAGEMENT_RATE_UNIFIED_PCT to `Social Posts (unified)` in an existing
master workbook, right after ENGAGEMENT_RATE_PCT, using the same formula as
build_master_data.add_unified_engagement_rate():

    (reactions + comments + shares + clicks) / impressions * 100

Exists because Hidalga_Master_Data_v1.xlsx carries manual tag fixes applied
after build_master_data.py ran, so re-running the full build would lose them.
This edits the workbook in place with openpyxl and leaves every other cell,
sheet, and style alone. Safe to re-run: an existing column is overwritten.

Usage (from data/):
    python3 add_unified_engagement_rate.py [--workbook Hidalga_Master_Data_v1.xlsx]
"""
from __future__ import annotations
import argparse
import copy
import os
import sys

import pandas as pd
import openpyxl
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_master_data import add_unified_engagement_rate  # noqa: E402

SHEET = "Social Posts (unified)"
COL = "ENGAGEMENT_RATE_UNIFIED_PCT"
ANCHOR = "ENGAGEMENT_RATE_PCT"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workbook", default=os.path.join(HERE, "Hidalga_Master_Data_v1.xlsx"))
    args = ap.parse_args(argv)

    df = pd.read_excel(args.workbook, sheet_name=SHEET)
    df = add_unified_engagement_rate(df.drop(columns=[COL], errors="ignore"))

    wb = openpyxl.load_workbook(args.workbook)
    ws = wb[SHEET]
    headers = [c.value for c in ws[1]]
    if COL in headers:
        col_idx = headers.index(COL) + 1
    else:
        col_idx = headers.index(ANCHOR) + 2
        # insert_cols doesn't shift column widths, so carry them over by hand
        widths = {i: ws.column_dimensions[get_column_letter(i)].width for i in range(col_idx, ws.max_column + 1)}
        ws.insert_cols(col_idx)
        for i, w in widths.items():
            ws.column_dimensions[get_column_letter(i + 1)].width = w
        anchor_cell = ws.cell(row=1, column=col_idx - 1)
        head = ws.cell(row=1, column=col_idx, value=COL)
        head.font, head.fill, head.alignment = (copy.copy(anchor_cell.font), copy.copy(anchor_cell.fill),
                                                copy.copy(anchor_cell.alignment))
        ws.column_dimensions[get_column_letter(col_idx)].width = len(COL) + 2

    for i, v in enumerate(df[COL], start=2):
        ws.cell(row=i, column=col_idx, value=None if pd.isna(v) else float(v))
    wb.save(args.workbook)

    summary = df.groupby("PLATFORM").agg(
        posts=(COL, "size"), null=(COL, lambda s: s.isna().sum()),
        old_median=(ANCHOR, "median"), unified_median=(COL, "median"), unified_mean=(COL, "mean"))
    print(f"Wrote {COL} to '{SHEET}' in {args.workbook}\n")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
