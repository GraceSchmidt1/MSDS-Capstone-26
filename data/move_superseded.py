#!/usr/bin/env python3
"""
move_superseded.py
===================
Archives every raw export file that build_master_data.py did NOT use as a
winning source into data/data_prev/, preserving the original filenames.
Nothing is ever deleted -- files only move within data/.

Run this only after build_master_data.py has completed successfully (so the
pre-discard diffs in dedup_discrepancies.csv have actually been run against
these files).
"""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PREV = os.path.join(HERE, "data_prev")
P = lambda name: os.path.join(HERE, name)

# (filename, reason) -- see DATA_CLEANING_PLAN.md for the family reasoning
MOVES = [
    # Family B -- fully superseded by manual-loomly-exporting.xlsx (plan §1a):
    # its date range (2025-12-02..2026-03-27) is entirely covered by
    # manual-loomly's wider range.
    ("Facebook-data.xlsx", "Family B: superseded by manual-loomly-exporting.xlsx (§1a)"),
    ("Instagram-data.xlsx", "Family B: superseded by manual-loomly-exporting.xlsx (§1a)"),
    ("linkedin-data.xlsx", "Family B: superseded by manual-loomly-exporting.xlsx (§1a)"),
    ("032826-Facebook-data(data-cleaned).csv", "Family B: CSV re-export of Facebook-data.xlsx's data-cleaned sheet; inferior duplicate (flattened rows, encoding issues) of an already-superseded file"),
    ("032826-Instagram-data(data-cleaned).csv", "Family B: CSV re-export of Instagram-data.xlsx's data-cleaned sheet; inferior duplicate of an already-superseded file"),
    ("032826-linkedin-data(data-cleaned).csv", "Family B: CSV re-export of linkedin-data.xlsx's data-cleaned sheet; inferior duplicate of an already-superseded file"),

    # Family A subsets -- 082026-Social-Media-Data.xlsx is the widest pull;
    # diffed against these in build_master_data.py (see dedup_discrepancies.csv).
    ("052026-Social-Media-Data.xlsx", "Family A: earlier partial pull, strict subset of 082026-Social-Media-Data.xlsx (no follower-counts sheet to diff)"),
    ("062026-Social-Media-Data.xlsx", "Family A: earlier partial pull, diffed against 082026-Social-Media-Data.xlsx follower counts -- see dedup_discrepancies.csv"),
    ("072026-Social-Media-Data.xlsx", "Family A: earlier partial pull, diffed against 082026-Social-Media-Data.xlsx follower counts -- see dedup_discrepancies.csv"),

    # Family A' / H -- pre-2026 account-level weekly rollups, excluded per §1c
    ("20251028-Social-Media-Data.xlsx", "Family A': pre-2026 weekly account rollup, predates ICP/pillar taxonomy -- excluded from pipeline per plan §1c"),
    ("SocialMediaDashboards.xlsx", "Family H: hand-maintained weekly account dashboard, predates taxonomy -- excluded from pipeline per plan §1c"),

    # Family C subsets -- hidalga-technologies_content_1788892250539.xls is the
    # widest/most recent rolling export; diffed in build_master_data.py.
    ("hidalga-012026-082026.xls", "Family C: rolling-window subset, diffed against the widest LinkedIn content export -- see dedup_discrepancies.csv"),
    ("hidalga-technologies_content_062026.xls", "Family C: rolling-window subset, diffed against the widest LinkedIn content export -- see dedup_discrepancies.csv"),
    ("hidalga-technologies_content_1776889133559.xls", "Family C: rolling-window subset, diffed against the widest LinkedIn content export -- see dedup_discrepancies.csv"),
    ("hidalga-technologies_content_1785352077908.xls", "Family C: rolling-window subset, diffed against the widest LinkedIn content export -- see dedup_discrepancies.csv"),
    ("hidalga-technologies_content_1786998188809.xls", "Family C: rolling-window subset, diffed against the widest LinkedIn content export -- see dedup_discrepancies.csv"),

    # Family D subsets -- hidalga-technologies_visitors_1788892288178.xls (365-row,
    # 2025-09-07..2026-09-06) is the widest; diffed in build_master_data.py.
    ("hidalga-012026082026.xls", "Family D: rolling-window subset, diffed against the widest LinkedIn visitors export -- see dedup_discrepancies.csv"),
    ("hidalga-technologies_visitors_062026.xls", "Family D: rolling-window subset, missing the Job function sheet the winner has -- diffed against the widest visitors export"),
    ("hidalga-technologies_visitors_1776889282185 copy.xls", "Family D: verified data-identical (Visitor metrics sheet, cell-for-cell) to hidalga-technologies_visitors_1776889282185.xls -- both kept, not deleted, both subsets of the widest export"),
    ("hidalga-technologies_visitors_1776889282185.xls", "Family D: rolling-window subset, diffed against the widest visitors export -- see dedup_discrepancies.csv"),
    ("hidalga-technologies_visitors_1779310462161.xls", "Family D: rolling-window subset (2026-01-01..05-18); verified data-identical (Visitor metrics sheet, cell-for-cell) to 052026-hidalga-technologies_visitors.xls -- both kept, both subsets of the widest export"),
    ("hidalga-technologies_visitors_1785352089815.xls", "Family D: rolling-window subset, diffed against the widest visitors export -- see dedup_discrepancies.csv"),
    ("hidalga-technologies_visitors_1786998157043.xls", "Family D: rolling-window subset, diffed against the widest visitors export -- see dedup_discrepancies.csv"),
    ("052026-hidalga-technologies_visitors.xls", "Family D: rolling-window subset (2026-01-01..05-18); verified data-identical (Visitor metrics sheet, cell-for-cell) to hidalga-technologies_visitors_1779310462161.xls -- both kept, both subsets of the widest export"),

    # Family F narrow monthly extracts -- subsets of the wide Content Analytics
    # workbook, used only for cross-checking, not as pipeline inputs.
    ("Hidalga Content Analytics 2026-01-01 - 2026-01-31.xlsx", "Family F: narrow monthly subset of the wide Content Analytics workbook (Oct 2025-Sep 2026); cross-check only, not a pipeline input"),
    ("Hidalga Content Analytics 2026-02-01 - 2026-02-28.xlsx", "Family F: narrow monthly subset of the wide Content Analytics workbook; cross-check only, not a pipeline input"),
    ("Hidalga Content Analytics 2026-03-01 - 2026-03-27.xlsx", "Family F: narrow monthly subset of the wide Content Analytics workbook; cross-check only, not a pipeline input"),

    # Family G -- single month, fully covered by winners above
    ("Marketing-Analytics-April-2026.xlsx", "Family G: single-month (Apr 2026) extract fully covered by manual-loomly-exporting.xlsx and the wide Content Analytics workbook"),

    # Family Z -- prior pipeline OUTPUTS, not raw inputs. Used only as the
    # regression check inside build_master_data.py's QA pass.
    ("Hidalga_Cleaned_Data(old08172026).xlsx", "Family Z: prior pipeline output, not a raw source -- kept only as historical reference"),
    ("Hidalga_Cleaned_Data_2026-08.xlsx", "Family Z: prior pipeline output, used as the regression-check baseline in build_master_data.py's QA pass"),
]

# Never touch these even though they live in data/: Excel lock files, macOS
# metadata, and anything build_master_data.py actively reads.
SKIP = {".DS_Store", "~$Hidalga Content Analytics 2025-10-01 - 2026-09-10.xlsx",
        "~$manual-loomly-exporting.xlsx"}


def main():
    os.makedirs(PREV, exist_ok=True)
    notes = ["# data_prev — archived superseded/duplicate source files",
             "",
             "Moved here by move_superseded.py after build_master_data.py produced",
             "Hidalga_Master_Data_v1.xlsx. Nothing was deleted. See",
             "DATA_CLEANING_PLAN.md for the full family-by-family reasoning, and",
             "dedup_discrepancies.csv (in data/) for the actual before/after diffs",
             "run against several of these files prior to archiving them.",
             ""]

    moved, missing = [], []
    for fname, reason in MOVES:
        src = P(fname)
        if not os.path.exists(src):
            missing.append(fname)
            continue
        dst = os.path.join(PREV, fname)
        shutil.move(src, dst)
        moved.append(fname)
        notes.append(f"- `{fname}`\n  {reason}")

    with open(os.path.join(PREV, "NOTES.md"), "w") as f:
        f.write("\n".join(notes) + "\n")

    print(f"Moved {len(moved)} files to {PREV}")
    if missing:
        print(f"Not found (already moved or renamed?): {missing}")

    remaining = sorted(
        f for f in os.listdir(HERE)
        if os.path.isfile(P(f)) and f not in SKIP and not f.endswith(".py")
    )
    print("\nRemaining in data/ (winners + outputs + reference):")
    for f in remaining:
        print(" ", f)


if __name__ == "__main__":
    main()
