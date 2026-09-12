# MSDS-Capstone-26

## Data pipeline

Raw marketing exports live in `data/`. They arrive as repeated, overlapping
pulls of the same growing Loomly/LinkedIn accounts rather than distinct
datasets, so the core task is picking the single best (most complete, most
recent) source per record rather than concatenating everything. The full
reasoning behind every source-selection decision is in
[`DATA_CLEANING_PLAN.md`](DATA_CLEANING_PLAN.md).

### Scripts (in `data/`)

- **`clean_hidalga_data.py`** — the original monthly cleaner. Point it at
  exactly 3 fresh exports (social workbook + LinkedIn content export +
  LinkedIn visitors export) and it produces one cleaned workbook. Kept as-is
  for that ongoing monthly use.
- **`build_master_data.py`** — the one-time (v1) historical consolidation.
  Reuses `clean_hidalga_data.py`'s normalizers (label/pillar parsing,
  offset-sheet reading, format handling) and adds the source-selection,
  cross-file diffing, and LinkedIn engagement ground-truth logic described in
  the plan. Run with `python3 build_master_data.py` from `data/`.
- **`move_superseded.py`** — archives every raw file `build_master_data.py`
  did *not* use as a winning source into `data/data_prev/`, with a `NOTES.md`
  explaining why each one was set aside. Nothing is ever deleted, only moved.

### Current outputs (in `data/`)

- **`Hidalga_Master_Data_v1.xlsx`** — the consolidated dataset. Sheets:
  Social Posts (unified), LinkedIn Posts (detail), LinkedIn Daily Metrics,
  Visitor Daily Metrics, Visitor Demographics, Follower Counts, Loomly
  Clicks, Instagram Stories, LinkedIn New Followers (daily), Competitors,
  plus QA sheets (join gaps, monthly summary, thin cells, label coverage).
- **`merge_decisions.csv`** — every judgment call made while merging (e.g.
  each LinkedIn post row that had its Loomly-reported engagement numbers
  overwritten with LinkedIn's own native metrics), so it's auditable.
- **`dedup_discrepancies.csv`** — before/after diffs run against superseded
  files prior to archiving them, catching cases where an earlier partial
  pull's numbers were later revised (e.g. mid-month LinkedIn/visitor metrics
  that finalize after a delay).

### `data/data_prev/`

Holds the ~30 superseded, duplicate, or subset export files that were
replaced by a wider/later pull of the same underlying data (see its
`NOTES.md` for the reason each file was archived). `data/` itself now holds
only the winning source files, the reference taxonomy PDF, the scripts, and
the pipeline outputs above.

### Result summary (v1 run)

- 378 unified social posts (122 Facebook, 125 Instagram, 131 LinkedIn),
  spanning Sep 2025–Sep 2026, zero duplicates on (platform, date, title).
- 127 of 131 LinkedIn posts carry LinkedIn-native engagement numbers as the
  metric of record (per the plan's FB/IG = Loomly, LinkedIn = LinkedIn-native
  ground-truth rule); the remaining 4 (posts newer than the LinkedIn export's
  cutoff) fall back to Loomly and are flagged for follow-up.
- Regression-checked against the prior `Hidalga_Cleaned_Data_2026-08.xlsx`
  output: the only count differences are 3 previously double-counted posts
  that the wider, later source correctly collapsed to one.
