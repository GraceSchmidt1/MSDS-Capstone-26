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
  for that ongoing monthly use. Two parsing fixes landed since the v1 run:
  `parse_labels()` now keeps *every* canonical content pillar a post is
  tagged with instead of only the first match (posts tagged with more than
  one pillar were silently losing the second); and the Loomly click-data
  sheet is now matched by normalized name instead of one exact literal, so
  a rename between export tools (`loomly-click-data` vs `Loomly_click`)
  doesn't make `build_loomly()` silently return an empty sheet.
- **`build_master_data.py`** — the one-time (v1) historical consolidation.
  Reuses `clean_hidalga_data.py`'s normalizers (label/pillar parsing,
  offset-sheet reading, format handling) and adds the source-selection,
  cross-file diffing, and LinkedIn engagement ground-truth logic described in
  the plan. Run with `python3 build_master_data.py` from `data/`.
- **`move_superseded.py`** — archives every raw file `build_master_data.py`
  did *not* use as a winning source into `data/data_prev/`, with a `NOTES.md`
  explaining why each one was set aside. Nothing is ever deleted, only moved.
- **`backfill_tags.py`** (repo root) — heuristic keyword-based tag suggester
  for posts in `Social Posts (unified)` that predate the current ICP/pillar
  taxonomy. Never silently fills in ground truth: it scores each untagged
  post's text against keyword sets drawn from the reference guide and writes
  every suggestion (auto-applied, needs-review, or insufficient-text) to
  `data/tag_backfill_review.csv` for a human to eyeball. Run with
  `python3 backfill_tags.py --input data/Hidalga_Master_Data_v1.xlsx --output <out.xlsx> --review data/tag_backfill_review.csv`.

### Current outputs (in `data/`)

- **`Hidalga_Master_Data_v1.xlsx`** — the consolidated dataset. Sheets:
  Social Posts (unified), LinkedIn Posts (detail), LinkedIn Daily Metrics,
  Visitor Daily Metrics, Visitor Demographics, Follower Counts, Loomly
  Clicks (89 rows — see "Loomly click attribution" below), Instagram
  Stories, LinkedIn New Followers (daily), Competitors, plus QA sheets
  (join gaps, monthly summary, thin cells, label coverage).
- **`merge_decisions.csv`** — every judgment call made while merging (e.g.
  each LinkedIn post row that had its Loomly-reported engagement numbers
  overwritten with LinkedIn's own native metrics), so it's auditable.
- **`dedup_discrepancies.csv`** — before/after diffs run against superseded
  files prior to archiving them, catching cases where an earlier partial
  pull's numbers were later revised (e.g. mid-month LinkedIn/visitor metrics
  that finalize after a delay).
- **`tag_backfill_review.csv`** — the human-review output of
  `backfill_tags.py`: every previously-untagged post with its suggested
  ICP segment(s)/content pillar, matched keywords, and a decision flag
  (`auto_applied`, `needs_review_*`, or `insufficient_text`).

### `data/data_prev/`

Holds the ~30 superseded, duplicate, or subset export files that were
replaced by a wider/later pull of the same underlying data (see its
`NOTES.md` for the reason each file was archived). `data/` itself now holds
only the winning source files, the reference taxonomy PDF, the scripts, and
the pipeline outputs above.

### Result summary (v1 run)

- 374 unified social posts (118 Facebook, 125 Instagram, 131 LinkedIn),
  spanning Sep 2025–Sep 2026, zero duplicates on (platform, date, title).
  (4 Facebook rows with no title/body/labels at all — dead Loomly export
  stubs with nothing to tag — were removed during the labeling pass below.)
- 127 of 131 LinkedIn posts carry LinkedIn-native engagement numbers as the
  metric of record (per the plan's FB/IG = Loomly, LinkedIn = LinkedIn-native
  ground-truth rule); the remaining 4 (posts newer than the LinkedIn export's
  cutoff) fall back to Loomly and are flagged for follow-up.
- Regression-checked against the prior `Hidalga_Cleaned_Data_2026-08.xlsx`
  output: the only count differences are 3 previously double-counted posts
  that the wider, later source correctly collapsed to one.

### ICP/pillar tagging backfill

Historical posts (Sep 2025 onward) predated the current Q1 ICP segment/content
pillar taxonomy and were largely untagged. `backfill_tags.py` generated
keyword-based suggestions (`data/tag_backfill_review.csv`), which were then
manually reviewed and applied, plus hand-fixes for non-canonical/corrupted
values (mismatched casing like `"team & Culture"`, old pillar names like
`"Patient-Centered Stories"`, and a handful of rows where format tags like
`Video`/`blog-link` or raw ICP text had leaked into the `CONTENT_PILLAR`
field instead of a real pillar). `Instagram Stories` also had its single
combined `LABELS` column split into separate `ICP_SEGMENTS` and
`CONTENT_PILLAR` columns to match `Social Posts (unified)`.

Current state, per `QA Label Coverage`: **100% ICP and 100% content-pillar
coverage** across Facebook, Instagram, and LinkedIn — every value is one of
the 4 canonical ICP segments / 5 canonical content pillars, no free text.

### Loomly click attribution (planned — no data changes made)

`Loomly Clicks` (89 rows) carries `SHORT_LINK`, `DESTINATION_URL`,
`CREATION_DATE`, `TOTAL_CLICKS`, and source/location breakdowns per link,
but nothing that maps a click record to one specific post — `LINK_TAGS` on
`Social Posts (unified)` holds category labels (e.g. `Financial-ICP-link`,
`generic-website-link`), not the literal `loom.ly` short link, so there's
no direct join key between the two sheets.

Decision: attribute clicks to posts by matching `CREATION_DATE` to posting
date against LinkedIn posts, rather than adding a join key to the data
now. This is a modeling-stage join, not a cleaning-stage fix — no
adjustments to `Loomly Clicks` or `Social Posts (unified)` are needed for
it.
