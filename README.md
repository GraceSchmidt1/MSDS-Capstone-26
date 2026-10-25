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
  the plan. Run with `python3 build_master_data.py` from `data/`. It refuses to
  run if `Hidalga_Master_Data_v1.xlsx` already exists, because v1 now carries
  manual tag fixes a rebuild would erase. Use `merge_into_master.py` for new data.
- **`merge_into_master.py`** — append-only update of the master with a fresh
  export (see "Adding new data" below).
- **`move_superseded.py`** — archives every raw file `build_master_data.py`
  did *not* use as a winning source into `data/data_prev/`, with a `NOTES.md`
  explaining why each one was set aside. Nothing is ever deleted, only moved.
- **`add_unified_engagement_rate.py`** — adds `ENGAGEMENT_RATE_UNIFIED_PCT`
  to `Social Posts (unified)` in an existing master workbook, in place, without
  re-running the full build (which would lose the manual tag fixes). The same
  logic lives in `build_master_data.add_unified_engagement_rate()` for future
  builds. Run with `python3 add_unified_engagement_rate.py` from `data/`.
- **`backfill_tags.py`** (repo root) — heuristic keyword-based tag suggester
  for posts in `Social Posts (unified)` that predate the current ICP/pillar
  taxonomy. Never silently fills in ground truth: it scores each untagged
  post's text against keyword sets drawn from the reference guide and writes
  every suggestion (auto-applied, needs-review, or insufficient-text) to
  `data/tag_backfill_review.csv` for a human to eyeball. Run with
  `python3 backfill_tags.py --input data/Hidalga_Master_Data_v1.xlsx --output <out.xlsx> --review data/tag_backfill_review.csv`.

### Adding new data (append-only updates)

```bash
cd data
# 1. clean the new export(s) to a scratch location (never to a Hidalga_Master_Data file)
python3 clean_hidalga_data.py <new exports...> -o /tmp/cleaned_YYYYMM.xlsx
# 2. append into the next master version
python3 merge_into_master.py --master Hidalga_Master_Data_v1.xlsx \
    --incoming /tmp/cleaned_YYYYMM.xlsx --pull-date YYYY-MM-DD
```

What `merge_into_master.py` guarantees:

- **The current master is never written to.** The output is the next version
  (`_v2`, `_v3`, …), which starts as a byte copy of the master. Rows are only
  appended. The script refuses to run if the output file already exists or is
  the master itself.
- **New rows are added, existing rows are never overwritten.** Each sheet has a
  dedup key (e.g. PLATFORM + DATE + POST_TITLE for posts, DATE for daily sheets,
  SHORT_LINK for Loomly clicks). Incoming rows with a new key are appended;
  rows whose key already exists leave the master row exactly as it was.
- **Nothing incoming is thrown away.** When an already-present row comes in
  with different values (e.g. a later pull with more impressions), each
  differing value is recorded in the `Unapplied Updates (log)` sheet as
  master value vs. incoming value. That sheet is also append-only.
- **It checks itself.** After writing, every original row of every sheet is
  re-read and compared cell-for-cell against the master, and the master's
  file hash is checked. Any difference deletes the output and exits with an error.
- **It stops on an unfamiliar export shape.** An incoming sheet without a
  defined key, or columns the master doesn't have, stop the run instead of
  being guessed at.
- New posts get `ENGAGEMENT_RATE_UNIFIED_PCT`, `SOURCE_PULL_DATE`, and
  `IN_EXPERIMENT_PERIOD` filled in. If the cleaned workbook includes the
  LinkedIn content export, new LinkedIn posts get native LinkedIn metrics.
- Each merge appends a summary to `merge_decisions.csv`.

Not automated: newly appended posts are tagged only by the raw Loomly LABELS
parse, so check them for missing ICP/pillar tags (the September test had 3
posts with no pillar). The `QA *` sheets are not regenerated by a merge.

`build_master_data.py`, `clean_hidalga_data.py`, and `backfill_tags.py` all
refuse to write over an existing master workbook.

**Verified (2026-10-25)** on a copy of v1 with
`data_csv/092026-Social-Media-Data.xlsx` (38 posts, Aug 26–Sep 29):

| Check | Result |
|---|---|
| 21 posts overlapping v1 | matched existing rows, not appended, master values kept |
| 17 posts newer than v1 | appended (374 → 391), 0 duplicates |
| Follower counts / Loomly clicks | +3 (Sep months) / +6 new links |
| Differing values on already-present rows | 145 logged, none applied |
| Original rows in output | all identical cell-for-cell; v1 file hash unchanged |
| Re-merging the same export | 0 rows appended |
| Output exists / output = master / edited cell | refused / refused / caught by verify |

### Merge history

| Version | Source | Pull date | Posts added | Notes |
|---|---|---|---|---|
| v1 | `build_master_data.py` (full consolidation) | 2026-09-25 | 374 | manual tag fixes applied afterward |
| v2 | `data/092026-Social-Media-Data.xlsx` (cumulative Jan 1–Sep 29, 313 posts) | 2026-10-25 | 17 (Sep 11–29) | 296 already in v1 kept as-is; +3 follower-count months, +6 Loomly links; 1,665 differing values in `Unapplied Updates (log)`. "Cut Denial Rework Before It Hits the Queue" (Sep 14, all 3 platforms) has no content pillar yet. |

### Current outputs (in `data/`)

- **`Hidalga_Master_Data_v2.xlsx`** — **current master** (v1 + September
  2026 posts, 391 social posts). Same sheets as v1 plus `Unapplied Updates (log)`.
  Use this one for modeling. See "Merge history" above.
- **`Hidalga_Master_Data_v1.xlsx`** — the original consolidated dataset, kept frozen. Sheets:
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

### Unified engagement rate (modeling target)

`ENGAGEMENT_RATE_PCT` mixes three definitions: Facebook's derived rate,
Instagram's Loomly rate (engagement ÷ **reach**), and LinkedIn's native rate.
`ENGAGEMENT_RATE_UNIFIED_PCT` applies one formula to every platform:
(reactions + comments + shares + clicks) ÷ impressions × 100.

- Facebook and LinkedIn: identical to `ENGAGEMENT_RATE_PCT` (LinkedIn's native
  rate turns out to be exactly this formula), except the 4 LinkedIn posts that
  fall back to Loomly numbers, which are now recomputed.
- Instagram: `LIKES` stands in for reactions, and the export has no shares or
  clicks columns, so the rate is effectively (likes + comments) ÷ impressions.
  Median drops from 37.5% to 17.1%. Instagram is therefore still not perfectly
  comparable (no clicks component), so keep a platform term in any pooled model.
- Null for the one Facebook post with 0 impressions (2025-12-02).

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
