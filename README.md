# MSDS-Capstone-26

## Data pipeline

Raw marketing exports live in `data/`. They arrive as repeated, overlapping
pulls of the same growing Loomly/LinkedIn accounts rather than distinct
datasets, so the core task is picking the single best (most complete, most
recent) source per record rather than concatenating everything. The full
reasoning behind every source-selection decision is in
[`DATA_CLEANING_PLAN.md`](DATA_CLEANING_PLAN.md).

### Project status (2026-10-25)

Data preparation, exploratory analysis, and the **Stage 1 engagement-rate
regression** are complete. **Current master: `data/Hidalga_Master_Data_v3.xlsx`**:
391 posts (124 Facebook, 130 Instagram, 137 LinkedIn), Sep 23 2025 – Sep 29 2026.

Done so far:

1. **Consolidation (v1).** ~35 overlapping raw exports reduced to one winning
   source per data family, deduplicated, with superseded files archived to
   `data/data_prev/` (see `DATA_CLEANING_PLAN.md`).
2. **ICP/pillar tag backfill.** Keyword suggestions plus manual review for
   posts that predate the taxonomy.
3. **Unified engagement rate.** Added `ENGAGEMENT_RATE_UNIFIED_PCT` =
   (reactions + comments + shares + clicks) ÷ impressions × 100, one formula
   for all platforms, as the modeling target (see section below).
4. **Append-only update pipeline.** `merge_into_master.py` adds new exports
   as a new master version without changing existing rows. Build, clean, and
   backfill scripts now refuse to overwrite a master workbook.
5. **September 2026 merge (v2).** +17 posts (Sep 11–29), +3 follower-count
   months, +6 Loomly links.
6. **Corrections (v3).** 40 posts' content pillar moved from `LABELS_RAW` into
   `CONTENT_PILLAR`, one garbled-label post given its reviewed pillar,
   `IN_EXPERIMENT_PERIOD` set for Q2–Q3 2026, and QA sheets regenerated.
   100% ICP and pillar coverage. Follower Counts month labels later
   standardized in v3 (Jan … Sept); LinkedIn's January follower count of 0 is
   kept as exported.
7. **Exploratory analysis.** `notebooks/01_eda_master_v3.ipynb`, merged to
   `main` via PR #1 (see below).
8. **Stage 1 regression on engagement rate.** Per-platform models with a pooled
   check, built in `modeling/` with tests and versioned run outputs, written up
   in `notebooks/02_regression_engagement_rate.ipynb` (`modeling` branch; see
   "Stage 1 modeling" below).

Scope decisions:

- **No pre-Sep-2025 backfill.** Older post-level exports aren't accessible,
  so modeling uses the 391 posts available rather than the proposal's ~550 target.
- **Experiment-period posts stay in.** Q2–Q3 2026 content experiments were
  general and still help the modeling, so they're flagged with
  `IN_EXPERIMENT_PERIOD` instead of being excluded.
- **Most recent pull wins for metrics.** Values already in the master
  are from the latest pull (Sep 25), so none of the logged differences from
  the September workbook were applied (see "Merge history").
- **Compare within platform, not across.** Platforms differ in audience size,
  impression counting, and engagement components (Instagram has no clicks or
  shares), so EDA and models treat each platform separately; pooling is a check.
- **The duplicated LinkedIn metrics on 2026-03-20 are left as is** (two posts
  sharing one native metrics row; flagged in the EDA, not corrected).

Stage 1 classification (`classification` branch, merged in PR #3): C1–C5 and
the quantitative evaluation framework, items 1–7, written up in notebook 03.

Stage 2 (`recommendations` branch): a recommendation tool for the marketing
team, built on how past posts performed (no LLM). See "Using the recommendation
tool" below; results in `FINDINGS.md`. Remaining: PR to `main`.

### Exploratory analysis (`EDA` branch)

[`notebooks/01_eda_master_v3.ipynb`](notebooks/01_eda_master_v3.ipynb) is a
documented EDA of v3, run from `notebooks/`. It reads the master read-only
(file hash checked at the end), drops no rows (unstable posts are flagged,
not removed), and compares **within each platform** only. Group means are shown
with 95% bootstrap CIs and an impression-weighted rate. Figures are saved to
`notebooks/figures/`. Sections: data inventory and quality → target
distributions → pillar / ICP / format → timing → monthly trends and experiment
period → cross-posting and tag combinations → supporting sheets (followers,
LinkedIn daily, visitors, Loomly clicks, Stories, competitors) → data-quality
list, findings, modeling implications.

## Stage 1 modeling: engagement-rate regression (`modeling` branch)

**Question:** which content features, controlling for reach, are associated
with a post's engagement rate, and how well can it be predicted?
**Write-up:** [`notebooks/02_regression_engagement_rate.ipynb`](notebooks/02_regression_engagement_rate.ipynb)
(reads the committed runs; does not refit, except a checked Stage 2 preview).

### Design

| | |
|---|---|
| Unit | one row per platform post; ICP segments and pillars as 0/1 columns (no exploded rows) |
| Target | `ENGAGEMENT_RATE_UNIFIED_PCT`, in percentage points; 390 posts (the 0-impression post has no rate) |
| Main analysis | one model per platform (FB 123, IG 130, LI 137); pooled model with platform intercepts as a check |
| Features | `log2_impressions` (reach control), 4 ICP, 4 pillar (ref: Thought Leadership), format (ref: Image), weekend, after 11am, LinkedIn paid boost |
| Models | reference (mean), OLS (HC3 SEs; pooled: clustered by content piece), ridge, lasso, elastic net |
| Validation | repeated 5-fold × 10; regularized penalty tuned by nested CV; cross-posted content kept in one fold |
| Trust checks | 500 bootstrap refits, permutation importance on held-out folds, 5 sensitivity variants |

### Code (`modeling/`)

- `config.yaml`: every setting (master path, vocabularies, reference
  categories, formats per platform, CV, seed, sensitivity variants).
- `features.py`: `build_features()` for real posts, `features_from_spec()` for
  hypothetical posts (Stage 2); one shared encoder, so both encode identically.
- `models.py`: CV splits, OLS, regularized models, VIF / influence, bootstrap
  stability, permutation importance. `sensitivity.py`: one-change variants.
- `run_regression.py` / `run_sensitivity.py`: CLIs that write a run folder.
- `tests/`: 53 tests (no silent drops, one row per post, encoding matches tags,
  determinism, spec round-trip, nested CV uses training posts only, …).

```bash
python -m pytest modeling/tests -q
python -m modeling.run_regression --tag <name>     # steps 2–4, ~1 min
python -m modeling.run_sensitivity --tag <name>    # step 5, ~1 min
```

Each run writes `outputs/runs/<date>_<tag>/` with CSVs, figures, and
`run_info.json` (master path + SHA-256, git commit and clean/dirty flag,
config snapshot, package versions, seed). Reruns are byte-identical. To use a new
master, change `master:` in the config and rerun with a new tag.

Committed runs: `2026-10-25_step2_baseline`, `_step3_regularized`,
`_step4_importance` (contains steps 2–4), `_step5_sensitivity`.

### Results

Held-out R² (elastic net vs. reference mean): **Instagram 0.44** (0.57 with a
log target), LinkedIn ≈0 (MAE 8% lower than the mean; 0.19 without one
3-impression outlier), **Facebook ≈ mean (not predictable)**, pooled 0.24.

| Finding | Effect | Robust? |
|---|---|---|
| LinkedIn **Document** posts | ≈ +12 pts vs. Image (+8 to +18 across variants) | **Yes**: 5/5 variants, 96% of refits, holds with the experiment flag. All 13 ran Jul–Sep 2026, so a period-specific factor can't be fully excluded. |
| Instagram **reach** | ≈ −6 pts per doubling of impressions | **Yes**: 100% of refits, the dominant feature |
| **Video** | not different within any platform once reach is controlled | The EDA's "video is worst" was a reach effect |
| Instagram Carousel, LinkedIn Multi-Image | +2 to +5 pts | Suggestive (positive throughout, CI often crosses 0) |
| ICP segment, pillar, day/time, paid boost, experiment period | ≈ 0 | No detectable effect at this sample size (OLS CIs, stability, importance agree) |

Coefficients are associations, not causal effects.

## Stage 1 classification and evaluation results (`classification` branch)

**Question:** once reach is accounted for, which content features separate
high-engaging posts from low? **Write-up:**
[`notebooks/03_classification_evaluation.ipynb`](notebooks/03_classification_evaluation.ipynb)
(reads the committed runs; does not refit). Label: rate above the median
reach-expected rate (spline in log2 impressions, per platform). Full log in
`FINDINGS.md`.

| Finding | Evidence | Robust? |
|---|---|---|
| **Format matters** on LinkedIn and pooled | Joint Wald test survives BH (LinkedIn p = 0.037, pooled 0.001); LinkedIn Format explains 36% of rate-vs-reach variance | **Yes**: every label and knot count |
| LinkedIn **Document** | lasso odds ratio ≈ 2.7; ≈ +14 pts over Image after reach, Hedges' g ≈ 2.2 | **Yes** as an effect size; single-feature p 0.03 → 0.51 after BH (13 posts) |
| **Link** posts | LinkedIn OR ≈ 0.23; pooled Link survives BH (0.035) | **Yes** |
| Instagram Carousel, Events & Partnerships, Operations / Financial Leader | OR ≈ 1.9 / 2.5 / 0.5–0.6; g ≈ ±0.5 | Suggestive: hold at 3–4 spline knots, not 5 |
| Pillar, ICP segment | no BH-significant ANOVA / Welch / Kruskal test, no Tukey pair, ω² ≤ 0.05 | No detectable difference |

Held-out AUC (best model, mean ± SD over 50 splits): FB 0.64 ± 0.11, IG 0.67 ±
0.10, LI 0.66 ± 0.10, pooled 0.64 ± 0.05; differences under ~0.05 between
models are noise. Use robust or clustered SEs: naive SEs are up to 1.8× too small
(unequal variance, not repeated posts).

## Using the recommendation tool (Stage 2, `recommendations` branch)

**What it does:** reads the master workbook and reports, for each platform,
which choices (format, content pillar, ICP segment, timing, LinkedIn boost) have
done better or worse than usual, what changed in the last 90 days, and, if you
give it a content plan, how each planned post is likely to do. It runs on past
performance only; no LLM or internet access is needed. A run takes about 30
seconds.

**One-time setup:** Python 3.13, then `python -m pip install -r requirements.txt` from the repo root.

**After each data update:**

```bash
# 1. add the new export to the next master version (see "Adding new data" below)
# 2. point modeling/config.yaml at it:     master: data/Hidalga_Master_Data_v4.xlsx
# 3. run (from the repo root), with or without a plan:
python -m modeling.run_recommend --tag <name>
python -m modeling.run_recommend --tag <name> --plan my_plan.csv
```

Then open `outputs/runs/<date>_<name>/recommendations.html` in a browser (it
works offline and can be emailed), or `recommendations.xlsx` for every table.

**Planning posts:** copy `modeling/plan_template.csv`, one row per planned post:

| Column | Example | Notes |
|---|---|---|
| `post_id` | LI-1 | optional; your own label |
| `platform` | LinkedIn | Facebook, Instagram or LinkedIn |
| `date` | 2026-11-17 | YYYY-MM-DD (weekend is read from it) |
| `time` | 9:30 AM | or 14:00 |
| `format` | Document | must be a format used on that platform |
| `pillars` | Thought Leadership & Industry Insights | several separated by `;` |
| `icp_segments` | Oncology Operations Leader; Oncology Financial Leader | several separated by `;` |
| `impressions` | | optional; blank = typical for that platform and format |
| `boosted` | no | LinkedIn only; yes / no |

Rows with a problem (unknown format, misspelled pillar, bad date) are listed
with the reason; the rest are still scored.

**Reading the report:**

- Every post is compared with the **typical rate for its reach** on its
  platform, because small-audience posts naturally have higher engagement
  rates. **Lift** = rate − that typical rate.
- **Do more / Do less:** the choice clearly beats (or trails) its baseline,
  after correcting for the number of comparisons, and the result holds however
  reach is modelled. **Promising / Watch:** leaning one way, keep testing.
  **No clear difference:** choose on strategy. **Not enough posts yet:** fewer
  than 10.
- **What to test next** lists unsettled choices with roughly how many more
  posts would settle each.
- **Planned posts** get an expected rate and a chance to beat their reach,
  with ranges. Each platform's scores are marked usable or rough
  (Facebook's are rough: its engagement is hard to predict).
- These are associations in past posts, not proof of cause.

**Current results (v3 master, run `2026-11-17_recommendations`):** LinkedIn
**Document: do more** (+14 pts vs. Image) and **Link: do less** (−3 pts);
LinkedIn Multi-Image and Instagram Carousel / Events & Partnerships are
promising; Instagram Video is one to watch. Facebook and LinkedIn posts beat
their reach far more often in the last 90 days (71% / 67% vs. ~40% before).
On LinkedIn that is mostly the Document posts. On Facebook it is part of a
quarter-to-quarter swing on a very small audience, not the Q2–Q3 experiment
period (see `FINDINGS.md`); the report carries a caution note for Facebook,
set in `recommend: platform_notes` in the config.

Settings (thresholds, 90-day window, how many refits) are in the `recommend:`
section of `modeling/config.yaml`. Code: `modeling/recommend.py` (verdicts,
plan scoring, recent check), `modeling/report.py` (Excel / HTML),
`modeling/run_recommend.py` (the command). Tests: `python -m pytest modeling/tests -q`.

## Quantitative evaluation framework (added 2026-11-02; done 2026-11-04)

**Status:** items 1–2 are in `run_regression` / `run_classification`; items 3–7 are
`modeling/evaluate.py`, run with `python -m modeling.run_evaluation --tag <name>`
(run `outputs/runs/2026-11-04_eval_framework`). Results and the reporting rule
they imply are in `FINDINGS.md`. The table below is the original plan. Candidate
new features (day of week, posting hour, segment count) are not yet added.

**Goal:** every methodological claim gets a defined statistical test and an effect
size, not only stability or importance checks (e.g., section 5 of notebook 02
reports refit shares and permutation importance with no test or threshold).
Statistical significance at α = 0.05; practical significance through effect
sizes and confidence intervals.

| # | Test | Already in place | To do |
|---|---|---|---|
| 1 | **Regression:** coefficient t-tests, 95% CIs, standardized β; R², adjusted R², RMSE, MAE | OLS coefficients with HC3 CIs and p-values; R² / adjusted R² (`fit_stats.csv`); CV MAE and RMSE | Add t-statistics and standardized β to the OLS table; report as "a one-unit increase in X is associated with β pts (p = …)" |
| 2 | **High vs. low classification:** median-split label; logistic (Wald χ², odds ratios); random forest benchmark; accuracy, precision, recall, F1, ROC-AUC | `above_median` label; logistic odds ratios with CIs and p-values (C2); CV AUC, balanced accuracy, Brier, log loss | Wald χ² per feature and jointly per feature group; accuracy / precision / recall / F1 in the CV scores; random forest benchmark (+ permutation importance); report `above_median` alongside the main label (`reach_spline` since C5; `reach_adjusted` before) |
| 3 | **Which drivers matter most:** ranked table (feature, effect size, p-value) | Pieces exist separately (OLS, regularized, importance) | One ranked table per platform: standardized β (regression), odds ratio (logistic), permutation importance (tree-based benchmark) |
| 4 | **Do segments / pillars differ?** One-way ANOVA (F, p), Tukey HSD post-hoc | Not started | Per platform, for content pillar and ICP segment; check with Welch ANOVA or Kruskal–Wallis, since rates are skewed with unequal variances. Multi-segment posts handled as in row 5 |
| 5 | **Multi-segment dependence:** naive vs. clustered SEs | Main models avoid the problem (one row per post, 0/1 segment columns); pooled model clusters by content piece | Build the exploded data (one row per post × ICP segment); compare naive OLS SEs with post-clustered SEs (table: model, coefficient, SE, ratio). Same comparison for cross-posted content in the pooled model |
| 6 | **Robustness:** k-fold CV, mean ± SD | Done: 5-fold × 10 repeats on shared splits; 5 sensitivity variants | Report as "AUC = x ± y across folds"; optional 10-fold check (only ~12–14 posts per test fold per platform) |
| 7 | **Practical significance:** Cohen's d, standardized β, odds ratios, partial R² | Odds ratios; effects in percentage points | Cohen's d for 0/1 features (e.g., Document vs. Image); partial R² per feature and feature group |
| – | **Write-up** | – | A "Results evaluation" methods paragraph in the notebooks / report |

**Candidate new features** (from row 1):

- Day of week (from `DATE`; currently only a weekend flag) and posting hour
  (from `POST_TIME`; currently only an after-11am flag). Both are complete for
  all 391 posts.
- Number of ICP segments per post (`ICP_SEGMENTS`, `;`-separated, 1–3 per post).
- Post length and call-to-action presence: **blocked**. `POST_BODY` in the
  master is truncated at 200 characters (median 199), so length can't be
  measured and a CTA check would only see the opening. Needs full post text
  from Loomly / native exports, or these are dropped.
- More features means more tests at ~125 posts per platform: report new
  features as exploratory or adjust p-values (Holm or Benjamini–Hochberg).

## Data pipeline: details

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
- **`apply_master_corrections.py`** — applies reviewed one-off corrections
  (pillar fixes, experiment-period flag) to a master, writes the next version,
  logs every changed cell to `Corrections (log)`, regenerates the QA sheets,
  and verifies nothing else changed. Same never-overwrite rules as the merge.
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
# 2. append into the next master version (always pass the current master)
python3 merge_into_master.py --master Hidalga_Master_Data_v3.xlsx \
    --incoming /tmp/cleaned_YYYYMM.xlsx --pull-date YYYY-MM-DD
# 3. if new posts need tag fixes, add them to apply_master_corrections.py and run
python3 apply_master_corrections.py --master Hidalga_Master_Data_v4.xlsx
```

Export a fresh workbook from Loomly for each update. If you append new posts
to last month's workbook instead, its older posts carry stale metrics, which
the merge will only log, never apply.

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
| v2 | `data/092026-Social-Media-Data.xlsx` (cumulative Jan 1–Sep 29, 313 posts) | 2026-10-25 | 17 (Sep 11–29) | 296 already in v1 kept as-is; +3 follower-count months, +6 Loomly links; 1,665 differing values in `Unapplied Updates (log)` (see note below) |
| v3 | `apply_master_corrections.py` on v2 | — | 0 | 260 cells corrected, see `Corrections (log)`: 40 blank `CONTENT_PILLAR` filled from `LABELS_RAW`; "Cut Denial Rework Before It Hits the Queue" (3 rows) → Thought Leadership & Industry Insights; `IN_EXPERIMENT_PERIOD` = True for Apr 1–Sep 30 2026 (217 posts); QA sheets regenerated |

**About v2's `Unapplied Updates (log)`.** The 092026 workbook is the August
Social-Media-Data workbook with September's posts added; it isn't a fresh
pull of the older posts. For every overlapping Facebook/Instagram post and
Loomly link, its numbers are identical to the August export, and lower than
v1's, which come from the Sep 25 manual Loomly pull. So the logged "incoming"
values are older, not newer. Following the rule of keeping the most recent
pull, the master's values were kept and none were applied. LinkedIn differences are Loomly vs.
LinkedIn-native numbers and are never applied (plan §1b). **To refresh
metrics in future, export a new workbook from Loomly rather than appending
to the previous month's file.** Otherwise the log only holds stale numbers.

**Experiment period caveat.** `IN_EXPERIMENT_PERIOD` covers every post dated
Apr 1–Sep 30 2026, so it is the same as a "Q2–Q3 2026" time indicator. A
model can't tell the experiment's effect apart from a time trend over the same
months. Treat its coefficient as period effect, not causal experiment effect.

### Current outputs (in `data/`)

- **`Hidalga_Master_Data_v3.xlsx`** — **current master** (391 social posts,
  Sep 2025–Sep 2026, 100% ICP and pillar coverage). Same sheets as v1 plus
  `Unapplied Updates (log)` and `Corrections (log)`. Use this one for
  exploration and modeling. See "Merge history" above.
- **`Hidalga_Master_Data_v2.xlsx`** — v1 + September 2026 posts, before corrections.
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

### Notes for exploration and modeling (v3)

Readiness check on `Social Posts (unified)`: 1 null target (0 impressions),
every `POST_TIME` parses ("1:34 PM CDT" format, CDT/CST), no ICP or pillar
cell below 3 posts per platform. Things charts and models need to handle:

- **Very small posts make rates jumpy.** 27 posts have < 10 impressions
  (e.g. 3 impressions → 66.7%). Use impression-weighted means or a minimum-impressions
  filter, and show uncertainty.
- **Formats depend on the platform.** Carousel is Instagram-only;
  Document and Simple Status are LinkedIn-only. Compare formats within a
  platform, or group thin formats before pooling.
- **Instagram has no clicks**, so its unified rate is not fully comparable.
  Plot per platform or keep a platform term. Medians: FB 12.9%, IG 16.9%, LI 7.2%.
- **Small groups.** Patient-Centered Storytelling and Product Development
  (9–13 posts per platform), Financial Leader (22–25). Report confidence
  intervals.
- **Multi-label tags.** 119 posts have more than one ICP segment and 12 more
  than one pillar. Use 0/1 columns at the post level, or explode to one row
  per segment and cluster standard errors by post (proposal §5).
- **Experiment flag = time period** (see caveat above).
- **10 LinkedIn posts** (the newest) still use Loomly metrics rather than
  LinkedIn-native ones until a fresh LinkedIn content export is merged.

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

Current state (v3), per the regenerated `QA Label Coverage`: **100% ICP and
100% content-pillar coverage** across Facebook, Instagram, and LinkedIn. Every
value is one of the 4 canonical ICP segments / 5 canonical content pillars,
with no free text. (v1's QA sheet also said 100%, but it was never regenerated
after the manual fixes. 40 posts actually had their pillar in `LABELS_RAW`
with `CONTENT_PILLAR` blank; v3 fixes this.)

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
