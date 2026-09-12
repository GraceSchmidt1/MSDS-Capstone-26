# Hidalga Capstone — Data Consolidation Plan for Claude Code

> **Update after user review:** three of the four open questions from the
> first pass are now resolved, and one new file changes the source
> priority significantly. See §1a and §3 for what changed. Sections below
> are otherwise unchanged from the original inventory/verdicts, except
> where §1a supersedes them.

## 0. Context Claude Code needs

Goal: consolidate ~35 raw export files into one clean, deduplicated dataset
matching the schema in the capstone proposal (Section 4): a unified
`Social Posts` table (PLATFORM, DATE, POST_TIME, FORMAT, POST_TITLE,
POST_BODY, LABELS_RAW, ICP_SEGMENTS, CONTENT_PILLAR, LINK_TAGS, metric
columns, ENGAGEMENT_RATE_PCT), plus supporting tables (LinkedIn post detail,
LinkedIn daily metrics, visitor daily metrics, visitor demographics,
follower counts, Loomly clicks).

There is already a working script, `clean_hidalga_data.py`, that does this
correctly for **one month's worth of exactly 3 files** (a Social-Media-Data
workbook + a LinkedIn content export + a LinkedIn visitors export). Read it
first — it has the header-offset logic, multi-row-per-post parsing, and
label-parsing regexes already solved. **Do not rewrite that logic from
scratch; extend it.** The new work is: (a) figuring out which of the many
files is authoritative when several cover the same dates, and (b) adding
parsers for two new file shapes it doesn't handle yet (see Family F and
Family C below).

**Critical framing: most of these files are not separate data — they are
repeated exports of the same growing dataset, pulled at different times.**
Naively concatenating them will multiply post counts by 3-8x with massive
duplication. The core task is selecting the single best (most complete,
most recent) source per record, not merging everything.

## 1a. New primary source (supersedes Family A and Family B)

`manual-loomly-exporting.xlsx` was re-uploaded with real content (the first
version was empty). It has four sheets: `facebook`, `instagram`, `linkedin`
(each in the exact same layout as Family A/B — DATE/FORMAT/POST/LABELS +
metrics, multi-row-per-post), plus `Loomly_click`.

Its coverage is wider than Family A and Family B **combined**:

| sheet | range | count |
|---|---|---|
| facebook | 2025-10-01 → 2026-09-09 | 122 |
| instagram | 2025-09-23 → 2026-09-09 | 125 |
| linkedin | 2025-10-02 → 2026-09-09 | 131 |
| Loomly_click | 2025-09-22 → 2026-09-10 | 89 links |

Spot-check confirms it's the same underlying Loomly account as Family A,
just pulled later and more completely: the 2026-08-24 Facebook post appears
in both this file and `082026-Social-Media-Data.xlsx` with identical
FORMAT/POST/LABELS, but IMPRESSIONS is 14 here vs. 6 there — i.e. this is a
more mature, later snapshot of the same post, consistent with the
"prefer most recent" precedence rule already in the plan.

**Revised verdict: `manual-loomly-exporting.xlsx` becomes the primary
post-level source for Facebook, Instagram, and LinkedIn-via-Loomly content
and labels, replacing Family A and Family B entirely for that purpose.**

- Family A (`082026-Social-Media-Data.xlsx`) is still needed, but **only**
  for its `follower counts` sheet (monthly per-channel — nothing else in
  the file set replaces this grain).
- Family B (`Facebook-data.xlsx`, `Instagram-data.xlsx`, `linkedin-data.xlsx`)
  is now fully superseded — drop from the pipeline entirely.
- This file's own `Loomly_click` sheet replaces Family A's
  `loomly-click-data` sheet as the primary Loomly-clicks source (same
  reasoning: wider range, later pull).
- Family F (wide Content Analytics workbook) is still worth keeping, but
  now only for two things it uniquely provides: the **Instagram Stories**
  sheet, and a cross-check on FB/IG/LI post text and labels in the
  2025-10 to 2026-03 overlap window. It's no longer needed for FORMAT
  backfill, since this master file already has FORMAT natively.

## 1b. Engagement-number source of truth (resolves prior open question #4)

Per your decision: **use LinkedIn's own native export (Family C,
`hidalga-technologies_content_1788892250539.xls`) as the engagement-number
ground truth for LinkedIn**, and **use Loomly's numbers (now
`manual-loomly-exporting.xlsx`) as ground truth for Facebook and
Instagram.**

Concretely, in the merge step (§2 step 4 below):
- FB/IG rows: ENGAGEMENT_RATE_PCT and all raw metric columns come from
  `manual-loomly-exporting.xlsx`, no LinkedIn-export equivalent exists for
  these platforms anyway.
- LinkedIn rows: ENGAGEMENT_RATE_PCT and metric columns come from Family C
  (`Engagement rate`, `Impressions`, `Clicks`, `Likes`, `Comments`,
  `Reposts`, joined by DATE, ideally confirmed by post-title match since
  Family C has one row per post with no ambiguity). The LinkedIn rows
  parsed from `manual-loomly-exporting.xlsx`'s `linkedin` sheet are still
  used for FORMAT, LABELS, and POST title/body (Family C's "All posts"
  sheet doesn't carry the ICP/pillar tags), but their Loomly-reported
  engagement numbers are **not** used as the metric of record — keep them
  in a `LOOMLY_ENGAGEMENT_RATE_PCT_RAW` column for reference/QA only, so
  the discrepancy between the two tools stays visible rather than silently
  dropped.

## 1c. Pre-2026 dashboard data — confirmed skip

Per your decision, Family H (`SocialMediaDashboards.xlsx`) and Family A'
(`20251028-Social-Media-Data.xlsx`) are excluded from the pipeline
entirely. No further action needed on these two files — leave them out of
the `raw/` staging folder Claude Code builds in §2 below.

## 1. Source inventory, by family, with a verdict for each

> **Note:** the verdicts for Family A and Family B below reflect the
> *original* inventory pass. §1a supersedes both — Family A survives only
> for its `follower counts` sheet, and Family B is dropped entirely. Read
> this section for the reasoning on cumulative/rolling exports (it still
> applies to Families C–F), but defer to §1a on what to actually feed into
> the pipeline for post content.

### Family A — Monthly cumulative social workbooks
`052026-Social-Media-Data.xlsx`, `062026-...`, `072026-...`, `082026-...`
(sheets: Facebook/Instagram/Linkedin/loomly-click-data/follower counts)

Each one is **not** a distinct month — each starts at 2026-01-01 and simply
extends further (052026 → through 5/18, 062026 → through 6/15, 072026 →
through 7/27, 082026 → through 8/24). They're the same growing export
re-downloaded over time.

**Verdict: use only `082026-Social-Media-Data.xlsx` (the latest/widest).**
Discard 052026/062026/072026 as strict subsets — but before discarding, run
a diff on the overlapping date range to confirm metrics weren't silently
revised between pulls (e.g., late-arriving engagement). Log any mismatches;
don't just assume they're identical.

`follower counts` sheet here is **monthly, per-channel** (POSTS, FOLLOWERS,
ENGAGEMENT, IMPRESSIONS by MONTH) — keep this, it's a different grain than
Family E below, not a duplicate.

### Family B — Per-platform "master" workbooks
`Facebook-data.xlsx`, `Instagram-data.xlsx`, `linkedin-data.xlsx`
(each has a raw sheet + a `data-cleaned` metrics-only sheet)

These cover **2025-12-02 to 2026-03-27** — i.e., they include Dec 2025,
which Family A does not (Family A starts Jan 1, 2026). **This is your only
source for Dec 2025 posts** — important for the backfill goal.

**Verdict: primary source for Dec 2025–Mar 27 2026.** For the overlap
window with Family A (Jan–Mar), compare LABELS coverage and prefer
whichever has more complete tagging per post; reconcile disagreements.

Their companion files `032826-Facebook-data_data-cleaned_.csv`,
`032826-Instagram-...csv`, `032826-linkedin-...csv` are the **same
`data-cleaned` sheet, exported to CSV** — inferior duplicates (CSV export
flattens the multi-row post layout and loses the datetime/time cell-type
distinction the parser relies on; also has stray `\xa0` bytes — read with
`encoding='latin1'` if you touch it at all). **Verdict: ignore these CSVs
unless the xlsx originals are ever unreadable.**

### Family C — LinkedIn native content exports ("Metrics" + "All posts")
`hidalga-012026-082026.xls`, `hidalga-technologies_content_062026.xls`,
`..._1776889133559.xls`, `..._1785352077908.xls`, `..._1786998188809.xls`,
`..._1788892250539.xls`

All six are rolling-window exports from LinkedIn's own analytics tool, with
heavy overlap. Their date ranges:

| file | All posts range | Metrics range |
|---|---|---|
| `_1788892250539.xls` | **2025-09-09 → 2026-09-04** | 2025-09-07 → 2026-09-06 |
| `hidalga-012026-082026.xls` | 2026-01-01 → 2026-08-21 | 2026-01-01 → 2026-08-23 |
| `_1786998188809.xls` | 2026-01-01 → 2026-08-14 | 2026-01-01 → 2026-08-15 |
| `_content_062026.xls` | 2026-03-18 → 2026-06-12 | 2026-03-17 → 2026-06-14 |
| `_1785352077908.xls` | 2026-04-30 → 2026-07-27 | 2026-04-29 → 2026-07-27 |
| `_1776889133559.xls` | 2026-03-23 → 2026-04-17 | 2026-03-22 → 2026-04-20 |

**Verdict: `_1788892250539.xls` is the widest and most recent — use it as
the sole Family C source.** It fully contains every other file's window.
Discard the rest (but spot-check a few overlapping rows for revised metrics
before discarding). This feeds the `LinkedIn Posts (detail)` and
`LinkedIn Daily Metrics` sheets, with LinkedIn's *native* metrics (Views,
Offsite Views, Follows, CTR, Content Type) — these supplement, they don't
replace, the Loomly-reported LinkedIn numbers in Family A/F, since the two
tools attribute engagement differently. Keep both, joined by DATE (+ post
title match where possible), don't overwrite one with the other.

### Family D — LinkedIn visitor exports ("Visitor metrics" + demographics)
Nine files total. Same rolling-window pattern as Family C:

- `hidalga-technologies_visitors_1788892288178.xls`: **2025-09-07 →
  2026-09-06 (365 rows)** — widest, use this one.
- `_1776889282185.xls` and `_1776889282185_copy.xls` are **byte-identical
  duplicates** (same 30-row range) — delete the copy.
- `_1779310462161.xls` and `052026-hidalga-technologies_visitors.xls` cover
  the identical range (2026-01-01→05-18, 138 rows) — near-certainly
  duplicates too; diff them to confirm, then keep one.
- `hidalga-012026082026.xls`, `_062026.xls`, `_1785352089815.xls`,
  `_1786998157043.xls` are all subsets of the widest file's range.

**Verdict: use only `_1788892288178.xls`.** Discard the other eight after a
spot-check diff (don't just trust filenames — the "062026" file is missing
the "Job function" sheet that the others have, so check sheet-completeness
too, not just row counts).

### Family E — LinkedIn new-followers export
`hidalga-technologies_followers_1788892297560.xls` (sheet: "New followers",
daily, 2025-09-07 onward)

Single file, no duplicates. **Verdict: use directly.** This is daily-grain
and LinkedIn-specific — complementary to Family A's monthly per-channel
follower counts, not a duplicate. Add as a new `LinkedIn New Followers
(daily)` sheet (the current script has no slot for this — it's new).

### Family F — Loomly "Content Analytics" reports (a different export type)
`Hidalga_Content_Analytics_2025-10-01_-_2026-09-10.xlsx` (wide) plus three
narrow monthly ones for Jan/Feb/Mar 2026.

This is a **structurally different, cleaner format** than Family A: one row
per post (no multi-row date/time/title/body split), DATE is a single
datetime with time included, POST is the full text with `\n` separating
title/body, LABELS is a clean comma-joined string. It also includes an
**Instagram Stories sheet** — a content type not present anywhere else —
and `Loom.ly - Links` / `Loom.ly - Account` rollups.

The wide file spans **2025-10-01 to 2026-08-31** (post dates), which is
wider than every other post-level source except Family C/D/E. The three
monthly files (Jan/Feb/Mar 2026) are narrow subsets of it — use them only
to cross-check, not as inputs.

**Two things to watch:**
1. **No FORMAT column** in these Posts sheets (unlike Family A/B). You'll
   need to backfill FORMAT for these rows by joining on (PLATFORM, DATE) to
   Family A or Family B, and leave it null where no match exists.
2. **This is your best backfill source pre-2026** (Oct–Dec 2025) alongside
   Family B — cross-check the two where their ranges overlap (Dec 2025) for
   consistency before trusting either exclusively.

Also note: its dates run through **Aug 31, 2026**, and Family C's LinkedIn
export runs through **Sep 4, 2026** — both past the proposal's stated
Aug 17/21 cutoff. See open question #1 below before deciding whether to
include these in the training window.

**Verdict: primary post-level source alongside Family A/B; use for FORMAT
backfill target, Instagram Stories, and Loomly link/account rollups.**

### Family G — `Marketing-Analytics-April-2026.xlsx`
Single-month (April 2026) extract in the Family B raw multi-row layout.
Fully covered by Family A (082026) and Family F for that month.

**Verdict: cross-check only, do not merge as an input.**

### Family H — `SocialMediaDashboards.xlsx` / Family A' — `20251028-Social-Media-Data.xlsx`
These two are **the same underlying weekly account-level rollup** (Views,
Followers, Reach, Interactions — not per-post data), dated from March 2025,
predating the current ICP/pillar taxonomy entirely. `SocialMediaDashboards`
even has a sheet with a manual note about needing a pivot-table refresh —
this is a hand-maintained dashboard artifact, not a clean export.

**Verdict: exclude from the core post-level pipeline.** These aren't
posts, they're weekly account trend snapshots from a pre-taxonomy period,
in a messy pivot-table format. Flag to the user: worth the cleanup effort
only if Grace wants pre-2026 account-level trend context; not needed for
the Stage 1 engagement model.

### Family I — `Hidalga_Technologies__Inc_competitors.xlsx`
Competitor follower/post/engagement snapshot at two dates (9/7/2025,
9/6/2026). **Verdict: keep as its own small reference table** for the
Stage 2 market-research agent — do not merge into Social Posts.

### Family J — `manual-loomly-exporting.xlsx`
**Completely empty (0 rows, 0 columns).** Flag to the user — either it
needs to be re-exported, or it can be deleted from the source folder.

### Family K — `Q1_ICP_Segments_and_Content_Pillar_Reference_Guide.pdf`
Not data — this is the **authoritative taxonomy definition**. Use it to
validate the label-parsing lookup tables in `clean_hidalga_data.py`
(`ICP_SEGMENTS`, `PILLAR_MAP`). Cross-checked already: the script's mapping
of raw label variants (`"Team & Culture"` → `"Team Spotlights & Company
Culture"`, `"Product Dev. & Innovations"` → `"Product Development &
Innovations"`, `"Patient-Centered Stories"` → `"Patient-Centered
Storytelling"`) is consistent with the PDF's canonical pillar names — keep
it, but re-verify once Family F's raw LABELS strings are parsed, since that
source may contain label spelling variants not seen before (e.g., its
sample rows show `"Oncology Clinic Leader , Oncology Operations Leader ,
Th..."` — note the stray space before commas; confirm the parser's
substring matching tolerates that).

### Family Z — Prior pipeline outputs (not raw inputs!)
`Hidalga_Cleaned_Data_2026-08.xlsx` and `Hidalga_Cleaned_Data_old08172026_.xlsx`

These are **previous runs of `clean_hidalga_data.py`**, not source data.
**Do not feed them into the new pipeline as inputs** — that would
double-count everything they contain. Use them only as a regression check:
the new pipeline's output should reproduce `Hidalga_Cleaned_Data_2026-08`'s
Social Posts numbers (284 rows: 94 FB + 92 IG + 98 LI, Jan 1–Aug 24 2026)
exactly for that overlapping window, before it's trusted to extend further
back and forward in time.

## 2. Pipeline steps for Claude Code

1. **Set up a `raw/` staging folder**, copy in only the "winning" file from
   each family per the verdicts above (082026 workbook, the three Family B
   workbooks, `_1788892250539.xls` [content], `_1788892288178.xls`
   [visitors], the followers file, the wide Content Analytics file,
   competitors file). Everything else moves to a `superseded/` folder with
   a one-line note on why, not deleted outright — the user may want to
   audit the dedup decisions.

2. **Before discarding any "superseded" file**, run a quick diff pass: for
   each date that appears in both a winning file and a superseded one,
   compare the metric values. Write a `dedup_discrepancies.csv` log of any
   row where they disagree (this catches late-arriving engagement or
   silent corrections — don't silently assume subset files are byte-exact
   copies just because the date range nests).

3. **Extend `clean_hidalga_data.py`** rather than rewriting it:
   - Its existing `build_social_posts` logic already handles Family A's
     multi-row layout — reuse for Family A and Family B (same layout).
   - Write a new parser for Family F's flat one-row-per-post layout
     (much simpler: no offset/multi-row logic needed, just clean text and
     split LABELS on commas).
   - After parsing Family F, backfill its missing FORMAT column via a
     (PLATFORM, DATE) join against Family A/B; leave null on no match.
   - Add the Instagram Stories sheet from Family F as a new output sheet
     (`Instagram Stories`) — it's a distinct content type, don't force it
     into the Social Posts schema.
   - Add a `LinkedIn New Followers (daily)` output sheet from Family E.

4. **Merge Family A + Family B + Family F into one Social Posts table**,
   matched on **(PLATFORM, DATE)** as the primary dedup key (add POST_TITLE
   similarity as a tie-breaker for days with multiple posts on the same
   platform). Precedence when the same (platform, date) appears in more
   than one winning source: prefer the source with more complete
   LABELS/ICP tagging; if tied, prefer Family F (cleaner parse); log every
   resolved conflict to `merge_decisions.csv` so it's auditable.

5. **Re-run label parsing** (`parse_labels`) against the merged table,
   validate every resulting ICP segment and pillar name against the PDF's
   canonical list (Family K), and report a coverage table (% tagged by
   platform) matching the format already in the proposal (Section 4) so
   the numbers are directly comparable.

6. **No cutoff — keep everything** (§1d). Instead of splitting off recent
   data, add the `IN_EXPERIMENT_PERIOD` covariate column described in §1d
   and populate `SOURCE_PULL_DATE` on every row so any future distribution
   shift stays traceable rather than needing to guess.

7. **Consolidate Family C (LinkedIn detail) and Family D (visitors)** using
   their single winning wide files directly — no merge conflicts expected
   since each is already the superset, but still run the join to Social
   Posts by (DATE) for LinkedIn detail and flag any post-date in Social
   Posts with no matching detail row (and vice versa) so gaps are visible
   rather than silently dropped.

8. **Keep Family I (competitors) and Family K (taxonomy) as separate
   reference tables**, not merged into Social Posts.

9. **Final QA pass before writing output:**
   - Row-count reconciliation table: expected vs. actual per platform,
     with a note on why the number differs from the proposal's 272/94/90/88
     (it will, since more data has been backfilled and the live feed has
     continued — that's expected, not an error).
   - Full duplicate scan on the final Social Posts table by (PLATFORM,
     DATE, POST_TITLE) — zero duplicates should remain.
   - Confirm ENGAGEMENT_RATE_PCT is populated consistently across all
     platforms using one formula (per the proposal's stated definition),
     not a mix of native vs. derived rates.
   - Report thin cells (ICP segment × platform combinations with fewer
     than 3 posts) per the proposal's Section 7 mitigation plan.

10. **Output**: `Hidalga_Master_Data_v1.xlsx` (see §1e for why it's
    versioned rather than one-off), same sheet structure as the existing
    script's output plus the three new sheets (Instagram Stories, LinkedIn
    New Followers daily, Competitors), plus the `merge_decisions.csv` and
    `dedup_discrepancies.csv` audit logs alongside it so every judgment
    call is traceable back to source files. This first version becomes the
    base that future exports get upserted into per §1e, rather than a
    finished, one-time deliverable.

## 3. Open questions

Resolved: pre-2026 dashboard data → skip (§1c). Family J → re-exported,
now in use (§1a). Engagement source of truth → LinkedIn native for
LinkedIn, Loomly (now the manual export) for FB/IG (§1b). Cutoff date →
**no cutoff — keep everything, including September 2026** (§1d).

All open questions are now resolved. See §1d and §1e below for how this
changes the pipeline design.

## 1d. No training-window cutoff — keep everything, including September

Per your decision: don't hold out recent posts. Every post in every
winning source, through whatever the newest export's max date is, goes
into the final Social Posts table — no truncation.

This changes how Section 7's "confounding from in-flight experiments"
concern gets handled, since exclusion is off the table:

- Add an `IN_EXPERIMENT_PERIOD` boolean covariate column (date range TBD by
  Grace/Joshua — whenever the coming-quarter experiments actually run) so
  the modeling stage can control for it statistically instead of the data
  pipeline pre-emptively cutting it. This is a modeling decision to make
  later, not a cleaning decision — the pipeline's job is just to make sure
  the column exists and is populated, not to decide how it's used.
- Keep `SOURCE_FILE` and a new `SOURCE_PULL_DATE` column (see §1e) on every
  row regardless of date, so if a distribution shift does show up later,
  it's traceable back to which export batch introduced which rows.
- The QA report (§2 step 9) should still surface a simple month-by-month
  post-count and engagement-rate summary so any shift is visible at a
  glance, even though nothing is being excluded.

## 1e. Designing for progressive/incremental Loomly exports

You'll be pulling more Loomly data over time (next month, the month after,
etc.), and want each new export folded into the same cleaned dataset
rather than re-running a one-off script from scratch each time. This
changes the pipeline from "clean these N files once" to "maintain one
growing master dataset that new exports get merged into." Concretely:

1. **Split the pipeline into two scripts, not one:**
   - `clean_source.py` — takes a single raw export (any of the shapes in
     §1a/§1c: the Loomly manual-export layout, the Family C LinkedIn
     layout, the Family D visitor layout, the Family F content-analytics
     layout) and normalizes it into the *same* tidy per-row schema used in
     the final output, tagging every row with `SOURCE_FILE` and
     `SOURCE_PULL_DATE` (the date the export was generated — parse from
     filename timestamp/suffix where present, otherwise from file
     metadata or ask the user once per file).
   - `merge_into_master.py` — takes the current master consolidated
     workbook plus one or more freshly-cleaned exports from step 1, and
     upserts: for each incoming row, match on (PLATFORM, DATE, POST_TITLE)
     against the master; if no match, append; if a match exists, compare
     `SOURCE_PULL_DATE` and keep whichever is newer (later pull = more
     mature metrics, per the precedent in §1a), logging every overwrite to
     `merge_decisions.csv` with old value → new value so nothing changes
     silently.

2. **This means Claude Code should build the "most complete source per
   family" logic from §1a not as a one-time manual selection, but as a
   standing rule inside `merge_into_master.py`**: "newest `SOURCE_PULL_DATE`
   wins on conflict" is exactly the rule that made `manual-loomly-exporting.xlsx`
   beat `082026-Social-Media-Data.xlsx` in this pass — encode it once,
   and it keeps working automatically as new exports arrive next month,
   without needing another manual file-by-file audit like this one.

3. **First run:** treat this consolidation as building the master from the
   current best sources (§1a/§1c) — `manual-loomly-exporting.xlsx`,
   Family A's `follower counts` sheet, Family C, Family D, Family E, and
   Family F's Instagram Stories sheet. That produces
   `Hidalga_Master_Data.xlsx` (v1).

4. **Every future run:** just run `clean_source.py` on whatever new export
   arrived, then `merge_into_master.py` against the existing
   `Hidalga_Master_Data.xlsx` to produce the next version. No need to
   re-audit the whole file inventory each time — the upsert rule handles
   staleness automatically. Keep each version (`Hidalga_Master_Data_v1.xlsx`,
   `_v2.xlsx`, ...) rather than overwriting, so the modeling stage can
   always point at a specific frozen snapshot if reproducibility matters
   later.

5. **One thing this doesn't automate:** if a genuinely *new* export shape
   shows up (a tool export you haven't seen before, not just a later pull
   of a familiar one), `clean_source.py` will need a new parser branch —
   that's still a manual step, flagged clearly rather than silently
   dropped or mis-parsed as an existing shape.


