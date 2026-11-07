# Findings log

A running record of results and methodological findings, newest stage last, each with the run folder or
file it comes from, so they can be cited in notebooks and the report. Effects are associations, not
causal. "Stable" = kept in ≥80% of 500 lasso bootstrap refits with ≥95% sign agreement.

Odds ratios (OR) are the median over bootstrap refits of the lasso logistic unless marked "plain logit".
AUC is the held-out mean over 5-fold × 10 repeats (0.5 = chance).

## Data

- **391 posts** (FB 124, IG 130, LI 137), Sep 23 2025 – Sep 29 2026, master `data/Hidalga_Master_Data_v3.xlsx`.
  390 have an engagement rate (one post has 0 impressions).
- **`POST_BODY` is truncated at 200 characters** (median 199). Post length can't be measured and a CTA check
  would only see the opening; full text needed from Loomly / native exports. `POST_TIME` and `DATE` are
  complete for all posts (day of week and posting hour are available as features). Found 2026-11-02.
- Each post carries 1–3 ICP segments (`;`-separated in `ICP_SEGMENTS`).
- Two LinkedIn posts on 2026-03-20 share one native metrics row (kept as is; flagged in the EDA).

## Stage 1 regression: engagement rate (runs `2026-10-25_step2…step5`)

Source: README "Results", notebook 02.

| Finding | Effect | Robustness |
|---|---|---|
| Held-out R² (elastic net) | IG 0.44 (0.57 with log target); LI ≈ 0 (0.19 without one 3-impression outlier); FB ≈ mean; pooled 0.24 | — |
| LinkedIn Document vs. Image | ≈ +12 pts (+8 to +18 across variants) | 5/5 variants, 96% of refits. All 13 Document posts ran Jul–Sep 2026 |
| Instagram reach | ≈ −6 pts per doubling of impressions | 100% of refits; dominant feature |
| Video | no difference within platform once reach is controlled | EDA's "video worst" was a reach effect |
| IG Carousel, LI Multi-Image | +2 to +5 pts | suggestive only (CI often crosses 0) |
| ICP, pillar, day/time, boost, experiment period | ≈ 0 | no detectable effect at this n |

## Stage 1 classification: high vs. low engagement

### Main label: reach-adjusted (run `2026-11-02_c3_penalized`, commits `c56c718`, `d09004f`)

"High" = rate above what reach alone predicts (median residual of a median-regression line of rate on
log2 impressions, fit per platform on training posts only).

Held-out AUC:

| Platform | Base rate | Plain logit | Tree | Ridge | Lasso | Elastic net |
|---|---|---|---|---|---|---|
| Facebook | 0.50 | 0.51 | 0.56 | 0.55 | 0.50 | 0.50 |
| Instagram | 0.50 | 0.54 | 0.62 | 0.57 | 0.51 | 0.50 |
| LinkedIn | 0.50 | 0.65 | 0.58 | **0.66** | 0.62 | 0.63 |
| Pooled | 0.50 | 0.57 | 0.58 | 0.57 | 0.54 | 0.55 |

- **Only LinkedIn has real signal** once reach is removed. Penalized ridge is best (AUC 0.66, Brier 0.24).
- **Facebook and Instagram: lasso / elastic net drop every feature** (predict the base rate). Ridge and
  elastic net chose the strongest penalty tried (C at the grid edge), i.e. they prefer the null model.
- **Stable LinkedIn effects:** Document OR ≈ 3.6, Team Spotlights pillar ≈ 2.3, Link ≈ 0.32,
  Simple Status ≈ 0.41. Pooled: Document ≈ 2.5, Link ≈ 0.41.
- **Plain logit, 95% CI excluding 1:** LI Team Spotlights OR 4.3 (p = 0.016), LI Link 0.18 (p = 0.012),
  LI Document 6.8 (p = 0.029); pooled Link 0.23 (p = 0.004), pooled Document 5.6 (p = 0.04).
- **Permutation importance (elastic net, AUC drop):** Format is the only group that matters
  (LI 0.11, pooled 0.06); everything else ≈ 0.

### Label-definition sensitivity (C4; runs `2026-11-02_c4_label_above_median`, `…_top_tercile`)

Best-model AUC by label:

| Platform | Reach-adjusted | Above median | Top third |
|---|---|---|---|
| Facebook | 0.56 | 0.62 | 0.63 |
| Instagram | 0.62 | 0.77 | 0.79 |
| LinkedIn | 0.66 | 0.66 | 0.64 |
| Pooled | 0.58 | 0.62 | 0.67 |

- **Instagram's higher AUC under raw-rate labels is almost all reach:** shuffling reach costs 0.25–0.28 AUC
  (more than all other groups combined). Low-reach posts have higher rates; consistent with the regression.
  Under raw-rate labels only, IG Carousel (OR ≈ 2.0–2.4) and Events & Partnerships (≈ 2.3–2.7) are stable.
- **LinkedIn is label-independent:** AUC 0.64–0.66 under all three; Format is the dominant group each time
  (AUC drop 0.11–0.16).
- **Robust across all three labels:** LinkedIn / pooled **Document** (OR 3.1–4.1 LI; 2.5–3.5 pooled) and
  pooled **Link** (OR 0.24–0.41; LI Link stable under 2 of 3). Simple Status (≈ 0.4) and Team Spotlights
  (≈ 2) stable under 2 of 3.
- **Facebook is weak and unstable under every label.**
- **Artifact:** under top-third, Facebook's plain logit did not converge: all 3 weekend posts are "low"
  (perfect separation), giving a weekend OR ≈ 0 with a meaningless CI. Don't report that row; penalized
  models are unaffected.

## Evaluation framework, items 1–3 (runs `2026-11-03_eval_regression`, `…_eval_classification`, `…_eval_label_above_median`, `…_eval_label_top_tercile`; commits `7622674`, `c4bcde2`)

Reach-adjusted label unless noted. Outputs were checked against the earlier committed runs: every existing
number is unchanged; only new columns, models and files were added. The official run CSVs are
byte-identical to the 2026-11-02 trial runs, and the linear models under the two alternative labels
match the C4 runs exactly.

**Regression: standardized β (SDs of rate per SD of feature) and robust t:**

| Platform | Largest effects (std β, t, p) |
|---|---|
| Instagram | reach −0.77 (t −4.6, p < 0.001); Carousel +0.17 (t 1.9, p 0.058) |
| LinkedIn | Document +0.41 (t 2.2, p 0.030); Multi-Image +0.15 (t 1.7, p 0.088) |
| Facebook | Link −0.27 (t −2.0, p 0.041; FB is not predictable, so treat as noise until multiple testing is handled) |
| Pooled | reach −0.46 (t −3.2, p 0.001); Platform: Instagram +0.24 (p < 0.001); Document +0.17 (p 0.050) |

**Logistic: joint Wald χ² per feature group:** **Format is the only group that matters anywhere:**
LinkedIn χ² = 16.8 (df 5, p = 0.005), pooled χ² = 19.6 (df 6, p = 0.003). Every other group on every
platform has p > 0.14. Reach p ≈ 0.7 everywhere (expected: the label removes linear reach).

**Random forest benchmark (300 trees, untuned): the best model everywhere except LinkedIn:**

| Platform | Forest AUC | Best linear / tree AUC | Forest F1 |
|---|---|---|---|
| Facebook | **0.66** | 0.56 (tree) | 0.60 |
| Instagram | **0.68** | 0.62 (tree) | 0.54 |
| LinkedIn | 0.65 | **0.66** (ridge) | 0.58 |
| Pooled | **0.65** | 0.58 (tree) | 0.57 |

- **The forest's top group on FB, IG and pooled is reach** (AUC drop 0.12 / 0.09 / 0.07), even though the
  label is reach-adjusted. The adjustment is a *straight line* in log2 impressions, so any curved reach
  pattern stays in the label, and only the forest can use it. **The reach-adjusted label is not fully
  reach-free.** Open question: adjust reach non-linearly (e.g. a spline) and re-check.
- Forest also uses ICP segment on Facebook (AUC drop 0.08) and pillar on Instagram (0.05), which no linear
  model detects; possible interactions, unconfirmed.
- On LinkedIn the forest adds nothing over ridge: the signal there is additive (format).
- **Precision / recall / F1 at a 0.5 cut** are near 0 for lasso / elastic net on FB and IG because they
  predict the base rate (just under 0.5) for every post, so nothing is called "high". AUC and balanced
  accuracy are the fairer comparison there.

**Random forest under all three labels** (forest AUC; best other model in brackets):

| Platform | Reach-adjusted | Above median | Top third |
|---|---|---|---|
| Facebook | **0.66** (0.56 tree) | **0.70** (0.62 tree) | **0.67** (0.63 logit) |
| Instagram | **0.68** (0.62 tree) | **0.79** (0.77 logit) | **0.79** (0.79 logit) |
| LinkedIn | 0.65 (**0.66** ridge) | 0.63 (**0.66** ridge) | 0.63 (**0.64** ridge) |
| Pooled | **0.65** (0.58 tree) | **0.71** (0.62 tree) | **0.72** (0.67 logit) |

Forest permutation importance, top group (AUC drop, reach-adjusted / above median / top third):

- **Facebook: reach** 0.12 / 0.16 / 0.16 (2.5–97.5% interval over splits above 0 only under above-median; FB has
  124 posts, so per-split intervals are wide). **Facebook is predictable only by the forest, and mainly
  through reach,** under every label. Linear models with the same reach feature get ≈ 0.50–0.63, so the
  reach–rate relationship on Facebook is likely curved (non-linear), not absent.
- **Instagram: reach** 0.09 / 0.21 / 0.25 (interval above 0 under both raw labels). Under raw-rate labels the forest barely beats logistic
  (the reach effect there is mostly linear); under the reach-adjusted label the leftover curve gives it
  +0.06.
- **LinkedIn: format** 0.09 / 0.09 / 0.09; reach ≈ 0 under every label. Forest never beats ridge, so the
  LinkedIn signal is additive (format) and label-independent.
- **Pooled: reach** 0.07 / 0.14 / 0.15, then **format** (0.05–0.06); both intervals above 0 under all
  three labels.
- Strengthens the open question above: a non-linear (spline) reach adjustment is the next test of
  whether anything besides reach drives Facebook / Instagram.
- **Top-third F1 is low** (forest FB 0.09, IG 0.36) because only a third of posts are "high" and the
  cut is 0.5; compare on AUC / balanced accuracy.

## C5: spline reach-adjusted label (run `2026-11-03_eval_label_reach_spline`, commits `7d1069d` code, `593b567`, `f0a8b42`)

`reach_spline` = as reach-adjusted, but the expected (median) rate is a cubic spline in log2 impressions
(4 knots at training-reach quantiles, linear beyond the outer knots), fit per platform on training posts
only. Made the main label on 2026-11-03. Knot sensitivity (3 and 5 knots) from trial runs in the
scratchpad, not committed.

**Reach is removed much better than by the straight line.** Forest reach importance (AUC drop),
line → spline 3 / 4 / 5 knots:

| Platform | Line | 3 knots | 4 knots | 5 knots |
|---|---|---|---|---|
| Facebook | 0.12 | 0.08 | 0.06 | 0.03 |
| Instagram | 0.09 | 0.01 | 0.01 | 0.01 |
| LinkedIn | ≈ 0 | ≈ 0 | ≈ 0 | 0.02 |
| Pooled | 0.07 | 0.03 | 0.03 | 0.02 |

Held-out AUC, line → 4 knots: forest about unchanged (FB 0.66 → 0.64, IG 0.68 → 0.67, LI 0.65 → 0.65,
pooled 0.65 → 0.64), but it now leans on content (IG format 0.06, ICP 0.06; FB ICP 0.05, intervals cross
0). **Linear models gain once reach is out:** IG ridge 0.57 → 0.66, plain logit 0.54 → 0.65, elastic
net 0.50 → 0.59; FB ridge 0.55 → 0.60; pooled ridge 0.57 → 0.60. LinkedIn unchanged (ridge 0.66).

Plain-logit joint Wald tests: Format LI χ² = 16.3 (df 5, p = 0.006), pooled 27.3 (df 6, p < 0.001; was
19.6 under the line); IG Format p = 0.06, IG ICP p = 0.10; everything else p > 0.12.

Stable lasso effects (odds ratios) and whether they hold across knots:

| Effect | 3 knots | 4 knots | 5 knots | Verdict |
|---|---|---|---|---|
| LI Document | 2.8 | 2.7 | 2.9 | robust (also every earlier label) |
| LI Link | 0.22 | 0.23 | 0.22 | robust |
| LI Team Spotlights | 1.9 | 1.9 | 2.2 | robust |
| Pooled Link / Document | 0.35 / 1.9 | 0.36 / 1.8 | 0.28 / 2.5 | robust |
| Pooled Carousel | 1.9 | 1.6 | 1.7 | new; robust across knots |
| IG Carousel | 2.2 | 1.9 | – | holds at 3–4 knots only |
| IG Events & Partnerships | 2.5 | 2.5 | – | holds at 3–4 knots only |
| IG Operations / Financial Leader | 0.49 / 0.57 | 0.47 / 0.61 | – | holds at 3–4 knots only |

- **Instagram content effects are knot-sensitive:** at 5 knots the IG lasso keeps no feature in even
  half the refits (IG ridge AUC falls to 0.60). A more flexible reach curve may absorb content that is
  correlated with reach, or 5 knots overfits ~100 training posts. Report the IG effects as
  *suggestive*: they match the raw-rate labels (C4: Carousel ≈ 2.0–2.4, Events ≈ 2.3–2.7) and 3 and
  4 knots, but not 5.
- Facebook stays weak: no effect is stable at 4 or 5 knots.

## Evaluation framework, items 3–7 (run `2026-11-04_eval_framework`, commits `00c9260` code, `50d578b`, `493168d`)

Two outcomes: **rate** and **rate vs. reach** (rate minus the spline-expected rate for the post's reach).
Items 3 and 6 read the committed runs `2026-11-03_eval_regression` and `2026-11-03_eval_label_reach_spline`.
BH = Benjamini–Hochberg false-discovery-rate adjustment; Holm = family-wise.

**Item 4: pillars and ICP segments do not differ detectably on any platform.**
- Single-tag posts (87–133 per test), ANOVA / Welch / Kruskal–Wallis: no test has BH-adjusted p < 0.05
  (smallest 0.18). Uncorrected, only Facebook pillar on rate vs. reach approaches it (ANOVA p = 0.059,
  Kruskal p = 0.051; post-clustered test on all posts p = 0.037, BH 0.23).
- Effect sizes are small: ω² ≤ 0.05 everywhere (pillar or ICP explains ≤ 5% of rate variance).
- **Tukey HSD: no pair of pillars or segments differs** on any platform or outcome.

**Item 5: duplicated rows do not distort the SEs; unequal variance does.**
- Post × tag rows (multi-segment posts repeated): post-clustered SE / naive SE = 0.64–1.39 (median
  ≈ 0.8–1.0); clustered / HC3 = 0.76–1.00. Repeating a post across its segments does not inflate
  uncertainty here. Two uncorrected flips (FB Patient-Centered Storytelling, LI Technical Leader become
  p < 0.05 when clustered), neither survives multiple testing.
- Pooled model (cross-posted content): clustered / naive up to **1.8× for reach and Document**, 1.6× for
  Platform: LinkedIn, 1.25× for Link; pooled Document p goes 0.001 (naive) → 0.050 (clustered). But
  clustered / HC3 is 0.69–1.06: the inflation is heteroscedasticity (e.g. Document posts' rates vary
  widely), not cross-posting. **Naive (non-robust) SEs should not be used; robust or clustered SEs are.**
  Simple Status goes the other way (0.53×, p 0.11 → 0.003) with only 6 posts: don't trust that p.

**Item 7: effect sizes** (Hedges' g on rate vs. reach unless noted; formats vs. Image; 95% CI excluding 0):

| Platform | Feature | g (95% CI) | Size |
|---|---|---|---|
| LinkedIn | Document (13 posts) | **2.17 (1.50 to 2.84)**, +14 pts | very large |
| LinkedIn | Multi-Image (10) | 1.52 (0.83 to 2.22) | large |
| LinkedIn | Link (18) / Simple Status (6) | −1.11 / −1.10 | large |
| Instagram | Carousel (28) | 0.49 (0.06 to 0.92) | medium |
| Instagram | Events & Partnerships (23) | 0.48 (0.03 to 0.94) | medium |
| Instagram | Operations Leader (87) | −0.45 (−0.82 to −0.08) | medium |
| Instagram | Video, raw rate only | −1.79 on rate, n.s. on rate vs. reach | reach effect |
| Facebook | Product Development pillar (9) / Operations Leader (82) | 0.82 / 0.41 | CIs near 0 |
| Pooled | Document / Multi-Image / Link | 1.87 / 0.51 / −0.48 | |

CIs are unadjusted. Partial R² (share of remaining variance a group explains, in-sample OLS):
- **LinkedIn Format 0.19 of rate, 0.36 of rate vs. reach**; every other LinkedIn group ≤ 0.05.
- **Instagram reach 0.42 of raw rate**, ≈ 0 of rate vs. reach (the spline removes it); format, ICP and
  pillar 0.02–0.06.
- Facebook: every group ≤ 0.09. Pooled: Format 0.09 / 0.15, reach 0.08 of rate.
- Pillar and ICP never exceed 0.08 anywhere, consistent with item 4.

**Item 6: CV as mean ± SD across the 50 splits** (`cv_mean_sd.csv`):
- Regression R² (elastic net): IG 0.44 ± 0.15, pooled 0.24 ± 0.08, FB −0.05 ± 0.07, LI −0.01 ± 0.30.
  The LinkedIn SD shows how unstable that number is.
- Classification AUC, reach-spline label (ridge / forest): FB 0.60 ± 0.10 / 0.64 ± 0.11, IG 0.66 ± 0.11 /
  0.67 ± 0.10, LI 0.66 ± 0.10 / 0.65 ± 0.08, pooled 0.60 ± 0.05 / 0.64 ± 0.05. Per-platform SDs ≈ 0.1: the
  per-platform AUCs overlap, so differences of < 0.05 between models are not meaningful.

**Item 3: ranked drivers (`driver_ranking.csv`). Per-feature p-values mostly do not survive multiple
testing:**
- Within each platform (BH over its 13–17 features), only **Instagram reach** (regression) survives on a
  single platform. **LinkedIn Document: regression p 0.030 → BH 0.51; logistic p 0.061 → BH 0.35.**
- Pooled survivors: Platform: Instagram, reach, Video, Simple Status (regression, also Holm), and
  **Link** (logistic, BH and Holm 0.035).
- **Group-level tests do survive:** BH across feature groups on the joint Wald χ² (reach-spline
  label): LinkedIn Format p = 0.006 → 0.037, pooled Format < 0.001 → 0.001; no other group < 0.24.
- Features with ≥ 2 independent lines of support (BH-significant p and / or stable lasso in regression
  and logistic): LinkedIn Document; Instagram Carousel, Financial Leader, reach; pooled Link,
  Document, Carousel, Video, Simple Status, reach, Platform: Instagram.
- **Reporting rule that follows:** claim *format matters* (group test, survives correction, large effect
  sizes, stable across labels) rather than single-feature p-values. Present single features with effect
  sizes, CIs and stability, flagged as exploratory where their adjusted p ≥ 0.05.

## Stage 2: recommendation tool (run `2026-11-07_recommendations`, commits `853e7b7`, `00c2e40`, `47c1d5a`, `a65a9d4` code; `91f953b`, `eb92eff` run)

`python -m modeling.run_recommend` (`modeling/recommend.py`, `report.py`): verdicts per choice from past posts,
planned-post scores, a 90-day check, and Excel + HTML reports for the marketing team. No LLM. Lift = rate −
typical rate for the post's reach (the `reach_spline` rule fit on all of the platform's posts), i.e. Stage 1's
rate vs. reach.

**Verdicts on v3** (formats vs. Image; pillars / segments vs. posts without them):

| Platform | Do more / Do less | Promising / Watch |
|---|---|---|
| LinkedIn | **Document** +14.1 pts (13 posts, 77% beat their reach); **Link** −2.9 pts (18, 11%) | Multi-Image +5.2 (promising); Technical Leader −2.0 (watch) |
| Instagram | – | Carousel +2.6, Events & Partnerships +2.5 (promising); Video −2.0, Operations Leader −2.3 (watch) |
| Facebook | – | Operations Leader +3.8 (promising) |

Pillar, segment, timing and boost otherwise show no clear difference; weekend posts, boosts, FB Link, LI Simple
Status and two FB pillars have < 10 posts. Matches Stage 1 (robust: LI Document / Link; suggestive: IG content).

**Design findings (each changed the tool):**
- **Compare with a baseline, not with "all other posts".** Option vs. the rest flagged LinkedIn Image as
  underperforming (−2.7 pts, CI excludes 0) only because Documents lift the rest. Now formats vs. Image, as in
  Stage 1's effect sizes (diffs and Hedges' g reproduce `cohens_d_table` exactly; tested).
- **Correct within a factor, gated by the factor test.** BH across all ~20 options of a platform left LinkedIn
  Document at BH p 0.078 ("Promising"). With the format Wald test first (LI p < 0.001) and BH within the 5 format
  comparisons, Document is p 0.008 → 0.020: "Do more". Same logic as Stage 1's "format matters" rule.
- **Knot check reproduces Stage 1's knot sensitivity.** IG Carousel / Video are "Do more" / "Do less" at 3 and
  4 knots but "Promising" / "Watch" at 5; LinkedIn verdicts identical at 3 / 4 / 5. Firm verdicts must hold at
  every checked knot count, otherwise downgraded with a note.
- **Welch vs. Stage 1's g interval:** IG Video vs. Image on lift has Welch p 0.017 (video lifts vary little),
  while Stage 1's normal-approximation g CI (−0.39) crossed 0. The knot check downgrades it to "Watch" anyway.
- **Planned posts: typical rate + predicted lift, not a direct rate model.** An elastic net of rate (linear in
  reach) put a planned LinkedIn Link post at 8.5% vs. a typical 7.1% for its reach, contradicting "Do less" and
  its 26% chance to beat its reach. Predicting lift from the content choices and adding the spline's typical
  rate keeps scores and verdicts consistent (Link now 5.6%).
- **Recent check needs a platform row and a difference-in-differences.** Comparing each option's recent posts
  with its own older posts flagged almost every Facebook option, because the whole platform rose.

**Planned-post score quality** (held-out, spline refit per split): lift R² FB 0.01, IG 0.02, LI 0.12; ridge
AUC FB 0.60, IG 0.66, LI 0.66. Content choices predict *how much* a post beats its reach only on LinkedIn;
IG scores are usable through the chance to beat its reach; FB scores are flagged as rough guides.

**Last 90 days (Jul 2 – Sep 29, 2026):**
- **Facebook and LinkedIn did better than usual for their reach:** 71% / 67% of recent posts beat their reach
  vs. 38% / 41% before (mean lift +7.2 / +6.0 pts vs. −0.6 / +0.1; Welch p < 0.001 / 0.008). Instagram in line
  (41% vs. 49%).
- **LinkedIn:** the 13 Documents (all in this window) explain most of the mean lift (non-Document recent lift
  +1.1 pts), but not all of the share beating reach (non-Document 61% vs. 41%). Recent LinkedIn posts also had
  far lower reach (median 87 vs. 316 impressions).
- **Facebook:** no format explains it (Multi-Image 15, Image 15, Video 5 recent posts). The whole window falls
  in the Q2–Q3 experiment period. Open question: the Stage 1 regression found the experiment flag ≈ 0 on raw
  rate with a linear reach term; on spline lift the period looks positive for FB. Worth a direct check.
- After BH, only FB **Video** (gap to other FB posts −13.7 pts) and FB **Team Spotlights** (−9.5) fell behind,
  on 5–6 recent posts each: worth a look, not firm.

## Methodological and code findings

- **saga all-zero intercept (2026-11-02):** when every coefficient is zero, scikit-learn's saga stops after
  one pass with the intercept off by up to ~2 probability points. Fixed by setting the exact null intercept
  (log-odds of the training share) in `classify._fit`. Nonzero fits match an exact solver (liblinear).
- **saga tolerance:** 1e-5 gives results identical to 1e-6 and is ~1.7× faster; 1e-4 shifted one pooled
  split's inner tuning (AUC change 0.013 on that split). Full C3 run: 14.9 → 6.0 min with tol 1e-5 plus
  reusing the CV fits for permutation importance.
- **Regression reproduces byte-identically** after the C3 refactor of shared plot/importance code
  (step 4 and step 5 CSVs). A marker-dict bug introduced during the refactor was caught by this check.
- **Committed `step4_importance/figures/cv_performance.png` has broken value labels** (a stray comment
  swallowed `textcoords`; fixed in commit `8fa1bd2`). Notebook 02 displays the broken figure; regenerate
  if the figure is used.
- **`run_info` "uncommitted_changes" counts untracked files,** including the new run folder itself, so a
  first run is always flagged. Pattern: commit the run folder, rerun with `--overwrite`, commit
  `run_info.json` ("reran outputs post … commits"). The C2 run `2026-10-25_c2_logistic_tree` is still
  flagged; its outputs are reproduced cleanly inside the C3 run.
- **Don't run two runs in parallel for the clean rerun:** the first to finish rewrites its `run_info.json`,
  which makes the tree dirty for the second (seen on the C4 reruns, 2026-11-02; fixed by rerunning the
  second alone). Commits: C4 runs `8df206e`, clean run_info `04d9bee`.
- **Several runs in a row (2026-11-03):** run all, commit the folders, then rerun each with `--overwrite`;
  after each rerun copy its `run_info.json` aside and `git checkout` it so the next rerun sees a clean
  tree; restore all and commit together. Scratchpad trial runs (output outside the repo) can run in
  parallel with an official run without dirtying it.
- **Plain-logit crash on perfect separation (2026-11-03):** with a 3-knot spline label, all 3 Facebook
  weekend posts are "low" and the full-data statsmodels logit raises `LinAlgError` (singular matrix),
  stopping `run_classification` before CV. The 3-knot trial used a scratchpad wrapper that skips that
  one fit. **Fixed 2026-11-04:** `fit_logit` returns None on a singular fit; `logistic_fit_stats.csv` records
  it as not estimable and lists `separating_features` (now reported for every fit, e.g. the top-third FB
  weekend case); effects / group tests are skipped for that platform and its figure panel says so. A
  3-knot rerun with the repo code gives CV, stability and importance outputs identical to the wrapper run.
- **openpyxl stamps the save time** into `docProps/core.xml` and the zip entries, so a rerun changed the workbook
  and `run_info` then flagged the tree dirty (git status was read after writing). Fixed 2026-11-07 (`64bcdb4`):
  timestamps pinned to the newest post's date, and `run_recommend` reads git status before writing. A
  verification rerun changes only `run_info.json` (created time, commit).
