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

## Evaluation framework, items 1–3 (trial runs 2026-11-02; official runs pending)

Reach-adjusted label unless noted. Trial outputs were checked against the committed runs: every existing
number is unchanged; only new columns, models and files were added.

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
