"""
Stage 1 evaluation framework: the README's "Planned: quantitative evaluation framework", items 3–7.

Two outcomes throughout, per platform:
  rate            engagement rate (pts), the regression target
  rate_vs_reach   rate minus the expected (median) rate for the post's reach, from the main label's reach
                  rule fit on all of the platform's posts: how far above or below typical for its reach.
                  Raw rate is confounded with reach (small-reach posts have high rates), so group
                  differences in rate alone can be reach differences.

  item 4  group_tests       do content pillars / ICP segments differ?
                            Single-tag posts (independent rows): one-way ANOVA, Welch ANOVA, Kruskal–Wallis,
                            η², ω², Tukey HSD pairs. All posts: one row per post × tag, OLS on tag dummies with
                            post-clustered SEs, joint Wald test that every tag has the same mean.
  item 5  clustered_se      naive vs. HC3 vs. clustered SEs: post × tag rows clustered by post; pooled model
                            clustered by content piece (cross-posted copies)
  item 7  effect_sizes      Hedges' g (bias-corrected Cohen's d, 95% CI) per 0/1 feature; partial R² per
                            feature and feature group
  item 6  cv_mean_sd        held-out metrics as "mean ± SD" over the CV splits, read from committed runs
  item 3  driver_ranking    one table per platform joining regression, logistic, stability, importance and
                            Hedges' g, with Holm and Benjamini–Hochberg adjusted p-values
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

from modeling.features import POOLED, FeatureSet, feature_group, split_tags
from modeling.targets import make_label_rule

FACTORS = {"Content pillar": "CONTENT_PILLAR", "ICP segment": "ICP_SEGMENTS"}


def outcomes(fs: FeatureSet, config: dict) -> dict[str, pd.Series]:
    """The two outcomes for a feature set; rate_vs_reach uses the main label's reach rule."""
    rule = make_label_rule(config["classification"]["main_label"], config).fit(fs)
    if not hasattr(rule, "_expected_platform"):
        raise ValueError("rate_vs_reach needs a reach-adjusted main label")
    return {"rate": fs.y, "rate_vs_reach": (fs.y - rule.expected_rate(fs)).rename("rate_vs_reach")}


def post_tags(prepared: pd.DataFrame, fs: FeatureSet, column: str) -> pd.Series:
    """Tag list per modelled post (index = MASTER_ROW, aligned with fs)."""
    return prepared.set_index("MASTER_ROW").loc[fs.X.index, column].map(split_tags)


# ─────────────────────────────────────────────────────────────────────────────
# Item 4: group differences
# ─────────────────────────────────────────────────────────────────────────────
def oneway_tests(y: pd.Series, tag: pd.Series) -> dict:
    """Classic one-way tests on independent rows (one tag per row); groups as given."""
    names = list(dict.fromkeys(tag))
    groups = [y[tag == g].to_numpy() for g in names]
    n, k = len(y), len(groups)
    grand = y.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(((y - grand) ** 2).sum())
    ms_within = (ss_total - ss_between) / (n - k)
    f, p = stats.f_oneway(*groups)
    welch = anova_oneway(y.to_numpy(), tag.to_numpy(), use_var="unequal")
    h, p_kw = stats.kruskal(*groups)
    return {"n_posts": n, "n_groups": k, "anova_f": f, "anova_p": p,
            "welch_f": float(welch.statistic), "welch_p": float(welch.pvalue), "kruskal_h": h, "kruskal_p": p_kw,
            "eta2": ss_between / ss_total,
            "omega2": (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)}


def tukey_pairs(y: pd.Series, tag: pd.Series) -> pd.DataFrame:
    """Tukey HSD: every pair of groups, mean difference (group2 − group1) with family-wise adjusted p and CI."""
    t = pairwise_tukeyhsd(y.to_numpy(), tag.to_numpy()).summary().data
    out = pd.DataFrame(t[1:], columns=t[0])
    return out.rename(columns={"meandiff": "mean_diff", "p-adj": "p_adj", "lower": "ci_lo", "upper": "ci_hi"})


def exploded(y: pd.Series, tags: pd.Series, keep: list[str]) -> pd.DataFrame:
    """One row per post × tag (tags in `keep` only); `post` identifies the post for clustering."""
    ex = pd.DataFrame({"post": y.index, "y": y.to_numpy(), "tag": tags.to_numpy()}).explode("tag")
    return ex[ex["tag"].isin(keep)].reset_index(drop=True)


def tag_ols(ex: pd.DataFrame, reference: str):
    """OLS of y on tag dummies (reference tag omitted): naive, HC3 and post-clustered fits."""
    X = sm.add_constant(pd.get_dummies(ex["tag"]).drop(columns=reference).astype(float))
    model = sm.OLS(ex["y"].to_numpy(), X)
    return {"naive": model.fit(), "HC3": model.fit(cov_type="HC3"),
            "clustered": model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(ex["post"])[0]})}


def joint_wald(res, names: list[str]) -> tuple[float, int, float]:
    """Joint Wald χ² that every named coefficient is 0, using the fit's own covariance."""
    idx = [list(res.params.index).index(c) for c in names]
    b, V = res.params.to_numpy()[idx], res.cov_params().to_numpy()[np.ix_(idx, idx)]
    chi2 = float(b @ np.linalg.solve(V, b))
    return chi2, len(idx), float(stats.chi2.sf(chi2, len(idx)))


def group_tests(prepared: pd.DataFrame, fs: FeatureSet, outs: dict, min_group: int):
    """
    Item 4 (and the post × tag half of item 5) for one platform. Tags with fewer than `min_group`
    single-tag posts are left out of both analyses so they compare the same groups.
    Returns (omnibus rows, Tukey pairs, SE-comparison rows).
    """
    omnibus, pairs, se_rows = [], [], []
    for factor, column in FACTORS.items():
        tags = post_tags(prepared, fs, column)
        single = tags.map(len) == 1
        first = tags[single].map(lambda t: t[0])
        counts = first.value_counts()
        keep = [t for t in counts.index if counts[t] >= min_group]
        reference = counts.index[0]                                  # most common tag
        dropped = sorted(set(t for ts in tags for t in ts) - set(keep))
        for outcome, y in outs.items():
            ys, tag = y[single][first.isin(keep)], first[first.isin(keep)]
            base = {"platform": fs.platform, "factor": factor, "outcome": outcome}
            row = {**base, **oneway_tests(ys, tag), "groups_left_out": "; ".join(dropped)}
            ex = exploded(y, tags, keep)
            fits = tag_ols(ex, reference)
            dummies = [c for c in fits["clustered"].params.index if c != "const"]
            chi2, df, p = joint_wald(fits["clustered"], dummies)
            row.update({"all_posts_rows": len(ex), "all_posts_posts": ex["post"].nunique(),
                        "clustered_wald_chi2": chi2, "clustered_df": df, "clustered_p": p})
            omnibus.append(row)
            pairs.append(tukey_pairs(ys, tag).assign(**base))
            for c in dummies:
                se = {k: float(r.bse[c]) for k, r in fits.items()}
                se_rows.append({"model": f"{factor} dummies, post × tag rows (ref: {reference})", **base,
                                "term": c, "coef": float(fits["naive"].params[c]),
                                "se_naive": se["naive"], "se_hc3": se["HC3"], "se_clustered": se["clustered"],
                                "ratio_clustered_to_naive": se["clustered"] / se["naive"],
                                "p_naive": float(fits["naive"].pvalues[c]),
                                "p_clustered": float(fits["clustered"].pvalues[c]),
                                "rows": len(ex), "clusters": ex["post"].nunique()})
    return omnibus, pairs, se_rows


def adjust(df: pd.DataFrame, pcols: list[str], by: list[str], method: str = "fdr_bh", suffix: str = "_bh"):
    """Add multiplicity-adjusted copies of p-value columns, adjusted within each `by` family (NaN kept)."""
    df = df.copy()
    for col in pcols:
        df[col + suffix] = np.nan
        for _, idx in df.groupby(by).groups.items():
            p = df.loc[idx, col].dropna()
            if len(p):
                df.loc[p.index, col + suffix] = multipletests(p.to_numpy(), method=method)[1]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Item 5: naive vs clustered SEs in the pooled model (cross-posted content)
# ─────────────────────────────────────────────────────────────────────────────
def pooled_se_comparison(fs: FeatureSet) -> pd.DataFrame:
    if fs.platform != POOLED:
        raise ValueError("pooled feature set expected")
    model = sm.OLS(fs.y, sm.add_constant(fs.X, has_constant="add"))
    fits = {"naive": model.fit(), "HC3": model.fit(cov_type="HC3"),
            "clustered": model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(fs.groups)[0]})}
    rows = []
    for c in fs.X.columns:
        se = {k: float(r.bse[c]) for k, r in fits.items()}
        rows.append({"model": "pooled OLS, one row per post (clustered by content piece)", "platform": POOLED,
                     "factor": feature_group(c), "outcome": "rate", "term": c, "coef": float(fits["naive"].params[c]),
                     "se_naive": se["naive"], "se_hc3": se["HC3"], "se_clustered": se["clustered"],
                     "ratio_clustered_to_naive": se["clustered"] / se["naive"],
                     "p_naive": float(fits["naive"].pvalues[c]), "p_clustered": float(fits["clustered"].pvalues[c]),
                     "rows": len(fs.y), "clusters": fs.groups.nunique()})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Item 7: effect sizes
# ─────────────────────────────────────────────────────────────────────────────
def hedges_g(a, b) -> dict:
    """Bias-corrected standardized mean difference (a − b) / pooled SD, with a normal-approximation 95% CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n0 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n0 - 1) * b.var(ddof=1)) / (n1 + n0 - 2))
    d = (a.mean() - b.mean()) / sp
    g = (1 - 3 / (4 * (n1 + n0) - 9)) * d
    se = np.sqrt((n1 + n0) / (n1 * n0) + g ** 2 / (2 * (n1 + n0)))
    return {"cohens_d": d, "hedges_g": g, "ci_lo": g - 1.96 * se, "ci_hi": g + 1.96 * se}


def cohens_d_table(fs: FeatureSet, outs: dict, config: dict) -> pd.DataFrame:
    """
    Per 0/1 feature: posts with it vs. a comparison group. Formats are compared with the reference format
    (e.g. Document vs. Image); everything else with posts without the feature.
    """
    fmt_cols = [c for c in fs.X.columns if c.startswith("Format: ")]
    is_reference_format = ~fs.X[fmt_cols].any(axis=1).to_numpy()     # the reference format has no column
    rows = []
    for c in fs.X.columns:
        v = fs.X[c].to_numpy()
        if not set(np.unique(v)) <= {0.0, 1.0}:
            continue
        if c in fmt_cols:
            other, versus = is_reference_format, config["formats"]["reference"]
        else:
            other, versus = v == 0, "posts without it"
        with_ = v == 1
        if with_.sum() < 2 or other.sum() < 2:
            continue
        for outcome, y in outs.items():
            y = y.to_numpy()
            rows.append({"platform": fs.platform, "feature": c, "outcome": outcome, "compared_with": versus,
                         "n_with": int(with_.sum()), "n_compared": int(other.sum()),
                         "mean_with": y[with_].mean(), "mean_compared": y[other].mean(),
                         "diff_pts": y[with_].mean() - y[other].mean(), **hedges_g(y[with_], y[other])})
    return pd.DataFrame(rows)


def partial_r2_table(fs: FeatureSet, outs: dict) -> pd.DataFrame:
    """
    Partial R² of each feature and feature group: the share of the variance left unexplained without it
    that it explains, (SSE_without − SSE_full) / SSE_without, from plain OLS refits on the same posts.
    """
    rows = []
    units = [("feature", c, [c]) for c in fs.X.columns]
    for g in dict.fromkeys(feature_group(c) for c in fs.X.columns):
        units.append(("group", g, [c for c in fs.X.columns if feature_group(c) == g]))
    for outcome, y in outs.items():
        full = sm.OLS(y.to_numpy(), sm.add_constant(fs.X, has_constant="add")).fit()
        for level, unit, cols in units:
            reduced = sm.OLS(y.to_numpy(), sm.add_constant(fs.X.drop(columns=cols), has_constant="add")).fit()
            rows.append({"platform": fs.platform, "outcome": outcome, "level": level, "unit": unit,
                         "n_features": len(cols), "partial_r2": (reduced.ssr - full.ssr) / reduced.ssr,
                         "full_r2": full.rsquared})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Items 6 and 3: tables assembled from committed run folders (no refitting)
# ─────────────────────────────────────────────────────────────────────────────
REGRESSION_METRICS = ["r2", "mae", "rmse"]
CLASSIFICATION_METRICS = ["auc", "balanced_accuracy", "f1", "brier"]


def cv_mean_sd(regression_dir: Path, classification_dir: Path) -> pd.DataFrame:
    """Held-out metrics per platform and model as mean ± SD over the CV splits."""
    rows = []
    for task, folder, metrics in [("regression", regression_dir, REGRESSION_METRICS),
                                  ("classification", classification_dir, CLASSIFICATION_METRICS)]:
        s = pd.read_csv(Path(folder) / "cv_summary.csv")
        for _, r in s.iterrows():
            for m in metrics:
                rows.append({"task": task, "platform": r["platform"], "model": r["model"], "metric": m,
                             "mean": r[f"{m}_mean"], "sd": r[f"{m}_std"],
                             "report": f"{r[f'{m}_mean']:.2f} ± {r[f'{m}_std']:.2f}"})
    return pd.DataFrame(rows)


def driver_ranking(regression_dir: Path, classification_dir: Path, effects: pd.DataFrame,
                   stable_freq: float, stable_sign: float) -> pd.DataFrame:
    """
    One row per platform × feature. p-values are adjusted within each platform (Holm: family-wise;
    BH: false discovery rate), separately for the regression and the logistic tests. Ranked by the smaller
    BH-adjusted p, then by |standardized β|. `evidence` lists what supports the feature.
    """
    reg, cls = Path(regression_dir), Path(classification_dir)
    ols = pd.read_csv(reg / "coefficients_ols.csv").query("feature != 'const'")
    t = ols[["platform", "feature", "coef_pts", "std_beta", "p_value"]].rename(columns={"p_value": "p_reg"})
    logit = pd.read_csv(cls / "logistic_effects.csv").query("feature != 'const'")
    t = t.merge(logit[["platform", "feature", "odds_ratio", "p_value"]].rename(columns={"p_value": "p_logit"}),
                on=["platform", "feature"], how="left")
    for name, folder, model in [("reg", reg, "Lasso"), ("logit", cls, "Lasso logistic")]:
        st = pd.read_csv(folder / "stability.csv").query("model == @model")
        st = st[["platform", "feature", "selection_freq", "sign_agreement"]]
        t = t.merge(st.rename(columns={"selection_freq": f"lasso_freq_{name}",
                                       "sign_agreement": f"lasso_sign_{name}"}), on=["platform", "feature"], how="left")
    for name, folder, metric in [("reg_mae_rise", reg, "mae_rise_mean"), ("logit_auc_drop", cls, "auc_drop_mean")]:
        imp = pd.read_csv(folder / "importance.csv").query("level == 'feature'")
        t = t.merge(imp[["platform", "unit", metric]].rename(columns={"unit": "feature", metric: f"importance_{name}"}),
                    on=["platform", "feature"], how="left")
    g = effects.query("outcome == 'rate_vs_reach'")[["platform", "feature", "hedges_g", "ci_lo", "ci_hi",
                                                     "compared_with"]]
    t = t.merge(g.rename(columns={"hedges_g": "hedges_g_vs_reach", "ci_lo": "g_ci_lo", "ci_hi": "g_ci_hi"}),
                on=["platform", "feature"], how="left")
    for col in ["p_reg", "p_logit"]:
        t = adjust(t, [col], ["platform"], "holm", "_holm")
        t = adjust(t, [col], ["platform"], "fdr_bh", "_bh")

    def evidence(r):
        e = []
        if r["p_reg_bh"] < 0.05:
            e.append("regression p (BH)")
        if r["p_logit_bh"] < 0.05:
            e.append("logistic p (BH)")
        for name in ("reg", "logit"):
            if r[f"lasso_freq_{name}"] >= stable_freq and r[f"lasso_sign_{name}"] >= stable_sign:
                e.append(f"stable lasso ({name})")
        return "; ".join(e)
    t["evidence"] = t.apply(evidence, axis=1)
    t["n_evidence"] = t["evidence"].map(lambda s: len(s.split("; ")) if s else 0)
    t["best_p_bh"] = t[["p_reg_bh", "p_logit_bh"]].min(axis=1)
    t["abs_std_beta"] = t["std_beta"].abs()
    t = t.sort_values(["platform", "best_p_bh", "abs_std_beta"], ascending=[True, True, False])
    t["rank"] = t.groupby("platform").cumcount() + 1
    return t.drop(columns="abs_std_beta")
