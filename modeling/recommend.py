"""
Stage 2 recommendations: what is performing well vs. what isn't, per platform.

Every post is judged against its own reach. The expected rate for a post's reach comes from the main label's
reach rule (the Stage 1 `reach_spline`), fit on all of the platform's posts, so

    lift = rate − expected rate for that reach            (pts; ≈ 0 for a typical post)
    beat = 1 if the post is above the typical rate for its reach (the Stage 1 label)

For each option the marketing team chooses, posts with the option are compared with a baseline, as in the
Stage 1 effect sizes (evaluate.cohens_d_table):
    Format          each format vs. the reference format (Image), which is the "Baseline" row
    Timing          Weekend vs. Weekday, 11am or later vs. Before 11am (the second is the baseline)
    Paid boost      Boosted vs. Not boosted (LinkedIn)
    Pillar / ICP    each tag vs. posts without it (posts carry several tags, so there is no single baseline)

    option_table    posts, share that beat their reach, mean lift with a bootstrap 95% CI, difference vs. the
                    baseline (bootstrap CI), Hedges' g, Welch t-test p and its Benjamini–Hochberg value within
                    the factor, and how many posts would settle it (power calculation)
    factor_tests    does the factor matter at all? Joint Wald test of its columns in one OLS of lift on every
                    option (HC3 SEs), BH-adjusted across the platform's factors
    add_verdicts    plain-language verdict per option (rules below; thresholds in config `recommend:`)

Verdicts, checked in order (the factor test gates the option tests, as in Stage 1's "format matters" rule):
    Baseline                the comparison option itself
    Not enough posts yet    fewer than `min_posts` posts with the option or in its comparison group
    Do more / Do less       its factor survives BH (p_bh < alpha) and so does its own test within the factor
    Promising / Watch       its own unadjusted p < alpha, or its factor survives BH and its bootstrap CI for the
                            difference excludes 0: positive → "Promising, keep testing", negative → "Watch, tends
                            to underperform"
    No clear difference     otherwise
Knot check: a "Do more" / "Do less" stands only if the verdict is the same when the reach spline uses each of the
`knot_check` knot counts instead; otherwise it is downgraded to "Promising" / "Watch" and `note` says so
(Stage 1 found Instagram content effects held at 3–4 knots, not 5).
The comparisons are descriptive (after reach, but other choices made on the same posts are not held fixed).
Stage 1 (notebooks 02–03) has the adjusted models.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower

from modeling.evaluate import hedges_g, joint_wald
from modeling.features import FeatureSet, feature_group, split_tags
from modeling.targets import make_label_rule

VERDICTS = ["Do more", "Promising, keep testing", "No clear difference", "Watch, tends to underperform",
            "Do less", "Not enough posts yet", "Baseline"]


def performance(fs: FeatureSet, config: dict) -> pd.DataFrame:
    """Per post: rate, expected rate for its reach, lift, and whether it beat its reach (index = MASTER_ROW)."""
    rule = make_label_rule(config["classification"]["main_label"], config).fit(fs)
    expected = rule.expected_rate(fs)
    return pd.DataFrame({"rate": fs.y.to_numpy(), "expected_rate": expected, "lift": fs.y.to_numpy() - expected,
                         "beat": rule.label(fs)}, index=fs.y.index)


def option_masks(prepared: pd.DataFrame, fs: FeatureSet, config: dict) -> list[tuple]:
    """
    (factor, option, mask, comparison mask, compared_with) over fs rows for every choice on the platform.
    A baseline option has comparison mask None.
    """
    rows = prepared.set_index("MASTER_ROW").loc[fs.X.index]
    ref = config["formats"]["reference"]
    is_ref = rows["FORMAT"].eq(ref).to_numpy()
    out = [("Format", f, rows["FORMAT"].eq(f).to_numpy(), None if f == ref else is_ref, ref)
           for f in config["formats"][fs.platform]]
    for factor, col, vocab in [("Content pillar", "CONTENT_PILLAR", config["pillars"]),
                               ("ICP segment", "ICP_SEGMENTS", config["icp_segments"])]:
        tags = rows[col].map(lambda v: set(split_tags(v)))
        for t in vocab:
            m = tags.map(lambda s, t=t: t in s).to_numpy()
            out.append((factor, t, m, ~m, "posts without it"))
    binaries = [("Timing", "WEEKEND", "Weekend", "Weekday"), ("Timing", "AFTER_11", "11am or later", "Before 11am")]
    if fs.platform == "LinkedIn":
        binaries.append(("Paid boost", "BOOSTED", "Boosted", "Not boosted"))
    for factor, col, yes, no in binaries:
        on = rows[col].eq(1).to_numpy()
        out += [(factor, yes, on, ~on, no), (factor, no, ~on, None, no)]
    return out


def _boot_means(rng, a, b, n_boot):
    """Bootstrap means of a and of (a − b), resampling each group separately."""
    ia = rng.integers(0, len(a), (n_boot, len(a)))
    ib = rng.integers(0, len(b), (n_boot, len(b)))
    ma = a[ia].mean(axis=1)
    return ma, ma - b[ib].mean(axis=1)


def posts_needed(g: float, n_with: int, n_without: int, config: dict) -> float:
    """
    Posts with the option needed for `power` to detect an effect of size g (two-sided Welch-style t-test at
    alpha), keeping the current ratio of other posts; returns the additional posts beyond n_with, capped at
    `max_posts_needed` (inf when the effect is negligible, |g| < 0.1).
    """
    r = config["recommend"]
    if not np.isfinite(g) or abs(g) < 0.1:
        return np.inf
    ratio = max(n_without, 1) / max(n_with, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # a failed solve is reported as "too many to settle"
        try:
            n1 = TTestIndPower().solve_power(effect_size=abs(g), alpha=r["alpha"], power=r["power"], ratio=ratio)
        except Exception:
            return np.inf
    extra = max(0.0, np.ceil(n1) - n_with)
    return extra if extra <= r["max_posts_needed"] else np.inf


def option_table(prepared: pd.DataFrame, fs: FeatureSet, config: dict, perf: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per option on the platform (see module docstring). Deterministic for a fixed config seed."""
    r = config["recommend"]
    perf = performance(fs, config) if perf is None else perf
    lift, beat = perf["lift"].to_numpy(), perf["beat"].to_numpy()
    rng = np.random.default_rng(config["seed"])
    rows = []
    for factor, option, m, cmp, versus in option_masks(prepared, fs, config):
        a = lift[m]
        row = {"platform": fs.platform, "factor": factor, "option": option, "posts": int(m.sum()),
               "compared_with": "–" if cmp is None else versus, "comparison_posts": np.nan if cmp is None else int(cmp.sum()),
               "share_beat": beat[m].mean() if m.any() else np.nan,
               "mean_lift": a.mean() if m.any() else np.nan, "median_lift": np.median(a) if m.any() else np.nan}
        if m.any():
            row["lift_ci_lo"], row["lift_ci_hi"] = np.quantile(_boot_means(rng, a, a, r["n_boot"])[0], [0.025, 0.975])
        if cmp is not None and m.sum() >= 2 and cmp.sum() >= 2:
            b = lift[cmp]
            diff = _boot_means(rng, a, b, r["n_boot"])[1]
            g = hedges_g(a, b)
            row.update({"diff_vs_baseline": a.mean() - b.mean(), "diff_ci_lo": np.quantile(diff, 0.025),
                        "diff_ci_hi": np.quantile(diff, 0.975), "hedges_g": g["hedges_g"],
                        "p_value": stats.ttest_ind(a, b, equal_var=False).pvalue,
                        "posts_needed": posts_needed(g["hedges_g"], int(m.sum()), int(cmp.sum()), config)})
        rows.append(row)
    t = pd.DataFrame(rows)
    t["p_bh"] = np.nan                                                # BH within each factor
    for _, idx in t.groupby("factor").groups.items():
        p = t.loc[idx, "p_value"].dropna()
        if len(p):
            t.loc[p.index, "p_bh"] = multipletests(p.to_numpy(), method="fdr_bh")[1]
    return t


def factor_tests(fs: FeatureSet, config: dict, perf: pd.DataFrame | None = None) -> pd.DataFrame:
    """Joint Wald test per factor in one OLS of lift on every option column (reach already removed), HC3 SEs."""
    perf = performance(fs, config) if perf is None else perf
    X = fs.X.drop(columns=config["features"]["reach"])
    X = X.loc[:, X.nunique() > 1]                                     # an option nobody used carries no test
    res = sm.OLS(perf["lift"].to_numpy(), sm.add_constant(X, has_constant="add")).fit(cov_type="HC3")
    factor_of = {"Timing": "Timing", "Paid boost": "Paid boost"}
    rows = []
    for g in dict.fromkeys(feature_group(c) for c in X.columns):
        cols = [c for c in X.columns if feature_group(c) == g]
        chi2, df, p = joint_wald(res, cols)
        rows.append({"platform": fs.platform, "factor": factor_of.get(g, g), "wald_chi2": chi2, "df": df,
                     "p_value": p})
    t = pd.DataFrame(rows)
    t["p_bh"] = multipletests(t["p_value"], method="fdr_bh")[1]
    return t


def verdict(row, factor_p_bh: float, config: dict) -> str:
    r = config["recommend"]
    alpha = r["alpha"]
    if row["compared_with"] == "–":
        return "Baseline"
    if row["posts"] < r["min_posts"] or row["comparison_posts"] < r["min_posts"] or pd.isna(row.get("p_value")):
        return "Not enough posts yet"
    factor_ok = factor_p_bh < alpha
    up = row["diff_vs_baseline"] > 0
    if factor_ok and row["p_bh"] < alpha:
        return "Do more" if up else "Do less"
    ci_clear = row["diff_ci_lo"] > 0 or row["diff_ci_hi"] < 0
    if row["p_value"] < alpha or (factor_ok and ci_clear):
        return "Promising, keep testing" if up else "Watch, tends to underperform"
    return "No clear difference"


def add_verdicts(options: pd.DataFrame, factors: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Attach each option's factor test and its verdict."""
    f = factors.set_index(["platform", "factor"])
    out = options.copy()
    key = list(zip(out["platform"], out["factor"]))
    out["factor_p_bh"] = [f["p_bh"].get(k, np.nan) for k in key]
    out["verdict"] = [verdict(r, fp, config) for (_, r), fp in zip(out.iterrows(), out["factor_p_bh"])]
    return out


DOWNGRADE = {"Do more": "Promising, keep testing", "Do less": "Watch, tends to underperform"}


def _verdicts(prepared: pd.DataFrame, fs: FeatureSet, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    perf = performance(fs, config)
    factors = factor_tests(fs, config, perf)
    return add_verdicts(option_table(prepared, fs, config, perf), factors, config), factors


def with_knots(config: dict, n_knots: int) -> dict:
    """Copy of config whose main (spline) label uses n_knots."""
    c = copy.deepcopy(config)
    c["classification"]["labels"][c["classification"]["main_label"]]["n_knots"] = n_knots
    return c


def recommendations(prepared: pd.DataFrame, fs: FeatureSet, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(options with verdicts after the knot check, factor tests) for one platform."""
    options, factors = _verdicts(prepared, fs, config)
    options["note"] = ""
    spec = config["classification"]["labels"][config["classification"]["main_label"]]
    if spec["type"] != "reach_spline":
        return options, factors
    other = {k: _verdicts(prepared, fs, with_knots(config, k))[0]["verdict"] for k in config["recommend"]["knot_check"]}
    for k, v in other.items():
        options[f"verdict_{k}_knots"] = v.to_numpy()
    strong = options["verdict"].isin(list(DOWNGRADE))
    for i in options.index[strong]:
        failed = [k for k, v in other.items() if v[i] != options.at[i, "verdict"]]
        if failed:
            options.at[i, "note"] = (f"{options.at[i, 'verdict']} at {spec['n_knots']} knots but not at "
                                     f"{', '.join(map(str, failed))}: depends on how reach is modelled")
            options.at[i, "verdict"] = DOWNGRADE[options.at[i, "verdict"]]
    return options, factors
