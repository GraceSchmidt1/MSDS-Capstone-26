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

Planned posts (a content calendar CSV, see plan_template.csv):
    plan_specs      validate each row (unknown platform / format / tag, bad date or time, …) into a spec or an
                    error message; blank impressions → the median of past posts in that platform and format
    fit_scorers     per platform: elastic net of lift on the content choices, ridge logistic of "beat its
                    reach", and bootstrap refits for a range
    score_plan      typical rate for the reach, expected lift and rate, chance to beat its reach, and the
                    verdicts that apply to its choices
    model_quality   held-out lift R² and AUC per platform, and whether the scores are usable or rough guides

Recent check (recent_check): the last `recent_days` up to the newest post. Each recent post vs. its reach, and
the platform overall recent vs. older, and per option whether its gap to the platform's other posts changed
(flagged only when clear: BH-adjusted p < alpha).
"""
from __future__ import annotations

import copy
import datetime as dt
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


# ─────────────────────────────────────────────────────────────────────────────
# R2: scoring planned posts (a content calendar)
# ─────────────────────────────────────────────────────────────────────────────
# Plan CSV, one row per planned post. Required: platform, date, time, format, pillars, icp_segments.
# Optional: post_id, impressions (blank → typical for the platform and format), boosted (LinkedIn; yes/no).
# Several pillars / segments are separated by ";", as in the master.
PLAN_REQUIRED = ["platform", "date", "time", "format", "pillars", "icp_segments"]
PLAN_OPTIONAL = ["post_id", "impressions", "boosted"]
TIME_FORMATS = ["%I:%M %p", "%I:%M%p", "%I %p", "%I%p", "%H:%M"]
RATE_MODEL, BEAT_MODEL = "Elastic net", "Ridge logistic"


def read_plan(path) -> pd.DataFrame:
    plan = pd.read_csv(path, dtype=str, keep_default_na=False)
    plan.columns = [c.strip().lower() for c in plan.columns]
    missing = [c for c in PLAN_REQUIRED if c not in plan.columns]
    if missing:
        raise ValueError(f"Plan file is missing columns: {', '.join(missing)} (required: {', '.join(PLAN_REQUIRED)})")
    for c in PLAN_OPTIONAL:
        if c not in plan.columns:
            plan[c] = ""
    plan = plan.map(str.strip)
    plan["post_id"] = [v or f"row {i + 2}" for i, v in enumerate(plan["post_id"])]   # CSV row number
    return plan


def _parse_time(value: str):
    for fmt in TIME_FORMATS:
        try:
            return dt.datetime.strptime(value.upper().replace(".", ""), fmt).hour
        except ValueError:
            pass
    return None


def typical_impressions(prepared: pd.DataFrame, platform: str, fmt: str, config: dict) -> tuple[float, str]:
    """Median impressions of the platform's posts in that format, or of all its posts if the format is rare."""
    rows = prepared[(prepared["PLATFORM"] == platform) & (prepared["IMPRESSIONS"] > 0)]
    same = rows.loc[rows["FORMAT"] == fmt, "IMPRESSIONS"]
    if len(same) >= config["recommend"]["min_posts"]:
        return float(same.median()), f"median of {len(same)} past {platform} {fmt} posts"
    return float(rows["IMPRESSIONS"].median()), f"median of all {len(rows)} past {platform} posts"


def plan_specs(plan: pd.DataFrame, prepared: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Validate each planned post; returns the plan with `spec` (dict for features_from_spec) or `error`."""
    out = []
    for _, r in plan.iterrows():
        errors, spec, source = [], None, ""
        platform = r["platform"].strip().title().replace("Linkedin", "LinkedIn")
        if platform not in config["platforms"]:
            errors.append(f"unknown platform '{r['platform']}'")
        elif r["format"] not in config["formats"][platform]:
            errors.append(f"format '{r['format']}' not used on {platform} (one of: {', '.join(config['formats'][platform])})")
        pillars, icp = split_tags(r["pillars"]), split_tags(r["icp_segments"])
        bad = [t for t in pillars if t not in config["pillars"]] + [t for t in icp if t not in config["icp_segments"]]
        if bad:
            errors.append(f"unknown pillar / segment: {', '.join(bad)}")
        if not pillars or not icp:
            errors.append("needs at least one pillar and one ICP segment")
        date = pd.to_datetime(r["date"], errors="coerce")
        if pd.isna(date):
            errors.append(f"unreadable date '{r['date']}' (use YYYY-MM-DD)")
        hour = _parse_time(r["time"])
        if hour is None:
            errors.append(f"unreadable time '{r['time']}' (e.g. 9:30 AM or 14:00)")
        impressions = pd.to_numeric(r["impressions"], errors="coerce") if r["impressions"] else np.nan
        if r["impressions"] and not (impressions > 0):
            errors.append(f"impressions must be a positive number, got '{r['impressions']}'")
        boosted = r["boosted"].lower() in ("yes", "y", "true", "1")
        if boosted and platform != "LinkedIn":
            errors.append("boosted applies to LinkedIn posts only")
        if not errors:
            if np.isnan(impressions):
                impressions, source = typical_impressions(prepared, platform, r["format"], config)
            else:
                source = "given in plan"
            spec = {"icp": icp, "pillars": pillars, "format": r["format"], "impressions": impressions,
                    "weekend": date.dayofweek >= 5, "hour": hour, "boosted": boosted}
        out.append({**r.to_dict(), "platform": platform if platform in config["platforms"] else r["platform"], "spec": spec, "impressions_used": spec and impressions,
                    "impressions_source": source, "error": "; ".join(errors)})
    return pd.DataFrame(out)


def _content(fs: FeatureSet, config: dict) -> pd.DataFrame:
    """Feature columns without reach (reach is handled by the spline, i.e. the typical rate)."""
    return fs.X.drop(columns=config["features"]["reach"])


def fit_scorers(fs: FeatureSet, config: dict) -> dict:
    """
    Final models on all of the platform's posts (penalty tuned on the shared CV splits, as in Stage 1):
      lift model    elastic net of lift (rate − typical rate for the reach) on the content choices, so
                    expected rate = typical rate for the reach + predicted lift
      beat model    ridge logistic of "beat its reach" (the Stage 1 label) on all features
    plus `plan_boot` refits on resampled posts with the penalty fixed, for a range around each score.
    """
    from modeling.classify import _fit, _fixed_penalty_logistic, fit_penalized
    from modeling.models import _fixed_penalty_model, cv_splits, make_regularized

    splits = cv_splits(fs, config["cv"]["n_splits"], config["cv"]["n_repeats"], config["seed"])
    rule = make_label_rule(config["classification"]["main_label"], config).fit(fs)
    beat, lift = rule.label(fs), fs.y.to_numpy() - rule.expected_rate(fs)
    Xc, X = _content(fs, config).to_numpy(), fs.X.to_numpy()
    lift_model = make_regularized(RATE_MODEL, splits, config).fit(Xc, lift)
    clf = fit_penalized(fs, beat, BEAT_MODEL, config, splits)
    rng = np.random.default_rng(config["seed"])
    boots = []
    for _ in range(config["recommend"]["plan_boot"]):
        idx = rng.integers(0, len(lift), len(lift))
        if beat[idx].min() == beat[idx].max():
            continue
        boots.append((_fixed_penalty_model(RATE_MODEL, lift_model, config).fit(Xc[idx], lift[idx]),
                      _fit(_fixed_penalty_logistic(BEAT_MODEL, clf, config), X[idx], beat[idx])))
    return {"rule": rule, "lift": lift_model, "beat": clf, "boots": boots, "columns": list(fs.X.columns),
            "content_columns": list(_content(fs, config).columns)}


def model_quality(fs: FeatureSet, config: dict) -> dict:
    """
    Held-out quality of the two scoring models on the shared CV splits (nested tuning, as in Stage 1). For the
    lift model the reach spline is refit on each split's training posts, so held-out posts never shape their
    own target. lift_r2 > 0: the content choices predict lift better than the training-mean lift does.
    """
    from modeling.classify import cv_score_classifier, penalized_fit_predict
    from modeling.models import cv_splits, regularized_fit_predict

    splits = cv_splits(fs, config["cv"]["n_splits"], config["cv"]["n_repeats"], config["seed"])
    Xc, y, groups = _content(fs, config).to_numpy(), fs.y.to_numpy(), fs.groups.to_numpy()
    fit_predict = regularized_fit_predict(RATE_MODEL, config)
    r2 = []
    for tr, te in splits:
        rule = make_label_rule(config["classification"]["main_label"], config).fit(fs, tr)
        lift_tr, lift_te = y[tr] - rule.expected_rate(fs, tr), y[te] - rule.expected_rate(fs, te)
        pred = fit_predict(Xc[tr], lift_tr, Xc[te], groups[tr])
        r2.append(1 - ((lift_te - pred) ** 2).sum() / ((lift_te - lift_tr.mean()) ** 2).sum())
    r2 = np.array(r2)
    auc = cv_score_classifier(fs, splits, config["classification"]["main_label"], config,
                              penalized_fit_predict(BEAT_MODEL, config))["auc"]
    q = config["recommend"]["reliable"]
    reliable = r2.mean() >= q["r2"] or auc.mean() >= q["auc"]
    return {"platform": fs.platform, "posts": len(y), "lift_model": RATE_MODEL, "lift_r2_mean": r2.mean(),
            "lift_r2_sd": r2.std(), "beat_model": BEAT_MODEL, "auc_mean": auc.mean(), "auc_sd": auc.std(),
            "reliability": "usable" if reliable else "weak: treat scores as rough guides"}


def _option_notes(spec: dict, options: pd.DataFrame) -> str:
    """The verdicts that apply to a planned post, skipping baselines and 'No clear difference'."""
    v = options.set_index(["factor", "option"])["verdict"]
    picks = [("Format", spec["format"])] + [("Content pillar", t) for t in spec["pillars"]]
    picks += [("ICP segment", t) for t in spec["icp"]]
    picks += [("Timing", "Weekend" if spec["weekend"] else "Weekday")]
    picks += [("Timing", "11am or later" if spec["hour"] >= 11 else "Before 11am")]
    if ("Paid boost", "Boosted") in v.index:
        picks += [("Paid boost", "Boosted" if spec["boosted"] else "Not boosted")]
    keep = [f"{o}: {v[(f, o)]}" for f, o in picks
            if (f, o) in v.index and v[(f, o)] not in ("Baseline", "No clear difference")]
    return "; ".join(keep)


def score_plan(specs: pd.DataFrame, scorers: dict, options: dict, config: dict) -> pd.DataFrame:
    """
    Per valid planned post:
      typical_rate          the usual rate for a post with that reach on the platform (the reach spline)
      expected_lift         predicted lift from its content choices (elastic net), lo / hi = 2.5 / 97.5% over
                            the bootstrap refits (uncertainty of the estimate, not the spread of single posts,
                            which is much wider)
      expected_rate         typical_rate + expected_lift (pts, floored at 0), lo / hi likewise
      chance_beat           ridge-logistic probability that it beats the typical rate for its reach, lo / hi
      notes                 verdicts that apply to its choices (see option tables)
    """
    from modeling.features import features_from_spec

    rows = []
    for _, r in specs.iterrows():
        out = {k: r[k] for k in ["post_id", "platform", "date", "time", "format", "pillars", "icp_segments",
                                 "boosted", "impressions_used", "impressions_source", "error"]}
        if r["spec"] is not None:
            s = scorers[r["platform"]]
            x = features_from_spec(r["spec"], r["platform"], config)[s["columns"]].to_numpy()
            reach = FeatureSet(r["platform"], pd.DataFrame(x, columns=s["columns"]), pd.Series([0.0]), pd.Series(["plan"]))
            typical = float(s["rule"].expected_rate(reach)[0])
            xc = pd.DataFrame(x, columns=s["columns"])[s["content_columns"]].to_numpy()
            lift = float(s["lift"].predict(xc)[0])
            beat = float(s["beat"].predict_proba(x)[0, 1])
            bl = np.array([m.predict(xc)[0] for m, _ in s["boots"]])
            bb = np.array([c.predict_proba(x)[0, 1] for _, c in s["boots"]])
            lo, hi = np.quantile(bl, [0.025, 0.975])
            out.update({"typical_rate": typical, "expected_lift": lift, "expected_lift_lo": lo,
                        "expected_lift_hi": hi, "expected_rate": max(typical + lift, 0.0),
                        "expected_rate_lo": max(typical + lo, 0.0), "expected_rate_hi": max(typical + hi, 0.0),
                        "chance_beat": beat, "chance_beat_lo": np.quantile(bb, 0.025),
                        "chance_beat_hi": np.quantile(bb, 0.975), "notes": _option_notes(r["spec"], options[r["platform"]])})
        rows.append(out)
    return pd.DataFrame(rows).assign(notes=lambda d: d.get("notes", pd.Series("", index=d.index)).fillna(""))


# ─────────────────────────────────────────────────────────────────────────────
# R3: recent check (has anything changed lately?)
# ─────────────────────────────────────────────────────────────────────────────
def recent_window(prepared: pd.DataFrame, config: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end) of the last `recent_days`, ending at the newest post in the master."""
    end = prepared["DATE_TS"].max()
    return end - pd.Timedelta(days=config["recommend"]["recent_days"] - 1), end


def recent_check(prepared: pd.DataFrame, fs: FeatureSet, options: pd.DataFrame, config: dict,
                 perf: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    recent_posts    every post in the window: rate, typical rate for its reach, lift, whether it beat its
                    reach, and `standout` = among the platform's top / bottom `standout_share` of lifts ever
    recent_options  first row "All posts": recent vs. older lift for the whole platform (Welch t-test), i.e.
                    is the platform doing better or worse than usual lately. Then per option: recent posts,
                    recent and older share beat and mean lift, and `gap_change` = how much its gap to the
                    platform's other posts changed (recent gap − older gap), tested as the interaction in
                    OLS lift ~ option × recent (HC3), BH-adjusted over the platform's options. So an option is
                    flagged only if it moved differently from the rest of the platform: "Pulling ahead recently"
                    / "Falling behind recently" (p_bh < alpha);
                    "Too few recent posts" below `recent_min_posts`; else "In line"
    Verdicts use all posts equally; this only flags changes for a closer look.
    """
    r = config["recommend"]
    perf = performance(fs, config) if perf is None else perf
    rows = prepared.set_index("MASTER_ROW").loc[fs.X.index]
    start, end = recent_window(prepared, config)
    recent = rows["DATE_TS"].between(start, end).to_numpy()
    lift, beat = perf["lift"].to_numpy(), perf["beat"].to_numpy()
    lo, hi = np.quantile(lift, [r["standout_share"], 1 - r["standout_share"]])
    posts = rows.loc[recent, ["DATE_TS", "POST_TITLE", "FORMAT", "CONTENT_PILLAR", "ICP_SEGMENTS", "IMPRESSIONS"]]
    posts = posts.assign(**perf.loc[recent, ["rate", "expected_rate", "lift"]],
                         result=np.where(beat[recent] == 1, "Beat its reach", "Below its reach"),
                         standout=np.select([lift[recent] >= hi, lift[recent] <= lo],
                                            ["Top post", "Bottom post"], ""))
    posts = (posts.rename(columns={"DATE_TS": "date", "POST_TITLE": "title", "FORMAT": "format",
                                   "CONTENT_PILLAR": "pillars", "ICP_SEGMENTS": "icp_segments",
                                   "IMPRESSIONS": "impressions", "expected_rate": "typical_rate"})
             .reset_index().rename(columns={"MASTER_ROW": "master_row"})
             .assign(platform=fs.platform).sort_values("date", ascending=False))

    verdict_of = options.set_index(["factor", "option"])["verdict"]
    old_all = ~recent
    out = [{"platform": fs.platform, "factor": "All posts", "option": "All posts", "verdict": "",
            "recent_posts": int(recent.sum()), "recent_share_beat": beat[recent].mean(),
            "recent_mean_lift": lift[recent].mean(), "older_posts": int(old_all.sum()),
            "older_share_beat": beat[old_all].mean(), "older_mean_lift": lift[old_all].mean(), "gap_change": np.nan,
            "p_value": stats.ttest_ind(lift[recent], lift[old_all], equal_var=False).pvalue}]
    out[0]["trend"] = ("In line" if out[0]["p_value"] >= r["alpha"] else
                       "Platform doing better recently" if out[0]["recent_mean_lift"] > out[0]["older_mean_lift"]
                       else "Platform doing worse recently")
    for factor, option, m, _, _ in option_masks(prepared, fs, config):
        new, old = m & recent, m & ~recent
        row = {"platform": fs.platform, "factor": factor, "option": option,
               "verdict": verdict_of.get((factor, option), ""), "recent_posts": int(new.sum()),
               "recent_share_beat": beat[new].mean() if new.any() else np.nan,
               "recent_mean_lift": lift[new].mean() if new.any() else np.nan, "older_posts": int(old.sum()),
               "older_share_beat": beat[old].mean() if old.any() else np.nan,
               "older_mean_lift": lift[old].mean() if old.any() else np.nan, "gap_change": np.nan, "p_value": np.nan}
        cells = [new, old, ~m & recent, ~m & ~recent]
        if new.sum() < r["recent_min_posts"] or min(c.sum() for c in cells[1:]) < 2:
            row["trend"] = "Too few recent posts"
        else:
            X = sm.add_constant(np.column_stack([m, recent, m & recent]).astype(float))
            res = sm.OLS(lift, X).fit(cov_type="HC3")
            row["gap_change"], row["p_value"] = float(res.params[3]), float(res.pvalues[3])
        out.append(row)
    out = pd.DataFrame(out)
    tested = out.index[1:][out["p_value"].iloc[1:].notna()]          # option rows, BH within the platform
    out["p_bh"] = np.nan
    out.loc[0, "p_bh"] = out.loc[0, "p_value"]
    if len(tested):
        out.loc[tested, "p_bh"] = multipletests(out.loc[tested, "p_value"], method="fdr_bh")[1]
        out.loc[tested, "trend"] = np.where(out.loc[tested, "p_bh"] >= r["alpha"], "In line",
                                            np.where(out.loc[tested, "gap_change"] > 0, "Pulling ahead recently",
                                                     "Falling behind recently"))
    return posts.reset_index(drop=True), out
