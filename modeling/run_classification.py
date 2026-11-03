#!/usr/bin/env python3
"""
Run the Stage 1 high- vs. low-engagement classifier and write a self-describing run folder.

    python -m modeling.run_classification --tag c3_penalized              (from the repo root)
    python -m modeling.run_classification --tag ... --label above_median  (another label definition)

Writes outputs/runs/<YYYY-MM-DD>_<tag>/ containing
    run_info.json            master file + sha256, git commit, config snapshot, versions, seed, label used
    label_summary.csv        posts / high / share high per label definition and platform (all posts)
    label_rules.csv          the label rule fit on all posts (thresholds, reach line)
    logistic_fit_stats.csv   per model: n, share high, pseudo R², AIC, converged
    logistic_effects.csv     odds ratio (95% CI) and probability-point effect (95% CI) per feature
    tree_leaves.csv          shallow tree on all posts: rule per leaf, posts, share high
    tree_importance.csv      share of the tree's split gain per feature
    coefficients_penalized.csv  ridge / lasso / elastic net logistic: log-odds per SD (importance), odds
                             ratio per unit, whether kept (nonzero), chosen C and mix
    stability.csv            bootstrap: share of refits keeping each feature, sign agreement, odds-ratio interval
    permutation_scores.csv   drop in held-out AUC per split when a feature / group is shuffled;
                             importance.csv: mean and 2.5–97.5% spread across splits
    cv_scores.csv            every CV split for every model (AUC, balanced accuracy, Brier, log loss);
                             cv_summary.csv: mean and SD per model. Labels are refit on each training fold.
    figures/                 effects, tree leaves, CV performance, stability, importance

Models: base rate (reference), logistic regression, shallow decision tree (step C2); ridge, lasso, elastic
net logistic with nested-CV penalty (step C3). Per platform = main, pooled = check.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import pandas as pd

from modeling import plots
from modeling.classify import (PENALIZED, base_rate_fit_predict, bootstrap_stability_logistic, cv_score_classifier,
                               fit_logit, fit_penalized, logistic_fit_predict, logit_effects, logit_fit_stats,
                               make_tree, penalized_coef_table, penalized_fit_predict,
                               permutation_importance_logistic, tree_fit_predict, tree_importance, tree_leaves)
from modeling.features import DEFAULT_CONFIG, POOLED, REPO_ROOT, build_features, feature_group, file_sha256, \
    load_config, load_master, prepare_posts
from modeling.models import cv_splits, summarize_importance
from modeling.run_regression import git_info, package_versions
from modeling.targets import label_summary, make_label_rule

CLASSIFICATION_METRICS = [("auc", "Held-out ROC AUC (0.5 = chance)", 0.5),
                          ("balanced_accuracy", "Held-out balanced accuracy", 0.5),
                          ("brier", "Held-out Brier score (lower is better)", 0.25)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--tag", required=True, help="short run name, e.g. c2_logistic_tree")
    ap.add_argument("--label", default=None, help="label definition (default: classification.main_label)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run folder with the same name")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    label_name = args.label or config["classification"]["main_label"]
    if label_name not in config["classification"]["labels"]:
        sys.exit(f"Unknown label {label_name!r}; choose from {list(config['classification']['labels'])}")
    out = REPO_ROOT / config["output_dir"] / f"{dt.date.today().isoformat()}_{args.tag}"
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} already exists; use a new --tag or pass --overwrite.")
    (out / "figures").mkdir(parents=True, exist_ok=True)

    posts, li_detail, master_sha = load_master(config)
    prepared = prepare_posts(posts, li_detail, config)
    model_sets = config["platforms"] + [POOLED]
    cv = config["cv"]
    models = [("base rate (reference)", base_rate_fit_predict), ("Logistic", logistic_fit_predict(config)),
              ("Decision tree", tree_fit_predict(config))]
    cls = config["classification"]

    effects, stats_rows, leaves, tree_imp, cv_rows, rules = [], [], [], [], [], []
    pen_coefs, stab_rows, perm_rows = [], [], []
    for p in model_sets:
        fs = build_features(prepared, p, config)
        rule = make_label_rule(label_name, config).fit(fs)
        y = rule.label(fs)
        if p != POOLED:
            rules.append(rule.describe())
        res = fit_logit(fs, y)
        effects.append(logit_effects(res, fs, y))
        stats_rows.append(logit_fit_stats(res, fs, y))
        tree = make_tree(config).fit(fs.X.to_numpy(), y)
        leaves.append(tree_leaves(tree, fs, y))
        tree_imp.append(tree_importance(tree, fs))
        splits = cv_splits(fs, cv["n_splits"], cv["n_repeats"], config["seed"])
        perm_fits = []                       # the permutation model's per-split fits, reused for importance
        pen = [(n, penalized_fit_predict(n, config, perm_fits if n == cls["permutation"]["model"] else None))
               for n in PENALIZED]
        for name, fp in models + pen:
            cv_rows.append(cv_score_classifier(fs, splits, label_name, config, fp).assign(platform=p, model=name))
        for name in PENALIZED:
            pipe = fit_penalized(fs, y, name, config, splits)
            pen_coefs.append(penalized_coef_table(pipe, fs, name, config))
            if name in cls["stability"]["models"]:
                stab_rows.append(bootstrap_stability_logistic(fs, y, name, pipe, config))
        perm_rows.append(permutation_importance_logistic(fs, splits, label_name, cls["permutation"]["model"],
                                                         config, feature_group, fitted=perm_fits))
        print(f"  {p}: done", flush=True)

    effects = pd.concat(effects, ignore_index=True)
    leaves = pd.concat(leaves, ignore_index=True)
    cv_scores = pd.concat(cv_rows, ignore_index=True)
    metric_cols = ["auc", "balanced_accuracy", "brier", "log_loss"]
    cv_summary = cv_scores.groupby(["platform", "model"], sort=False)[metric_cols].agg(["mean", "std"]).round(3)
    cv_summary.columns = [f"{m}_{s}" for m, s in cv_summary.columns]
    cv_summary["splits_with_fit_warning"] = cv_scores.groupby(["platform", "model"], sort=False)["fit_warning"].sum()

    label_summary(build_features(prepared, POOLED, config), config).to_csv(out / "label_summary.csv", index=False)
    pd.concat(rules, ignore_index=True).to_csv(out / "label_rules.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(out / "logistic_fit_stats.csv", index=False)
    effects.to_csv(out / "logistic_effects.csv", index=False)
    leaves.to_csv(out / "tree_leaves.csv", index=False)
    pd.concat(tree_imp, ignore_index=True).to_csv(out / "tree_importance.csv", index=False)
    pd.concat(pen_coefs, ignore_index=True).to_csv(out / "coefficients_penalized.csv", index=False)
    stability = pd.concat(stab_rows, ignore_index=True)
    stability.to_csv(out / "stability.csv", index=False)
    perm = pd.concat(perm_rows, ignore_index=True)
    perm.to_csv(out / "permutation_scores.csv", index=False)
    importance = summarize_importance(perm, metric="auc_drop")
    importance.to_csv(out / "importance.csv", index=False)
    cv_scores.to_csv(out / "cv_scores.csv", index=False)
    cv_summary.reset_index().to_csv(out / "cv_summary.csv", index=False)

    ame = effects.rename(columns={"ame_ci_lo": "ci_lo", "ame_ci_hi": "ci_hi"})
    ame_note = ("Dot = average change in the probability a post is high, in percentage points; line = 95% CI "
                "(robust SEs). A CI crossing 0 means no detectable effect. Associations, not causal effects.")
    ame_x = f"Change in P(high), pts  [label: {label_name}]"
    plots.coef_plot(ame, config["platforms"], "Logistic regression: how each feature shifts the chance of high "
                    "engagement, within each platform", out / "figures" / "logistic_effects_by_platform.png",
                    value="ame_pts", xlabel=ame_x, note=ame_note)
    plots.coef_plot(ame, [POOLED], "Logistic regression, pooled check (platform intercepts, SEs clustered by "
                    "content)", out / "figures" / "logistic_effects_pooled.png", value="ame_pts", xlabel=ame_x,
                    note=ame_note)
    plots.tree_leaves_plot(leaves, config["platforms"], out / "figures" / "tree_leaves_by_platform.png")
    pen_models = ("Lasso logistic", "Elastic net logistic")
    plots.stability_plot(stability, config["platforms"], out / "figures" / "stability_by_platform.png", pen_models)
    plots.stability_plot(stability, [POOLED], out / "figures" / "stability_pooled.png", pen_models)
    perm_model, auc_x = cls["permutation"]["model"], "Drop in held-out AUC when shuffled"
    plots.importance_plot(importance, model_sets, "group", f"Permutation importance by feature group ({perm_model}, "
                          "held-out folds)", out / "figures" / "importance_groups.png", "auc_drop", auc_x)
    plots.importance_plot(importance, config["platforms"], "feature", f"Permutation importance by feature "
                          f"({perm_model}, held-out folds)", out / "figures" / "importance_features.png", "auc_drop",
                          auc_x)
    plots.cv_plot(cv_scores, model_sets, out / "figures" / "cv_performance.png", metrics=CLASSIFICATION_METRICS,
                  title="Cross-validated classification (5-fold × 10 repeats; labels refit per fold; ±1 SD)")

    run_info = {
        "tag": args.tag, "created": dt.datetime.now().isoformat(timespec="seconds"), "label": label_name,
        "master": config["master"], "master_sha256": master_sha, "git": git_info(),
        "packages": package_versions(), "seed": config["seed"], "config": config,
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))
    if file_sha256(REPO_ROOT / config["master"]) != master_sha:
        sys.exit("Master workbook changed during the run; results discarded.")

    print(f"Wrote {out.relative_to(REPO_ROOT)}")
    print(cv_summary[["auc_mean", "auc_std", "balanced_accuracy_mean", "brier_mean",
                      "splits_with_fit_warning"]].to_string())


if __name__ == "__main__":
    main()
