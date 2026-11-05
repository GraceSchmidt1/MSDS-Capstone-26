#!/usr/bin/env python3
"""
Run the Stage 1 engagement-rate regression and write a self-describing run folder.

    python -m modeling.run_regression --tag baseline          (from the repo root)

Writes outputs/runs/<YYYY-MM-DD>_<tag>/ containing
    run_info.json            master file + sha256, git commit (+ uncommitted-changes flag), config
                             snapshot, package versions, seed: everything needed to reproduce the run
    fit_stats.csv            per model: n, features, R², adjusted R², AIC, residual SD
    coefficients_ols.csv     effect sizes: coefficient (pts), 95% CI, t statistic, p-value, effect per SD
                             (pts), standardized β, posts with feature
    stability.csv            bootstrap: share of refits keeping each feature, sign agreement, 95% interval
    importance.csv           permutation importance (feature + group): rise in held-out MAE when shuffled
    coefficients_regularized.csv  ridge / lasso / elastic net: coefficient per SD (importance) and in pts,
                             whether kept (nonzero), chosen penalty
    cv_scores.csv            every CV split for every model; cv_summary.csv: mean and SD per model
                             (regularized models scored with nested CV: penalty tuned on training posts only)
    vif.csv                  multicollinearity per feature
    influence.csv            per-post residuals, leverage, Cook's distance
    excluded_posts.csv       posts not modelled, with the reason
    figures/                 coefficient plot, diagnostics

Models: reference (mean), baseline OLS, ridge, lasso, elastic net; per platform = main, pooled = check.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform as py_platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from modeling import plots
from modeling.features import (DEFAULT_CONFIG, POOLED, REPO_ROOT, build_features, file_sha256, load_config,
                               load_master, prepare_posts)
from modeling.models import (REGULARIZED, coef_table, cv_score, cv_splits, fit_ols, fit_regularized, fit_stats,
                             influence_table, ols_fit_predict, reference_fit_predict, regularized_coef_table,
                             regularized_fit_predict, vif_table, bootstrap_stability,
                             permutation_importance_cv, summarize_importance)
from modeling.features import feature_group


def git_info() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain"))}


def package_versions() -> dict:
    import matplotlib, scipy, sklearn, statsmodels, yaml  # noqa: E401
    return {"python": py_platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "statsmodels": statsmodels.__version__, "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "pyyaml": yaml.__version__}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--tag", required=True, help="short run name, e.g. baseline")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run folder with the same name")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    out = REPO_ROOT / config["output_dir"] / f"{dt.date.today().isoformat()}_{args.tag}"
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} already exists; use a new --tag or pass --overwrite.")
    (out / "figures").mkdir(parents=True, exist_ok=True)

    posts, li_detail, master_sha = load_master(config)
    prepared = prepare_posts(posts, li_detail, config)
    model_sets = config["platforms"] + [POOLED]
    cv = config["cv"]

    coefs, stats_rows, cv_rows, vifs, infls, excluded, reg_coefs = [], [], [], [], [], [], []
    stab_rows, perm_rows = [], []
    for p in model_sets:
        fs = build_features(prepared, p, config)
        res = fit_ols(fs)
        coefs.append(coef_table(res, fs))
        stats_rows.append(fit_stats(res, fs))
        vifs.append(vif_table(fs))
        infls.append(influence_table(res, fs))
        excluded.append(fs.excluded)
        splits = cv_splits(fs, cv["n_splits"], cv["n_repeats"], config["seed"])
        for name, fp in [("reference (mean)", reference_fit_predict), ("OLS", ols_fit_predict)]:
            cv_rows.append(cv_score(fs, splits, fp).assign(platform=p, model=name))
        for name in REGULARIZED:
            cv_rows.append(cv_score(fs, splits, regularized_fit_predict(name, config)).assign(platform=p, model=name))
            pipe = fit_regularized(fs, name, config, splits)
            reg_coefs.append(regularized_coef_table(pipe, fs, name))
            if name in config["stability"]["models"]:
                stab_rows.append(bootstrap_stability(fs, name, pipe, config))
        perm_rows.append(permutation_importance_cv(fs, splits, config["permutation"]["model"], config,
                                                   feature_group))
        print(f"  {p}: done", flush=True)

    coefs = pd.concat(coefs, ignore_index=True)
    cv_scores = pd.concat(cv_rows, ignore_index=True)
    cv_summary = (cv_scores.groupby(["platform", "model"], sort=False)[["mae", "rmse", "r2"]]
                  .agg(["mean", "std"]).round(3))
    cv_summary.columns = [f"{m}_{s}" for m, s in cv_summary.columns]
    infl = pd.concat(infls, ignore_index=True)

    reg_coefs = pd.concat(reg_coefs, ignore_index=True)
    coefs.to_csv(out / "coefficients_ols.csv", index=False)
    reg_coefs.to_csv(out / "coefficients_regularized.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(out / "fit_stats.csv", index=False)
    cv_scores.to_csv(out / "cv_scores.csv", index=False)
    cv_summary.reset_index().to_csv(out / "cv_summary.csv", index=False)
    pd.concat(vifs, ignore_index=True).to_csv(out / "vif.csv", index=False)
    infl.to_csv(out / "influence.csv", index=False)
    pd.concat(excluded, ignore_index=True).drop_duplicates().to_csv(out / "excluded_posts.csv", index=False)

    plots.coef_plot(coefs, config["platforms"], "Baseline OLS: effect of each feature, within each platform",
                    out / "figures" / "coef_ols_by_platform.png")
    plots.coef_plot(coefs, [POOLED], "Baseline OLS, pooled check (platform intercepts, SEs clustered by content)",
                    out / "figures" / "coef_ols_pooled.png")
    plots.diagnostics_plot(infl, config["platforms"], out / "figures" / "diagnostics_ols.png")
    plots.coef_compare_plot(coefs, reg_coefs, config["platforms"],
                            "OLS vs. regularized estimates, within each platform",
                            out / "figures" / "coef_compare_by_platform.png")
    plots.coef_compare_plot(coefs, reg_coefs, [POOLED], "OLS vs. regularized estimates, pooled check",
                            out / "figures" / "coef_compare_pooled.png")
    plots.cv_plot(cv_scores, model_sets, out / "figures" / "cv_performance.png")

    stability = pd.concat(stab_rows, ignore_index=True)
    perm = pd.concat(perm_rows, ignore_index=True)
    importance = summarize_importance(perm)
    stability.to_csv(out / "stability.csv", index=False)
    perm.to_csv(out / "permutation_scores.csv", index=False)
    importance.to_csv(out / "importance.csv", index=False)
    plots.stability_plot(stability, config["platforms"], out / "figures" / "stability_by_platform.png")
    plots.stability_plot(stability, [POOLED], out / "figures" / "stability_pooled.png")
    perm_model = config["permutation"]["model"]
    plots.importance_plot(importance, model_sets, "group",
                          f"Permutation importance by feature group ({perm_model}, held-out folds)",
                          out / "figures" / "importance_groups.png")
    plots.importance_plot(importance, config["platforms"], "feature",
                          f"Permutation importance by feature ({perm_model}, held-out folds)",
                          out / "figures" / "importance_features.png")

    run_info = {
        "tag": args.tag, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "master": config["master"], "master_sha256": master_sha, "git": git_info(),
        "packages": package_versions(), "seed": config["seed"], "config": config,
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))
    if file_sha256(REPO_ROOT / config["master"]) != master_sha:
        sys.exit("Master workbook changed during the run; results discarded.")

    print(f"Wrote {out.relative_to(REPO_ROOT)}")
    print(pd.DataFrame(stats_rows)[["platform", "n_posts", "n_features", "r2", "adj_r2"]].round(3).to_string(index=False))
    print(cv_summary[["mae_mean", "r2_mean", "r2_std"]].to_string())


if __name__ == "__main__":
    main()
