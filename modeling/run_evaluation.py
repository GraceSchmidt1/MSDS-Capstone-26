#!/usr/bin/env python3
"""
Run the Stage 1 evaluation framework (items 3–7) and write a self-describing run folder.

    python -m modeling.run_evaluation --tag eval_framework          (from the repo root)

Outcomes: rate (engagement rate, pts) and rate_vs_reach (rate minus the expected rate for the post's reach,
from the main label's reach rule). Writes outputs/runs/<YYYY-MM-DD>_<tag>/ containing
    run_info.json            master + sha256, git commit, config, packages, and the source runs read
                             (with their own git commits)
    group_tests.csv          item 4: pillar / ICP differences per platform and outcome. Single-tag posts:
                             ANOVA F, Welch F, Kruskal–Wallis H, η², ω². All posts (post × tag rows):
                             joint Wald χ² with post-clustered SEs. *_bh = Benjamini–Hochberg within outcome
    tukey_pairs.csv          item 4: Tukey HSD for every pair of groups (single-tag posts)
    clustered_se.csv         item 5: naive vs. HC3 vs. clustered SEs (post × tag rows; pooled content pieces)
    cohens_d.csv             item 7: Hedges' g (95% CI) per 0/1 feature; formats vs. the reference format
    partial_r2.csv           item 7: partial R² per feature and feature group (plain OLS refits)
    cv_mean_sd.csv           item 6: held-out metrics as mean ± SD over the CV splits (from the source runs)
    driver_ranking.csv       item 3: per platform × feature: std β, odds ratio, Holm / BH-adjusted p-values,
                             lasso stability, permutation importance, Hedges' g, evidence, rank
    figures/                 Hedges' g by platform (rate vs. reach)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import pandas as pd

from modeling import plots
from modeling.evaluate import (adjust, cohens_d_table, cv_mean_sd, driver_ranking, group_tests, outcomes,
                               partial_r2_table, pooled_se_comparison)
from modeling.features import (DEFAULT_CONFIG, POOLED, REPO_ROOT, build_features, file_sha256, load_config,
                               load_master, prepare_posts)
from modeling.run_regression import git_info, package_versions


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--tag", required=True, help="short run name, e.g. eval_framework")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run folder with the same name")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    ev = config["evaluation"]
    sources = {k: REPO_ROOT / v for k, v in ev["sources"].items()}
    for k, folder in sources.items():
        if not (folder / "run_info.json").exists():
            sys.exit(f"Source run for {k} not found: {folder}")
    out = REPO_ROOT / config["output_dir"] / f"{dt.date.today().isoformat()}_{args.tag}"
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} already exists; use a new --tag or pass --overwrite.")
    (out / "figures").mkdir(parents=True, exist_ok=True)

    posts, li_detail, master_sha = load_master(config)
    prepared = prepare_posts(posts, li_detail, config)

    omnibus, pairs, se_rows, effects, partial = [], [], [], [], []
    for p in config["platforms"] + [POOLED]:
        fs = build_features(prepared, p, config)
        outs = outcomes(fs, config)
        if p == POOLED:
            se_rows.append(pooled_se_comparison(fs))
        else:
            o, pr, se = group_tests(prepared, fs, outs, ev["min_group_posts"])
            omnibus += o
            pairs += pr
            se_rows.append(pd.DataFrame(se))
        effects.append(cohens_d_table(fs, outs, config))
        partial.append(partial_r2_table(fs, outs))
        print(f"  {p}: done", flush=True)

    omnibus = adjust(pd.DataFrame(omnibus), ["anova_p", "welch_p", "kruskal_p", "clustered_p"], ["outcome"])
    effects = pd.concat(effects, ignore_index=True)
    ranking = driver_ranking(sources["regression"], sources["classification"], effects,
                             ev["stable"]["selection_freq"], ev["stable"]["sign_agreement"])

    omnibus.to_csv(out / "group_tests.csv", index=False)
    pd.concat(pairs, ignore_index=True).to_csv(out / "tukey_pairs.csv", index=False)
    pd.concat(se_rows, ignore_index=True).to_csv(out / "clustered_se.csv", index=False)
    effects.to_csv(out / "cohens_d.csv", index=False)
    pd.concat(partial, ignore_index=True).to_csv(out / "partial_r2.csv", index=False)
    cv_mean_sd(sources["regression"], sources["classification"]).to_csv(out / "cv_mean_sd.csv", index=False)
    ranking.to_csv(out / "driver_ranking.csv", index=False)

    g = effects.query("outcome == 'rate_vs_reach'").rename(columns={"hedges_g": "g"})
    plots.coef_plot(g, config["platforms"], "Standardized differences (Hedges' g): rate relative to reach, posts "
                    "with each feature vs. comparison", out / "figures" / "hedges_g_by_platform.png", value="g",
                    xlabel="Hedges' g (SDs)", note="Dot = Hedges' g; line = 95% CI. Formats are compared with "
                    f"{config['formats']['reference']}, other features with posts without them. |g| ≈ 0.2 small, "
                    "0.5 medium, 0.8 large. Associations, not causal effects.")

    run_info = {
        "tag": args.tag, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "master": config["master"], "master_sha256": master_sha, "git": git_info(),
        "sources": {k: {"folder": ev["sources"][k],
                        "git": json.loads((f / "run_info.json").read_text())["git"]} for k, f in sources.items()},
        "packages": package_versions(), "seed": config["seed"], "config": config,
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))
    if file_sha256(REPO_ROOT / config["master"]) != master_sha:
        sys.exit("Master workbook changed during the run; results discarded.")

    print(f"Wrote {out}")
    print(omnibus[["platform", "factor", "outcome", "anova_p", "welch_p", "kruskal_p", "clustered_p",
                   "omega2"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
