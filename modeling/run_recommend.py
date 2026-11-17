#!/usr/bin/env python3
"""
Stage 2: recommendations from how past posts performed, for the marketing team.

    python -m modeling.run_recommend --tag <name>                        (from the repo root)
    python -m modeling.run_recommend --tag <name> --plan my_plan.csv     (also score planned posts)

Reads the master named in modeling/config.yaml (`master:`). After a data update, point `master:` at the new
version and rerun. Writes outputs/runs/<YYYY-MM-DD>_<tag>/ containing
    recommendations.html     one-page summary per platform (open in a browser; works offline)
    recommendations.xlsx     the same, with every table (Summary, What's working, What to test next, Recent posts,
                             Recent changes, Planned posts, Model check)
    options.csv              per platform × choice: posts, share beat, lift, difference vs. baseline (95% CI),
                             Hedges' g, p / BH p, posts needed, verdict, knot check, note
    factor_tests.csv         joint Wald test per factor (format, pillar, ICP, timing, boost), BH within platform
    model_quality.csv        held-out lift R² and AUC of the planned-post scores, and whether they are usable
    recent_posts.csv         posts in the last `recent_days`: rate vs. typical for its reach, standouts
    recent_options.csv       platform and per-choice recent vs. older, trend flags
    plan_input.csv           copy of the plan file (with --plan)
    plan_scores.csv          planned posts: typical rate, expected lift / rate, chance to beat its reach (with --plan)
    figures/                 options_<platform>.png
    run_info.json            master + sha256, plan file + sha256, git commit, config, packages
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from modeling import plots, report
from modeling.features import (DEFAULT_CONFIG, REPO_ROOT, build_features, file_sha256, load_config, load_master,
                               prepare_posts)
from modeling.recommend import (fit_scorers, model_quality, performance, plan_specs, read_plan, recent_check,
                                recent_window, recommendations, score_plan)
from modeling.run_regression import git_info, package_versions


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--tag", required=True, help="short run name, e.g. recommendations")
    ap.add_argument("--plan", default=None, help="CSV of planned posts to score (see modeling/plan_template.csv)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run folder with the same name")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    plan_path = Path(args.plan).resolve() if args.plan else None
    if plan_path and not plan_path.exists():
        sys.exit(f"Plan file not found: {plan_path}")
    plan = read_plan(plan_path) if plan_path else None             # fail early on a malformed plan
    out = REPO_ROOT / config["output_dir"] / f"{dt.date.today().isoformat()}_{args.tag}"
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} already exists; use a new --tag or pass --overwrite.")
    (out / "figures").mkdir(parents=True, exist_ok=True)

    git = git_info()                      # before anything is written, so this run's own files never count
    posts, li_detail, master_sha = load_master(config)
    prepared = prepare_posts(posts, li_detail, config)

    options, factors, quality, recent_posts, recent_opts, scorers, figures = {}, [], [], [], [], {}, {}
    for p in config["platforms"]:
        fs = build_features(prepared, p, config)
        options[p], f = recommendations(prepared, fs, config)
        factors.append(f)
        rp, ro = recent_check(prepared, fs, options[p], config, performance(fs, config))
        recent_posts.append(rp)
        recent_opts.append(ro)
        quality.append(model_quality(fs, config))
        if plan is not None:
            scorers[p] = fit_scorers(fs, config)
        figures[p] = out / "figures" / f"options_{p.lower()}.png"
        plots.options_plot(options[p], p, figures[p])
        print(f"  {p}: done", flush=True)

    opts = pd.concat(options.values(), ignore_index=True)
    factors = pd.concat(factors, ignore_index=True)
    quality = pd.DataFrame(quality)
    recent_posts = pd.concat(recent_posts, ignore_index=True)
    recent_opts = pd.concat(recent_opts, ignore_index=True)
    scores = score_plan(plan_specs(plan, prepared, config), scorers, options, config) if plan is not None else None

    opts.to_csv(out / "options.csv", index=False)
    factors.to_csv(out / "factor_tests.csv", index=False)
    quality.to_csv(out / "model_quality.csv", index=False)
    recent_posts.to_csv(out / "recent_posts.csv", index=False)
    recent_opts.to_csv(out / "recent_options.csv", index=False)
    if scores is not None:
        shutil.copyfile(plan_path, out / "plan_input.csv")
        scores.to_csv(out / "plan_scores.csv", index=False)

    notes = config["recommend"].get("platform_notes") or {}
    start, end = recent_window(prepared, config)
    n = {p: len(build_features(prepared, p, config).y) for p in config["platforms"]}
    meta = (f"Data: {config['master']} · {sum(n.values())} posts ("
            + ", ".join(f"{p} {k}" for p, k in n.items()) + f") through {end:%b %d, %Y}. "
            f"Last 90 days = {start:%b %d} – {end:%b %d, %Y}. Run {dt.date.today():%b %d, %Y}.")
    q_sheet, f_sheet = report.model_sheet(quality, factors)
    report.write_excel(out / "recommendations.xlsx", {
        "Summary": report.summary_table(opts, recent_opts, quality, notes),
        "What's working": report.options_sheet(opts),
        "What to test next": report.test_next_sheet(opts),
        "Recent posts": report.recent_posts_sheet(recent_posts),
        "Recent changes": report.recent_changes_sheet(recent_opts),
        "Planned posts": report.plan_sheet(scores),
        "Model check": [q_sheet, f_sheet],
    }, stamp=end)
    report.write_html(out / "recommendations.html", title="Social content recommendations", meta=meta,
                      options=opts, recent_opts=recent_opts, quality=quality, figures=figures, scores=scores,
                      platforms=config["platforms"], notes=notes)

    run_info = {
        "tag": args.tag, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "master": config["master"], "master_sha256": master_sha,
        "plan": None if plan_path is None else {
            "file": str(plan_path.relative_to(REPO_ROOT) if plan_path.is_relative_to(REPO_ROOT) else plan_path),
            "sha256": file_sha256(plan_path)},
        "git": git, "packages": package_versions(), "seed": config["seed"], "config": config,
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))
    if file_sha256(REPO_ROOT / config["master"]) != master_sha:
        sys.exit("Master workbook changed during the run; results discarded.")

    print(f"Wrote {out}")
    print(f"Open {out / 'recommendations.html'}")
    print(report.summary_table(opts, recent_opts, quality, notes).to_string(index=False))


if __name__ == "__main__":
    main()
