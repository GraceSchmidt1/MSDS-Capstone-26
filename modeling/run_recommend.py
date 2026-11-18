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

The analysis itself is in modeling/pipeline.py, which the web page (index.html) also runs in the browser.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from modeling import pipeline
from modeling.features import DEFAULT_CONFIG, REPO_ROOT, file_sha256, load_config
from modeling.recommend import read_plan
from modeling.run_regression import git_info


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
    git = git_info()                      # before anything is written, so this run's own files never count
    result = pipeline.run(out, config, plan=plan, plan_path=plan_path, tag=args.tag, git=git,
                          log=lambda m: print(m, flush=True))
    if file_sha256(REPO_ROOT / config["master"]) != result["master_sha256"]:
        sys.exit("Master workbook changed during the run; results discarded.")

    print(f"Wrote {out}")
    print(f"Open {out / 'recommendations.html'}")
    print(result["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
