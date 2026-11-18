"""
Stage 2 pipeline shared by the command line (modeling/run_recommend.py) and the web page (index.html, which runs
this file in the browser with Pyodide): read a master, build the recommendations, optionally score a plan, and
write every output file into one folder. See run_recommend.py for the list of files.
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import pandas as pd

from modeling import plots, report
from modeling.features import REPO_ROOT, build_features, file_sha256, load_master, prepare_posts
from modeling.recommend import (fit_scorers, model_quality, performance, plan_specs, recent_check, recent_window,
                                recommendations, score_plan)
from modeling.run_regression import package_versions

TITLE = "Social content recommendations"


def run(out: Path, config: dict, *, master_path: Path | None = None, plan: pd.DataFrame | None = None,
        plan_path: Path | None = None, plan_name: str | None = None, tag: str | None = None, git: dict | None = None,
        run_date: dt.date | None = None, log=print) -> dict:
    """Write all outputs to `out`. `master_path` defaults to config["master"] (relative to the repo); `plan` is the
    read_plan() frame of the file at `plan_path`. Returns master_sha256, meta, summary and the report's sections."""
    run_date = run_date or dt.date.today()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    posts, li_detail, master_sha = load_master(config, master_path)
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
        log(f"  {p}: done")

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
            f"Last 90 days = {start:%b %d} – {end:%b %d, %Y}. Run {run_date:%b %d, %Y}.")
    summary = report.summary_table(opts, recent_opts, quality, notes)
    q_sheet, f_sheet = report.model_sheet(quality, factors)
    report.write_excel(out / "recommendations.xlsx", {
        "Summary": summary,
        "What's working": report.options_sheet(opts),
        "What to test next": report.test_next_sheet(opts),
        "Recent posts": report.recent_posts_sheet(recent_posts),
        "Recent changes": report.recent_changes_sheet(recent_opts),
        "Planned posts": report.plan_sheet(scores),
        "Model check": [q_sheet, f_sheet],
    }, stamp=end)
    report_args = dict(options=opts, recent_opts=recent_opts, quality=quality, figures=figures, scores=scores,
                       platforms=config["platforms"], notes=notes)
    report.write_html(out / "recommendations.html", title=TITLE, meta=meta, **report_args)

    if plan_path is not None and plan_name is None:
        plan_name = str(plan_path.relative_to(REPO_ROOT) if plan_path.is_relative_to(REPO_ROOT) else plan_path)
    run_info = {
        "tag": tag, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "master": config["master"], "master_sha256": master_sha,
        "plan": None if plan_path is None else {"file": plan_name, "sha256": file_sha256(plan_path)},
        "git": git, "packages": package_versions(), "seed": config["seed"], "config": config,
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))
    return {"master_sha256": master_sha, "meta": meta, "summary": summary,
            "sections": report.html_sections(**report_args)}
