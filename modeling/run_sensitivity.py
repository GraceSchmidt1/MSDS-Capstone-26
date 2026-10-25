#!/usr/bin/env python3
"""
Step 5: sensitivity runs. Refit the main specification with one change at a time (variants in
config.yaml → sensitivity.variants) and check whether the conclusions hold.

    python -m modeling.run_sensitivity --tag step5_sensitivity          (from the repo root)

Writes outputs/runs/<YYYY-MM-DD>_<tag>/ containing
    run_info.json                 master + sha256, git commit, config snapshot, versions, seed
    sensitivity_ols.csv           OLS effect (pts) with 95% CI for every variant × model × feature
    sensitivity_regularized.csv   elastic-net effect (pts) and kept/dropped for every variant
    sensitivity_cv.csv            elastic-net held-out MAE / R² per variant (all on the rate scale)
    robustness_summary.csv        per platform × key feature: does the main OLS finding survive each variant?
    variant_notes.csv             what each variant changed (e.g. which post was dropped)
    figures/sensitivity_key_effects.png
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import pandas as pd

from modeling import plots
from modeling.features import (DEFAULT_CONFIG, POOLED, REPO_ROOT, feature_columns, file_sha256, load_config,
                               load_master, prepare_posts)
from modeling.run_regression import git_info, package_versions
from modeling.sensitivity import (fit_ols_variant, ols_effects, prepare_variant, regularized_variant,
                                  robustness_summary)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    sens = config["sensitivity"]
    out = REPO_ROOT / config["output_dir"] / f"{dt.date.today().isoformat()}_{args.tag}"
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} already exists; use a new --tag or pass --overwrite.")
    (out / "figures").mkdir(parents=True, exist_ok=True)

    posts, li_detail, master_sha = load_master(config)
    prepared = prepare_posts(posts, li_detail, config)
    model_sets = config["platforms"] + [POOLED]
    known = set(feature_columns(POOLED, config, sensitivity=True))
    bad = [f for f in sens["key_features"] if not isinstance(f, str) or f not in known]
    if bad:
        sys.exit(f"sensitivity.key_features not recognized (quote them in YAML?): {bad}")

    ols_rows, reg_rows, cv_rows, notes = [], [], [], []
    for variant_name, variant in sens["variants"].items():
        for p in model_sets:
            vd = prepare_variant(prepared, p, variant, config)
            ols_rows.append(ols_effects(fit_ols_variant(vd), vd).assign(variant=variant_name, platform=p))
            scores, coefs = regularized_variant(vd, sens["model"], config)
            reg_rows.append(coefs.assign(variant=variant_name, platform=p, model=sens["model"]))
            cv_rows.append(scores.assign(variant=variant_name, platform=p))
            notes.append({"variant": variant_name, "platform": p, "n_posts": len(vd.fs.y),
                          "n_features": vd.fs.X.shape[1], "weighted": vd.weights is not None,
                          "target_transform": vd.transform or "none",
                          "dropped_master_rows": ", ".join(map(str, vd.dropped_rows))})
        print(f"  {variant_name}: done", flush=True)

    ols = pd.concat(ols_rows, ignore_index=True)
    reg = pd.concat(reg_rows, ignore_index=True)
    cv = pd.concat(cv_rows, ignore_index=True)
    cv_summary = cv.groupby(["platform", "variant"], sort=False)[["mae", "r2"]].agg(["mean", "std"]).round(3)
    cv_summary.columns = [f"{m}_{s}" for m, s in cv_summary.columns]
    robust = robustness_summary(ols, sens["key_features"])

    ols.to_csv(out / "sensitivity_ols.csv", index=False)
    reg.to_csv(out / "sensitivity_regularized.csv", index=False)
    cv_summary.reset_index().to_csv(out / "sensitivity_cv.csv", index=False)
    robust.to_csv(out / "robustness_summary.csv", index=False)
    pd.DataFrame(notes).to_csv(out / "variant_notes.csv", index=False)
    plots.sensitivity_plot(ols, sens["key_features"], list(sens["variants"]), model_sets,
                           out / "figures" / "sensitivity_key_effects.png")

    run_info = {"tag": args.tag, "created": dt.datetime.now().isoformat(timespec="seconds"),
                "master": config["master"], "master_sha256": master_sha, "git": git_info(),
                "packages": package_versions(), "seed": config["seed"], "config": config}
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))
    if file_sha256(REPO_ROOT / config["master"]) != master_sha:
        sys.exit("Master workbook changed during the run; results discarded.")

    print(f"Wrote {out.relative_to(REPO_ROOT)}")
    print(cv_summary.to_string())


if __name__ == "__main__":
    main()
