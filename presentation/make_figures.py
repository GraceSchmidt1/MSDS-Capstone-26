"""
Charts for the final presentation and report (presentation/img/). Same palette and style as modeling/plots.py, sized
for slides. Reads the current master and the committed runs; writes nothing else. Run from the repo root:
    python -m presentation.make_figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modeling.features import REPO_ROOT, build_features, load_config, load_master, prepare_posts
from modeling.plots import GRID, INK, INK_2, INK_MUTED, PLATFORM_COLOR, SURFACE
from modeling.recommend import performance

RUNS = REPO_ROOT / "outputs" / "runs"
RECS = RUNS / "2026-11-17_recommendations"
EVAL = RUNS / "2026-11-04_eval_framework"
OUT = Path(__file__).parent / "img"
PLATFORMS = ["Facebook", "Instagram", "LinkedIn"]
GOOD, BAD, LEAN = "#1d7a4f", "#b3261e", "#8a5a00"     # report.CSS --good / --bad / --lean
VERDICT_LABEL = {"Do more": ("▲ Do more", GOOD), "Do less": ("▼ Do less", BAD),
                 "Promising, keep testing": ("△ Promising", LEAN), "Watch, tends to underperform": ("▽ Watch", LEAN)}

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 12, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "savefig.dpi": 200})


def _save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def reach(prepared, config):
    """Rate vs impressions, with the reach-expected rate every post is compared with."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, p in zip(axes, PLATFORMS):
        fs = build_features(prepared, p, config)
        perf = performance(fs, config)
        imp = 2 ** fs.X["log2_impressions"]
        ax.scatter(imp, perf["rate"], s=22, color=PLATFORM_COLOR[p], alpha=0.6, edgecolors=SURFACE, linewidths=0.6)
        order = np.argsort(imp.to_numpy())
        ax.plot(imp.to_numpy()[order], perf["expected_rate"].to_numpy()[order], color=INK, lw=2)
        ax.set_xscale("log")
        ax.set_title(p)
        ax.set_xlabel("Impressions (log scale)")
        ax.set_ylim(-2, 70)
    axes[0].set_ylabel("Engagement rate (%)")
    axes[0].annotate("typical rate for the reach", xy=(0.97, 0.9), xycoords="axes fraction", ha="right",
                     color=INK, fontsize=11)
    fig.tight_layout()
    _save(fig, "reach.png")


def formats():
    """Format vs Image after reach: difference in lift, 95% bootstrap CI, with the tool's verdict."""
    o = pd.read_csv(RECS / "options.csv")
    o = o[(o["factor"] == "Format") & (o["compared_with"] == "Image")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    for ax, p in zip(axes, PLATFORMS):
        d = o[o["platform"] == p].sort_values("diff_vs_baseline")
        y = np.arange(len(d))
        ax.axvline(0, color=INK_MUTED, lw=1)
        ax.hlines(y, d["diff_ci_lo"], d["diff_ci_hi"], color=PLATFORM_COLOR[p], lw=3, alpha=0.5)
        ax.scatter(d["diff_vs_baseline"], y, s=70, color=PLATFORM_COLOR[p], zorder=3, edgecolors=SURFACE,
                   linewidths=1.5)
        labels = [f"{f} ({n})" for f, n in zip(d["option"], d["posts"])]
        ax.set_yticks(y, labels)
        for yi, (v, hi) in enumerate(zip(d["verdict"], d["diff_ci_hi"])):
            if v in VERDICT_LABEL:
                text, col = VERDICT_LABEL[v]
                ax.annotate(text, (hi, yi), xytext=(8, 0), textcoords="offset points", va="center", fontsize=11,
                            fontweight="semibold", color=col)
        ax.set_ylim(-0.6, len(d) - 0.4)
        ax.set_title(p)
        ax.set_xlabel("Points vs. Image, after reach")
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(-15, 32)
    fig.tight_layout()
    _save(fig, "formats.png")


def models():
    """How predictable each platform is: held-out R² (rate) and AUC (beats its reach), mean ± 1 SD over 50 splits."""
    cv = pd.read_csv(EVAL / "cv_mean_sd.csv")
    r2 = cv[(cv["metric"] == "r2") & (cv["model"] == "Elastic net")].set_index("platform")
    auc = cv[(cv["metric"] == "auc") & (cv["model"] == "Random forest")].set_index("platform")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, d, ref, label in [(axes[0], r2, 0, "Held-out R², engagement rate (elastic net)"),
                              (axes[1], auc, 0.5, "Held-out AUC, beats its reach (random forest)")]:
        y = np.arange(len(PLATFORMS))
        for yi, p in zip(y, PLATFORMS):
            m, s = d.loc[p, "mean"], d.loc[p, "sd"]
            ax.hlines(yi, m - s, m + s, color=PLATFORM_COLOR[p], lw=3, alpha=0.5)
            ax.scatter(m, yi, s=70, color=PLATFORM_COLOR[p], zorder=3, edgecolors=SURFACE, linewidths=1.5)
            ax.annotate(f"{m + 0.0:.2f}", (m, yi), xytext=(0, -16), textcoords="offset points", ha="center",
                        fontsize=11, color=INK_2)
        ax.axvline(ref, color=INK_MUTED, lw=1)
        ax.annotate("no better than guessing", (ref, -0.55), xytext=(4, 0), textcoords="offset points",
                    fontsize=10, color=INK_MUTED, va="center")
        ax.set_yticks(y, PLATFORMS)
        ax.set_ylim(len(PLATFORMS) - 0.4, -0.8)
        ax.set_title(label)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "models.png")


def quarters(prepared, config):
    """Share of posts that beat their reach, by quarter: Facebook swings, the others stay steadier."""
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for p in PLATFORMS:
        fs = build_features(prepared, p, config)
        perf = performance(fs, config)
        q = pd.PeriodIndex(pd.to_datetime(prepared.set_index("MASTER_ROW").loc[perf.index, "DATE"]), freq="Q")
        s = perf.groupby(q.astype(str))["beat"].agg(["mean", "size"])
        s = s[s["size"] >= 10]
        x = np.arange(len(s))
        ax.plot(x, s["mean"] * 100, color=PLATFORM_COLOR[p], lw=2.5 if p == "Facebook" else 2,
                marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
        nudge = {"Facebook": 9, "LinkedIn": -9}.get(p, 0)          # their Q3 points almost coincide
        ax.annotate(p, (x[-1], s["mean"].iloc[-1] * 100), xytext=(12, nudge), textcoords="offset points",
                    va="center", fontsize=12, color=INK)
        ticks = [t.replace("Q", " Q") for t in s.index]
    ax.axhline(50, color=INK_MUTED, lw=1, ls="--")
    ax.set_xticks(x, ticks)
    ax.set_xlim(-0.3, len(x) - 0.3)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Posts that beat their reach (%)")
    fig.tight_layout()
    _save(fig, "quarters.png")


def main():
    OUT.mkdir(exist_ok=True)
    config = load_config()
    posts, li_detail, _ = load_master(config)
    prepared = prepare_posts(posts, li_detail, config)
    reach(prepared, config)
    formats()
    models()
    quarters(prepared, config)


if __name__ == "__main__":
    main()
