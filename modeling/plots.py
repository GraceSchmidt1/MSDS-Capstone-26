"""
Charts for the modeling outputs, using the same style as the EDA notebook: one panel per platform,
platform identity colors (validated categorical slots), light grid, uncertainty always shown.
"""
from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

PLATFORM_COLOR = {"Facebook": "#2a78d6", "Instagram": "#eb6834", "LinkedIn": "#1baf7a", "pooled": "#52514e"}
INK, INK_2, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.labelcolor": INK_2, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.major.size": 0, "ytick.major.size": 0,
    "text.color": INK, "legend.frameon": False, "lines.linewidth": 2,
})


FEATURE_LABEL = {
    "log2_impressions": "Reach: impressions doubled",
    "WEEKEND": "Timing: weekend",
    "AFTER_11": "Timing: posted 11am or later",
    "BOOSTED": "LinkedIn: also paid-boosted",
    "IN_EXPERIMENT_PERIOD": "Period: Q2–Q3 2026 experiment",
}


def label(feature: str) -> str:
    return FEATURE_LABEL.get(feature, feature)


def coef_plot(coefs: pd.DataFrame, platforms: list[str], title: str, path):
    """Coefficient (pts) with 95% CI per feature, one panel per platform; intercept omitted."""
    c = coefs[(coefs["feature"] != "const") & coefs["platform"].isin(platforms)]
    features = list(dict.fromkeys(c["feature"]))           # union over the plotted platforms, first-seen order
    fig, axes = plt.subplots(1, len(platforms), figsize=(4.6 * len(platforms), 0.36 * len(features) + 1.6),
                             sharey=True, squeeze=False)
    for ax, p in zip(axes[0], platforms):
        d = c[c["platform"] == p].set_index("feature").reindex(features)
        y = np.arange(len(features))
        col = PLATFORM_COLOR[p]
        ax.axvline(0, color=INK_MUTED, lw=1)
        ax.hlines(y, d["ci_lo"], d["ci_hi"], color=col, lw=2, alpha=0.55)
        ax.scatter(d["coef_pts"], y, s=40, color=col, zorder=3, edgecolors=SURFACE, linewidths=1.2)
        for yi, missing in zip(y, d["coef_pts"].isna()):
            if missing:
                ax.annotate("not in model", (0.02, yi), xycoords=("axes fraction", "data"), va="center",
                            fontsize=8, color=INK_MUTED, style="italic")
        ax.set_title(p)
        ax.set_xlabel("Effect on engagement rate (percentage points)")
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(np.arange(len(features)))
    axes[0][0].set_yticklabels([textwrap.shorten(label(f), 44, placeholder="…") for f in features], fontsize=8.5)
    axes[0][0].invert_yaxis()
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.text(0.01, -0.01, "Dot = OLS coefficient; line = 95% CI (robust SEs). A CI crossing 0 means the data "
             "can't distinguish the effect from none. Associations, not causal effects.",
             fontsize=8, color=INK_2)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(path)
    plt.close(fig)


def diagnostics_plot(infl: pd.DataFrame, platforms: list[str], path):
    """Rows: residuals vs fitted, normal Q-Q of studentized residuals, Cook's distance by post."""
    fig, axes = plt.subplots(3, len(platforms), figsize=(4.4 * len(platforms), 9), squeeze=False)
    for j, p in enumerate(platforms):
        d = infl[infl["platform"] == p]
        col = PLATFORM_COLOR[p]
        ax = axes[0][j]
        ax.scatter(d["fitted"], d["resid"], s=16, color=col, alpha=0.75)
        ax.axhline(0, color=INK_MUTED, lw=1)
        ax.set_title(p)
        ax.set_xlabel("Fitted rate (%)")
        ax.set_ylabel("Residual (pts)" if j == 0 else "")

        ax = axes[1][j]
        osm, osr = stats.probplot(d["student_resid"], dist="norm", fit=False)
        ax.scatter(osm, osr, s=16, color=col, alpha=0.75)
        lim = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
        ax.plot(lim, lim, color=INK_MUTED, lw=1)
        ax.set_xlabel("Normal quantile")
        ax.set_ylabel("Studentized residual" if j == 0 else "")

        ax = axes[2][j]
        ax.vlines(np.arange(len(d)), 0, d["cooks_d"], color=col, lw=1.2)
        ax.axhline(4 / len(d), color=INK_MUTED, lw=1, ls="--")
        ax.annotate(f"4/n; {int(d['influential'].sum())} posts above", (0.98, 0.92), xycoords="axes fraction",
                    ha="right", fontsize=8, color=INK_2)
        ax.set_xlabel("Post (model order)")
        ax.set_ylabel("Cook's distance" if j == 0 else "")
    fig.suptitle("OLS diagnostics: residuals, normality, influential posts", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
