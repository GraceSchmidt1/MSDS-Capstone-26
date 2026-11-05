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


COEF_NOTE = ("Dot = OLS coefficient; line = 95% CI (robust SEs). A CI crossing 0 means the data "
             "can't distinguish the effect from none. Associations, not causal effects.")


def coef_plot(coefs: pd.DataFrame, platforms: list[str], title: str, path, value: str = "coef_pts",
              xlabel: str = "Effect on engagement rate (percentage points)", note: str = COEF_NOTE):
    """Effect (`value`, with ci_lo / ci_hi) per feature, one panel per platform; intercept omitted."""
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
        ax.scatter(d[value], y, s=40, color=col, zorder=3, edgecolors=SURFACE, linewidths=1.2)
        estimable = (c["platform"] == p).any()
        for yi, missing in zip(y, d[value].isna()):
            if missing and estimable:
                ax.annotate("not in model", (0.02, yi), xycoords=("axes fraction", "data"), va="center",
                            fontsize=8, color=INK_MUTED, style="italic")
        if not estimable:
            ax.text(0.5, 0.5, "not estimable:\nperfect separation\n(see logistic_fit_stats.csv)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9, color=INK_MUTED, style="italic")
        ax.set_title(p if estimable else f"{p} (not estimable)")
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(np.arange(len(features)))
    axes[0][0].set_yticklabels([textwrap.shorten(label(f), 44, placeholder="…") for f in features], fontsize=8.5)
    axes[0][0].invert_yaxis()
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.text(0.01, -0.01, note, fontsize=8, color=INK_2)
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


MODEL_MARKER = {"Ridge": ("D", "none"), "Lasso": ("^", "none"), "Elastic net": ("s", "full")}
LOGISTIC_MARKER = {f"{name} logistic": m for name, m in MODEL_MARKER.items()}   # classification (step C3)


def coef_compare_plot(ols: pd.DataFrame, reg: pd.DataFrame, platforms: list[str], title: str, path):
    """OLS coefficient + 95% CI (platform color) beside ridge / lasso / elastic net estimates (gray shapes)."""
    o = ols[(ols["feature"] != "const") & ols["platform"].isin(platforms)]
    r = reg[reg["platform"].isin(platforms)]
    features = list(dict.fromkeys(o["feature"]))
    fig, axes = plt.subplots(1, len(platforms), figsize=(4.6 * len(platforms), 0.4 * len(features) + 1.8),
                             sharey=True, squeeze=False)
    offsets = {"OLS": -0.27, "Ridge": -0.09, "Lasso": 0.09, "Elastic net": 0.27}
    for ax, p in zip(axes[0], platforms):
        y = np.arange(len(features))
        col = PLATFORM_COLOR[p]
        ax.axvline(0, color=INK_MUTED, lw=1)
        d = o[o["platform"] == p].set_index("feature").reindex(features)
        ax.hlines(y + offsets["OLS"], d["ci_lo"], d["ci_hi"], color=col, lw=2, alpha=0.5)
        ax.scatter(d["coef_pts"], y + offsets["OLS"], s=34, color=col, zorder=3, label="OLS (95% CI)")
        for name, (marker, fill) in MODEL_MARKER.items():
            m = r[(r["platform"] == p) & (r["model"] == name)].set_index("feature").reindex(features)
            ax.scatter(m["coef_pts"], y + offsets[name], s=30, marker=marker, zorder=4, linewidths=1.2,
                       edgecolors=INK_2, facecolors=INK_2 if fill == "full" else "none", label=name)
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
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles[0] = plt.Line2D([], [], color=INK_MUTED, marker="o", lw=2, label="OLS (95% CI), platform color")
    labels[0] = handles[0].get_label()
    fig.legend(handles, labels, loc="upper right", ncol=4, fontsize=8.5, bbox_to_anchor=(0.995, 1.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.text(0.01, -0.01, "Regularized estimates are shrunk toward 0; a lasso / elastic-net point at 0 means the "
             "feature was dropped. Each panel has its own x-scale.", fontsize=8, color=INK_2)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(path)
    plt.close(fig)


REGRESSION_METRICS = [("r2", "Held-out R² (higher is better)", 0),
                      ("mae", "Held-out MAE, pts (lower is better)", None)]


def cv_plot(cv_scores: pd.DataFrame, platforms: list[str], path, metrics=REGRESSION_METRICS,
            title: str = "Cross-validated performance (5-fold × 10 repeats; ±1 SD across splits)"):
    """Held-out scores per model: mean across splits with ±1 SD, one panel row per (metric, label, ref line)."""
    models = list(dict.fromkeys(cv_scores["model"]))
    fig, axes = plt.subplots(len(metrics), len(platforms), figsize=(4.4 * len(platforms), 2.8 * len(metrics)),
                             squeeze=False)
    for j, p in enumerate(platforms):
        d = cv_scores[cv_scores["platform"] == p].groupby("model", sort=False)
        for i, (metric, xlabel, ref) in enumerate(metrics):
            ax = axes[i][j]
            mean, sd = d[metric].mean().reindex(models), d[metric].std().reindex(models)
            yy = np.arange(len(models))
            ax.hlines(yy, mean - sd, mean + sd, color=PLATFORM_COLOR[p], lw=2, alpha=0.5)
            ax.scatter(mean, yy, s=40, color=PLATFORM_COLOR[p], zorder=3)
            for yi, v in zip(yy, mean):
                ax.annotate(f"{round(v, 2) + 0.0:.2f}", (v, yi), xytext=(0, -12),   # + 0.0: no "-0.00"
                            textcoords="offset points", ha="center",
                            fontsize=8, color=INK_2)
            if ref is not None:
                ax.axvline(ref, color=INK_MUTED, lw=1)
            if i == 0:
                ax.set_title("Pooled (check)" if p == "pooled" else p)
            ax.set_yticks(yy, models if j == 0 else [""] * len(models))
            ax.set_ylim(len(models) - 0.3, -0.6)        # inverted, with room for the labels below each dot
            ax.set_xlabel(xlabel)
            ax.grid(axis="y", visible=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def stability_plot(stab: pd.DataFrame, platforms: list[str], path, models=("Lasso", "Elastic net")):
    """Share of 500 bootstrap refits that keep each feature (`models`: lasso, elastic net), one panel per platform."""
    lasso, enet = models
    s = stab[stab["platform"].isin(platforms) & stab["model"].isin(models)]
    features = list(dict.fromkeys(s["feature"]))
    fig, axes = plt.subplots(1, len(platforms), figsize=(4.4 * len(platforms), 0.36 * len(features) + 1.8),
                             sharey=True, squeeze=False)
    for ax, p in zip(axes[0], platforms):
        y = np.arange(len(features))
        for x0 in (0.5, 0.8):
            ax.axvline(x0, color=INK_MUTED, lw=1, ls=":" if x0 == 0.5 else "--")
        for name, off in [(lasso, -0.13), (enet, 0.13)]:
            d = s[(s["platform"] == p) & (s["model"] == name)].set_index("feature").reindex(features)
            marker, fill = {**MODEL_MARKER, **LOGISTIC_MARKER}[name]
            ax.hlines(y + off, 0, d["selection_freq"], color=PLATFORM_COLOR[p], lw=2, alpha=0.45)
            ax.scatter(d["selection_freq"], y + off, s=30, marker=marker, zorder=3, linewidths=1.2,
                       edgecolors=INK_2, facecolors=INK_2 if fill == "full" else "none", label=name)
            for yi, missing in zip(y, d["selection_freq"].isna()):
                if missing and name == lasso:
                    ax.annotate("not in model", (0.02, yi), xycoords=("axes fraction", "data"), va="center",
                                fontsize=8, color=INK_MUTED, style="italic")
        en_mix = s.loc[(s["platform"] == p) & (s["model"] == enet), "l1_ratio"]
        if len(en_mix) and en_mix.iloc[0] < 0.5:
            ax.annotate(f"Elastic net chose a ridge-like mix (l1 = {en_mix.iloc[0]:.1f}),\nso it keeps nearly "
                        "everything: read Lasso here", (0.98, 0.005), xycoords="axes fraction", ha="right",
                        va="bottom", fontsize=8, color=INK_2, style="italic",
                        bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=GRID))
        ax.set_xlim(0, 1.02)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title("Pooled (check)" if p == "pooled" else p)
        ax.set_xlabel("Share of bootstrap refits keeping the feature")
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(np.arange(len(features)))
    axes[0][0].set_yticklabels([textwrap.shorten(label(f), 44, placeholder="…") for f in features], fontsize=8.5)
    axes[0][0].invert_yaxis()
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=8.5, bbox_to_anchor=(0.995, 1.0))
    fig.suptitle("Stability: how often each feature survives 500 bootstrap refits", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold")
    fig.text(0.01, -0.01, "Dashed line = 80% (stable); dotted = 50%. A feature kept in most refits is a robust "
             "signal; one kept rarely is noise at this sample size.", fontsize=8, color=INK_2)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(path)
    plt.close(fig)


def importance_plot(imp: pd.DataFrame, platforms: list[str], level: str, title: str, path, metric: str = "mae_rise",
                    xlabel: str = "Rise in held-out MAE when shuffled (pts)"):
    """Permutation importance: change in a held-out score (`metric`, e.g. MAE rise) when a feature / group is
    shuffled; ±95% over splits."""
    d_all = imp[(imp["level"] == level) & imp["platform"].isin(platforms)]
    units = list(dict.fromkeys(d_all.sort_values(f"{metric}_mean", ascending=False)["unit"]))
    fig, axes = plt.subplots(1, len(platforms), figsize=(4.4 * len(platforms), 0.38 * len(units) + 1.6),
                             sharey=True, squeeze=False)
    for ax, p in zip(axes[0], platforms):
        d = d_all[d_all["platform"] == p].set_index("unit").reindex(units)
        y = np.arange(len(units))
        ax.axvline(0, color=INK_MUTED, lw=1)
        ax.hlines(y, d[f"{metric}_lo"], d[f"{metric}_hi"], color=PLATFORM_COLOR[p], lw=2, alpha=0.5)
        ax.scatter(d[f"{metric}_mean"], y, s=40, color=PLATFORM_COLOR[p], zorder=3)
        for yi, missing in zip(y, d[f"{metric}_mean"].isna()):
            if missing:
                ax.annotate("not in model", (0.02, yi), xycoords=("axes fraction", "data"), va="center",
                            fontsize=8, color=INK_MUTED, style="italic")
        ax.set_title("Pooled (check)" if p == "pooled" else p)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(np.arange(len(units)))
    axes[0][0].set_yticklabels([textwrap.shorten(label(u), 44, placeholder="…") for u in units], fontsize=8.5)
    axes[0][0].invert_yaxis()
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.text(0.01, -0.01, "Dot = mean over 50 held-out folds; line = 2.5–97.5% across folds. Near 0 = the model "
             "doesn't rely on it. Each panel has its own x-scale.", fontsize=8, color=INK_2)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(path)
    plt.close(fig)


VARIANT_MARKER = {"experiment_period": "s", "impression_weighted": "D", "log1p_target": "^",
                  "no_reach_control": "v", "drop_most_influential": "P"}


def sensitivity_plot(ols: pd.DataFrame, key_features: list[str], variants: list[str], platforms: list[str], path):
    """Key OLS effects (pts, 95% CI) under each sensitivity variant; main spec in the platform color."""
    d_all = ols[ols["feature"].isin(key_features) & ols["platform"].isin(platforms)]
    features = [f for f in key_features if f in set(d_all["feature"])]
    n_v = len(variants)
    fig, axes = plt.subplots(1, len(platforms), figsize=(4.2 * len(platforms), 0.62 * len(features) * n_v / 3 + 2),
                             sharey=True, squeeze=False)
    step = 0.8 / n_v
    for ax, p in zip(axes[0], platforms):
        ax.axvline(0, color=INK_MUTED, lw=1)
        for k, v in enumerate(variants):
            d = d_all[(d_all["platform"] == p) & (d_all["variant"] == v)].set_index("feature").reindex(features)
            y = np.arange(len(features)) - 0.4 + step * (k + 0.5)
            is_main = v == "main"
            col = PLATFORM_COLOR[p] if is_main else INK_2
            ax.hlines(y, d["ci_lo"], d["ci_hi"], color=col, lw=2 if is_main else 1.2, alpha=0.55)
            ax.scatter(d["coef_pts"], y, s=34 if is_main else 22, color=col, zorder=3,
                       marker="o" if is_main else VARIANT_MARKER.get(v, "o"), label=v.replace("_", " "))
        for yi, f in enumerate(features):
            if d_all[(d_all["platform"] == p) & (d_all["feature"] == f)].empty:
                ax.annotate("not in model", (0.02, yi), xycoords=("axes fraction", "data"), va="center",
                            fontsize=8, color=INK_MUTED, style="italic")
        ax.set_title("Pooled (check)" if p == "pooled" else p)
        ax.set_xlabel("Effect on engagement rate (pts)")
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(np.arange(len(features)))
    axes[0][0].set_yticklabels([textwrap.shorten(label(f), 34, placeholder="…") for f in features], fontsize=8.5)
    axes[0][0].set_ylim(len(features) - 0.5, -0.5)
    handles, labels = axes[0][-1].get_legend_handles_labels()
    handles[0] = plt.Line2D([], [], color=INK_MUTED, marker="o", lw=2, label="main (platform color)")
    labels[0] = handles[0].get_label()
    fig.legend(handles, labels, loc="upper right", ncol=len(variants), fontsize=8, bbox_to_anchor=(0.995, 1.0))
    fig.suptitle("Sensitivity: key OLS effects under one-change-at-a-time variants", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold", y=1.04)
    fig.text(0.01, -0.02, "Dot = effect, line = 95% CI. log1p effects are converted to pts at the platform mean. "
             "'no reach control' has no reach row by design. Each panel has its own x-scale.",
             fontsize=8, color=INK_2)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(path)
    plt.close(fig)


def tree_leaves_plot(leaves: pd.DataFrame, platforms: list[str], path):
    """Each leaf of the shallow tree: share of its posts that are high, with a 95% Wilson interval."""
    from statsmodels.stats.proportion import proportion_confint
    fig, axes = plt.subplots(len(platforms), 1, squeeze=False,
                             figsize=(9, sum(0.42 * (leaves["platform"] == p).sum() + 0.9 for p in platforms)),
                             gridspec_kw={"height_ratios": [(leaves["platform"] == p).sum() + 1 for p in platforms]})
    for ax, p in zip(axes[:, 0], platforms):
        d = leaves[leaves["platform"] == p].sort_values("share_high", ascending=False)
        lo, hi = proportion_confint(d["high"], d["posts"], method="wilson")
        y = np.arange(len(d))
        ax.axvline(0.5, color=INK_MUTED, lw=1, ls=":")
        ax.hlines(y, lo, hi, color=PLATFORM_COLOR[p], lw=2, alpha=0.5)
        ax.scatter(d["share_high"], y, s=40, color=PLATFORM_COLOR[p], zorder=3)
        for yi, (_, r) in zip(y, d.iterrows()):
            ax.annotate(f"{r['high']}/{r['posts']}", (1.0, yi), xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK_2)
        ax.set_yticks(y, ["\n".join(textwrap.wrap(" AND ".join(label(c.removeprefix("not ")) if not c.startswith("not ")
                                                             else "not " + label(c[4:]) for c in r.split(" AND ")), 60))
                          for r in d["rule"]], fontsize=8)
        ax.set_ylim(len(d) - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title("Pooled (check)" if p == "pooled" else p)
        ax.grid(axis="y", visible=False)
    axes[-1, 0].set_xlabel("Share of the leaf's posts labelled high")
    fig.suptitle("Shallow decision tree (all posts): the groups it splits posts into", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold")
    fig.text(0.01, -0.01, "Dot = share high, line = 95% Wilson interval, right = high / posts. Dotted line = 50% "
             "(the overall share). Rules describe these posts; CV scores say how well they generalize.",
             fontsize=8, color=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
