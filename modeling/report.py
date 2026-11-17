"""
Stage 2 reports for the marketing team, built from the tables in modeling/recommend.py:

    recommendations.xlsx    Summary · What's working · What to test next · Recent posts · Recent changes ·
                            Planned posts · Model check
    recommendations.html    one page per run: the summary per platform, stat tiles for the recent check, the
                            option chart, planned posts, and how to read it (works offline; images embedded)

Every number in the reports is also in the run folder's CSVs.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import numpy as np
import pandas as pd

ACTION_ORDER = ["Do more", "Do less", "Promising, keep testing", "Watch, tends to underperform"]
FIRM = {"Do more", "Do less"}
FACTOR_WORD = {"Content pillar": "pillar", "ICP segment": "ICP segment"}

HOW_TO_READ = [
    ("Typical rate for its reach", "Posts with fewer impressions usually have higher engagement rates. Every "
     "post is compared with the usual rate for a post of its reach on the same platform, so a choice isn't "
     "credited or blamed for reach."),
    ("Lift", "Rate minus the typical rate for its reach, in percentage points (pts)."),
    ("Difference vs. baseline", "How much higher or lower the lift is for posts with the choice. Formats are "
     "compared with Image; weekend with weekday; 11am or later with before 11am; boosted with not boosted; "
     "pillars and segments with posts without them."),
    ("Do more / Do less", "The whole factor (e.g. format) makes a difference, the choice clearly differs from "
     "its baseline after correcting for the number of comparisons, and this holds however reach is modelled."),
    ("Promising / Watch", "Leaning one way, but not yet conclusive. Keep testing."),
    ("No clear difference", "Choose on strategy; the data doesn't show it changing engagement."),
    ("Not enough posts yet", "Fewer than 10 posts with the choice (or its baseline)."),
    ("Posts needed", "Roughly how many more posts with the choice would settle it, if the current difference "
     "is real (80% power). '200+' means too small to settle soon."),
    ("Caveat", "These are associations in past posts, not proof of cause. Other choices made on the same "
     "posts are not held fixed."),
]


# ─────────────────────────────────────────────────────────────────────────────
# Plain-language tables
# ─────────────────────────────────────────────────────────────────────────────
def option_name(factor: str, option: str) -> str:
    return option if factor in ("Format", "Timing", "Paid boost") else f"{option} ({FACTOR_WORD[factor]})"


def _pts(v) -> str:
    return "–" if pd.isna(v) else f"{v:+.1f} pts"


def _needed(v) -> str:
    return "–" if pd.isna(v) else ("200+" if np.isinf(v) else f"{int(v)}")


def finding_line(r) -> str:
    """e.g. 'Document: +14.1 pts vs Image (13 posts, 77% beat their reach)'."""
    vs = r["compared_with"]
    vs = "posts without it" if vs == "posts without it" else vs
    return (f"{option_name(r['factor'], r['option'])}: {_pts(r['diff_vs_baseline'])} vs {vs} "
            f"({r['posts']} posts, {r['share_beat']:.0%} beat their reach)")


def summary_table(options: pd.DataFrame, recent_opts: pd.DataFrame, quality: pd.DataFrame,
                  notes: dict[str, str] | None = None) -> pd.DataFrame:
    """Key findings per platform; `notes` (config recommend.platform_notes) come first as a "Note" row."""
    rows = []
    for p, g in options.groupby("platform", sort=False):
        q = quality.set_index("platform").loc[p]
        allp = recent_opts[(recent_opts["platform"] == p) & (recent_opts["factor"] == "All posts")].iloc[0]
        if (notes or {}).get(p):
            rows.append({"Platform": p, "Type": "Note", "Finding": notes[p], "Detail": ""})
        rows.append({"Platform": p, "Type": "Scores", "Finding": f"Planned-post scores are {q['reliability']}",
                     "Detail": f"held-out AUC {q['auc_mean']:.2f}, lift R² {q['lift_r2_mean']:.2f}"})
        rows.append({"Platform": p, "Type": "Last 90 days", "Finding": allp["trend"],
                     "Detail": f"{allp['recent_share_beat']:.0%} of {allp['recent_posts']} recent posts beat their "
                               f"reach (before: {allp['older_share_beat']:.0%})"})
        acts = g[g["verdict"].isin(ACTION_ORDER)].copy()
        acts["o"] = acts["verdict"].map(ACTION_ORDER.index)
        for _, r in acts.sort_values(["o", "p_value"]).iterrows():
            rows.append({"Platform": p, "Type": r["verdict"], "Finding": finding_line(r), "Detail": r["note"]})
        trends = recent_opts[(recent_opts["platform"] == p)
                             & recent_opts["trend"].isin(["Pulling ahead recently", "Falling behind recently"])]
        for _, r in trends.iterrows():
            rows.append({"Platform": p, "Type": r["trend"], "Finding": option_name(r["factor"], r["option"]),
                         "Detail": f"gap to other posts changed by {_pts(r['gap_change'])} "
                                   f"({r['recent_posts']} recent posts)"})
    return pd.DataFrame(rows)


def options_sheet(options: pd.DataFrame) -> pd.DataFrame:
    o = options[options["compared_with"] != "–"]
    return pd.DataFrame({
        "Platform": o["platform"], "Factor": o["factor"], "Choice": o["option"], "Verdict": o["verdict"],
        "Compared with": o["compared_with"], "Posts": o["posts"], "Baseline posts": o["comparison_posts"],
        "Share beat their reach": o["share_beat"], "Avg lift (pts)": o["mean_lift"],
        "Difference vs baseline (pts)": o["diff_vs_baseline"], "Low (95%)": o["diff_ci_lo"],
        "High (95%)": o["diff_ci_hi"], "Effect size (g)": o["hedges_g"], "p": o["p_value"],
        "p adjusted": o["p_bh"], "Factor p adjusted": o["factor_p_bh"],
        "Posts needed": o["posts_needed"].map(_needed), "Note": o["note"]})


def test_next_sheet(options: pd.DataFrame) -> pd.DataFrame:
    """Unsettled choices that look worth settling: leaning one way, or untested, ordered by posts needed."""
    o = options[options["verdict"].isin(["Promising, keep testing", "Watch, tends to underperform",
                                         "Not enough posts yet", "No clear difference"])
                & np.isfinite(options["posts_needed"].fillna(np.inf))
                & (options["posts_needed"] > 0)].copy()
    o["o"] = o["verdict"].map({"Promising, keep testing": 0, "Watch, tends to underperform": 1,
                               "Not enough posts yet": 2, "No clear difference": 3})
    o = o.sort_values(["platform", "o", "posts_needed"])
    return pd.DataFrame({
        "Platform": o["platform"], "Choice": [option_name(f, c) for f, c in zip(o["factor"], o["option"])],
        "Verdict now": o["verdict"], "Posts so far": o["posts"], "Difference vs baseline (pts)": o["diff_vs_baseline"],
        "About how many more posts would settle it": o["posts_needed"].astype(int)})


def recent_posts_sheet(posts: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Platform": posts["platform"], "Date": posts["date"].dt.date, "Title": posts["title"],
        "Format": posts["format"], "Pillars": posts["pillars"], "ICP segments": posts["icp_segments"],
        "Impressions": posts["impressions"], "Rate (%)": posts["rate"], "Typical for its reach (%)": posts["typical_rate"],
        "Lift (pts)": posts["lift"], "Result": posts["result"], "Standout": posts["standout"],
        "Master row": posts["master_row"]})


def recent_changes_sheet(recent_opts: pd.DataFrame) -> pd.DataFrame:
    r = recent_opts
    return pd.DataFrame({
        "Platform": r["platform"], "Choice": [option_name(f, o) if f != "All posts" else "All posts"
                                              for f, o in zip(r["factor"], r["option"])],
        "Trend": r["trend"], "Verdict (all posts)": r["verdict"], "Recent posts": r["recent_posts"],
        "Recent share beat": r["recent_share_beat"], "Recent avg lift (pts)": r["recent_mean_lift"],
        "Older posts": r["older_posts"], "Older share beat": r["older_share_beat"],
        "Older avg lift (pts)": r["older_mean_lift"], "Change in gap to other posts (pts)": r["gap_change"],
        "p adjusted": r["p_bh"]})


def plan_sheet(scores: pd.DataFrame | None) -> pd.DataFrame:
    if scores is None:
        return pd.DataFrame({"Planned posts": ["No plan given. Run with --plan <file.csv> "
                                               "(template: modeling/plan_template.csv) to score planned posts."]})
    s = scores
    return pd.DataFrame({
        "Post": s["post_id"], "Platform": s["platform"], "Date": s["date"], "Time": s["time"], "Format": s["format"],
        "Pillars": s["pillars"], "ICP segments": s["icp_segments"], "Impressions used": s["impressions_used"],
        "Impressions from": s["impressions_source"], "Typical rate for its reach (%)": s.get("typical_rate"),
        "Expected rate (%)": s.get("expected_rate"), "Expected rate low": s.get("expected_rate_lo"),
        "Expected rate high": s.get("expected_rate_hi"), "Chance to beat its reach": s.get("chance_beat"),
        "Chance low": s.get("chance_beat_lo"), "Chance high": s.get("chance_beat_hi"),
        "What applies": s["notes"], "Problem with this row": s["error"]})


def model_sheet(quality: pd.DataFrame, factors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = quality.rename(columns={"posts": "Posts", "lift_r2_mean": "Lift R² (held-out)", "lift_r2_sd": "± SD",
                                "auc_mean": "AUC (held-out)", "auc_sd": "AUC ± SD", "reliability": "Scores are",
                                "platform": "Platform", "lift_model": "Lift model", "beat_model": "Beat model"})
    f = factors.rename(columns={"platform": "Platform", "factor": "Factor", "wald_chi2": "Wald χ²", "df": "df",
                                "p_value": "p", "p_bh": "p adjusted"})
    return q, f


# ─────────────────────────────────────────────────────────────────────────────
# Excel
# ─────────────────────────────────────────────────────────────────────────────
PERCENT_COLS = ("Share beat", "share beat", "Chance")


def _number_format(col: str, values: pd.Series) -> str | None:
    if any(k in col for k in PERCENT_COLS):
        return "0%"
    if not pd.api.types.is_float_dtype(values):
        return None
    words = col.split()
    precise = "p" in words or any(k in col for k in ("R²", "AUC", "χ²", "SD"))
    return "0.000" if precise else "0.0"


def _format_block(ws, df: pd.DataFrame, start: int):
    """Header style and number formats for one table whose header is on sheet row start + 1."""
    from openpyxl.styles import Alignment, Font, PatternFill

    head = PatternFill("solid", fgColor="EDEDEA")
    for cell in ws[start + 1][:len(df.columns)]:
        cell.font = Font(bold=True)
        cell.fill = head
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for j, col in enumerate(df.columns, start=1):
        fmt = _number_format(col, df[col])
        if fmt:
            for row in ws.iter_rows(min_row=start + 2, max_row=start + 1 + len(df), min_col=j, max_col=j):
                row[0].number_format = fmt


def _column_widths(ws, frames: list[pd.DataFrame]):
    from openpyxl.utils import get_column_letter

    for j in range(1, max(len(f.columns) for f in frames) + 1):
        texts = [str(c) for f in frames if j <= len(f.columns) for c in [f.columns[j - 1]]
                 + list(f.iloc[:200, j - 1])]
        ws.column_dimensions[get_column_letter(j)].width = min(max(max(map(len, texts)) + 2, 9), 60)


def _pin_timestamps(path: Path, stamp: pd.Timestamp):
    """
    openpyxl stamps the save time into docProps/core.xml and the zip entries; replace both with `stamp` so the
    same data gives a byte-identical workbook.
    """
    import re
    import zipfile

    iso = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    with zipfile.ZipFile(path) as z:
        members = [(i, z.read(i.filename)) for i in z.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for info, data in members:
            if info.filename == "docProps/core.xml":
                data = re.sub(rb"(<dcterms:(?:created|modified)[^>]*>)[^<]*", rb"\g<1>" + iso.encode(), data)
            pinned = zipfile.ZipInfo(info.filename, date_time=stamp.timetuple()[:6])
            pinned.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(pinned, data)


def write_excel(path: Path, sheets: dict[str, pd.DataFrame | list[pd.DataFrame]], stamp: pd.Timestamp):
    """Write the sheets (a list = several tables stacked on one sheet); file timestamps pinned to `stamp`."""
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, frames in sheets.items():
            frames = frames if isinstance(frames, list) else [frames]
            start = 0
            for df in frames:
                df.to_excel(xw, sheet_name=name, index=False, startrow=start)
                _format_block(xw.sheets[name], df, start)
                start += len(df) + 3
            _column_widths(xw.sheets[name], frames)
            xw.sheets[name].freeze_panes = "A2"
    _pin_timestamps(path, stamp)


# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#6b6a65;--line:#e6e5e0;
--good:#1d7a4f;--bad:#b3261e;--lean:#8a5a00;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1d1d1c;--ink:#f2f1ec;--ink2:#c3c2bc;--muted:#8f8e88;
--line:#34332f;--good:#5fc596;--bad:#f2867e;--lean:#e0b25a;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 12px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:20px 0 8px;color:var(--ink2)}.meta{color:var(--ink2);font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(220px,100%),1fr));gap:12px;margin:12px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.tile .n{font-size:28px;font-weight:600;line-height:1.1}.tile .l{color:var(--ink2);font-size:13px;margin-top:4px}
ul.find{list-style:none;padding:0;margin:0}ul.find li{padding:8px 0;border-bottom:1px solid var(--line)}
.tag{display:inline-block;min-width:220px;margin-right:10px;font-weight:600;font-size:13px}
.tag.good{color:var(--good)}.tag.bad{color:var(--bad)}.tag.lean{color:var(--lean)}
.note{color:var(--muted);font-size:13px}.platform-note{border-left:3px solid var(--lean);padding:6px 12px;margin:8px 0 4px;color:var(--ink2);font-size:14px}ul.find .note{margin-left:230px}
figure{margin:16px 0;background:#fcfcfb;border:1px solid var(--line);border-radius:10px;padding:8px;overflow-x:auto}
figure img{width:100%;min-width:640px;height:auto;display:block}
.tbl{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--ink2);font-weight:600}td:first-child{white-space:nowrap}td.num{text-align:right;font-variant-numeric:tabular-nums}
dl{display:grid;grid-template-columns:minmax(140px,220px) 1fr;gap:6px 16px;font-size:14px}dt{font-weight:600}dd{margin:0}
@media (max-width:600px){dl{grid-template-columns:1fr}.tag{min-width:0;display:block}ul.find .note{margin-left:0}}
"""

TAG_CLASS = {"Do more": "good", "Do less": "bad", "Promising, keep testing": "lean",
             "Watch, tends to underperform": "lean", "Pulling ahead recently": "lean",
             "Falling behind recently": "lean"}


def _img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _e(v) -> str:
    return html.escape(str(v))


def _plan_html(scores: pd.DataFrame) -> str:
    rows = []
    for _, s in scores.iterrows():
        if s["error"]:
            cells = [s["post_id"], s["platform"], s["format"]]
            rows.append("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in cells)
                        + f'<td colspan="4" class="note">Not scored: {_e(s["error"])}</td></tr>')
            continue
        rows.append(
            f"<tr><td>{_e(s['post_id'])}</td><td>{_e(s['platform'])}</td><td>{_e(s['format'])}</td>"
            f"<td class='num'>{s['typical_rate']:.1f}%</td>"
            f"<td class='num'>{s['expected_rate']:.1f}% <span class='note'>({s['expected_rate_lo']:.1f}–"
            f"{s['expected_rate_hi']:.1f})</span></td>"
            f"<td class='num'>{s['chance_beat']:.0%} <span class='note'>({s['chance_beat_lo']:.0%}–"
            f"{s['chance_beat_hi']:.0%})</span></td><td>{_e(s['notes']) or '–'}</td></tr>")
    return ("<div class='tbl'><table><thead><tr><th>Post</th><th>Platform</th><th>Format</th>"
            "<th>Typical for its reach</th><th>Expected rate</th><th>Chance to beat its reach</th>"
            "<th>What applies</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def write_html(path: Path, *, title: str, meta: str, options: pd.DataFrame, recent_opts: pd.DataFrame,
               quality: pd.DataFrame, figures: dict[str, Path], scores: pd.DataFrame | None, platforms: list[str],
               notes: dict[str, str] | None = None):
    q = quality.set_index("platform")
    parts = [f"<title>{_e(title)}</title><style>{CSS}</style><main>",
             f"<h1>{_e(title)}</h1><p class='meta'>{_e(meta)}</p>"]
    for p in platforms:
        g = options[options["platform"] == p]
        allp = recent_opts[(recent_opts["platform"] == p) & (recent_opts["factor"] == "All posts")].iloc[0]
        parts.append(f"<h2>{_e(p)}</h2>")
        if (notes or {}).get(p):
            parts.append(f"<p class='platform-note'>{_e(notes[p])}</p>")
        parts.append("<div class='tiles'>"
                     f"<div class='tile'><div class='n'>{allp['recent_share_beat']:.0%}</div><div class='l'>of the "
                     f"last 90 days' {allp['recent_posts']} posts beat their reach (before: "
                     f"{allp['older_share_beat']:.0%}). {_e(allp['trend'])}.</div></div>"
                     f"<div class='tile'><div class='n'>{q.loc[p, 'auc_mean']:.2f}</div><div class='l'>held-out "
                     f"AUC of the planned-post score: {_e(q.loc[p, 'reliability'])}</div></div></div>")
        acts = g[g["verdict"].isin(ACTION_ORDER)].copy()
        acts["o"] = acts["verdict"].map(ACTION_ORDER.index)
        items = [f"<li><span class='tag {TAG_CLASS[r['verdict']]}'>{_e(r['verdict'])}</span>{_e(finding_line(r))}"
                 + (f"<div class='note'>{_e(r['note'])}</div>" if r["note"] else "") + "</li>"
                 for _, r in acts.sort_values(["o", "p_value"]).iterrows()]
        trends = recent_opts[(recent_opts["platform"] == p)
                             & recent_opts["trend"].isin(["Pulling ahead recently", "Falling behind recently"])]
        items += [f"<li><span class='tag lean'>{_e(r['trend'])}</span>{_e(option_name(r['factor'], r['option']))}: "
                  f"gap to other {_e(p)} posts changed by {_pts(r['gap_change'])} ({r['recent_posts']} recent posts)</li>"
                  for _, r in trends.iterrows()]
        parts.append("<h3>What the data says</h3><ul class='find'>" + ("".join(items) or
                     "<li class='note'>No choice stands out yet: pick formats, pillars and segments on strategy.</li>")
                     + "</ul>")
        parts.append(f"<figure><img alt='{_e(p)}: difference vs. baseline for each choice, with 95% intervals and "
                     f"verdicts' src='{_img(figures[p])}'></figure>")
    if scores is not None:
        parts.append("<h2>Planned posts</h2>" + _plan_html(scores)
                     + "<p class='note'>Ranges show the uncertainty of the estimate; single posts vary much more.</p>")
    parts.append("<h2>How to read this</h2><dl>"
                 + "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in HOW_TO_READ) + "</dl>")
    parts.append("<p class='note'>Full tables: recommendations.xlsx and the CSVs in this folder.</p></main>")
    path.write_text("<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width, "
                    "initial-scale=1'>" + "".join(parts), encoding="utf-8")
