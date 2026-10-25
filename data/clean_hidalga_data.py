#!/usr/bin/env python3
"""
clean_hidalga_data.py
=====================
Clean and consolidate Hidalga's three monthly marketing data exports into one
unified workbook with seven tidy sheets. Run this each month against the new
exports — the file formats are stable, so the same command produces the same
output structure every time.

INPUTS (three files, any order — matched by content/sheet names, not filename):
  1. Social media workbook   (.xlsx)  sheets: Facebook, Instagram, Linkedin,
                                              loomly-click-data, follower counts
  2. LinkedIn content export (.xls)   sheets: Metrics, All posts
  3. LinkedIn visitors export(.xls)   sheets: Visitor metrics, Location,
                                              Job function, Seniority,
                                              Industry, Company size

OUTPUT: one .xlsx with sheets:
  Social Posts (unified) | LinkedIn Posts (detail) | LinkedIn Daily Metrics |
  Visitor Daily Metrics  | Visitor Demographics    | Follower Counts | Loomly Clicks

USAGE
  # Point at a folder — the script auto-detects which file is which:
  python clean_hidalga_data.py --input-dir ./exports/2026-09 \
                               --output ./cleaned/Hidalga_Cleaned_2026-09.xlsx

  # Or pass the three files explicitly (any order):
  python clean_hidalga_data.py file1.xlsx file2.xls file3.xls -o out.xlsx

  # Auto-name the output from a month label:
  python clean_hidalga_data.py --input-dir ./exports/sep --month 2026-09

Requires: pandas, openpyxl, xlrd   (pip install pandas openpyxl xlrd)
"""
from __future__ import annotations
import argparse
import datetime as dt
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    import openpyxl
except ImportError:
    sys.exit("Missing dependency: openpyxl  (pip install openpyxl)")


# ─────────────────────────────────────────────────────────────────────────────
# Reference maps (edit here if Loomly/Planner tag vocabulary changes)
# ─────────────────────────────────────────────────────────────────────────────
ICP_SEGMENTS = [
    "Oncology Operations Leader",
    "Oncology Financial Leader",
    "Oncology Technical Leader",
    "Oncology Clinic Leader",
]

# (raw substring found in LABELS, canonical pillar name). Order matters:
# longer / more specific variants first.
PILLAR_MAP = [
    ("Thought Leadership & Industry Insights", "Thought Leadership & Industry Insights"),
    ("Events & Partnerships", "Events & Partnerships"),
    ("Patient-Centered Storytelling", "Patient-Centered Storytelling"),
    ("Patient-Centered Stories", "Patient-Centered Storytelling"),
    ("Product Development & Innovations", "Product Development & Innovations"),
    ("Product Dev. & Innovations", "Product Development & Innovations"),
    ("Team & Culture", "Team Spotlights & Company Culture"),
]

FORMAT_LABELS = ["Video", "Image", "Link", "Carousel", "Multi-Image", "Document"]

FORMAT_NORMALIZE = {
    "image": "Image", "video": "Video", "link": "Link", "carousel": "Carousel",
    "simple status": "Simple Status", "multi-image": "Multi-Image",
    "multi image": "Multi-Image", "document": "Document", "poll": "Poll",
}

# Expected column layout per platform sheet (used to align even if trailing
# blank columns drift). Matched case-insensitively against the header row.
PLATFORM_COLS = {
    "Facebook": ["DATE", "FORMAT", "POST", "LABELS", "IMPRESSIONS", "REACTIONS",
                 "COMMENTS", "SHARES", "CLICKS", "VIDEO VIEWS"],
    "Instagram": ["DATE", "FORMAT", "POST", "LABELS", "LIKES", "COMMENTS",
                  "IMPRESSIONS", "REACH", "ENGAGEMENT", "ENGAGEMENT-RATE",
                  "SAVES", "REELS-PLAYS"],
    "Linkedin": ["DATE", "FORMAT", "POST", "LABELS", "SHARES", "CLICKS",
                 "ENGAGEMENT-RATE", "REACTIONS", "IMPRESSIONS", "COMMENTS"],
}
PLATFORM_DISPLAY = {"Facebook": "Facebook", "Instagram": "Instagram", "Linkedin": "LinkedIn"}


# ─────────────────────────────────────────────────────────────────────────────
# Small normalizers
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).replace("\xa0", " ").replace("\\n", "\n")
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s or None


def norm_format(v):
    if v is None:
        return None
    s = re.sub(r"^[^\w]+", "", str(v).replace("\xa0", " ")).strip()
    return FORMAT_NORMALIZE.get(s.lower(), s or None)


def is_datetime_cell(v):
    """True for a real datetime.datetime (post row), False for datetime.time
    (the time sub-row) or strings/None."""
    return isinstance(v, dt.datetime)


def is_time_cell(v):
    return isinstance(v, dt.time) and not isinstance(v, dt.datetime)


def parse_labels(lbl):
    """Return (segments list, canonical pillar(s) joined with '; ', residual
    link tags). Loomly's LABELS field concatenates every tag on a post with
    no delimiter (e.g. "CarouselOncology Operations LeaderThought Leadership
    & Industry InsightsPatient-Centered Stories"), and posts are routinely
    tagged with more than one content pillar -- so every canonical pillar
    whose raw substring appears gets kept (in PILLAR_MAP order, deduped),
    matching how multi-pillar posts are represented in Hidalga_Master_Data_v1
    rather than silently keeping only the first match. Matching stays
    case-sensitive: several residual link tags legitimately end in lowercase
    "-link" (e.g. "blog-link", "Financial-ICP-link"), which a case-insensitive
    match against the FORMAT_LABELS "Link" would wrongly clip."""
    if lbl is None:
        return [], None, None
    s = str(lbl)
    segs = [seg for seg in ICP_SEGMENTS if seg in s]
    pillars = []
    for raw, canon in PILLAR_MAP:
        if raw in s and canon not in pillars:
            pillars.append(canon)
    residual = s
    for seg in ICP_SEGMENTS:
        residual = residual.replace(seg, "|")
    for raw, _ in PILLAR_MAP:
        residual = residual.replace(raw, "|")
    for f in FORMAT_LABELS:
        residual = residual.replace(f, "|")
    link_tags = [t.strip() for t in re.split(r"\|+", residual) if t.strip()]
    return segs, ("; ".join(pillars) if pillars else None), ("; ".join(link_tags) if link_tags else None)


def read_offset_sheet(path, sheet, header_row):
    """Read a sheet whose real header is on `header_row` (0-indexed), with a
    description banner above it."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    hdr = raw.iloc[header_row].tolist()
    df = raw.iloc[header_row + 1:].copy()
    df.columns = [clean_text(h) or f"col{i}" for i, h in enumerate(hdr)]
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# File-type detection (so filename order/naming doesn't matter)
# ─────────────────────────────────────────────────────────────────────────────
def sheet_names(path):
    try:
        return set(pd.ExcelFile(path).sheet_names)
    except Exception:
        return set()


def classify_files(paths):
    """Return dict role->path for: social_xlsx, li_content, li_visitors."""
    roles = {}
    for p in paths:
        names = {n.lower() for n in sheet_names(p)}
        if {"facebook", "instagram", "linkedin"} & names and "follower counts" in names:
            roles["social_xlsx"] = p
        elif {"metrics", "all posts"} <= names:
            roles["li_content"] = p
        elif "visitor metrics" in names or {"seniority", "job function"} & names:
            roles["li_visitors"] = p
    return roles


# ─────────────────────────────────────────────────────────────────────────────
# 1. Social platform sheets (FB / IG / LI)  ->  unified post table
# ─────────────────────────────────────────────────────────────────────────────
def build_social_posts(social_xlsx, warnings):
    wb = openpyxl.load_workbook(social_xlsx, read_only=True)
    records = []

    for sheet, expected in PLATFORM_COLS.items():
        if sheet not in wb.sheetnames:
            warnings.append(f"Social workbook missing sheet '{sheet}' — skipped.")
            continue

        ws = wb[sheet]
        header = [clean_text(c) for c in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
        # Map expected column name -> actual index (case-insensitive)
        idx = {}
        upper = [(h or "").upper() for h in header]
        for col in expected:
            idx[col] = upper.index(col.upper()) if col.upper() in upper else None
        post_i = idx.get("POST")
        plat = PLATFORM_DISPLAY[sheet]

        cur = None
        for r in ws.iter_rows(min_row=2, values_only=True):
            c0 = r[0] if len(r) else None

            if is_datetime_cell(c0):
                cur = {"row": list(r), "time": None, "title": None, "body": []}
                records.append((sheet, cur))
            else:
                if cur is None:
                    continue
                if is_time_cell(c0):
                    cur["time"] = c0.strftime("%I:%M %p").lstrip("0")
                elif isinstance(c0, str) and re.search(r"\d", c0):
                    cur["time"] = clean_text(c0)
                if post_i is not None and post_i < len(r) and r[post_i] is not None:
                    txt = clean_text(r[post_i])
                    if txt:
                        if cur["title"] is None:
                            cur["title"] = txt
                        else:
                            cur["body"].append(txt)

        # materialize
        for s, p in [(s, p) for s, p in records if s == sheet and "materialized" not in p]:
            p["materialized"] = True
            row = p["row"]

            def get(col):
                i = idx.get(col)
                return row[i] if i is not None and i < len(row) else None

            date = get("DATE")
            labels_raw = get("LABELS")
            segs, pillar, link_tags = parse_labels(labels_raw)

            if plat == "Facebook":
                imp = get("IMPRESSIONS") or 0
                num = sum((get(k) or 0) for k in ["REACTIONS", "COMMENTS", "SHARES", "CLICKS"])
                er = round(num / imp * 100, 4) if imp else None
            else:
                raw_er = get("ENGAGEMENT-RATE")
                er = round(float(raw_er) * 100, 4) if raw_er not in (None, "") else None

            records_out.append({
                "SOURCE_FILE": os.path.basename(social_xlsx),
                "PLATFORM": plat,
                "DATE": pd.to_datetime(date).date() if date is not None else None,
                "POST_TIME": p["time"],
                "FORMAT": norm_format(get("FORMAT")),
                "POST_TITLE": p["title"],
                "POST_BODY": "\n".join(p["body"]) or None,
                "LABELS_RAW": clean_text(labels_raw),
                "ICP_SEGMENTS": "; ".join(segs) if segs else None,
                "CONTENT_PILLAR": pillar,
                "LINK_TAGS": link_tags,
                "IMPRESSIONS": get("IMPRESSIONS"),
                "REACH": get("REACH"),
                "LIKES": get("LIKES"),
                "REACTIONS": get("REACTIONS"),
                "COMMENTS": get("COMMENTS"),
                "SHARES": get("SHARES"),
                "CLICKS": get("CLICKS"),
                "SAVES": get("SAVES"),
                "VIDEO_VIEWS": get("VIDEO VIEWS"),
                "REELS_PLAYS": get("REELS-PLAYS"),
                "ENGAGEMENT_RAW": get("ENGAGEMENT"),
                "ENGAGEMENT_RATE_PCT": er,
            })

    return pd.DataFrame(records_out)


# module-level accumulators reset per run
records_out: list = []


# ─────────────────────────────────────────────────────────────────────────────
# 2. LinkedIn content export
# ─────────────────────────────────────────────────────────────────────────────
def build_li_content(li_content):
    posts = read_offset_sheet(li_content, "All posts", 1)
    posts.insert(0, "PLATFORM", "LinkedIn")
    posts.insert(0, "SOURCE_FILE", os.path.basename(li_content) + " (All posts)")
    for c in posts.columns:
        if posts[c].dtype == object:
            posts[c] = posts[c].map(clean_text)
    for c in ["Created date", "Campaign start date", "Campaign end date"]:
        if c in posts.columns:
            posts[c] = pd.to_datetime(posts[c], errors="coerce").dt.date
    if "Engagement rate" in posts.columns:
        posts["Engagement rate (%)"] = pd.to_numeric(posts["Engagement rate"], errors="coerce") * 100
    if "Click through rate (CTR)" in posts.columns:
        posts["CTR (%)"] = pd.to_numeric(posts["Click through rate (CTR)"], errors="coerce") * 100

    metrics = read_offset_sheet(li_content, "Metrics", 1)
    metrics.insert(0, "SOURCE_FILE", os.path.basename(li_content) + " (Metrics)")
    metrics = metrics.rename(columns={"Date": "DATE"})
    metrics["DATE"] = pd.to_datetime(metrics["DATE"], errors="coerce").dt.date
    metrics = metrics.dropna(subset=["DATE"]).reset_index(drop=True)
    for c in [c for c in metrics.columns if str(c).lower().startswith("engagement rate")]:
        metrics[str(c) + " (%)"] = pd.to_numeric(metrics[c], errors="coerce") * 100

    return posts, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 3. Visitor export
# ─────────────────────────────────────────────────────────────────────────────
def build_visitors(li_visitors):
    daily = read_offset_sheet(li_visitors, "Visitor metrics", 0)
    daily.insert(0, "SOURCE_FILE", os.path.basename(li_visitors) + " (Visitor metrics)")
    daily = daily.rename(columns={"Date": "DATE"})
    daily["DATE"] = pd.to_datetime(daily["DATE"], errors="coerce").dt.date
    daily = daily.dropna(subset=["DATE"]).reset_index(drop=True)

    demo_frames = []
    for sheet in ["Location", "Job function", "Seniority", "Industry", "Company size"]:
        if sheet not in sheet_names(li_visitors):
            continue
        raw = pd.read_excel(li_visitors, sheet_name=sheet, header=None)
        dim_name = clean_text(raw.iloc[0, 0])
        df = raw.iloc[1:].copy()
        df.columns = ["DIMENSION_VALUE", "TOTAL_VIEWS"]
        df["DIMENSION_VALUE"] = df["DIMENSION_VALUE"].map(clean_text)
        df["TOTAL_VIEWS"] = pd.to_numeric(df["TOTAL_VIEWS"], errors="coerce")
        df.insert(0, "DIMENSION", dim_name)
        demo_frames.append(df.dropna(subset=["DIMENSION_VALUE"]).reset_index(drop=True))

    demographics = pd.concat(demo_frames, ignore_index=True) if demo_frames else pd.DataFrame()
    if not demographics.empty:
        demographics.insert(0, "SOURCE_FILE", os.path.basename(li_visitors) + " (demographics)")
    return daily, demographics


# ─────────────────────────────────────────────────────────────────────────────
# 4. follower counts + loomly clicks (both live in the social workbook)
# ─────────────────────────────────────────────────────────────────────────────
def build_followers(social_xlsx):
    if "follower counts" not in sheet_names(social_xlsx):
        return pd.DataFrame()
    fol = read_offset_sheet(social_xlsx, "follower counts", 0)
    fol.insert(0, "SOURCE_FILE", os.path.basename(social_xlsx) + " (follower counts)")
    if "CHANNEL" in fol.columns:
        fol["CHANNEL"] = fol["CHANNEL"].map(lambda x: clean_text(str(x)) if x is not None else None)
    return fol.dropna(how="all").reset_index(drop=True)


def _loomly_sheet_name(social_xlsx):
    """The Loomly click-data sheet's name has drifted across export tools
    (seen: "loomly-click-data", "Loomly_click") even though the row layout
    is identical -- match on normalized text instead of one exact literal,
    so a rename doesn't silently make build_loomly() return empty."""
    for name in sheet_names(social_xlsx):
        norm = re.sub(r"[^a-z]", "", name.lower())
        if "loomly" in norm and "click" in norm:
            return name
    return None


def build_loomly(social_xlsx):
    sheet = _loomly_sheet_name(social_xlsx)
    if sheet is None:
        return pd.DataFrame()
    wb = openpyxl.load_workbook(social_xlsx, read_only=True)
    ws = wb[sheet]
    recs, cur = [], None

    for r in ws.iter_rows(min_row=2, values_only=True):
        c0, c1, c2, c3, c4 = (list(r) + [None] * 5)[:5]
        if is_datetime_cell(c0):
            cur = {"CREATION_DATE": pd.to_datetime(c0).date(), "SHORT_LINK": clean_text(c1),
                   "TOTAL_CLICKS": c2, "DESTINATION_URL": None, "POST_TIME": None,
                   "src": [], "loc": []}
            recs.append(cur)
            if c3 is not None:
                cur["src"].append(clean_text(str(c3)))
            if c4 is not None:
                cur["loc"].append(clean_text(str(c4)))
        elif is_time_cell(c0):
            if cur is not None:
                cur["POST_TIME"] = c0.strftime("%H:%M")
                cur["DESTINATION_URL"] = clean_text(c1)
        else:
            if cur is None:
                continue
            if c3 is not None:
                cur["src"].append(clean_text(str(c3)))
            if c4 is not None:
                cur["loc"].append(clean_text(str(c4)))

    def fold(items):
        items = [i for i in items if i is not None]
        out, j = [], 0
        while j < len(items):
            lab = items[j]
            if j + 1 < len(items) and re.fullmatch(r"\d+", str(items[j + 1])):
                out.append(f"{lab}={items[j + 1]}"); j += 2
            else:
                out.append(f"{lab}"); j += 1
        return "; ".join(out) if out else None

    rows = [{
        "SOURCE_FILE": os.path.basename(social_xlsx) + f" ({sheet})",
        "CREATION_DATE": c["CREATION_DATE"], "POST_TIME": c["POST_TIME"],
        "SHORT_LINK": c["SHORT_LINK"], "DESTINATION_URL": c["DESTINATION_URL"],
        "TOTAL_CLICKS": c["TOTAL_CLICKS"],
        "SOURCE_BREAKDOWN": fold(c["src"]), "LOCATION_BREAKDOWN": fold(c["loc"]),
    } for c in recs]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Workbook writer with formatting
# ─────────────────────────────────────────────────────────────────────────────
def write_workbook(path, sheets: dict):
    from openpyxl.styles import Font, PatternFill, Alignment
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            (df if not df.empty else pd.DataFrame({"(no data)": []})).to_excel(
                xw, sheet_name=name[:31], index=False)

    wb = openpyxl.load_workbook(path)
    head_fill = PatternFill("solid", fgColor="1F3A5F")
    head_font = Font(bold=True, color="FFFFFF", size=10)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        for c in ws[1]:
            c.fill, c.font = head_fill, head_font
            c.alignment = Alignment(vertical="center", wrap_text=True)
        ws.row_dimensions[1].height = 28
        for col in ws.columns:
            letter = col[0].column_letter
            maxlen = max((len(str(c.value)) for c in col[:60] if c.value is not None), default=8)
            ws.column_dimensions[letter].width = min(max(maxlen + 2, 10), 48)
    wb.save(path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description="Clean Hidalga monthly marketing exports into one workbook.")
    ap.add_argument("files", nargs="*", help="The three export files (any order).")
    ap.add_argument("--input-dir", help="Folder containing exactly the three exports.")
    ap.add_argument("-o", "--output", help="Output .xlsx path.")
    ap.add_argument("--month", help="Month label (e.g. 2026-09) used to auto-name output if -o omitted.")
    args = ap.parse_args(argv)

    # gather candidate files
    paths = list(args.files)
    if args.input_dir:
        for f in sorted(os.listdir(args.input_dir)):
            if f.lower().endswith((".xls", ".xlsx")) and not f.startswith("~$"):
                paths.append(os.path.join(args.input_dir, f))
    paths = [p for p in dict.fromkeys(paths)]  # de-dupe, keep order

    if not paths:
        ap.error("No input files. Pass files or --input-dir.")

    roles = classify_files(paths)
    missing = {"social_xlsx", "li_content", "li_visitors"} - set(roles)
    if missing:
        print("WARNING: could not identify these roles:", ", ".join(sorted(missing)), file=sys.stderr)
        print("Detected:", {k: os.path.basename(v) for k, v in roles.items()}, file=sys.stderr)

    warnings = []
    global records_out
    records_out = []

    sheets, counts = {}, {}

    if "social_xlsx" in roles:
        social_posts = build_social_posts(roles["social_xlsx"], warnings)
        sheets["Social Posts (unified)"] = social_posts
        sheets["Follower Counts"] = build_followers(roles["social_xlsx"])
        sheets["Loomly Clicks"] = build_loomly(roles["social_xlsx"])
        counts["social posts"] = len(social_posts)
        if not social_posts.empty:
            counts["by platform"] = social_posts["PLATFORM"].value_counts().to_dict()

    if "li_content" in roles:
        li_posts, li_metrics = build_li_content(roles["li_content"])
        sheets["LinkedIn Posts (detail)"] = li_posts
        sheets["LinkedIn Daily Metrics"] = li_metrics
        counts["LI posts"] = len(li_posts)
        counts["LI daily rows"] = len(li_metrics)

    if "li_visitors" in roles:
        vis_daily, demographics = build_visitors(roles["li_visitors"])
        sheets["Visitor Daily Metrics"] = vis_daily
        sheets["Visitor Demographics"] = demographics
        counts["visitor daily rows"] = len(vis_daily)
        counts["demographic rows"] = len(demographics)

    if not sheets:
        ap.error("None of the three known file types were found among the inputs.")

    # canonical sheet order
    order = ["Social Posts (unified)", "LinkedIn Posts (detail)", "LinkedIn Daily Metrics",
             "Visitor Daily Metrics", "Visitor Demographics", "Follower Counts", "Loomly Clicks"]
    sheets = {k: sheets[k] for k in order if k in sheets}

    # output path
    out = args.output
    if not out:
        label = args.month or dt.date.today().strftime("%Y-%m")
        out = f"Hidalga_Cleaned_Data_{label}.xlsx"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if os.path.basename(out).startswith("Hidalga_Master_Data") and os.path.exists(out):
        ap.error(f"{out} is a master workbook; write elsewhere and use merge_into_master.py to add it.")

    write_workbook(out, sheets)

    # report
    print("Detected files:")
    for role, p in roles.items():
        print(f"  {role:12s} -> {os.path.basename(p)}")
    print("\nRow counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    for w in warnings:
        print("  WARN:", w)
    print(f"\nWrote {out}")
    return out


if __name__ == "__main__":
    main()
