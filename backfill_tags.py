#!/usr/bin/env python3
"""
backfill_tags.py
=================
Heuristic keyword-based tag suggestion for posts missing ICP_SEGMENTS and/or
CONTENT_PILLAR in the Social Posts (unified) sheet -- primarily the Sep 2025
-> ~Feb 2026 posts that predate the current tagging taxonomy.

THIS SCRIPT DOES NOT SILENTLY FILL IN GROUND TRUTH. It scores each untagged
post's title+body text against keyword sets drawn directly from
Q1_ICP_Segments_and_Content_Pillar_Reference_Guide.pdf, and buckets every
previously-null row into one of three outcomes:

  - AUTO-APPLIED   : score is high and unambiguous enough to fill in directly
  - NEEDS REVIEW   : some signal, but not confident enough to auto-apply
  - INSUFFICIENT TEXT: no post title/body to work from at all

Every row -- auto-applied or not -- gets written to a review CSV with its
matched keywords and confidence, so nothing is a black box. Re-running the
script is idempotent: it only ever touches rows where ICP_SEGMENTS /
CONTENT_PILLAR is currently null, never rows that already carry a real tag
(manual or already-backfilled).

USAGE
  python backfill_tags.py --input Hidalga_Master_Data_v1.xlsx \
                           --output Hidalga_Master_Data_v1_tagged.xlsx \
                           --review tag_backfill_review.csv

Requires: pandas, openpyxl
"""
from __future__ import annotations
import argparse
import re
import sys

import pandas as pd

try:
    import openpyxl  # noqa: F401
except ImportError:
    sys.exit("Missing dependency: openpyxl  (pip install openpyxl)")


# ─────────────────────────────────────────────────────────────────────────────
# Keyword sets, drawn from Q1_ICP_Segments_and_Content_Pillar_Reference_Guide.pdf
# Edit these lists as false positives/negatives surface during review --
# this is meant to be tuned, not treated as final on the first run.
# ─────────────────────────────────────────────────────────────────────────────
ICP_KEYWORDS = {
    "Oncology Operations Leader": [
        "practice administrator", "executive director", "director of operations",
        "clinic administrator", "regional operations", "operational reliability",
        "firefighting", "patient delay", "trustworthy dashboard", "economic buyer",
        "time to treatment", "staffing", "cycle time", "order to infusion",
        "standardize workflow", "standardizing workflow", "measurable roi",
        "vendor accountability", "operational outcome", "operational rework",
        "coo",
    ],
    "Oncology Financial Leader": [
        "revenue cycle", "billing manager", "patient financial services",
        "finance director", "controller", "cfo", "revenue integrity",
        "denial reduction", "first pass approval", "days in accounts receivable",
        "clean claims", "appeal win rate", "leakage detection", "payer contract",
        "predictable cash flow", "audit defensibility", "defensible",
        "prior authorization denial", "reimbursement", "financial outcome",
        "baseline", "payback period", "audit log", "traceability", "denials",
    ],
    "Oncology Technical Leader": [
        "it director", "systems administrator", "ciso", "clinical informaticist",
        "system reliability", "screen scraping", "deployment model",
        "escalation path", "integration simplicity", "sso", "rbac",
        "audit trails", "uptime", "workflow disruption", "support burden",
        "interoperability", "fhir", "hl7", "data governance", "soc 2", "hipaa",
        "product roadmap", "security officer", "cio",
    ],
    "Oncology Clinic Leader": [
        "medical director", "physician partner", "lead oncologist",
        "nursing director", "infusion nurse manager", "app lead",
        "clinical operations director", "chief clinical officer",
        "clinical judgment", "patient impact", "explainability",
        "override or correct", "keeping patients on schedule",
        "administrative burden on clinical", "peer to peer",
        "peer-to-peer", "retroactive documentation", "guideline consistency",
        "treatment delay",
    ],
}

PILLAR_KEYWORDS = {
    "Thought Leadership & Industry Insights": [
        "industry trend", "case studies", "case study", "beta test outcome",
        "data-driven insight", "healthcare inefficienc", "expert interview",
        "new data alert", "physicians are concerned", "% of physicians",
        "study shows", "report finds", "survey finds",
    ],
    "Product Development & Innovations": [
        "product update", "new feature", "feature launch", "behind-the-scenes",
        "anticipated outcome", "beta test", "halopa", "new capability",
        "marketing site", "prior authorization automation", "roadmap",
        "sneak peek",
    ],
    "Events & Partnerships": [
        "live event", "conference", "partnership", "competition highlight",
        "milestone", "circle", "attending", "summit", "webinar", "nasdaq",
        "award", "grant", "startup crawl", "session", "pitch", "accepted into",
        "honored to", "we're looking forward to",
    ],
    "Patient-Centered Storytelling": [
        "patient journey", "before-and-after", "personal narrative",
        "healthcare delay", "patient story", "treatment access", "biomarker",
        "oncology patient", "navigating high-cost", "prior authorization",
    ],
    "Team Spotlights & Company Culture": [
        "team member", "spotlight", "motivational quote", "employee-driven",
        "joins the team", "meet ", "congratulations", "promotion", "advisor",
        "company culture", "anniversary", "thank you, ", "welcome to the team",
        "welcome ", "we recognize the",
    ],
}


def _phrase_pattern(phrase: str) -> re.Pattern:
    """Whole-word/phrase boundary match, so short keywords like 'coo' or
    'sso' don't fire inside unrelated words like 'coordination' or 'lessons'."""
    return re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)")


_COMPILED_CACHE: dict[str, re.Pattern] = {}


def _compiled(phrase: str) -> re.Pattern:
    if phrase not in _COMPILED_CACHE:
        _COMPILED_CACHE[phrase] = _phrase_pattern(phrase)
    return _COMPILED_CACHE[phrase]


def score_categories(text: str, keyword_map: dict[str, list[str]]) -> dict[str, tuple[int, list[str]]]:
    """Return {category: (hit_count, matched_phrases)} for every category."""
    t = text.lower()
    out = {}
    for cat, phrases in keyword_map.items():
        hits = [p for p in phrases if _compiled(p).search(t)]
        out[cat] = (len(hits), hits)
    return out


def suggest_pillar(text: str) -> tuple[str | None, str, list[str]]:
    """Return (suggested_pillar_or_None, decision, matched_keywords)."""
    scores = score_categories(text, PILLAR_KEYWORDS)
    ranked = sorted(scores.items(), key=lambda kv: kv[1][0], reverse=True)
    top_cat, (top_score, top_hits) = ranked[0]
    second_score = ranked[1][1][0] if len(ranked) > 1 else 0

    if top_score == 0:
        return None, "needs_review_no_signal", []
    if top_score >= 2 and (top_score - second_score) >= 1:
        return top_cat, "auto_applied", top_hits
    return top_cat, "needs_review_weak_signal", top_hits


def suggest_segments(text: str) -> tuple[list[str], str, dict[str, list[str]]]:
    """Return (auto_segments, decision, {segment: matched_keywords} for all
    segments with any hit, for review visibility)."""
    scores = score_categories(text, ICP_KEYWORDS)
    hit_segments = {cat: hits for cat, (n, hits) in scores.items() if n > 0}

    if not hit_segments:
        return [], "needs_review_no_signal", {}

    # Auto-apply only segments with >=2 distinct keyword hits -- multi-label
    # false positives are costlier here since they duplicate rows downstream
    # in the segment-explosion step.
    strong = {cat: hits for cat, (n, hits) in scores.items() if n >= 2}
    if strong:
        return sorted(strong.keys()), "auto_applied", hit_segments

    return [], "needs_review_weak_signal", hit_segments


def main(argv=None):
    ap = argparse.ArgumentParser(description="Suggest ICP segment / content pillar tags for untagged posts.")
    ap.add_argument("--input", required=True, help="Path to Hidalga_Master_Data_v*.xlsx")
    ap.add_argument("--output", required=True, help="Path to write the updated workbook.")
    ap.add_argument("--review", required=True, help="Path to write the human-review CSV.")
    ap.add_argument("--sheet", default="Social Posts (unified)")
    args = ap.parse_args(argv)

    xl = pd.ExcelFile(args.input)
    sheets = {name: pd.read_excel(args.input, sheet_name=name) for name in xl.sheet_names}
    df = sheets[args.sheet].copy()

    if "ICP_SEGMENTS_SOURCE" not in df.columns:
        df["ICP_SEGMENTS_SOURCE"] = df["ICP_SEGMENTS"].apply(lambda v: "original" if pd.notna(v) else None)
    if "CONTENT_PILLAR_SOURCE" not in df.columns:
        df["CONTENT_PILLAR_SOURCE"] = df["CONTENT_PILLAR"].apply(lambda v: "original" if pd.notna(v) else None)

    review_rows = []
    n_icp_auto = n_icp_review = n_icp_no_text = 0
    n_pillar_auto = n_pillar_review = n_pillar_no_text = 0

    for i, row in df.iterrows():
        needs_icp = pd.isna(row["ICP_SEGMENTS"])
        needs_pillar = pd.isna(row["CONTENT_PILLAR"])
        if not (needs_icp or needs_pillar):
            continue

        text = " ".join(str(x) for x in [row.get("POST_TITLE"), row.get("POST_BODY")] if pd.notna(x)).strip()
        has_text = len(text) >= 8  # a handful of chars is not enough signal

        rec = {
            "ROW_INDEX": i,
            "PLATFORM": row.get("PLATFORM"),
            "DATE": row.get("DATE"),
            "POST_TITLE": row.get("POST_TITLE"),
            "POST_BODY_SNIPPET": (str(row.get("POST_BODY"))[:120] if pd.notna(row.get("POST_BODY")) else None),
        }

        if needs_icp:
            if not has_text:
                rec["ICP_DECISION"] = "insufficient_text"
                n_icp_no_text += 1
            else:
                segs, decision, matched = suggest_segments(text)
                rec["ICP_SUGGESTED"] = "; ".join(segs) if segs else None
                rec["ICP_DECISION"] = decision
                rec["ICP_MATCHED_KEYWORDS"] = "; ".join(
                    f"{cat}:[{','.join(kw)}]" for cat, kw in matched.items()
                ) if matched else None
                if decision == "auto_applied":
                    df.at[i, "ICP_SEGMENTS"] = rec["ICP_SUGGESTED"]
                    df.at[i, "ICP_SEGMENTS_SOURCE"] = "auto_backfill"
                    n_icp_auto += 1
                else:
                    n_icp_review += 1

        if needs_pillar:
            if not has_text:
                rec["PILLAR_DECISION"] = "insufficient_text"
                n_pillar_no_text += 1
            else:
                pillar, decision, matched = suggest_pillar(text)
                rec["PILLAR_SUGGESTED"] = pillar
                rec["PILLAR_DECISION"] = decision
                rec["PILLAR_MATCHED_KEYWORDS"] = "; ".join(matched) if matched else None
                if decision == "auto_applied":
                    df.at[i, "CONTENT_PILLAR"] = pillar
                    df.at[i, "CONTENT_PILLAR_SOURCE"] = "auto_backfill"
                    n_pillar_auto += 1
                else:
                    n_pillar_review += 1

        review_rows.append(rec)

    # Anything still null after auto-apply is explicitly marked for review,
    # so it's never ambiguous whether a blank cell was checked or skipped.
    for i, row in df.iterrows():
        if pd.isna(row["ICP_SEGMENTS"]) and pd.isna(row["ICP_SEGMENTS_SOURCE"]):
            df.at[i, "ICP_SEGMENTS_SOURCE"] = "needs_manual_review"
        if pd.isna(row["CONTENT_PILLAR"]) and pd.isna(row["CONTENT_PILLAR_SOURCE"]):
            df.at[i, "CONTENT_PILLAR_SOURCE"] = "needs_manual_review"

    sheets[args.sheet] = df
    with pd.ExcelWriter(args.output, engine="openpyxl") as xw:
        for name, sdf in sheets.items():
            sdf.to_excel(xw, sheet_name=name[:31], index=False)

    pd.DataFrame(review_rows).to_csv(args.review, index=False)

    print(f"ICP_SEGMENTS   : {n_icp_auto} auto-applied, {n_icp_review} need review, {n_icp_no_text} insufficient text")
    print(f"CONTENT_PILLAR : {n_pillar_auto} auto-applied, {n_pillar_review} need review, {n_pillar_no_text} insufficient text")
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.review}  <- open this and eyeball the auto_applied rows + resolve needs_review rows")


if __name__ == "__main__":
    main()
