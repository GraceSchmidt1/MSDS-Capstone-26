"""
Entry point for the web page (index.html → site/worker.js, in Pyodide): optionally add new exports to the master
(data/clean_hidalga_data.py + data/merge_into_master.py), run the pipeline, and hand back what the page shows and
offers for download. Also runs locally, which is how the tests check it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from modeling import pipeline, report
from modeling.features import REPO_ROOT, load_config, split_tags
from modeling.recommend import read_plan

DATA_DIR = REPO_ROOT / "data"
EXPORTS = {"social_xlsx": "Loomly social media workbook (.xlsx)", "li_content": "LinkedIn content export (.xls)",
           "li_visitors": "LinkedIn visitors export (.xls)"}
NOT_USED = {"li_followers": "LinkedIn followers export (.xls)"}


class InputError(ValueError):
    """A problem with an uploaded file, worded for the page."""


def _data_scripts():
    if str(DATA_DIR) not in sys.path:
        sys.path.insert(0, str(DATA_DIR))
    import clean_hidalga_data
    import merge_into_master
    return clean_hidalga_data, merge_into_master


def next_master_name(name: str) -> str:
    m = re.search(r"_v(\d+)\.xlsx$", name)
    return f"{name[:m.start()]}_v{int(m.group(1)) + 1}.xlsx" if m else f"{Path(name).stem}_updated.xlsx"


def _new_post_warnings(new_posts: list[dict], has_li_export: bool, config: dict) -> list[str]:
    warnings = []
    untagged = [p for p in new_posts if not p["Pillar"] or not p["ICP segments"]]
    if untagged:
        warnings.append(f"{len(untagged)} new post(s) have no pillar or no ICP segment in Loomly; they count as posts "
                        "without one. Fix the labels in Loomly, or add corrections with "
                        "data/apply_master_corrections.py.")
    odd = sorted({f"{p['Format']} ({p['Platform']})" for p in new_posts
                  if p["Platform"] in config["formats"] and p["Format"] not in config["formats"][p["Platform"]]})
    if odd:
        warnings.append(f"New post format(s) the tool doesn't know yet: {', '.join(odd)}. Add them to `formats` in "
                        "modeling/config.yaml before the recommendations can run.")
    loomly_li = [p for p in new_posts if p["Platform"] == "LinkedIn" and p["Numbers from"] == "Loomly"]
    if loomly_li and not has_li_export:
        warnings.append(f"{len(loomly_li)} new LinkedIn post(s) use Loomly's numbers because the LinkedIn content "
                        "export wasn't included. They keep those numbers: the master never overwrites rows it "
                        "already has.")
    elif loomly_li:
        warnings.append(f"{len(loomly_li)} new LinkedIn post(s) weren't found in the LinkedIn content export (posted "
                        "after it was pulled?) and use Loomly's numbers.")
    return warnings


def update(master_path, master_name: str, exports: list, pull_date: dt.date, work, config: dict) -> dict:
    """Clean `exports` ([(path, original name)]) and merge them into the master as the next version, in `work`.
    Returns the files found (and missing), rows added per sheet, the new posts, warnings, and the new master's path
    and name (None when nothing was new: the master already had everything)."""
    clean_hidalga_data, merge_into_master = _data_scripts()
    work = Path(work)
    paths = [str(p) for p, _ in exports]
    names = dict(zip(paths, (n for _, n in exports)))
    found = {p: clean_hidalga_data.classify_files([p]) for p in paths}
    unknown = [names[p] for p in paths if not found[p]]
    if unknown:
        raise InputError(f"Not one of the exports the tool knows: {', '.join(unknown)}. Expected: "
                         + "; ".join(EXPORTS.values()) + ".")
    for role, label in {**EXPORTS, **NOT_USED}.items():
        same = [names[p] for p in paths if role in found[p]]
        if len(same) > 1:
            raise InputError(f"Two files look like the {label}: {', '.join(same)}. Keep just the newest one.")
    roles = clean_hidalga_data.classify_files(paths)

    out = {"files": [{"file": names[p], "export": EXPORTS.get(r) or NOT_USED[r], "used": r in EXPORTS}
                     for r, p in roles.items()],
           "missing": [label for r, label in EXPORTS.items() if r not in roles],
           "added": [], "new_posts": [], "warnings": [], "logged": 0, "new_master_path": None,
           "new_master_name": None, "pull_date": pull_date.isoformat()}
    sheets, _, _, warnings = clean_hidalga_data.clean(paths)
    out["warnings"] += warnings
    if not sheets:
        out["warnings"].append("The LinkedIn followers export isn't used by the master, so nothing was added.")
        return out

    cleaned = work / "cleaned_exports.xlsx"
    clean_hidalga_data.write_workbook(str(cleaned), sheets)
    new_name = next_master_name(master_name)
    new_path = work / new_name
    try:
        summary, logged = merge_into_master.update_master(str(master_path), str(cleaned), str(new_path), pull_date)
    except merge_into_master.MergeError as e:
        raise InputError(f"The exports couldn't be added to {master_name}: {e}") from None
    out["added"] = [{"sheet": s, "in_master": b, "in_exports": i, "added": a, "already_there": k}
                    for s, b, i, a, k in summary]
    out["logged"] = logged
    if not any(a["added"] for a in out["added"]):
        new_path.unlink()
        return out

    n_new = next((a["added"] for a in out["added"] if a["sheet"] == config["posts_sheet"]), 0)
    if n_new:                                           # appended rows are at the end of the sheet
        posts = pd.read_excel(new_path, sheet_name=config["posts_sheet"]).tail(n_new)
        out["new_posts"] = [{"Platform": r["PLATFORM"], "Date": pd.Timestamp(r["DATE"]).strftime("%Y-%m-%d"),
                             "Post": r["POST_TITLE"], "Format": r["FORMAT"],
                             "Pillar": "; ".join(split_tags(r["CONTENT_PILLAR"])),
                             "ICP segments": "; ".join(split_tags(r["ICP_SEGMENTS"])),
                             "Numbers from": r["ENGAGEMENT_SOURCE"]} for _, r in posts.iterrows()]
        out["warnings"] += _new_post_warnings(out["new_posts"], "li_content" in roles, config)
    out["new_master_path"], out["new_master_name"] = str(new_path), new_name
    return out


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(master_path, master_name: str, plan_path=None, plan_name: str | None = None, exports=None,
        pull_date: dt.date | str | None = None, out=None, log=print) -> dict:
    """Returns meta, summary rows, report sections (heading, html), the report CSS, and every output file as bytes
    (plus all of them as one zip). With `exports` ([(path, name)]) the master is updated first: `update` describes
    what was added and `new_master` holds the next version (also among the files). If the analysis can't run on the
    updated master, `analysis_error` says why and the update and new master are still returned."""
    config = load_config()
    try:
        plan = read_plan(plan_path) if plan_path else None
    except (ValueError, UnicodeDecodeError) as e:
        raise InputError(f"{plan_name}: {e}") from None
    try:
        found = set(pd.ExcelFile(master_path).sheet_names)
    except (ValueError, zipfile.BadZipFile, OSError):
        found = set()
    if not {config["posts_sheet"], config["linkedin_detail_sheet"]} <= found:
        raise InputError(f"{master_name} doesn't look like a master workbook: it needs the sheets "
                         f"'{config['posts_sheet']}' and '{config['linkedin_detail_sheet']}'.")
    out = Path(out or tempfile.mkdtemp(prefix="rec_"))
    upd, new_master = None, None
    if exports:
        log("Adding the new exports to the master…")
        pull_date = dt.date.fromisoformat(str(pull_date)) if pull_date else dt.date.today()
        work = out / "update"
        work.mkdir(parents=True, exist_ok=True)
        upd = update(master_path, master_name, exports, pull_date, work, config)
        if upd["new_master_path"]:
            new_master = {"name": upd["new_master_name"], "bytes": Path(upd["new_master_path"]).read_bytes()}
            master_path, master_name = upd["new_master_path"], upd["new_master_name"]
    config["master"] = master_name

    res_dir = out / "run"
    result, error = None, None
    try:
        result = pipeline.run(res_dir, config, master_path=Path(master_path), plan=plan,
                              plan_path=Path(plan_path) if plan_path else None, plan_name=plan_name, tag="web",
                              log=log)
    except ValueError as e:                             # e.g. a new format not yet in config.yaml
        if upd is None:
            raise
        error = str(e)
    if result is not None and upd is not None:          # record the update in run_info.json
        info_path = res_dir / "run_info.json"
        info = json.loads(info_path.read_text())
        info["update"] = {"exports": [{"file": n, "sha256": _sha(p)} for p, n in exports],
                          "pull_date": upd["pull_date"], "added": upd["added"], "new_master": upd["new_master_name"]}
        info_path.write_text(json.dumps(info, indent=2, default=str))

    files = {} if result is None else {p.relative_to(res_dir).as_posix(): p.read_bytes()
                                       for p in sorted(res_dir.rglob("*")) if p.is_file()}
    if new_master:
        files[new_master["name"]] = new_master["bytes"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return {"meta": result["meta"] if result else f"Data: {master_name}",
            "summary": result["summary"].to_dict("records") if result else [],
            "sections": [list(s) for s in result["sections"]] if result else [], "css": report.CSS,
            "files": files, "zip": buf.getvalue(), "title": pipeline.TITLE,
            "update": None if upd is None else {k: v for k, v in upd.items() if k != "new_master_path"},
            "new_master": new_master, "analysis_error": error}
