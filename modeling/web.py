"""
Entry point for the web page (index.html → site/worker.js, in Pyodide): run the pipeline on uploaded files and hand
back what the page shows and offers for download. Also runs locally, which is how the tests check it.
"""
from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

from modeling import pipeline, report
from modeling.features import load_config
from modeling.recommend import read_plan


class InputError(ValueError):
    """A problem with an uploaded file, worded for the page."""


def run(master_path, master_name: str, plan_path=None, plan_name: str | None = None, out=None, log=print) -> dict:
    """Returns meta, summary rows, report sections (heading, html), the report CSS, and every output file as bytes
    (plus all of them as one zip)."""
    config = load_config()
    config["master"] = master_name
    try:
        plan = read_plan(plan_path) if plan_path else None
    except (ValueError, UnicodeDecodeError) as e:
        raise InputError(f"{plan_name}: {e}") from None
    out = Path(out or tempfile.mkdtemp(prefix="rec_"))
    try:
        result = pipeline.run(out, config, master_path=Path(master_path), plan=plan,
                              plan_path=Path(plan_path) if plan_path else None, plan_name=plan_name, tag="web",
                              log=log)
    except (ValueError, zipfile.BadZipFile) as e:
        if isinstance(e, zipfile.BadZipFile) or "Worksheet named" in str(e) or "Excel file format" in str(e):
            raise InputError(f"{master_name} doesn't look like a master workbook: it needs the sheets "
                             f"'{config['posts_sheet']}' and '{config['linkedin_detail_sheet']}'.") from None
        raise
    files = {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return {"meta": result["meta"], "summary": result["summary"].to_dict("records"),
            "sections": [list(s) for s in result["sections"]], "css": report.CSS, "files": files,
            "zip": buf.getvalue(), "title": pipeline.TITLE}
