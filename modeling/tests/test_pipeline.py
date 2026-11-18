"""Tests for modeling/pipeline.py as the web page calls it. Run from the repo root:  python -m pytest modeling/tests -q"""
import json
import shutil

import pandas as pd
import pytest

from modeling import pipeline
from modeling.features import REPO_ROOT, file_sha256, load_config
from modeling.recommend import read_plan

CONFIG = load_config()
TEMPLATE = REPO_ROOT / "modeling" / "plan_template.csv"


@pytest.fixture(scope="module")
def upload(tmp_path_factory):
    """A run the way the web page makes it: master copied outside the repo, shown by its file name; no plan."""
    tmp = tmp_path_factory.mktemp("web")
    master = tmp / "uploaded_master.xlsx"
    shutil.copyfile(REPO_ROOT / CONFIG["master"], master)
    cfg = dict(CONFIG, master=master.name)
    result = pipeline.run(tmp / "out", cfg, master_path=master, log=lambda m: None)
    return tmp / "out", result, master


def test_master_path_and_name(upload):
    out, result, master = upload
    info = json.loads((out / "run_info.json").read_text())
    assert info["master"] == "uploaded_master.xlsx"
    assert info["master_sha256"] == result["master_sha256"] == file_sha256(master)
    assert info["plan"] is None and info["git"] is None
    assert result["meta"].startswith("Data: uploaded_master.xlsx · ")


def test_no_plan_files_or_section(upload):
    out, result, _ = upload
    assert not (out / "plan_input.csv").exists() and not (out / "plan_scores.csv").exists()
    assert [h for h, _ in result["sections"]] == CONFIG["platforms"] + ["How to read this"]


def test_sections_make_up_the_page(upload):
    out, result, _ = upload
    page = (out / "recommendations.html").read_text(encoding="utf-8")
    assert "".join(html for _, html in result["sections"]) in page
    for p in CONFIG["platforms"]:
        assert dict(result["sections"])[p].startswith(f"<h2>{p}</h2>")


def test_summary_matches_workbook(upload):
    out, result, _ = upload
    sheet = pd.read_excel(out / "recommendations.xlsx", sheet_name="Summary", keep_default_na=False)
    assert sheet["Finding"].tolist() == result["summary"]["Finding"].tolist()


def test_plan_name_for_an_uploaded_plan(tmp_path):
    plan = tmp_path / "next_month.csv"
    shutil.copyfile(TEMPLATE, plan)
    cfg = dict(CONFIG, platforms=["LinkedIn"])
    result = pipeline.run(tmp_path / "out", cfg, plan=read_plan(plan), plan_path=plan, plan_name=plan.name,
                          log=lambda m: None)
    info = json.loads((tmp_path / "out" / "run_info.json").read_text())
    assert info["plan"] == {"file": "next_month.csv", "sha256": file_sha256(plan)}
    assert [h for h, _ in result["sections"]] == ["LinkedIn", "Planned posts", "How to read this"]
