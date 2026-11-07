"""Tests for modeling/report.py and the run_recommend CLI. Run from the repo root:  python -m pytest modeling/tests -q"""
import json

import numpy as np
import openpyxl
import pandas as pd
import pytest
import yaml

from modeling import report
from modeling.features import DEFAULT_CONFIG, REPO_ROOT, file_sha256, load_config
from modeling.run_recommend import main

CONFIG = load_config()
TEMPLATE = REPO_ROOT / "modeling" / "plan_template.csv"
SHEETS = ["Summary", "What's working", "What to test next", "Recent posts", "Recent changes", "Planned posts",
          "Model check"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One end-to-end run with the template plan, written to a temporary output folder."""
    tmp = tmp_path_factory.mktemp("rec")
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    cfg["output_dir"] = str(tmp / "runs")
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    main(["--config", str(path), "--tag", "test", "--plan", str(TEMPLATE)])
    return next((tmp / "runs").iterdir())


def test_run_writes_every_file(run):
    for f in ["options.csv", "factor_tests.csv", "model_quality.csv", "recent_posts.csv", "recent_options.csv",
              "plan_input.csv", "plan_scores.csv", "recommendations.xlsx", "recommendations.html", "run_info.json"]:
        assert (run / f).exists(), f
    for p in CONFIG["platforms"]:
        assert (run / "figures" / f"options_{p.lower()}.png").exists()
    info = json.loads((run / "run_info.json").read_text())
    assert info["master_sha256"] == file_sha256(REPO_ROOT / CONFIG["master"])
    assert info["plan"]["sha256"] == file_sha256(TEMPLATE)


def test_refuses_to_overwrite(run):
    cfg = run.parent.parent / "config.yaml"
    with pytest.raises(SystemExit, match="already exists"):
        main(["--config", str(cfg), "--tag", "test"])


def test_workbook_sheets_and_summary(run):
    wb = openpyxl.load_workbook(run / "recommendations.xlsx")
    assert wb.sheetnames == SHEETS
    summary = pd.read_excel(run / "recommendations.xlsx", sheet_name="Summary")
    opts = pd.read_csv(run / "options.csv")
    acts = opts[opts["verdict"].isin(report.ACTION_ORDER)]
    assert (summary["Type"].isin(report.ACTION_ORDER)).sum() == len(acts)
    assert set(summary["Platform"]) == set(CONFIG["platforms"])
    working = pd.read_excel(run / "recommendations.xlsx", sheet_name="What's working")
    assert len(working) == (opts["compared_with"] != "–").sum()                  # baselines left out
    planned = pd.read_excel(run / "recommendations.xlsx", sheet_name="Planned posts")
    assert len(planned) == len(pd.read_csv(TEMPLATE))


def test_html_has_every_platform_and_finding(run):
    page = (run / "recommendations.html").read_text()
    assert page.count("data:image/png;base64,") == len(CONFIG["platforms"])        # charts embedded
    for p in CONFIG["platforms"]:
        assert f"<h2>{p}</h2>" in page
    opts = pd.read_csv(run / "options.csv", keep_default_na=False, na_values=[""])
    for _, r in opts[opts["verdict"].isin(report.ACTION_ORDER)].iterrows():
        assert report.html.escape(report.finding_line(r)) in page
    assert "Planned posts" in page and "How to read this" in page


# ── helpers ─────────────────────────────────────────────────────────────────
def test_option_names_and_finding_line():
    assert report.option_name("Format", "Document") == "Document"
    assert report.option_name("ICP segment", "Oncology Clinic Leader") == "Oncology Clinic Leader (ICP segment)"
    assert report.option_name("Content pillar", "Events & Partnerships") == "Events & Partnerships (pillar)"
    r = pd.Series({"factor": "Format", "option": "Document", "compared_with": "Image", "diff_vs_baseline": 14.11,
                   "posts": 13, "share_beat": 0.769})
    assert report.finding_line(r) == "Document: +14.1 pts vs Image (13 posts, 77% beat their reach)"


def test_test_next_sheet_keeps_only_unsettled_finite():
    opts = pd.DataFrame({"platform": "LinkedIn", "factor": "Format",
                         "option": ["A", "B", "C", "D", "E"], "posts": 10, "diff_vs_baseline": 1.0,
                         "verdict": ["Do more", "Promising, keep testing", "No clear difference",
                                     "Not enough posts yet", "Promising, keep testing"],
                         "posts_needed": [5.0, 20.0, np.inf, 8.0, 0.0]})
    t = report.test_next_sheet(opts)
    assert list(t["Choice"]) == ["B", "D"]                                       # promising first


def test_number_formats():
    f = report._number_format
    assert f("Share beat their reach", pd.Series([0.5])) == "0%"
    assert f("Factor p adjusted", pd.Series([0.5])) == "0.000"
    assert f("Lift R² (held-out)", pd.Series([0.5])) == "0.000"
    assert f("Avg lift (pts)", pd.Series([0.5])) == "0.0"
    assert f("Posts", pd.Series([3])) is None


def test_plan_sheet_without_plan():
    assert "--plan" in report.plan_sheet(None).iloc[0, 0]


def test_workbook_is_byte_identical_for_the_same_data(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0], "Share beat": [0.5, 0.25]})
    stamp = pd.Timestamp("2026-09-29")
    a, b = tmp_path / "a.xlsx", tmp_path / "b.xlsx"
    report.write_excel(a, {"S": df}, stamp)
    report.write_excel(b, {"S": df}, stamp)
    assert a.read_bytes() == b.read_bytes()
    assert openpyxl.load_workbook(a).properties.created == stamp.to_pydatetime()
