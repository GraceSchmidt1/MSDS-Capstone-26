"""Tests for modeling/web.py and the web page's files. Run from the repo root:  python -m pytest modeling/tests -q"""
import io
import json
import re
import subprocess
import sys
import zipfile

import pytest

from modeling import report, web
from modeling.features import REPO_ROOT, load_config

CONFIG = load_config()
MASTER = REPO_ROOT / CONFIG["master"]
TEMPLATE = REPO_ROOT / "modeling" / "plan_template.csv"
WORKER = (REPO_ROOT / "site" / "worker.js").read_text()


def worker_modules() -> list[str]:
    listed = re.search(r"const MODULES = \[(.*?)\];", WORKER, re.S).group(1)
    return [m.strip().strip('"') for m in listed.split(",")]


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return web.run(MASTER, "my_master.xlsx", TEMPLATE, "next_month.csv", out=tmp_path_factory.mktemp("web"),
                   log=lambda m: None)


def test_returns_what_the_page_shows(result):
    assert result["meta"].startswith("Data: my_master.xlsx · ")
    assert [h for h, _ in result["sections"]] == CONFIG["platforms"] + ["Planned posts", "How to read this"]
    assert {"Platform", "Type", "Finding", "Detail"} == set(result["summary"][0])
    assert result["css"] == report.CSS


def test_files_and_zip(result):
    files = result["files"]
    for f in ["recommendations.xlsx", "recommendations.html", "options.csv", "plan_scores.csv", "run_info.json"]:
        assert f in files
    assert all(f"figures/options_{p.lower()}.png" in files for p in CONFIG["platforms"])
    with zipfile.ZipFile(io.BytesIO(result["zip"])) as z:
        assert {n: z.read(n) for n in z.namelist()} == files
    info = json.loads(files["run_info.json"])
    assert info["master"] == "my_master.xlsx" and info["plan"]["file"] == "next_month.csv" and info["tag"] == "web"


def test_not_a_master(tmp_path):
    with pytest.raises(web.InputError, match="doesn't look like a master workbook"):
        web.run(REPO_ROOT / "data" / "Hidalga Technologies, Inc competitors.xlsx", "x.xlsx", out=tmp_path)
    fake = tmp_path / "fake.xlsx"
    fake.write_bytes(TEMPLATE.read_bytes())
    with pytest.raises(web.InputError, match="doesn't look like a master workbook"):
        web.run(fake, "fake.xlsx", out=tmp_path / "o")


def test_bad_plan(tmp_path):
    plan = tmp_path / "plan.csv"
    plan.write_text("platform,date,format\nLinkedIn,2026-12-01,Document\n")
    with pytest.raises(web.InputError, match=r"^plan\.csv: Plan file is missing columns: time, pillars, icp_segments"):
        web.run(MASTER, "m.xlsx", plan, "plan.csv", out=tmp_path / "o")


def test_worker_loads_every_module_a_run_needs(tmp_path):
    """The worker fetches a fixed list of modules; a run (with a plan, so scoring imports run too) must need no others."""
    listed = set(worker_modules())
    code = ("import sys; from modeling import web; "
            f"web.run({str(MASTER)!r}, 'm.xlsx', {str(TEMPLATE)!r}, 'p.csv', out={str(tmp_path)!r}, log=lambda m: None); "
            "print(' '.join(m.split('.', 1)[1] for m in sys.modules if m.startswith('modeling.')))")
    used = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert set(used.stdout.split()) | {"__init__"} == listed


def test_page_files_are_committed():
    """GitHub Pages serves the committed files of main, with case-sensitive paths: everything the page and the worker
    fetch must be tracked under exactly that name (a local server or macOS would hide a mistake in either)."""
    tracked = set(subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True,
                                 check=True).stdout.splitlines())
    page = (REPO_ROOT / "index.html").read_text()
    refs = re.findall(r'(?:href|src)="((?!https?:|data:|#)[^"]+)"', page) + ["site/worker.js"]
    refs += [f"modeling/{m}.py" for m in worker_modules()]
    refs += ["modeling/config.yaml", CONFIG["master"], ".nojekyll"]   # .nojekyll: keeps modeling/__init__.py
    assert [r for r in refs if r not in tracked] == []


def test_page_colours_match_the_report():
    tokens = lambda css: re.findall(r"--[a-z0-9]+:#[0-9a-f]{6}", css)
    page = (REPO_ROOT / "site" / "style.css").read_text()
    assert tokens(page) == tokens(report.CSS) and len(tokens(page)) == 18
