"""Tests for the web page's master update (modeling/web.py with exports). Run from the repo root:
    python -m pytest modeling/tests -q
The September exports are already in v3, so v1 + the September Loomly workbook is used to check a real update: it must
rebuild the committed v2."""
import io
import json
import zipfile

import pandas as pd
import pytest

from modeling import web
from modeling.features import REPO_ROOT, load_config

DATA = REPO_ROOT / "data"
CONFIG = load_config()
SOCIAL = DATA / "092026-Social-Media-Data.xlsx"
LI_CONTENT = DATA / "hidalga-technologies_content_1788892250539.xls"
LI_VISITORS = DATA / "hidalga-technologies_visitors_1788892288178.xls"
LI_FOLLOWERS = DATA / "hidalga-technologies_followers_1788892297560.xls"
QUIET = dict(log=lambda m: None)


def ex(*paths):
    return [(p, p.name) for p in paths]


@pytest.fixture
def linkedin_only(monkeypatch):
    """Analyse one platform only, to keep the tests that aren't about the results quick."""
    monkeypatch.setattr(web, "load_config", lambda: dict(load_config(), platforms=["LinkedIn"]))


@pytest.fixture(scope="module")
def v2(tmp_path_factory):
    return web.run(DATA / "Hidalga_Master_Data_v1.xlsx", "Hidalga_Master_Data_v1.xlsx", exports=ex(SOCIAL),
                   pull_date="2026-10-25", out=tmp_path_factory.mktemp("v2"), **QUIET)


def test_update_rebuilds_v2(v2):
    u = v2["update"]
    assert [(a["sheet"], a["added"]) for a in u["added"]] == [
        ("Social Posts (unified)", 17), ("Follower Counts", 3), ("Loomly Clicks", 6)]  # merge_decisions.csv
    assert v2["new_master"]["name"] == "Hidalga_Master_Data_v2.xlsx"
    built = pd.read_excel(io.BytesIO(v2["new_master"]["bytes"]), sheet_name=None)
    committed = pd.read_excel(DATA / "Hidalga_Master_Data_v2.xlsx", sheet_name=None)
    assert list(built) == list(committed)
    for sheet, df in committed.items():
        drop = ["MERGED_AT", "INCOMING_FILE"]                     # merge time; the cleaned file's scratch name
        assert built[sheet].drop(columns=drop, errors="ignore").equals(df.drop(columns=drop, errors="ignore")), sheet


def test_update_reports_files_posts_and_warnings(v2):
    u = v2["update"]
    assert u["files"] == [{"file": SOCIAL.name, "export": web.EXPORTS["social_xlsx"], "used": True}]
    assert u["missing"] == [web.EXPORTS["li_content"], web.EXPORTS["li_visitors"]]
    assert len(u["new_posts"]) == 17 and {p["Platform"] for p in u["new_posts"]} == set(CONFIG["platforms"])
    li_loomly = sum(p["Platform"] == "LinkedIn" and p["Numbers from"] == "Loomly" for p in u["new_posts"])
    assert any(w.startswith(f"{li_loomly} new LinkedIn post(s) use Loomly's numbers because") for w in u["warnings"])
    untagged = sum(not p["Pillar"] or not p["ICP segments"] for p in u["new_posts"])
    assert untagged and any(w.startswith(f"{untagged} new post(s) have no pillar") for w in u["warnings"])
    assert "new_master_path" not in u and u["pull_date"] == "2026-10-25"


def test_results_use_the_new_master(v2):
    assert v2["analysis_error"] is None and v2["sections"]
    assert v2["meta"].startswith("Data: Hidalga_Master_Data_v2.xlsx · ")
    info = json.loads(v2["files"]["run_info.json"])
    assert info["master"] == "Hidalga_Master_Data_v2.xlsx"
    assert info["update"]["new_master"] == "Hidalga_Master_Data_v2.xlsx" and info["update"]["pull_date"] == "2026-10-25"
    assert [e["file"] for e in info["update"]["exports"]] == [SOCIAL.name]
    with zipfile.ZipFile(io.BytesIO(v2["zip"])) as z:
        assert z.read("Hidalga_Master_Data_v2.xlsx") == v2["new_master"]["bytes"]


def test_nothing_new_keeps_the_master(tmp_path, linkedin_only):
    r = web.run(DATA / "Hidalga_Master_Data_v3.xlsx", "Hidalga_Master_Data_v3.xlsx",
                exports=ex(SOCIAL, LI_CONTENT, LI_VISITORS, LI_FOLLOWERS), out=tmp_path, **QUIET)
    assert r["new_master"] is None and sum(a["added"] for a in r["update"]["added"]) == 0
    assert r["update"]["missing"] == [] and r["update"]["new_posts"] == []
    assert [f["used"] for f in r["update"]["files"]] == [True, True, True, False]       # followers: not used
    assert r["meta"].startswith("Data: Hidalga_Master_Data_v3.xlsx · ") and r["sections"]
    assert not any(n.endswith(".xlsx") and n.startswith("Hidalga_Master") for n in r["files"])


def test_linkedin_exports_only(tmp_path, linkedin_only):
    r = web.run(DATA / "Hidalga_Master_Data_v3.xlsx", "v3.xlsx", exports=ex(LI_CONTENT, LI_VISITORS), out=tmp_path,
                **QUIET)
    assert r["update"]["missing"] == [web.EXPORTS["social_xlsx"]]
    assert {a["sheet"] for a in r["update"]["added"]} == {"LinkedIn Posts (detail)", "LinkedIn Daily Metrics",
                                                         "Visitor Daily Metrics", "Visitor Demographics"}


def test_followers_export_alone_adds_nothing(tmp_path, linkedin_only):
    r = web.run(DATA / "Hidalga_Master_Data_v3.xlsx", "v3.xlsx", exports=ex(LI_FOLLOWERS), out=tmp_path, **QUIET)
    assert r["new_master"] is None and r["update"]["added"] == []
    assert any("followers export isn't used" in w for w in r["update"]["warnings"])


def test_unknown_and_duplicate_exports(tmp_path):
    with pytest.raises(web.InputError, match="Not one of the exports the tool knows: competitors.xlsx"):
        web.run(DATA / "Hidalga_Master_Data_v3.xlsx", "v3.xlsx",
                exports=[(DATA / "Hidalga Technologies, Inc competitors.xlsx", "competitors.xlsx")], out=tmp_path)
    with pytest.raises(web.InputError, match="Two files look like the Loomly social media workbook"):
        web.run(DATA / "Hidalga_Master_Data_v3.xlsx", "v3.xlsx",
                exports=ex(SOCIAL, DATA / "082026-Social-Media-Data.xlsx"), out=tmp_path)


def test_new_format_still_returns_the_new_master(tmp_path, monkeypatch):
    """A format the config doesn't list stops the analysis, but the update and the new master still come back."""
    cfg = load_config()
    cfg["platforms"] = ["Facebook"]
    cfg["formats"] = dict(cfg["formats"], Facebook=[f for f in cfg["formats"]["Facebook"] if f != "Image"])
    monkeypatch.setattr(web, "load_config", lambda: cfg)
    r = web.run(DATA / "Hidalga_Master_Data_v1.xlsx", "Hidalga_Master_Data_v1.xlsx", exports=ex(SOCIAL),
                out=tmp_path, **QUIET)
    assert r["analysis_error"].startswith("Facebook has formats not listed") and r["sections"] == []
    assert r["new_master"]["name"] == "Hidalga_Master_Data_v2.xlsx"
    assert any(w.startswith("New post format(s) the tool doesn't know yet: Image (Facebook)") for w in
               r["update"]["warnings"])


def test_next_master_name():
    assert web.next_master_name("Hidalga_Master_Data_v3.xlsx") == "Hidalga_Master_Data_v4.xlsx"
    assert web.next_master_name("Hidalga_Master_Data_v9.xlsx") == "Hidalga_Master_Data_v10.xlsx"
    assert web.next_master_name("master copy.xlsx") == "master copy_updated.xlsx"
