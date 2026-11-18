// Runs the recommendation tool (modeling/web.py) in Pyodide, off the page's main thread.
// Messages in:  {type: "run", master: {name, bytes} | null (null = the master named in config.yaml), plan: {name, bytes} | null}
// Messages out: {type: "status", text} · {type: "ready", repoMaster} · {type: "result", ...} · {type: "error", text}
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/";
const PACKAGES = ["numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "matplotlib", "pyyaml", "micropip"];
const PYPI = ["openpyxl==3.1.5"];
// modeling.web and every module it imports (modeling/tests/test_web.py checks this list).
const MODULES = ["__init__", "web", "pipeline", "features", "recommend", "evaluate", "targets", "models", "classify",
                 "plots", "report", "run_regression"];
const ROOT = new URL("../", self.location.href);

importScripts(PYODIDE + "pyodide.js");

const status = (text) => postMessage({ type: "status", text });

async function fetchBytes(path) {
  const r = await fetch(new URL(path, ROOT));
  if (!r.ok) throw new Error(`Could not load ${path} (${r.status})`);
  return new Uint8Array(await r.arrayBuffer());
}

async function setup() {
  status("Loading Python (the first visit downloads about 40 MB; later visits use the browser's copy)…");
  const py = await loadPyodide({ indexURL: PYODIDE });
  status("Loading the analysis libraries…");
  await py.loadPackage(PACKAGES);
  await py.pyimport("micropip").install(PYPI);
  status("Loading the recommendation tool…");
  py.FS.mkdirTree("/repo/modeling");
  py.FS.mkdirTree("/in");
  const files = [...MODULES.map((m) => `modeling/${m}.py`), "modeling/config.yaml"];
  const data = await Promise.all(files.map(fetchBytes));
  files.forEach((f, i) => py.FS.writeFile(`/repo/${f}`, data[i]));
  py.runPython(`
import os, sys
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, "/repo")
from modeling import web
from modeling.features import load_config
`);
  return py;
}

const ready = setup().then((py) => {
  postMessage({ type: "ready", repoMaster: py.runPython("load_config()['master']") });
  return py;
});
ready.catch((e) => postMessage({ type: "error", text: `Could not start Python: ${e.message}`, fatal: true }));

self.onmessage = async ({ data: msg }) => {
  if (msg.type !== "run") return;
  const py = await ready;
  try {
    let masterPath, masterName;
    if (msg.master) {
      masterPath = "/in/master.xlsx";
      masterName = msg.master.name;
      py.FS.writeFile(masterPath, new Uint8Array(msg.master.bytes));
    } else {
      masterName = py.runPython("load_config()['master']");
      masterPath = "/in/repo_master.xlsx";
      status(`Downloading ${masterName}…`);
      py.FS.writeFile(masterPath, await fetchBytes(masterName));
      masterName = masterName.split("/").pop();
    }
    let planPath = null, planName = null;
    if (msg.plan) {
      planPath = "/in/plan.csv";
      planName = msg.plan.name;
      py.FS.writeFile(planPath, new Uint8Array(msg.plan.bytes));
    }
    status("Analysing posts…");
    const web = py.pyimport("modeling.web");
    let res;
    try {
      res = web.run.callKwargs({ master_path: masterPath, master_name: masterName, plan_path: planPath,
                                 plan_name: planName, log: (m) => status(`Analysing posts… ${m.trim()}`) });
    } catch (e) {
      const text = String(e.message);
      const input = text.match(/InputError: ([\s\S]*?)\s*$/);
      postMessage({ type: "error", text: input ? input[1] : `Something went wrong: ${text.trim().split("\n").pop()}` });
      return;
    }
    const out = res.toJs({ dict_converter: Object.fromEntries });
    res.destroy();
    const transfer = [out.zip.buffer, ...Object.values(out.files).map((b) => b.buffer)];
    postMessage({ type: "result", ...out }, transfer);
  } catch (e) {
    postMessage({ type: "error", text: e.message });
  }
};
