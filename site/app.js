// Page logic: pick files, hand them to the worker (site/worker.js), show the results as tabs and offer downloads.
const $ = (id) => document.getElementById(id);
const worker = new Worker("site/worker.js");
const state = { master: null, plan: null, useRepo: false, ready: false, running: false, repoMaster: null, urls: [] };
const TYPE_CLASS = { "Do more": "good", "Do less": "bad", "Promising, keep testing": "lean",
                     "Watch, tends to underperform": "lean", "Pulling ahead recently": "lean",
                     "Falling behind recently": "lean" };
const MIME = { xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", html: "text/html",
               zip: "application/zip" };

function setStatus(text) { $("status").textContent = text; }
function showError(text) { $("error").textContent = text; $("error").hidden = !text; }
function refresh() {
  $("run").disabled = !state.ready || state.running || !(state.master || state.useRepo);
  $("use-repo").disabled = !state.ready || state.running;
}

// ── Files ──────────────────────────────────────────────────────────────────
function choose(kind, file) {
  showError("");
  const ext = kind === "master" ? ".xlsx" : ".csv";
  if (file && !file.name.toLowerCase().endsWith(ext)) {
    showError(`${file.name}: the ${kind === "master" ? "master workbook" : "plan"} must be a ${ext} file.`);
    return;
  }
  state[kind] = file || null;
  if (kind === "master" && file) state.useRepo = false;
  const label = kind === "master" && state.useRepo ? `${state.repoMaster} (saved in the repo)` : file?.name;
  const chosen = $(`${kind}-chosen`);
  chosen.hidden = !label;
  chosen.textContent = label ? `✓ ${label}` : "";
  if (label) {
    const clear = document.createElement("button");
    clear.type = "button";
    clear.className = "link";
    clear.textContent = "remove";
    clear.addEventListener("click", () => { if (kind === "master") state.useRepo = false; choose(kind, null); });
    chosen.append(clear);
  }
  $(`${kind}-drop`).classList.toggle("done", Boolean(label));
  refresh();
}

for (const kind of ["master", "plan"]) {
  const drop = $(`${kind}-drop`), input = $(`${kind}-file`);
  input.addEventListener("change", () => { choose(kind, input.files[0]); input.value = ""; });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    if (e.dataTransfer.files.length) choose(kind, e.dataTransfer.files[0]);
  });
}
document.querySelectorAll("[data-pick]").forEach((b) => b.addEventListener("click", () => $(b.dataset.pick).click()));
$("use-repo").addEventListener("click", () => { state.useRepo = true; choose("master", null); });
// A file dropped anywhere else would make the browser open it and leave the page.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

// ── Run ────────────────────────────────────────────────────────────────────
let timer = null, statusText = "", started = 0;
$("run").addEventListener("click", async () => {
  showError("");
  state.running = true;
  refresh();
  started = Date.now();
  statusText = "Starting…";
  timer = setInterval(() => setStatus(`${statusText} (${Math.round((Date.now() - started) / 1000)} s)`), 1000);
  const pack = async (f) => f && { name: f.name, bytes: await f.arrayBuffer() };
  const msg = { type: "run", master: state.useRepo ? null : await pack(state.master), plan: await pack(state.plan) };
  worker.postMessage(msg, [msg.master?.bytes, msg.plan?.bytes].filter(Boolean));
});

function finish() {
  clearInterval(timer);
  state.running = false;
  refresh();
}

worker.onmessage = ({ data: m }) => {
  if (m.type === "status") {
    statusText = m.text;
    if (!state.running) setStatus(m.text);
  } else if (m.type === "ready") {
    state.ready = true;
    state.repoMaster = m.repoMaster.split("/").pop();
    $("repo-name").textContent = `(${state.repoMaster})`;
    setStatus("Ready. Add the master workbook, then make recommendations (takes about a minute).");
    refresh();
  } else if (m.type === "error") {
    finish();
    setStatus(m.fatal ? "Python could not start. Check the internet connection and reload the page." : "Stopped.");
    showError(m.text);
  } else if (m.type === "result") {
    finish();
    setStatus(`Done in ${Math.round((Date.now() - started) / 1000)} s. Pick a tab below, or download the files.`);
    showResults(m);
  }
};

// ── Results ────────────────────────────────────────────────────────────────
function el(tag, props = {}, text) {
  const e = Object.assign(document.createElement(tag), props);
  if (text !== undefined) e.textContent = text;
  return e;
}

function overview(rows) {
  const table = el("table", { className: "overview" });
  table.appendChild(el("thead")).appendChild(el("tr"))
    .append(...["Type", "Finding", "Detail"].map((h) => el("th", {}, h)));
  const body = table.appendChild(el("tbody"));
  let platform = null;
  for (const r of rows) {
    if (r.Platform !== platform) {
      platform = r.Platform;
      body.appendChild(el("tr", { className: "platform-row" })).append(el("td", { colSpan: 3 }, platform));
    }
    const tr = body.appendChild(el("tr"));
    tr.append(el("td", { className: `type ${TYPE_CLASS[r.Type] ?? ""}` }, r.Type), el("td", {}, r.Finding),
              el("td", { className: "note" }, r.Detail ?? ""));
  }
  const wrap = el("div", { className: "tbl" });
  wrap.append(table);
  const frag = document.createDocumentFragment();
  frag.append(el("h2", {}, "Overview"),
              el("p", { className: "note" }, "Key findings for each platform. The platform tabs explain each one " +
                                               "and show every choice on a chart."), wrap);
  return frag;
}

function download(id, name, bytes, type) {
  const url = URL.createObjectURL(new Blob([bytes], { type }));
  state.urls.push(url);
  Object.assign($(id), { href: url, download: name });
  return url;
}

function showResults(r) {
  state.urls.forEach((u) => URL.revokeObjectURL(u));
  state.urls = [];
  let style = $("report-css");
  if (!style) document.head.append(style = el("style", { id: "report-css" }));
  style.textContent = r.css;
  $("meta").textContent = r.meta;

  const day = new Date().toISOString().slice(0, 10);
  download("dl-xlsx", `${day}_recommendations.xlsx`, r.files["recommendations.xlsx"], MIME.xlsx);
  const reportUrl = download("dl-html", `${day}_recommendations.html`, r.files["recommendations.html"], MIME.html);
  download("dl-zip", `${day}_recommendations.zip`, r.zip, MIME.zip);
  $("open-report").onclick = () => window.open(reportUrl, "_blank");

  const tabs = $("tabs"), panels = $("panels");
  tabs.replaceChildren();
  panels.replaceChildren();
  const sections = [["Overview", null], ...r.sections];
  sections.forEach(([name, html], i) => {
    const tab = el("button", { type: "button", id: `tab-${i}`, tabIndex: i ? -1 : 0 }, name);
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-controls", `panel-${i}`);
    tab.setAttribute("aria-selected", String(!i));
    const panel = el("div", { id: `panel-${i}`, hidden: Boolean(i), tabIndex: 0 });
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", `tab-${i}`);
    if (html === null) panel.append(overview(r.summary));
    else panel.innerHTML = html;       // built by modeling/report.py, which escapes every value
    tab.addEventListener("click", () => select(i));
    tabs.append(tab);
    panels.append(panel);
  });
  tabs.onkeydown = (e) => {
    const n = sections.length, cur = [...tabs.children].findIndex((t) => t.getAttribute("aria-selected") === "true");
    const next = { ArrowRight: (cur + 1) % n, ArrowLeft: (cur - 1 + n) % n, Home: 0, End: n - 1 }[e.key];
    if (next === undefined) return;
    e.preventDefault();
    select(next);
    tabs.children[next].focus();
  };
  $("results").hidden = false;
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function select(i) {
  [...$("tabs").children].forEach((t, j) => {
    t.setAttribute("aria-selected", String(i === j));
    t.tabIndex = i === j ? 0 : -1;
  });
  [...$("panels").children].forEach((p, j) => { p.hidden = i !== j; });
}
