// Page logic: pick files, hand them to the worker (site/worker.js), show the results as tabs and offer downloads.
const $ = (id) => document.getElementById(id);
const worker = new Worker("site/worker.js");
const state = { master: null, plan: null, exports: [], useRepo: false, ready: false, running: false, repoMaster: null,
                urls: [] };
const TYPE_CLASS = { "Do more": "good", "Do less": "bad", "Promising, keep testing": "lean",
                     "Watch, tends to underperform": "lean", "Pulling ahead recently": "lean",
                     "Falling behind recently": "lean" };
const MIME = { xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", html: "text/html",
               zip: "application/zip" };

const today = () => {                    // local date, YYYY-MM-DD
  const d = new Date();
  return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
};
$("pull-date").value = today();

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
// New exports: several files, any subset; which export each one is gets worked out in Python from its sheets.
function addExports(files) {
  showError("");
  const bad = [...files].filter((f) => !/\.xlsx?$/i.test(f.name)).map((f) => f.name);
  if (bad.length) showError(`${bad.join(", ")}: exports are .xlsx or .xls files.`);
  for (const f of files) {
    if (/\.xlsx?$/i.test(f.name) && !state.exports.some((g) => g.name === f.name)) state.exports.push(f);
  }
  renderExports();
}

function renderExports() {
  const list = $("exports-chosen");
  list.replaceChildren(...state.exports.map((f, i) => {
    const li = el("li", {}, `✓ ${f.name}`);
    const clear = el("button", { type: "button", className: "link" }, "remove");
    clear.addEventListener("click", () => { state.exports.splice(i, 1); renderExports(); });
    li.append(clear);
    return li;
  }));
  list.hidden = $("pull-row").hidden = !state.exports.length;
  $("exports-drop").classList.toggle("done", state.exports.length > 0);
}

{
  const drop = $("exports-drop"), input = $("exports-file");
  input.addEventListener("change", () => { addExports(input.files); input.value = ""; });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    addExports(e.dataTransfer.files);
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
  $("results").hidden = true;            // never leave an earlier run's results under a new run's error
  state.running = true;
  refresh();
  started = Date.now();
  statusText = "Starting…";
  timer = setInterval(() => setStatus(`${statusText} (${Math.round((Date.now() - started) / 1000)} s)`), 1000);
  const pack = async (f) => f && { name: f.name, bytes: await f.arrayBuffer() };
  const msg = { type: "run", master: state.useRepo ? null : await pack(state.master), plan: await pack(state.plan),
                exports: await Promise.all(state.exports.map(pack)), pullDate: $("pull-date").value || today() };
  worker.postMessage(msg, [msg.master?.bytes, msg.plan?.bytes, ...msg.exports.map((f) => f.bytes)].filter(Boolean));
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
    const secs = Math.round((Date.now() - started) / 1000);
    setStatus(m.new_master ? `Done in ${secs} s. Made ${m.new_master.name}: download it below and keep it.`
                           : `Done in ${secs} s. Pick a tab below, or download the files.`);
    showResults(m);
    if (m.analysis_error) {
      showError(`The new master was built, but the recommendations couldn't run on it: ${m.analysis_error}`);
    }
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

function table(cols, rows, cls = {}) {
  const t = el("table", { className: "update" });
  const numeric = (c) => rows.length > 0 && typeof rows[0][c] === "number";
  t.appendChild(el("thead")).appendChild(el("tr"))
    .append(...cols.map((c) => el("th", { className: numeric(c) ? "num" : "" }, c)));
  const body = t.appendChild(el("tbody"));
  for (const r of rows) {
    body.appendChild(el("tr")).append(...cols.map((c) => {
      const v = r[c] ?? "", empty = v === "";
      const td = el("td", { className: [numeric(c) ? "num" : "", c === "Date" ? "nowrap" : "",
                                         empty && cls.flagEmpty ? "missing" : ""].join(" ").trim() },
                    empty && cls.flagEmpty ? "none" : String(v));
      td.dataset.label = c;                // shown before the value when rows stack on phones
      return td;
    }));
  }
  const wrap = el("div", { className: "tbl" });
  wrap.append(t);
  return wrap;
}

function dataUpdate(u, newMaster, masterName) {
  const frag = document.createDocumentFragment();
  const posts = u.new_posts.length, pulled = `exports pulled ${u.pull_date}`;
  frag.append(el("h2", {}, "Data update"));
  if (newMaster) {
    frag.append(el("p", { className: "update-lead" },
                   `Added ${posts} new post${posts === 1 ? "" : "s"} and the other new rows below to make ` +
                   `${newMaster.name} (${pulled}). The results in the other tabs use it.`));
    const btn = el("a", { className: "btn btn-key", href: $("dl-master").href, download: newMaster.name },
                   `Download ${newMaster.name}`);
    frag.append(btn, el("p", { className: "note" },
      "Keep this file: next time, drop it into box 1. To make it the master saved in the repo, add it to data/ " +
      "and set master: in modeling/config.yaml."));
  } else if (u.added.length) {
    frag.append(el("p", { className: "update-lead" },
                   `Nothing new: everything in these exports is already in ${masterName}, so no new master was ` +
                   `made. The results use ${masterName}.`));
  }
  if (u.warnings.length) {
    const ul = el("ul", { className: "warn" });
    ul.append(...u.warnings.map((w) => el("li", {}, w)));
    frag.append(ul);
  }
  frag.append(el("h3", {}, "Files"),
              table(["File", "Export"], u.files.map((f) => ({ File: f.file,
                                                               Export: f.used ? f.export : `${f.export}: not used` }))));
  if (u.missing.length) {
    frag.append(el("p", { className: "note" }, `Not provided this time: ${u.missing.join("; ")}.`));
  }
  if (u.added.length) {
    frag.append(el("h3", {}, "Rows added"),
                table(["Sheet", "In the master", "In the exports", "Added", "Already there"],
                      u.added.map((a) => ({ Sheet: a.sheet, "In the master": a.in_master,
                                            "In the exports": a.in_exports, Added: a.added,
                                            "Already there": a.already_there }))));
    if (u.logged) {
      frag.append(el("p", { className: "note" },
        `${u.logged} values in the exports differ from the master's (usually newer numbers for older posts). ` +
        "The master's values are kept; the differences are listed in the master's \"Unapplied Updates (log)\" sheet."));
    }
  }
  if (posts) {
    frag.append(el("h3", {}, "New posts"),
                table(["Platform", "Date", "Post", "Format", "Pillar", "ICP segments", "Numbers from"], u.new_posts,
                      { flagEmpty: true }));
  }
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

  const day = today(), ran = Boolean(r.files["recommendations.xlsx"]);
  $("dl-master").hidden = !r.new_master;
  if (r.new_master) {
    download("dl-master", r.new_master.name, r.new_master.bytes, MIME.xlsx);
    $("dl-master").textContent = `Download new master (${r.new_master.name})`;
  }
  for (const id of ["dl-xlsx", "dl-html", "dl-zip", "open-report"]) $(id).hidden = !ran;
  if (ran) {
    download("dl-xlsx", `${day}_recommendations.xlsx`, r.files["recommendations.xlsx"], MIME.xlsx);
    const reportUrl = download("dl-html", `${day}_recommendations.html`, r.files["recommendations.html"], MIME.html);
    download("dl-zip", `${day}_recommendations.zip`, r.zip, MIME.zip);
    $("open-report").onclick = () => window.open(reportUrl, "_blank");
  }

  const tabs = $("tabs"), panels = $("panels");
  tabs.replaceChildren();
  panels.replaceChildren();
  const sections = [...(r.update ? [["Data update", "update"]] : []), ...(ran ? [["Overview", null]] : []),
                    ...r.sections];
  sections.forEach(([name, html], i) => {
    const tab = el("button", { type: "button", id: `tab-${i}`, tabIndex: i ? -1 : 0 }, name);
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-controls", `panel-${i}`);
    tab.setAttribute("aria-selected", String(!i));
    const panel = el("div", { id: `panel-${i}`, hidden: Boolean(i), tabIndex: 0 });
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", `tab-${i}`);
    if (html === "update") panel.append(dataUpdate(r.update, r.new_master, r.meta.replace(/^Data: (.*?)( ·.*)?$/, "$1")));
    else if (html === null) panel.append(overview(r.summary));
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
