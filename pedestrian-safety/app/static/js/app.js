/* =====================================================================
   Crowd Safety Console - front end (no build step, plain JS)
   ===================================================================== */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const el = (tag, attrs = {}, html = "") => { const e = document.createElement(tag); Object.assign(e, attrs); if (html) e.innerHTML = html; return e; };
const svgEl = (tag, attrs = {}) => { const e = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
const nf = (v, d = 0) => v == null || Number.isNaN(v) ? "–" : Number(v).toLocaleString("en-IN", { maximumFractionDigits: d, minimumFractionDigits: d });
const pct = (v) => v == null ? "–" : `${Math.round(v * 100)}%`;
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const LEVELS = ["Low", "Medium", "High", "Very High"];
const LV_COLOR = ["var(--lv0)", "var(--lv1)", "var(--lv2)", "var(--lv3)"];
const RISK_ICON = { "Normal": "🟢", "Watch": "🟡", "Act now": "🔴" };
const REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;

async function api(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({ error: `Server returned ${r.status}` }));
  if (!r.ok || j.error) throw new Error(j.error || `Request failed (${r.status})`);
  return j;
}
function fmtWhen(w) {
  const d = new Date(w);
  return d.toLocaleString("en-AU", { weekday: "short", day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", hour12: false });
}
function countUp(node, to, dec = 0, ms = 900) {
  if (REDUCE || to == null) { node.textContent = nf(to, dec); return; }
  const t0 = performance.now();
  const tick = (t) => { const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3);
    node.textContent = nf(to * e, dec); if (k < 1) requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
}

/* ---------------------------------------------------------------- state */
const S = { meta: null, when: null, page: "overview", loaded: {}, sim: null, lastUpload: null };

/* ---------------------------------------------------------------- router */
const PAGES = { overview: loadOverview, plan: loadPlan, watch: loadWatch, camera: initCamera, agent: initAgent, results: loadResults };
function go(page) { location.hash = page; }
function route() {
  const page = (location.hash || "#overview").slice(1);
  if (!PAGES[page]) return go("overview");
  S.page = page;
  $$(".page").forEach((p) => (p.hidden = p.id !== `page-${page}`));
  $$(".rail nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === page));
  const sec = $(`#page-${page}`);
  $("#page-title").textContent = sec.dataset.title;
  document.title = `${sec.dataset.title} – Crowd Safety Console`;
  sec.style.animation = "none"; sec.offsetHeight; sec.style.animation = "";
  if (S.sim) page === "overview" ? S.sim.start() : S.sim.stop();
  const key = page + "|" + S.when;
  if (!S.loaded[key]) { S.loaded[key] = true; PAGES[page]().catch(showError(sec)); }
  window.scrollTo({ top: 0 });
}
const showError = (sec) => (e) => {
  console.error(e);
  const box = el("p", { className: "error" }, esc(e.message));
  sec.prepend(box); setTimeout(() => box.remove(), 8000);
};

/* ---------------------------------------------------------------- boot */
async function boot() {
  S.meta = await api("/api/meta");
  const m = S.meta;
  // replay clock
  const sel = $("#clock-moment");
  sel.innerHTML = `<option value="">Jump to…</option>` + m.moments.map((x) => `<option value="${x.when}">${esc(x.label)}</option>`).join("");
  const t = $("#clock-time");
  t.min = m.data_start.replace(" ", "T"); t.max = "2030-12-31T23:00";
  setClock(m.default_when, false);
  sel.addEventListener("change", () => { if (sel.value) setClock(sel.value); sel.value = ""; });
  t.addEventListener("change", () => t.value && setClock(t.value));
  // agent pill
  const pill = $("#agent-pill");
  pill.classList.add(m.agent.llm ? "on" : "off");
  pill.querySelector("span").textContent = m.agent.llm ? `Agent: Gemini` : "Agent: offline plan";
  pill.title = m.agent.llm ? m.agent.model : "No GEMINI_API_KEY in .env: the fixed backup plan answers";
  // camera selects
  for (const id of ["#watch-cam", "#cam-select", "#a-cam"]) {
    const s = $(id);
    s.insertAdjacentHTML("beforeend", m.cameras.map((c) => `<option value="${c.camera_id}">${esc(c.camera_id)} (${esc(c.sensor_name)})</option>`).join(""));
  }
  $$("[data-go]").forEach((b) => b.addEventListener("click", () => go(b.dataset.go)));
  // hero simulation
  S.sim = new CrowdSim($("#crowd"));
  window.addEventListener("hashchange", route);
  route();
}
function setClock(w, refresh = true) {
  S.when = w.slice(0, 13) + ":00";
  $("#clock-time").value = S.when;
  if (refresh) { S.loaded = {}; route(); }
}

/* ================================================================ OVERVIEW */
async function loadOverview() {
  const d = await api(`/api/status?when=${S.when}`);
  $("#status-when").textContent = `Cameras at ${fmtWhen(S.when)}`;
  const ev = d.day.events.map((e) => e.event_name);
  if (d.day.public_holiday) ev.unshift(`${d.day.public_holiday} (public holiday)`);
  $("#status-day").textContent = ev.length ? `Today: ${ev.join(", ")}` : "No listed events today.";
  const o = d.overall;
  const ov = $("#overall");
  ov.className = `overall risk r${o.risk_code}`;
  ov.innerHTML = `<span class="dot"></span>Overall: ${esc(o.risk_level)}`;
  // simulation pressure follows the worst camera
  const maxP = Math.max(...d.cameras.map((c) => c.forecast.p_very_high || 0));
  S.sim.setPressure([0.12, 0.55, 0.95][o.risk_code] * 0.7 + maxP * 0.3);

  const box = $("#cams"); box.innerHTML = "";
  d.cameras.forEach((c, i) => {
    const lv = LEVELS.indexOf(c.forecast.predicted_level);
    const card = el("article", { className: `cam risk r${c.risk.risk_code}` });
    card.style.animation = `cellIn .5s var(--ease) ${i * 70}ms backwards`;
    card.innerHTML = `
      <span class="tag">${esc(c.risk.risk_level)}</span>
      <h4>${esc(c.camera.camera_id)}</h4>
      <p class="sub">${esc(c.camera.camera_name)}</p>
      <div class="meter" aria-label="Forecast level ${esc(c.forecast.predicted_level)}">${[0, 1, 2, 3].map((k) => `<i class="${k <= lv ? "on" : ""} l${lv}"></i>`).join("")}</div>
      <div class="row"><span>Forecast<br><b>${esc(c.forecast.predicted_level)}</b></span>
        <span>Chance of Very High<br><b class="num" data-p="${c.forecast.p_very_high}">0%</b></span></div>
      <div class="row"><span>This hour ${c.actual ? `<b>${nf(c.actual.count)}</b>` : "<b>–</b>"}</span><span>usual <b>${nf(c.forecast.usual_people_this_hour)}</b></span></div>`;
    box.append(card);
    const b = card.querySelector("[data-p]");
    countUpPct(b, c.forecast.p_very_high);
  });
}
function countUpPct(node, v) {
  if (REDUCE) { node.textContent = pct(v); return; }
  const t0 = performance.now();
  const tick = (t) => { const k = Math.min(1, (t - t0) / 900), e = 1 - Math.pow(1 - k, 3);
    node.textContent = `${Math.round(v * 100 * e)}%`; if (k < 1) requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
}

/* ================================================================ PLAN AHEAD */
async function loadPlan() {
  const input = $("#plan-date");
  if (!input.dataset.bound) {
    input.dataset.bound = 1;
    input.addEventListener("change", () => input.value && renderPlan(input.value).catch(showError($("#page-plan"))));
  }
  input.value = S.when.slice(0, 10);
  await renderPlan(input.value);
}
async function renderPlan(date) {
  const grid = $("#plan-grid");
  grid.innerHTML = `<div class="status-line" style="grid-column:1/-1"><span class="spin"></span>Forecasting 24 hours for every camera…</div>`;
  const d = await api(`/api/plan?date=${date}`);
  const ev = d.day.events.map((e) => `${e.event_name} (${e.start_hour}:00–${e.end_hour}:00)`);
  if (d.day.public_holiday) ev.unshift(`${d.day.public_holiday} (public holiday)`);
  $("#plan-day").textContent = `${d.weekday}${ev.length ? ": " + ev.join(", ") : ", no listed events"}`;
  grid.innerHTML = `<div></div>` + Array.from({ length: 24 }, (_, h) => `<div class="h">${String(h).padStart(2, "0")}</div>`).join("");
  const tip = $("#plan-tip");
  d.rows.forEach((row, r) => {
    grid.append(el("div", { className: "name" }, `${esc(row.camera.camera_id)}<small>${esc(row.camera.sensor_name)}</small>`));
    row.hours.forEach((h) => {
      const lv = LEVELS.indexOf(h.level);
      const c = el("div", { className: `cell l${lv}${h.warning ? " warn" : ""}${h.replay ? "" : " future"}` });
      c.style.setProperty("--i", h.hour + r * 1.5);
      c.setAttribute("role", "cell");
      c.setAttribute("aria-label", `${row.camera.camera_id} ${h.hour}:00 forecast ${h.level}`);
      c.addEventListener("pointerenter", () => {
        tip.hidden = false;
        tip.innerHTML = `<b>${esc(row.camera.camera_id)}, ${String(h.hour).padStart(2, "0")}:00</b>
          Forecast <strong>${esc(h.level)}</strong><br>Chance of Very High ${pct(h.p_very_high)}<br>
          Usual ${nf(h.usual)} people / hour${h.actual_level ? `<br>What happened: ${esc(h.actual_level)}` : ""}
          ${h.warning ? "<br><strong>Early warning raised</strong>" : ""}`;
      });
      c.addEventListener("pointermove", (e) => { tip.style.left = Math.min(e.clientX + 14, innerWidth - 220) + "px"; tip.style.top = e.clientY + 14 + "px"; });
      c.addEventListener("pointerleave", () => (tip.hidden = true));
      grid.append(c);
    });
  });
}

/* ================================================================ LIVE WATCH */
const W = { data: null, i: 0, timer: null, logged: new Set() };
async function loadWatch() {
  const sel = $("#watch-cam");
  if (!sel.dataset.bound) {
    sel.dataset.bound = 1;
    sel.addEventListener("change", () => loadWatchData().catch(showError($("#page-watch"))));
    $("#w-play").addEventListener("click", togglePlay);
    $("#w-next").addEventListener("click", () => { stopPlay(); setHour(Math.min(23, W.i + 1)); });
    $("#w-back").addEventListener("click", () => { stopPlay(); setHour(Math.max(0, W.i - 1)); });
    let rz;
    window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => { if (W.data) { W.logged = new Set(); $("#w-log").innerHTML = ""; drawChart(); setHour(W.i, true); } }, 150); });
  }
  await loadWatchData();
}
async function loadWatchData() {
  stopPlay();
  const day = S.when.slice(0, 10);
  $("#w-time").textContent = "Loading…";
  W.data = await api(`/api/watch?location=${$("#watch-cam").value}&when=${day}T23:00&hours=24`);
  $("#w-sensor").textContent = `Sensor ${W.data.sensor}, ${fmtWhen(day + "T00:00").split(",").slice(0, 2).join(",")}`;
  W.logged = new Set(); $("#w-log").innerHTML = "";
  drawChart();
  setHour(parseInt(S.when.slice(11, 13), 10), true);
}
function drawChart() {
  const svg = $("#w-chart"), d = W.data, s = d.series;
  const w = svg.clientWidth || 800, h = svg.clientHeight || 360, m = { l: 52, r: 16, t: 14, b: 30 };
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`); svg.innerHTML = "";
  const max = Math.max(d.lines.p95 * 1.12, ...s.map((p) => Math.max(p.actual || 0, p.usual || 0) * 1.08));
  const x = (i) => m.l + (i / (s.length - 1)) * (w - m.l - m.r);
  const y = (v) => h - m.b - (v / max) * (h - m.t - m.b);
  W.x = x; W.y = y; W.h = h; W.m = m;
  const defs = svgEl("defs");
  defs.innerHTML = `<linearGradient id="gradA" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#5aa7ff" stop-opacity=".28"/><stop offset="1" stop-color="#5aa7ff" stop-opacity="0"/></linearGradient>
    <clipPath id="reveal"><rect id="reveal-r" x="0" y="0" width="0" height="${h}"/></clipPath>`;
  svg.append(defs);
  for (let k = 0; k <= 4; k++) {
    const v = (max / 4) * k;
    svg.append(svgEl("line", { class: "grid", x1: m.l, x2: w - m.r, y1: y(v), y2: y(v) }));
    const t = svgEl("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }); t.textContent = nf(v); svg.append(t);
  }
  s.forEach((p, i) => { if (i % 3 === 0) { const t = svgEl("text", { x: x(i), y: h - 8, "text-anchor": "middle" }); t.textContent = `${String(p.hour).padStart(2, "0")}:00`; svg.append(t); } });
  svg.append(svgEl("line", { class: "vhline", x1: m.l, x2: w - m.r, y1: y(d.lines.p95), y2: y(d.lines.p95) }));
  const vt = svgEl("text", { x: w - m.r, y: y(d.lines.p95) - 6, "text-anchor": "end", fill: "var(--act)" }); vt.textContent = "Very High line"; vt.style.fill = "#ef8a7f"; svg.append(vt);
  const path = (key) => s.map((p, i) => p[key] == null ? null : `${x(i)},${y(p[key])}`).filter(Boolean);
  if (!s.some((p) => p.actual != null)) { const t = svgEl("text", { x: (m.l + w - m.r) / 2, y: h / 2, "text-anchor": "middle" }); t.textContent = "No sensor data for this day: showing the usual pattern only"; svg.append(t); }
  svg.append(svgEl("polyline", { class: "usual", points: path("usual").join(" ") }));
  // actual line, split wherever the sensor has no data for an hour
  const runs = []; let cur = [];
  s.forEach((p, i) => { if (p.actual == null) { if (cur.length) runs.push(cur); cur = []; } else cur.push(i); });
  if (cur.length) runs.push(cur);
  const g = svgEl("g", { "clip-path": "url(#reveal)" });
  runs.forEach((run) => {
    const pts = run.map((i) => `${x(i)},${y(s[i].actual)}`).join(" ");
    if (run.length > 1) g.append(svgEl("polygon", { class: "area", points: `${x(run[0])},${y(0)} ${pts} ${x(run[run.length - 1])},${y(0)}` }));
    g.append(run.length > 1 ? svgEl("polyline", { class: "actual", points: pts }) : svgEl("circle", { cx: x(run[0]), cy: y(s[run[0]].actual), r: 2.5, fill: "#5aa7ff" }));
  });
  svg.append(g);
  W.dots = svgEl("g"); svg.append(W.dots);
  W.ph = svgEl("line", { class: "playhead", y1: m.t, y2: h - m.b }); svg.append(W.ph);
  W.phDot = svgEl("circle", { class: "ph-dot", r: 4.5 }); svg.append(W.phDot);
  W.dotsDrawn = new Set();
}
function setHour(i, instant = false) {
  const d = W.data; if (!d) return;
  W.i = i; const s = d.series, p = s[i], x = W.x(i);
  const r = $("#reveal-r");
  r.style.transition = instant || REDUCE ? "none" : "width .55s cubic-bezier(.2,.8,.2,1)";
  r.setAttribute("width", x + 1);
  W.ph.setAttribute("x1", x); W.ph.setAttribute("x2", x);
  W.phDot.setAttribute("cx", x); W.phDot.setAttribute("cy", W.y(p.actual ?? p.usual ?? 0));
  $("#w-time").textContent = p.label;
  // warning dots for every hour reached so far
  for (let k = 0; k <= i; k++) {
    const q = s[k];
    if (q.warning && !W.dotsDrawn.has(k)) {
      W.dotsDrawn.add(k);
      W.dots.append(svgEl("circle", { class: `wdot${q.actual_level === "Very High" ? " hit" : ""}`, cx: W.x(k), cy: W.y(q.actual ?? q.usual), r: 6 }));
    }
    if (q.warning && !W.logged.has(k)) {
      W.logged.add(k);
      const hit = q.actual_level === "Very High";
      const li = el("li", {}, `<b>${String((q.hour + 23) % 24).padStart(2, "0")}:00</b> warned for ${String(q.hour).padStart(2, "0")}:00
        <em class="${hit ? "ok" : "miss"}">${q.actual_level ? (hit ? "came true" : `was ${esc(q.actual_level)}`) : "future"}</em>`);
      $("#w-log").prepend(li);
    }
  }
  if (!W.logged.size) $("#w-log").innerHTML = `<li class="muted">No warnings yet.</li>`;
  else $$("#w-log li.muted").forEach((n) => n.remove());
  // side panels
  const a = $("#w-actual");
  p.actual != null ? countUp(a, p.actual, 0, 450) : (a.textContent = "–");
  $("#w-vs").textContent = p.actual != null && p.usual ? `${(p.actual / p.usual).toFixed(1)}× the usual ${nf(p.usual)} for this hour (${p.actual_level})` : `Usual for this hour: ${nf(p.usual)}`;
  const nx = s[i + 1];
  const probs = $("#w-probs");
  if (!nx) { probs.innerHTML = `<p class="muted">End of day.</p>`; $("#w-warn").hidden = true; return; }
  probs.innerHTML = LEVELS.map((L, k) => `<div class="prob"><span>${L}</span><span class="bar"><i style="background:${LV_COLOR[k]}" data-w="${(nx.probabilities[L] || 0) * 100}"></i></span><span>${pct(nx.probabilities[L])}</span></div>`).join("");
  requestAnimationFrame(() => $$("#w-probs i").forEach((b) => (b.style.width = b.dataset.w + "%")));
  const wb = $("#w-warn");
  if (nx.warning) { wb.hidden = false; wb.style.animation = "none"; wb.offsetHeight; wb.style.animation = ""; } else wb.hidden = true;
}
function togglePlay() { W.timer ? stopPlay() : startPlay(); }
function startPlay() {
  if (W.i >= 23) setHour(0, true);
  $("#w-play span").textContent = "Pause";
  W.timer = setInterval(() => { if (W.i >= 23) return stopPlay(); setHour(W.i + 1); }, 750);
}
function stopPlay() { clearInterval(W.timer); W.timer = null; const b = $("#w-play span"); if (b) b.textContent = "Play the day"; }

/* ================================================================ CAMERA CHECK */
const C = { sample: null, file: null, bound: false };
async function initCamera() {
  if (C.bound) return; C.bound = true;
  const m = S.meta, samples = $("#samples");
  if (!m.samples.length) samples.innerHTML = `<p class="muted small">No WiderPerson test photos found. Upload your own crowd photo.</p>`;
  m.samples.forEach((sm) => {
    const b = el("button", { type: "button", title: sm.name }, `<img src="${sm.url}" alt="Crowd sample ${sm.id + 1}" loading="lazy">`);
    b.addEventListener("click", () => { $$("#samples button").forEach((x) => x.classList.remove("sel")); b.classList.add("sel"); C.sample = sm.id; C.file = null; showStage(sm.url); });
    samples.append(b);
  });
  const camSel = $("#cam-select");
  const setArea = () => { const c = m.cameras.find((x) => x.camera_id === camSel.value); $("#cam-area").value = c ? c.visible_area_m2 : 200; };
  camSel.addEventListener("change", setArea); setArea();
  $("#cam-file").addEventListener("change", (e) => pickFile(e.target.files[0]));
  const stage = $("#stage");
  stage.addEventListener("dragover", (e) => { e.preventDefault(); stage.classList.add("drag"); });
  stage.addEventListener("dragleave", () => stage.classList.remove("drag"));
  stage.addEventListener("drop", (e) => { e.preventDefault(); stage.classList.remove("drag"); pickFile(e.dataTransfer.files[0]); });
  $("#cam-run").addEventListener("click", () => runCamera().catch((e) => { $("#scan").hidden = true; $("#cam-run").disabled = false; showError($("#page-camera"))(e); }));
}
function pickFile(f) {
  if (!f || !f.type.startsWith("image/")) return;
  C.file = f; C.sample = null; $$("#samples button").forEach((x) => x.classList.remove("sel"));
  showStage(URL.createObjectURL(f));
}
function showStage(url) {
  const img = $("#stage-img");
  img.src = url; $("#frame").hidden = false; $("#drop").hidden = true; $("#stage").classList.add("has-img");
  $("#stage-boxes").innerHTML = ""; $("#cam-result").hidden = true; $("#cam-run").disabled = false;
}
async function runCamera() {
  const fd = new FormData();
  if (C.file) fd.append("image", C.file); else fd.append("sample", C.sample);
  fd.append("camera_id", $("#cam-select").value);
  fd.append("area_m2", $("#cam-area").value);
  fd.append("when", S.when);
  $("#cam-run").disabled = true; $("#scan").hidden = false; $("#stage-boxes").innerHTML = "";
  $("#cam-hint").textContent = "Finding people, then labelling each one…";
  const r = await api("/api/camera", { method: "POST", body: fd });
  $("#scan").hidden = true; $("#cam-run").disabled = false;
  $("#cam-hint").textContent = `${r.classes.people_detected} people found by YOLO, ${r.classes.confirmed_people} confirmed by the image ensemble.`;
  if (C.file) S.lastUpload = r.image_url;
  drawBoxes($("#stage-boxes"), r.detections);
  showCameraResult(r);
}
function drawBoxes(svg, det) {
  svg.innerHTML = "";
  if (!det || !det.width) return;
  svg.setAttribute("viewBox", `0 0 ${det.width} ${det.height}`);
  svg.setAttribute("preserveAspectRatio", "none");
  const boxes = [...det.boxes].sort((a, b) => a.y1 - b.y1);
  const n = boxes.length, total = Math.min(1800, n * 40);
  boxes.forEach((b, i) => {
    const w = b.x2 - b.x1, h = b.y2 - b.y1;
    const r = svgEl("rect", { x: b.x1, y: b.y1, width: w, height: h, rx: 3, class: b.label, pathLength: 100 });
    r.style.setProperty("--len", 100);
    r.style.setProperty("--d", `${(i / Math.max(1, n)) * total}ms`);
    svg.append(r);
  });
}
function showCameraResult(r) {
  const box = $("#cam-result"); box.hidden = false;
  const k = r.risk;
  const rc = $("#cam-risk"); rc.className = `risk-card risk r${k.risk_code}`;
  rc.innerHTML = `<div class="lvl">${RISK_ICON[k.risk_level] || ""} ${esc(k.risk_level)}</div><p>${esc(k.actions[0])}</p>`;
  const c = r.classes.counts;
  const items = [["Walking", c.pedestrian, "var(--c-ped)"], ["Riders", c.rider, "var(--c-rider)"],
                 ["Partly hidden", c.partially_visible, "var(--c-hidden)"], ["Not a person", c.background, "var(--c-bg)"]];
  $("#cam-counts").innerHTML = items.map(([n, v, col]) => `<div class="count"><b data-v="${v}">0</b><span><i style="background:${col}"></i>${n}</span></div>`).join("");
  $$("#cam-counts b").forEach((b) => countUp(b, +b.dataset.v));
  gauge($("#g-density"), k.density_per_m2 ?? 0, 5, (v) => v.toFixed(2), k.density_per_m2 >= 4 ? "var(--act)" : k.density_per_m2 >= 2 ? "var(--watch)" : "var(--ok)");
  gauge($("#g-hidden"), r.classes.partly_hidden_share, 1, (v) => `${Math.round(v * 100)}%`, r.classes.partly_hidden_share >= 0.4 ? "var(--watch)" : "var(--c-ped)");
  $("#cam-reasons").innerHTML = k.reasons.map((x) => `<li>${esc(x)}</li>`).join("");
}
function gauge(svg, v, max, fmt, color) {
  const L = Math.PI * 50;
  svg.innerHTML = `<path class="track" d="M10 62 A50 50 0 0 1 110 62"/><path class="val" d="M10 62 A50 50 0 0 1 110 62" stroke="${color}" stroke-dasharray="${L}" stroke-dashoffset="${L}"/><text x="60" y="58">0</text>`;
  const val = svg.querySelector(".val"), t = svg.querySelector("text");
  requestAnimationFrame(() => requestAnimationFrame(() => val.setAttribute("stroke-dashoffset", L * (1 - Math.min(1, v / max)))));
  if (REDUCE) { t.textContent = fmt(v); return; }
  const t0 = performance.now();
  const tick = (now) => { const k = Math.min(1, (now - t0) / 1000), e = 1 - Math.pow(1 - k, 3); t.textContent = fmt(v * e); if (k < 1) requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
}

/* ================================================================ AGENT */
const TOOL_NAMES = {
  list_locations: "Looked up the cameras and sensors", get_events: "Checked the events calendar",
  get_baseline: "Checked what is normal", forecast_footfall: "Forecast the next hour (XGBoost)",
  detect_people: "Found people in the photo (YOLO)", classify_people: "Labelled each person (image ensemble)",
  assess_risk: "Applied the safety rules",
};
let agentBound = false;
async function initAgent() {
  const m = S.meta;
  const imgSel = $("#a-img");
  imgSel.innerHTML = `<option value="">No photo</option>` + m.samples.map((s) => `<option value="${s.id}">Sample crowd photo ${s.id + 1}</option>`).join("");
  const day = S.when.slice(0, 10), hour = S.when.slice(11, 16);
  const nyeYear = new Date(m.data_end).getFullYear();
  const chips = [
    { q: `Will CAM-1 be dangerously crowded at ${day} ${hour}?` },
    { q: `What events or holidays are on ${day}?` },
    { q: `We are planning New Year's Eve. How risky will CAM-2 be at ${nyeYear}-12-31 22:00?` },
    { q: "Check this camera photo and tell me the risk.", img: m.samples.length ? "0" : "", cam: m.cameras[0]?.camera_id },
    { q: "Which cameras can you monitor?" },
  ];
  $("#chips").innerHTML = "";
  chips.forEach((c) => {
    const b = el("button", { type: "button" }, esc(c.q));
    b.addEventListener("click", () => { $("#q").value = c.q; if (c.img !== undefined) imgSel.value = c.img; if (c.cam) $("#a-cam").value = c.cam; });
    $("#chips").append(b);
  });
  if (agentBound) return; agentBound = true;
  $("#a-run").addEventListener("click", askAgent);
  $("#q").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) askAgent(); });
}
function stepSummary(tool, r) {
  if (!r) return "";
  if (r.error) return `Failed: ${esc(r.error)}`;
  switch (tool) {
    case "list_locations": return `Knows ${r.cameras.length} cameras and ${nf(r.sensor_count)} street sensors.`;
    case "get_events": return `${esc(r.weekday)} ${esc(r.date)}: ${r.events.length ? r.events.map((e) => esc(e.event_name)).join(", ") : "no listed events"}${r.public_holiday ? `, public holiday (${esc(r.public_holiday)})` : ""}.`;
    case "get_baseline": return `Usually ${nf(r.usual_people_per_hour)} people/hour on ${esc(r.weekday)} at ${r.hour}:00. Very High above ${nf(r.level_lines_people_per_hour.Very_High_above)}.`;
    case "forecast_footfall": return `<strong>${esc(r.predicted_level)}</strong> forecast for ${esc(r.forecast_for_hour)} at ${esc(r.sensor)}, chance of Very High ${pct(r.p_very_high)}. Last hour ${nf(r.people_1h_before)} vs usual ${nf(r.usual_people_this_hour)}.${r.early_warning ? " <strong>Early warning raised.</strong>" : ""}${r.mode.startsWith("planning") ? " (Planning mode: future date.)" : ""}`;
    case "detect_people": return `Found ${nf(r.people_detected)} people in a ${esc(r.image_size)} photo.`;
    case "classify_people": return `${nf(r.confirmed_people)} confirmed: ${r.counts.pedestrian} walking, ${r.counts.rider} riders, ${r.counts.partially_visible} partly hidden (${pct(r.partly_hidden_share)}). ${r.rejected_as_background} rejected as not a person.`;
    case "assess_risk": return `${RISK_ICON[r.risk_level] || ""} <strong>${esc(r.risk_level)}</strong>: ${esc(r.reasons[0])}`;
    default: return "";
  }
}
function askAgent() {
  const q = $("#q").value.trim();
  if (!q) { $("#q").focus(); return; }
  const trace = $("#trace"); trace.innerHTML = "";
  trace.append(el("div", { className: "qbubble" }, esc(q)));
  const status = el("div", { className: "status-line" }, `<span class="spin"></span><span>Starting…</span>`);
  const steps = el("div", { className: "steps" });
  trace.append(steps, status);
  const params = new URLSearchParams({ q, when: S.when });
  if ($("#a-cam").value) params.set("camera", $("#a-cam").value);
  if ($("#a-img").value !== "") params.set("sample", $("#a-img").value);
  $("#a-run").disabled = true;
  const es = new EventSource(`/api/agent/stream?${params}`);
  let running = null;
  es.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    if (ev.type === "status") status.querySelector("span:last-child").textContent = ev.text + "…";
    if (ev.type === "tool_start") {
      running = el("div", { className: "step running" }, `<div class="t"><b>${esc(TOOL_NAMES[ev.tool] || ev.tool)}</b><code>${esc(ev.tool)}(${esc(Object.entries(ev.args || {}).filter(([, v]) => v != null).map(([k, v]) => `${k}=${typeof v === "string" ? `"${k === "image_path" ? v.split(/[\\/]/).pop() : v}"` : v}`).join(", "))})</code></div><p class="what">Running…</p>`);
      steps.append(running);
      status.querySelector("span:last-child").textContent = "Running a tool…";
    }
    if (ev.type === "step") {
      const node = running || steps.lastElementChild; running = null;
      node.classList.remove("running");
      node.querySelector(".what").innerHTML = stepSummary(ev.tool, ev.result);
      const det = el("details", {}, `<summary>Raw result</summary><pre>${esc(JSON.stringify(ev.result, null, 2))}</pre>`);
      node.append(det);
    }
    if (ev.type === "done") { es.close(); status.remove(); $("#a-run").disabled = false; trace.append(reportCard(ev)); }
    if (ev.type === "error") { es.close(); status.remove(); $("#a-run").disabled = false; trace.append(el("p", { className: "error" }, esc(ev.text))); }
  };
  es.onerror = () => { es.close(); status.remove(); $("#a-run").disabled = false; trace.append(el("p", { className: "error" }, "Lost connection to the server. Is app.py still running?")); };
}
function reportCard(ev) {
  const r = ev.risk;
  const card = el("article", { className: `report risk r${r ? r.risk_code : 0}` });
  card.innerHTML = `<div class="band"><span class="lvl">${r ? `${RISK_ICON[r.risk_level] || ""} ${esc(r.risk_level)}` : "Answer"}</span><small>${esc(ev.mode)}, ${ev.seconds}s</small></div>`;
  if (ev.image_url && ev.detections) {
    const shot = el("div", { className: "shot" }, `<div class="frame"><img src="${ev.image_url}" alt="Analysed camera photo"></div>`);
    const svg = svgEl("svg", { class: "boxes" }); shot.firstChild.append(svg); card.append(shot);
    setTimeout(() => drawBoxes(svg, ev.detections), 50);
  }
  card.append(el("div", { className: "body" }, esc(ev.answer)));
  return card;
}

/* ================================================================ RESULTS */
async function loadResults() {
  const d = await api("/api/results");
  const k = $("#kpis"); k.innerHTML = "";
  const kpi = (val, label, sub, dec = 3) => k.insertAdjacentHTML("beforeend", `<div class="kpi"><b data-v="${val}" data-d="${dec}">–</b><span>${label}</span><small>${sub}</small></div>`);
  const sm = d.sensor_models || [];
  const ml = sm.filter((r) => !String(r.model).startsWith("Rule"));
  const best = ml.sort((a, b) => b.val_macro_f1 - a.val_macro_f1)[0];
  const rules = sm.filter((r) => String(r.model).startsWith("Rule"));
  if (best) {
    kpi(best.test_macro_f1, `Sensor model macro F1 (${esc(best.model)})`, `Best rule: ${nf(Math.max(...rules.map((r) => r.test_macro_f1)), 3)}`);
    kpi(best.test_very_high_recall * 100, "Very High hours caught (%)", `Best rule: ${nf(Math.max(...rules.map((r) => r.test_very_high_recall)) * 100, 0)}%`, 0);
  }
  const ew = (d.early_warning || []).find((r) => String(r.method).startsWith("AI"));
  if (ew) kpi(ew.warned_at_onset * 100, "Surges warned 1 hour ahead (%)", `Unusual surges: ${pct(ew.unusual_warned_at_onset)}; rules: 0%`, 0);
  const im = (d.image_models || []).find((r) => r.type === "ensemble");
  if (im) { kpi(im.test_macro_f1, "Image ensemble macro F1", `Accuracy ${pct(im.test_accuracy)}`); }
  const ag = d.agent_eval && (d.agent_eval.llm || d.agent_eval.offline);
  if (ag) kpi(parseFloat(ag.tool_selection_correct) || 0, "Agent picked the right tools (%)", `${d.agent_eval.llm ? "Gemini" : "Offline plan"}; numbers grounded ${esc(ag.numbers_grounded)}`, 0);
  $$("#kpis b").forEach((b) => countUp(b, +b.dataset.v, +b.dataset.d, 1100));

  const bars = (rows, key, colorOf, labelOf = (r) => r.model) => `<div class="hbars">${rows.map((r) => `<div class="hbar ${String(r.model).startsWith("Rule") ? "rule" : ""}"><span>${esc(labelOf(r))}</span><span class="b"><i style="background:${colorOf(r)}" data-w="${r[key] * 100}"></i></span><span>${nf(r[key], 3)}</span></div>`).join("")}</div>`;
  $("#r-sensor").innerHTML = sm.length ? bars([...sm].sort((a, b) => b.test_macro_f1 - a.test_macro_f1), "test_macro_f1",
    (r) => String(r.model).startsWith("Rule") ? "#4a5873" : r.model === best?.model ? "var(--watch)" : "#5aa7ff") + `<p class="muted small" style="margin-top:10px">Macro F1. Gray bars are simple rules with no AI.</p>` : `<p class="muted">Run sensor_pipeline/02_train_compare.py first.</p>`;
  const imr = d.image_models || [];
  const tc = { single: "#5aa7ff", hybrid: "#2bc4a0", ensemble: "var(--watch)" };
  $("#r-image").innerHTML = imr.length ? bars(imr, "test_macro_f1", (r) => tc[r.type]) + `<p class="muted small" style="margin-top:10px">Macro F1. Blue: single backbone. Green: hybrid (two backbones fused). Amber: soft-voting ensemble.</p>` : `<p class="muted">Download phase3_outputs.zip from Kaggle first.</p>`;
  requestAnimationFrame(() => requestAnimationFrame(() => $$(".hbar i").forEach((i) => (i.style.width = i.dataset.w + "%"))));

  const e = d.early_warning || [];
  $("#r-early").innerHTML = e.length ? `<table class="t"><thead><tr><th>Method</th><th>Surges warned 1 h ahead</th><th>Unusual surges warned</th><th>Very High hours caught</th><th>False alarms per sensor per week</th></tr></thead><tbody>
    ${e.map((r) => `<tr class="${String(r.method).startsWith("AI") ? "best" : ""}"><td>${esc(r.method)}</td><td>${pct(r.warned_at_onset)}</td><td>${pct(r.unusual_warned_at_onset)}</td><td>${pct(r.surge_hours_caught)}</td><td>${nf(r.false_alarms_per_sensor_week, 2)}</td></tr>`).join("")}</tbody></table>
    ${(d.early_warning_events || []).length ? `<p class="muted small" style="margin-top:10px">Event days never seen in training: ${d.early_warning_events.map((x) => `${esc(x.event)} (${esc(x.date)}): AI caught ${pct(x.caught_by_model)} of Very High hours vs ${pct(x.caught_by_usual_rule)} for the usual-level rule`).join("; ")}.</p>` : ""}`
    : `<p class="muted">Run sensor_pipeline/03_early_warning.py first.</p>`;

  const g = $("#gallery"); g.innerHTML = "";
  d.figures.forEach((src) => {
    const b = el("button", { type: "button" }, `<img src="${src}" alt="${esc(src.split("/").pop().replace(".png", "").replaceAll("_", " "))}" loading="lazy">`);
    b.addEventListener("click", () => { const lb = $("#lightbox"); lb.querySelector("img").src = src; lb.showModal(); });
    g.append(b);
  });
  const lb = $("#lightbox");
  lb.querySelector(".close").onclick = () => lb.close();
  lb.onclick = (ev) => { if (ev.target === lb) lb.close(); };
}

boot().catch((e) => { document.body.insertAdjacentHTML("afterbegin", `<p class="error" style="margin:20px">Could not start: ${esc(e.message)}</p>`); });
