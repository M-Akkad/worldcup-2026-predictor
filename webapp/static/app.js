/* WK·26 ODDS ENGINE, frontend logic (vanilla JS, no build step).
   User-facing text goes through t() / data-i18n; see i18n.js. */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => (x == null ? "–" : (100 * x).toFixed(d) + "%");
const num = (n) => n.toLocaleString(LANG === "en" ? "en" : "nl");

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" }, ...opts });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || res.statusText);
  return body;
}

let toastTimer = null;
function toast(msg, err = false) {
  let el = $(".toast");
  if (!el) { el = document.createElement("div"); document.body.appendChild(el); }
  el.className = "toast" + (err ? " err" : "");
  el.textContent = msg;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.remove(), 4200);
}

/* tabs */
const TABS = ["dashboard", "groups", "results", "predict", "data"];
function showTab(name) {
  if (!TABS.includes(name)) name = "dashboard";
  $$("main > section").forEach(s => s.hidden = s.dataset.tab !== name);
  $$("nav button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  localStorage.setItem("tab", name);
  if (location.hash.slice(1) !== name) history.replaceState(null, "", "#" + name);
  if (name === "results" && !resultsLoaded) {
    resultsLoaded = true; loadFixtures(); loadScorecard();
  }
}
$$("nav button").forEach(b => b.addEventListener("click", () => showTab(b.dataset.tab)));
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

/* status chips */
async function loadStatus() {
  try {
    const s = await api("/api/status");
    $("#chip-matches").innerHTML = t("chip_matches", { n: num(s.matches) });
    const cond = s.played_group_matches + s.decided_ko_matches;
    const live = $("#chip-cond");
    live.innerHTML = cond > 0 ? t("chip_cond", { n: cond }) : t("chip_waiting");
    live.classList.toggle("live", cond > 0);
    $("#chip-run").innerHTML = s.latest_run
      ? t("chip_run", { model: esc(s.latest_run.model), n: num(s.latest_run.n_sims) })
      : t("chip_norun");
  } catch (e) { /* leave chips empty */ }
}

/* dashboard */
let tableData = [];
let sortKey = "win_title", sortDir = -1;

async function loadDashboard() {
  let run;
  try { run = await api("/api/runs/latest"); }
  catch {
    $("#hero-body").innerHTML = `<p class="muted">${t("hero_norun")}</p>`;
    return;
  }
  const tbl = run.table;
  const fav = tbl[0];
  const ko = run.conditioned_ko ? t("cond_ko", { n: run.conditioned_ko }) : "";
  $("#hero-body").innerHTML = `
    <div class="eyebrow">${t("hero_fav_eyebrow", { model: esc(run.model), n: num(run.n_sims) })}</div>
    <h1><span class="fav-name">${esc(fav.team)}</span></h1>
    <div class="fav-odds">${pct(fav.win_title)} ${t("fav_worldtitle")} ·
      ${pct(fav.reach_final)} ${t("fav_final")} · Elo ${Math.round(fav.elo)}</div>
    <p class="small muted">${t("cond_main", { g: run.conditioned_group })}${ko}.</p>`;

  const bars = $("#title-bars");
  bars.innerHTML = tbl.slice(0, 10).map((r, i) => `
    <div class="barrow">
      <span class="team">${i + 1}. ${esc(r.team)}</span>
      <div class="track"><div class="fill ${i === 0 ? "gold" : ""}"
           data-w="${(100 * r.win_title / tbl[0].win_title).toFixed(1)}"></div></div>
      <span class="val">${pct(r.win_title)}</span>
    </div>`).join("");
  requestAnimationFrame(() =>
    $$(".fill", bars).forEach(f => f.style.width = f.dataset.w + "%"));

  tableData = tbl;
  renderTable();
  loadHistory(tbl.slice(0, 5).map(r => r.team));
}

const COL_TIPS = {
  team: "tip_team", group: "tip_group", elo: "tip_elo", win_group: "tip_groupwin",
  reach_ko: "tip_ko", reach_qf: "tip_qf", reach_sf: "tip_sf",
  reach_final: "tip_final", win_title: "tip_title",
};

function renderTable() {
  const cols = [["team", "col_team"], ["group", "col_group"], ["elo", "col_elo"],
    ["win_group", "col_groupwin"], ["reach_ko", "col_ko"],
    ["reach_qf", "col_qf"], ["reach_sf", "col_sf"],
    ["reach_final", "col_final"], ["win_title", "col_title"]];
  const q = ($("#table-search").value || "").toLowerCase();
  const rows = tableData
    .filter(r => r.team.toLowerCase().includes(q))
    .sort((a, b) => {
      const va = a[sortKey === "group" ? "grp" : sortKey];
      const vb = b[sortKey === "group" ? "grp" : sortKey];
      return (va > vb ? 1 : va < vb ? -1 : 0) * sortDir;
    });
  $("#prob-table").innerHTML = `
    <thead><tr>${cols.map(([k, lk]) =>
      `<th data-k="${k}" class="${k === sortKey ? "sorted" : ""}"
           title="${esc(t(COL_TIPS[k]))}">${esc(t(lk))}${
        k === sortKey ? (sortDir < 0 ? " ↓" : " ↑") : ""}</th>`).join("")}
    </tr></thead>
    <tbody>${rows.map(r => `
      <tr><td>${esc(r.team)}</td><td class="num">${esc(r.grp)}</td>
      <td class="num">${Math.round(r.elo)}</td>
      <td class="num">${pct(r.win_group)}</td><td class="num">${pct(r.reach_ko)}</td>
      <td class="num">${pct(r.reach_qf)}</td><td class="num">${pct(r.reach_sf)}</td>
      <td class="num">${pct(r.reach_final)}</td><td class="num">${pct(r.win_title)}</td>
      </tr>`).join("")}</tbody>`;
  $$("#prob-table th").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.k;
    if (sortKey === k) sortDir *= -1; else { sortKey = k; sortDir = -1; }
    renderTable();
  }));
}
$("#table-search").addEventListener("input", renderTable);

const PALETTE = ["#e9ba4d", "#ff5c39", "#7fb0d6", "#9fd68a", "#d68ac9"];

async function loadHistory(topTeams) {
  const { runs } = await api("/api/runs/history");
  const box = $("#history-box");
  if (runs.length < 2) {
    box.innerHTML = `<p class="small muted">${t("p_history_grow")}</p>`;
    return;
  }
  const W = 900, H = 210, P = 34;
  const maxY = Math.max(.05, ...runs.flatMap(r => topTeams.map(t => r.win_title[t] || 0)));
  const x = i => P + (W - 2 * P) * (runs.length === 1 ? 0 : i / (runs.length - 1));
  const y = v => H - P + (P * 2 - H) * (v / maxY);
  let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img"
    aria-label="${esc(t("chart_aria"))}">`;
  for (const f of [0, .5, 1]) {
    svg += `<line x1="${P}" x2="${W - P}" y1="${y(maxY * f)}" y2="${y(maxY * f)}"
      stroke="#2e5c3c" stroke-dasharray="3 5"/>
      <text x="4" y="${y(maxY * f) + 3}">${(100 * maxY * f).toFixed(0)}%</text>`;
  }
  topTeams.forEach((team, ti) => {
    const pts = runs.map((r, i) => `${x(i)},${y(r.win_title[team] || 0)}`).join(" ");
    svg += `<polyline points="${pts}" fill="none" stroke="${PALETTE[ti]}"
      stroke-width="2.4"/>`;
  });
  svg += `</svg>`;
  box.innerHTML = svg + `<div class="legend">` + topTeams.map((t, i) =>
    `<span><i style="background:${PALETTE[i]}"></i>${esc(t)}</span>`).join("") + `</div>`;
}

/* groups */
async function loadGroups() {
  const { groups, has_run } = await api("/api/groups");
  $("#groupgrid").innerHTML = groups.map(g => `
    <div class="groupcard">
      <h3>${t("group_label", { g: esc(g.group) })}</h3>
      ${g.teams.map(tm => `
        <div class="trow">
          <span>${esc(tm.team)}${tm.host ? `<span class="hostbadge">${t("badge_host")}</span>` : ""}</span>
          <span class="num small muted">${tm.elo ? Math.round(tm.elo) : "–"}</span>
          <span class="num small">${pct(tm.win_group, 0)}</span>
          <span class="mini"><i style="width:${100 * (tm.win_group || 0)}%"></i></span>
        </div>`).join("")}
      ${g.played.length ? `<div style="margin-top:8px">${g.played.map(m =>
        `<span class="playedchip">${esc(m.home)} <b>${m.home_score}–${m.away_score}</b> ${esc(m.away)}</span>`
      ).join("")}</div>` : ""}
    </div>`).join("");
  $("#groups-note").hidden = has_run;
}

/* results */
let resultsLoaded = false;

async function loadFixtures() {
  const tbl = $("#fixtures-table"), empty = $("#fixtures-empty");
  tbl.innerHTML = `<tbody><tr><td class="small muted">${t("fx_calc")}</td></tr></tbody>`;
  let data;
  try { data = await api("/api/fixtures?model=ensemble"); }
  catch (e) { tbl.innerHTML = ""; toast(e.message, true); return; }
  const fx = data.fixtures;
  if (!fx.length) { tbl.innerHTML = ""; empty.hidden = false; return; }
  empty.hidden = true;
  const bar = (p, cls) => `<span class="seg ${cls}" style="width:${(100 * p).toFixed(1)}%"></span>`;
  tbl.innerHTML = `
    <thead><tr><th>${t("th_date")}</th><th>${t("th_match")}</th>
      <th>${t("th_winprob")} <span class="small muted">${t("sub_hda")}</span></th>
      <th><abbr title="${esc(t("tip_expgoals"))}">${t("th_expgoals")}</abbr></th>
      <th>${t("th_expresult")}</th><th>${t("th_favourite")}</th></tr></thead>
    <tbody>${fx.map(m => {
      const p = m.pred, s = m.pred_score;
      return `<tr>
        <td class="num small">${m.date.slice(5)}</td>
        <td>${esc(m.home)} <span class="muted">v</span> ${esc(m.away)}
            ${m.neutral ? "" : `<span class="hostbadge">${t("badge_home")}</span>`}</td>
        <td><div class="wdl" title="${pct(p.home,0)} / ${pct(p.draw,0)} / ${pct(p.away,0)}">
            ${bar(p.home,"h")}${bar(p.draw,"d")}${bar(p.away,"a")}</div>
            <span class="small muted">${pct(p.home,0)} / ${pct(p.draw,0)} / ${pct(p.away,0)}</span></td>
        <td class="num small">${m.pred_xg.home.toFixed(1)}–${m.pred_xg.away.toFixed(1)}</td>
        <td class="num"><b>${s.home}–${s.away}</b> <span class="small muted">${pct(s.prob,0)}</span></td>
        <td class="small">${esc(m.favourite)} <b>${pct(m.favourite_prob,0)}</b></td>
      </tr>`;
    }).join("")}</tbody>`;
}

async function loadScorecard() {
  const tbl = $("#scorecard-table"), sum = $("#scorecard-summary"),
        empty = $("#scorecard-empty");
  sum.innerHTML = `<span class="small muted">${t("sc_calc")}</span>`;
  tbl.innerHTML = "";
  let sc;
  try { sc = await api("/api/scorecard?model=ensemble"); }
  catch (e) { sum.innerHTML = ""; toast(e.message, true); resultsLoaded = false; return; }
  const s = sc.summary;
  if (!s.n) { sum.innerHTML = ""; empty.hidden = false; return; }
  empty.hidden = true;
  sum.innerHTML = `
    <div><span class="gv">${s.n}</span><span class="gl">${t("sc_n")}</span></div>
    <div><span class="gv">${pct(s.favourite_hit_rate, 0)}</span><span class="gl">${t("sc_fav")}</span></div>
    <div><span class="gv">${pct(s.mean_p_actual, 0)}</span><span class="gl">${t("sc_pact")}</span></div>
    <div><span class="gv">${s.mean_rps.toFixed(3)}</span><span class="gl"><abbr title="${esc(t("tip_rps_quality"))}">${t("sc_quality")}</abbr> ${t("sc_lower_better")}</span></div>`;
  tbl.innerHTML = `
    <thead><tr><th>${t("th_date")}</th><th>${t("th_match")}</th>
      <th>${t("th_predicted")} <span class="small muted">${t("sub_hda")}</span></th>
      <th><abbr title="${esc(t("tip_expgoals"))}">${t("th_expgoals")}</abbr></th>
      <th>${t("th_result")}</th>
      <th><abbr title="${esc(t("tip_realchance"))}">${t("th_realchance")}</abbr></th></tr></thead>
    <tbody>${sc.matches.map(m => {
      const p = m.pred, ac = m.actual;
      const res = `${ac.home}–${ac.away}`;
      const surprise = m.p_actual < 0.25;
      return `<tr>
        <td class="num small">${m.date.slice(5)}</td>
        <td>${esc(m.home)} <span class="muted">v</span> ${esc(m.away)}
            ${m.neutral ? "" : `<span class="hostbadge">${t("badge_home")}</span>`}</td>
        <td class="num small">${pct(p.home,0)} / ${pct(p.draw,0)} / ${pct(p.away,0)}</td>
        <td class="num small">${m.pred_xg.home.toFixed(1)}–${m.pred_xg.away.toFixed(1)}</td>
        <td class="num"><b>${res}</b></td>
        <td class="num ${surprise ? "miss" : (m.favourite_correct ? "win" : "")}">${pct(m.p_actual,0)}</td>
      </tr>`;
    }).join("")}</tbody>`;
}

/* predictor */
async function fillTeamSelects() {
  const { teams } = await api("/api/teams");
  const opts = teams.map(t => `<option>${esc(t)}</option>`).join("");
  $("#sel-home").innerHTML = opts;
  $("#sel-away").innerHTML = opts;
  $("#sel-home").value = "Netherlands";
  $("#sel-away").value = "Brazil";
}

$("#predict-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const btn = $("#predict-btn"); btn.disabled = true;
  try {
    const params = new URLSearchParams({
      home: $("#sel-home").value, away: $("#sel-away").value,
      model: $("#sel-pmodel").value,
      neutral: $("#chk-neutral").checked,
    });
    const odds = $("#inp-odds").value.trim();
    if (odds) {
      params.set("odds", odds);
      params.set("market_weight", $("#inp-mw").value || "0.5");
    }
    const p = await api("/api/predict?" + params);
    renderPrediction(p);
  } catch (e) { toast(e.message, true); }
  btn.disabled = false;
});

let lastPrediction = null;
function renderPrediction(p) {
  lastPrediction = p;
  $("#predict-out").hidden = false;
  $("#pred-title").innerHTML =
    `${esc(p.home)} <span class="muted small">(Elo ${Math.round(p.elo.home)})</span>
     – ${esc(p.away)} <span class="muted small">(Elo ${Math.round(p.elo.away)})</span>
     <span class="muted small">· ${p.neutral ? t("venue_neutral") : t("venue_home")}</span>`;
  $("#pred-outcome").innerHTML = `
    <div class="cell home"><div class="pct">${pct(p.probs.home)}</div>${t("pred_win", { team: esc(p.home) })}</div>
    <div class="cell draw"><div class="pct">${pct(p.probs.draw)}</div>${t("pred_draw")}</div>
    <div class="cell away"><div class="pct">${pct(p.probs.away)}</div>${t("pred_win", { team: esc(p.away) })}</div>`;
  $("#pred-market").textContent = p.market
    ? t("pred_market", { probs: p.market.map(x => pct(x)).join(" / ") })
    : "";

  const gl = p.goals;
  const goalHint = p.model === "elo" ? "" :
    `<p class="small muted" style="margin:6px 0 0">${t("goal_hint", { model: esc(p.model) })}</p>`;
  $("#pred-goals").innerHTML = `
    <div class="goalgrid">
      <div><span class="gv">${gl.exp_total.toFixed(2)}</span><span class="gl">${t("goal_total")}</span>
        <span class="small muted">${esc(p.home)} ${gl.exp_home.toFixed(2)} – ${gl.exp_away.toFixed(2)} ${esc(p.away)}</span></div>
      <div><span class="gv">${pct(gl.over25)}</span><span class="gl">${t("goal_over25")}</span>
        <span class="small muted">${t("goal_under25", { x: pct(1 - gl.over25) })}</span></div>
      <div><span class="gv">${pct(gl.btts)}</span><span class="gl">${t("goal_btts")}</span>
        <span class="small muted">${t("goal_over1535", { a: pct(gl.over15), b: pct(gl.over35) })}</span></div>
    </div>${goalHint}`;
  $("#pred-scores").innerHTML = p.top_scorelines.map(s =>
    `<span class="playedchip"><b>${s.home_goals}–${s.away_goals}</b> ${pct(s.prob)}</span>`).join("");

  const m = p.matrix, mx = Math.max(...m.flat());
  let heat = `<div class="hd"></div>` +
    [...Array(7).keys()].map(j => `<div class="hd">${j}</div>`).join("");
  for (let i = 0; i < 7; i++) {
    heat += `<div class="hd">${i}</div>` + m[i].map(v => {
      const a = Math.pow(v / mx, .65);
      return `<div class="cell" style="background:rgba(255,92,57,${(0.06 + 0.9 * a).toFixed(2)})"
        title="${pct(v, 2)}">${v > mx * .12 ? pct(v, 0) : ""}</div>`;
    }).join("");
  }
  $("#pred-heat").innerHTML = heat;
}

/* data & model */
async function loadAdjustments() {
  const { adjustments } = await api("/api/adjustments");
  $("#adj-list").innerHTML = adjustments.length ? adjustments.map(a => `
    <div class="adjrow">
      <span class="d ${a.delta >= 0 ? "up" : "down"}">${a.delta > 0 ? "+" : ""}${a.delta}</span>
      <span><b>${esc(a.team)}</b> <span class="muted small">${esc(a.reason || "")}</span></span>
      <button class="x" title="${esc(t("adj_delete"))}" data-team="${esc(a.team)}">✕</button>
    </div>`).join("")
    : `<p class="small muted">${t("adj_empty")}</p>`;
  $$("#adj-list .x").forEach(b => b.addEventListener("click", async () => {
    await api("/api/adjustments/" + encodeURIComponent(b.dataset.team),
              { method: "DELETE" });
    loadAdjustments(); loadStatus();
  }));
}

$("#adj-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  try {
    await api("/api/adjustments", { method: "POST", body: JSON.stringify({
      team: $("#adj-team").value, delta: parseFloat($("#adj-delta").value),
      reason: $("#adj-reason").value }) });
    $("#adj-form").reset();
    toast(t("toast_adj_saved"));
    loadAdjustments(); loadStatus();
  } catch (e) { toast(e.message, true); }
});

$("#btn-fetch").addEventListener("click", async () => {
  const b = $("#btn-fetch"); b.disabled = true;
  try {
    const r = await api("/api/update/fetch", { method: "POST" });
    const n = (r.added || 0) + (r.updated || 0);
    if (n > 0) {
      toast(t("toast_fetch", { n }));
      resultsLoaded = false;
      loadStatus(); loadGroups();
    } else {
      toast(r.note || t("toast_fetch_none"), false);
    }
  } catch (e) { toast(e.message, true); }
  b.disabled = false;
});

$("#result-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  try {
    const r = await api("/api/update/result", { method: "POST", body: JSON.stringify({
      home: $("#res-home").value, away: $("#res-away").value,
      home_score: parseInt($("#res-hs").value), away_score: parseInt($("#res-as").value),
      date: $("#res-date").value || null, neutral: !$("#res-homegame").checked }) });
    toast(r.added ? t("toast_result_added")
      : (r.updated ? t("toast_result_updated") : t("toast_result_dupe")));
    $("#res-hs").value = ""; $("#res-as").value = "";
    resultsLoaded = false;
    loadStatus(); loadGroups();
  } catch (e) { toast(e.message, true); }
});

$("#sim-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const btn = $("#sim-btn"); btn.disabled = true;
  $("#sim-log").textContent = t("sim_started");
  try {
    const { job_id } = await api("/api/simulate", { method: "POST", body: JSON.stringify({
      model: $("#sim-model").value, n_sims: parseInt($("#sim-n").value),
      use_adjustments: $("#sim-adj").checked }) });
    const t0 = Date.now();
    const poll = async () => {
      const j = await api("/api/jobs/" + job_id);
      if (j.status === "running") {
        $("#sim-log").textContent = t("sim_busy", { s: ((Date.now() - t0) / 1000).toFixed(0) });
        setTimeout(poll, 800);
      } else if (j.status === "done") {
        $("#sim-log").textContent = t("sim_done", { s: ((Date.now() - t0) / 1000).toFixed(1), id: j.run_id });
        toast(t("toast_sim_done"));
        btn.disabled = false;
        loadStatus(); loadDashboard(); loadGroups();
        showTab("dashboard");
      } else {
        $("#sim-log").textContent = t("job_failed", { e: j.error });
        toast(j.error, true);
        btn.disabled = false;
      }
    };
    poll();
  } catch (e) { toast(e.message, true); btn.disabled = false; }
});

$("#bt-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const btn = $("#bt-btn"); btn.disabled = true;
  $("#bt-out").innerHTML = `<p class="small muted">${t("bt_running")}</p>`;
  try {
    const r = await api("/api/backtest?" + new URLSearchParams({
      model: $("#bt-model").value, start: $("#bt-start").value }));
    const row = (k, a, b) => `<tr><td>${metricName(k)}</td>
      <td class="num">${a.toFixed(4)}</td><td class="num">${b.toFixed(4)}</td></tr>`;
    $("#bt-out").innerHTML = `
      <table><thead><tr><th>${t("th_metric")}</th><th>${t("th_model")}</th>
        <th><abbr title="${esc(t("tip_baserate"))}">${t("th_baserate")}</abbr></th></tr></thead>
      <tbody>
        ${row("rps", r.model.rps, r.baseline.rps)}
        ${row("log_loss", r.model.log_loss, r.baseline.log_loss)}
        ${row("brier", r.model.brier, r.baseline.brier)}
        ${row("accuracy", r.model.accuracy, r.baseline.accuracy)}
      </tbody></table>
      <p class="small muted"><abbr title="${esc(t("tip_m_ece"))}">ECE</abbr>
      ${t("bt_ece_line", { x: r.ece.toFixed(4), n: r.model.n })}</p>`;
  } catch (e) { $("#bt-out").innerHTML = ""; toast(e.message, true); }
  btn.disabled = false;
});

/* parameter tuning */
const METRIC_LABEL = { rps: "RPS", log_loss: "log-loss", brier: "Brier" };
const METRIC_TIP = { rps: "tip_m_rps", log_loss: "tip_m_ll", brier: "tip_m_brier",
  accuracy: "tip_m_acc", ece: "tip_m_ece" };
const metricName = k => {
  const label = k === "log_loss" ? "log-loss" : k.toUpperCase();
  return METRIC_TIP[k] ? `<abbr title="${esc(t(METRIC_TIP[k]))}">${label}</abbr>` : label;
};

function renderTuneResult(j) {
  const ml = METRIC_LABEL[j.metric] || j.metric;
  const verdict = j.improved
    ? `<span class="tag good">${t("verdict_better")}</span>`
    : `<span class="tag bad">${t("verdict_worse")}</span>`;
  const rows = ["rps", "log_loss", "brier", "accuracy", "ece"]
    .filter(k => k in j.best_metrics)
    .map(k => {
      const better = k === "accuracy"
        ? j.best_metrics[k] > j.baseline_metrics[k]
        : j.best_metrics[k] < j.baseline_metrics[k];
      return `<tr><td>${metricName(k)}</td>
        <td class="num ${better ? "win" : ""}">${j.best_metrics[k].toFixed(4)}</td>
        <td class="num">${j.baseline_metrics[k].toFixed(4)}</td></tr>`;
    }).join("");
  const cfg = Object.entries(j.best_config)
    .map(([k, v]) => `<div><code>${esc(k)}</code><b>${(+v).toFixed(3)}</b></div>`)
    .join("");
  $("#tune-out").innerHTML = `
    <p class="small">${verdict} &nbsp; ${ml}: <b>${j.best_value.toFixed(4)}</b>
      ${t("tune_vs", { v: j.baseline_value.toFixed(4) })}
      <span class="muted">${t("tune_meta", { n: j.n_trials, m: esc(j.model) })}</span></p>
    <table class="mini-metrics"><thead><tr><th>${t("th_metric")}</th><th>${t("th_found")}</th>
      <th>${t("th_default")}</th></tr></thead><tbody>${rows}</tbody></table>
    <details class="explain" style="margin-top:10px">
      <summary>${t("tune_bestparams")}${j.improved ? t("tune_adopt") : ""}</summary>
      <div class="cfggrid">${cfg}</div>
      <p class="small muted" style="margin-bottom:0">${j.improved ? t("tune_note_better") : t("tune_note_worse")}</p>
    </details>`;
}

$("#tune-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const btn = $("#tune-btn"); btn.disabled = true;
  const bar = $("#tune-bar"), fill = $("#tune-bar i");
  bar.hidden = false; fill.style.width = "0%";
  $("#tune-out").innerHTML = "";
  $("#tune-log").textContent = t("tune_started");
  try {
    const { job_id } = await api("/api/tune", { method: "POST", body: JSON.stringify({
      model: $("#tune-model").value, metric: $("#tune-metric").value,
      n_trials: parseInt($("#tune-trials").value) }) });
    const t0 = Date.now();
    const poll = async () => {
      const j = await api("/api/jobs/" + job_id);
      if (j.status === "running") {
        const frac = j.total ? (j.done / j.total) : 0;
        fill.style.width = (100 * frac).toFixed(1) + "%";
        const ml = METRIC_LABEL[j.metric] || j.metric;
        $("#tune-log").textContent =
          t("tune_trial", { done: j.done || 0, total: j.total }) +
          (j.best != null ? t("tune_best", { ml, v: j.best.toFixed(4) }) : "") +
          ` · ${((Date.now() - t0) / 1000).toFixed(0)}s`;
        setTimeout(poll, 700);
      } else if (j.status === "done") {
        fill.style.width = "100%";
        $("#tune-log").textContent = t("tune_done", { s: ((Date.now() - t0) / 1000).toFixed(0) });
        toast(j.improved ? t("toast_tune_better") : t("toast_tune_same"));
        renderTuneResult(j);
        btn.disabled = false;
      } else {
        bar.hidden = true;
        $("#tune-log").textContent = t("job_failed", { e: j.error });
        toast(j.error, true);
        btn.disabled = false;
      }
    };
    poll();
  } catch (e) { bar.hidden = true; $("#tune-log").textContent = "";
    toast(e.message, true); btn.disabled = false; }
});

/* language switch: re-render dynamic content when the language changes */
$("#lang-btn").addEventListener("click", () => setLang(LANG === "nl" ? "en" : "nl"));
window.addEventListener("langchange", () => {
  loadStatus();
  loadDashboard();
  loadGroups();
  loadAdjustments();
  resultsLoaded = false;
  const cur = document.querySelector("main > section:not([hidden])");
  if (cur && cur.dataset.tab === "results") { resultsLoaded = true; loadFixtures(); loadScorecard(); }
  if (lastPrediction && !$("#predict-out").hidden) renderPrediction(lastPrediction);
});

/* init */
(async function init() {
  applyI18n();
  showTab(location.hash.slice(1) || localStorage.getItem("tab") || "dashboard");
  loadStatus();
  fillTeamSelects().catch(() => {});
  loadDashboard();
  loadGroups();
  loadAdjustments();
})();
