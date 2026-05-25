/* ===========================================================================
 * live.js — live substrate game frontend
 *
 * Reuses the rendering patterns from view.js but sourced from /api/live/*
 * and driven by a polling loop while phases are computed.
 *
 * State shape mirrors view.js, plus a few live-only fields:
 *   live.state    : public state from /api/live/state (started, ended, etc.)
 *   live.bundle   : the full payload {summary,snapshots,messages,board,log}
 *   live.polling  : true while the page is polling for updates
 * =========================================================================== */

const POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "RUSSIA", "TURKEY"];

const POWER_COLORS = {
  AUSTRIA: "#b03a48", ENGLAND: "#2a4365", FRANCE: "#4a90b8",
  GERMANY: "#4a4538", RUSSIA: "#b8855e", TURKEY: "#c8a04a",
};

const POWER_BRIEFS = {
  MARSHAL_VEIL: "I plan in arcs. I keep my word when watched and remember when others do not.",
  CARDINAL_FOX: "I trade in stories and what they imply. I prefer a beautiful turn to a safe one.",
  PARSON_HAWTHORNE: "My word is given carefully and kept absolutely.",
  BARON_KORVIN: "I trust no one before they have earned it twice.",
  ARCHITECT_LIRA: "I look at the whole table and design the equilibrium I prefer.",
  PLAYER_DEFAULT: "An LLM-driven Diplomacy player.",
};

const STATUS_COLOR = {
  active: "#2f7a4a", proto: "#b8893a", retired: "#8a8275",
  kept: "#2f7a4a", broken: "#b03a48", pending: "#8c7d5b",
  confirmed: "#2f7a4a", refuted: "#b03a48", open: "#4a78a8", partial: "#7a6a8a",
};

const state = {
  // bundle data (from /api/live/bundle)
  bundle: null,            // {summary, snapshots, messages, board, log, state}
  phases: [],
  phase: null,             // currently-displayed phase (last completed by default)

  // selected for the right pane
  power: null,
  foveated: false,
  selectedNodeId: null,

  // map calibration
  calibration: null,

  // live-mode flags
  live: {
    public: null,          // public state object from /api/live/state
    polling: false,
    pollHandle: null,
    followLatest: true,    // if true, scrubber jumps to newest phase as it arrives
  },
};


/* ---------------------------------------------------------------------------
 * Bootstrap & start overlay
 * ------------------------------------------------------------------------- */

async function init() {
  // Load calibration up-front (static)
  try {
    const r = await fetch("/static/map_calibration.json");
    state.calibration = r.ok ? await r.json() : null;
  } catch (e) { state.calibration = null; }

  // Check whether a session is already running. If yes, skip start overlay.
  const sresp = await fetch("/api/live/state");
  const pub = await sresp.json();
  if (pub.started) {
    hideStartOverlay();
    await refreshBundle();
    startPolling();
  } else {
    setupStartOverlay();
  }
}

function setupStartOverlay() {
  document.getElementById("start-btn").addEventListener("click", onStartClick);
}

async function onStartClick() {
  const btn = document.getElementById("start-btn");
  btn.disabled = true;
  document.getElementById("start-error").textContent = "";

  const radio = document.querySelector('input[name="llm-kind"]:checked');
  const llm_kind = radio.value;
  const model = radio.dataset.model;
  const max_phases = parseInt(document.getElementById("start-max-phases").value, 10) || 12;

  const llm_options = {};
  if (model) llm_options.model = model;

  try {
    const r = await fetch("/api/live/start", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({llm_kind, llm_options, max_phases}),
    });
    const data = await r.json();
    if (!r.ok || data.error) {
      document.getElementById("start-error").textContent = data.error || "Start failed";
      btn.disabled = false;
      return;
    }
    hideStartOverlay();
    await refreshBundle();
    startPolling();
  } catch (e) {
    document.getElementById("start-error").textContent = "Network error: " + e.message;
    btn.disabled = false;
  }
}

function hideStartOverlay() {
  document.getElementById("start-overlay").classList.add("hidden");
  document.getElementById("app").classList.remove("hidden");
  setupAppHandlers();
  initMapSVG();
}


/* ---------------------------------------------------------------------------
 * App-level handlers (after start)
 * ------------------------------------------------------------------------- */

function setupAppHandlers() {
  document.getElementById("phase-scrubber").addEventListener("input", e => {
    const i = parseInt(e.target.value, 10);
    if (i >= 0 && i < state.phases.length) {
      state.phase = state.phases[i];
      state.live.followLatest = (i === state.phases.length - 1);
      state.selectedNodeId = null;
      renderAll();
    }
  });
  document.getElementById("phase-prev").addEventListener("click", () => {
    const i = state.phases.indexOf(state.phase);
    if (i > 0) {
      state.phase = state.phases[i - 1];
      state.live.followLatest = false;
      state.selectedNodeId = null;
      buildPhaseScrubber();
      renderAll();
    }
  });
  document.getElementById("phase-next").addEventListener("click", () => {
    const i = state.phases.indexOf(state.phase);
    if (i < state.phases.length - 1) {
      state.phase = state.phases[i + 1];
      state.live.followLatest = (i + 1 === state.phases.length - 1);
      state.selectedNodeId = null;
      buildPhaseScrubber();
      renderAll();
    }
  });
  document.getElementById("fovea-toggle").addEventListener("change", e => {
    state.foveated = e.target.checked;
    renderGraph();
  });
  document.getElementById("run-next-btn").addEventListener("click", onRunNext);
  document.getElementById("auto-play-btn").addEventListener("click", onAutoPlayToggle);
  ["msg-filter-self", "msg-filter-to-self", "msg-filter-other"].forEach(id => {
    document.getElementById(id).addEventListener("change", renderMessages);
  });
}

async function onRunNext() {
  document.getElementById("run-next-btn").disabled = true;
  try {
    await fetch("/api/live/run_next_phase", {method: "POST"});
  } catch (e) {
    console.error("run_next failed:", e);
  }
  // The phase will run on the server; polling will pick up state changes.
}

async function onAutoPlayToggle() {
  const btn = document.getElementById("auto-play-btn");
  const isOn = btn.classList.contains("active");
  if (isOn) {
    await fetch("/api/live/auto_pause", {method: "POST"});
  } else {
    await fetch("/api/live/auto_play", {method: "POST"});
  }
}


/* ---------------------------------------------------------------------------
 * Polling loop
 *
 * Behavior:
 *   - poll /api/live/state every ~1.5s
 *   - if completed_phases length grew, fetch the bundle and re-render
 *   - if computing_phase changed, update the "thinking" banner
 *   - if ended, stop polling
 * ------------------------------------------------------------------------- */

function startPolling() {
  if (state.live.polling) return;
  state.live.polling = true;
  pollTick();
}

function scheduleNextPoll(delayMs) {
  if (state.live.pollHandle) clearTimeout(state.live.pollHandle);
  state.live.pollHandle = setTimeout(() => {
    if (state.live.polling) pollTick();
  }, delayMs);
}

async function pollTick() {
  try {
    const r = await fetch("/api/live/state");
    const pub = await r.json();
    const prev = state.live.public;
    state.live.public = pub;

    updateLiveControls(pub);

    const prevCompleted = prev?.completed_phases?.length || 0;
    const nowCompleted = pub.completed_phases.length;
    if (nowCompleted > prevCompleted || !state.bundle) {
      await refreshBundle();
    }
    updateThinkingBanner(pub);

    if (pub.ended || pub.last_error) {
      scheduleNextPoll(3000);
    } else {
      scheduleNextPoll(pub.is_running_phase ? 1200 : 1800);
    }
  } catch (e) {
    console.warn("poll error:", e);
    scheduleNextPoll(3000);
  }
}

function updateLiveControls(pub) {
  const runBtn = document.getElementById("run-next-btn");
  const autoBtn = document.getElementById("auto-play-btn");
  if (pub.ended) {
    runBtn.disabled = true;
    autoBtn.disabled = true;
    autoBtn.textContent = "Game ended";
    autoBtn.classList.remove("active");
  } else {
    runBtn.disabled = pub.is_running_phase || pub.auto_play;
    autoBtn.disabled = false;
    autoBtn.textContent = pub.auto_play ? "Pause auto-play" : "Auto-play";
    autoBtn.classList.toggle("active", pub.auto_play);
  }
  // Run meta string in topbar
  const meta = document.getElementById("run-meta");
  if (meta) {
    const bits = [];
    if (pub.llm_kind) bits.push(pub.llm_kind);
    if (pub.llm_model) bits.push(pub.llm_model);
    if (pub.completed_phases) bits.push(`${pub.completed_phases.length}/${pub.max_phases||"?"} phases`);
    if (pub.ended) bits.push("ENDED: " + (pub.end_reason||""));
    meta.textContent = bits.join(" · ");
  }
}

function updateThinkingBanner(pub) {
  const banner = document.getElementById("thinking-banner");
  if (!banner) return;
  if (pub.is_running_phase && pub.computing_phase) {
    document.getElementById("thinking-phase").textContent = phaseDisplay(pub.computing_phase);
    banner.classList.remove("hidden");
  } else {
    banner.classList.add("hidden");
  }
}


/* ---------------------------------------------------------------------------
 * Bundle refresh — fetches everything, updates phases array, possibly
 * advances the displayed phase.
 * ------------------------------------------------------------------------- */

async function refreshBundle() {
  const r = await fetch("/api/live/bundle");
  state.bundle = await r.json();

  const allPhases = new Set([
    ...Object.keys(state.bundle.snapshots || {}),
    ...Object.keys(state.bundle.board || {}),
  ]);
  state.phases = Array.from(allPhases).sort(comparePhases);

  if (state.phases.length === 0) {
    return;
  }

  // Choose displayed phase
  if (!state.phase || !state.phases.includes(state.phase)) {
    state.phase = state.phases[state.phases.length - 1];
  } else if (state.live.followLatest) {
    state.phase = state.phases[state.phases.length - 1];
  }

  // Default power: pick one with a snapshot for current phase, or first
  if (!state.power) {
    state.power = POWERS.find(p => snapshotFor(p, state.phase)) || POWERS[0];
  }

  buildPhaseScrubber();
  renderAll();
}


/* ---------------------------------------------------------------------------
 * Below this line: rendering. Adapted from view.js — same DOM IDs so the
 * shared CSS works, slight tweaks for live mode (phase log panel).
 * ------------------------------------------------------------------------- */

const SEASON_ORDER = { SPRING: 0, FALL: 1, WINTER: 2 };
const KIND_ORDER = { MOVES: 0, RETREATS: 1, ADJUSTMENTS: 2 };

function phaseKey(phase) {
  const parts = phase.split("-");
  if (parts.length !== 3) return [9999, 9, 9];
  return [
    parseInt(parts[0], 10) || 0,
    SEASON_ORDER[parts[1]] ?? 9,
    KIND_ORDER[parts[2]] ?? 9,
  ];
}

function comparePhases(a, b) {
  const ka = phaseKey(a), kb = phaseKey(b);
  for (let i = 0; i < 3; i++) if (ka[i] !== kb[i]) return ka[i] - kb[i];
  return 0;
}

function phaseDisplay(phase) {
  if (!phase) return "—";
  const [y, s, k] = phase.split("-");
  return `${s.toLowerCase().replace(/^./, c=>c.toUpperCase())} ${y} · ${k.toLowerCase()}`;
}

function snapshotFor(power, phase) {
  const direct = state.bundle?.snapshots?.[phase]?.[power];
  if (direct) return direct;
  const idx = state.phases.indexOf(phase);
  for (let i = idx - 1; i >= 0; i--) {
    const ph = state.phases[i];
    const sn = state.bundle?.snapshots?.[ph]?.[power];
    if (sn) return sn;
  }
  return null;
}

function boardFor(phase) { return state.bundle?.board?.[phase] || {}; }
function messagesFor(phase) { return state.bundle?.messages?.[phase] || []; }
function summaryFor(power) { return state.bundle?.summary?.agents?.[power] || null; }


/* ---------------- Power tabs / scrubber ---------------- */

function buildPowerTabs() {
  const nav = document.getElementById("power-tabs");
  nav.innerHTML = "";
  POWERS.forEach(p => {
    const snap = snapshotFor(p, state.phase);
    const arch = snap?.archetype || "PLAYER_DEFAULT";
    const sc = boardFor(state.phase)[p]?.sc_count ?? "—";
    const btn = document.createElement("button");
    btn.className = "power-tab" + (p === state.power ? " selected" : "");
    btn.style.setProperty("--power-color", POWER_COLORS[p]);
    btn.dataset.power = p;
    btn.innerHTML = `
      <span class="archetype">${arch.replace(/_/g, " ")}</span>
      <span class="power-name">${p}</span>
      <span class="sc-badge">${sc} SC</span>
    `;
    btn.addEventListener("click", () => {
      state.power = p;
      state.selectedNodeId = null;
      renderAll();
    });
    nav.appendChild(btn);
  });
}

function buildPhaseScrubber() {
  const scrubber = document.getElementById("phase-scrubber");
  scrubber.min = 0;
  scrubber.max = Math.max(0, state.phases.length - 1);
  scrubber.value = state.phases.indexOf(state.phase);
}


/* ---------------- Render orchestration ---------------- */

function renderAll() {
  document.getElementById("phase-label").textContent = phaseDisplay(state.phase);
  document.getElementById("msg-phase-label").textContent = phaseDisplay(state.phase);
  buildPowerTabs();
  renderCharacterCard();
  renderInspectorDefault();
  renderMap();
  renderGraph();
  renderMessages();
  renderPhaseLog();
}


/* ---------------- Character card ---------------- */

function renderCharacterCard() {
  const snap = snapshotFor(state.power, state.phase);
  const sum = summaryFor(state.power) || {};
  const arch = snap?.archetype || sum.archetype || "PLAYER_DEFAULT";

  const archEl = document.getElementById("char-archetype");
  archEl.textContent = `${arch.replace(/_/g, " ")} · ${state.power}`;
  archEl.style.color = POWER_COLORS[state.power];

  document.getElementById("char-brief").textContent =
    POWER_BRIEFS[arch] || POWER_BRIEFS.PLAYER_DEFAULT;

  const s = snap?.summary || {};
  const stats = [
    ["Beliefs", `${s.beliefs_active||0} active · ${s.beliefs_proto||0} proto`],
    ["Intents", `${s.intents_active||0} active · ${(s.intents_total||0)-(s.intents_active||0)} other`],
    ["Promises kept", `${s.incoming_commitments_kept||0} / ${(s.incoming_commitments_kept||0)+(s.incoming_commitments_broken||0)+(s.incoming_commitments_pending||0)}`],
    ["Predictions", `${s.predictions_confirmed||0} ✓ · ${s.predictions_refuted||0} ✗ · ${s.predictions_open||0} ◌`],
    ["SC", String(sum.final_sc ?? "—")],
    ["Self-credibility", sum.self_commitments ? `${sum.self_commitments.kept}/${sum.self_commitments.kept+sum.self_commitments.broken}` : "—"],
  ];
  document.getElementById("char-stats").innerHTML = stats.map(([k,v]) =>
    `<span class="stat-label">${k}</span><span class="stat-value">${v}</span>`
  ).join("");
}


/* ---------------- Inspector default ---------------- */

function renderInspectorDefault() {
  const snap = snapshotFor(state.power, state.phase);
  const detail = document.getElementById("inspector-detail");
  if (!snap) {
    detail.innerHTML = `<div class="hint">No snapshot for ${state.power} at this phase.</div>`;
    return;
  }
  const beliefs = snap.beliefs || [];
  const incoming = snap.incoming_commitments || [];
  const self_c = snap.self_commitments || [];
  const preds = snap.recent_predictions || [];

  const html = [];
  html.push(`<h3>Beliefs (${beliefs.length})</h3>`);
  if (beliefs.length === 0) {
    html.push(`<div class="hint">No beliefs yet.</div>`);
  } else {
    const sorted = [...beliefs].sort((a,b) => {
      const order = {active: 0, proto: 1, retired: 2};
      return (order[a.status]||9) - (order[b.status]||9);
    });
    for (const b of sorted) html.push(beliefRowHTML(b));
  }
  html.push(`<h3>Commitments — others to me (${incoming.length})</h3>`);
  if (incoming.length === 0) html.push(`<div class="hint">None.</div>`);
  else for (const c of incoming) html.push(commitmentRowHTML(c, "in"));

  html.push(`<h3>Commitments — me to others (${self_c.length})</h3>`);
  if (self_c.length === 0) html.push(`<div class="hint">None.</div>`);
  else for (const c of self_c) html.push(commitmentRowHTML(c, "out"));

  const recentPreds = preds.slice(0, 8);
  html.push(`<h3>Recent predictions (${recentPreds.length} of ${preds.length})</h3>`);
  for (const p of recentPreds) html.push(predictionRowHTML(p));

  detail.innerHTML = html.join("");

  detail.querySelectorAll(".belief-row").forEach(el => {
    el.addEventListener("click", () => {
      state.selectedNodeId = el.dataset.id;
      renderInspectorDetail();
      highlightGraphNode(el.dataset.id);
    });
  });
  detail.querySelectorAll(".commit-row").forEach(el => {
    el.addEventListener("click", () => {
      state.selectedNodeId = el.dataset.id;
      renderInspectorDetail();
      highlightGraphNode(el.dataset.id);
    });
    el.addEventListener("mouseenter", () => highlightCommitOnMap(el.dataset.id));
    el.addEventListener("mouseleave", () => clearMapHighlights());
  });
  detail.querySelectorAll(".pred-row").forEach(el => {
    el.addEventListener("click", () => {
      state.selectedNodeId = el.dataset.id;
      renderInspectorDetail();
      highlightGraphNode(el.dataset.id);
    });
  });
}

function beliefRowHTML(b) {
  return `<div class="belief-row" data-id="${b.id}">
    <div class="head">${escapeHTML(b.head)}</div>
    <div class="meta-line">
      <span class="pill ${b.status}">${b.status}</span>
      about ${b.about} · ${b.type} · hp ${(b.hp||0).toFixed(2)} ·
      ev <span style="color:${STATUS_COLOR.confirmed}">+${b.evidence_for_count||0}</span>
      <span style="color:${STATUS_COLOR.broken}">-${b.evidence_against_count||0}</span>
    </div>
  </div>`;
}

function commitmentRowHTML(c, dir) {
  const who = dir === "in" ? c.speaker : c.to;
  const whoLabel = dir === "in" ? `${who} →` : `→ ${who}`;
  const what = describeCommit(c);
  return `<div class="commit-row" data-id="${c.id}" data-target-prov="${c.target_province||''}" data-counter="${who||''}">
    <span class="who">${whoLabel}</span>
    <span class="what">${escapeHTML(what)}</span>
    <span class="pill ${c.status}">${c.status}</span>
  </div>`;
}

function describeCommit(c) {
  if (c.raw) return c.raw;
  if (c.type === "non_aggression") return `non-aggression with ${c.counterparty || c.to || c.speaker}`;
  if (c.type === "demilitarize") return `demilitarize ${c.subject_province || ""} with ${c.counterparty || c.to || c.speaker}`;
  if (c.target_province) return `${c.type} ${c.target_province}`;
  return c.type;
}

function predictionRowHTML(p) {
  const conf = (p.confidence != null) ? p.confidence.toFixed(2) : "?";
  return `<div class="pred-row" data-id="${p.id}">
    <div class="pred-line">
      <span class="pill ${p.status}">${p.status}</span>
      ${p.about} · ${p.type} ${p.target||''} <span style="color:#8a8275">(conf ${conf})</span>
    </div>
    ${p.rationale ? `<div class="pred-rationale">${escapeHTML(p.rationale)}</div>` : ""}
  </div>`;
}

function escapeHTML(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}


/* ---------------- Inspector detail ---------------- */

function renderInspectorDetail() {
  const id = state.selectedNodeId;
  if (!id) { renderInspectorDefault(); return; }

  const snap = snapshotFor(state.power, state.phase);
  if (!snap) return;

  const find = arr => (arr || []).find(x => x.id === id);
  const target =
    find(snap.beliefs) ||
    find(snap.strategic_intents) ||
    find(snap.incoming_commitments) ||
    find(snap.self_commitments) ||
    find(snap.recent_predictions);

  const detail = document.getElementById("inspector-detail");
  if (!target) {
    detail.innerHTML = `<div class="hint">Node not found in current snapshot.</div>`;
    return;
  }

  const back = `<a href="javascript:void(0)" id="back-to-summary" style="font-size:11px; color:#8a8275">← back to all</a>`;
  const html = [back];

  if (id.startsWith("belief:")) {
    html.push(`<h3>Belief <span class="pill ${target.status}">${target.status}</span></h3>`);
    html.push(`<div class="head">${escapeHTML(target.head)}</div>`);
    if (target.body) html.push(`<div class="body">${escapeHTML(target.body)}</div>`);
    html.push(metaTable({
      "id": target.id, "about": target.about, "type": target.type,
      "hp": (target.hp||0).toFixed(3),
      "critic_score": (target.critic_score||0).toFixed(2),
      "times_foveated": target.times_foveated,
      "evidence_for": target.evidence_for_count,
      "evidence_against": target.evidence_against_count,
      "formed_at": target.formed_at,
      "last_updated": target.last_updated,
      "retire_reason": target.retire_reason || "—",
    }));
  } else if (id.startsWith("intent:")) {
    html.push(`<h3>Strategic intent <span class="pill ${target.status}">${target.status}</span></h3>`);
    html.push(`<div class="head">${escapeHTML(target.head)}</div>`);
    if (target.body) html.push(`<div class="body">${escapeHTML(target.body)}</div>`);
    html.push(metaTable({
      "id": target.id, "horizon": target.horizon,
      "target_powers": (target.target_powers||[]).join(", "),
      "target_provinces": (target.target_provinces||[]).join(", "),
      "predictions_confirmed": target.predictions_confirmed,
      "predictions_refuted": target.predictions_refuted,
      "supporting_plans": target.supporting_plan_count,
      "sc_delta": target.sc_delta_under_intent,
      "formed_at": target.formed_at,
    }));
  } else if (id.startsWith("cmt:")) {
    const dir = target.speaker ? "incoming" : "outgoing";
    const who = target.speaker || target.to;
    html.push(`<h3>Commitment (${dir}) <span class="pill ${target.status}">${target.status}</span></h3>`);
    html.push(`<div class="head">${escapeHTML(target.raw || describeCommit(target))}</div>`);
    html.push(metaTable({
      "id": target.id, "type": target.type,
      [dir === "incoming" ? "speaker" : "to"]: who,
      "subject_province": target.subject_province || "—",
      "target_province": target.target_province || "—",
      "deadline": target.deadline_phase,
      "resolved_at": target.resolved_at || "—",
      "evidence_count": target.evidence_count,
    }));
  } else if (id.startsWith("pred:")) {
    html.push(`<h3>Prediction <span class="pill ${target.status}">${target.status}</span></h3>`);
    html.push(`<div class="head">${target.about} · ${target.type} ${target.target||''}</div>`);
    if (target.rationale) html.push(`<div class="body">${escapeHTML(target.rationale)}</div>`);
    html.push(metaTable({
      "id": target.id, "window_kind": target.window_kind,
      "prediction_window": target.prediction_window,
      "confidence": (target.confidence||0).toFixed(2),
      "formed_at": target.formed_at,
    }));
  }

  detail.innerHTML = html.join("");
  document.getElementById("back-to-summary")?.addEventListener("click", () => {
    state.selectedNodeId = null;
    renderInspectorDefault();
    clearGraphHighlights();
  });
}

function metaTable(obj) {
  const rows = Object.entries(obj).map(([k,v]) =>
    `<span class="k">${k}</span><span class="v">${escapeHTML(v ?? "")}</span>`
  ).join("");
  return `<div class="meta">${rows}</div>`;
}


/* ---------------- Map ---------------- */

let _mapInited = false;
function initMapSVG() {
  if (state.calibration?.image_size) {
    const [w, h] = state.calibration.image_size;
    const svg = document.getElementById("map-svg");
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  }
  if (!document.getElementById("map-tooltip")) {
    const tip = document.createElement("div");
    tip.id = "map-tooltip";
    document.getElementById("map-wrap").appendChild(tip);
  }
  _mapInited = true;
}

function provCoord(code) { return state.calibration?.provinces?.[code] || null; }

function renderMap() {
  if (!_mapInited) initMapSVG();
  const svg = d3.select("#map-svg");
  svg.selectAll("*").remove();

  const board = boardFor(state.phase);
  const snap = snapshotFor(state.power, state.phase);

  const gCommit = svg.append("g").attr("class", "g-commit");
  const gSc     = svg.append("g").attr("class", "g-sc");
  const gUnits  = svg.append("g").attr("class", "g-units");
  const gHits   = svg.append("g").attr("class", "g-hits");

  const ownership = inferOwnership(board);
  for (const [code, owner] of Object.entries(ownership)) {
    const c = provCoord(code);
    if (!c) continue;
    gSc.append("circle")
      .attr("class", "sc-dot")
      .attr("cx", c[0]).attr("cy", c[1] + 18).attr("r", 6)
      .attr("fill", owner ? POWER_COLORS[owner] : "#aaa")
      .attr("stroke", "#1f1d18").attr("stroke-width", 1.2);
  }

  for (const power of POWERS) {
    const units = board[power]?.units || [];
    for (const u of units) {
      const c = provCoord(u.prov);
      if (!c) continue;
      const isSelected = (power === state.power);
      gUnits.append("rect")
        .attr("class", "unit-bg")
        .attr("x", c[0] - 16).attr("y", c[1] - 11)
        .attr("width", 32).attr("height", 22).attr("rx", 11)
        .attr("fill", POWER_COLORS[power])
        .attr("stroke", isSelected ? "#1f1d18" : "white")
        .attr("stroke-width", isSelected ? 2.5 : 1.5)
        .attr("opacity", isSelected ? 1.0 : 0.85);
      gUnits.append("text")
        .attr("class", "unit").attr("x", c[0]).attr("y", c[1])
        .attr("fill", "white").attr("stroke", "none")
        .text(`${u.kind} ${u.prov}`);
    }
  }

  if (snap) {
    for (const c of (snap.self_commitments || [])) {
      if (!c.target_province) continue;
      const tc = provCoord(c.target_province);
      if (!tc) continue;
      const color = c.status === "broken" ? STATUS_COLOR.broken
                  : c.status === "kept"   ? STATUS_COLOR.kept
                  : STATUS_COLOR.pending;
      gCommit.append("circle")
        .attr("class", `commit-marker commit-${c.id}`)
        .attr("cx", tc[0]).attr("cy", tc[1]).attr("r", 18)
        .attr("fill", "none").attr("stroke", color)
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", c.type === "not_move_to" ? "4 3" : "none")
        .attr("opacity", 0.0);
    }
  }

  if (state.calibration?.provinces) {
    for (const [code, [x, y]] of Object.entries(state.calibration.provinces)) {
      gHits.append("circle")
        .attr("class", "prov-hit")
        .attr("cx", x).attr("cy", y).attr("r", 26)
        .on("mouseenter", (e) => showProvTooltip(e, code, board))
        .on("mouseleave", hideProvTooltip)
        .on("click", () => onProvClick(code));
    }
  }
}

function inferOwnership(board) {
  const out = {};
  for (const [power, info] of Object.entries(board)) {
    for (const u of (info.units || [])) out[u.prov] = power;
  }
  return out;
}

function showProvTooltip(evt, code, board) {
  const tip = document.getElementById("map-tooltip");
  const ownership = inferOwnership(board);
  const owner = ownership[code];
  const lines = [`<b>${code}</b>`];
  if (owner) lines.push(`held by ${owner}`);
  const snap = snapshotFor(state.power, state.phase);
  if (snap) {
    const cmts = [
      ...(snap.self_commitments || []),
      ...(snap.incoming_commitments || []),
    ].filter(c => c.target_province === code || c.subject_province === code);
    if (cmts.length) lines.push(`<i>${cmts.length} commitment${cmts.length>1?"s":""} touch this</i>`);
  }
  tip.innerHTML = lines.join("<br>");
  tip.style.display = "block";
  const wrap = document.getElementById("map-wrap").getBoundingClientRect();
  tip.style.left = (evt.clientX - wrap.left + 12) + "px";
  tip.style.top = (evt.clientY - wrap.top + 12) + "px";
}

function hideProvTooltip() {
  const tip = document.getElementById("map-tooltip");
  if (tip) tip.style.display = "none";
}

function onProvClick(code) { flashGraphForProvince(code); }

function highlightCommitOnMap(commitId) {
  d3.selectAll(`#map-svg .commit-marker`).attr("opacity", 0);
  d3.selectAll(`#map-svg .commit-${commitId}`)
    .attr("opacity", 1).attr("r", 18)
    .transition().duration(800)
    .attr("r", 26).attr("opacity", 0.4);
}

function clearMapHighlights() {
  d3.selectAll(`#map-svg .commit-marker`).attr("opacity", 0);
}


/* ---------------- Mind graph ---------------- */

let _graphSim = null;

function renderGraph() {
  const svg = d3.select("#graph-svg");
  svg.selectAll("*").remove();

  const snap = snapshotFor(state.power, state.phase);
  if (!snap) {
    svg.append("text").attr("x", 20).attr("y", 30)
      .attr("fill", "#8a8275")
      .text(`No mind-state for ${state.power} at this phase.`);
    return;
  }

  const { nodes, links } = buildGraphData(snap);

  const defs = svg.append("defs");
  defs.append("marker").attr("id", "arrow")
    .attr("viewBox", "0 -5 10 10").attr("refX", 10).attr("refY", 0)
    .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
    .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "#8a8275");

  const w = svg.node().clientWidth || 600;
  const h = svg.node().clientHeight || 400;

  const others = POWERS.filter(p => p !== state.power);
  others.forEach((p, i) => {
    const angle = (i / others.length) * 2 * Math.PI - Math.PI/2;
    const node = nodes.find(n => n.id === `power:${p}`);
    if (node) {
      node.fx = w/2 + Math.cos(angle) * Math.min(w, h) * 0.30;
      node.fy = h/2 + Math.sin(angle) * Math.min(w, h) * 0.30;
    }
  });
  const self = nodes.find(n => n.id === `power:${state.power}`);
  if (self) { self.fx = w/2; self.fy = h/2; }

  if (_graphSim) _graphSim.stop();
  _graphSim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id)
      .distance(d => d.distance || 60).strength(0.5))
    .force("charge", d3.forceManyBody().strength(-160))
    .force("collide", d3.forceCollide().radius(d => (d.r || 8) + 4))
    .force("x", d3.forceX(w/2).strength(0.02))
    .force("y", d3.forceY(h/2).strength(0.02));

  const linkSel = svg.append("g").attr("class", "links")
    .selectAll("line").data(links).join("line")
    .attr("class", d => `link ${d.cls || ""}`);

  const nodeSel = svg.append("g").attr("class", "nodes")
    .selectAll("g").data(nodes).join("g")
    .attr("class", "node").attr("data-id", d => d.id)
    .style("opacity", 0)
    .on("click", (e, d) => onGraphNodeClick(d))
    .on("mouseenter", (e, d) => focusNode(d.id))
    .on("mouseleave", () => clearGraphHighlights());

  nodeSel.each(function(d) {
    const sel = d3.select(this);
    if (d.kind === "self") {
      sel.append("circle").attr("r", 18)
        .attr("fill", POWER_COLORS[state.power]).attr("stroke", "#1f1d18").attr("stroke-width", 2);
      sel.append("text").attr("dy", 1).attr("fill", "white")
        .style("font-weight", "700").text(state.power.slice(0,3));
    } else if (d.kind === "power") {
      sel.append("circle").attr("r", 14)
        .attr("fill", POWER_COLORS[d.power] || "#ddd").attr("opacity", 0.55)
        .attr("stroke", "#1f1d18").attr("stroke-width", 1);
      sel.append("text").attr("dy", 1).attr("fill", "white").text(d.power.slice(0,3));
    } else if (d.kind === "belief") {
      sel.append("circle").attr("r", 8)
        .attr("fill", STATUS_COLOR[d.status] || "#aaa")
        .attr("stroke", "#1f1d18").attr("stroke-width", 0.8);
    } else if (d.kind === "prediction") {
      sel.append("circle").attr("r", 5)
        .attr("fill", STATUS_COLOR[d.status] || "#888")
        .attr("stroke", "#1f1d18").attr("stroke-width", 0.6);
    } else if (d.kind === "intent") {
      sel.append("rect")
        .attr("x", -10).attr("y", -7).attr("width", 20).attr("height", 14)
        .attr("fill", "#8a5b9a").attr("opacity", 0.75)
        .attr("stroke", "#1f1d18").attr("stroke-width", 0.8);
    }
  });

  nodeSel.transition().duration(500).style("opacity", 1);

  _graphSim.on("tick", () => {
    linkSel
      .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    nodeSel.attr("transform", d => `translate(${d.x},${d.y})`);
  });
}

function buildGraphData(snap) {
  const nodes = [];
  const links = [];
  nodes.push({ id: `power:${state.power}`, kind: "self", power: state.power, r: 18 });
  for (const p of POWERS) {
    if (p === state.power) continue;
    nodes.push({ id: `power:${p}`, kind: "power", power: p, r: 14 });
  }

  let beliefs = snap.beliefs || [];
  if (state.foveated) beliefs = beliefs.filter(b => b.status !== "retired");
  for (const b of beliefs) {
    nodes.push({ id: b.id, kind: "belief", status: b.status, r: 8, about: b.about, _data: b });
    links.push({ source: `power:${b.about}`, target: b.id, distance: 50, cls: "evidence" });
  }

  let preds = snap.recent_predictions || [];
  if (state.foveated) preds = preds.filter(p => p.status === "open" || p.status === "confirmed");
  for (const p of preds) {
    nodes.push({ id: p.id, kind: "prediction", status: p.status, r: 5, about: p.about, _data: p });
    links.push({ source: `power:${p.about}`, target: p.id, distance: 35, cls: "evidence" });
  }

  const intents = snap.strategic_intents || [];
  for (const it of intents) {
    nodes.push({ id: it.id, kind: "intent", status: it.status, r: 10, _data: it });
    const targets = it.target_powers || [];
    if (targets.length === 0) {
      links.push({ source: `power:${state.power}`, target: it.id, distance: 70, cls: "evidence" });
    } else {
      for (const t of targets) {
        if (POWERS.includes(t)) {
          links.push({ source: `power:${t}`, target: it.id, distance: 70, cls: "evidence" });
        }
      }
    }
  }

  for (const c of (snap.self_commitments || [])) {
    if (!POWERS.includes(c.to)) continue;
    links.push({
      source: `power:${state.power}`, target: `power:${c.to}`,
      distance: 80, cls: `commitment-${c.status}`, _commit: c,
    });
  }
  for (const c of (snap.incoming_commitments || [])) {
    if (!POWERS.includes(c.speaker)) continue;
    links.push({
      source: `power:${c.speaker}`, target: `power:${state.power}`,
      distance: 80, cls: `commitment-${c.status}`, _commit: c,
    });
  }

  return { nodes, links };
}

function onGraphNodeClick(d) {
  if (d.kind === "self" || d.kind === "power") return;
  state.selectedNodeId = d.id;
  renderInspectorDetail();
  highlightGraphNode(d.id);
}

function highlightGraphNode(id) {
  if (!id) { clearGraphHighlights(); return; }
  d3.selectAll("#graph-svg .node").classed("faded", true).classed("highlight", false);
  d3.selectAll("#graph-svg .link").classed("faded", true).classed("highlight", false);
  d3.selectAll(`#graph-svg .node[data-id="${id}"]`)
    .classed("faded", false).classed("highlight", true);
  d3.selectAll("#graph-svg .link").filter(function(d) {
    return d.source.id === id || d.target.id === id;
  }).classed("faded", false).classed("highlight", true)
    .each(function(d) {
      const otherId = d.source.id === id ? d.target.id : d.source.id;
      d3.select(`#graph-svg .node[data-id="${otherId}"]`).classed("faded", false);
    });
}

function focusNode(id) { highlightGraphNode(id); }

function clearGraphHighlights() {
  d3.selectAll("#graph-svg .node").classed("faded", false).classed("highlight", false);
  d3.selectAll("#graph-svg .link").classed("faded", false).classed("highlight", false);
}

function flashGraphForProvince(prov) {
  const matches = [];
  d3.selectAll("#graph-svg .node").each(function(d) {
    const dat = d._data;
    if (!dat) return;
    if (dat.target_province === prov || dat.target === prov ||
        (dat.target_provinces && dat.target_provinces.includes(prov))) {
      matches.push(d.id);
    }
  });
  if (matches.length === 0) return;
  d3.selectAll("#graph-svg .node").classed("faded", true);
  matches.forEach(id => {
    d3.selectAll(`#graph-svg .node[data-id="${id}"]`).classed("faded", false).classed("highlight", true);
  });
  setTimeout(clearGraphHighlights, 1500);
}


/* ---------------- Messages ---------------- */

function renderMessages() {
  const list = document.getElementById("msg-list");
  list.innerHTML = "";
  const msgs = messagesFor(state.phase);
  const showSelf = document.getElementById("msg-filter-self").checked;
  const showToSelf = document.getElementById("msg-filter-to-self").checked;
  const showOther = document.getElementById("msg-filter-other").checked;

  const filtered = msgs.filter(m => {
    const isFromSelf = m.from === state.power;
    const isToSelf = m.to.includes(state.power) || m.to.includes("ALL");
    if (isFromSelf && showSelf) return true;
    if (!isFromSelf && isToSelf && showToSelf) return true;
    if (!isFromSelf && !isToSelf && showOther) return true;
    return false;
  });

  if (filtered.length === 0) {
    list.innerHTML = `<div class="hint" style="padding:12px; color:#8a8275; font-style:italic;">No messages match the current filters.</div>`;
    return;
  }

  for (const m of filtered) {
    const div = document.createElement("div");
    div.className = "msg";
    const fromColor = POWER_COLORS[m.from] || "#888";
    div.innerHTML = `
      <div class="from-to">
        <span style="color:${fromColor}; font-weight:600">${m.from}</span> →
        ${m.to.map(t => `<span style="color:${POWER_COLORS[t]||'#888'}">${t}</span>`).join(", ")}
      </div>
      <div class="text">${formatMessageText(m.text)}</div>
    `;
    list.appendChild(div);
  }
}

function formatMessageText(text) {
  const m = text.match(/^([\s\S]*?)\[\[commit\s*([\s\S]*?)\]\]\s*$/);
  if (!m) return escapeHTML(text);
  const prose = m[1].trim();
  const commit = m[2].trim();
  return `${escapeHTML(prose)}<span class="commit-block">${escapeHTML(commit)}</span>`;
}


/* ---------------- Phase log (live-only) ---------------- */

function renderPhaseLog() {
  const list = document.getElementById("phase-log-list");
  if (!list) return;
  list.innerHTML = "";
  const entries = state.bundle?.log || [];
  if (entries.length === 0) {
    list.innerHTML = `<div class="hint" style="padding:12px; color:#8a8275; font-style:italic;">No phases run yet.</div>`;
    return;
  }
  // Most recent first
  const reversed = [...entries].reverse();
  for (const e of reversed) {
    const div = document.createElement("div");
    div.className = "phase-log-entry" + (e.phase === state.phase ? " current" : "");

    const took = (e.finished_at && e.started_at)
      ? Math.round(e.finished_at - e.started_at) + "s"
      : "?";

    let lines = [`
      <div class="ple-header">
        <span class="ple-phase">${phaseDisplay(e.phase)}</span>
        <span>${took}</span>
      </div>
      <div class="ple-stats">
        ${e.n_messages||0} msgs · ${e.n_messages_with_commitspeak||0} cs ·
        ${e.n_orders||0} orders · ${e.n_predictions||0} predictions
      </div>
    `];

    // Per-agent lifecycle events
    const eventLines = [];
    for (const power of POWERS) {
      const a = e.per_agent?.[power];
      if (!a) continue;
      const bits = [];
      if (a.beliefs_promoted)   bits.push(`<span class="ev-promotion">+${a.beliefs_promoted} belief${a.beliefs_promoted>1?"s":""}</span>`);
      if (a.beliefs_retired)    bits.push(`<span class="ev-retirement">-${a.beliefs_retired} belief${a.beliefs_retired>1?"s":""}</span>`);
      if (a.intents_promoted)   bits.push(`<span class="ev-promotion">+${a.intents_promoted} intent${a.intents_promoted>1?"s":""}</span>`);
      if (a.intents_retired)    bits.push(`<span class="ev-retirement">-${a.intents_retired} intent${a.intents_retired>1?"s":""}</span>`);
      if (a.commitments_graded) bits.push(`${a.commitments_graded} cmt${a.commitments_graded>1?"s":""} graded`);
      if (bits.length) {
        eventLines.push(`<div class="ple-event"><span class="pwr" style="color:${POWER_COLORS[power]}">${power.slice(0,3)}</span>${bits.join(" · ")}</div>`);
      }
    }
    if (eventLines.length) {
      lines.push(`<div class="ple-events">${eventLines.join("")}</div>`);
    }

    // SC changes
    if (e.sc_changes && e.sc_changes.length) {
      const scLines = e.sc_changes.map(c => {
        const fromS = c.from
          ? `<span class="ple-sc-from" style="color:${POWER_COLORS[c.from]||'#888'}">${c.from.slice(0,3)}</span>`
          : `<span class="ple-sc-from">—</span>`;
        const toS = c.to
          ? `<span class="ple-sc-to" style="color:${POWER_COLORS[c.to]||'#888'}">${c.to.slice(0,3)}</span>`
          : `<span class="ple-sc-to">—</span>`;
        return `<div class="ple-sc-row">${c.prov}: ${fromS}<span class="ple-sc-arrow">→</span>${toS}</div>`;
      });
      lines.push(`<div class="ple-sc">${scLines.join("")}</div>`);
    }

    div.innerHTML = lines.join("");
    div.addEventListener("click", () => {
      state.phase = e.phase;
      state.live.followLatest = false;
      state.selectedNodeId = null;
      buildPhaseScrubber();
      renderAll();
    });
    list.appendChild(div);
  }
}


/* ---------------- Go ---------------- */

window.addEventListener("DOMContentLoaded", () => {
  init().catch(err => {
    console.error("init failed", err);
    document.body.insertAdjacentHTML("afterbegin",
      `<div style="padding:20px; background:#fee; color:#900; font-family:monospace;">
        Error: ${escapeHTML(err.message || String(err))}
      </div>`);
  });
});
