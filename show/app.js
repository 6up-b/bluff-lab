/* Bluff Lab — Show. Pure renderer of show/bundle/ (built by `blab pick-demo` from real runs).
   Nothing here invents a number: every value on screen is read from a bundle file.
   Presenter-paced: Space/→/PageDown next cue, ←/PageUp previous, R reset, H HUD, F fullscreen. */
"use strict";

const BUNDLE = (new URLSearchParams(location.search).get("bundle") || "bundle").replace(/\/$/, "");
const B = {};                    // loaded bundle data
const HAVE = {};                 // file → exists
const W = { events: [], bySeq: new Map(), reveals: new Map(), seats: {}, start: null, wid: "" };
const NAMES = { p1: "The Profiler", p2: "Player 2", p3: "Player 3" };
const RANKS = { K: "Kings", Q: "Queens", A: "Aces" };
const RANK1 = { K: "King", Q: "Queen", A: "Ace" };
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const f2 = (x) => (x === null || x === undefined || Number.isNaN(+x)) ? "—" : (+x).toFixed(2);
const fN = (x) => (x === null || x === undefined) ? "—" : (Number.isInteger(+x) ? String(+x) : (+x).toFixed(1));
const pct = (x) => (x === null || x === undefined) ? "—" : Math.round(+x * 100) + "%";
const SVGNS = "http://www.w3.org/2000/svg";

/* ================================================================ loading */
async function getText(p) {
  try { const r = await fetch(`${BUNDLE}/${p}`, { cache: "no-store" }); if (!r.ok) return null; return await r.text(); }
  catch { return null; }
}
async function getJSON(p) {
  const t = await getText(p); if (t === null) return null;
  try { HAVE[p] = true; return JSON.parse(t); } catch (e) { console.warn("bad json", p, e); return null; }
}
async function exists(p) {
  if (p in HAVE) return HAVE[p];
  try { const r = await fetch(`${BUNDLE}/${p}`, { method: "HEAD", cache: "no-store" }); HAVE[p] = r.ok; }
  catch { HAVE[p] = false; }
  return HAVE[p];
}
function parseJSONL(t) {
  return (t || "").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l)).sort((a, b) => a.seq - b.seq);
}

async function load() {
  const [world, twin] = await Promise.all([getText("world.jsonl"), getText("twin.jsonl")]);
  HAVE["world.jsonl"] = world !== null; HAVE["twin.jsonl"] = twin !== null;
  if (world) indexWorld(parseJSONL(world));
  if (twin) B.twin = parseJSONL(twin);
  const names = ["reveal", "dossier", "ablation", "longcon", "results", "forge", "cues"];
  const got = await Promise.all(names.map((n) => getJSON(`${n}.json`)));
  names.forEach((n, i) => { B[n] = got[i]; HAVE[`${n}.json`] = got[i] !== null; });
  cues = normalizeCues(B.cues || defaultCues());
  await markSkips();
}

function indexWorld(evs) {
  W.events = evs;
  for (const e of evs) W.bySeq.set(e.seq, e);
  W.start = (evs.find((e) => e.type === "world_start") || {}).payload || null;
  W.wid = evs.length ? evs[0].world : "";
  W.run = evs.length ? evs[0].run : "";
  for (const s of (W.start?.seats || [])) W.seats[s.seat] = s;
  // obs id → {reveal, claim}: the claim a reveal resolved is the last claim before it in that round.
  let lastClaim = null;
  for (const e of evs) {
    if (e.type === "deal") lastClaim = null;
    if (e.type === "claim") lastClaim = e;
    if (e.type === "reveal") {
      const id = e.payload.obs_id || `${e.world}:g${e.game}r${e.round}`;
      W.reveals.set(id, { reveal: e, claim: lastClaim });
    }
  }
}

/* ================================================================ cues */
let cues = [];
let idx = 0;                      // number of cues applied
const S = {};                     // presentation state
function resetState() {
  Object.assign(S, { view: "title", seq: null, frozen: false, inspect: null, evidence: false,
    dossierGame: null, dossierAbout: null, surgery: [], flipped: false, caption: "" });
}

function normalizeCues(raw) {
  return (raw || []).map((c, i) => {
    if (!c.actions) {               // SPEC §12.2 form: {"at_seq":412,"action":"freeze"}
      const a = { ...c }; if (a.at_seq !== undefined && a.seq === undefined) a.seq = a.at_seq;
      return { name: c.name || c.action, say: c.say || "", requires: c.requires || [], actions: [a] };
    }
    return { name: c.name || `cue ${i + 1}`, say: c.say || "", requires: c.requires || [], actions: c.actions };
  });
}

async function markSkips() {
  for (const c of cues) {
    const need = new Set(c.requires);
    for (const a of c.actions) {
      if (a.action === "audio" && a.file) need.add(a.file);
      if (a.action === "view" && a.name === "forge") need.add("forge.json");
      if (a.action === "view" && a.name === "split") need.add("twin.jsonl");
      if (["play_to", "freeze", "inspect"].includes(a.action)) need.add("world.jsonl");
      if (a.action === "surgery") {
        need.add("reveal.json");
        const v = variantOf(a.variant);
        if (B.reveal && (!v || !v.available)) c.skip = `skipped: no data (${a.variant} not run)`;
      }
      if (a.action === "dossier") need.add("dossier.json");
    }
    for (const f of need) if (!(await exists(f))) { c.skip = `skipped: no data (${f})`; break; }
  }
}

function applyAction(a, live) {
  switch (a.action) {
    case "view": S.view = a.name; break;
    case "play_to": S.seq = a.seq; S.frozen = false; S.view = "table"; break;
    case "freeze": S.seq = a.seq; S.frozen = true; S.view = "table"; break;
    case "inspect": S.inspect = a.seq; S.view = "inspector"; break;
    case "evidence": S.evidence = a.hyp_id || true; S.view = "inspector"; break;
    case "dossier": S.dossierGame = a.game; if (a.about) S.dossierAbout = a.about; S.view = "dossier"; break;
    case "surgery": if (!S.surgery.includes(a.variant)) S.surgery.push(a.variant); S.view = "surgery"; break;
    case "flip": S.flipped = true; break;
    case "audio": if (live) playAudio(a.file); break;
    case "caption": S.caption = a.text || ""; break;
    case "clear_caption": S.caption = ""; break;
    default: console.warn("unknown cue action", a);
  }
}
function applyCue(c, live) {
  if (c.skip) return;
  S.caption = c.keepCaption ? S.caption : "";
  for (const a of c.actions) applyAction(a, live);
}
function next() {
  if (idx >= cues.length) return;
  applyCue(cues[idx], true); idx++;
  while (idx < cues.length && cues[idx - 1].skip) { applyCue(cues[idx], true); idx++; }  // never land on a skip
  render();
}
function prev() {
  let target = Math.max(0, idx - 1);
  while (target > 0 && cues[target - 1].skip) target--;
  goto(target);
}
function goto(n) {
  // Rebuilding state (←, R, ?cue=) jumps without transitions so the frame is final immediately.
  document.body.classList.add("instant");
  resetState(); stopAudio();
  for (let i = 0; i < n; i++) applyCue(cues[i], false);
  idx = n; render();
  void document.body.offsetHeight;
  requestAnimationFrame(() => requestAnimationFrame(() => document.body.classList.remove("instant")));
}
function playAudio(file) {
  const el = $("#audio"); el.src = `${BUNDLE}/${file}`; el.currentTime = 0; el.play().catch(() => {});
}
function stopAudio() { const el = $("#audio"); el.pause(); }

/* Fallback when cues.json is missing: the FLOW.md order, built from reveal.json. */
function defaultCues() {
  const r = B.reveal; const c = [];
  if (!r) return [{ name: "title", actions: [{ action: "view", name: "title" }] }];
  const g = r.decision_game || (W.bySeq.get(r.seq) || {}).game;
  c.push({ name: "cold open: claim", actions: [{ action: "play_to", seq: r.claim_seq }] });
  if (r.ask_seq) c.push({ name: "ask", actions: [{ action: "play_to", seq: r.ask_seq }, { action: "audio", file: `voice/ask_${r.ask_seq}.mp3` }] });
  c.push({ name: "freeze", actions: [{ action: "freeze", seq: r.seq }] });
  c.push({ name: "inspect", actions: [{ action: "inspect", seq: r.seq }] });
  c.push({ name: "evidence", actions: [{ action: "evidence" }] });
  for (const gg of [4, 8, g]) c.push({ name: `dossier g${gg}`, actions: [{ action: "dossier", game: gg, about: r.decision?.target }] });
  for (const v of ["full", "as_of:0", "false_memory", "profiler_blind"]) c.push({ name: `surgery ${v}`, actions: [{ action: "surgery", variant: v }] });
  c.push({ name: "flip", actions: [{ action: "flip" }] });
  for (const v of ["ablation", "trust", "split", "results", "decisions", "forge", "close"]) c.push({ name: v, actions: [{ action: "view", name: v }] });
  return c;
}

/* ================================================================ helpers */
function variantOf(key) { return (B.reveal?.variants || []).find((v) => v.key === key) || null; }
function revealedClaimOfDecision(seq) {
  // The reveal that follows a decision in the same round (if it was challenged).
  const d = W.bySeq.get(seq); if (!d) return null;
  for (const e of W.events) {
    if (e.seq <= seq) continue;
    if (e.type === "deal" || e.type === "game_end") break;
    if (e.type === "claim") break;
    if (e.type === "reveal") return e;
  }
  return null;
}
function claimBefore(seq) {
  let c = null; for (const e of W.events) { if (e.seq > seq) break; if (e.type === "claim") c = e; if (e.type === "deal") c = null; }
  return c;
}
function cardEl(rank, { up = false, tableRank = null, sm = false } = {}) {
  const bad = up && tableRank && rank !== tableRank && rank !== "J";
  const cls = ["card", sm ? "sm" : "", up ? "up" : "", bad ? "bad" : "", up && rank === "J" ? "wild" : ""].join(" ");
  return `<div class="${cls}" title="${up ? esc(rank) : "face down"}">${up ? esc(rank) : ""}</div>`;
}
function isBluff(cards, tr) { return cards.some((c) => c !== tr && c !== "J"); }
function svg(w, h, inner, cls = "") { return `<svg class="${cls}" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" xmlns="${SVGNS}">${inner}</svg>`; }
function awaiting(title, body) { return `<div class="awaiting"><b>${esc(title)}</b>${body}</div>`; }
function seatName(seat, reveal = false) {
  const role = W.seats[seat]?.role;
  return esc(NAMES[seat] || seat) + (reveal && role && seat !== "p1" ? `<span class="role">${esc(role)}</span>` : "");
}

/* ================================================================ table */
function tableState(upto) {
  const st = { game: 0, round: 0, lives: {}, hands: {}, rank: null, pile: [], bubbles: {}, reveal: null, actor: null, winner: undefined };
  for (const e of W.events) {
    if (e.seq > upto) break;
    const p = e.payload;
    switch (e.type) {
      case "game_start": st.game = e.game; st.lives = { ...p.lives }; st.pile = []; st.bubbles = {}; st.reveal = null; st.winner = undefined; break;
      case "deal": st.round = e.round; st.game = e.game; st.rank = p.table_rank; st.pile = []; st.bubbles = {}; st.reveal = null;
        st.hands = Object.fromEntries(Object.entries(p.hands).map(([k, v]) => [k, v.length])); break;
      case "claim": st.hands[e.actor] = Math.max(0, (st.hands[e.actor] || 0) - p.count);
        st.pile.push({ actor: e.actor, count: p.count, cards: p.cards, seq: e.seq });
        st.bubbles[e.actor] = { text: p.utterance, kind: "claim" }; st.actor = e.actor; break;
      case "ask": st.bubbles[e.actor || "p1"] = { text: p.text, kind: "ask" }; break;
      case "answer": if (e.actor) st.bubbles[e.actor] = { text: p.text, kind: "answer" }; break;
      case "challenge": st.bubbles[e.actor] = { text: "Challenge.", kind: "act" }; st.actor = e.actor; break;
      case "reveal": st.reveal = p; break;
      case "life_lost": st.lives[p.seat] = p.lives_left; break;
      case "game_end": st.winner = p.winner; break;
    }
  }
  return st;
}

function renderTable() {
  const el = $("#v-table");
  if (!W.events.length) { el.innerHTML = awaiting("Awaiting data", "No <code>bundle/world.jsonl</code> yet. Run <code>blab pick-demo --run local</code> after a real local world exists."); return; }
  const upto = S.seq ?? 0;
  const st = tableState(upto);
  const flipShow = S.flipped && B.reveal && B.reveal.reveal_seq && upto < B.reveal.reveal_seq;   // flip shows the picked reveal
  const lastPile = st.pile[st.pile.length - 1];
  const revealCards = st.reveal ? st.reveal.cards : (flipShow ? B.reveal.reveal.cards : null);
  const pile = st.pile.map((pl, i) => {
    const up = revealCards && i === st.pile.length - 1;
    return pl.cards.map((c, j) => cardEl(up ? revealCards[j] ?? c : c, { up, tableRank: st.rank })).join("");
  }).join('<div style="width:18px"></div>');
  const seat = (s) => {
    const lives = st.lives[s] ?? 0, max = 3;
    const pips = Array.from({ length: max }, (_, i) => `<span class="${i < lives ? "on" : ""}">●</span>`).join("");
    const b = st.bubbles[s];
    const tag = b ? { claim: "says", ask: "asks", answer: "answers", act: "" }[b.kind] : "";
    const n = st.hands[s] ?? 0;
    return `<div class="seat ${s} ${lives <= 0 ? "out" : ""} ${st.actor === s ? "actor" : ""}">
      <div><span class="name">${seatName(s, S.flipped)}</span></div>
      <div class="lives">${pips}</div>
      <div class="hand">${Array.from({ length: n }, () => cardEl("?", { sm: true })).join("")}</div>
      <div class="bubble ${b ? "on" : ""} ${b?.kind === "ask" ? "ask" : ""}">${b ? `${tag ? `<span class="tag">${tag}</span>` : ""}${esc(b.text)}` : ""}</div>
    </div>`;
  };
  el.classList.toggle("frozen", !!S.frozen);
  el.innerHTML = `<div class="felt-oval"></div>
    <div class="gameplate">Game <b>${st.game}</b>, round <b>${st.round}</b></div>
    <div class="rankplate"><div class="lbl">Table rank</div><div class="rk">${esc(RANKS[st.rank] || "—")}</div></div>
    <div class="pile">${pile}</div>
    ${seat("p2")}${seat("p3")}${seat("p1")}
    <div class="freeze-stamp">Frozen at decision #${esc(S.seq)}</div>`;
  where(`Game ${st.game}, round ${st.round}`);
}

/* ================================================================ inspector */
function hypById(id) {
  const ck = B.dossier?.checkpoints; if (!ck) return null;
  for (const g of Object.keys(ck).sort((a, b) => b - a)) {
    const h = (ck[g].hypotheses || []).find((x) => x.id === id); if (h) return h;
  }
  return null;
}
function predFeatures(pred, out = new Set()) {
  if (!pred || typeof pred !== "object") return out;
  if (pred.f) out.add(pred.f);
  for (const k of ["all", "any"]) (pred[k] || []).forEach((p) => predFeatures(p, out));
  if (pred.not) predFeatures(pred.not, out);
  return out;
}

function gaugeSVG(pc, pf, { w = 1080, h = 170, label = true } = {}) {
  const x0 = 20, x1 = w - 20, X = (v) => x0 + (x1 - x0) * Math.max(0, Math.min(1, v));
  const y = 62, bh = 46;
  let s = `<rect x="${x0}" y="${y}" width="${x1 - x0}" height="${bh}" fill="#0c2426"/>`;
  if (pc !== null && pc !== undefined) s += `<rect x="${x0}" y="${y}" width="${X(pc) - x0}" height="${bh}" fill="var(--amber)"/>`;
  if (pf !== null && pf !== undefined && pc !== null && pc !== undefined) {
    const a = Math.min(pc, pf), b = Math.max(pc, pf);
    s += `<rect x="${X(a)}" y="${y}" width="${Math.max(0, X(b) - X(a))}" height="${bh}" fill="var(--ice)"/>`;
    s += `<rect x="${X(pf) - 3}" y="${y - 12}" width="6" height="${bh + 24}" fill="var(--bone)"/>`;
  }
  s += `<line x1="${X(.5)}" x2="${X(.5)}" y1="${y - 22}" y2="${y + bh + 22}" stroke="var(--bone)" stroke-width="2" stroke-dasharray="6 6"/>`;
  if (label) {
    s += `<text x="${X(.5)}" y="${y + bh + 50}" fill="var(--bone-dim)" font-size="22" text-anchor="middle">challenge above 0.50</text>`;
    if (pc !== null && pc !== undefined) s += X(pc) < x0 + 240
      ? `<text x="${x0}" y="${y - 20}" fill="var(--amber)" font-size="28" font-weight="700" text-anchor="start">card math ${f2(pc)}</text>`
      : `<text x="${X(pc)}" y="${y - 20}" fill="var(--amber)" font-size="28" font-weight="700" text-anchor="end">card math ${f2(pc)}</text>`;
    if (pf !== null && pf !== undefined) s += `<text x="${Math.min(X(pf) + 14, x1)}" y="${y - 20}" fill="var(--bone)" font-size="28" font-weight="700" text-anchor="${X(pf) > x1 - 260 ? "end" : "start"}">with memory ${f2(pf)}</text>`;
    for (const t of [0, .25, .5, .75, 1]) s += `<text x="${X(t)}" y="${y + bh + 22}" fill="var(--bone-faint)" font-size="18" text-anchor="middle">${t === .5 ? "" : t}</text>`;
  }
  return svg(w, h, s);
}

function memLineHTML(m, evidenceOpen) {
  const qt = m.q_true || [], qf = m.q_false || [];
  const rows = (m.evidence_obs || []).map((id) => W.reveals.get(id) ? { id, ...W.reveals.get(id) } : { id });
  const shown = rows.slice(-5);
  const evid = shown.map((r) => {
    if (!r.reveal) return `<div class="row"><span>${esc(r.id.split(":").pop())}</span><span class="u">not in this world's log</span><span></span></div>`;
    const tr = (W.events.find((e) => e.type === "deal" && e.game === r.reveal.game && e.round === r.reveal.round) || {}).payload?.table_rank;
    const p = r.reveal.payload;
    return `<div class="row"><span>Game ${r.reveal.game} r${r.reveal.round}</span>
      <span class="u">${esc(r.claim?.payload.utterance || "")}</span>
      <span class="cs">${p.cards.map((c) => cardEl(c, { up: true, tableRank: tr, sm: true })).join("")}
      <span class="res ${p.was_bluff ? "bluffword" : "honestword"}">${p.was_bluff ? "bluff" : "honest"}</span></span></div>`;
  }).join("");
  return `<div class="line">
    <div class="txt">${esc(m.text)}</div>
    <div class="meta"><span class="chip ${esc(m.status)}">${esc(m.status)}</span>
      <span class="chip ${esc(m.provenance)}">${m.provenance === "untrusted" ? "no evidence (injected)" : "from evidence"}</span>
      <span>Applies: ${m.n_true} reveals, bluff rate ${f2(qt[0])} (${f2(qt[1])}–${f2(qt[2])})</span>
      <span>Otherwise: ${m.n_false} reveals, ${f2(qf[0])} (${f2(qf[1])}–${f2(qf[2])})</span></div>
    <div class="evid ${evidenceOpen ? "on" : ""}">${evid}${rows.length > shown.length ? `<div class="more">and ${rows.length - shown.length} earlier reveals</div>` : ""}${rows.length ? "" : '<div class="more">No evidence ids recorded on this line.</div>'}</div>
  </div>`;
}

function claimCardHTML(d, ev) {
  const cl = d.claim || {}; const feats = cl.features || {};
  const hot = new Set();
  for (const m of d.memory_used || []) {
    const h = hypById(m.hyp_id); predFeatures(h?.predicate, hot);
    if (!h && m.aspect === "verbal_tell") hot.add("formal_register");
  }
  const keys = ["formal_register", "hand_size", "claim_count", "trusted_streak", "lives_target", "was_challenged_last_round"];
  const chips = keys.filter((k) => k in feats).map((k) => `<span class="feat ${hot.has(k) ? "hot" : ""}">${esc(k.replace(/_/g, " "))} <b>${esc(String(feats[k]))}</b></span>`).join("");
  const who = d.target || ev?.actor;
  return `<div class="claimcard"><div class="who">${seatName(who, S.flipped)} claims</div>
    <div class="what">${esc(cl.count)} ${esc((cl.count === 1 ? RANK1 : RANKS)[cl.rank] || cl.rank || "")}</div>
    <div class="said">“${esc(cl.utterance)}”</div><div class="feats">${chips}</div></div>`;
}

function qSourceText(d) {
  const src = d.q_source || "";
  if (src.startsWith("base:")) return `From memory: ${seatName(src.slice(5))}'s base bluff rate q = ${f2(d.q_used)}.`;
  if (src.startsWith("drift:")) return `From memory: ${seatName(src.slice(6))}'s recent bluff rate after a detected shift, q = ${f2(d.q_used)}.`;
  if (src === "logreg" || String(d.agent || "").startsWith("logreg")) return "Predicted directly by the logistic-regression baseline (no q).";
  if (!src) return d.q_used === null || d.q_used === undefined ? "No memory: card math only." : `q = ${f2(d.q_used)}.`;
  const m = (d.memory_used || []).find((x) => x.hyp_id === src);
  return m ? `From the memory line below: q = ${f2(d.q_used)}.` : `From memory (${esc(src)}): q = ${f2(d.q_used)}.`;
}
function renderInspector() {
  const el = $("#v-inspector");
  const e = W.bySeq.get(S.inspect);
  if (!e || e.type !== "decision") { el.innerHTML = awaiting("Awaiting data", `No decision event at seq ${esc(S.inspect)} in <code>world.jsonl</code>.`); return; }
  const d = e.payload, cm = d.card_math || {};
  const mem = d.memory_used || [];
  const focus = S.evidence === true ? mem[0]?.hyp_id : S.evidence;
  const evs = d.ev || {};
  const evCells = Object.entries(evs).map(([k, v]) => `<div class="cell ${k === d.action ? "chosen" : ""}">EV ${esc(k.replace("_", " "))}<span class="n">${f2(v)}</span></div>`).join("");
  el.innerHTML = `<div class="insp">
    <div>${claimCardHTML(d, e)}
      <div class="ev" style="margin-top:36px">${evCells}<span class="action-stamp">${esc((d.action || "").replace("_", " "))}</span></div>
      <div class="foot" style="margin-top:22px">Recorded at decision time (event #${e.seq}, memory version ${esc(d.memory_version)}).
        Card math: ${esc(cm.U)} unseen cards, ${esc(cm.m)} could match, claimant held ${esc(cm.h)}, claimed ${esc(cm.k)}.
        ${d.explore ? "<br><b>This was an exploration draw.</b>" : ""}</div>
    </div>
    <div>
      <div class="gauge-wrap"><h3>Probability this claim is a bluff</h3>${gaugeSVG(cm.p_cards, d.p_final)}
        <div class="foot">${qSourceText(d)} p = card math + (1 − card math) × q.</div></div>
      <div class="mem"><h3>${mem.length ? "Memory lines used" : "No specific memory line matched this claim"}</h3>
        ${mem.map((m) => memLineHTML(m, S.evidence && m.hyp_id === focus)).join("")}</div>
    </div></div>`;
  where(`Game ${e.game}, round ${e.round}, decision #${e.seq}`);
}

/* ================================================================ dossier */
function renderDossier() {
  const el = $("#v-dossier"); const D = B.dossier;
  if (!D?.checkpoints) { el.innerHTML = `<h2 class="title">Dossier</h2>` + awaiting("Awaiting data", "No <code>bundle/dossier.json</code> (built from <code>ckpt/gNN.json</code>)."); return; }
  const games = Object.keys(D.checkpoints).map(Number).sort((a, b) => a - b);
  const want = S.dossierGame ?? games[games.length - 1];
  const g = games.filter((x) => x <= want).pop() ?? games[0];
  const ck = D.checkpoints[g];
  const about = S.dossierAbout || B.reveal?.decision?.target || "p3";
  const hyps = (ck.hypotheses || []).filter((h) => !h.about || h.about === about);
  const grp = { confirmed: [], provisional: [], superseded: [], refuted: [] };
  for (const h of hyps) {
    if (h.valid_until_game !== null && h.valid_until_game !== undefined || h.superseded_by) grp.superseded.push(h);
    else (grp[h.status] || grp.provisional).push(h);
  }
  const line = (h, kind) => {
    const qt = h.q_true || [];
    const st = kind === "superseded"
      ? `valid game ${h.valid_from_game ?? "?"} to ${h.valid_until_game ?? "?"}${h.superseded_by ? `, replaced by ${esc(h.superseded_by)}` : ""}`
      : `${h.n_true ? `${h.n_true} reveals where it applies, bluff rate ${f2(qt[0])}` : "no reveals yet where it applies"}; ${h.n_false ?? 0} otherwise; since game ${h.valid_from_game ?? h.created_game ?? "?"}${h.provenance === "untrusted" ? "; no evidence (injected)" : ""}`;
    return `<div class="h"><span class="t">${esc(h.text)}</span><span class="st">${st}</span></div>`;
  };
  const block = (k, title) => `<div class="grp ${k}"><h4>${title} (${grp[k].length})</h4>${grp[k].map((h) => line(h, k)).join("") || '<div class="empty">none</div>'}</div>`;
  const base = ck.base_summary?.[about];
  const maxG = Math.max(...games, 20), w = 1776;
  const X = (v) => 20 + (w - 40) * v / maxG;
  let sl = `<line x1="20" x2="${w - 20}" y1="30" y2="30" stroke="var(--rule)" stroke-width="4"/>`;
  for (const t of games) sl += `<circle cx="${X(t)}" cy="30" r="${t === g ? 14 : 6}" fill="${t === g ? "var(--ice)" : "var(--bone-faint)"}"/>` +
    (t % 2 === 0 || t === g ? `<text x="${X(t)}" y="66" fill="${t === g ? "var(--ice)" : "var(--bone-faint)"}" font-size="20" text-anchor="middle">${t}</text>` : "");
  el.innerHTML = `<h2 class="title">${seatName(about, S.flipped)}: the file it built, as of game ${g}</h2>
    <p class="sub">From memory checkpoint g${String(g).padStart(2, "0")} (memory version ${esc(ck.memory_version ?? "?")}).
    ${base ? `Base bluff rate <b>${f2(base[0])}</b> (${f2(base[1])}–${f2(base[2])}).` : ""}</p>
    <div class="slider">${svg(w, 80, sl)}</div>
    <div class="doss">${block("confirmed", "Confirmed")}${block("provisional", "Provisional")}${block("superseded", "Superseded")}${block("refuted", "Refuted")}</div>`;
  where(`Dossier, game ${g}`);
}

/* ================================================================ surgery */
const VARIANT_TEXT = {
  "full": ["Full memory", "What it actually knew at this decision"],
  "as_of:0": ["Memory erased", "Checkpoint as of game 0; this game's reveals only"],
  "false_memory": ["False memory planted", null],
  "profiler_blind": ["Provenance-blind agent", null],
};
function renderSurgery() {
  const el = $("#v-surgery"); const R = B.reveal;
  if (!R) { el.innerHTML = `<h2 class="title">Memory surgery</h2>` + awaiting("Awaiting data", "No <code>bundle/reveal.json</code>. It holds real <code>replay_decision</code> outputs; nothing is drawn without it."); return; }
  const d = R.decision || {}; const cl = d.claim || {};
  const rv = R.reveal; const tr = cl.rank;
  const wasBluff = rv ? rv.was_bluff : null;
  const keys = ["full", "as_of:0", "false_memory", "profiler_blind"];
  const fm = variantOf("false_memory")?.variant;
  const cols = keys.map((k) => {
    const v = variantOf(k); const on = S.surgery.includes(k);
    const [t, sub0] = VARIANT_TEXT[k];
    const sub = sub0 || (k === "false_memory" ? `Adds “${esc(fm?.text || "")}”, no evidence behind it`
      : `Treats every line as true, including “${esc((v?.variant || fm)?.text || "")}”`);
    if (!v || !v.available || !v.result) {
      return `<div class="col absent ${on ? "on" : ""}"><div class="vt">${t}</div><div class="vs">${sub}</div><div class="p">not run</div>
        <div class="dp">${esc(v?.note || "no matching world in this run")}</div></div>`;
    }
    const r = v.result; const act = r.action;
    const right = wasBluff === null ? null : (act === "challenge") === !!wasBluff;
    return `<div class="col ${on ? "on" : ""}"><div class="vt">${t}</div><div class="vs">${sub}</div>
      <div class="p">${f2(r.p_final)}</div><div class="plbl">probability of bluff</div>
      ${gaugeSVG(r.card_math?.p_cards ?? d.card_math?.p_cards, r.p_final, { w: 360, h: 120, label: false })}
      <div class="act ${esc(act)}">${esc((act || "").replace("_", " "))}${right === null ? "" : `<span class="judge ${right ? "right" : "wrong"}">${right ? "right" : "wrong"}</span>`}</div>
      <div class="dp">${k === "full" ? (R.full_matches_log === false ? "warning: replay differs from log" : "matches the logged decision") : `${r.flipped ? "action flipped" : "same action"}; Δp ${r.delta_p >= 0 ? "+" : ""}${f2(r.delta_p)}`}</div>
</div>`;
  }).join("");
  const showCards = S.flipped && rv;
  const cards = (showCards ? rv.cards : (R.claim?.cards || Array(cl.count || 0).fill("?"))).map((c) => cardEl(c, { up: !!showCards, tableRank: tr })).join("");
  el.classList.toggle("flipped", !!S.flipped);
  el.innerHTML = `<div class="surg-head"><div><div class="who">Same cards, same claim, same model. Only the memory changes. ${seatName(d.target, S.flipped)}, game ${esc(R.decision_game ?? "")}:</div>
      <div class="said">“${esc(cl.utterance)}”</div></div>
      <div class="cards">${cards}<span class="verdict ${showCards ? "on" : ""} ${wasBluff ? "bluffword" : "honestword"}">${wasBluff ? "bluff" : "honest"}</span></div></div>
    <div class="cols">${cols}</div>
    ${R.selection && R.selection.exact === false ? `<div class="selnote">Closest match, not exact: ${esc(R.selection.note || "")}</div>` : ""}`;
  where(`Decision #${R.seq}, world ${R.world}`);
}

/* ================================================================ ablation */
function axisY(x0, x1, yScale, ticks, fmt = (t) => t) {
  return ticks.map((t) => `<line x1="${x0}" x2="${x1}" y1="${yScale(t)}" y2="${yScale(t)}" stroke="var(--rule)" stroke-width="1"/>
    <text x="${x0 - 14}" y="${yScale(t) + 7}" fill="var(--bone-faint)" font-size="20" text-anchor="end">${fmt(t)}</text>`).join("");
}
function renderAblation() {
  const el = $("#v-ablation"); const A = B.ablation;
  const head = `<h2 class="title">Does the memory help? Ablation on the same seeds</h2>`;
  if (!A?.bars?.length) { el.innerHTML = head + awaiting("Awaiting data", "No <code>bundle/ablation.json</code> yet. It comes from <code>blab ablate</code> branch worlds via <code>blab analyze</code>."); return; }
  const w = 1776, h = 600, x0 = 90, x1 = w - 20, yT = 30, yB = h - 90;
  const bars = A.bars; const n = bars.length;
  const top = Math.min(1, Math.max(...bars.map((b) => b.hi ?? b.mean ?? 0)) + 0.1);
  const Y = (v) => yB - (yB - yT) * (v / top);
  const bw = Math.min(180, (x1 - x0) / n * 0.6);
  const step = top <= 0.3 ? 0.05 : top <= 0.6 ? 0.1 : 0.25;
  const ticks = []; for (let t = 0; t <= top + 1e-9; t += step) ticks.push(+t.toFixed(2));
  let s = axisY(x0, x1, Y, ticks, (t) => pct(t));
  bars.forEach((b, i) => {
    const cx = x0 + (x1 - x0) * (i + 0.5) / n;
    const col = b.variant === "full" ? "var(--ice)" : b.variant.startsWith("minus") ? "var(--bluff)" : b.variant.startsWith("as_of") ? "var(--amber)" : "var(--violet)";
    if (b.mean !== null && b.mean !== undefined) {
      s += `<rect x="${cx - bw / 2}" y="${Y(b.mean)}" width="${bw}" height="${yB - Y(b.mean)}" fill="${col}" opacity=".9"/>`;
      if (b.lo !== null && b.hi !== null) s += `<line x1="${cx}" x2="${cx}" y1="${Y(b.lo)}" y2="${Y(b.hi)}" stroke="var(--bone)" stroke-width="3"/>
        <line x1="${cx - 16}" x2="${cx + 16}" y1="${Y(b.lo)}" y2="${Y(b.lo)}" stroke="var(--bone)" stroke-width="3"/>
        <line x1="${cx - 16}" x2="${cx + 16}" y1="${Y(b.hi)}" y2="${Y(b.hi)}" stroke="var(--bone)" stroke-width="3"/>`;
      s += `<text x="${cx}" y="${Y(b.hi ?? b.mean) - 14}" fill="var(--bone)" font-size="32" font-weight="700" text-anchor="middle">${pct(b.mean)}</text>`;
    }
    s += `<text x="${cx}" y="${yB + 36}" fill="var(--bone)" font-size="26" font-weight="600" text-anchor="middle">${esc(b.label || b.variant)}</text>
      <text x="${cx}" y="${yB + 66}" fill="var(--bone-faint)" font-size="20" text-anchor="middle">n = ${esc(b.n)}</text>`;
  });
  const fl = A.flips;
  el.innerHTML = head + `<p class="sub">Win rate in games ${esc(A.window || "11–20")}, memory frozen at game ${esc(A.at_game ?? 10)}, ${esc(A.table || "")} table. Bars: mean; whiskers: 95% bootstrap CI over seeds.</p>
    <div class="chart">${svg(w, h, s)}</div>
    <div class="foot">${fl && fl.n ? `Decision flips (full vs erased, same recorded states): ${esc(fl.flipped)} of ${esc(fl.n)}.` : ""} Source: ${esc(A.source || "")}</div>`;
  where(`Ablation, ${esc(A.source || "")}`);
}

/* ================================================================ trust curve */
function renderTrust() {
  const el = $("#v-trust"); const L = B.longcon;
  const head = `<h2 class="title">The long con: trust earned honestly, then spent</h2>`;
  if (!L?.games?.length) { el.innerHTML = head + awaiting("Awaiting data", "No <code>bundle/longcon.json</code> yet. It comes from <code>longcon</code> worlds via <code>blab analyze</code>."); return; }
  const w = 1300, h = 640, x0 = 90, x1 = w - 30, yT = 30, yB = h - 70;
  const G = L.games; const gmin = G[0], gmax = G[G.length - 1];
  const X = (g) => x0 + (x1 - x0) * (g - gmin) / Math.max(1, gmax - gmin);
  const Y = (v) => yB - (yB - yT) * v;
  let s = axisY(x0, x1, Y, [0, .25, .5, .75, 1], (t) => pct(t));
  for (const g of G) if (g % 2 === 0 || g === gmin) s += `<text x="${X(g)}" y="${yB + 34}" fill="var(--bone-faint)" font-size="20" text-anchor="middle">${g}</text>`;
  s += `<text x="${(x0 + x1) / 2}" y="${yB + 64}" fill="var(--bone-dim)" font-size="22" text-anchor="middle">game</text>`;
  const g0 = L.g0?.median ?? L.g0?.value;
  if (g0 !== null && g0 !== undefined) s += `<line x1="${X(g0)}" x2="${X(g0)}" y1="${yT}" y2="${yB}" stroke="var(--bluff)" stroke-width="2" stroke-dasharray="8 6"/>
    <text x="${X(g0) - 10}" y="${yB - 16}" fill="var(--bluff)" font-size="26" font-weight="700" text-anchor="end">⚑ turns at game ${f2(g0).replace(/\.00$/, "")}${L.n_worlds > 1 ? " (median)" : ""}</text>`;
  const line = (pts, color, width, dash = "") => {
    const ok = pts.filter((p) => p && p.mean !== null && p.mean !== undefined);
    if (!ok.length) return "";
    const band = ok.filter((p) => p.lo !== null && p.lo !== undefined && p.hi !== null && p.hi !== undefined);
    let out = "";
    if (band.length > 1) out += `<path d="M${band.map((p) => `${X(p.game)},${Y(p.hi)}`).join("L")}L${band.slice().reverse().map((p) => `${X(p.game)},${Y(p.lo)}`).join("L")}Z" fill="${color}" opacity=".12"/>`;
    out += `<path d="M${ok.map((p) => `${X(p.game)},${Y(p.mean)}`).join("L")}" fill="none" stroke="${color}" stroke-width="${width}" stroke-dasharray="${dash}" stroke-linejoin="round"/>`;
    return out;
  };
  const COL = { counter: "var(--amber)", bayes: "var(--violet)", bayes_discount: "#e6c98f", profiler: "var(--ice)" };
  s += line(L.true_honesty || [], "var(--bone)", 6);
  const agents = Object.keys(L.agents || {});
  for (const a of agents) s += line(L.agents[a], COL[a] || "var(--bone-dim)", 4, a === "counter" ? "10 8" : "");
  const det = L.detection?.profiler?.median;
  if (det !== null && det !== undefined) {
    const p = (L.agents?.profiler || []).find((q) => q.game === Math.round(det));
    s += `<text x="${X(det)}" y="${Y(p?.mean ?? 0.5) + 14}" fill="var(--ice)" font-size="40" text-anchor="middle">◎</text>`;
  }
  const sl = L.sleeper_seat ? seatName(L.sleeper_seat) : "The sleeper";
  const leg = [`<span><i style="background:var(--bone)"></i>${sl}'s true honesty</span>`]
    .concat(agents.map((a) => `<span><i style="background:${COL[a] || "var(--bone-dim)"}"></i>${esc(a)} estimate</span>`)).join("");
  const ec = L.exploitation_cost || {}; const rtd = L.rtd || {};
  const stat = Object.keys(ec).concat(Object.keys(rtd)).filter((v, i, a) => a.indexOf(v) === i).map((a) => `
    <div class="kv" style="font-size:28px;padding:12px 0;border-top:1px solid var(--rule)"><b style="color:${COL[a] || "var(--bone)"}">${esc(a)}</b>
      <span style="display:block;font-size:22px;color:var(--bone-dim)">${ec[a] && ec[a].n ? `${esc(L.exploitation_label || "exploitation cost")}: ${fN(ec[a].mean)}${ec[a].lo !== null && ec[a].lo !== undefined ? ` (${fN(ec[a].lo)}–${fN(ec[a].hi)})` : ""}` : ""}${rtd[a] ? `<br>rounds to detection, median ${fN(rtd[a].median)} (detected in ${esc(rtd[a].detected ?? rtd[a].n)} of ${esc(rtd[a].worlds ?? "?")} worlds)` : ""}</span></div>`).join("");
  const twist = L.twist?.holds ? `<div style="font-family:var(--voice);font-style:italic;font-size:38px;line-height:1.2;margin-top:24px">Naive memory was more exploitable than no memory.</div>
     <div class="foot">${esc(L.twist.metric || "bayes − counter")}: ${fN(L.twist.diff?.mean)} (${fN(L.twist.diff?.lo)}–${fN(L.twist.diff?.hi)})</div>` : "";
  el.innerHTML = head + `<p class="sub">${sl} plays honest, then turns. Every memory of it is true; the attack is assuming people don't change.</p>
    <div style="display:grid;grid-template-columns:${w}px 1fr;gap:48px"><div class="chart">${svg(w, h, s)}<div class="legend">${leg}</div></div>
    <div style="margin-top:26px">${stat}${twist}<div class="foot">n = ${esc(L.n_worlds ?? "?")} longcon worlds. ${esc(L.estimate_note || "")}</div></div></div>`;
  where(`Long con, ${esc(L.source || "")}`);
}

/* ================================================================ results */
function renderResults() {
  const el = $("#v-results"); const R = B.results;
  const head = `<h2 class="title">Results</h2>`;
  const grid = R?.show_grid;
  if (!grid?.rows?.length) { el.innerHTML = head + awaiting("Awaiting data", "No <code>bundle/results.json</code> yet (<code>blab analyze</code>)."); return; }
  const th = grid.columns.map((c) => `<th>${esc(c.label)}</th>`).join("");
  const rows = grid.rows.map((r) => `<tr class="${r.agent === "profiler" || r.agent.startsWith("profiler+") ? "ours" : ""}"><td>${esc(r.agent)}</td>${grid.columns.map((c) => {
    const m = r.cells?.[c.key];
    if (!m || m.mean === null || m.mean === undefined) return `<td class="na">—</td>`;
    const v = c.fmt === "pct" ? pct(m.mean) : f2(m.mean);
    const ci = (m.lo !== null && m.lo !== undefined) ? (c.fmt === "pct" ? `${pct(m.lo)}–${pct(m.hi)}` : `${f2(m.lo)}–${f2(m.hi)}`) : "";
    return `<td>${v}<span class="ci">${ci}${m.n !== undefined ? ` n ${m.n}` : ""}</span></td>`;
  }).join("")}</tr>`).join("");
  el.innerHTML = head + `<p class="sub">${esc(R.n_worlds ?? "?")} worlds, run ${esc(R.run ?? "")}${grid.respond_rule ? `, ${esc(grid.respond_rule)} respond rule` : ""}. Means with 95% bootstrap CIs over seeds.</p>
    <table class="grid"><thead><tr><th>Agent</th>${th}</tr></thead><tbody>${rows}</tbody></table>`;
  where(`Results, run ${esc(R.run ?? "")}`);
}

/* ================================================================ beliefs vs decisions */
function renderDecisions() {
  const el = $("#v-decisions"); const R = B.results; const D = R?.decisions;
  const head = `<h2 class="title">Better beliefs, better decisions?</h2>`;
  if (!D?.rows?.length) { el.innerHTML = head + awaiting("Awaiting data", "No paired profiler − counter comparison in <code>bundle/results.json</code> yet."); return; }
  const metrics = [...new Set(D.rows.map((r) => r.metric))];
  const panel = (m) => {
    const rows = D.rows.filter((r) => r.metric === m);
    const r0 = rows[0]; const scale = r0.fmt === "pct" ? 100 : 1;
    const ext = Math.max(...rows.flatMap((r) => [Math.abs(r.delta.lo ?? r.delta.mean), Math.abs(r.delta.hi ?? r.delta.mean)])) * 1.25 || 0.1;
    const w = 820, x0 = 210, x1 = w - 40, rh = 96, h = rows.length * rh + 70;
    const X = (v) => x0 + (x1 - x0) * (v + ext) / (2 * ext);
    const fmt = (v) => r0.fmt === "pct" ? `${v >= 0 ? "+" : ""}${(v * scale).toFixed(1)} pts` : `${v >= 0 ? "+" : ""}${v.toFixed(3)}`;
    let s = `<line x1="${X(0)}" x2="${X(0)}" y1="10" y2="${h - 50}" stroke="var(--bone-faint)" stroke-width="2" stroke-dasharray="6 6"/>
      <text x="${X(0)}" y="${h - 22}" fill="var(--bone-faint)" font-size="20" text-anchor="middle">no difference</text>
      <text x="${r0.better === "lower" ? x0 : x1}" y="${h - 22}" fill="var(--honest)" font-size="20" text-anchor="${r0.better === "lower" ? "start" : "end"}">${r0.better === "lower" ? "← memory better" : "memory better →"}</text>`;
    rows.forEach((r, i) => {
      const y = 30 + i * rh + 20, d = r.delta;
      const good = r.better === "lower" ? (d.hi !== null && d.hi < 0) : (d.lo !== null && d.lo > 0);
      const bad = r.better === "lower" ? (d.lo !== null && d.lo > 0) : (d.hi !== null && d.hi < 0);
      const col = good ? "var(--honest)" : bad ? "var(--bluff)" : "var(--amber)";
      s += `<text x="0" y="${y + 8}" fill="var(--bone)" font-size="26" font-weight="600">${esc(r.rule)} rule</text>
        <text x="0" y="${y + 34}" fill="var(--bone-faint)" font-size="19">n = ${esc(d.n)} paired seeds</text>`;
      if (d.lo !== null && d.lo !== undefined) s += `<line x1="${X(d.lo)}" x2="${X(d.hi)}" y1="${y}" y2="${y}" stroke="${col}" stroke-width="5"/>`;
      s += `<circle cx="${X(d.mean)}" cy="${y}" r="11" fill="${col}"/>
        <text x="${X(d.mean)}" y="${y - 22}" fill="var(--bone)" font-size="26" font-weight="700" text-anchor="middle">${fmt(d.mean)}</text>`;
    });
    return `<div><h3 style="font-size:30px;margin:0 0 6px">${esc(r0.label)}</h3><div class="foot" style="margin:0 0 14px">Profiler − no-memory counter, same seeds; whiskers 95% CI</div>${svg(w, h, s)}</div>`;
  };
  const hl = D.headline?.holds ? `<div style="font-family:var(--voice);font-style:italic;font-size:52px;line-height:1.15;margin-top:30px;max-width:1500px">${esc(D.headline.text)}</div>` : "";
  el.innerHTML = head + `<p class="sub">Does memory sharpen its beliefs, and do sharper beliefs win more games? Paired against the no-memory counter on the same seeds, per respond rule.</p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:80px;margin-top:34px">${metrics.map(panel).join("")}</div>${hl}
    <div class="foot" style="margin-top:18px">${(D.texts || []).map(esc).join("<br>")}</div>`;
  where(`Run ${esc(R.run ?? "")}`);
}

/* ================================================================ forge */
function renderForge() {
  const el = $("#v-forge"); const F = B.forge;
  const head = `<h2 class="title">Strategy forge: memory compiled into code, run as untrusted</h2>`;
  if (!F) { el.innerHTML = head + awaiting("Not available", "No <code>bundle/forge.json</code>."); return; }
  const kv = (k, v) => `<div class="kv"><span>${esc(k)}</span>${v}</div>`;
  el.innerHTML = head + `<div class="forge"><pre>${esc(F.source || F.module_source || "")}</pre><div>
    ${kv("Static gate", F.gate?.ok ? '<span class="honestword">passed</span>' : `<span class="bluffword">rejected</span> ${esc(F.gate?.reason || "")}`)}
    ${kv("Sandbox", esc(F.sandbox_id || "—") + (F.network_blocked_verified ? ' <span class="chip evidence">network blocked</span>' : (F.sandbox_id ? ' <span class="chip none">restricted network</span>' : "")))}
    ${kv("Win rate, same seeds", `${pct(F.win_rate)}${F.ci ? ` (${pct(F.ci[0])}–${pct(F.ci[1])})` : ""} vs Profiler ${pct(F.baseline_win_rate)}`)}
    ${kv("Games, crashes, timeouts", `${esc(F.games)}, ${esc(F.crashes)}, ${esc(F.timeouts)}`)}
    ${kv("Promoted", F.promoted ? '<span class="honestword">yes</span>' : "no")}</div></div>`;
  where(`Forge, ${esc(F.policy_id || "")}`);
}

/* ================================================================ split */
function gameWinners(evs) { return evs.filter((e) => e.type === "game_end").map((e) => ({ game: e.game, winner: e.payload.winner })); }
function renderSplit() {
  const el = $("#v-split");
  const head = `<h2 class="title">Same seed, same cards, two minds</h2>`;
  if (!B.twin) { el.innerHTML = head + awaiting("Not available", "No <code>bundle/twin.jsonl</code>."); return; }
  const side = (evs, label) => {
    const ws = gameWinners(evs); const start = (evs.find((e) => e.type === "world_start") || {}).payload;
    const pol = start?.seats?.find((s) => s.seat === "p1")?.policy || "?";
    const won = ws.filter((g) => g.winner === "p1").length;
    return `<div class="side"><h3>${esc(label)}: ${esc(pol)}</h3><div class="meta">World ${esc(evs[0]?.world)}, seed ${esc(start?.seed)}</div>
      <div class="strip-games">${ws.map((g) => `<div class="${g.winner === "p1" ? "win" : ""}">${g.game}</div>`).join("")}</div>
      <div class="total">${won}<span>of ${ws.length} games won</span></div></div>`;
  };
  el.innerHTML = head + `<p class="sub">Highlighted: games won by seat 1.</p><div class="split">${side(W.events, "World")}${side(B.twin, "Twin")}</div>`;
  where("Split screen");
}

/* ================================================================ title / close */
function provenance() {
  const P = B.reveal?.provenance || {};
  return { run: P.run || W.run, world: P.world || W.wid, seed: P.seed ?? W.start?.seed, table: P.table, rule: P.rule };
}
function renderTitle() {
  const P = provenance();
  $("#v-title").innerHTML = `<div class="big">Bluff Lab</div>
    <div class="one">How AI agents learn through memory without retraining, and whether that memory makes them better or more exploitable.</div>
    <div class="prov">${W.events.length ? `One recorded world, picked by rule, not by hand: run <b>${esc(P.run)}</b>, world <b>${esc(P.world)}</b>, seed <b>${esc(P.seed)}</b>${P.table ? `, ${esc(P.table)} table` : ""}.` : "Awaiting data: no bundle yet."}</div>`;
  where("");
}
function renderClose() {
  $("#v-close").innerHTML = `<div class="line">The question isn't how much an agent remembers. It's which memories it should trust.</div>
    <div class="line2">Every decision recorded and replayable, every belief tied to the reveals behind it.</div>`;
  where("");
}

/* ================================================================ render */
function where(t) { $("#strip-where").innerHTML = t; }
const RENDER = { title: renderTitle, table: renderTable, inspector: renderInspector, dossier: renderDossier, surgery: renderSurgery,
  ablation: renderAblation, trust: renderTrust, split: renderSplit, results: renderResults, decisions: renderDecisions, forge: renderForge, close: renderClose };
const SCENE = { title: "", table: "The table", inspector: "Mind inspector", dossier: "Dossier", surgery: "Memory surgery",
  ablation: "Ablation", trust: "Long con", split: "Split screen", results: "Results", decisions: "Beliefs and decisions", forge: "Strategy forge", close: "" };
function render() {
  for (const el of document.querySelectorAll(".view")) el.classList.toggle("on", el.id === `v-${S.view}`);
  (RENDER[S.view] || renderTitle)();
  $("#strip-scene").textContent = SCENE[S.view] ?? "";
  const cap = $("#caption"); cap.textContent = S.caption || ""; cap.classList.toggle("on", !!S.caption);
  renderHUD();
}
function renderHUD() {
  const h = $("#hud"); const cur = cues[idx - 1]; const nxt = cues[idx];
  const files = ["world.jsonl", "reveal.json", "dossier.json", "ablation.json", "longcon.json", "results.json", "cues.json", "twin.jsonl", "forge.json"]
    .map((f) => `${HAVE[f] ? "✓" : "✗"} ${f}`).join("  ");
  h.innerHTML = `<div class="now">${idx}/${cues.length} ${esc(cur?.name || "start")}</div>
    <div class="say">${esc(cur?.say || "")}</div>
    ${nxt ? `<div>next: <b>${esc(nxt.name)}</b>${nxt.skip ? ` <span class="warn">${esc(nxt.skip)}</span>` : ""}</div>` : "<div>end of cues</div>"}
    ${B.cues ? "" : '<div class="warn">no cues.json: using built-in FLOW order</div>'}
    <ol>${cues.map((c, i) => `<li class="${i === idx - 1 ? "cur" : ""} ${c.skip ? "skip" : ""}">${esc(c.name)}${c.skip ? ` (${esc(c.skip)})` : ""}</li>`).join("")}</ol>
    <div class="files">bundle: ${esc(BUNDLE)}/<br>${files}</div>`;
}

function fit() {
  const s = Math.min(innerWidth / 1920, innerHeight / 1080);
  $("#stage").style.transform = `translate(-50%, -50%) scale(${s})`;
  $("#stage").style.transformOrigin = "center center";
  $("#stage").style.left = "50%"; $("#stage").style.top = "50%";
}
addEventListener("resize", fit);
addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key;
  if (k === " " || k === "ArrowRight" || k === "PageDown") { e.preventDefault(); next(); }
  else if (k === "ArrowLeft" || k === "PageUp") { e.preventDefault(); prev(); }
  else if (k === "r" || k === "R") goto(0);
  else if (k === "h" || k === "H") $("#hud").classList.toggle("on");
  else if (k === "f" || k === "F") { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen().catch(() => {}); }
});

(async function main() {
  fit(); resetState();
  await load();
  const n = parseInt(new URLSearchParams(location.search).get("cue") || "0", 10);   // ?cue=N jumps (rehearsal)
  goto(Math.min(n, cues.length));
  if (new URLSearchParams(location.search).has("hud")) $("#hud").classList.add("on");
})();
