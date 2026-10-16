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
const MINUS = (t) => t.replace(/^-/, "\u2212");   // typographic minus for negative numbers
const f2 = (x) => (x === null || x === undefined || Number.isNaN(+x)) ? "—" : MINUS((+x).toFixed(2));
const fN = (x) => (x === null || x === undefined) ? "—" : MINUS(Number.isInteger(+x) ? String(+x) : (+x).toFixed(1));
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
  const names = ["reveal", "dossier", "ablation", "longcon", "results", "forge", "stack", "cues", "longcon2", "boundary", "memory", "embed_exploratory", "longcon3", "c02_confirm", "confirm"];
  const got = await Promise.all(names.map((n) => getJSON(`${n}.json`)));
  names.forEach((n, i) => { B[n] = got[i]; HAVE[`${n}.json`] = got[i] !== null; });
  const deck = new URLSearchParams(location.search).get("deck");      // ?deck=pitch → curated 3-minute cue list
  if (deck && /^[a-z0-9_-]+$/i.test(deck)) { const alt = await getJSON(`cues_${deck}.json`); if (alt) B.cues = alt; }
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
  Object.assign(S, { view: "title", seq: null, frozen: false, playFrom: null, pace: 1, inspect: null, evidence: false,
    dossierGame: null, dossierAbout: null, surgery: [], flipped: false, caption: "", stackFlow: false });
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
      if (a.action === "audio" && a.file) a.missing = !(await exists(a.file));   // no mp3: the cue still runs, silently
      if (a.action === "play_to") {   // opponent voice: claim/answer mp3 for the event this cue lands on, if bundled
        const e = W.bySeq.get(a.seq);
        const f = e && (e.type === "claim" || e.type === "answer") ? `voice/${e.type}_${e.seq}.mp3` : null;
        a.voice = f && (await exists(f)) ? f : null;
      }
      if (a.action === "view" && a.name === "forge") need.add("forge.json");
      if (a.action === "view" && a.name === "split") need.add("twin.jsonl");
      if (a.action === "view" && a.name === "stack") need.add("stack.json");
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
    case "view": S.view = a.name; if (a.flow) S.stackFlow = true; break;
    case "play_to": S.seq = a.seq; S.frozen = false; S.view = "table"; S.playFrom = a.from ?? null; S.pace = a.pace ?? 1; break;
    case "freeze": S.seq = a.seq; S.frozen = true; S.view = "table"; S.playFrom = null; S.pace = 1; break;
    case "inspect": S.inspect = a.seq; S.view = "inspector"; break;
    case "evidence": S.evidence = a.hyp_id || true; S.view = "inspector"; break;
    case "dossier": S.dossierGame = a.game; if (a.about) S.dossierAbout = a.about; S.view = "dossier"; break;
    case "surgery": if (!S.surgery.includes(a.variant)) S.surgery.push(a.variant); S.view = "surgery"; break;
    case "flip": S.flipped = true; break;
    case "audio": if (live && !a.missing) AQ.push(a.file); break;
    case "caption": S.caption = a.text || ""; break;
    case "clear_caption": S.caption = ""; break;
    default: console.warn("unknown cue action", a);
  }
}
function applyCue(c, live) {
  if (c.skip) return;
  S.caption = c.keepCaption ? S.caption : "";
  AQ.length = 0;
  for (const a of c.actions) applyAction(a, live);
  if (live) {   // Profiler ask first, then the opponent's line; audio never advances cues
    for (const a of c.actions) if (a.action === "play_to" && a.voice) AQ.push(a.voice);
    playQueue();
  }
}
function next() {
  if (idx >= cues.length) return;
  applyCue(cues[idx], true); idx++;
  while (idx < cues.length && cues[idx - 1].skip) { applyCue(cues[idx], true); idx++; }  // never land on a skip
  render(true);
}
function prev() {
  let target = Math.max(0, idx - 1);
  while (target > 0 && cues[target - 1].skip) target--;
  goto(target);
}
function goto(n) {
  // Rebuilding state (←, R, ?cue=) jumps without transitions so the frame is final immediately.
  document.body.classList.add("instant");
  resetState(); stopAudio(); tableStop(); T.built = false;
  for (let i = 0; i < n; i++) applyCue(cues[i], false);
  idx = n; render(false);
  void document.body.offsetHeight;
  requestAnimationFrame(() => requestAnimationFrame(() => document.body.classList.remove("instant")));
}
const AQ = [];   // this cue's audio files, played in order
function playQueue() {
  const el = $("#audio"); const f = AQ.shift();
  if (!f) return;
  el.onended = () => playQueue(); el.onerror = () => playQueue();   // a file that fails to load is skipped silently
  el.src = `${BUNDLE}/${f}`; el.currentTime = 0; el.play().catch(() => {});
}
function playAudio(file) { AQ.length = 0; AQ.push(file); playQueue(); }
function stopAudio() { AQ.length = 0; const el = $("#audio"); el.onended = null; el.onerror = null; el.pause(); }

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
/* ================================================================ table
   The table is a persistent DOM synced to tableModel(seq). Within one cue it can step through the
   real events between the previous cue's seq and this one (sub-animations); the cue itself only
   advances on a keypress. Opponents' cards are face-down and carry no rank in the DOM until a
   real `reveal` event; the Profiler's own hand (p1) is face up. */
const SEAT_IDS = ["p2", "p3", "p1"];
const TURN_XY = { p1: [760, 906], p2: [235, 546], p3: [1685, 546] };
const STEP_MS = { game_start: 300, deal: 1500, claim: 1300, play_on: 700, ask: 1700, answer: 1700, challenge: 1100, reveal: 1700, life_lost: 1200, game_end: 900 };
const T = { built: false, shown: null, target: null, timers: [] };
let SC = 1;                                   // stage scale (screen px per stage px)

function tableModel(upto) {
  const st = { game: 0, round: 0, lives: {}, hand1: [], counts: {}, rank: null, pile: [], bubbles: {}, acts: {},
    reveal: null, lost: null, winner: undefined, active: null, lastBubble: -1 };
  let i = 0;
  for (; i < W.events.length; i++) {
    const e = W.events[i]; if (e.seq > upto) break;
    const p = e.payload || {};
    switch (e.type) {
      case "game_start": Object.assign(st, { game: e.game, round: 0, lives: { ...p.lives }, pile: [], bubbles: {}, acts: {}, reveal: null, lost: null, winner: undefined, rank: null, counts: {}, hand1: [] }); break;
      case "deal": Object.assign(st, { round: e.round, game: e.game, rank: p.table_rank, pile: [], bubbles: {}, acts: {}, reveal: null, lost: null });
        st.hand1 = [...(p.hands.p1 || [])];
        st.counts = Object.fromEntries(Object.entries(p.hands).map(([k, v]) => [k, v.length])); break;
      case "claim": {
        st.counts[e.actor] = Math.max(0, (st.counts[e.actor] || 0) - p.count);
        if (e.actor === "p1") for (const c of p.cards) { const j = st.hand1.indexOf(c); if (j >= 0) st.hand1.splice(j, 1); }
        st.pile.push({ actor: e.actor, count: p.count, rank: p.claimed_rank, cards: p.cards, seq: e.seq });
        st.bubbles[e.actor] = { text: p.utterance, kind: "claim", seq: e.seq }; st.lastBubble = e.seq;
        st.acts[e.actor] = null; break;
      }
      case "play_on": st.acts[e.actor] = { text: "plays on", seq: e.seq }; break;
      case "ask": { const a = e.actor || "p1"; st.bubbles[a] = { text: p.text, kind: "ask", seq: e.seq }; st.lastBubble = e.seq; break; }
      case "answer": if (e.actor) { st.bubbles[e.actor] = { text: p.text, kind: "answer", seq: e.seq }; st.lastBubble = e.seq; } break;
      case "challenge": st.acts[e.actor] = { text: "Challenge!", call: true, seq: e.seq }; break;
      case "reveal": st.reveal = { ...p, seq: e.seq }; break;
      case "life_lost": st.lives[p.seat] = p.lives_left; st.lost = { seat: p.seat, seq: e.seq }; break;
      case "game_end": st.winner = p.winner; break;
    }
  }
  // whose turn: the next seat to act in this round, read from the log
  for (let j = i; j < W.events.length; j++) {
    const e = W.events[j];
    if (e.game !== st.game || ["deal", "game_end", "game_start", "reveal", "life_lost"].includes(e.type)) break;
    if (["claim", "play_on", "challenge", "ask", "answer", "decision"].includes(e.type) && e.actor) { st.active = e.actor; break; }
  }
  return st;
}

function cardFront(rank) {
  if (rank === "J") return `<i class="tl">J</i><b>★</b><s>Joker</s><i class="br">J</i>`;
  return `<i class="tl">${esc(rank)}</i><b>${esc(rank)}</b><s>${esc(RANK1[rank] || "")}</s><i class="br">${esc(rank)}</i>`;
}
function cardEl(rank, { up = false, tableRank = null, sm = false, xs = false } = {}) {
  const bad = up && tableRank && rank !== tableRank && rank !== "J";
  const cls = ["pc", sm ? "sm" : "", xs ? "xs" : "", up ? "up" : "", bad ? "bad" : "", up && rank === "J" ? "joker" : ""].join(" ");
  return `<div class="${cls}" title="${up ? esc(rank) : "face down"}"><div class="in"><div class="face back"></div><div class="face front">${up ? cardFront(rank) : ""}</div></div></div>`;
}
function nodeOf(html) { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstChild; }
function setFront(pc, rank, tableRank) {
  pc.querySelector(".front").innerHTML = cardFront(rank);
  pc.classList.toggle("joker", rank === "J");
  pc.classList.toggle("bad", !!tableRank && rank !== tableRank && rank !== "J");
  pc.title = rank;
}
function isBluff(cards, tr) { return cards.some((c) => c !== tr && c !== "J"); }
function svg(w, h, inner, cls = "") { return `<svg class="${cls}" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" xmlns="${SVGNS}">${inner}</svg>`; }
function awaiting(title, body) { return `<div class="awaiting"><b>${esc(title)}</b>${body}</div>`; }
function seatName(seat, reveal = false) {
  const role = W.seats[seat]?.role;
  return esc(NAMES[seat] || seat) + (reveal && role && seat !== "p1" ? `<span class="role">${esc(role)}</span>` : "");
}
const claimText = (n, r) => `${n} ${esc((n === 1 ? RANK1 : RANKS)[r] || r)}`;

function buildTable() {
  const el = $("#v-table");
  el.innerHTML = `<div class="room-light"></div><div class="tbl">
      <div class="rail"></div><div class="felt"></div>
      <div class="gameplate"></div>
      <div class="rankplate hide"><div class="lbl">Table</div><div class="rk"></div></div>
      <div class="pile"></div><div class="piletag"></div>
      <div class="verdict"><span class="v"></span><span class="why"></span></div>
      ${SEAT_IDS.map((s) => `<div class="seat ${s}" data-seat="${s}"><div class="glow"></div><div class="hand"></div>
        <div class="plate"><span class="name">${seatName(s)}</span></div><div class="chips"></div>
        <div class="act"></div><div class="bubble"></div></div>`).join("")}
    </div><div id="turn">turn</div><div class="frost"></div><div class="freeze-stamp"></div>`;
  T.built = true; T.shown = null;
}
const rectC = (n) => { const r = n.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; };

function pilePos(g, j, G, seq) {
  const x = (g - (G - 1) / 2) * 40 + j * 18, y = ((seq * 13) % 9) - 4;
  const r = ((seq * 37) % 23) - 11 + j * 4;
  return `translate(${x}px, ${y}px) rotate(${r}deg)`;
}

function syncTable(m, ev, anim) {
  const el = $("#v-table");
  const isDeal = anim && ev?.type === "deal";
  el.querySelector(".gameplate").innerHTML = m.game ? `Game <b>${m.game}</b>${m.round ? `, round <b>${m.round}</b>` : ""}${m.winner !== undefined ? `. ${m.winner ? `${seatName(m.winner)} wins the game` : "Draw"}` : ""}` : "";
  const rp = el.querySelector(".rankplate");
  rp.querySelector(".rk").textContent = RANKS[m.rank] || "";
  rp.classList.toggle("hide", !m.rank);
  if (isDeal) { rp.classList.remove("pop"); void rp.offsetWidth; rp.classList.add("pop"); }

  // hands (remember where removed cards were, for the slide into the pile)
  const from = {};
  for (const s of SEAT_IDS) {
    const hand = el.querySelector(`.seat.${s} .hand`);
    const alive = (m.lives[s] ?? 0) > 0 || m.counts[s] > 0;
    el.querySelector(`.seat.${s}`).classList.toggle("out", m.game > 0 && !alive);
    if (isDeal || !anim) {
      hand.innerHTML = s === "p1" ? m.hand1.map((c) => cardEl(c, { up: true })).join("")
        : Array.from({ length: m.counts[s] || 0 }, () => cardEl("?", { sm: true })).join("");
      if (isDeal) [...hand.children].forEach((c, i) => { c.classList.add("deal"); c.style.setProperty("--i", i + SEAT_IDS.indexOf(s) * 5);
        c.style.setProperty("--dx", s === "p1" ? "0px" : s === "p2" ? "540px" : "-540px"); c.style.setProperty("--dy", s === "p1" ? "-330px" : "100px"); });
      continue;
    }
    const kids = [...hand.children];
    if (s === "p1") {
      const want = [...m.hand1]; const keep = new Set();
      for (const k of kids) { const r = k.title; const j = want.indexOf(r); if (j >= 0) { want.splice(j, 1); keep.add(k); } }
      const gone = kids.filter((k) => !keep.has(k));
      from[s] = gone.map(rectC); gone.forEach((k) => k.remove());
      for (const r of want) hand.appendChild(nodeOf(cardEl(r, { up: true })));
    } else {
      const n = m.counts[s] || 0;
      const gone = kids.slice(n); from[s] = gone.map(rectC); gone.forEach((k) => k.remove());
      for (let i = kids.length; i < n; i++) hand.appendChild(nodeOf(cardEl("?", { sm: true })));
    }
  }

  // pile
  const pile = el.querySelector(".pile");
  const G = m.pile.length;
  const have = new Map([...pile.children].map((c) => [c.dataset.k, c]));
  const want = new Set();
  m.pile.forEach((g, gi) => g.cards.forEach((c, j) => {
    const k = `${g.seq}:${j}`; want.add(k);
    let n = have.get(k); const tf = pilePos(gi, j, G, g.seq);
    const revealed = m.reveal && gi === G - 1;
    if (!n) {
      n = nodeOf(cardEl(g.actor === "p1" ? c : "?", { up: g.actor === "p1" && anim }));
      if (g.actor === "p1") setFront(n, c, null);
      n.dataset.k = k; pile.appendChild(n);
      n.style.transform = tf;
      const src = anim && ev?.type === "claim" && ev.seq === g.seq ? (from[g.actor] || [])[j] : null;
      if (src) {
        const [cx, cy] = rectC(n);
        n.style.transition = "none";
        n.style.transform = `translate(${(src[0] - cx) / SC}px, ${(src[1] - cy) / SC}px) ${tf}`;
        void n.offsetWidth;
        n.style.transition = "transform .75s cubic-bezier(.3,.7,.2,1)";
        requestAnimationFrame(() => { n.style.transform = tf; if (g.actor === "p1") n.classList.remove("up"); });
      } else if (g.actor === "p1") n.classList.remove("up");
    } else { n.style.transform = tf; }
    if (revealed) {
      const rc = m.reveal.cards?.[j] ?? c;
      if (!n.classList.contains("up")) { setFront(n, rc, m.rank); n.classList.add("up"); }
    }
  }));
  for (const [k, n] of have) if (!want.has(k)) n.remove();
  const last = m.pile[G - 1];
  const lastSaid = last && m.bubbles[last.actor]?.seq !== last.seq ? W.bySeq.get(last.seq)?.payload?.utterance : null;
  el.querySelector(".piletag").innerHTML = last ? `${seatName(last.actor)} claims <b>${claimText(last.count, last.rank)}</b>${lastSaid ? `<span class="said">“${esc(lastSaid)}”</span>` : ""}` : "";

  // verdict
  const vd = el.querySelector(".verdict");
  if (m.reveal) {
    const R = m.reveal;
    vd.className = `verdict on ${R.was_bluff ? "bluff" : "honest"}`;
    vd.querySelector(".v").textContent = R.was_bluff ? "Bluff" : "Honest";
    vd.querySelector(".why").innerHTML = `${seatName(R.loser)} loses a life`;
  } else vd.className = "verdict";

  // seats: lives, bubbles, actions, spotlight
  for (const s of SEAT_IDS) {
    const se = el.querySelector(`.seat.${s}`);
    se.classList.toggle("active", m.active === s);
    const lives = m.lives[s] ?? 0;
    const chips = se.querySelector(".chips");
    const leaving = anim && ev?.type === "life_lost" && ev.payload.seat === s;
    if (!chips.children.length || !anim || isDeal || ev?.type === "game_start") {
      chips.innerHTML = Array.from({ length: 3 }, (_, i) => `<span class="chip-life ${i < lives ? "" : "gone"}"></span>`).join("");
    } else if (leaving) {
      const c = chips.children[lives];
      if (c && !c.classList.contains("gone")) { c.classList.add("leaving"); setTimeout(() => { c.classList.remove("leaving"); c.classList.add("gone"); }, 800); }
    }
    const b = m.bubbles[s]; const bu = se.querySelector(".bubble");
    if (b) {
      if (bu.dataset.seq !== String(b.seq)) {
        const tgt = String(W.bySeq.get(b.seq)?.payload?.target || "");
        bu.innerHTML = esc(b.text) + (b.kind === "ask" ? `<span class="tag">The Profiler asks ${seatName(tgt)}${HAVE_VOICE(b.seq) ? " (voice: ElevenLabs)" : ""}</span>`
          : b.kind === "answer" ? `<span class="tag">answers the question</span>` : "");
        bu.dataset.seq = b.seq;
        if (anim) { bu.classList.remove("on"); void bu.offsetWidth; }
      }
      bu.classList.add("on"); bu.classList.toggle("ask", b.kind === "ask"); bu.classList.toggle("old", b.seq < m.lastBubble);
    } else { bu.classList.remove("on", "old"); bu.dataset.seq = ""; }
    const a = m.acts[s]; const ae = se.querySelector(".act");
    ae.textContent = a ? a.text : ""; ae.classList.toggle("on", !!a); ae.classList.toggle("call", !!a?.call);
  }
  const tb = $("#turn");
  if (m.active && TURN_XY[m.active]) { tb.style.transform = `translate(${TURN_XY[m.active][0]}px, ${TURN_XY[m.active][1]}px)`; tb.classList.add("on"); }
  else tb.classList.remove("on");
  where(m.game ? `Game ${m.game}, round ${m.round}${B.stack?.daytona?.picked_world_box && B.stack?.picked_decision?.world === W.wid ? `, played on <span class="spons">Daytona</span> ${esc(B.stack.daytona.picked_world_box)}` : ""}` : "");
}
const HAVE_VOICE = (seq) => !!HAVE[`voice/ask_${seq}.mp3`];

function tableStop() { T.timers.forEach(clearTimeout); T.timers = []; }
function renderTable(live) {
  const el = $("#v-table");
  if (!W.events.length) { T.built = false; el.innerHTML = awaiting("Awaiting data", "No <code>bundle/world.jsonl</code> yet. Run <code>blab pick-demo --run local</code> after a real local world exists."); return; }
  const target = S.seq ?? 0;
  const interrupted = T.timers.length > 0;
  tableStop();
  if (!T.built) buildTable();
  if (interrupted && T.target !== null) { document.body.classList.add("instant"); syncTable(tableModel(T.target), null, false); T.shown = T.target; void el.offsetWidth; document.body.classList.remove("instant"); }
  T.target = target;
  let start = T.shown;
  if (live && S.playFrom !== null && S.playFrom !== undefined && T.shown !== S.seq) {
    start = S.playFrom - 1;
    document.body.classList.add("instant"); syncTable(tableModel(start), null, false); void el.offsetWidth; document.body.classList.remove("instant");
  }
  const steps = (live && start !== null && target > start) ? W.events.filter((e) => e.seq > start && e.seq <= target && STEP_MS[e.type]) : null;
  if (!steps || steps.length > 80) { syncTable(tableModel(target), null, false); T.shown = target; }
  else {
    let t = 0; const pace = S.pace || 1;
    steps.forEach((e, i) => {
      T.timers.push(setTimeout(() => { syncTable(tableModel(e.seq), e, true); T.shown = e.seq; }, t));
      t += STEP_MS[e.type] * (e.type === "deal" ? 1 : pace);
    });
    T.timers.push(setTimeout(() => { syncTable(tableModel(target), null, true); T.shown = target; T.timers = []; }, Math.max(0, t - 200)));
  }
  el.classList.toggle("frozen", !!S.frozen);
  const m = tableModel(target);
  const claimant = m.pile[m.pile.length - 1]?.actor;
  for (const s of SEAT_IDS) el.querySelector(`.seat.${s}`).classList.toggle("held-seat", !!S.frozen && s === claimant);
  for (const s of SEAT_IDS) el.querySelector(`.seat.${s} .bubble`).classList.toggle("held", !!S.frozen && s === claimant);
  el.querySelector(".freeze-stamp").textContent = S.frozen ? `Time stopped at decision ${S.seq}. The Profiler is to act.` : "";
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
  let s = `<rect x="${x0}" y="${y}" width="${x1 - x0}" height="${bh}" fill="#07100b" stroke="var(--brass-lo)" stroke-width="1"/>`;
  if (pc !== null && pc !== undefined) s += `<rect class="g-cards" x="${x0}" y="${y}" width="${X(pc) - x0}" height="${bh}" fill="var(--brass)"/>`;
  if (pf !== null && pf !== undefined && pc !== null && pc !== undefined) {
    const a = Math.min(pc, pf), b = Math.max(pc, pf);
    s += `<rect class="g-mem" style="transform-origin:${pf >= pc ? "left" : "right"} center" x="${X(a)}" y="${y}" width="${Math.max(0, X(b) - X(a))}" height="${bh}" fill="var(--pearl)"/>`;
    s += `<rect class="g-final" x="${X(pf) - 3}" y="${y - 12}" width="6" height="${bh + 24}" fill="var(--ivory)"/>`;
  }
  s += `<line x1="${X(.5)}" x2="${X(.5)}" y1="${y - 22}" y2="${y + bh + 22}" stroke="var(--bone)" stroke-width="2" stroke-dasharray="6 6"/>`;
  if (label) {
    s += `<text x="${X(.5)}" y="${y + bh + 50}" fill="var(--bone-dim)" font-size="22" text-anchor="middle">challenge above 0.50</text>`;
    if (pc !== null && pc !== undefined) s += X(pc) < x0 + 240
      ? `<text class="g-lbl-cards" x="${x0}" y="${y - 20}" fill="var(--brass-hi)" font-size="28" font-weight="600" text-anchor="start">card math ${f2(pc)}</text>`
      : `<text class="g-lbl-cards" x="${X(pc)}" y="${y - 20}" fill="var(--brass-hi)" font-size="28" font-weight="600" text-anchor="end">card math ${f2(pc)}</text>`;
    if (pf !== null && pf !== undefined) s += `<text class="g-lbl-mem" x="${Math.min(X(pf) + 14, x1)}" y="${y - 20}" fill="var(--pearl)" font-size="28" font-weight="600" text-anchor="${X(pf) > x1 - 260 ? "end" : "start"}">with memory ${f2(pf)}</text>`;
    for (const t of [0, .25, .5, .75, 1]) if (pf === null || pf === undefined || Math.abs(X(t) - X(pf)) > 34) s += `<text x="${X(t)}" y="${y + bh + 22}" fill="var(--bone-faint)" font-size="18" text-anchor="middle">${t === .5 ? "" : t}</text>`;
  }
  return svg(w, h, s, "gauge");
}

function memLineHTML(m, evidenceOpen) {
  const qt = m.q_true || [], qf = m.q_false || [];
  const rows = (m.evidence_obs || []).map((id) => W.reveals.get(id) ? { id, ...W.reveals.get(id) } : { id });
  const shown = rows.slice(-3);
  const evid = shown.map((r) => {
    if (!r.reveal) return `<div class="row"><span>${esc(r.id.split(":").pop())}</span><span class="u">not in this world's log</span><span></span></div>`;
    const tr = (W.events.find((e) => e.type === "deal" && e.game === r.reveal.game && e.round === r.reveal.round) || {}).payload?.table_rank;
    const p = r.reveal.payload;
    return `<div class="row"><span>Game ${r.reveal.game} r${r.reveal.round}</span>
      <span class="u">${esc(r.claim?.payload.utterance || "")}</span>
      <span class="cs">${p.cards.map((c) => cardEl(c, { up: true, tableRank: tr, xs: true })).join("")}
      <span class="res ${p.was_bluff ? "bluffword" : "honestword"}">${p.was_bluff ? "bluff" : "honest"}</span></span></div>`;
  }).join("");
  return `<div class="line panel">
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
  return `<div class="claimcard panel"><div class="who">${seatName(who, S.flipped)} claims</div>
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
function renderInspector(live) {
  const el = $("#v-inspector");
  el.classList.toggle("anim", !!live && !S.evidence);
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
  const pd = B.stack?.picked_decision;
  if (pd && pd.seq === e.seq && W.wid === pd.world) el.querySelector(".insp > div").insertAdjacentHTML("beforeend",
    `<div class="attrib"><b>Arize AX</b> decision span, trace <code>${esc(pd.arize_trace_id || "—")}</code><br><b>Daytona</b> world box <code>${esc(pd.box || "—")}</code></div>`);
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
    ${base ? `Base bluff rate <b>${f2(base[0])}</b> (${f2(base[1])}–${f2(base[2])}).` : ""}${surrealNote(about)}</p>
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
function renderSurgery(live) {
  const el = $("#v-surgery"); const R = B.reveal;
  const animFlip = !!live && !!S.flipped && !el.classList.contains("flipped");   // the flip keypress: turn the cards on screen
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
      return `<div class="col panel absent ${on ? "on" : ""}"><div class="vt">${t}</div><div class="vs">${sub}</div><div class="p">not run</div>
        <div class="dp">${esc(v?.note || "no matching world in this run")}</div></div>`;
    }
    const r = v.result; const act = r.action;
    const right = wasBluff === null ? null : (act === "challenge") === !!wasBluff;
    return `<div class="col panel ${on ? "on" : ""}"><div class="vt">${t}</div><div class="vs">${sub}</div>
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
      <div class="cards">${cards}<span class="verdict2 ${showCards ? "on" : ""} ${wasBluff ? "bluffword" : "honestword"}">${wasBluff ? "bluff" : "honest"}</span></div></div>
    <div class="cols">${cols}</div>
    ${R.selection && R.selection.exact === false ? `<div class="selnote">Closest match, not exact: ${esc(R.selection.note || "")}</div>` : ""}`;
  where(`Decision #${R.seq}, world ${R.world}`);
  if (animFlip) {   // start face down, then turn: cards, verdict and right/wrong all follow the existing transitions
    const pcs = [...el.querySelectorAll(".surg-head .cards .pc.up")]; const v2 = el.querySelector(".verdict2");
    el.classList.add("instant"); el.classList.remove("flipped"); pcs.forEach((c) => c.classList.remove("up")); v2?.classList.remove("on");
    void el.offsetWidth; el.classList.remove("instant");
    requestAnimationFrame(() => requestAnimationFrame(() => { el.classList.add("flipped"); pcs.forEach((c) => c.classList.add("up")); v2?.classList.add("on"); }));
  }
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
  const w = 1776, h = 640, x0 = 90, x1 = w - 20, yT = 30, yB = h - 130;
  const pts = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(1)}${""} pts`;
  const bars = A.bars; const n = bars.length;
  const top = Math.min(1, Math.max(...bars.map((b) => b.hi ?? b.mean ?? 0)) + 0.1);
  const Y = (v) => yB - (yB - yT) * (v / top);
  const bw = Math.min(180, (x1 - x0) / n * 0.6);
  const step = top <= 0.3 ? 0.05 : top <= 0.6 ? 0.1 : 0.25;
  const ticks = []; for (let t = 0; t <= top + 1e-9; t += step) ticks.push(+t.toFixed(2));
  let s = axisY(x0, x1, Y, ticks, (t) => pct(t));
  bars.forEach((b, i) => {
    const cx = x0 + (x1 - x0) * (i + 0.5) / n;
    const col = b.variant === "full" ? "var(--s-memory)" : b.variant.startsWith("minus") ? "var(--s-rose)" : b.variant.startsWith("as_of") ? "var(--s-brass)" : "var(--s-violet)";
    if (b.mean !== null && b.mean !== undefined) {
      s += `<rect x="${cx - bw / 2}" y="${Y(b.mean)}" width="${bw}" height="${yB - Y(b.mean)}" fill="${col}" rx="4"/>`;
      if (b.lo !== null && b.hi !== null) s += `<line x1="${cx}" x2="${cx}" y1="${Y(b.lo)}" y2="${Y(b.hi)}" stroke="var(--bone)" stroke-width="3"/>
        <line x1="${cx - 16}" x2="${cx + 16}" y1="${Y(b.lo)}" y2="${Y(b.lo)}" stroke="var(--bone)" stroke-width="3"/>
        <line x1="${cx - 16}" x2="${cx + 16}" y1="${Y(b.hi)}" y2="${Y(b.hi)}" stroke="var(--bone)" stroke-width="3"/>`;
      s += `<text x="${cx}" y="${Y(b.hi ?? b.mean) - 14}" fill="var(--bone)" font-size="32" font-weight="700" text-anchor="middle">${pct(b.mean)}</text>`;
    }
    s += `<text x="${cx}" y="${yB + 36}" fill="var(--bone)" font-size="26" font-weight="600" text-anchor="middle">${esc(b.label || b.variant)}</text>
      <text x="${cx}" y="${yB + 64}" fill="var(--bone-faint)" font-size="19" text-anchor="middle">n = ${esc(b.n_parents ?? b.n)}${b.delta_vs_full && b.delta_vs_full.mean !== null && b.delta_vs_full.mean !== undefined ? "" : ""}</text>
      ${b.delta_vs_full && b.delta_vs_full.mean !== null && b.delta_vs_full.mean !== undefined ? `<text x="${cx}" y="${yB + 92}" fill="var(--bone)" font-size="21" font-weight="600" text-anchor="middle">Δ vs full ${pts(b.delta_vs_full.mean)}</text>${b.delta_vs_full.lo !== null && b.delta_vs_full.lo !== undefined ? `<text x="${cx}" y="${yB + 116}" fill="var(--bone-dim)" font-size="18" text-anchor="middle">CI ${pts(b.delta_vs_full.lo)} to ${pts(b.delta_vs_full.hi)}</text>` : ""}` : ""}`;
  });
  const fl = A.flips;
  const arm = [A.agent === "profiler" ? "Profiler" : A.agent, A.reflection === "claude" ? "with real Claude reflection" : A.reflection ? `with ${A.reflection} reflection` : "",
    A.respond_rule ? `${A.respond_rule} respond rule` : "", A.n_parents ? `${A.n_parents} parent worlds, branches paired with their parent` : ""].filter(Boolean).map(esc).join(", ");
  el.innerHTML = head + `<p class="sub">${arm ? `${arm}. ` : ""}Win rate in games ${esc(A.window || "11–20")}, memory frozen at game ${esc(A.at_game ?? 10)}, ${esc(A.table || "")} table. Bars: mean; whiskers: 95% bootstrap CI.</p>
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
  for (const g of G) if (g % 2 === 0 || (g === gmin && !G.includes(gmin + 1))) s += `<text x="${X(g)}" y="${yB + 34}" fill="var(--bone-faint)" font-size="20" text-anchor="middle">${g}</text>`;
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
  const COL = { counter: "var(--s-brass)", bayes: "var(--s-rose)", bayes_discount: "var(--s-violet)", profiler: "var(--s-memory)" };
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
    .concat(agents.map((a) => `<span><i style="background:${COL[a] || "var(--bone-dim)"}"></i>${esc(a)} estimate</span>`))
    .concat(det !== null && det !== undefined ? [`<span><b style="color:var(--ice);font-size:1.3em;line-height:0">◎</b> profiler detects the turn (median)</span>`] : []).join("");
  const ec = L.exploitation_cost || {}; const rtd = L.rtd || {};
  const stat = Object.keys(ec).concat(Object.keys(rtd)).filter((v, i, a) => a.indexOf(v) === i).map((a) => `
    <div class="kv" style="font-size:28px;padding:12px 0;border-top:1px solid var(--rule)"><b style="color:${COL[a] || "var(--bone)"}">${esc(a)}</b>
      <span style="display:block;font-size:22px;color:var(--bone-dim)">${ec[a] && ec[a].n ? `${esc(L.exploitation_label || "exploitation cost")}: ${fN(ec[a].mean)}${ec[a].lo !== null && ec[a].lo !== undefined ? ` (${fN(ec[a].lo)}–${fN(ec[a].hi)})` : ""}` : ""}${rtd[a] ? `<br>rounds to detection, median ${fN(rtd[a].median)} (detected in ${esc(rtd[a].detected ?? rtd[a].n)} of ${esc(rtd[a].worlds ?? "?")} worlds)` : ""}</span></div>`).join("");
  const twist = "";   // the "naive memory more exploitable" line is not shown: lc02 at scale does not support it
  el.innerHTML = head + `<p class="sub">${sl} plays honest, then turns. Every memory of it is true; the attack is assuming people don't change.</p>
    <div style="display:grid;grid-template-columns:${w}px 1fr;gap:48px"><div class="chart">${svg(w, h, s)}<div class="legend">${leg}</div></div>
    <div style="margin-top:26px">${stat}${twist}<div class="foot">n = ${esc(L.n_worlds ?? "?")} longcon worlds. ${esc(L.estimate_note || "")}</div></div></div>`;
  where(`Long con, ${esc(L.source || "")}`);
}

/* ================================================================ results */
const OMIT_BOW = "Word-count classifier omitted (in r01 it was given the engineered tell feature); full results incl. it: DEVPOST.md / results.json.";
function renderResults() {
  const el = $("#v-results"); const R = B.results;
  const head = `<h2 class="title">Results</h2>`;
  const grid = R?.show_grid;
  if (!grid?.rows?.length) { el.innerHTML = head + awaiting("Awaiting data", "No <code>bundle/results.json</code> yet (<code>blab analyze</code>)."); return; }
  const th = grid.columns.map((c) => `<th>${esc(c.label)}</th>`).join("");
  const omitted = grid.rows.some((r) => r.agent === "logreg_bow");
  const rows = grid.rows.filter((r) => r.agent !== "logreg_bow").map((r) => `<tr class="${r.agent === "profiler" || r.agent.startsWith("profiler+") ? "ours" : ""}"><td>${esc(r.agent)}</td>${grid.columns.map((c) => {
    const m = r.cells?.[c.key];
    if (!m || m.mean === null || m.mean === undefined) return `<td class="na">—</td>`;
    const v = c.fmt === "pct" ? pct(m.mean) : f2(m.mean);
    const ci = (m.lo !== null && m.lo !== undefined) ? (c.fmt === "pct" ? `${pct(m.lo)}–${pct(m.hi)}` : `${f2(m.lo)}–${f2(m.hi)}`) : "";
    return `<td>${v}<span class="ci">${ci}${m.n !== undefined ? ` n ${m.n}` : ""}</span></td>`;
  }).join("")}</tr>`).join("");
  el.innerHTML = head + `<p class="sub">${esc(R.n_worlds ?? "?")} worlds, run ${esc(R.run ?? "")}${grid.respond_rule ? `, ${esc(grid.respond_rule)} respond rule` : ""}. Means with 95% bootstrap CIs over seeds.</p>
    <table class="grid"><thead><tr><th>Agent</th>${th}</tr></thead><tbody>${rows}</tbody></table>${omitted ? `<div class="foot" style="margin-top:12px">${esc(OMIT_BOW)}</div>` : ""}`;
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
    const fmt = (v) => MINUS(r0.fmt === "pct" ? `${v >= 0 ? "+" : ""}${(v * scale).toFixed(1)} pts` : `${v >= 0 ? "+" : ""}${v.toFixed(3)}`);
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
        <text x="${Math.abs(X(d.mean) - X(0)) < 70 ? X(d.mean) + (d.mean >= 0 ? 14 : -14) : X(d.mean)}" y="${y - 22}" fill="var(--bone)" font-size="26" font-weight="700" text-anchor="${Math.abs(X(d.mean) - X(0)) < 70 ? (d.mean >= 0 ? "start" : "end") : "middle"}">${fmt(d.mean)}</text>`;
    });
    return `<div><h3 style="font-size:30px;margin:0 0 6px">${esc(r0.label)}</h3><div class="foot" style="margin:0 0 14px">Profiler − no-memory counter, same seeds; whiskers 95% CI</div>${svg(w, h, s)}</div>`;
  };
  const hl = D.headline?.holds ? `<div style="font-family:var(--voice);font-style:italic;font-size:52px;line-height:1.15;margin-top:30px;max-width:1500px">${esc(D.headline.text)}</div>` : "";
  el.innerHTML = head + `<p class="sub">Does memory sharpen its beliefs, and do sharper beliefs win more games? Paired against the no-memory counter on the same seeds, per respond rule.</p>
    <div class="dec-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:80px;margin-top:34px">${metrics.map(panel).join("")}</div>${hl}
    <div class="foot" style="margin-top:18px">${(D.texts || []).map(esc).join("<br>")}</div>`;
  where(`Run ${esc(R.run ?? "")}`);
}

/* ================================================================ forge */
function renderForge() {
  const el = $("#v-forge"); const F = B.forge;
  const head = `<h2 class="title">Strategy forge: memory compiled into code, run as untrusted</h2>`;
  if (!F) { el.innerHTML = head + awaiting("Not available", "No <code>bundle/forge.json</code>."); return; }
  const kv = (k, v) => `<div class="kv"><span>${esc(k)}</span>${v}</div>`;
  // forge.json nests the sandbox result under entry.forge_result; read whichever level has it.
  const E = F.entry || {}; const R = F.forge_result || E.forge_result || {}; const V = F.eval || E.eval || {};
  const pick = (...vs) => vs.find((v) => v !== null && v !== undefined);
  const sid = pick(F.sandbox_id, R.sandbox_id, E.sandbox_id);
  const wr = pick(F.win_rate, R.win_rate, V.win_rate); const ci = pick(F.ci, R.ci, V.ci);
  const base = pick(F.baseline_win_rate, R.baseline_win_rate);
  const games = pick(F.eval?.games, R.games, V.games), crashes = pick(R.crashes, V.crashes), timeouts = pick(R.timeouts, V.timeouts);
  const promoted = pick(F.promoted, R.promoted, E.promoted);
  const blocked = F.network_blocked_verified || E.network_blocked_verified;
  const gate = F.gate || E.gate;
  const rows = [kv("Static gate", gate?.ok ? '<span class="honestword">passed</span>' : `<span class="bluffword">rejected</span> ${esc(gate?.reason || "")}`)];
  if (sid) rows.push(kv("Sandbox", `<code class="sid">${esc(sid)}</code>${blocked ? ' <span class="chip evidence">network blocked</span>' : ' <span class="chip none">restricted network</span>'}`));
  if (wr !== undefined) rows.push(kv("Win rate, same seeds", `${pct(wr)}${ci ? ` <small>(${pct(ci[0])}–${pct(ci[1])})</small>` : ""}${base !== undefined ? ` vs Profiler ${pct(base)}` : ""}`));
  if (games !== undefined) rows.push(kv("Games, crashes, timeouts", [games, crashes, timeouts].map((v) => esc(v ?? "—")).join(", ")));
  if (promoted !== undefined) rows.push(kv("Promoted", promoted ? '<span class="honestword">yes</span>' : "no"));
  // How many forges were tried: the shown one is picked by creation order, not by result.
  if (F.tally) rows.push(kv("All forges tried", Object.entries(F.tally).map(([t, x]) => `${esc(x.promoted)} of ${esc(x.tried)} promoted <small>(${esc(t)})</small>`).join("<br>")));
  const ctl = (F.all_runs || []).filter((r) => r.control && r.table === F.table);
  if (ctl.length) rows.push(kv("Control: memory withheld", ctl.map((r) => pct(r.win_rate)).join(", ")));
  el.innerHTML = head + `<div class="forge"><pre>${esc(F.source || F.module_source || E.module_source || "")}</pre><div>${rows.join("")}</div></div>`;
  where(`Forge${pick(F.policy_id, E.policy_id, R.policy_id) ? `, ${esc(pick(F.policy_id, E.policy_id, R.policy_id))}` : ""}`);
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
    return `<div class="side panel"><h3>${esc(label)}: ${esc(pol)}</h3><div class="meta">World ${esc(evs[0]?.world)}, seed ${esc(start?.seed)}</div>
      <div class="strip-games">${ws.map((g) => `<div class="${g.winner === "p1" ? "win" : ""}">${g.game}</div>`).join("")}</div>
      <div class="total">${won}<span>of ${ws.length} games won</span></div></div>`;
  };
  el.innerHTML = head + `<p class="sub">Highlighted: games won by seat 1.</p><div class="split">${side(W.events, "World")}${side(B.twin, "Twin")}</div>`;
  where("Split screen");
}

/* ================================================================ how it's built (stack.json) */
const dash = (v) => (v === null || v === undefined || v === "") ? "—" : esc(String(v));
function surrealNote(about) {
  const ex = B.stack?.surrealdb?.supersession_example;
  if (!ex || ex.world !== W.wid || !String(ex.entity || "").endsWith(`.${about}`)) return "";
  return `<br><span class="attrib"><b>SurrealDB Agent Memory</b> keeps this belief's history: <code>${esc(ex.entity)}</code>, key <code>${esc(ex.key)}</code>, ${dash(ex.history_len_from_log)} versions.</span>`;
}
let stackTimers = [];
function renderStack(live) {
  const el = $("#v-stack"); const K = B.stack;
  stackTimers.forEach(clearTimeout); stackTimers = [];
  const head = `<h2 class="title">How it's built: the house and its partners</h2>`;
  if (!K) { el.innerHTML = head + awaiting("Awaiting data", "No <code>bundle/stack.json</code>."); return; }
  const d = K.daytona || {}, c = K.claude || {}, sd = K.surrealdb || {}, az = K.arize || {}, el11 = K.elevenlabs, pd = K.picked_decision || {};
  const R = B.reveal || {}; const dec = R.decision || {};
  const boxes = (d.boxes || []).map((b) => `${esc(b.id)}`).join(", ");
  const L = [
    ["eleven", "ElevenLabs", "the voice", "Speaks the Profiler's questions at the table.",
      el11 ? `voice <b>${dash(el11.voice_id)}</b>, <b>${dash(el11.mp3s)}</b> lines voiced` : "not voiced in this run"],
    ["claude", "Claude Opus 5.5", "on AWS Bedrock", "Reflects between games: reads the reveals and proposes beliefs to test. Also writes Strategy Forge code.",
      `<b>${dash(c.reflections)}</b> real reflections, <b>${dash(c.proposed_hypotheses)}</b> beliefs proposed`],
    ["surreal", "SurrealDB", "Agent Memory", "Holds each belief about each opponent, with its history: what replaced what, valid from game to game. Dossiers are read as of a game.",
      `<b>${dash(sd.supersessions_in_logs)}</b> supersessions; e.g. <b>${dash(sd.supersession_example?.entity)}</b>`],
    ["arize", "Arize AX", "traces and experiments", "A span for every decision and reflection, and an experiment per agent scoring its frozen decisions (Brier).",
      `project <b>${dash(az.project)}</b>, <b>${dash(az.decision_spans_in_picked_world)}</b> decision spans in this world`],
    ["daytona", "Daytona", "world boxes and forge sandbox", "Stateful boxes where many worlds play at once: events, checkpoints, live memory. AI-written strategy code runs in a separate network-blocked sandbox.",
      `<b>${dash(Object.values(d.worlds_per_box || {}).reduce((a, b) => a + b, 0) || null)}</b> worlds on boxes ${boxes || "—"}${d.worlds_total ? ` (of ${esc(d.worlds_total)} in the run)` : ""}${(() => { const f = d.forge_sandbox_id || B.forge?.sandbox_id || B.forge?.entry?.sandbox_id || B.forge?.entry?.forge_result?.sandbox_id; return f ? `; forge sandbox <b>${esc(String(f).slice(0, 8))}</b>` : ""; })()}`],
  ];
  const at = (k) => ({ claim: "daytona", decide: "daytona", trace: "arize", reveal: "daytona", reflect: "claude", store: "surreal", back: "surreal" })[k];
  const steps = [
    ["claim", `${seatName(dec.target || "p3")} claims in world <code>${dash(R.world)}</code> on Daytona <code>${dash(pd.box)}</code> (event ${dash(R.claim_seq)}).`],
    ["decide", `The Profiler reads memory version ${dash(dec.memory_version)} and decides: <b>${esc((dec.action || "—").replace("_", " "))}</b>, p ${f2(dec.p_final)} (event ${dash(R.seq)}).`],
    ["trace", `The decision lands in Arize AX as trace <code>${dash(pd.arize_trace_id)}</code>.`],
    ["reveal", `The cards are turned: ${R.reveal ? (R.reveal.was_bluff ? "bluff" : "honest") : "—"} (event ${dash(R.reveal_seq)}).`],
    ["reflect", `After the game, Claude reads the reveals and proposes beliefs (sample request <code>${dash(c.sample_reflect_trace)}</code>).`],
    ["store", `Beliefs that pass the evidence test are written to SurrealDB; the old version is superseded, not erased.`],
    ["back", `Next game, the Profiler reads that memory again. No retraining.`],
  ];
  const flow = S.stackFlow;
  el.innerHTML = head + `<p class="sub">Every number on this card comes from run <b>${dash(K.run)}</b>.</p>
    <div class="stack"><div class="layers">${L.map(([k, wm, sm, does, real]) => `<div class="layer panel" data-k="${k}">
      <div class="wm">${wm}<small>${sm}</small></div><div class="does">${does}</div><div class="real">${real}</div></div>`).join("")}</div>
      <div class="flow panel"><h3>One decision, through the house</h3>${steps.map(([k, t], i) => `<div class="step ${flow && !live ? "on" : ""}" data-k="${at(k)}"><span class="n">${i + 1}</span><span>${t}</span></div>`).join("")}</div></div>`;
  where(`Run ${dash(K.run)}`);
  if (flow && live) {
    const st = [...el.querySelectorAll(".step")];
    st.forEach((s, i) => stackTimers.push(setTimeout(() => {
      s.classList.add("on");
      el.querySelectorAll(".layer").forEach((l) => l.classList.toggle("hot", l.dataset.k === s.dataset.k));
    }, 300 + i * 1100)));
    stackTimers.push(setTimeout(() => el.querySelectorAll(".layer").forEach((l) => l.classList.remove("hot")), 300 + st.length * 1100 + 600));
  }
}


/* ================================================================ lc02, checkpoint branches, capability map, memory (polish) */
const ciPct = (m) => !m || m.mean === null || m.mean === undefined ? "—" : `${pct(m.mean)}${m.lo !== null && m.lo !== undefined ? `<span class="ci">${pct(m.lo)}–${pct(m.hi)}${m.n !== undefined ? ` · n ${m.n}` : ""}</span>` : ""}`;
const ciNum = (m, f = fN) => !m || m.mean === null || m.mean === undefined ? "—" : `${f(m.mean)}${m.lo !== null && m.lo !== undefined ? `<span class="ci">${f(m.lo)}–${f(m.hi)}${m.n !== undefined ? ` · n ${m.n}` : ""}</span>` : ""}`;
const kmTxt = (k) => !k ? "—" : (k.median === null || k.median === undefined ? `undefined<span class="ci">fewer than half detected · n ${esc(k.n ?? "?")}</span>`
  : `${fN(k.median)}<span class="ci">${k.lo !== null && k.lo !== undefined ? `${fN(k.lo)}–${k.hi === null || k.hi === undefined ? "∞" : fN(k.hi)}` : ""}${k.n !== undefined ? ` · ${esc(k.n_events ?? "?")}/${esc(k.n)} detected` : ""}</span>`);
const sgn = (v, f) => v === null || v === undefined ? "—" : MINUS(`${v > 0 ? "+" : ""}${f(v)}`);
const dPts = (v) => sgn(v, (x) => `${(x * 100).toFixed(1)} pts`);
const dCI = (d, f) => !d || d.mean === null || d.mean === undefined ? "—"
  : `${f(d.mean)}${d.lo !== null && d.lo !== undefined ? `<span class="ci">${f(d.lo)} to ${f(d.hi)} · n ${esc(d.n)}</span>` : `<span class="ci">n ${esc(d.n ?? "?")}, no CI</span>`}`;

function renderLongcon2() {
  const el = $("#v-longcon2"); const L = B.longcon2;
  const head = `<h2 class="title">The long con at scale</h2>`;
  if (!L?.arms) { el.innerHTML = head + awaiting("Awaiting data", "No <code>bundle/longcon2.json</code> yet."); return; }
  const order = ["counter", "bayes", "bayes_discount", "profiler", "profiler+claude", "profiler_blind"];
  const rows = order.map((l) => [l, L.arms[`${l}|threshold`]]).filter(([, a]) => a).map(([l, a]) => {
    const lab = l === "profiler" ? "profiler <small>stub reflection</small>" : l === "profiler+claude" ? "profiler <small>Claude reflection</small>" : esc(l);
    return `<tr class="${l.startsWith("profiler") && l !== "profiler_blind" ? "ours" : ""}"><td>${lab}</td><td>${ciPct(a.detection_rate)}</td><td>${kmTxt(a.rtd_km_median)}</td>
      <td>${ciNum(a.exploitation)}</td><td>${ciPct(a.false_alarm_belief)}</td></tr>`;
  }).join("");
  el.innerHTML = head + `<p class="sub">Pre-registered run ${esc(L.run || "")}: ${esc(L.n_worlds ?? "?")} worlds, seeds ${esc(L.seeds || "")}, threshold respond rule. A sleeper plays honest, then turns at G₀. Means with 95% bootstrap CIs over seeds.</p>
    <table class="grid lc2"><thead><tr><th>Agent</th><th>Detects the turn</th><th>Rounds to detection<br><small>Kaplan–Meier median</small></th><th>Sleeper bluffs let through<br><small>games ≥ G₀, per world</small></th><th>False alarm<br><small>belief change, honest table</small></th></tr></thead><tbody>${rows}</tbody></table>
    <div class="foot" style="margin-top:18px">Detection is near 100% because the pre-registered rule fires on two consecutive p_final &gt; 0.5. Claude-reflection Profiler (n = ${esc(L.arms["profiler+claude|threshold"]?.n_longcon ?? "?")}) is its own row, never pooled with stub reflection.${L.arms["logreg_bow|threshold"] ? ` ${esc(OMIT_BOW.replace("results.json", "longcon2.json"))}` : ""}</div>`;
  where(`Long con at scale, ${esc(L.run || "")}`);
}

function renderBranches() {
  const el = $("#v-branches"); const C = B.longcon2?.checkpoint_branches;
  const head = `<h2 class="title">Checkpoint branches: change one memory, replay the rest</h2>`;
  if (!C?.interventions) { el.innerHTML = head + awaiting("Awaiting data", "No checkpoint branches in <code>bundle/longcon2.json</code> yet."); return; }
  const I = C.interventions; const base = I.none?.exploitation?.mean;
  const lab = { none: "untouched", erase: "erase memory", false_memory: "false memory", drift_off: "drift detector off" };
  const rows = ["none", "erase", "false_memory", "drift_off"].filter((k) => I[k]).map((k) => {
    const x = I[k]; const d = k === "none" || base === undefined || x.exploitation?.mean === undefined ? null : x.exploitation.mean - base;
    return `<tr class="${k === "none" ? "ours" : ""}"><td>${lab[k]}</td><td>${ciNum(x.exploitation)}</td><td class="dl">${d === null ? "—" : sgn(d, (v) => v.toFixed(1))}</td><td>${kmTxt(x.rtd_km_median)}</td><td>${ciNum(x.lives_lost_after_g0)}</td></tr>`;
  }).join("");
  el.innerHTML = head + `<p class="sub">Each branch restores the Profiler's checkpoint from game G₀−1 and replays the same seeds and cards with one change to memory. Parent arm: ${esc(C.parent_arm || "")}.</p>
    <table class="grid lc2"><thead><tr><th>Checkpoint branch</th><th>Sleeper bluffs let through<br><small>games ≥ G₀, per world</small></th><th>Difference<br><small>vs untouched, in means</small></th><th>Rounds to detection<br><small>Kaplan–Meier median</small></th><th>Lives lost after G₀</th></tr></thead><tbody>${rows}</tbody></table>
    <div class="foot" style="margin-top:18px">Checkpoint branches: restored from game G₀−1, same seeds and cards, learning on, stub reflection. 95% bootstrap CIs over seeds; the difference column compares means.</div>`;
  where("Checkpoint branches");
}

function renderBoundary() {
  const el = $("#v-boundary"); const Bd = B.boundary; const M = Bd?.capability_map;
  const head = `<h2 class="title">Where memory helps: the capability map</h2>`;
  if (!M?.rows) { el.innerHTML = head + `<p class="sub">Pre-registered experiment b01: raw observations for every agent, fresh seeds 4000–4023.</p>` + awaiting("Awaiting data", "No <code>bundle/boundary.json</code> (capability_map) yet."); return; }
  const fx = (metric) => metric === "win_rate" ? dPts : metric === "revealed_to_discovery" ? (v) => sgn(v, (x) => x.toFixed(1)) : (v) => sgn(v, (x) => x.toFixed(3));
  const ok = (h, st) => st === "not run" ? '<span class="na">not run</span>' : h === true ? '<span class="honestword">✓ holds</span>' : h === false ? '<span class="bluffword">✗ did not hold</span>' : '<span class="na">no CI</span>';
  const MET = { win_rate: "win rate", brier_first10: "Brier, first 10 reveals", brier_after_flip10: "Brier, 10 after the flip", revealed_to_discovery: "reveals to discovery", brier: "Brier" };
  const bow = (x) => x?.a === "logreg_bow" || x?.b === "logreg_bow";   // word-count classifier rows are omitted (footnote says where they are)
  const omitted = M.rows.some(bow) || Object.values(Bd.hypotheses || {}).flat().some(bow);
  const rows = M.rows.filter((r) => !bow(r)).map((r) => { const f = fx(r.metric); const ci = Array.isArray(r.ci) ? r.ci : [];
    return `<tr class="${r.status === "not run" ? "notrun" : ""}"><td>${esc(r.cue || "")}<small>${esc(r.hypothesis || "")}${r.a ? ` · ${esc(r.a)} vs ${esc(r.b)}` : ""}</small></td>
      <td>${esc(r.winner ?? "—")}</td>
      <td>${r.delta === null || r.delta === undefined ? "—" : `${f(r.delta)}<span class="ci">${ci[0] != null ? `${f(ci[0])} to ${f(ci[1])}` : "no CI"}${r.n !== undefined ? ` · n ${esc(r.n)}` : ""}</span>`}</td>
      <td>${esc(MET[r.metric] || r.metric || "—")}</td><td>${ok(r.holds, r.status)}</td></tr>`; }).join("");
  const H = Bd.hypotheses || {};
  const hyp = Object.keys(H).sort().flatMap((k) => (Array.isArray(H[k]) ? H[k] : [H[k]])).filter((h) => !bow(h)).map((h) => { const f = fx(h.metric);
    return `<div class="hyp"><b>${esc(h.name || "")}</b>${esc(h.text || "")}<span>${ok(h.holds, h.status)}${h.mean != null ? ` <span class="ci">Δ ${f(h.mean)}${h.lo != null ? `, CI ${f(h.lo)} to ${f(h.hi)}` : ""}${h.n !== undefined ? `, n ${esc(h.n)}` : ""}</span>` : ""}</span></div>`; }).join("");
  el.innerHTML = head + `<p class="sub">Pre-registered (${esc(Bd.prereg || "PREREG.md")}), run ${esc(Bd.run || "")}, ${esc(Bd.info || "raw")} observations, paired 95% CIs. Holds only if the CI excludes 0 in the predicted direction.</p>
    <table class="grid lc2 cmap"><thead><tr><th>Cue type</th><th>Winner</th><th>Δ (95% CI)</th><th>Metric</th><th>Pre-registered claim</th></tr></thead><tbody>${rows}</tbody></table>
    ${hyp ? `<div class="hyps">${hyp}</div>` : ""}${(() => { const n = String(M.note || ""); const k = n.indexOf("Deviation:");
      return k < 0 ? `<div class="foot" style="margin-top:8px">${esc(n)}</div>`
        : `<div class="deviation"><b>Deviation from the pre-registration.</b> ${esc((t => t.charAt(0).toUpperCase() + t.slice(1))(n.slice(k + 10).trim()))}</div><div class="foot" style="margin-top:6px">${esc(n.slice(0, k).trim().replace(" or logreg_bow − profiler (r01 win rate)", ""))}</div>`; })()}${omitted ? `<div class="foot" style="margin-top:6px">${esc(OMIT_BOW)}</div>` : ""}`;
  where(`Capability map, ${esc(Bd.run || "")}`);
}

function renderMemory() {
  const el = $("#v-memory"); const M = B.memory;
  const head = `<h2 class="title">Does memory help?</h2>`;
  if (!M?.rules) { el.innerHTML = head + awaiting("Awaiting data", "No <code>bundle/memory.json</code> yet."); return; }
  const rule = (M.rules_order || Object.keys(M.rules))[0] || "threshold"; const R = M.rules[rule] || {};
  const hasD = (t) => Object.values(R[t]?.delta_vs_baseline || {}).some((d) => d?.win_rate?.mean != null || d?.brier?.mean != null);
  const tables = ["all", ...Object.keys(R).filter((t) => t !== "all")].filter((t) => R[t] && hasD(t));   // tables with no paired seeds are left out (listed in the foot)
  const allArms = (R.all || R[tables[0]])?.arms?.filter((a) => a !== M.baseline) || [];
  // logreg_bow is left out of this headline table (word counts over a tiny fixed sentence bank); logreg_embed is the named classifier baseline.
  const arms = allArms.filter((a) => a !== "logreg_bow" && a !== "logreg_embed");
  const clsArms = allArms.filter((a) => a === "logreg_embed");
  const brier = (v) => sgn(v, (x) => x.toFixed(3));
  const cell = (t, a) => { const d = R[t]?.delta_vs_baseline?.[a]; if (!d || (d.win_rate?.mean == null && d.brier?.mean == null)) return `<td class="na">—</td>`;
    const one = (m, f) => !m || m.mean == null ? "—" : `${f(m.mean)} <span class="ci">${m.lo != null ? `${f(m.lo)} to ${f(m.hi)}` : "no CI"}</span>`;
    return `<td><div>${one(d.win_rate, dPts)}</div><div class="br">${one(d.brier, brier)}</div></td>`; };
  const rowOf = (a, lab) => `<tr class="${a.startsWith("llm_") || a === "profiler_plus" ? "ours" : ""}"><td>${lab || esc(a)}</td>${tables.map((t) => cell(t, a)).join("")}</tr>`;
  const rows = arms.map((a) => rowOf(a)).join("") + (clsArms.length ? `<tr class="divider"><td colspan="${tables.length + 1}">classifier baselines</td></tr>`
    + clsArms.map((a) => rowOf(a, `${esc(a)} <span class="ci">sentence embeddings</span>`)).join("") : "");
  const X = B.embed_exploratory?.deltas?.["profiler_plus-logreg_embed"]?.pooled_by_seed?.win;
  const expl = X?.mean == null ? "" : `<div class="explo"><span class="tag">exploratory</span> profiler_plus − logreg_embed, paired by seed: win rate ${dPts(X.mean)} <span class="ci">${dPts(X.lo)} to ${dPts(X.hi)} · n ${esc(X.n)}</span>. Baseline chosen after mem01 ran, and the weaker of our two classifiers on these tables. Confirmatory rerun: c02.</div>`;
  const setting = M.setting === "raw" ? "raw observations" : M.setting === "engineered" ? "engineered-feature setting" : esc(M.setting || "");
  el.innerHTML = head + `<div class="memhead"><span class="setting">${setting}</span><span class="hl">${esc(M.headline || "")}</span></div>
    <p class="sub">Paired Δ vs the no-memory arm <b>${esc(M.baseline || "")}</b> on the same seeds, ${esc(rule)} rule, run ${esc(M.run || "")}. Win rate higher is better; Brier lower is better. 95% bootstrap CI where n allows.</p>
    <table class="grid lc2 mem"><thead><tr><th>Arm</th>${tables.map((t) => `<th>${t === "all" ? "all tables" : esc(t)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>${expl}
    <div class="foot" style="margin-top:14px">Each cell: win-rate Δ (top), Brier Δ (below), 95% CI. Paired seeds per cell: ${tables.map((t) => `${t === "all" ? "all" : esc(t)} n ${esc(Math.max(0, ...Object.values(R[t].delta_vs_baseline || {}).map((d) => d?.win_rate?.n || 0)))}`).join(", ")}.${Object.keys(R).filter((t) => t !== "all" && !hasD(t)).length ? ` No paired seeds yet: ${Object.keys(R).filter((t) => t !== "all" && !hasD(t)).map(esc).join(", ")}.` : ""}${allArms.includes("logreg_bow") ? " Word-count classifier omitted from this table (in r01 it was given the engineered tell feature); full results incl. it: DEVPOST.md / memory.json." : ""}</div>`;
  where(`Does memory help?, ${esc(M.run || "")}`);
}

/* ================================================================ title / close */
function provenance() {
  const P = B.reveal?.provenance || {};
  return { run: P.run || W.run, world: P.world || W.wid, seed: P.seed ?? W.start?.seed, table: P.table, rule: P.rule };
}
function renderTitle() {
  const P = provenance();
  $("#v-title").innerHTML = `<div class="crest">a card room for agents that remember</div><div class="big">Bluff Lab</div>
    <div class="one">How AI agents learn through memory without retraining, and whether that memory makes them better or more exploitable.</div>
    <div class="suits">${["K", "Q", "A", "J"].map((r) => cardEl(r, { up: true })).join("")}</div>
    <div class="prov">${W.events.length ? `One recorded world, picked by rule, not by hand: run <b>${esc(P.run)}</b>, world <b>${esc(P.world)}</b>, seed <b>${esc(P.seed)}</b>${P.table ? `, ${esc(P.table)} table` : ""}.` : "Awaiting data: no bundle yet."}</div>`;
  where("");
}
function renderClose() {
  $("#v-close").innerHTML = `<div class="line">The question isn’t how much an agent remembers. It’s which memories it should trust, and when to stop.</div>
    <div class="line2">Every decision recorded and replayable, every belief tied to the reveals behind it.</div>`;
  where("");
}

function renderLongcon3() {
  const el = $("#v-longcon3"); const L = B.longcon3;
  const head = `<h2 class="title">The long con, full Profiler: reflection on vs off</h2>`;
  if (!L?.arms) { el.innerHTML = head + awaiting("Awaiting data", "No <code>bundle/longcon3.json</code> yet."); return; }
  const lab = { counter: "no memory <small>counter</small>", bayes_discount: "discounted Bayes", logreg_embed: "embedding classifier",
    profiler_plus_noreflect: "profiler_plus <small>Claude reflection OFF</small>", profiler_plus: "profiler_plus <small>Claude reflection ON</small>" };
  const rows = ["counter", "bayes_discount", "logreg_embed", "profiler_plus_noreflect", "profiler_plus"].filter((k) => L.arms[k]).map((k) => {
    const a = L.arms[k];
    return `<tr class="${k.startsWith("profiler_plus") ? "ours" : ""}"><td>${lab[k]}</td><td>${ciNum(a.exploitation)}</td><td>${kmTxt(a.rtd_km_median)}</td>
      <td>${ciPct(a.false_alarm_belief)}</td><td>${ciPct(a.win_rate_longcon)}</td></tr>`;
  }).join("");
  const H = L.hypotheses || {}; const h32 = H.H32 || {}; const h31 = H.H31 || {}; const h33 = H.H33 || {};
  const verdict = (h) => h.holds ? "held" : h.opposite_direction ? "did not hold: opposite direction" : h.holds === false ? "did not hold" : "report only";
  el.innerHTML = head + `<p class="sub">Pre-registered run ${esc(L.run || "lc03")}: ${esc(L.n_worlds ?? "?")} worlds, ${Array.isArray(L.seeds) ? `${L.seeds.length} fresh seeds` : `fresh seeds ${esc(L.seeds || "?")} (n = 12)`}, raw observations, threshold respond rule. A sleeper plays honest, then turns. Means with 95% bootstrap CIs over seeds.</p>
    <table class="grid lc2"><thead><tr><th>Agent</th><th>Sleeper bluffs let through<br><small>after the turn, per world</small></th><th>Rounds to detection<br><small>Kaplan–Meier median</small></th><th>False alarm<br><small>belief change, honest table</small></th><th>Win rate<br><small>long con table</small></th></tr></thead><tbody>${rows}</tbody></table>
    <div class="foot" style="margin-top:16px;font-size:20px;line-height:1.45;color:var(--ivory)"><b>Takeaway:</b> with Claude's reflection switched on, the same agent let through
      <b>${h32.mean !== undefined ? sgn(h32.mean, (v) => v.toFixed(1)) : "?"}</b> more of the betrayer's bluffs
      (95% CI ${h32.lo !== undefined ? `${sgn(h32.lo, (v) => v.toFixed(1))} to ${sgn(h32.hi, (v) => v.toFixed(1))}` : "?"}, ${esc(h32.n ?? "?")} paired seeds). The memory that helps against consistent players is what a long con exploits.</div>
    <div class="foot" style="margin-top:10px">Pre-registered: H31 (beats discounted Bayes) ${esc(verdict(h31))}; H32 (reflection on lets through fewer) ${esc(verdict(h32))}; H33 false alarms ${h33.mean !== undefined ? pct(h33.mean) : "?"} (report only, mostly from the drift detector). Mechanism (our reading, not measured): hypotheses written during the honest phase encode earned trust.</div>`;
  where(`Long con, full Profiler, ${esc(L.run || "lc03")}`);
}

function renderTwoSided() {
  const el = $("#v-twosided"); const C = B.c02_confirm; const L = B.longcon3;
  const head = `<h2 class="title">Better and more exploitable, at once</h2>`;
  const on = C?.deltas?.["profiler_plus-profiler_plus_noreflect"]?.pooled_by_seed?.win; const h32 = L?.hypotheses?.H32;
  if (!on || !h32) { el.innerHTML = head + awaiting("Awaiting data", "Needs <code>bundle/c02_confirm.json</code> and <code>bundle/longcon3.json</code>."); return; }
  const h13 = (B.confirm?.hypotheses || []).find((h) => h.name === "H13" && h.holds);
  const a = L.arms || {}; const km = (k) => a[k]?.rtd_km_median?.median; const wr = (k) => a[k]?.win_rate_longcon?.mean;
  const card = (kicker, big, unit, ci, note, tone) => `<div style="flex:1;border:1px solid ${tone};border-radius:6px;padding:34px 38px;background:rgba(10,20,15,.6)">
      <div style="font-family:var(--display);font-style:italic;font-size:30px;color:var(--ivory-dim)">${kicker}</div>
      <div style="font-family:var(--display);font-weight:700;font-size:150px;line-height:1.05;color:${tone}">${big}</div>
      <div style="font-size:26px;color:var(--ivory)">${unit}</div>
      <div style="font-size:20px;color:var(--ivory-dim);margin-top:8px">${ci}</div>
      <div style="font-size:19px;color:var(--ivory-dim);margin-top:14px;line-height:1.4">${note}</div></div>`;
  const f1 = (v) => (v * 100).toFixed(1);
  el.innerHTML = head + `<p class="sub">Fresh seeds, raw observations, threshold respond rule. 95% bootstrap CIs over seeds.</p>
    <div style="display:flex;gap:40px;margin-top:22px">
      ${h13 ? card("Overall, on fresh seeds (c01)", `+${f1(h13.mean)}`, "points of win rate vs a word-count classifier",
        `95% CI +${f1(h13.lo)} to +${f1(h13.hi)} · ${esc(h13.n)} fresh seeds · pre-registered`, `Raw text only, four opponent types. Claude's reflection on vs off: +${f1(on.mean)} (CI +${f1(on.lo)} to +${f1(on.hi)}, c02, fresh seeds; not pre-registered there), +9.5 earlier (mem01).`, "#8fd18f")
      : card("Across four opponent types (c02)", `+${f1(on.mean)}`, "points of win rate with reflection on",
        `95% CI +${f1(on.lo)} to +${f1(on.hi)} · ${esc(on.n)} fresh seeds`, "Replicates the earlier +9.5 (mem01). Not one of c02's pre-registered hypotheses; labelled as such.", "#8fd18f")}
      ${card("Against a long con (lc03)", `+${h32.mean.toFixed(1)}`, "more of the betrayer's bluffs let through",
        `95% CI +${h32.lo.toFixed(1)} to +${h32.hi.toFixed(1)} · ${esc(h32.n)} fresh seeds · pre-registered`, `Noticed the turn later (${esc(km("profiler_plus") ?? "?")} vs ${esc(km("profiler_plus_noreflect") ?? "?")} rounds). Win rate held (${wr("profiler_plus") !== undefined ? pct(wr("profiler_plus")) : "?"} vs ${wr("profiler_plus_noreflect") !== undefined ? pct(wr("profiler_plus_noreflect")) : "?"}).`, "#e2735f")}
    </div>
    <div class="foot" style="margin-top:22px;font-size:22px;line-height:1.45;color:var(--ivory)"><b>Takeaway:</b> Claude's hypotheses are earned trust. They win more games, and they're exactly what a long con exploits.
      <span style="color:var(--ivory-dim);font-size:18px">Mechanism is our reading, not measured.</span></div>`;
  where("Fresh-seed results: c01, c02, lc03");
}

/* ================================================================ render */
function where(t) { $("#strip-where").innerHTML = t; }
const RENDER = { title: renderTitle, table: renderTable, inspector: renderInspector, dossier: renderDossier, surgery: renderSurgery,
  ablation: renderAblation, trust: renderTrust, split: renderSplit, results: renderResults, decisions: renderDecisions, forge: renderForge, stack: renderStack, close: renderClose,
  longcon2: renderLongcon2, branches: renderBranches, boundary: renderBoundary, memory: renderMemory, longcon3: renderLongcon3, twosided: renderTwoSided };
const SCENE = { title: "", table: "The table", inspector: "Mind inspector", dossier: "Dossier", surgery: "Memory surgery",
  ablation: "Ablation", trust: "Long con", split: "Split screen", results: "Results", decisions: "Beliefs and decisions", forge: "Strategy forge", stack: "How it's built", close: "",
  longcon2: "Long con at scale", branches: "Checkpoint branches", boundary: "Capability map", memory: "Does memory help?", longcon3: "Long con, full Profiler", twosided: "One memory, two opponents" };
let lastView = null;
function render(live = false) {
  const prevView = lastView; lastView = S.view;
  const turning = live && prevView === "table" && S.view === "inspector";
  for (const el of document.querySelectorAll(".view")) {
    el.classList.remove("turn-out", "turn-in");
    el.classList.toggle("on", el.id === `v-${S.view}`);
  }
  if (S.view !== "table") tableStop();
  (RENDER[S.view] || renderTitle)(live);
  if (turning) {        // the table tilts away around the agent's seat; its mind turns in like a card
    const tv = $("#v-table"), iv = $("#v-inspector");
    void tv.offsetWidth; tv.classList.add("turn-out"); iv.classList.add("turn-in");
    const done = () => { tv.classList.remove("turn-out"); iv.classList.remove("turn-in"); };
    iv.addEventListener("animationend", (e) => { if (e.target === iv) done(); }, { once: false });
    setTimeout(done, 1600);
  }
  $("#strip-scene").textContent = SCENE[S.view] ?? "";
  const cap = $("#caption"); cap.textContent = S.caption || ""; cap.classList.toggle("on", !!S.caption);
  renderHUD();
}
function renderHUD() {
  const h = $("#hud"); const cur = cues[idx - 1]; const nxt = cues[idx];
  const files = ["world.jsonl", "reveal.json", "dossier.json", "ablation.json", "longcon.json", "results.json", "cues.json", "twin.jsonl", "forge.json", "stack.json"]
    .map((f) => `${HAVE[f] ? "✓" : "✗"} ${f}`).join("  ");
  h.innerHTML = `<div class="now">${idx}/${cues.length} ${esc(cur?.name || "start")}</div>
    <div class="say">${esc(cur?.say || "")}</div>
    ${nxt ? `<div>next: <b>${esc(nxt.name)}</b>${nxt.skip ? ` <span class="warn">${esc(nxt.skip)}</span>` : ""}</div>` : "<div>end of cues</div>"}
    ${B.cues ? "" : '<div class="warn">no cues.json: using built-in FLOW order</div>'}
    <ol>${cues.map((c, i) => `<li class="${i === idx - 1 ? "cur" : ""} ${c.skip ? "skip" : ""}">${esc(c.name)}${c.skip ? ` (${esc(c.skip)})` : ""}</li>`).join("")}</ol>
    <div class="files">bundle: ${esc(BUNDLE)}/<br>${files}</div>`;
}

function fit() {
  const s = Math.min(innerWidth / 1920, innerHeight / 1080); SC = s;
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
// Touch navigation (phones/tablets only; mouse clicks are left alone so a presenter's stray click never advances):
// tap the right half → next, left half → back; swipe left → next, swipe right → back.
let touchStart = null;
addEventListener("pointerdown", (e) => {
  if (e.pointerType === "mouse") return;
  touchStart = { x: e.clientX, y: e.clientY };
}, { passive: true });
addEventListener("pointerup", (e) => {
  if (e.pointerType === "mouse" || !touchStart) return;
  const dx = e.clientX - touchStart.x, dy = e.clientY - touchStart.y; touchStart = null;
  if (e.target.closest && e.target.closest("a, button, input, select, textarea, #hud")) return;
  document.body.classList.add("touched");
  if (Math.abs(dx) > 40 && Math.abs(dx) > Math.abs(dy)) { if (dx < 0) next(); else prev(); return; }   // swipe
  if (Math.abs(dx) > 12 || Math.abs(dy) > 12) return;                                                 // a drag, not a tap
  if (e.clientX < innerWidth / 2) prev(); else next();                                                // tap
});
addEventListener("pointercancel", () => { touchStart = null; });

(async function main() {
  fit(); resetState();
  await load();
  const n = parseInt(new URLSearchParams(location.search).get("cue") || "0", 10);   // ?cue=N jumps (rehearsal)
  goto(Math.min(n, cues.length));
  if (new URLSearchParams(location.search).has("hud")) $("#hud").classList.add("on");
})();
