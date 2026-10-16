# Bluff Lab — implementation spec (handoff document)

> **For the implementing agent.** This file is self-contained: build from it.
> - `NOTES.md` holds the pitch and framing. You don't need it to build.
> - **Hard deadline: Devpost submission at 15:30 on 2026-10-16.** Feature freeze at 14:40.
> - Work through the milestones in **§16 in order**. Each one has an acceptance check, and the system must stay demoable after every milestone.
> - Legend: ✅ = verified in vendor docs today · ⚠️ = unverified, check it in the spike (§15) before relying on it.

---

## 0. Hard rules (do not violate)

1. **Never fabricate results.**
   - Every number shown in the UI, video or Devpost comes from a real run's event logs or `results.json`.
   - The presentation layer may stage pacing and cut between scenes, but it must never invent data.
   - If a feature isn't real by the freeze, cut it or label it "illustrative."
2. **Agents never see hidden information.** Policies receive only `observation(seat)` (§3.4), never engine objects. Hidden cards appear only in the god-view event log, for evaluation and replay.
3. **Never `eval`/`exec` LLM output in our process.** Hypothesis predicates use the JSON DSL (§5.2). AI-written strategy code runs only inside a Daytona forge sandbox, in a child process (§7.4).
4. **Secrets live in `settings.env` and are never committed.**
   - **First action:** create a `.gitignore` containing `settings.env`, `runs/`, `__pycache__/` and `.venv/`.
   - The repo goes public for Devpost.
5. **Mind Inspector shows recorded decision variables only.** It never shows an LLM's after-the-fact explanation of a decision.
6. **Freeze the event schema (§4) at Milestone 1.** It's the contract between the Lab and the Show.
7. **Python 3.12.** Keep dependencies minimal (§17).

---

## 1. What we're building

**Flagship claim:** a controlled experiment in how AI agents learn through persistent memory **without retraining**, and in **whether that memory makes them more capable or more exploitable**.

We claim causal evidence ("same model + same state + different memory → different decision, measurably better or worse outcomes"). We do not claim proof of intelligence.

**The environment:** a 3-player bluffing card game. It provides hidden information, adversarial incentives, repeated play and **ground truth**: a challenge reveals the cards, so we always know who lied.

**Signature features** (each fully specified in §7):

| Feature | One line | Priority |
|---|---|---|
| **Mind Inspector** | Freeze any decision and see the exact recorded numbers, which memories were used, and the evidence behind each | P0 |
| **Memory Surgery** | Re-run the *same* decision, or the same games on the same seeds, with memory erased, rolled back, minus one line, false or transplanted | P0 (headline) |
| **The Long Con** | An opponent earns trust with honest play, then defects. Every memory is true; the attack is assumed stationarity. Measures rounds to detection and the cost of naive memory | P0 |
| **Strategy Forge** | The agent compiles its memory into a Python strategy, which runs as untrusted code in a network-blocked Daytona sandbox and is scored | P1 (demo if time) |
| **Voice interrogation** | Before deciding, the agent asks the claimant a question (ElevenLabs TTS). Answers are evidence | P1 |

**Two deliverables:**
- **Lab** — the real system: Daytona boxes running many worlds, SurrealDB memory, Arize traces, and experiments.
- **Show** — a static web app that replays one real recorded world through a presenter-driven cue file (§12).

They share only the event log format (§4) and the Show bundle format (§12.1). Two agents can build them in parallel.

**Naming:**
- Our agent is **the Profiler**.
- Scripted opponent roles: `sleeper`, `telltale`, `telltale_overt`, `control`, `colluder`.
- On screen, opponents are **"Player 2" / "Player 3"**, with the role hidden until the reveal. No animal names.

---

## 2. Architecture

```
┌──────────────────────── Orchestrator (laptop) — `blab` CLI ─────────────────────────┐
│ provision · run · reflect worker · ablate · forge · voice · collect · analyze · pick │
│   reflect:  box ⇄ orchestrator ⇄ SurrealDB Agent Memory ⇄ Claude (claude-opus-5-5)   │
│   forge:    Claude → static gate → ephemeral Daytona forge box (network_block_all)   │
│   voice:    ElevenLabs TTS/STT                                                       │
│   tracing:  Arize AX (reflection, forge, voice spans)                                │
└────────▲───────────────────────────▲────────────────────────────────────────────────┘
         │ HTTPS preview URL + token │ (port 8000 on each box)
 ┌───────┴───────┐           ┌───────┴───────┐            ┌──────────────────────┐
 │ World box 1   │           │ World box 2   │   …        │ Forge box (ephemeral)│
 │ worldd :8000  │           │ worldd :8000  │            │ forge_eval + policy  │
 │ K worlds,     │           │ K worlds,     │            └──────────────────────┘
 │ events, ckpts,│           │ events, ckpts,│
 │ live memory   │           │ live memory   │──► Arize AX decision spans (otlp.arize.com ✅ reachable)
 └───────────────┘           └───────────────┘
```

**Why the orchestrator, not the box, talks to SurrealDB:**
- On Tier 1/2, Daytona sandboxes have org-enforced restricted outbound network that can't be overridden per sandbox ✅.
- The always-reachable "essential services" include `*.anthropic.com` ✅, `otlp.arize.com` / `app.arize.com` ✅, `api.elevenlabs.io` ✅, `github.com` ✅ and `pypi.org` ✅.
- **SurrealDB's per-context host is not on that list** ⚠️.
- So at each reflection boundary the box pauses the world, and the orchestrator does the memory and LLM round-trip (§8.3).
- If the spike shows SurrealDB *is* reachable from a box, reflection may move into the box later; the interface stays the same.

**Scaling:**
- **Horizontal:** an experiment matrix of agents × opponent tables × seeds is spread across N world boxes.
- **Vertical:** each box runs K worlds concurrently. LLM-bound worlds run as asyncio tasks; CPU-bound worlds run in a `ProcessPoolExecutor`, sized from the cgroup quota (§8.1).
- **Limits ✅:** max 4 vCPU / 8 GB RAM / 10 GB disk per sandbox; Tier-1 pool 10 vCPU in total; 300 sandbox creations per minute.
- **Default plan:** 2 world boxes × 4 vCPU, plus forge boxes on demand (1 vCPU each, deleted after use).

---

## 3. Game engine (`blab/engine.py`)

### 3.1 Rules (implement exactly)
- **Seats** `p1, p2, p3`. `p1` is always the agent under test. Seat *order* rotates per game: game g starts with seat index `g mod 3`.
- **Lives:** 3 each. At 0 a player is eliminated. **The game ends** when one player remains (the winner) or after 30 rounds (the most lives wins; a tie is a draw).
- **Deck:** 6 K, 6 Q, 6 A, 2 J (Jokers are wild), 20 cards in total. **Each round:** shuffle, deal 5 to each living player, draw a `table_rank` uniformly from {K, Q, A}.
- **First player of a round:** the loser of the previous challenge if still alive, otherwise the next living seat. They must **play** (there's nothing to challenge yet).
- **On your turn,** if there is a pending claim by the previous actor, choose one:
  - **CHALLENGE** it, or
  - **PLAY** (accept it and make your own claim): put 1–3 cards from your hand face-down and claim they're all `table_rank`, with an `utterance` string.
- **Empty hand:** if you have 0 cards, you **must challenge** the pending claim. If there's no pending claim, you're skipped.
- **Resolving a challenge:** reveal the claimed cards. **The claim is a bluff iff any card ≠ `table_rank` and ≠ J.**
  - Bluff → the claimant loses a life.
  - Otherwise → the challenger loses a life.
  - The round ends either way. If every hand is empty with no pending claim, the round ends with no loss.
- Only the **immediate next actor** can challenge a claim.

### 3.2 Determinism
- Each world has a seed. Use `numpy.random.Generator(PCG64(seed))` for dealing, plus separate child streams per seat policy (`SeedSequence(seed).spawn(4)`).
- `engine.snapshot() -> dict` (JSON-serializable: deck, hands, pile, table_rank, lives, turn, RNG states, history) and `Engine.restore(dict)`.
- Same seed + same policies + same memory → identical event log.

### 3.3 Features (shared by decisions, the predicate DSL and logreg)

Computed for a claim by claimant `c`, from the observer's point of view:

| Feature | Type | Meaning |
|---|---|---|
| `hand_size` | int | claimant's cards **after** this play |
| `claim_count` | int | cards in this claim (1–3) |
| `trusted_streak` | int | claimant's consecutive unchallenged claims across games (reset when challenged) |
| `lives_self`, `lives_target` | int | observer's lives, claimant's lives |
| `game_index`, `round_index` | int | 1-based |
| `was_challenged_last_round` | bool | claimant was challenged in the previous round |
| `table_rank` | str | K/Q/A |
| `utterance` | str | claimant's line, for `contains` / `regex` / `no_contractions` |
| `formal_register` | bool | precomputed: the utterance has no contractions and ≥ 4 words |

### 3.4 Observation (the only thing a policy receives)

```json
{"seat":"p1","game":12,"round":3,"table_rank":"K","my_hand":["K","Q","J","A"],
 "lives":{"p1":2,"p2":3,"p3":1},"hand_sizes":{"p2":2,"p3":4},
 "pending_claim":{"by":"p3","count":2,"utterance":"I have placed two Kings.","features":{...}} ,
 "round_claims":[{"by":"p2","count":1,"utterance":"That's a King."}],
 "revealed_history":[{"game":11,"round":2,"by":"p3","count":1,"was_bluff":true,"features":{...},"utterance":"..."}],
 "legal_actions":["challenge","play"]}
```

### 3.5 Card math (`p_cards`)

This is the probability that the claimant **could not** have been truthful, given my hand and uniform dealing:
- `U` = 20 − |my hand at deal| (unseen cards)
- `m` = 8 − (my matching cards incl. J)
- `h` = claimant's hand size *before* the play
- `k` = `claim_count`
- `p_cards = 1 − P(X ≥ k)`, where `X ~ Hypergeometric(U, m, h)`

---

## 4. Event log — the contract (`blab/events.py`, Pydantic + JSONL)

Each line:

```json
{"v":1,"run":"r03","world":"w017","box":"b2","game":7,"round":3,"seq":412,
 "ts":"2026-10-16T13:02:11.512Z","type":"decision","actor":"p1","payload":{},"trace_id":"…"}
```

`seq` is monotonic per world. The log is a **god view**: hidden cards are included, and the UI hides them.

| `type` | `payload` |
|---|---|
| `world_start` | `{seats:[{seat,policy,role,params}], seed, games, box, snapshot_id, variant?:{…}, parent_world?, from_game?}` |
| `game_start` | `{seat_order, lives}` |
| `deal` | `{table_rank, hands:{p1:[],p2:[],p3:[]}}` |
| `ask` | `{target, template_id, text, p_final, reason_aspect, audio?}` (§7.5) |
| `answer` | `{text, latency_ms, source:"scripted"\|"stt"\|"typed"}` |
| `claim` | `{count, claimed_rank, cards:[], utterance, features:{}}` |
| `decision` | see §7.1 (the Mind Inspector record) |
| `challenge` / `play_on` | `{target}` / `{}` |
| `reveal` | `{cards, was_bluff, loser, claimant, challenger}` |
| `life_lost` | `{seat, lives_left}` |
| `game_end` | `{winner, lives}` |
| `reflect_request` | `{seat, memory_version, new_obs:[obs_id]}` |
| `reflect_result` | `{seat, memory_version, proposed:[hyp], llm_trace_id, surreal_trace_id, latency_ms, error?}` |
| `hypothesis` | `{id, aspect, text, predicate, status, n_true, n_false, q_true:[mean,lo,hi], q_false:[…], provenance, valid_from_game, superseded_by?}` |
| `belief_superseded` | `{aspect, old_id, new_id, reason:"drift"\|"llm"\|"refuted", at_game, at_round}` |
| `surgery` | `{op, variant, by:"operator", surreal_uncertainty_id?}` |
| `forge_result` | `{policy_id, sandbox_id, gate:{ok,reason}, games, win_rate, ci, crashes, timeouts, promoted, baseline_win_rate}` |
| `world_end` | `{win_rate, games_won, brier, rtd?, revealed_to_detection?, exploitation_cost?, false_alarm?}` |

---

## 5. Agents and memory

### 5.1 Our agents (seat `p1`)

**Shared estimator:** a grid posterior over strategic bluff propensity `q ∈ {0, 0.01, …, 1}`. For a revealed claim with recorded `p_c`:
- likelihood if a bluff = `p_c + (1 − p_c)·q`
- likelihood if honest = `(1 − p_c)(1 − q)`
- prior: uniform
- prediction: `p_final = p_cards + (1 − p_cards)·E[q | cell]`

**Respond rule (all agents unless stated):**
- challenge iff `p_final > 0.5`
- otherwise challenge with probability ε = 0.1 (exploration); log `explore` and `propensity`

**Play rule (all our agents):**
- if holding matching cards (incl. J), play 1–2 of them honestly
- else a forced bluff of 1 card
- utterance `"{n} {rank}s."`

The research object is detection, not our own bluffing.

| id | Memory / estimator | Role in experiments |
|---|---|---|
| `counter` | none; `p_final = p_cards` | no-memory baseline |
| `bayes` | one grid posterior per opponent over **all** revealed claims, forever | naive memory |
| `bayes_discount` | same, with likelihood weights γ^(games ago), γ = 0.85 | recency baseline |
| `logreg_bow` (P1) | online L2 logistic regression (numpy, lr 0.1, λ 1e-3) on [logit(p_cards), all numeric features, `formal_register`, 256-dim hashed utterance unigrams + bigrams (lowercased, apostrophes kept)], fit on revealed claims; predicts P(bluff) directly | strongest statistical baseline |
| `llm_episodic` (P1, ≤ 10 worlds) | per respond decision, Claude gets the last 30 revealed claims for that opponent (features + utterance + outcome) and returns `{p_bluff, reason}` | does structured memory beat raw recall? |
| `profiler_amnesic` | Profiler with memory reset at each game start | the control for "learning without retraining" |
| **`profiler`** ⭐ | `bayes` base rate per opponent **plus** the hypothesis graph (§5.2): `E[q]` comes from the most specific *confirmed* hypothesis whose predicate matches, else the base rate; drift detection (§5.3); provenance-aware (unsupported lines get weight 0) | flagship |
| `profiler_blind` | `profiler`, but treats every memory line as confirmed regardless of provenance or evidence | false-memory experiment |
| `forge:<id>` | an AI-written `decide(obs)` (§7.4) | Strategy Forge |

### 5.2 Hypothesis graph (`blab/hypotheses.py`)

**Hypothesis:**

```json
{"id":"h_w017_p3_004","about":"p3","aspect":"verbal_tell",
 "text":"Player 3 speaks formally (no contractions) when bluffing",
 "predicate":{"f":"formal_register","op":"==","v":true},
 "status":"provisional","provenance":"evidence","created_game":6,
 "valid_from_game":6,"valid_until_game":null,"supersedes":null}
```

- **Aspects** (closed set; each becomes the SurrealDB attribute key): `baseline_bluff`, `trigger`, `verbal_tell`, `pressure`, `exploits_trust`, `collusion`.
- **Predicate DSL:**
  - leaf `{"f":feature,"op":op,"v":value}`, with `op ∈ {==, !=, <, <=, >, >=, contains, regex, no_contractions}`
  - combinators `{"all":[…]}`, `{"any":[…]}`, `{"not":{…}}`
  - `regex` is compiled with a 50 ms timeout guard; reject patterns over 100 chars
  - **an evaluator, never `eval`**
- **Scoring** (at every game end, on revealed claims *about that opponent* observed **after** `created_game`):
  - Split by whether the predicate matched; compute grid posteriors `q_true` and `q_false`.
  - `P(q_true − q_false > 0.1)` is estimated from 2,000 samples of the grid posteriors.
  - **confirmed:** `n_true ≥ 4` and P > 0.9.
  - **refuted:** `n_true ≥ 6` and P < 0.2.
  - **otherwise:** provisional.
  - A confirmed hypothesis that fails over the last 2 games (P < 0.3 on that window) becomes `refuted`.
- **Supersession:** when a new confirmed hypothesis has the same `about` + `aspect` as an older confirmed one, the older one gets `valid_until_game = now`, `superseded_by = new.id`, and a `belief_superseded` event is emitted.
- **Provenance:** `evidence` (proposed by reflection and scored on reveals) or `untrusted` (injected by surgery). The `profiler` gives an `untrusted` line weight 0 unless it is independently confirmed by scoring.
- **Memory file** per seat: `{memory_version, base:{p2:{grid},p3:{grid}}, hypotheses:[…], observations:[obs ids]}`. Bump `memory_version` on every change.

### 5.3 Drift detection (the Long Con mechanism, numeric, in `profiler`)
- At each game end, for each opponent, compare the grid posterior from the last W = 8 revealed claims against all earlier ones.
- If `P(q_recent − q_old > 0.2) > 0.9`: supersede `baseline_bluff` with a new belief computed from the recent window only (`reason:"drift"`), and emit `belief_superseded`.
- The LLM's next reflection is told about the shift and may propose an `exploits_trust` hypothesis.

### 5.4 Reflection (LLM, between games)
- **Trigger:** at the end of every game g, when g is a multiple of `reflect_every` (2), for `profiler` and `profiler_blind`.
- **Input:** the frozen system prompt (rules, feature table, DSL grammar, aspect set, output schema), then the revealed observations since the last reflection (≤ 40) and the current hypotheses with their stats.
- **Output:** `ReflectionOut{hypotheses:[{about, aspect, text, predicate, rationale, supersedes_id|null}] (≤ 3)}`.
- Invalid predicates are dropped and logged. A new hypothesis starts as `provisional` with `provenance:"evidence"`.
- **Fallback:** on timeout (60 s), refusal or error, keep the old memory and emit `reflect_result.error`.

### 5.5 Scripted opponents (`blab/agents/scripted.py`)

**Respond policy** (all scripted): challenge iff `p_cards > 0.6`, or with probability 0.1.

**Play policy:** strategic bluff probability `s(game)`.
- **Bluff:** play 1–2 cards, at least one non-matching.
- **Honest:** play 1..min(3, matching) matching cards.
- **Forced bluff:** with no matching cards, play 1 card.

| role | `s(game)` | Utterance |
|---|---|---|
| `sleeper` | 0 until G₀ ~ U{5..8}; 0.3 at G₀; 0.6 from G₀+1 | casual bank only (no tell) |
| `control` | 0 always | casual bank |
| `telltale` | 0.35 | formal bank with probability 0.75 if bluffing, else 0.25; otherwise the casual bank |
| `telltale_overt` | 0.35 | "Trust me, it's {n} {rank}s." with probability 0.8 if bluffing, else 0.15; else "{n} {rank}s." |
| `colluder` (pair, P2) | 0.3 | casual; **never challenges its partner** |

**Banks** (≥ 8 each):
- Casual (contractions): "That's two Kings.", "It's a King, don't worry.", "Here's a couple of Kings."…
- Formal (none): "I have placed two Kings.", "These are Kings.", "I am playing one King."…

**Scripted answers to `ask` (§7.5):** the Telltale answers from its formal bank when bluffing (probability 0.75); everyone else answers from the casual bank.

---

## 6. Experiments and metrics (`blab/orchestrator/analyze.py`)

**Matrix (`matrix.yaml`):**

```yaml
games: 20
reflect_every: 2
tables:                                   # opponents for seats p2, p3
  telltale:  [control, telltale]          # learning + Surgery Reveal
  longcon:   [sleeper, telltale]          # Long Con + tell together
  falsealarm:[control, control]
agents: [counter, bayes, bayes_discount, logreg_bow, profiler_amnesic, profiler, profiler_blind]
llm_agents_extra: {llm_episodic: 10}
seeds: {llm: 30, non_llm: 300}            # profiler* = llm
ablate: {agents: [profiler], at_game: 10, learning: false,
         variants: [full, "as_of:0", "as_of:4", "as_of:6", "as_of:8", "minus:verbal_tell", "false_memory", "transplant:same_role", "transplant:other_role"]}
false_memory_text: {about: p3, aspect: baseline_bluff, text: "Player 3 never bluffs", q: 0.0}
```

**Metrics** (bootstrap 95% CI over seeds for each; compare paired by seed where possible):

| Metric | Definition |
|---|---|
| Win rate | games won / games, overall and for games 11–20 |
| Brier | mean (p_final − was_bluff)² over respond decisions whose claim was revealed |
| Ablation Δ | per seed, win rate (g11–20) of each variant − `full` (same seeds, learning frozen) |
| Decision flips | number of identical decision states whose action differs between `full` and `as_of:0` (via `/replay`) |
| Discovery game | first game at which a `verbal_tell` hypothesis is `confirmed` (Profiler); for logreg, the first game its tell coefficient is > 0 with the 80% CI excluding 0 |
| Revealed-to-discovery | revealed Telltale claims observed before discovery (sample efficiency) |
| RTD | R_b = the sleeper's first strategic bluff round; R_d = first round with `p_final(sleeper) > 0.5` on 2 consecutive respond decisions, **or** a `belief_superseded` about the sleeper; RTD = R_d − R_b in rounds |
| Exploitation cost | lives `p1` lost to the sleeper's claims between R_b and R_d (and, separately, over the games after G₀) |
| False-alarm rate | fraction of `falsealarm` worlds where "detection" fires |
| Trust curve | per game: the sleeper's true honesty rate (1 − realized strategic bluff rate) vs. each agent's estimate (1 − E[q]) |
| Interrogation Δ (P1) | profiler ± ask, on win rate and revealed-to-discovery |
| Forge Δ (P1) | forge policy win rate − profiler win rate, same seeds |

**Output:** `runs/<run>/results.json` (all of the above, per agent/variant/table, with CIs) and `runs/<run>/summary.md`.

---

## 7. Signature features (full specs)

### 7.1 Mind Inspector (P0)

**Purpose:** freeze any decision and show exactly what the agent knew, what it believed, which memories it used and why it acted. It is auditable and reproducible.

**Data:** the `decision` event payload. The agent writes it *at decision time*. Nothing is recomputed or explained after the fact.

```json
{"kind":"respond","target":"p3",
 "claim":{"count":2,"rank":"K","utterance":"I have placed two Kings.","features":{"hand_size":2,"formal_register":true,"...":"..."}},
 "card_math":{"U":15,"m":6,"h":4,"k":2,"p_cards":0.39},
 "memory_used":[{"hyp_id":"h_w017_p3_004","aspect":"verbal_tell","text":"Player 3 speaks formally when bluffing",
                 "status":"confirmed","provenance":"evidence","n_true":9,"n_false":14,
                 "q_true":[0.71,0.55,0.84],"q_false":[0.12,0.05,0.22],"evidence_obs":["w017:g4r2","w017:g6r1","…"],
                 "surreal":{"entity":"agent/w017.p3","key":"verbal_tell","fact_ref":"…"}}],
 "q_used":0.61,"q_source":"h_w017_p3_004",
 "p_final":0.76,"ev":{"challenge":0.52,"play_on":-0.52},"action":"challenge",
 "explore":false,"propensity":1.0,"memory_version":23,
 "ask":{"asked":true,"template_id":"formal_callout"},
 "trace":{"arize_trace_id":"…","surreal_trace_id":"…"}}
```

**UI (Show and Lab console):**
- **Claim card:** count, rank, utterance, highlighted features.
- **Waterfall:** `p_cards` → (+memory) → `p_final`, with the threshold line at 0.5.
- **Memory lines used:** status chip, provenance chip, evidence counts, q-intervals. **Click → the evidence list**: each revealed round, with cards shown and a jump-to-replay link.
- **EV per action,** and the chosen action.
- Links: Arize trace, SurrealDB fact.
- **Dossier panel** (the adversary profile): for the selected opponent, all hypotheses grouped as **confirmed / provisional / superseded (struck through, with valid-from and valid-until games) / refuted**. A **game slider** shows the dossier *as of* game N, from box checkpoints, cross-checked against SurrealDB `asOf`.

**Acceptance:**
- For a random `seq`, every number in the panel equals the event payload.
- Evidence ids resolve to `reveal` events.
- The dossier at game N equals the memory checkpoint `g<N>`.

### 7.2 Memory Surgery (P0, the headline)

**Purpose:** a causal test. Hold the model (θ = model + prompt + policy config) and the state fixed, change only memory, and measure the effect. Two levels:

1. **Decision-level** (`/replay`, instant):
   - `a_full = π_θ(s, M)` vs. `a_var = π_θ(s, M′)` for the same recorded state `s`.
   - Returns the full decision payload under `M′` plus `{delta_p, flipped, delta_ev}`.
   - Shows memory **mattered**.
2. **Outcome-level** (`/branch` and `blab ablate`):
   - Restore the checkpoint at game 10 and play games 11–20 on the **same seeds** with `M′`.
   - **Learning is frozen** (`learning:false`: no reflection, no scoring, no drift updates), so we measure the memory itself.
   - Shows memory **helped** or **hurt**. Required, because a changed decision is not an improved decision.

**Memory variants** (one schema for both endpoints):

| variant | M′ | Question it answers |
|---|---|---|
| `full` | unchanged | reference |
| `as_of:N` | memory checkpoint after game N (`as_of:0` = empty) | dose-response: does win rate rise with memory age? |
| `minus:<aspect>` | full minus all hypotheses with that aspect (e.g. `verbal_tell`) | which memory line carries the gain |
| `false_memory` | full + an `untrusted` line from `false_memory_text` | can an injected, evidence-free memory flip decisions? `profiler` should flag/ignore it; `profiler_blind` should be exploited |
| `transplant:same_role` / `transplant:other_role` | the memory learned about a same-role (or different-role) opponent in another world, remapped onto this seat | positive transfer vs. overgeneralization |

**SurrealDB side of a surgery** (orchestrator, §10.2):
- `false_memory` also writes the line to SurrealDB as a conversational `turn` fact. Its source kind differs from the evidence-backed `upsert` beliefs, so SurrealDB raises an **`uncertainty` record** instead of superseding ✅. Store the id in the `surgery` event.
- `erase` (`as_of:0`) also calls `POST /forget` with `dryRun:true` to show what would be expired (audit, nothing lost).
- `transplant` reads the source world's beliefs from SurrealDB (a cross-scope read).

**The Show's principal moment, the "Surgery Reveal" card:**
- Freeze the Profiler's respond decision on a Telltale formal-register claim in game ~12.
- Show `full` / `as_of:0` / `false_memory` (Profiler, plus the `profiler_blind` result) side by side, with p and action for each.
- Then **flip the cards** (the `reveal`).
- Pick a decision where `full` is correct and the variants differ. Pick it honestly from the real run.

**Acceptance:**
- `/replay` with `full` reproduces the logged decision exactly.
- `as_of:0` equals `profiler_amnesic`'s computation on the same state.
- Branches on the same seed with `full` reproduce the original games 11–20 (with learning frozen, identical whenever the original also had no updates; otherwise report the drift).

### 7.3 The Long Con (P0)

**Purpose:** show that memory can make an agent **more exploitable**, and measure whether time-aware memory recovers.
- The `sleeper` plays honestly (only forced bluffs) for G₀ ∈ [5, 8] games, building a true record of honesty, then bluffs strategically.
- This is **behavioral memory poisoning**: no memory is false; the attack is the learner assuming the opponent doesn't change. Contrast it with `false_memory`, which is an injected lie.

**Mechanisms:**
- `bayes` (naive memory) trusts the sleeper more than `counter` does.
- The `profiler` detects drift (§5.3) and supersedes `baseline_bluff`. In SurrealDB, the old belief is closed with `valid_until` and stays in history ✅.

**Metrics:** RTD, exploitation cost, false-alarm rate, trust curve (§6). The pitch twist, "naive memory is more exploitable than no memory," must come from the measured exploitation cost. Report it either way.

**UI:**
- **Trust curve chart:** the sleeper's true honesty vs. the estimates of `counter`, `bayes` and `profiler`, by game, with a ⚑ at G₀ and a ◎ at detection.
- **Supersession timeline** from SurrealDB `GET …/history/baseline_bluff`.

**Acceptance:**
- RTD is computed for every `longcon` world.
- The trust curve renders from `results.json`.
- At least one real `belief_superseded` event links to a SurrealDB history entry.

### 7.4 Strategy Forge (P1, demo if time)

**Purpose:** level-2 learning. The agent compiles its memory into executable strategy code, which is treated as **untrusted** and evaluated safely in isolation.

**Pipeline** (`blab forge --world w017 --seat p1`):

1. **Generate** (Claude, `effort:"medium"`). Input: the confirmed dossier for the world's opponents, the observation schema (§3.4), the action schema, and a reference `decide()` (the `counter` policy). Output: `ForgeOut{module_source, rationale, targets:[aspect]}`.

   Required interface:

   ```python
   def decide(obs: dict) -> dict:
       # returns {"type":"challenge"} or {"type":"play","cards":[indices into obs["my_hand"]],"utterance":str}
   ```

2. **Static gate** (orchestrator):
   - `ast.parse` must succeed.
   - Imports must be in {math, random, statistics, collections, itertools, functools}.
   - Forbidden names/attributes: `open, exec, eval, compile, __import__, globals, locals, vars, getattr, setattr, delattr, input, breakpoint`, and any attribute starting with `__`.
   - Source ≤ 20 KB; must define `decide`.
   - On failure → `forge_result{gate:{ok:false,reason}}`, then optionally one repair attempt with the reason fed back.
3. **Sandbox eval:** create a forge box with:
   - snapshot/image `blab-forge`, `Resources(cpu=1, memory=1)`
   - **`network_block_all=True`** ✅ (applies on Tier 1? ⚠️)
   - `ephemeral=True`, `auto_stop_interval=10`
   - labels `{project:"blufflab", role:"forge"}`

   Upload `blab/` and `policy.py`, then run:

   ```
   python -m blab.forge_eval --policy policy.py --table telltale --seeds <same 30 seeds> --games 20 --decide-timeout-ms 50
   ```

   - **The engine runs in the parent process. The policy runs in a child process (`multiprocessing`, spawn), receives only observation JSON over a pipe, and returns action JSON.** The policy never holds hidden state.
   - Illegal action / exception / timeout → substitute the legal fallback (challenge if a claim is pending, else play 1 card) and count it.
   - stdout = JSON `{win_rate, ci, crashes, timeouts, illegal, mean_ms}`. Delete the box afterward.
4. **Score and promote:**
   - Compare with `profiler` on the same seeds.
   - Promote (register `forge:<id>` as an agent) if the CI lower bound of the Δ is > 0.
   - Record in SurrealDB: entity `agent/forge.<id>`, attributes `derived_from = <world>.<seat>@memory_version`, `win_rate`. This gives provenance from memory to code.
   - Emit `forge_result`. Trace the generate/gate/eval steps as Arize spans.

**UI (Show):** a code panel (the generated source, read-only), gate result, sandbox id + "network: blocked" badge (only if the spike verified it), eval vs. Profiler on the same seeds, and a promoted badge.

**Acceptance:**
- One end-to-end forge run produces a `forge_result` with a real sandbox id.
- The gate rejects a test module containing `import os`.

### 7.5 Voice interrogation (P1)

- **When** (respond decisions, Profiler only, ≤ 2 per game):
  - **probe** when `p_final ∈ [0.35, 0.65]`
  - **call-out** when `p_final > 0.65`, just before challenging
- **What:** templates filled from the recorded decision variables only:
  - `"Two cards left and you're claiming {k} {rank}s?"` (`hand_size`)
  - `"You've gone very formal all of a sudden."` (top memory line `verbal_tell`)
  - `"You've been very honest so far… still?"` (`exploits_trust`)
- **Answer as evidence:** scripted opponents answer per §5.5. The answer's `features` (incl. `formal_register`) are logged on `answer` and can be used by predicates (feature prefix `answer_`).
- **Sim:** text only. **Show/live:** ElevenLabs TTS for the agent's line; human answers via STT (P2) or a typed reply.
- **Pre-generate** the mp3s for the picked world's actual `ask` events (`blab voice`).

---

## 8. World box (`blab/worldd.py`)

### 8.1 Process
- FastAPI + uvicorn on `0.0.0.0:8000`.
- Worlds are asyncio tasks. CPU-only worlds run in a `ProcessPoolExecutor(max_workers = cpu_quota)`.
- **Compute `cpu_quota` from `/sys/fs/cgroup/cpu.max`** (`quota/period`). `nproc` reports the host's CPUs ✅.
- Requests authenticate with the Daytona preview token header plus our header `X-Blab-Key` (from env).

### 8.2 State on disk (what makes a box stateful)

```
/home/daytona/worlds/<wid>/events.jsonl
/home/daytona/worlds/<wid>/ckpt/g<NN>.json        # engine snapshot + every agent's memory after game NN (g00 = start)
/home/daytona/worlds/<wid>/memory/<seat>.json     # live memory (memory_version)
/home/daytona/worlds/<wid>/meta.json              # config, status, parent_world, variant
```

### 8.3 HTTP API

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/health` | | `{box_id, cpu_quota, worlds:{running,waiting,done}}` |
| POST | `/worlds` | `{wid, seats:[{seat,policy,role,params}], seed, games, reflect_every, ask:bool}` | `{wid}` |
| POST | `/worlds/{wid}/run` · `/pause` · `/step?n=` | | status |
| GET | `/worlds?status=` | `running\|awaiting_reflection\|done` | list |
| GET | `/worlds/{wid}/events?since=` · `/stream` (SSE) | | events |
| GET | `/worlds/{wid}/agents/{seat}/memory?as_of_game=` | | memory JSON (the dossier) |
| GET | `/worlds/{wid}/reflect_payload` | | new observations + current hypotheses for the waiting seat |
| PUT | `/worlds/{wid}/agents/{seat}/memory` | `{proposed:[hyp], reflect_meta}` | new `memory_version`; resumes the world |
| POST | `/worlds/{wid}/decisions/{seq}/replay` | `{memory_variant}` | decision payload + `{delta_p, flipped, delta_ev}` |
| POST | `/worlds/{wid}/branch` | `{new_wid, from_game, games, learning:false, memory_variant}` | `{wid}` (a new world on the same seeds) |
| POST | `/worlds/{wid}/agents/{seat}/surgery` | `{op:"inject"\|"remove", hyp?, aspect?}` | `memory_version` (a live edit, for the console) |
| GET | `/worlds/{wid}/results` · `/checkpoints` | | world_end metrics · list |

**`memory_variant` schema:**
- `{"type":"full"}`
- `{"type":"as_of","game":N}`
- `{"type":"minus","aspect":A}`
- `{"type":"false_memory","about":"p3","aspect":"baseline_bluff","text":"…","q":0.0}`
- `{"type":"transplant","from_box":B,"from_world":W,"from_seat":"p1","about_map":{"p3":"p3"}}`

For a transplant, the orchestrator fetches the source memory and passes it inline as `memory` instead of `from_*` when the boxes differ.

**Reflection handshake:**
1. At a reflection boundary, the world emits `reflect_request`, sets `awaiting_reflection` and waits.
2. The orchestrator: `GET reflect_payload` → SurrealDB write + retrieve → Claude → SurrealDB write → `PUT memory`.
3. The world scores, emits `hypothesis` events and resumes.
4. **If there's no reply within 60 s, it resumes with the old memory** and logs that. A box never blocks on the LLM.

---

## 9. Orchestrator CLI (`blab/orchestrator/cli.py`, Typer or argparse)

```
blab spike                                   # §15 checks; prints PASS/FAIL table
blab local --table telltale --agent profiler --seed 7     # run one world in-process (no Daytona) → runs/local/
blab provision --boxes 2 --cpu 4 --mem 8     # create world boxes, start worldd, store preview URLs+tokens in runs/boxes.json
blab run --matrix matrix.yaml --run r03      # schedule worlds across boxes (bin-pack; LLM world ≈0.05 vCPU, non-LLM ≈1)
blab reflect --run r03                       # async worker (≤8 concurrent Claude calls); runs until the run is done
blab ablate --run r03                        # create branch worlds per matrix.ablate
blab replay --world w017 --seq 412 --variants full,as_of:0,false_memory
blab forge --world w017 --seat p1
blab voice --run r03 --world w017            # TTS mp3s for that world's ask events
blab collect --run r03 && blab analyze --run r03
blab pick-demo --run r03                     # choose Show world + Surgery Reveal decision → show/bundle/
blab teardown --run r03                      # delete boxes (NOT before the awards ceremony)
```

**`blab local` must work without any cloud service.** It uses an in-process stub for reflection that proposes hypotheses from a canned list. It's the dev loop and the fallback.

---

## 10. Sponsor integrations (exact calls)

**Environment variables** (load from `settings.env`; never commit):

```
DAYTONA_API_KEY, ANTHROPIC_API_KEY,
ARIZE_SPACE_ID, ARIZE_API_KEY, ARIZE_PROJECT_NAME=bluff-lab,
SPECTRON_API_KEY,      # = your SurrealDB Agent Memory API key (sp-…). "Spectron" is just the product's old name, which the docs still use for env vars
SPECTRON_URL,          # the context's host, shown in Studio → Contexts → your context (each context has its own host)
SPECTRON_CONTEXT_ID,   # the context ID, shown in the same place
ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, BLAB_KEY,
BLAB_LLM_PROVIDER=anthropic|bedrock,   # if bedrock: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_REGION (workshop creds; temporary). ANTHROPIC_API_KEY is then unused
```

### 10.1 Daytona (`pip install daytona`) ✅

```python
from daytona import Daytona, CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams, Image, Resources, SessionExecuteRequest
d = Daytona()
img = (Image.debian_slim("3.12")
       .pip_install(["fastapi","uvicorn","numpy","pydantic","httpx","arize-otel","openinference-instrumentation"])
       .workdir("/home/daytona"))
sb = d.create(CreateSandboxFromImageParams(image=img, resources=Resources(cpu=4, memory=8, disk=8),
              auto_stop_interval=0,                              # ✅ REQUIRED: default 15 min, and in-box processes don't count as activity
              labels={"project":"blufflab","role":"world"}, env_vars={...}), timeout=0)   # labels/env_vars kwarg names ⚠️
sb.process.exec("git clone https://github.com/<user>/blufflab /home/daytona/app")       # github.com reachable ✅
sb.process.create_session("worldd")
sb.process.execute_session_command("worldd", SessionExecuteRequest(
    command="cd /home/daytona/app && uvicorn blab.worldd:app --host 0.0.0.0 --port 8000", run_async=True))   # ✅
link = sb.get_preview_link(8000)   # ✅ link.url; send link.token in header x-daytona-preview-token
# forge box:
fb = d.create(CreateSandboxFromSnapshotParams(snapshot="blab-forge", network_block_all=True, ephemeral=True, auto_stop_interval=10))  # ✅ params exist
```

- **Fallbacks:** if `git clone` of a private repo fails, upload a tarball with `sb.fs.upload_file(bytes, path)` ⚠️ (check the signature). Build the image once and reuse it as a named snapshot ⚠️.
- **Optional:** VM sandboxes (`snapshot="daytona-vm-small"`) support `sb.fork()` ✅. Not required; `/branch` is our portable fork.

### 10.2 SurrealDB Agent Memory (REST via `httpx`) ✅

- **Base:** `{SPECTRON_URL}/api/v1/{SPECTRON_CONTEXT_ID}`, header `Authorization: Bearer {SPECTRON_API_KEY}`.
  - The host looks like `https://srv1.spectron.aws-usw2.surreal.cloud`.
  - **`*.surreal.cloud` is not on Daytona's essential-services list**, so expect it to be blocked from boxes. That's why all SurrealDB calls go through the orchestrator (§2). Spike D2 should `curl` this exact host from a box.
- **SDK option** (the human reports this init works):

  ```python
  from surrealdb import Spectron
  client = Spectron(context=os.environ["SPECTRON_CONTEXT_ID"],
                    endpoint=os.environ["SPECTRON_URL"],
                    api_key=os.environ["SPECTRON_API_KEY"])
  ```

  - Use it only if its method names map cleanly onto the calls below (inspect with `dir(client)` / `help(client)`).
  - Otherwise use raw REST via `httpx`, which is fully documented.
  - Either way, wrap SurrealDB access in `blab/orchestrator/memory.py`, so the rest of the code doesn't care which one is used.
- **Setup:**
  - Create the context in Studio (studio.surrealdb.com → Contexts → API key).
  - **Register scopes before writing** ✅: `lab`, `lab/world/<wid>` (shared table: public reveals), `lab/world/<wid>/agent/<seat>` (private beliefs). Use `POST /scopes` or the CLI `agent-memory scopes create`. Exact REST body ⚠️.
- **Entity:** type `agent` (closed vocabulary ✅), name `<wid>.<seat>` (the opponent being modeled, e.g. `w017.p3`).
- **Game time → timestamps:** `2026-01-01T00:00:00Z + game·1h + round·1min`, passed as `observed_at`, so `asOf` matches game numbers.

| Operation | Call |
|---|---|
| Revealed observation (shared) | `POST /facts {"infer":"triples", ...triples..., "scopes":[["lab/world/w017"]], "observed_at":T}`. **Triples body shape ⚠️: ask Martin / test.** Fallback: `{"text":"Game 6 round 1: Player 3 claimed two Kings formally and was bluffing.","infer":"full",...}` |
| Belief per aspect (private) | `POST /facts` triples on entity `agent/w017.p3`, key = aspect, value = hypothesis text + counts → `source.kind:"upsert"` ✅. A rewrite of the same entity + key + scope + kind **supersedes** (old value gets `valid_until`) ✅ |
| False memory | `POST /facts {"text":"Player 3 never bluffs.","infer":"full","scopes":[[agent scope]],"session_id":"untrusted-channel"}` → `source.kind:"turn"` ≠ `upsert` → an **`uncertainty`** record, never a supersession ✅ |
| Retrieval for reflection | `POST /query {"query":"…","k":20,"lens":[["lab/world/w017"],["lab/world/w017/agent/p1"]],"source":"reflect:w017:g8"}`. Log `trace.traceId` ✅. **Use `lens`; `scope` is silently ignored on reads ✅.** A flat array is OR; nested is AND ✅ |
| Dossier as of game N | `GET /entities/agent/w017.p3?asOf=T` ✅ · `GET /profile` ✅ |
| Supersession chain | `GET /entities/agent/w017.p3/history/baseline_bluff` ✅ |
| Conflicts | `GET /uncertainty?entity=…` ✅ |
| Soft erase (audit) | `POST /forget {"query":"…","dryRun":true}` ✅ |
| Audit | `GET /traces/{id}` ✅ |

- **Supersession rules ✅:** same entity + key + scope + source kind → supersede, subject to a confidence floor of 0.7 if a confidence is sent (we don't send one). A different kind → `uncertainty`.
- **Don't** rely on raw SurrealQL.

### 10.3 Arize AX ✅

```python
# pip install arize-otel openinference-instrumentation openinference-instrumentation-anthropic "arize>=8"
from arize.otel import register
from openinference.instrumentation import OITracer, TraceConfig
tp = register(space_id=os.environ["ARIZE_SPACE_ID"], api_key=os.environ["ARIZE_API_KEY"], project_name="bluff-lab")
tracer = OITracer(tp.get_tracer(__name__), config=TraceConfig())
with tracer.start_as_current_span("decision") as s:
    s.set_attribute("openinference.span.kind", "CHAIN"); s.set_attribute("session.id", wid)
    s.set_attribute("input.value", json.dumps(claim)); s.set_attribute("output.value", action)
    for k, v in flat(decision).items(): s.set_attribute(f"blab.{k}", v)
tp.force_flush()   # before a world ends / process exits
```

- **Where tracing happens:** boxes trace `decision` spans. The orchestrator traces `reflection` (Anthropic auto-instrumentation), `forge.*`, `voice.*` and `surgery`.
- **Experiments (v8 SDK) ✅:**
  - `from arize import ArizeClient`; `c = ArizeClient(api_key=…)`
  - `ds = c.datasets.create(space_id=…, name="frozen-decisions-r03", examples=df)`, where rows = revealed respond decisions (input = state + memory snapshot id; `was_bluff` label)
  - `c.experiments.run(name=f"{agent}-r03", dataset_id=ds.id, task=recompute, evaluators=[brier], concurrency=10)`, with `def brier(output, dataset_row) -> EvaluationResult(score=…, label="brier")`
  - One experiment per agent for side-by-side comparison.
- **Ignore Phoenix and v7 snippets.** Ingestion lags seconds to a minute.

### 10.4 Claude (`pip install anthropic`)

- `AsyncAnthropic()`, **model `claude-opus-5-5`**. Effort: `low` for reflection and `llm_episodic`; `medium` for the forge.
- **Structured output:** `await client.messages.parse(model=…, max_tokens=16000, system=[{"type":"text","text":STABLE_SYSTEM,"cache_control":{"type":"ephemeral"}}], messages=[…], output_format=ReflectionOut, output_config={"effort":"low"})` → `resp.parsed_output`.
- **Don't send `thinking`** (it's adaptive on this model; `disabled` returns a 400).
- **Prompt caching:** keep the system prompt byte-stable (no timestamps) and put volatile content in `messages`.
- **Robustness:**
  - Check `stop_reason == "refusal"` → keep the old memory.
  - **Enable server-side refusal fallbacks** (beta `server-side-fallback-2026-07-01`, `fallbacks:"default"`), because a deception-themed prompt may trip a classifier ⚠️ (confirm the call shape).
  - `RateLimitError` → honor `retry-after`.
  - At most 8 concurrent calls.

**Provider switch: Amazon Bedrock (the hackathon AWS account).** Put it behind `blab/orchestrator/llm.py` with `BLAB_LLM_PROVIDER=anthropic|bedrock`.
- **Client:** `pip install "anthropic[bedrock]"` → `from anthropic import AsyncAnthropicBedrockMantle`, then `AsyncAnthropicBedrockMantle(aws_region="us-west-2")`. ⚠️ The async class name is inferred from the sync `AnthropicBedrockMantle`; check it with `dir(anthropic)`. Same `messages.create` / `messages.parse` surface.
- **Endpoint:** `https://bedrock-mantle.{region}.api.aws/anthropic`.
- **Model IDs** take the `anthropic.` prefix: `anthropic.claude-opus-5-5`, `anthropic.claude-sonnet-5-5`, `anthropic.claude-haiku-5-5`. A first-party ID sent to Bedrock returns a 400.
- **Credentials:** the standard AWS chain (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION`). Workshop credentials are **temporary**, so refresh them before long runs and before the demo.
- **Supported on Bedrock:** structured outputs (`messages.parse` / `output_config.format`), prompt caching and effort, all ✅.
- **Not on Bedrock:** server-side `fallbacks` (just handle `stop_reason == "refusal"` by keeping the old memory), Message Batches and fast mode ✅.
- **Spike C1b:**
  - `aws bedrock list-foundation-models --by-provider anthropic --region us-west-2` shows the models.
  - One `messages.parse` call succeeds, so the account has access.
  - Note the account's requests-per-minute and tokens-per-minute quotas. Workshop quotas can be low; drop concurrency from 8 to 4 if you get throttled.
- **Network:** `*.api.aws` is **not** on Daytona's essential-services list (only `*.<region>.amazonaws.com` is), so Bedrock from inside a box is probably blocked ⚠️. All reflection and forge calls already run on the orchestrator. Run the in-box per-decision agent `llm_episodic` with `blab local` on the laptop (≤ 10 worlds) when using Bedrock.

### 10.5 ElevenLabs (P1) ✅

- **TTS:** `POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128`, header `xi-api-key`, JSON `{"text":…,"model_id":"eleven_multilingual_v2"}` → mp3 bytes. A lower-latency model ID comes from `GET /v1/models` ⚠️.
- **STT (P2):** `POST https://api.elevenlabs.io/v1/speech-to-text`, multipart `model_id=scribe_v2`, `file=@answer.webm`.
- All audio calls are made from the orchestrator / Show side.

---

## 11. Lab console (P2)

A minimal page served by `blab serve` that lists boxes and worlds, tails a world's events, opens its Mind Inspector, and runs `/replay` live. If time is short, the Show covers everything.

## 12. Show — presentation app (`show/`)

### 12.1 Bundle (`show/bundle/`, produced by `blab pick-demo`; all real data)

| File | Contents |
|---|---|
| `world.jsonl` | events of the picked `telltale` or `longcon` world (Profiler) |
| `twin.jsonl` | the same seed with `bayes` (for the split screen) |
| `reveal.json` | the Surgery Reveal decision `seq` + real `/replay` outputs for `full`, `as_of:0`, `false_memory` (+ the `profiler_blind` result) |
| `dossier.json` | memory per game (checkpoints) + SurrealDB history/uncertainty responses |
| `ablation.json` | dose-response + `minus:verbal_tell` + transplant, with CIs and decision flips |
| `longcon.json` | trust curve, RTD distribution, exploitation cost per agent |
| `results.json` | the full metrics |
| `forge.json` (optional) | generated source, gate, eval |
| `voice/*.mp3` (optional) | TTS for the picked world's `ask` events |
| `cues.json` | the director script |

### 12.2 App
- Static `index.html` + `app.js` + `style.css`, with no build step. It must run from `python -m http.server` offline.
- Charts: inline SVG, or Chart.js from cdnjs pinned to an exact version.
- **Views:**
  - **Table:** 3 seats, cards face-down, lives, utterance bubbles, roles hidden.
  - **Mind Inspector + Dossier** (§7.1).
  - **Surgery Reveal card** (§7.2).
  - **Ablation chart:** `as_of:0 → 4 → 6 → 8 → full` bars with CIs, plus a `minus:verbal_tell` bar.
  - **Trust curve** (§7.3).
  - **Results** grid.
  - **Forge** panel.
  - **Split screen** (`world` vs. `twin`).
- **Director:** `cues.json = [{"at_seq":412,"action":"freeze"},{"action":"surgery","variant":"as_of:0"},{"action":"flip"},{"action":"view","name":"ablation"},…]`. `Space` / `→` = next cue, `←` = previous, `R` = reset. **Presenter-paced, never timer-paced.**
- **Default cold open (no second person needed):** cue the picked game-12 Telltale claim. Play the Profiler's real `ask` line as audio ("You've gone very formal all of a sudden."), then freeze.
- **Acceptance:** with only `show/bundle/` and no network, the full cue sequence plays end to end. Record it as the <2 min Devpost video and as the stage backup.

---

## 13. Analysis and pick-demo rules (`blab pick-demo`)

- **World:** a `profiler` world on the `telltale` (or `longcon`) table whose discovery game is **closest to the median** discovery game. Not the best case.
- **Surgery Reveal decision:** in that world, a respond decision in games 10–14 on a formal-register claim where `full` challenges and is correct (the reveal shows a bluff), and `as_of:0` flips to `play_on`. If none exists, pick the closest and **say so in the writeup**.
- **Write `show/bundle/PROVENANCE.md`:** the run id, world id, seed, decision `seq` and selection rule. This is the honesty record.

## 14. Repo layout

```
.gitignore            settings.env, runs/, __pycache__/, .venv/
blab/engine.py        rules, features, card math, snapshot/restore, observation()
blab/events.py        pydantic event models + JSONL writer   ← contract
blab/hypotheses.py    DSL evaluator, grid posteriors, scoring, supersession, drift
blab/agents/          ours.py (counter, bayes, bayes_discount, logreg_bow, profiler*, llm_episodic), scripted.py, banks.py
blab/world.py         World runner (games loop, checkpoints, reflection handshake, variants/branching)
blab/worldd.py        FastAPI box server
blab/forge_eval.py    in-sandbox evaluator (parent engine, child policy)
blab/orchestrator/    cli.py boxes.py reflect.py memory.py(SurrealDB) llm.py tracing.py forge.py voice.py analyze.py pick.py
show/                 index.html app.js style.css bundle/
matrix.yaml  live.yaml  SPEC.md  NOTES.md  README.md
tests/                test_engine.py test_hypotheses.py test_replay.py test_forge_gate.py
```

---

## 15. Spike checklist (`blab spike`, about 30 min)

| # | Check | Pass when | Fallback if fail |
|---|---|---|---|
| D1 | Create a world box (`auto_stop_interval=0`), start worldd, `GET /health` via the preview link | 200 | upload a tarball instead of git clone |
| D2 | From inside the box: `curl -sI https://api.anthropic.com`, `https://otlp.arize.com`, the SurrealDB host, `https://example.com` | Anthropic + Arize OK; record SurrealDB | if SurrealDB is reachable, it can move in-box later (optional) |
| D3 | Forge box with `network_block_all=True` → `curl https://api.anthropic.com` | blocked | the UI says "restricted network," not "no network"; rely on the gate + child process |
| S1 | Register scopes; triples write ×2, same key → `history/{key}` shows 2 links | supersession visible | write beliefs as `infer:"full"` sentences; keep the box-local graph authoritative |
| S2 | A `turn` write contradicting it → `GET /uncertainty` | 1 record | show the box-side `untrusted` flag only |
| S3 | `GET entity?asOf=` before the second write | old value | use box checkpoints for the dossier slider |
| A1 | One span from the laptop and one from the box appear in AX | visible | the orchestrator emits decision spans from `events.jsonl` |
| A2 | `experiments.run` with `brier` on 5 rows | experiment visible | compute in pandas, screenshot it |
| C1 | `messages.parse(ReflectionOut)` on a canned history | valid hypotheses | retry once, then use the stub |
| E1 | TTS one line | mp3 plays | skip voice |

---

## 16. Milestones (build in this order; check each one before moving on)

| # | Milestone | Acceptance | Target |
|---|---|---|---|
| M0 | `.gitignore`, repo skeleton, `blab spike` | spike table printed | 12:55 |
| M1 | `engine.py` + `events.py` (**frozen**) + scripted opponents + `counter`/`bayes` + `blab local` | 20-game local world → valid JSONL; `test_engine` passes; same seed ⇒ identical log | 13:20 |
| M2 | **Vertical slice:** `hypotheses.py` + `profiler` (stub reflection), checkpoints, `/replay` logic (in-process) | Profiler confirms a `verbal_tell` on `telltale` within 12 games for ≥ 1 seed; `replay(full)` == logged decision; `replay(as_of:0)` changes p | 13:45 |
| M3 | Show v0: table + Mind Inspector + Dossier + Surgery Reveal card on M2's real JSONL | the cue sequence plays offline | 13:45 (parallel) |
| M4 | Real reflection (Claude) + SurrealDB memory sync + Arize spans | a local world with live LLM + SurrealDB history visible + traces in AX | 14:05 |
| M5 | `worldd` on Daytona + `provision`/`run`/`reflect` + the matrix (reduced if needed) | ≥ 2 boxes, ≥ 60 worlds done | 14:20 |
| M6 | `ablate` + `analyze` + `pick-demo` → bundle; Show ablation + trust curve | `results.json` with CIs; bundle plays | 14:40 **FREEZE** |
| M7 (P1) | Strategy Forge end to end; `logreg_bow`; voice mp3s | `forge_result` with a real sandbox id | only if M6 is done |
| M8 | Devpost writeup + <2 min video from the Show + repo public | submitted | 15:20 |

**Cut order if behind:**
`colluder` → STT → `llm_episodic` → transplant → forge promotion (keep generate + gate + eval) → `bayes_discount` → `logreg_bow` → Lab console.

**Never cut:** engine + event log, the Profiler with real hypotheses, `/replay` (`full` / `as_of:0` / `false_memory`), the ablation branches (`full` / `as_of:0` / `as_of:6` / `minus:verbal_tell`), the Long Con trust curve, and the Show.

## 17. Dependencies

`numpy`, `pydantic>=2`, `fastapi`, `uvicorn`, `httpx`, `typer`, `pyyaml`, `daytona`, `anthropic`, `arize-otel`, `openinference-instrumentation`, `openinference-instrumentation-anthropic`, `arize` (v8). No other heavy dependencies (logreg is plain numpy).

## 18. Open questions (ask the human or the sponsor tables; don't block on them)

1. **SurrealDB (Martin):**
   - the exact `infer:"triples"` body
   - can triples carry `source.ref`
   - the context host domain (for the Daytona allowlist)
   - the `POST /scopes` body
   - rate limits on the $50 credit
2. **Daytona:**
   - does `network_block_all` apply on Tier 1
   - can they raise our tier
   - the exact `labels`/`env_vars` kwargs
3. **Arize:** filtering on custom attributes in the AX UI; the expected ingestion lag.
4. **The human:** GitHub repo URL (public), the ElevenLabs voice ID, and whether to show `telltale` or `telltale_overt` in the Show (decided by M2 results).
