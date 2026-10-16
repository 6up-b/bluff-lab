# Internal interfaces (for parallel implementers)

SPEC.md is the source of truth; this file pins down the seams between modules that are being
built **concurrently by different agents**. Do not edit a module owned by someone else — if
you need a change in it, write it down under "Requests" at the bottom of this file.

Run everything with `.venv/bin/python` (Python 3.12, deps installed). Tests: `.venv/bin/python -m pytest -q`.
CLI: `.venv/bin/python -m blab.orchestrator.cli <cmd>` (or `pip install -e .` → `blab`).

## Already written (shared, owned by the lead — read, don't rewrite)
- `blab/events.py` — **FROZEN event contract** (§4). Use `EventWriter.emit(type, payload, game=, round=, actor=)`.
  Payload models validate on emit. You may add *optional* fields only (payload models allow extra keys).
- `blab/grid.py` — grid posterior over q (§5.1): `uniform, update(post,p_c,was_bluff,weight), posterior(claims),
  mean, interval(post,mass)->[mean,lo,hi], p_diff_greater(a,b,delta,rng,n), to_list/from_list`.
- `blab/env.py` — `load_env()` reads `settings.env` (aliases `SURREALDB_API_KEY`→`SPECTRON_API_KEY`). `has(*keys)`.
- `blab/orchestrator/cli.py` — Typer dispatcher; each command lazily imports `module:function` (see that file for
  exact names/kwargs). Implement the function it points at; sync or async both work.

## Ownership
| Owner | Files |
|---|---|
| core | `blab/engine.py`, `blab/agents/{scripted,banks,ours}.py`, `blab/world.py`, `blab/replay.py`, `tests/test_engine.py`, `tests/test_replay.py`, `matrix.yaml` |
| hypotheses | `blab/hypotheses.py`, `tests/test_hypotheses.py` |
| integrations | `blab/worldd.py`, `blab/orchestrator/{spike,boxes,reflect,memory,llm,tracing,voice,forge}.py`, `blab/forge_eval.py`, `tests/test_forge_gate.py` |
| show | `show/**`, `blab/orchestrator/pick.py` (bundle builder), `blab/orchestrator/analyze.py` |

## Seats, ids
- Seats `p1,p2,p3`; `p1` is the agent under test. World ids `w001…`; local runs write `runs/local/<wid>/`.
- Observation id of a revealed claim: `obs_id(world, game, round)` → `"w017:g4r2"` (one reveal per round max).

## Observation (§3.4) — the only thing a policy receives
Exactly the JSON in SPEC §3.4, plus:
- every `revealed_history` entry: `{obs_id, game, round, by, challenger, count, was_bluff, cards, utterance, features, p_cards}`
  where `p_cards` is computed **from the observing seat's own hand at that deal** (observer-specific, no hidden info),
  and `cards` are the revealed cards (public after a challenge).
- `pending_claim.features` includes the §3.3 features **and** `card_math: {U,m,h,k,p_cards}` for this observer.
- `revealed_history` covers ALL games so far in this world (public record). Agents decide what they remember.

## Policy interface (`blab/agents/*`)
```python
class Policy:
    id: str                       # "counter", "profiler", "scripted:telltale", ...
    def __init__(self, seat: str, rng: np.random.Generator, params: dict): ...
    def act(self, obs: dict) -> dict
        # {"type":"challenge"} | {"type":"play","cards":[indices into obs["my_hand"]],"utterance":str}
        # Our agents also return "decision": <§7.1 payload dict> for respond AND play decisions.
    # memory-bearing agents only (duck-typed; World checks hasattr):
    learning: bool
    def get_memory(self) -> dict          # JSON-serializable, includes memory_version
    def set_memory(self, mem: dict) -> None
    def on_game_end(self, obs: dict, game: int) -> list[tuple[str, dict]]   # events to emit: ("hypothesis",{...}), ("belief_superseded",{...})
    def wants_reflection(self, game: int) -> bool
    def reflect_payload(self) -> dict     # {seat, memory_version, new_obs:[revealed entries], hypotheses:[...]}
    def apply_reflection(self, proposed: list[dict], game: int) -> list[tuple[str, dict]]
```
**Memory timing rule (makes /replay exact):** checkpoint memory changes only at game end (scoring, drift,
reflection). During game g the agent decides from `memory = ckpt g<g-1>` **plus** in-game working memory =
the current game's entries in `obs.revealed_history`. Hence `as_of:0` (empty checkpoint + in-game) ==
`profiler_amnesic`. Each decision records `memory_version`, `obs`, and `rng_u` (the exploration uniform draw)
so that `replay(full)` reproduces the logged decision bit-for-bit.

## Memory file (§5.2)
```json
{"memory_version": 23, "agent": "profiler", "seat": "p1",
 "base": {"p2": {"grid": [...101 floats], "n": 14}, "p3": {...}},
 "hypotheses": [ <hypothesis dicts, §5.2 shape + scoring fields> ],
 "observations": ["w017:g4r2", ...],
 "revealed": [ <revealed_history entries remembered> ]}
```

## hypotheses.py API (owned by hypotheses; core imports it)
```python
ASPECTS = ("baseline_bluff","trigger","verbal_tell","pressure","exploits_trust","collusion")
FEATURES: dict[str, type]                       # §3.3 table (+ "answer_*" prefix allowed)
def validate_predicate(pred: dict) -> tuple[bool, str]
def eval_predicate(pred: dict, features: dict) -> bool        # evaluator, never eval()
def specificity(pred: dict) -> int                            # # of leaves; "most specific" tie-break
def new_hypothesis(*, wid, about, aspect, text, predicate, created_game, provenance="evidence",
                   idx: int, supersedes=None) -> dict
def score_hypothesis(h: dict, revealed: list[dict], rng: np.random.Generator) -> dict
    # revealed = entries about h["about"] with game > h["created_game"] (caller may pass all; fn filters)
    # sets n_true,n_false,q_true,q_false (each [mean,lo,hi]), p_gap, status per §5.2, evidence_obs
def recent_window_check(h, revealed, now_game, rng) -> bool   # confirmed→refuted rule over last 2 games
def supersede(hyps: list[dict], now_game: int, now_round: int) -> list[tuple[str, dict]]  # belief_superseded events; mutates
def detect_drift(revealed_about_opp: list[dict], rng, W=8, delta=0.2, thresh=0.9) -> tuple[bool, np.ndarray|None, float]
def matching_q(hyps, features, about, base_post, *, trust_untrusted=False) -> tuple[float, str|None, list[dict]]
    # E[q] from the most specific CONFIRMED hypothesis whose predicate matches (q_true mean), else
    # q_false of a confirmed hypothesis that does NOT match (more informative than base), else base mean.
    # untrusted lines count only if trust_untrusted (profiler_blind) or independently confirmed.
    # returns (q_used, q_source, memory_used[] entries in §7.1 shape)
def hypothesis_event(h: dict) -> dict          # payload for the "hypothesis" event
```

## World runner (`blab/world.py`, owned by core)
```python
class World:
    def __init__(self, *, wid, seats: list[dict], seed: int, games: int, reflect_every=2, ask=False,
                 out_dir: str|Path|None, run="local", box="local", variant: dict|None=None,
                 learning=True, reflector: "Reflector|None"=None, parent_world=None, from_game=None): ...
    status: str        # "created"|"running"|"awaiting_reflection"|"paused"|"done"
    async def run(self) -> dict          # returns world_end payload
    def pause(self); def resume(self)
    events: EventWriter
# Reflector = async callable (wid, seat, payload: dict) -> dict | None
#   returns {"proposed":[{about,aspect,text,predicate,rationale,supersedes_id}], "llm_trace_id":..., "surreal_trace_id":..., "source":"claude"|"stub"}
#   World wraps it in asyncio.wait_for(…, 60) and keeps the old memory on timeout/None/error.
def stub_reflector(...)                  # canned hypotheses (incl. formal_register==true verbal_tell) — no network
def run_local(table, agent, seed, games, reflect_every, out, llm, ask, wid) -> dict
def replay_decision(world_dir, seq, variant: dict) -> dict   # decision payload under M' + {delta_p, flipped, delta_ev}
def branch(world_dir, new_wid, from_game, games, variant, learning=False) -> World
```
Disk layout per world (§8.2): `<out>/<wid>/events.jsonl`, `ckpt/gNN.json` (engine snapshot + every agent's memory
after game NN; g00 = start), `memory/<seat>.json`, `meta.json`.
`memory_variant` dicts exactly as SPEC §8.3. With `llm=True`, run_local uses `blab.orchestrator.reflect.local_reflector`
(integrations) instead of the stub.

## Show bundle
SPEC §12.1. Built only from real runs (`runs/…`). Never hand-write numbers into bundle files.

## Requests
(append here: "<from> → <owner>: what you need")
hypotheses → core: §3.3 `formal_register` needs ≥ 4 words, so formal-bank lines like "These are Kings." (3 words) do NOT register as formal — keep every formal-bank line ≥ 4 words (and every casual line with a contraction), or the telltale's tell is diluted.
show → core/lead: run the Show's telltale profiler worlds with `--ask` so real `ask` events exist; the cold-open audio cue (`play_to` the ask + `voice/ask_<seq>.mp3`) is auto-skipped without one. pick-demo prefers decisions that used a confirmed verbal_tell line in games 10–14; with seed 7's g9 refutation, the picked decision (seq 840) used only the base rate (q_source base:p3) — more seeds or the WINDOW_MIN_TRUE fix should give a tell-line decision.
show → integrations (voice): write TTS mp3s to `<world_dir>/voice/ask_<seq>.mp3` (seq = the `ask` event's seq); pick-demo copies `<world_dir>/voice/*.mp3` into `show/bundle/voice/`.
show → integrations (forge): write the Show's forge result to `runs/<run>/forge.json` = the `forge_result` payload + `source` (module text) + `network_blocked_verified: bool` (true only if spike D3 passed); pick-demo copies it into the bundle.
show → whoever runs ablate: branch worlds must carry `world_start.variant` + `parent_world` + `from_game` (core's branch does), and include a `full` branch per seed so Δ-vs-full is paired. analyze reads every `events.jsonl` under `runs/<run>/` at any depth, so `collect` may use any layout.

show → design: handoff done (0a0d1be) — app.js gained a `decisions` view ("Better beliefs, better decisions?", cue after results) rendering `results.json.decisions` = {rows:[{rule, metric:"brier"|"win_rate", label, better:"lower"|"higher", fmt, delta:{mean,lo,hi,n}}], headline:{holds,text}, texts:[…]}; headline shown only if holds. Other new results.json fields for you: `claims` (every outcome sentence with `holds`, generated from data — never show `text` when holds is false unless it is the neutral variant), `show_grids[rule]` (one grid per respond rule; `show_grid` = primary rule, has `respond_rule`), `deltas[rule][table]["profiler − counter"|"profiler − bayes"|"profiler − logreg_bow"]`, `metric_defs`, `longcon.twist.{holds,diff,holds_spec,diff_spec}`, new grid column `false_discovery` (null-tell table). logreg_bow decisions have `q_used:null`, `q_source:"logreg"` — the waterfall must not show a memory segment as "q".
show(data) → design: new bundle file `show/bundle/stack.json` (from run artifacts only; null = unavailable): {run, daytona:{boxes:[{id,sandbox_id,cpu,mem}]|null (null when the run did not use boxes), worlds_total, worlds_per_box:{box:n}|null, picked_world_box, forge_sandbox_id, forge_network:"blocked"|null, spike:{check:status}}, claude:{provider, model, reflections (non-stub, no error), reflections_total, proposed_hypotheses, sample_reflect_trace}, surrealdb:{traces, sample_surreal_trace_id, supersessions_in_logs, supersession_example:{entity,key,history_len_from_log,reason,world}}, arize:{project, sample_arize_trace_id, decision_spans_in_picked_world, sample_reflect_trace}, elevenlabs:{voice_id,mp3s,world}|null, picked_decision:{world,box,seq,arize_trace_id,surreal_trace_id}}. forge.json is now normalised: top-level forge_result fields (policy_id, sandbox_id, gate, games, win_rate, ci, crashes, timeouts, promoted, baseline_win_rate, network) + source, rationale, targets, eval, network_blocked_verified, for_picked_world, provider. reveal.json.provenance gains respond_rule + reflection ("stub" ⇒ do not claim the LLM found the tell).
show(data) → design: please add a "Does memory help?" scene from new bundle file `show/bundle/memory.json` = {run, source, setting:"raw"|"engineered", baseline:"llm_nomem"|"counter", headline, note, buckets:["g1–5","g6–10","g11–15","g16–20"], rules:{<rule>:{<table>|"all":{arms:[…], metrics:{arm:{win_rate,win_rate_late,brier:{mean,lo,hi,n}, n_worlds}}, delta_vs_baseline:{arm:{win_rate,win_rate_late,brier:{mean,lo,hi,n}}} (paired by seed), curves:{arm:{win_rate:[4×{mean,lo,hi,n}], brier:[…]}}}}}, claims:{"<arm>|<metric>|<rule>":{holds,sign,text}}}. Show the headline verbatim (it is conditional: "same model…" only when baseline is llm_nomem), label `setting` on screen (r01 = "engineered-feature setting"), and only show claim texts as written. Suggested: Δ-vs-baseline dot+CI per arm (win rate, Brier), plus learning curves by bucket. Built from runs/mem01 when it exists (BLAB_MEMORY_RUN overrides), else from the Show run.
forge → show/design: `runs/<run>/forge.json` is a JSON **list**, one entry per forge run (latest last): {policy_id ("forge:<world>.p1.v<memory_version>"), source_world ("r01/w…"), source_seat, table, source_memory_version, source_reflection ("claude"|…), respond_rule, model, provider ("bedrock"|"anthropic"), effort, rationale, targets:[aspect], module_source (generated code, show read-only), dossier:[confirmed hyps fed to Claude: {id, about, aspect, text, predicate, n_true, n_false, q_true, q_false, p_gap, evidence_count}], gate:{ok,reason}, repair_attempted, sandbox_id (real Daytona id; box deleted after), network:"blocked"|"restricted" (in-box probe of api.anthropic.com), network_blocked_verified:bool, eval:{games, seeds, win_rate, ci:[lo,hi], crashes, timeouts, illegal, decisions, mean_ms, per_seed}, compare:{<agent>[_<other rule>][_eps0] for agent in profiler/logreg_bow/counter (e.g. profiler, profiler_eps0, profiler_ev, profiler_ev_eps0), counter_module (the reference counter decide() given to Claude, same child harness): {win_rate, ci, respond_rule, eps (null = agent's own ε=0.1 exploration as in the runs; 0 = greedy), reflection}, profiler_claude_r01?: {win_rate, ci, n_worlds, seeds, note} (different seeds — context only), paired_delta_vs_<agent>:[mean,lo,hi] (forge − agent, per seed, same seeds), promotion_rule, promoted_spec_rule_vs_profiler_eps0.1, simplified_rule_forge_ci_lo_gt_profiler_mean}, promoted (SPEC rule — paired Δ vs profiler CI-lo > 0 — AND the same vs the greedy profiler_eps0, since the forge module never explores), control (null, or "empty_dossier" = same prompt with the memory withheld: policy_id ends ".nomem"; show it as the control, never as the forge result), code (git rev), registered_agent? , arize_trace_id, created, forge_result (the §4 ForgeResult payload)}. Eval seeds 3001–3030 × 20 games, held out from every source/tuning seed; all compared policies use those same seeds and opponents. Baseline profilers use stub reflection (deterministic, local). Collected world logs are not modified (no forge_result event is appended). **show: `pick.forge_bundle` does `d.get(...)` on forge.json, which raises on a list — please take the entry whose source_world ends with the picked wid, else the last one; the per-world files `runs/<run>/forge/<wid>_p1.json` keep the old dict shape (world, source, forge_result, eval.network, attempts) and work with pick unchanged. Show text: label the comparison "vs Profiler (stub reflection), same 30 seeds"; `baseline_win_rate` = that profiler.**
core → llm-arms: raw mode ready (7681ebb). Enable it per world with p1 seat param `info: "raw"` (e.g. `seats=make_seats(t, a, seed); seats[0]["params"]["info"]="raw"`, or `run_local(..., p1_params={"info":"raw"})`). World then hands the policy `blab.agents.common.strip_obs(obs)`: no `formal_register`/`answer_formal_register`/`answer_has_contraction` in any feature dict, plus a neutral `word_count` (and `answer_word_count`). God-view `claim` events keep everything. State text: `blab.agents.common.public_obs_text(obs, *, about=None, max_history=None) -> str`, deterministic (the same obs gives byte-identical text); pass it the stripped obs. DSL: `hypotheses.validate_predicate(pred, mode="raw")` / `new_hypothesis(..., mode="raw")` reject formal_register and no_contractions. Profiler-family reflect payloads carry `info:"raw"`. `stub_reflector` raises on raw payloads, so raw memory agents need real Claude reflection.
core → integrations (BLOCKING for a valid raw-mode experiment): `blab/orchestrator/llm.py`'s system prompt lists `formal_register`, `no_contractions`, `answer_formal_register`, and an example hypothesis "speaks formally (no contractions) when bluffing". That leaks the tell. When `payload["info"] == "raw"`, please use a raw prompt variant: drop formal_register / no_contractions / answer_formal_register, list `word_count` (int: words in the utterance) and `answer_word_count`, and use a tell-neutral example (e.g. `{"all":[{"f":"hand_size","op":"<=","v":2},{"f":"claim_count","op":">=","v":2}]}`). Validate with `hypotheses.validate_predicate(pred, mode="raw")`. Raw-mode predicates that fail validation are also dropped on the core side.

show(data) → integrations (voice): the r01 Show world is `w0002` (Claude reflection, threshold rule); its cold-open ask is seq 1024. Please run `blab voice --run r01 --world w0002` (→ runs/r01/w0002/voice/ask_<seq>.mp3); then re-run `pick-demo --run r01` to copy it. Forge: pick-demo copies runs/r01/forge.json (or runs/r01/forge/<wid>_p1.json, preferring the picked world) into the bundle, normalised. Design FYI: ablation.json now has `respond_rule` and per-bar `n_by_reflection`; branch Δs pair by parent world.
## Status: hypotheses
DONE — `blab/hypotheses.py` (full API + extras `apply_drift`, `is_always_true`, `has_contraction`, `formal_register`; `score_hypothesis(..., now_game=None)` applies the recent-window rule when given; memory_used entries add `used_as`/`matched`/`q_override`), 68 tests in `tests/test_hypotheses.py` pass; power: median 14 revealed telltale claims to confirm formal_register (p_cards 0.1–0.3).

## Status: core
- M1 engine/scripted/counter/bayes/World/run_local: DONE — `.venv/bin/python -m blab.orchestrator.cli local --table telltale --agent profiler --seed 7` → `runs/local/telltale-profiler-s7/` (wid default `<table>-<agent>-s<seed>`; events validate on emit).
core → all (FYI): decision events carry `obs` with `revealed_history` trimmed to the CURRENT game (`obs.revealed_history_scope:"current_game"`); earlier games live in ckpt memory (timing rule). Claim payloads carry god-view `bluff_kind` ("strategic"|"forced"|null). Tables: `longcon` = p2 telltale, p3 sleeper (swapped vs SPEC: p1 only ever judges p3's claims, since only the next actor can challenge). Extra table `telltale_overt`.
core → hypotheses/lead: the §5.2 recent-window rule (P<0.3 over last 2 games ⇒ refuted, absorbing) fires on noise: seed 7 confirms the formal tell at g5 then refutes it at g9 with overall p_gap 0.99. Workaround in core (spec-compatible): reflection may re-propose a refuted predicate as a NEW line (stub does so only when the refuted line's overall p_gap > 0.9); it re-confirmed at g15 with a `belief_superseded{reason:"refuted"}`. Consider requiring WINDOW_MIN_TRUE ≥ 4.

## Status: show
- FLOW.md (storyboard, scene list, fallbacks, cue format): DONE — `show/FLOW.md`.
- App (`show/index.html`, `app.js`, `style.css`, vendored fonts in `show/vendor/`; no build, offline): DONE — table, inspector + evidence, dossier slider, surgery reveal + flip, ablation, trust curve, split, results, forge, close; HUD (H), Space/→/PageDown, ←/PageUp, R, F; `?cue=N` jumps, `?bundle=dir` loads another bundle; missing files ⇒ "awaiting data" plates / auto-skipped cues.
- `blab pick-demo --run local|<run>` (`blab/orchestrator/pick.py`): DONE — §13 rule, real `replay_decision` for full/as_of:0/false_memory/profiler_blind (agent override), writes world/twin/reveal/dossier/ablation/longcon/results/cues + PROVENANCE.md.
- `blab analyze --run <run>` (`blab/orchestrator/analyze.py`): DONE (v1) — win rate (+g11–20), Brier, discovery game, revealed-to-discovery, RTD, missed sleeper bluffs, lives lost to sleeper, false alarm (spec + belief-only), trust curve, ablation bars with paired Δ, twist test; bootstrap CIs over seeds → results.json, summary.md, ablation.json, longcon.json. Decision-flip count not yet computed (null).
- M2 profiler/profiler_blind/profiler_amnesic/bayes_discount + stub reflection + checkpoints + replay/branch: DONE — `.venv/bin/python -m pytest -q tests/test_engine.py tests/test_replay.py` (26 pass). Replay: `... cli replay telltale-profiler-s1 1043 --variants full,as_of:0,false_memory,false_memory@profiler_blind` (`<variant>@<agent>` re-runs the same state with another agent). Branch: `... cli branch telltale-profiler-s7 --from-game 10 --variant as_of:0` → `runs/local/<wid>.b10.as_of-0/`.
- M2 acceptance: stub-reflection profiler confirms the formal_register verbal_tell about the telltale within 12 games on 28/30 seeds (1–30; median discovery g7; misses: 15, 22); telltale_overt 30/30 (median g5). Tell open+confirmed at every ckpt g9–g14 on seeds 1,2,4,6,8,9,11,13,14,19,21,23,25,26,27,28. Surgery-Reveal-shaped decisions (formal claim, full challenges, reveal=bluff, as_of:0 → play_on): seed 1 seq 1043 (g10), 1306 (g13).
- Ask (§7.5, sim text): `... cli local --ask` → `ask`/`answer` events; answer features enter the claim as `answer_*`; decision.ask records it; replay stays exact.
- Real runs in `runs/local/`: telltale × {profiler,bayes,counter,profiler_blind,profiler_amnesic} × seeds {7,1}; branches `telltale-profiler-s{7,1}.b10.{full,as_of-0,as_of-6,minus-verbal_tell,false_memory}`; profiler on telltale_overt/longcon/falsealarm s7; longcon bayes/counter s7; `telltale-profiler-s7-ask`.
- For worldd (integrations): `World(..., reflector=async fn(wid, seat, payload))` — to do the box handshake, pass a reflector that awaits a Future resolved by `PUT /memory`; while waiting `world.status == "awaiting_reflection"` and `world.pending_reflection` holds the payload. Return `{"proposed":[...], "source":..., "llm_trace_id":..., "surreal_trace_id":...}` or `{"error":...}`. Timeout default 60 s (`reflect_timeout=`). `World.pause()/resume()` gate between games.
- Honest findings (20 seeds × 20 games, telltale): win rate counter 0.39, bayes 0.26, profiler 0.29, amnesic 0.13; Brier profiler 0.281 < bayes 0.291 < counter 0.319. Memory improves calibration but NOT win rate under the §5.1 0.5 threshold (3-player: a wrong challenge costs me a life, a right one is shared with the other opponent). Longcon (sleeper at p3): exploitation cost counter 24.5 > profiler 5.9 > bayes 4.7 (counter rarely challenges) — the "naive memory is more exploitable" twist does NOT reproduce. falsealarm: drift fires on 7/20 control worlds.
- Task 1 (win rate vs Brier) — diagnosis, seeds 1–20, telltale, threshold rule: challenge rate / accuracy: counter 0.25/0.60, bayes 0.39/0.51, profiler 0.47/0.54, amnesic 0.55/0.43. p1 lives lost to wrong challenges vs own claim caught: counter 547/416, bayes 812/215, profiler 841/168, amnesic 1012/124 (no other cause). P(p1 loses a life later in the round | played on) = 0.26–0.34, so playing on is NOT free. p1 starts 36–39% of rounds for every agent. Cause: memory agents estimate q ≈ 0.4 for the telltale, so p_final > 0.5 fires on many ~50/50 claims, and each wrong challenge costs a life for certain.
- New respond-rule param `respond_rule: "threshold"(default, §5.1) | "ev"`, the same code path for every agent in ours.py. With "ev", P(I lose a life this round) is 1 − p_final if I challenge and L if I play on. Exactly one life is lost per round, so the value of an opponent's life cancels: **challenge iff p_final > 1 − L**, with L = (k+1)/(n+3) (Beta(1,2) prior, mean 1/3). Here n counts p1's past non-forced play_ons and k counts the rounds it then lost. L comes from the memory checkpoint (`memory.self`); counter and amnesic always use the prior 1/3. Decisions record `ev={challenge:-(1-p), play_on:-L}`, `respond_rule`, `ev_detail{L,L_n,L_k,threshold,formula}`. Rule chosen before evaluation; evaluated on seeds 101–130, telltale (win ± se): threshold: counter .362±.024, bayes .262, bayes_discount .247, amnesic .110, profiler .283±.019 (paired profiler−counter −.078±.030). ev: counter .325, bayes .327, bayes_discount .338, amnesic .330, profiler .340±.021 (paired +.015±.028; Brier .253). The ev rule closes the gap, but memory still gives no significant win-rate gain.
- Task 2 (hypotheses.py, minimal): **recent-window refute now needs ≥ 4 matched claims in the window and P < 0.2** (SPEC §5.2: P < 0.3, no minimum). Tuned on seeds 1001–1030; fresh seeds 2001–2030, telltale: true tell refuted on noise 11/30 → 5/30; discovery ≤ 12 games 25/30 both before and after; a real 2-game reversal (test fixture, P = 0.159) is still refuted. **Drift unchanged (SPEC 0.9)**; the knobs are now module constants `DRIFT_THRESH/DRIFT_MIN_OLD/DRIFT_DELTA`. Fresh seeds at 0.9: falsealarm false alarms 11/30; longcon drift detects the sleeper after G0 in 11/30 (median 2 games late), RTD(any route) median 13.5 rounds. At 0.95: false alarms 2/30, but detection drops to 2/30 and median RTD rises to 15, so I did not adopt it. min_old = 8 had no effect. Most false alarms are about p3 (12 of 16 at 0.9), the opponent p1 judges directly. My guess (not verified): misspecified p_cards (it uses the claimant's current hand size) combined with p1's changing challenge selection. After the change: M2 acceptance 28/30 seeds within 12 games; tell confirmed at every ckpt g9–g14 on 23/30 seeds. runs/local regenerated with the current code.
- Not done: logreg_bow.

## Status: integrations
Spike (`blab spike`, results in `runs/spike.json`, 2026-10-16 ~12:55):

| # | Status | Notes |
|---|---|---|
| D1 | PASS | box from default snapshot, `auto_stop_interval=0`, repo uploaded as tarball (`sb.fs.upload_file(bytes, path)`), uvicorn `/health` → 200 via preview URL + `x-daytona-preview-token` |
| D2 | PASS | from a box: bedrock-runtime.us-west-2 / api.anthropic.com / openrouter.ai / otlp.arize.com / app.arize.com / api.elevenlabs.io / pypi / github = reachable; **example.com and the SurrealDB host = blocked** → Surreal stays on the orchestrator (as specced) |
| D2b | PASS | real Bedrock call (`us.anthropic.claude-opus-5-5`) from inside a box with AWS creds passed via `env_vars` |
| D3 | PASS | `network_block_all=True` box: anthropic + example.com blocked (UI may say "network: blocked") |
| S1 | PASS | triples body = `{"infer":"triples","triples":[{"entity":{"type":"agent","name":"w017.p3"},"key":"<aspect>","value":"…"}],"scopes":[["lab/world/w017/agent/p1"]],"observed_at":T}`; same key twice → `history/<key>` shows 2 links, `supersededBy` set |
| S2 | FAIL | `infer:"full"` text writes extract to an `other` entity, no uncertainty record (`/uncertainty` returns `{"unknowns":[]}`); a `session_id` other than the key's own is rejected (403). Fallback: box-side `untrusted` flag only |
| S3 | PASS | `GET /entities/agent/<wid>.p3?asOf=T` returns the old value |
| A1 | PASS | laptop span; box-side `decision` spans also verified (decision.trace.arize_trace_id on box runs) |
| A2 | SKIP | experiments not exercised |
| C1 | PASS ×3 | bedrock 8.3 s, anthropic 7.6 s, openrouter 6.6 s; all return `formal_register==true` verbal_tell on the canned history |
| E1 | PASS | ElevenLabs TTS mp3 (default premade voice chosen via GET /v1/voices; logged in voice/manifest.json) |

Notes for everyone:
- settings.env has `SPECTRON_URL` and `SPECTRON_CONTEXT_ID` values swapped; `memory.conf()` detects and swaps them (fix the file when convenient).
- Org limit: 10 GiB RAM total across sandboxes → `blab provision` clamps to 4 GiB/box for 2 boxes (2×4 vCPU/4 GiB + 1 GiB forge).
- LLM provider order (`BLAB_LLM_PROVIDER=auto`): bedrock → anthropic → openrouter (`anthropic/claude-opus-5.5`), falls through on expired STS creds / refusals / errors. Bedrock + direct both use `messages.parse(output_format=ReflectionOut)`; direct adds server-side `fallbacks:"default"`.
- Box world dirs are `/home/daytona/worlds/<run>/<wid>/`. `GET /export?run=` tars a run for `blab collect`.
- worldd: `POST /worlds` extra fields `reflector: orchestrator|stub|claude|none` (non-LLM worlds run in the ProcessPool), `start: true`, `table`.

Implemented + tested: spike, llm, reflect (local_reflector verified in `blab local --llm`: Bedrock + Surreal trace ids land in reflect_result), worldd (handshake / events / SSE / memory / surgery / replay / branch / results / checkpoints / export), boxes (provision, run_matrix boxes + `--local-only`, collect, ablate boxes + local, teardown, redeploy), tracing, memory, voice, forge_eval gate (`tests/test_forge_gate.py`).

integrations → core: `run_local` hard-codes `run="local"` in events; please add a `run` kwarg (boxes passes run ids for `--local-only` runs). Also matrix.yaml is still missing — `blab run` needs it (scratch test matrix used: tables telltale, agents counter/bayes/profiler).
- Verified end to end (13:2x): `blab run` on 2 boxes (t02: 8 worlds, Bedrock reflection via `blab reflect` handshake, Surreal trace ids, box decision spans) + `blab ablate` (10 branch worlds on boxes incl. false_memory/transplant) + `blab collect`; `--local-only` run + local ablate (t03); `blab forge w007 --run t02` → real forge sandbox id, gate ok, network BLOCKED, 100 games eval (promotion rule used: forge CI-lo > Profiler(stub) mean on same seeds).
- After code changes on the laptop: `python -c "from blab.orchestrator.boxes import redeploy; redeploy()"` re-uploads the repo and restarts worldd on both boxes (kills running worlds).
- Box-side decision spans add `decision.trace.arize_trace_id` + envelope `trace_id` (both stripped by `canonical()`); `BLAB_TRACE_DECISIONS=off|llm|all` (default llm = async/LLM worlds only).
- [13:16] r01 launched (commit 9b06a5e deployed via git archive): 4368 worlds = 48 real-Claude profiler (telltale+longcon × 12 seeds × {threshold,ev}; telltale ones ask=True) + 480 stub profiler/profiler_blind (4 tables × 30 seeds × 2 rules) on boxes + 1920 non-LLM (4 agents × 4 tables × seeds 1–60 × 2 rules) on boxes + 1920 non-LLM seeds 61–120 on the laptop. p1 params carry respond_rule + reflection (claude|stub|none). Estimate ~12 min + collect. Log: scratchpad r01.log.
- [13:19] r01: all 2448 box worlds scheduled (69 s, 0 errors); the 1920 laptop non-LLM worlds (seeds 61–120) are done (21 s). logreg_bow landed after launch, so I ran it on the laptop into the same run: 960 worlds (`wl0001…`, 4 tables × 120 seeds × 2 rules, specs in runs/r01/worlds_logreg.json, 13 s). Real-Claude reflections are running on the orchestrator at 16 concurrent, spread across Bedrock and Anthropic. Known issue: a few game-2 reflections timed out (source "timeout", old memory kept) because the reflect worker only started after scheduling finished. Fixed in boxes.py for future runs; the worker now runs during scheduling.
- **PRE-REGISTRATION plus01 (written 13:25, before any plus01 run).** Question: does `profiler_plus` ("LLM proposes, statistics disposes") beat `logreg_bow`?
  - **Agents:** counter, logreg_bow, profiler, profiler_plus. profiler and profiler_plus use REAL Claude reflection (`blab.orchestrator.reflect.local_reflector`), at most 12 concurrent LLM calls. profiler reflects every 2 games, profiler_plus every game.
  - **profiler_plus design (fixed now):**
    - Per-opponent logistic calibrator fit by L2-MAP Newton at game end, with prior N(w0, I) where w0 = [bias 0, logit(p_cards) 1, others 0]. With no data it reduces to counter (p = p_cards).
    - Likelihood weights 0.9^(games ago), so non-stationary opponents can be un-learned.
    - Features: logit(p_cards), the base numeric features scaled to [0, 1] (no formal_register, no BoW), plus one binary feature per usable hypothesis. Usable = open, not refuted, not always-true, and evidence provenance (untrusted only if independently confirmed).
    - Hypotheses are scored on ALL revealed claims about the opponent except the batch of reveals shown to the LLM when the line was proposed (the false-discovery guard). No supersession between non-drift lines.
    - Decisions record each hypothesis feature's coefficient with a 95% Laplace interval and its contribution to the logit.
  - **Tables:** telltale, tell_conjunctive, tell_flipper, longcon. New roles were DESIGNED AFTER seeing logreg_bow's weakness; label them so.
    - tell_conjunctive: s = 0.35. Formal with p 0.85 if (bluff and hand_size ≤ 2) or (honest and hand_size > 2), else 0.15.
    - tell_flipper: telltale until G1 ~ U{8..10}, then the tell inverts.
    - answer_tell (with ask) only if time.
  - **Runs:** seeds 3001–3012 (cut to 3001–3008 if the wall-time estimate requires it), 20 games, respond_rule threshold (primary) and ev (secondary). Output in runs/plus01/.
  - **Metrics:** win rate (all games and g11–20); Brier over revealed respond decisions; discovery game on telltale (profiler/plus: first confirmed verbal_tell about p3; logreg: memory tell_coef discovery_game).
  - **"Beats":** paired (by seed) Δ win rate, profiler_plus − logreg_bow, per table and rule, with a bootstrap 95% CI (2000 resamples). "Beats" = CI lower bound > 0. Brier Δ reported the same way (lower is better). Anything else is reported as "no detectable difference" or "worse".
  - **Amendment (13:28, still before any plus01 run).** Measured real reflection latency is about 20 s per call. 12 seeds × 4 tables × 2 rules × 30 calls is about 2,900 calls, which does not fit the window. So: threshold rule (primary) with REAL Claude on seeds 3001–3008 (about 960 calls, about 27 min at 12 concurrent). The ev rule runs on the same seeds with the STUB reflector and is labeled "ev/stub" (a secondary, mechanism-only comparison).
- [13:25] r01 DONE. runs/r01/ has 5928 world dirs (+ worlds.json, worlds_logreg.json, placement.json, ablate.json, reflect_log.json). The box part took 403 s, including 451 real-Claude reflections (Bedrock + Anthropic); 453 of 480 reflect_results have source claude and 27 timed out (5.6%; mostly game 2 before the worker started), with old memory kept. That gives 957 ask events in telltale LLM worlds. Ablate: 588 branch worlds (84 telltale profiler parents (24 claude + 60 stub) × 7 variants at g10, learning frozen) all finished on the boxes; collect took ~60 s. Boxes are still up (no teardown).
- Raw-information mode (7681ebb): p1 param `info:"raw"`, plus `blab/agents/common.py` (strip_obs, raw_features, public_obs_text); world.py edits: `World.observe()` (strips for raw seats) used for decision and game-end observations; raw answer features; new tables tell_conjunctive / tell_flipper / answer_tell plus the tell_flipper G1 param; stub refuses raw. Engineered mode is unchanged (all 113 tests pass); old logreg weights are zero-padded for the new raw-only `word_count` column. The plus01 engineered run was stopped before producing any worlds (superseded by mem01).
- mem01 (core arms) is running since 13:30. Runner: `/tmp/.../scratchpad/mem01_core.py` (copied to `runs/mem01/core_runner.py`). Arms counter, logreg_bow, profiler, profiler_plus, all with `info:"raw"` and respond_rule threshold. Tables telltale, tell_conjunctive, tell_flipper, longcon; seeds 3001–3012; 20 games; real Claude reflection (local_reflector, at most 10 concurrent calls; at most 14 worlds at once, seed-ascending, so seeds finish completely in order). Wids `<table>-<agent>-raw-s<seed>` in runs/mem01/. **Runner-side raw prompt:** llm.py has no raw variant yet, so the runner patches `llm.STABLE_SYSTEM` in its own process. The formal_register row becomes word_count, the no_contractions op is removed, and both examples are replaced with tell-neutral ones (asserted: no "formal" left in the prompt). `llm._validate_pred` is patched to `validate_predicate(pred, "raw")`. The pre-registered plus01 "beats" criterion (paired bootstrap 95% CI lower bound > 0) is applied to mem01.
- Mechanism check only, NOT evidence (engineered features + the stub's canned predicates, which encode our knowledge of the tells; seeds 3001–3012, threshold). Paired Δ win rate profiler_plus − logreg_bow, ± se: telltale +.088±.039, tell_conjunctive +.100±.034, tell_flipper +.046±.041, longcon +.067±.034. The calibrator works when given good predicates; the real question is the raw/Claude result.

## Status: llm-arms
- **PRE-REGISTRATION mem01 / LLM arms (written 13:37, before any mem01 LLM-arm world was run; a 4-game pilot on seed 2999, excluded, measured cost).**
  - **Question:** same model, same prompt, same raw information. Does memory help, and which kind?
  - **Agents** (`blab/agents/llm_agents.py`; registered in `ours.AGENTS` via a one-line import at the end of ours.py). Each respond decision is one Sonnet 5.5 call (`us.anthropic.claude-sonnet-5-5` on Bedrock via AWS_SONNET_API_KEY, falling back to `claude-sonnet-5-5` direct). Settings: effort low, default temperature, cached byte-stable system prompt, structured output `{p_bluff, reason}`. The user message is core's `public_obs_text` over the current game only, plus the card-math line (U m h k p_cards), then the memory block. **The memory block is the only thing that differs between arms:**
    - `llm_nomem`: no memory block ("no memory").
    - `llm_episodic`: the last 30 revealed claims by the claimant in previous games, in core's claim-line format.
    - `llm_dossier`: the claimant's hypotheses (text, predicate, status, n_true/n_false, q_true/q_false, provenance; up to 12, confirmed first). They are maintained by ours.Profiler's scoring, supersession and drift, plus Claude reflection every 2 games through `reflect.local_reflector`, with the raw reflection prompt and raw DSL validation copied verbatim from core's mem01 runner. This is the same machinery as the raw profiler arm.
    - Every arm uses the same respond rule (threshold 0.5, ε = 0.1 with recorded rng_u) and the same play rule as ours.py.
    - Forced responses (no legal play) make no call: p_final = p_cards in every arm.
    - When a call times out or fails (30 s budget), p_final = p_cards and `llm.error` is recorded. These are counted and reported.
    - `reason` is the model's pre-decision rationale, recorded at decision time.
    - Replay: responses are cached by sha256(system + user) in `<world>/llm_cache.jsonl`. replay(full) reuses the cached response and is exact. A memory variant changes the prompt, so it makes a fresh call.
  - **Reference arms:** counter, logreg_bow, profiler, profiler_plus (raw, real reflection) come from core's runner in runs/mem01 (same seeds, tables and wids). I do not re-run them.
  - **Tables, seeds, games:** telltale, longcon, tell_flipper (tell_conjunctive is dropped for budget). Seeds 3001–3008, 20 games, threshold rule, info: raw. Wids are `<table>-<arm>-raw-s<seed>` in runs/mem01/.
  - **Budget (measured):** the pilot made about 13 calls per game (≈ 260 per world) at 2.4 s mean latency, with at most 16 concurrent calls (decisions and reflections share one semaphore; core's reference arms use their own ≤ 10). The full plan is ≈ 18.7k calls ≈ 50 min, which does not fit. The queue therefore runs seed-major (seed → table → arm), stops launching new worlds at 14:08, and has a hard stop at 14:25. Only finished worlds are analysed. The seeds that complete are decided by queue order, not by results.
  - **Metrics** (per arm × table; bootstrap 95% CI over seeds, 2000 resamples; paired by seed Δ vs llm_nomem):
    - win rate, overall and games 11–20;
    - Brier over revealed respond decisions;
    - Brier learning curve by game bucket (1–5 / 6–10 / 11–15 / 16–20);
    - tell discovery (dossier arms): first confirmed verbal_tell line about the tell seat, with its text;
    - longcon / tell_flipper: strategic bluffs by the changing seat (p3) let through after the change point (first strategic sleeper bluff, or game G1), and recovery = rounds until p_final > 0.5 on 2 consecutive strategic bluffs by that seat;
    - LLM error count.
  - **"Memory helps" criterion:** primary = pooled (table × seed pairs) paired Δ win rate, arm − llm_nomem. An arm "helps" iff the 95% CI lower bound > 0, and "hurts" iff the upper bound < 0. Otherwise the result is "no detectable difference". Secondary measures are reported the same way: Δ Brier (helps iff CI upper bound < 0), Δ win rate in games 11–20, and per-table Δ. Raw transcript vs structured is compared as llm_episodic vs llm_dossier on the same Δs.
