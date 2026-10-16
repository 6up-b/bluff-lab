# Show flow — the < 2 min presentation

One recorded world, chosen by rule (SPEC §13), replayed by a presenter-paced cue file.
Every number on screen is read from `show/bundle/` (built by `blab pick-demo` from real runs).
Staging (pacing, cuts, captions) is ours; the numbers are not.

**Controls:** `Space` / `→` next cue · `←` previous cue · `R` reset · `H` presenter HUD · `F` fullscreen.
No timers ever advance a scene. Brief CSS transitions only.

**Seq placeholders:** `C` = seq of the picked Telltale `claim`, `A` = its `ask` event (if any),
`D` = the Profiler's `decision` on it, `V` = the `reveal`. `pick.py` fills these in when it writes
`cues.json`.

| # | Scene / view | On screen | Cue (keypress) → action | Presenter says | Feeds from |
|---|---|---|---|---|---|
| 0 | `title` | "Bluff Lab", one-liner, provenance strip (run, world, seed, "chosen near the median by rule") | load | — (wait for the room) | `PROVENANCE` fields in `reveal.json` |
| 1 | `table` cold open | Table at game 12: three seats, face-down cards, lives. Player 3's claim bubble appears | `play_to C` | "Game twelve. Player 3 puts down two cards and claims two Kings." | `world.jsonl` up to `C` |
| 2 | `table` ask | The Profiler's line appears as a bubble from seat P1; audio plays | `play_to A` + `audio voice/<A>.mp3` + bubble from the `ask` event text | (let the audio play) "You've gone very formal all of a sudden." — "That's a frozen model. Nobody told it to watch for that." | `ask` event; `voice/*.mp3` optional |
| 3 | `table` freeze | Table dims, "frozen at decision D" stamp | `freeze D` | "Let's freeze that decision and look inside." | `decision` event `D` |
| 4 | `inspector` | Claim card (count, rank, utterance, highlighted `formal_register`), the waterfall bar `p_cards` → +memory → `p_final` against the 0.5 line, EV per action, chosen action | `inspect D` | "Card math alone says [p_cards]. Memory holds a confirmed line…" | `decision.payload` (card_math, q_used, p_final, ev, action) |
| 5 | `inspector` evidence | The memory line used: status chip, provenance chip, n for/against, q-intervals, and its evidence list (each revealed round, cards face-up) | `evidence` | "…Player 3 drops contractions when bluffing. Where that applies, [n_true] revealed claims, it bluffed at [q_true]; otherwise [q_false]. That pushes it to [p_final], so it challenges." (Say "applies on", not "for/against": n_true/n_false count reveals where the predicate held / didn't.) | `decision.memory_used[]`, `reveal` events matched by `obs_id` |
| 6 | `dossier` | Player 3's dossier as of game 4 → 8 → 12: confirmed / provisional / superseded (struck, valid g–g) / refuted | `dossier` × 3 (pick.py chooses: game ≤ 4, the discovery game, the checkpoint the decision used = decision game − 1) | "Here is the file it built on Player 3, game by game. Nobody wrote this; reflection proposed it, reveals confirmed it." | `dossier.json` (from `ckpt/gNN.json`) |
| 7 | `surgery` | The Surgery Reveal card: same claim header; columns appear one per keypress: **full**, **as_of:0 (erased)**, **false_memory**, **profiler_blind + false_memory**; each shows p_final bar + action | `surgery full`, `surgery as_of:0`, `surgery false_memory`, `surgery profiler_blind` | "Same cards, same claim, same model; only the memory changes." "No memory: [p_0]. It trusts." "Plant 'Player 3 never bluffs': ours sees no evidence and still challenges; a provenance-blind agent swallows it." | `reveal.json` (real `replay_decision` outputs) |
| 8 | `surgery` flip | The claimed cards flip face-up; bluff/honest stamp; each column marked right/wrong | `flip` | "It was a bluff." (or whatever the reveal says — it is real) | `reveal.json.reveal` (= `reveal` event `V`) |
| 9 | `ablation` | Dose-response bars: win rate g11–20 for as_of:0 → 4 → 6 → 8 → full, plus minus:verbal_tell, 95% CIs, n worlds; decision-flip count | `view ablation` | "A changed decision isn't an improved one. So we ablate at scale on the same seeds: [a] → [b]; remove the one line: [c]." | `ablation.json` |
| 10 | `trust` | Long Con trust curve: sleeper's true honesty vs counter / bayes / profiler estimates by game, ⚑ G₀, ◎ detection | `view trust` | "Player 2 played honest for [G₀] games, then turned. Every memory was true." Twist line **only if** `longcon.json.exploitation_cost` shows bayes > counter: "Naive memory lost [x] lives — more than no memory at all." Ours: "closed its old belief after [RTD] rounds." | `longcon.json` |
| 11 | `split` (optional) | Same seed, two minds: Profiler world vs Bayes twin, lives and per-game winner | `view split` | "Same seed, same cards. Different memory." | `twin.jsonl` + `world.jsonl` |
| 12 | `results` | Results grid: win rate, win rate g11–20, Brier, discovery game, RTD, exploitation cost — per agent with CIs and n | `view results` | "All of it from [N] worlds on Daytona; every decision traced." | `results.json` |
| 13 | `forge` (optional) | Generated strategy source (read-only), static gate result, sandbox id, network badge, eval vs Profiler, promoted badge | `view forge` | "It can even compile its memory into code — which we run as untrusted, in a sandbox." | `forge.json` |
| 14 | `close` | The closing line, large | `view close` + caption | "Memory makes agents better, and it makes them exploitable. The question isn't how much an agent remembers. It's which memories it should trust." | — |

Target pacing (video < 2:00): 0–2 ≈ 0:15, 3–8 ≈ 0:55, 9 ≈ 0:15, 10 ≈ 0:20, 12 ≈ 0:10, 14 ≈ 0:05. Scenes 11 and 13 are Devpost/Q&A extras; drop them from the video if over time.

## Fallbacks (P1 features cut or data missing)

- **No `ask` event / no voice mp3** (voice is P1): scene 2 is skipped (HUD: "skipped: no data"); cold open becomes scene 1 with the claim utterance itself as the hook — "Player 3 says: *I have placed two Kings.* Card math says…".
- **No `profiler_blind` world on the same seed:** the 4th surgery column renders "not run" and its cue is skipped; the false-memory line then says only "ours flags it and still challenges."
- **No exact Surgery Reveal match** (§13 rule fails): `pick.py` picks the closest decision and `reveal.json.selection.exact=false`; the card shows "closest match, not exact" and PROVENANCE.md says why. Say it on stage.
- **No `ablation.json` / `longcon.json` yet:** the chart renders an explicit "awaiting data" plate (never a placeholder chart). Cut the scene from the video if still missing at freeze.
- **Twist not measured** (bayes exploitation cost not > counter with CI): drop the twist sentence; say the measured result instead.
- **No `twin.jsonl` / `forge.json`:** scenes 11 / 13 are skipped automatically.
- **Telltale tell not learned → `telltale_overt` used:** the inspector highlights the `contains "trust me"` feature instead; change the line to "says 'trust me' when bluffing."

## Cue file format (`bundle/cues.json`)

```json
[{"name":"cold open", "say":"Game twelve…", "actions":[{"action":"view","name":"table"},{"action":"play_to","seq":412}]},
 {"name":"ask", "requires":["voice/ask_398.mp3"], "actions":[{"action":"play_to","seq":398},{"action":"audio","file":"voice/ask_398.mp3"}]}]
```
- A cue is one keypress. `actions` run in order. A top-level `action` (SPEC §12.2 form) is accepted as a one-action cue.
- `requires`: bundle files the cue needs. If any is missing, the cue is skipped and the HUD says "skipped: no data".
- Actions: `view {name}` · `play_to {seq}` · `freeze {seq}` · `inspect {seq}` · `evidence {hyp_id?}` · `dossier {game, about?}` · `surgery {variant}` · `flip` · `audio {file}` · `caption {text}` (presenter text, never data) · `clear_caption`.
- `←` rebuilds state by replaying cues 0…i−1 from reset (audio is not replayed), so going back is always consistent.

## Status of the current bundle (built from `runs/local`, see `bundle/PROVENANCE.md`)

Rebuild any time with `.venv/bin/python -m blab.orchestrator.cli pick-demo --run local` (or `--run <cloud run>`
after `blab collect`). Preview: `cd show && python -m http.server 8000` → `http://localhost:8000/?hud`
(`?cue=N` jumps to cue N for rehearsal; `?bundle=<dir>` loads another bundle).
The HUD lists every cue; cues whose data is missing are marked "skipped: no data" and are stepped over.

## Data and honesty notes (owned by the data side: analyze.py / pick.py)

- **Two respond rules.** r01 runs every agent under `threshold` (SPEC default, the primary rule) and `ev`
  (follow-up: challenge iff p_final > 1 − L). `results.json.show_grid` is the primary rule, and `show_grids[rule]`
  holds both. Never mix rules in one comparison; every paired Δ is within one rule.
- **Reflection source.** `reveal.json.provenance.reflection` is `claude` or `stub`. pick-demo prefers Claude-reflection
  worlds. If the picked world says `stub`, the hypotheses came from a canned list. Do not say "nobody told it
  to watch for that"; say "the line was proposed from a fixed list and confirmed by the reveals".
- **Outcome sentences come from `results.json.claims` only.** Each has `holds` and a `text` whose wording already
  follows the CI: "no detectable difference" when the CI includes 0. Show a headline only when `holds` is true.
  The "better beliefs ≠ better decisions" beat is `claims.beliefs_not_decisions` (calibration better, CI
  excludes 0, and no rule shows a significant win gain).
- **The logreg_bow baseline** may be the stronger predictor: `claims.vs_logreg_<rule>`. The Profiler's case is
  auditability, surgery and provenance (`claims.false_discovery_<rule>` and `false_discovery_lower_<rule>`, measured
  on the null-tell falsealarm table). Say only what those claims support.
- **Metrics.** Both the §6 versions and the corrected versions are in `metric_defs`:
  - false alarm: `false_alarm` (§6) vs `false_alarm_belief` (corrected).
  - exploitation: `lives_lost_to_sleeper` (§6) vs `missed_bluffs`, "sleeper bluffs let through" (corrected).
  - The Show uses the corrected versions, labelled as such.
- **logreg_bow decisions** have `q_used: null` and `q_source: "logreg"`. They have no memory segment in the
  waterfall.
- **Two settings.** r01 observations include engineered tell features, notably the precomputed `formal_register`
  boolean. Label any r01 comparison (logreg_bow, the results grid) "engineered-feature setting". mem01 (if present)
  is the raw-observation experiment: same model (Claude Sonnet 5.5), same prompt, same observation, different
  memory. `bundle/memory.json` gives `setting`, `baseline` (llm_nomem, or counter as a fallback, with `note`), and
  `rules[rule][table|"all"]` → `{arms, metrics[arm], delta_vs_baseline[arm][metric], curves[arm].{win_rate,brier}[bucket]}`.
  It also has `buckets` and data-generated `claims` (`holds` + `text`). If mem01 isn't there, memory.json comes from
  r01 and says `setting: "engineered"`.
