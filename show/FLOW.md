# Show flow — the < 2 min presentation

One recorded world, chosen by rule (SPEC §13), replayed by a presenter-paced cue file.
Every number on screen is read from `show/bundle/` (built by `blab pick-demo` from real runs).
Staging (pacing, cuts, captions) is ours; the numbers are not.

**Controls:** `Space` / `→` next cue · `←` previous cue · `R` reset · `H` presenter HUD · `F` fullscreen.
No timers ever advance a scene. Brief CSS transitions only.

**Seq placeholders:** `C` = seq of the picked Telltale `claim`, `A` = its `ask` event (if any),
`D` = the Profiler's `decision` on it, `V` = the `reveal`. `pick.py` fills these in when it writes
`cues.json`.

Cue numbers below are for the current r01 bundle (world w0002, decision 1026); `pick.py` regenerates them for any bundle.
The cold-open round is picked by rule in `cues()`: **the first round of the picked world (lowest game, then round) that ends in a challenge and a reveal** (game 1 round 1 when it has one). The cue records the rule and seqs in its `cold_open` field.

| # | Scene / view | On screen | Cue → action | Presenter says | Feeds from |
|---|---|---|---|---|---|
| 0 | `title` | Gold "Bluff Lab", one-liner, the four card faces, provenance strip | `view title` | — (wait for the room) | `reveal.json.provenance` |
| 1 | `table` cold open: deal | Felt table; 5 cards dealt to each seat (only the Profiler's face up); "Table: Kings" plate; chips for lives; turn button on the first player | `play_to deal` (`from` = deal, animated) + caption | "Three players, five cards each. One table rank per round. Jokers are wild." | `deal` event |
| 2 | cold open: first claim | Turn button + spotlight on the claimant; speech bubble with the real utterance; cards slide face-down to the pile; "claims 1 King" | `play_to claim` | "Player 2 puts down two cards: *'Don't sweat it, two Kings.'* You may lie." | `claim` |
| 3 | cold open: next claim | "plays on" on the responder, then its own claim | `play_to claim` | "The next player can call it, or play on." | `play_on`, `claim` |
| 4 | cold open: the ask (if the round has one) | The Profiler's question bubble (voiced if an mp3 exists), the answer bubble | `play_to answer` + `audio` | "Before deciding, the Profiler can ask one question." | `ask`, `answer` |
| 5 | cold open: the call | "Challenge!" stamp on the caller | `play_to challenge` | "It calls it." | `challenge` |
| 6 | cold open: the reveal | The claimed cards flip face-up; Bluff/Honest stamp; a chip leaves the loser's stack | `play_to life_lost` | "Caught lying, you lose a life. Wrong call, the caller loses one." | `reveal`, `life_lost` |
| 7 | game N: the claim | Cut to the decision round: deal, then the round plays through (paced 0.6×) to the picked claim; turn button lands on the Profiler | `play_to C` (`from` = deal, `pace` .6) + caption | "Game ten. Nine games in. Player 3: *'The cards are one King.'*" | `world.jsonl` |
| 8 | the ask | The Profiler's question bubble, ElevenLabs voice plays; the answer | `play_to answer` + `audio voice/ask_A.mp3` | (let the audio play) "That's a frozen model. Nobody told it to watch for that." | `ask`, `answer`, `voice/` |
| 9 | freeze | Room desaturates, frost vignette, "Time stopped at decision D"; the claim and the Profiler's seat stay lit | `freeze D` | "The Profiler has to decide. Let's stop time and look inside." | `decision` D |
| 10 | `inspector` (the turn) | The table tilts away around the Profiler's seat and the Mind Inspector turns in like a card (1.1–1.25 s; crossfade under reduced motion). Card math bar grows, then memory, then p_final; EV and action; Arize trace + Daytona box | `inspect D` | "Card math alone says [p_cards]. Memory says…" | `decision.payload`, `stack.json` |
| 11 | evidence | The memory line's evidence: the last 3 revealed rounds, cards face-up | `evidence` | "…Player 3 drops contractions when bluffing: [n_true] reveals, bluff rate [q_true]." | `memory_used[]`, `reveal` events |
| 12–13 | `dossier` | The file on Player 3 at game 4 → decision game − 1; SurrealDB history line | `dossier` × 2 | "Nobody wrote this; reflection proposed it, reveals confirmed it." | `dossier.json`, `stack.json` |
| 14–17 | `surgery` | Four columns appear one per key: full, erased, false memory, provenance-blind | `surgery …` | "Same cards, same claim, same model; only the memory changes." | `reveal.json` |
| 18 | surgery flip | The claimed card flips; bluff stamp; right/wrong per column | `flip` | "It was a bluff." | `reveal.json.reveal` |
| 19 | `ablation` | Dose-response bars with CIs | `view ablation` | "A changed decision isn't an improved one. So we ablate at scale." | `ablation.json` |
| 20 | `trust` | Long Con trust curve | `view trust` | "Every memory was true; the attack is assuming people don't change." | `longcon.json` |
| 21 | `split` (optional) | Same seed, Profiler vs Bayes twin, won games as chips | `view split` | "Same seed, same cards. Different memory." | `twin.jsonl` |
| 22 | `results` | Results grid | `view results` | "All of it from real worlds." | `results.json` |
| 23 | `decisions` | Beliefs vs decisions, paired deltas | `view decisions` | "Memory sharpens beliefs; the decision rule matters too." | `results.json.decisions` |
| 24 | `stack` | How it's built: five layers (ElevenLabs, Claude on Bedrock, SurrealDB, Arize AX, Daytona) with real counts and ids | `view stack` | "Daytona runs the worlds, Claude reflects, SurrealDB remembers, Arize traces, ElevenLabs speaks." | `stack.json` |
| 25 | `stack` flow | The picked decision walks through the layers step by step (claim → decision → trace → reveal → reflection → memory → back); each layer lights as it is used | `view stack` + `flow` | "Follow this one decision through the stack." | `stack.json`, `reveal.json` |
| 26 | `forge` (optional) | Generated strategy, gate, sandbox id, network badge | `view forge` | "It compiles memory into code, run as untrusted in a sandbox." | `forge.json` |
| 27 | `close` | The closing line | `view close` | "The question isn't how much an agent remembers. It's which memories it should trust." | — |

Target pacing (video < 2:00): 0–6 ≈ 0:25, 7–11 ≈ 0:30, 12–18 ≈ 0:30, 19–20 ≈ 0:20, 24–25 ≈ 0:10, 27 ≈ 0:05. Scenes 21, 23, 26 are Devpost/Q&A extras.
Animations run inside a cue (deal, slides, flips, the turn); a keypress mid-animation snaps to the cue's end state and continues. `?cue=N` and `←` rebuild without animation.

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
