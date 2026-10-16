# Bluff Lab

A controlled experiment in how AI agents learn through persistent memory without retraining, and in whether
that memory makes them more capable or more exploitable.

The testbed is a 3-player bluffing card game with ground truth. Every challenge reveals the cards, so we always
know who lied. Our agent, **the Profiler**, keeps a typed, time-aware memory of each opponent: a base bluff rate,
plus hypotheses written as machine-checkable predicates. Between games, a frozen Claude model proposes those
hypotheses, and later reveals score them (provisional → confirmed → refuted or superseded). Only confirmed lines
affect decisions, and lines without evidence get weight 0. Every decision is recorded, and you can replay any of
them under a different memory.

- **Mind Inspector:** the recorded decision variables for any decision: card math, the memory line used and its
  evidence, p(bluff), EV and the action.
- **Memory Surgery:** the same recorded state under `full` / `as_of:N` / `minus:<aspect>` / `false_memory` /
  `transplant` memory. Decision level uses `/replay`. Outcome level uses `/branch`: games 11–20 on the same
  seeds with learning frozen.
- **The Long Con:** a sleeper opponent is honest for 5–8 games, then defects. We measure rounds to detection,
  the bluffs it gets through, and the trust curve.
- **Strategy Forge:** memory is compiled into a Python strategy, checked by a static gate, and run as
  untrusted code in a network-blocked Daytona sandbox.

`SPEC.md` is the full specification. `NOTES.md` covers the pitch and framing. `INTERFACES.md` describes the
module seams.

## Quick start (local, no cloud)

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
# one 20-game world, stub reflection (no network) → runs/local/telltale-profiler-s7/
.venv/bin/python -m blab.orchestrator.cli local --table telltale --agent profiler --seed 7
# re-run one recorded decision under different memories
.venv/bin/python -m blab.orchestrator.cli replay telltale-profiler-s7 <decision seq> --variants full,as_of:0,false_memory
# outcome-level surgery: restore game 10, play 11–20 on the same seeds with memory erased
.venv/bin/python -m blab.orchestrator.cli branch telltale-profiler-s7 --from-game 10 --variant as_of:0
# metrics with bootstrap CIs → runs/local/results.json + summary.md
.venv/bin/python -m blab.orchestrator.cli analyze --run local
```

Tables are `telltale`, `longcon`, `falsealarm` and `telltale_overt`. Agents are `counter`, `bayes`,
`bayes_discount`, `logreg_bow`, `profiler_amnesic`, `profiler` and `profiler_blind`. Real Claude reflection
uses `--llm`.

## The Show (presentation app)

A static page that replays one real recorded world through a presenter-paced cue file. There is no build step,
and it works offline.

```bash
.venv/bin/python -m blab.orchestrator.cli pick-demo --run local     # or --run r01 → show/bundle/
cd show && python -m http.server 8000                                  # open http://localhost:8000/?hud
```

| Key | Action |
|---|---|
| Space / → / PageDown | next cue |
| ← / PageUp | previous cue |
| R | reset |
| H | presenter HUD |
| F | fullscreen |

`?cue=N` jumps to a cue.

`pick-demo` chooses the world and the Surgery Reveal decision by a fixed rule (SPEC §13), never by hand. It
writes every bundle file from real event logs and real `replay_decision` outputs, and records the choice in
`show/bundle/PROVENANCE.md`. Panels whose data doesn't exist render "awaiting data", and cues whose files are
missing are skipped.

## Cloud run (Daytona)

```bash
# create settings.env with the keys listed below (gitignored; never commit it)
.venv/bin/python -m blab.orchestrator.cli spike                  # PASS/FAIL table for every integration
.venv/bin/python -m blab.orchestrator.cli provision --boxes 2 --cpu 4 --mem 8
.venv/bin/python -m blab.orchestrator.cli run --matrix matrix.yaml --run r01
.venv/bin/python -m blab.orchestrator.cli reflect --run r01      # orchestrator-side Claude + SurrealDB loop
.venv/bin/python -m blab.orchestrator.cli ablate --run r01
.venv/bin/python -m blab.orchestrator.cli collect --run r01 && .venv/bin/python -m blab.orchestrator.cli analyze --run r01
.venv/bin/python -m blab.orchestrator.cli pick-demo --run r01
.venv/bin/python -m blab.orchestrator.pick --devpost --run r01   # DEVPOST.md → DEVPOST.filled.md from results.json
```

`run --local-only` runs the same matrix in-process (ProcessPoolExecutor) without Daytona.

Environment variables (in `settings.env`):

- `DAYTONA_API_KEY`
- AWS Bedrock credentials or `ANTHROPIC_API_KEY`
- `ARIZE_SPACE_ID`, `ARIZE_API_KEY`
- `SPECTRON_API_KEY`, `SPECTRON_URL`, `SPECTRON_CONTEXT_ID` (SurrealDB Agent Memory)
- `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`
- `BLAB_KEY`

Every integration degrades to a no-op or SKIP when its keys are missing.

## Architecture

```
orchestrator (laptop, `blab` CLI) ── provision · run · reflect · ablate · forge · voice · collect · analyze · pick
   │  reflect: box ⇄ orchestrator ⇄ SurrealDB Agent Memory ⇄ Claude (Opus 5.5 via Bedrock)
   │  tracing: Arize AX · voice: ElevenLabs
   ▼  HTTPS preview URL + token
Daytona world boxes: worldd (FastAPI :8000), K worlds each → events.jsonl, ckpt/gNN.json, memory/<seat>.json
Daytona forge box (ephemeral, network_block_all): forge_eval runs the engine in the parent process and the
   untrusted policy in a child process
```

| Path | Contents |
|---|---|
| `blab/engine.py` | rules, features, card math, snapshot/restore |
| `blab/events.py` | the frozen event contract between Lab and Show |
| `blab/hypotheses.py` | predicate DSL evaluator (never `eval`), grid posteriors, scoring, supersession, drift |
| `blab/agents/` | scripted opponents; `counter`, `bayes`, `logreg_bow`, `profiler*` |
| `blab/world.py`, `blab/replay.py` | world runner, checkpoints, `/replay`, `/branch` |
| `blab/worldd.py` | box server |
| `blab/orchestrator/` | integrations, `analyze.py` (metrics), `pick.py` (Show bundle + Devpost fill) |
| `show/` | the presentation app |

The event log is the only contract between the Lab and the Show: the Show is a pure renderer of real logs.

## Sponsor usage

- **Daytona:** stateful world boxes run the experiment matrix in parallel. Each box holds worlds, checkpoints
  and live memory, and serves `/replay` and `/branch`. Ephemeral forge sandboxes use `network_block_all` to
  run AI-written strategy code.
- **Arize AX:** OpenTelemetry spans for decisions (from the boxes) and for reflection, forge, voice and surgery
  (from the orchestrator). Frozen-decision datasets feed one experiment per agent with a Brier evaluator.
- **SurrealDB Agent Memory:** each opponent is an entity and each belief is an attribute. Rewriting a belief
  supersedes the old one, so history and `asOf` show what the agent believed and when. A false memory written
  through a different source raises an `uncertainty` record. Reflection retrieves through scoped lenses.
- **Anthropic Claude (Opus 5.5, via AWS Bedrock):** reflection between games returns structured hypotheses
  validated against a closed predicate DSL. Claude also writes the Strategy Forge code. The LLM is never in
  the move-by-move loop.
- **ElevenLabs:** TTS for the Profiler's real `ask` lines, used in the Show's cold open.

## Honesty rules

- No number is ever written by hand. Every number in the Show and in the Devpost text comes from
  `runs/<run>/results.json` or the event logs.
- Outcome sentences are generated from the data with their own conditionals.
- Agents only ever see `observation(seat)`. Hidden cards exist only in the god-view log.
- LLM output is never executed in our process.
