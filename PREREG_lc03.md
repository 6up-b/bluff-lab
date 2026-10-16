# PREREG lc03: Long Con with the stronger Profiler (profiler_plus)

Committed on main before any lc03 world is run. This commit's hash and timestamp are the evidence.
lc02 stays exactly as reported; nothing here re-runs or re-scores it.

## Why

lc02 ran the Long Con with the original `profiler`. Since then `profiler_plus` (mem01/c02) is our strongest memory
agent. lc03 asks whether it catches the sleeper sooner and lets fewer of its bluffs through than the best
Bayesian arm of lc02 (`bayes_discount`), and whether that is due to Claude reflection (vs the reflection-refused
control `profiler_plus_noreflect`).

## Seeds and n (wall-time estimate made before any lc03 world ran)

- Reserved fresh range: 10000–10023. Seeds used anywhere before: 1–130, 1001–1030, 2001–2030, 2999, 3001–3030,
  4000–4023, 5000–5023, 6000–6119, 7000–7099, 8000–8023, 9000–9023.
- **Run: 10000–10011 (n = 12).** Estimate: a profiler_plus world in c02 (running now, same reflector) takes about
  400 s (20 sequential Opus reflections). Cap: 6 concurrent Claude calls (shared Bedrock), so 6 worlds at a time.
  24 seeds × 2 tables = 48 profiler_plus worlds ≈ 8 waves × 6.7 min ≈ 54 min: does not finish by 15:35 from a
  ~14:58 launch. n = 16 is 32 worlds = 6 waves ≈ 40 min, also too late. **n = 12** is 24 worlds = 4 waves ≈ 27 min,
  done ≈ 15:25 if Bedrock latency holds. 10012–10023 stay reserved and unused.

## Setup (same as lc02 except the arms; reflection as in c02 / mem01 core)

- Tables (`blab.world.TABLES`): **longcon** = [telltale, sleeper] (sleeper at p3, G0 ~ U{5..8} from the seed, as
  in lc02) and **falsealarm** = [control, control].
- p1 seat, `info: "raw"`, threshold respond rule, 20 games per world, `reflect_every: 2` world setting.
- Reflection (profiler_plus only): real Claude, **Bedrock `us.anthropic.claude-opus-5-5`, effort low**, structured
  output, via `blab.orchestrator.reflect.local_reflector`. Raw prompt: `llm.STABLE_SYSTEM_RAW` (llm.py selects it
  when the reflect payload has `info: "raw"`; the runner asserts no tell leak in it). profiler_plus reflects after
  every game (its default). 60 s timeout; a failure keeps the old memory and is counted.
  Backend note (checked before running): with both Bedrock and Anthropic API keys present, llm.py alternates
  the two Claude backends per call (same model Opus 5.5, effort low, same raw prompt), exactly as in c02's run;
  the backend per reflection is logged in reflect_result and reported.
- Runs on the laptop process (box2 not needed: the Claude cap, not CPU, is the bottleneck). Runner
  `scripts/lc03_run.py`, matrix `matrix_longcon3.yaml`, output `runs/lc03/<table>-<arm>-raw-s<seed>`.

## Arms

- `profiler_plus`: unchanged (mem01 definition), real Claude reflection.
- `profiler_plus_noreflect`: profiler_plus with reflection refused (world reflector `stub_reflector`, which raises on
  raw payloads), exactly c02's control. No network.
- Non-LLM references on the same seeds, unchanged: `counter`, `bayes_discount`, `logreg_embed` (b01 definition;
  embeddings from `blab/data/embed_cache.json` only; a cache miss raises and the world is reported as FAILED).
  The runner checks before launching that the cache covers every enumerated sentence.

## Metrics (exactly lc02's, `blab/orchestrator/analyze_longcon2._world`, reused by import)

- Detection rate (longcon worlds with a strategic sleeper bluff, R_b): detection by the §6 rule after R_b.
- Kaplan–Meier median RTD (rounds R_d − R_b), undetected worlds censored at the end of game 20.
- Exploitation = sleeper bluffs p1 let through (played on) in games ≥ G0.
- False alarm = `false_alarm_belief` (a belief_superseded fires) on falsealarm. Supplementary, as in lc02:
  `false_alarm_any` (§6 detection rule fires against a control opponent).
- Also: lives lost to the sleeper in games ≥ G0, and p1 win rate (world_end `win_rate`) per table.

## Hypotheses (paired by seed; 2000-resample bootstrap over seeds, numpy default_rng(0); 95% percentile CI)

- **H31:** profiler_plus lets through fewer sleeper bluffs after G0 than bayes_discount.
  Δ = exploitation(profiler_plus) − exploitation(bayes_discount) per seed on longcon. Holds iff CI upper < 0.
- **H32:** profiler_plus lets through fewer sleeper bluffs after G0 than profiler_plus_noreflect. Same Δ form.
  Holds iff CI upper < 0.
- **H33 (report only):** profiler_plus false-alarm rate (`false_alarm_belief`) on falsealarm, with its CI. No holds flag.

## Analysis and stopping rule

- `scripts/lc03_analyze.py` → `runs/lc03/longcon3.json` (lc02's longcon2.json shape, plus `hypotheses` with `holds`
  and `deviations`). Every arm × table cell is reported; also paired Δs vs bayes_discount for every arm. No exclusions.
- Order: all non-LLM arms and profiler_plus_noreflect for all 12 seeds first; then profiler_plus seed-major
  (seed 10000 → 10011, longcon then falsealarm), at most 6 worlds at a time.
- **No profiler_plus world is launched after 15:30; the run is stopped at 15:38.** A hypothesis uses every seed
  whose longcon world is complete for both arms it compares. Seeds that did not complete are reported as "not run".
  The runner prints no results, and no result is looked at before the run stops or finishes.
- If fewer than 12 seeds complete, that is a deviation, not a re-specification.
