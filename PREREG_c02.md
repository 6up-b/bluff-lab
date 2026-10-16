# PREREG c02: the memory agent vs the sentence-embedding baseline

Committed on main before any c02 test-seed world is run. This commit's hash and timestamp are the evidence.
r01, mem01, b01 and c01 stay exactly as reported. Nothing here re-runs or re-scores them.

## Why

The user decided that **`logreg_embed` (sentence embeddings) is the statistical baseline from now on**, not
`logreg_bow`: hashed word counts get too strong a prior from the scripted opponents' fixed template vocabulary.
mem01 tested profiler_plus against logreg_bow. c02 repeats that test against logreg_embed on fresh seeds.

A separate exploratory comparison runs logreg_embed on the mem01 seeds (3001–3012), in `runs/mem01`, with the
analysis in `runs/mem01/embed_exploratory.json`. Its worlds ran before this commit; the paired Δs had not been
computed yet. It is labeled "exploratory: baseline chosen after mem01 was run", and it is not part of this test.
It uses mem01's seeds only. No c02 seed has been run.

## Seeds

- **Test: 9000–9011 (n = 12).** Seeds already used anywhere: 1–130, 1001–1030, 2001–2030, 2999, 3001–3030,
  4000–4023, 5000–5023, 6000–6119, 7000–7099, 8000–8023. 9000–9023 were reserved. Only 9000–9011 are run.
- **Why n = 12, not 24 (wall-time estimate, made before any c02 world ran):** a profiler_plus world in mem01 took
  about 390 s: 20 sequential Opus reflections at about 19 s each. The cap is 6 concurrent Claude calls (shared
  Bedrock). 24 seeds × 4 tables = 96 worlds is about 96 × 390 / 6 ≈ 104 min, which does not fit before 15:40.
  12 seeds is 48 worlds, about 52 min. That is borderline from a 14:40 launch.

## Information, rules, reflection (same as mem01 core)

- p1 seat, `info: "raw"`, threshold respond rule, 20 games per world, `reflect_every: 2` world setting.
- Reflection: real Claude. **Bedrock `us.anthropic.claude-opus-5-5`, effort low**, structured output, through
  `blab.orchestrator.reflect.local_reflector`. Raw prompt: main's `llm.STABLE_SYSTEM_RAW`, which is
  byte-identical to the mem01 runner's in-process patch (checked), with raw predicate validation. 60 s
  reflection timeout. A failure or timeout keeps the old memory, as in every previous run, and is counted.
- profiler_plus reflects after every game (`plus_reflect_every` = 1, its default; this is mem01's schedule).

## Arms (main as of this commit; no shared module changed)

- `counter`: unchanged.
- `logreg_embed`: unchanged (b01 definition): a 64-dim fixed random projection (seed 0) of sentence embeddings
  (`openai/text-embedding-3-small`, from `blab/data/embed_cache.json` only, with no network at decision time; a
  cache miss raises) plus raw numeric features. Online L2 logistic regression, λ = 1.0, lr = 0.1. The cache
  covers every sentence these roles can say: there are 665 enumerated sentences and 0 misses.
- `profiler_plus`: unchanged (mem01 definition).
- `profiler_plus_noreflect`: profiler_plus with reflection refused, exactly like mem01's post-hoc `plus_nohyp`
  control. The world's reflector is `stub_reflector`, which raises on raw payloads, so memory never gains a
  hypothesis and no network is used. That leaves the calibrator alone: logit p_cards, numeric features and
  recency weighting. Its refused reflections are counted, but they are by design, not failures. (This is
  NOT c01's `profiler_plus_noreflect`, which adds the record check and the population prior.)
- `profiler_hybrid` is not included: it was not on main at launch.

## Tables

telltale, tell_conjunctive, tell_flipper, longcon (the mem01 core tables). Every table × arm × seed is one
world: 4 × 4 × 12 = 192 worlds. Wids: `runs/c02/<table>-<arm>-raw-s<seed>`.

## Hypotheses (verbatim)

- **H21:** profiler_plus's win rate is higher than logreg_embed's, pooled.
- **H22:** profiler_plus's Brier is lower than logreg_embed's, pooled.
- **H23 (control):** profiler_plus_noreflect's win rate does not beat logreg_embed's, pooled (the CI is reported).

Operationalization:
- Win rate = p1's world `win_rate` over 20 games. Brier = the world_end `brier` (mean (p_final − was_bluff)² over
  p1's respond decisions on revealed claims), as in mem01.
- "Pooled" = per seed, the mean over the 4 tables of (arm − logreg_embed). The bootstrap is over seeds.
- H21 holds iff the 95% CI lower bound is > 0. H22 holds iff the upper bound is < 0. H23 holds iff the lower bound
  is ≤ 0 (no detectable win-rate advantage of the calibrator alone over logreg_embed).

## Analysis (`blab/orchestrator/analyze_c02.py` → `runs/c02/confirm.json`)

- Paired by seed. 2000-resample bootstrap over seeds (numpy default_rng(0)), 95% percentile CI.
- No exclusions. Every cell is reported: per table × arm mean win rate and Brier, and paired Δ vs logreg_embed
  for every arm, per table and pooled. The secondary pooling over table × seed pairs (mem01's) is also
  reported, with no holds flag.
- **Stopping rule (pre-registered, compute-driven, independent of results):** the non-LLM arms run first for all
  seeds. Then profiler_plus runs seed-major (seed 9000→9011, table order as above), 6 worlds at a time. No world
  is launched after **15:30**, and the run is stopped at **15:40**. A hypothesis uses every seed for which all 4
  of its tables are complete for both arms it compares. Seeds that did not complete are reported as "not run",
  with their count. The runner prints no results, and no result is looked at before the run stops (or
  finishes).
- If fewer than 12 seeds complete, that is reported as a deviation. It is not a re-specification.
- Runner: `scripts/c02_confirm.py`.
