# PREREG — reg01: "The Regulars" (identity-keyed memory) + surgical forgetting

Written 2026-10-16 ~14:20 EDT, before any reg01 test-seed world was run. This commit's hash and timestamp are
the evidence. The only things run before this commit are a reflection-latency pilot on seed 9900 (not a test
seed; excluded, used only to choose the reflection model for wall time: Sonnet 5.5 low effort, 7.8 s and 6.9 s).

Question: does memory that follows WHO it meets win more against returning opponents, and can one person be
forgotten surgically without touching what the agent knows about everyone else?

## Confirmed by the user (lead relayed at ~14:20; moved out of "Reconstructed")

- **H8 (identity), confirmed by the user.** On return encounters (a regular's 2nd–5th match), profiler_plus's
  win rate is higher than profiler_plus_noid's AND higher than logreg_pooled's. Paired by seed, aggregated over
  all four regulars. Holds only if both paired-bootstrap 95% CIs exclude 0 in the predicted direction.
- **H9 (strongest baseline), confirmed by the user.** profiler_plus vs logreg_identity on return encounters.
  Report only, no directional claim.
- **Pool, confirmed by the user.** p2 = control (fixed, every match). p3 = one regular per match:
  - Avery = `telltale`: formal line w.p. 0.75 when bluffing, 0.25 when honest.
  - Blake = `reverse_telltale` (NEW role): formal w.p. 0.75 when HONEST, 0.25 when bluffing. Same banks, same
    s = 0.35. Avery and Blake appear equally often, so a learner that pools them sees the effects cancel.
  - Casey = `telltale_overt`. Drew = `tell_conjunctive`.
- **Reflection-off control** `profiler_plus_noreflect` is pre-registered (same agent, identity-keyed, no
  reflection, so no hypotheses: a per-name calibrator on the base numeric features only).

## Amended by the user (replaces the reconstructed text)

1. **Schedule.** 20 matches = 5 blocks, each block a random permutation of the 4 regulars, drawn from
   `numpy.random.default_rng([seed, 8080])`. No regular plays two matches in a row, including across block
   boundaries: a block whose first regular equals the previous block's last is re-drawn (same generator).
   Each regular plays exactly once per block. Every arm sees the identical schedule for a given seed.
2. **H10 timing.** Forget at the END of match 16 (after block 4, i.e. after game 48). Measure in matches
   17–20 (block 5), where each regular appears exactly once.
3. **H10 metric** (paired by seed: forget branch vs no-forget, same seed, learning ON in both):
   - Forgotten regular: the win-rate Δ (forget − no-forget) in their block-5 match is < 0, 95% CI excludes 0.
   - Every other regular: the whole 95% CI of their Δ lies within [−5, +5] pts (equivalence on the CI).
   - H10 holds for a forgotten name only if both parts hold. Run forget:Avery and forget:Blake as separate
     branch sets (H10a, H10b).
   - Also reported: whether the forgotten regular's hypotheses were re-proposed by the end of match 20
     (relearning), and the SurrealDB /forget dryRun + real responses.
   - Power note (written before running, not a change): each regular plays 3 games in block 5, so a per-seed
     Δ is a multiple of 1/3. With n ≤ 24 seeds the ±5-pt equivalence is expected to be very hard to reach;
     it is scored strictly as written. Supplementary (no claim): Brier Δ on p1's respond decisions in block 5.
4. **Decision replays.** Take the no-forget season's first non-forced respond decision against Avery after
   game 48, and the first one against Casey. Replay each with the memory checkpoint at the start of its game,
   (a) unchanged (must reproduce the logged p_final exactly) and (b) with forget:Avery applied. Log Δp and
   whether the action flipped (same rng_u). Prediction: Avery shifts; Casey is identical with Δp = 0 exactly.
5. **Raw mode, clean prompt.** p1 has `info: "raw"`; reflection uses `llm.STABLE_SYSTEM_RAW` (picked when the
   payload has `info: "raw"`); `formal_register` / `no_contractions` are rejected by the DSL, so Avery's and
   Blake's register tells must be discovered from raw text (contains/regex). Regular names never appear in a
   prompt except as speaker labels (implementation: the reflection call sees the regular as `p3`, see below).

## Fixed design (lead's spec, not garbled)

- **Season world:** 60 games = 20 matches × 3 games, one engine and one seat-RNG stream for the whole season
  (seed = world seed). p1 (ours) judges p3's claims, as in longcon. At each match start p3's scripted policy is
  replaced by the scheduled regular (same p3 RNG stream).
- **Identity is public:** every claim carries the speaker's name (known identity, "voice diarization"): in
  p1's observation, `by`/`challenger` = "p3" is replaced by the current regular's name. Raw text only.
- **Arms (all p1, raw mode, threshold respond rule p_final > 0.5, ε = 0.1):**
  - `counter` (p_final = p_cards)
  - `logreg_match`: logreg_bow (raw), memory reset at every match start
  - `logreg_pooled`: logreg_bow (raw), persistent, keyed by seat (ignores the name)
  - `logreg_identity`: logreg_bow (raw), persistent, one model per name (the honest strongest baseline)
  - `profiler_plus`: identity-keyed memory (hypotheses carry `about: "<name>"`, calibrator per name)
  - `profiler_plus_noid`: the same agent, all regulars under one key (`p3`)
  - `profiler_plus_noreflect`: identity-keyed, no reflection
- **Reflection:** at the end of each match (games 3, 6, …, 60). The payload is filtered to the current
  regular's (and p2's) reveals and hypotheses; the regular is relabelled `p3` for the LLM call and proposals
  about `p3` are relabelled back to the name. noid sends `p3` unchanged. Same prompt, same model, same filter
  rule for both, so the key is the only difference.
- **Reflection model (pre-registered deviation for wall time):** the default reflection model (Opus 5.5,
  Bedrock, effort low) has a measured median latency of ~40 s per call (lc02 logs), which makes 20 serial
  reflections per season infeasible before 15:05. reg01 uses `us.anthropic.claude-sonnet-5-5` on Bedrock,
  effort low, same raw prompt and schema, for every reflecting arm. ≤ 6 concurrent Claude calls. Reflection
  timeout 180 s; a failed/timed-out reflection keeps the old memory and is counted.
- **SurrealDB:** not on the per-reflection path in season runs (latency). Box memory is authoritative. For the
  forgetting demo, the no-forget profiler_plus memory at game 48 of seed 8000 is mirrored to SurrealDB as
  entities `person/<name>` (scope `lab/reg01/<wid>`), then `POST /forget` with `dryRun: true`, then the real
  call (or the soft-delete equivalent the API supports). Responses go in the branch world's `surgery` event
  (optional fields only) and in regulars.json.
- **forget:<name>** (new memory variant): removes every hypothesis about the name, its base-rate cell, its
  calibrator / logreg model, every revealed observation by it (and its obs ids), and its drift notes, from the
  box memory.
- **No-forget comparator:** the parent season's own matches 17–20 (identical state at game 48, learning ON,
  same seeds). Checkpoint-restore exactness is verified with a non-LLM arm before running.

## Seeds, metrics, analysis

- Fresh seeds **8000–8023**. Run order ascending. If the full matrix cannot finish by ~15:05, seeds are cut
  from the top (keep ≥ 12, i.e. at least 8000–8011); never arms or tables. The cut is recorded in `deviations`.
- Win rate: fraction of games p1 wins (last player standing, as in `compute_metrics`), per match.
- Encounter index k = 1..5 per regular (its k-th match). First meeting = k = 1; returns = k ≥ 2.
- Brier: p1's non-forced respond decisions on p3's claims, truth = god-view bluff (claim event).
- Per seed, a metric is averaged over the relevant matches, then paired across arms by seed. Paired bootstrap
  over seeds, 2000 resamples, 95% CI. A claim holds only if its CI excludes 0 (H10 others: CI within ±5 pts).
- No exclusions. Every cell reported. No tuning on test seeds. Losses are reported.
- Outputs: `runs/reg01/regulars.json`, `blab/orchestrator/analyze_regulars.py`.

## Reconstructed from a garbled spec (items the user has not yet confirmed)

- The `surgery` event stores the SurrealDB response in optional fields only (event schema frozen).
- "Brier per arm by encounter index" uses the definition above.

---

## Post-run record (appended 14:46 after the runs; the text above is unchanged from commit 1e43a74)

- All 24 seeds (8000–8023) ran; no seed cut. 7 arms x 24 seasons + 2 x 24 forget branches. 1152 Sonnet 5.5 reflections, 0 errors.
- **H8: does not hold.** Returns win rate, profiler_plus − profiler_plus_noid −0.023 [−0.051, +0.006]; − logreg_pooled +0.032 [+0.007, +0.058].
- **H9 (report only):** profiler_plus − logreg_identity, returns win rate +0.034 [+0.008, +0.059], Brier +0.006 [−0.001, +0.014].
- **H10a (forget:Avery): does not hold.** Avery Δwin −0.069 [−0.208, +0.069]; others' CIs not within ±5 pts. Supplementary Brier Δ: Avery +0.112 [+0.086, +0.140], others within ±0.01 (CIs span 0).
- **H10b (forget:Blake): does not hold.** Blake Δwin −0.083 [−0.194, +0.014]; Brier Δ +0.094 [+0.066, +0.122].
- Relearning: by match 20, every seed had new hypotheses about the forgotten regular (mean 2.7 vs 8.5 before). The same predicate was re-proposed in 13/24 seeds (Avery) and 12/24 (Blake).
- Decision replays: Avery 24/24 shifted (mean |Δp| 0.27, 6 flipped); Casey 24/24 Δp = 0 exactly; every unchanged replay reproduced the logged p_final.
- SurrealDB: semantic /forget dryRun → {deleted: 0} for both names. DELETE /entities/person/<name> soft-deleted Avery (after one 500 + retry) and Blake; Casey and Drew were untouched.
- Deviations: runs/reg01/deviations.json (also in regulars.json).
