# PREREG c01: a stronger Profiler, tested fairly

Committed on main before any c01 test-seed world is run. This commit's hash and timestamp are the evidence. The
code is commit `eeb06de` on branch `exp/confirm`; it is frozen for the run except for bug fixes, which will be
listed as deviations. b01, r01 and mem01 stay exactly as reported, losses included. Nothing is re-run on their
seeds.

## Seeds

- **Test: 5000–5023** (fresh, 24 seeds). Seeds already used anywhere: 1–130, 1001–1030, 2001–2030, 2999,
  3001–3030, 4000–4023, 6000–6119. None of them is used by c01.
- **Dev: 7000–7099.** Used for the leak audit, the population prior and smoke tests only. No parameter is tuned
  on test seeds.

## Step 1: leak audit of the b01 relational tell (dev seeds, done before this commit)

Question: can "this statement is false" be predicted from the utterance ALONE, with no public record? Method:
L2 logistic regression (λ = 1.0, set a priori) on 256 hashed unigram+bigram BoW features plus logreg_embed's
64-dim projected sentence embedding, 5-fold CV grouped by seed. AUC of the out-of-fold scores, with a
2000-resample bootstrap over seeds for the 95% CI. Leak iff the CI lower bound is > 0.55.
Script: `scripts/c01_audit.py`. Data: p1 = counter, raw, 20 games, seeds 7000–7099.

| role | statements | AUC | 95% CI | leak |
|---|---|---|---|---|
| b01 `tell_relational` (full utterance) | 24,801 | 0.667 | [0.660, 0.674] | **yes** |
| c01 `tell_relational_v2` (full utterance) | 18,776 | 0.557 | [0.547, 0.565] | no (lower bound < 0.55) |
| c01 `tell_relational_v2` (statement sentence only) | 18,776 | 0.501 | [0.491, 0.511] | no |

**The b01 cell leaked through wording.** By kind: count statements 0.63, hand-size statements 0.69, same/different
rank statements 0.68. Causes: the "same"/"different" wording changes with truth while P(same rank) = 1/3; ±1
perturbations at range boundaries make some numbers (for example, 5 cards left after a play) impossible from the
utterance alone. This is an audit note on b01, not a re-run. It is logged in `runs/b01/boundary.json`
`deviations`. In b01, logreg_embed's H1 advantage on tell_relational may partly come from this leak.

**Rebuilt tell, `tell_relational_v2`** (`blab/agents/scripted.py`, `banks.REL2_PREV`):
- Utterance = claim line (formal or casual bank, 50/50, independent of bluffing) + ONE statement naming the
  previous round's table rank: "Last round's rank was {R}." / "The previous round was {R}." / "Last round we
  played {R}." / "The round before this one was {R}." The template is drawn independently of truth.
- False w.p. 0.7 when bluffing and 0.1 when honest (same as b01). A false statement names one of the two other
  ranks, uniformly. Ranks are iid uniform, so the named rank is uniform and independent of the current rank in
  both cases: the wording is identical and only the rank word differs. There are no negation forms.
- Round 1 of a game has no previous round in the game, so it has the claim line only.
- The residual 0.557 AUC of the full utterance comes from the claim line's count (strategic bluffs are 1–2
  cards), which every agent sees as `claim_count`. The statement alone is at chance (0.501).
- Table `tell_relational_v2` = [control, tell_relational_v2]. Embeddings for the 12 new sentences were added to
  `blab/data/embed_cache.json` (same model). logreg_embed therefore sees every utterance.

## Step 2: per-claim record check `llm_record_check` ∈ {"true","false","none"}

- Before each respond decision of profiler_plus_c01, profiler_hybrid and profiler_plus_noreflect, Claude Sonnet
  (`us.anthropic.claude-sonnet-5-5`, Bedrock, effort low, structured output) gets the GENERIC question "Does any
  factual statement in the pending utterance contradict the public record? true / false / none". It is given the
  utterance and the public record only: core's `public_obs_text` over the current game, with the private lines
  removed (our hand, legal actions and the hand-derived p_cards). The prompt never mentions any tell. System
  prompt: `blab/agents/c01.py` `RC_SYSTEM`.
- Cached by sha256(utterance, public text) in a run-wide cache shared by all arms. Identical states therefore get
  identical answers, and replay is exact. Every call and cache hit is logged (tokens, latency) in
  `runs/c01/rc_log.jsonl`.
- **Skip rule (a priori, to save cost):** a single-sentence utterance is the claim itself, which the public
  record cannot contradict (the cards are hidden; card math is already in p_cards). It gets "none" without a
  call. In practice only `tell_relational_v2` utterances (round ≥ 2) have two sentences.
- A failed call gets "none" and is counted.
- It is a DSL feature (`hypotheses.FEATURES["llm_record_check"]`). The c01 reflection prompt is the clean raw
  prompt `STABLE_SYSTEM_RAW` plus one neutral feature-table row describing it (`STABLE_SYSTEM_RAW_RC`, selected
  only when a raw payload carries `record_check: true`). It is attached to every judged claim's features, so it
  appears in revealed claims and in reflection payloads.
- **Disclosure:** the statistical baselines (logreg_bow, logreg_embed, bayes_discount_pp) do not get it. Reading
  words against the record is the capability under test.

## Step 3: population prior (dev seeds)

The pooled grid posterior over q from CONTROL opponents' revealed claims (seat p2, role control) on
tell_relational_v2, tell_flipper and tell_conjunctive. Settings: p1 = counter, raw, 20 games, seeds 7000–7099.
That is 15,532 claims (8,200 bluffs). It is tempered to **N0 = 10 effective claims** (each likelihood raised to
10/15,532; N0 set a priori, about one opponent's first two games of reveals). Result: mean 0.32, 95% [0.03, 0.68].
Committed as `blab/data/c01_pop_prior.json` (script `scripts/c01_pop_prior.py`). It is applied identically
wherever a Bayesian-family agent has a prior:
- grid agents start from it instead of uniform (`bayes_discount_pp`; bayes and profiler would too, but they
  are not c01 arms);
- logistic calibrators (profiler_plus_c01, profiler_hybrid, profiler_plus_noreflect) use the offset
  logit(p_cards + (1 − p_cards)·E[q]) with E[q] = 0.32, in place of logit(p_cards). The prior coefficient is 1,
  as before.

## Arms (exact definitions; code `blab/agents/c01.py`, `blab/agents/ours.py`)

All arms: p1 seat, `info: "raw"`, threshold respond rule (challenge iff p_final > 0.5, else ε = 0.1 exploration
with the recorded rng_u), the same play rule, 20 games, reflection by Claude Opus (`us.anthropic.claude-opus-5-5`,
Bedrock, effort low) through `reflect.local_reflector`, raw prompt.
- `logreg_bow`: unchanged (r01/b01 definition).
- `logreg_embed`: unchanged (b01 definition, λ 1.0, lr 0.1).
- `bayes_discount` → **`bayes_discount_pp`**: bayes_discount (γ = 0.85) starting from the population prior.
- `profiler_plus` → **`profiler_plus_c01`**: mem01's profiler_plus, a per-opponent L2-MAP logistic calibrator
  whose features are the base numeric features plus one binary feature per usable hypothesis. Prior N(w0, I),
  likelihood weights 0.9^(games ago), exact refit at game end. The holdout of the proposing batch is kept, and
  untrusted lines get weight 0 unless independently confirmed. Changes from mem01:
  (a) the population-prior offset;
  (b) llm_record_check attached to claim features, so reflection may propose predicates on it (it is NOT a
  built-in calibrator feature);
  (c) reflection after games **2, 4, 8 and 12** only (mem01 reflected after every game). This is compute-driven
  and set a priori: 20 Opus reflections × 192 worlds do not fit before the deadline.
- **`profiler_hybrid`**: profiler_plus_c01 plus two built-in calibrator features, rc_contradiction =
  [llm_record_check == "true"] and rc_consistent = [llm_record_check == "false"], each with prior N(0, 1) like
  every non-offset coefficient, plus full memory semantics:
  - hypotheses are stored with provenance;
  - untrusted lines get weight 0 unless independently confirmed;
  - **supersession** (`hypotheses.supersede`: the newest confirmed line per (about, aspect) closes older ones,
    which leave the calibrator), applied after scoring and after each reflection, which profiler_plus does
    not do;
  - memory-surgery and forgetting variants (`replay.py` full / as_of / minus / false_memory / transplant);
  - replay-exact (checked on a dev world: 0 mismatches).
  Regularization: prior variance 1.0 and γ = 0.9, set a priori and identical to profiler_plus. There is no
  learning rate: the MAP is refit exactly by damped Newton at every game end.
  **Difference from profiler_plus_c01:** the record check is a built-in calibrator feature, and supersession is on.
- **`profiler_plus_noreflect`**: profiler_hybrid with reflection off (no hypotheses), i.e. the calibrator on base
  features + population-prior offset + built-in record-check features. It isolates what Claude reflection adds.

## Tables

tell_relational_v2 (rebuilt; replaces tell_relational because the audit failed), tell_flipper, tell_conjunctive,
longcon. Every table × arm × seed is one world (4 × 6 × 24 = 576 worlds).

## Hypotheses (verbatim)

- **H11:** profiler_hybrid's Brier on the first 10 reveals is lower than logreg_embed's, on relational and on
  flipper (two tests).
- **H12:** profiler_hybrid's win rate is higher than logreg_bow's, pooled.
- **H13:** profiler_plus's win rate is higher than logreg_bow's, pooled (replication of mem01).

Operationalization:
- "First 10 reveals" = p1's first 10 respond decisions on claims by the tell opponent (p3) whose cards were
  revealed. Brier = mean (p_final − was_bluff)²; it uses fewer than 10 if fewer exist (as in
  analyze_boundary).
- "Relational" = tell_relational_v2.
- "Pooled" win rate = per seed, the mean over the 4 tables of (arm − logreg_bow) world win rates. Bootstrap
  over seeds.
- H13 uses profiler_plus_c01 (the definition above), so it replicates mem01's claim with the c01 version. Exact
  identity is impossible because of the reflection schedule and the population prior.

## Analysis (`blab/orchestrator/analyze_confirm.py` → `runs/c01/confirm.json`)

- Paired by seed. 2000-resample bootstrap over seeds, 95% percentile CI. A hypothesis **holds iff the CI excludes
  0 in the predicted direction**. No exclusions. All cells are reported (per-cell win rate, Brier, Brier on the
  first 10 reveals, paired Δs vs logreg_bow and logreg_embed for every arm).
- **Stopping rule (pre-registered, compute-driven, independent of results):** phase A (no-LLM arms) runs all
  seeds. Phase B queue order: for seed 5000→5023, for each table: profiler_hybrid, profiler_plus_c01; then
  profiler_plus_noreflect on tell_relational_v2 for every seed. No world is launched after **15:30**, and the run
  is stopped at **15:42** at the latest. A hypothesis uses every seed for which all of its cells are complete.
  Seeds and cells that did not run are reported as "not run" with their count. No result is looked at before
  the run stops.
- Concurrency: 8 worker processes, one world each at a time, so at most 8 concurrent Claude calls (shared
  Bedrock).
- Reflection failures and timeouts keep the old memory, as in every previous run, and are counted. Record-check
  failures give "none" and are counted.
- The record check's call count, cache hits, tokens and estimated cost are reported.

---

## Post-run record (appended 15:32; the text above is unchanged from commit 7a0cd49)

All 576 worlds ran (24 seeds × 4 tables × 6 arms), finishing at 15:30:06 with 0 failures. All 768 Opus
reflections succeeded. The stopping rule was not triggered. Full numbers are in `runs/c01/confirm.json` (copy:
`show/bundle/confirm.json`).

| | Δ (a − b) | 95% CI | holds |
|---|---|---|---|
| H11a relational_v2, Brier first 10: hybrid − logreg_embed | +0.008 | [−0.017, +0.033] | no |
| H11b flipper, Brier first 10: hybrid − logreg_embed | +0.014 | [−0.009, +0.039] | no |
| H12 pooled win rate: hybrid − logreg_bow | +0.080 | [+0.054, +0.107] | **yes** |
| H13 pooled win rate: profiler_plus_c01 − logreg_bow | +0.081 | [+0.057, +0.104] | **yes** |

Exploratory, added after the run started (the lead's request arrived after test-seed worlds had run):
- H14 hybrid − logreg_embed, pooled win rate: +0.158 [+0.134, +0.184].
- H15 profiler_plus_c01 − logreg_embed, pooled win rate: +0.159 [+0.134, +0.183].
- Whole-world Brier for hybrid − logreg_embed is negative on all 4 tables (−0.020 to −0.039, every CI excludes 0).

Record check: 6,102 Sonnet calls and 589 cache hits. 158 calls failed and were scored "none". Tokens: 3.0M
input, 0.23M output, 3.7M cache-read. Estimated cost about $9 at first-party rates. Accuracy on test decisions
was 0.80–0.83 (descriptive), with errors mostly false "contradiction" answers.

Deviations: see `confirm.json` `deviations`.
