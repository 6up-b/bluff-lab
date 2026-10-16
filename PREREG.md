# PREREG — capability-boundary experiment (b01) and Long Con at scale (lc02)

Committed before any b01 or lc02 world is run. This commit's hash and timestamp are the evidence.

Capability-boundary experiment (designed after r01, where `logreg_bow` beat the Profiler on the engineered, lexical, stationary `telltale`).

- Info mode: RAW for all agents. No `formal_register` or other engineered tell features. Every agent sees the same raw utterance and answer text plus the public log.
- Seeds: 4000–4023 (fresh). Respond rule: threshold (primary). n = 24 paired seeds per cell.
- Agents: counter, bayes, bayes_discount, logreg_bow, logreg_embed, profiler (Claude reflection), profiler_amnesic.
- Tables: [tell_relational, control], [tell_flipper, control], [answer_tell, control].

Hypotheses:
- H1 (relational): on tell_relational, profiler's Brier over the first 10 revealed claims by the tell opponent is lower than logreg_embed's and logreg_bow's.
- H2 (shifting): on tell_flipper, profiler's Brier over the 10 reveals after the flip is lower than logreg_embed's, and not worse than bayes_discount's.
- H3 (answer): on answer_tell, profiler needs fewer revealed claims to discovery than logreg_embed.
- H4 (control replication): on the r01 engineered telltale, logreg_bow ≥ profiler (already observed; reported as the control).

Analysis:
- Paired bootstrap over seeds (2000 resamples), 95% CI.
- A claim "holds" only if the CI excludes 0.
- No exclusions. All cells reported. Claude and stub never pooled.

## Filled-in details

- **Seed substitution (deviation from the drafted text, made before running):** the draft said seeds 1000–1023 for b01 and 2000–2119 for lc02. Those are not fresh: seeds 1001–1030 were used to tune the hypothesis refute rule, and seeds 2001–2030 were used to decide the drift threshold (see INTERFACES.md "Status: core"). Seeds already used anywhere in this project: 1–130, 1001–1030, 2001–2030, 2999, 3001–3030. Fresh ranges used instead:
  - b01 (capability boundary): **4000–4023**
  - lc02 (Long Con at scale): **6000–6119** (real-Claude Profiler on 6000–6023)
- Runs: `runs/b01` (matrix `matrix_boundary.yaml`, analysis `blab/orchestrator/analyze_boundary.py`), `runs/lc02` (matrix `matrix_longcon2.yaml`, analysis `blab/orchestrator/analyze_longcon2.py`).
- Disclosure: the opponent roles `tell_relational`, `tell_flipper` and `answer_tell` were designed after r01 showed `logreg_bow` winning on the engineered `telltale`. They test where LLM memory should help; they are not a neutral sample of opponents.
- For `tell_relational`, the Profiler may use a generic per-claim LLM check ("Does any factual statement in this utterance contradict the public game record below? true / false / none"), which is not told about the tell. Disclosure: the LLM reads words against the record; the statistical baselines cannot. That is the capability under test.
- For `answer_tell`, the engine asks the tell opponent the same neutral question on every claim in every world, and the answer text is public, so every agent receives identical information.
- `logreg_embed` hyperparameters are fixed a priori: 64-dim fixed random projection (seed 0) of sentence embeddings + raw numeric features, online L2 logistic regression, λ = 1.0, lr = 0.1. No tuning on test seeds.

## Long Con at scale (lc02)

- Agents: counter, bayes, bayes_discount, logreg_bow, profiler (stub reflection on 6000–6119; real Claude reflection on 6000–6023, analyzed separately), profiler_blind; logreg_embed if available before the run starts.
- Metrics, defined before running: detection rate; median RTD among detected worlds; Kaplan-Meier median RTD with undetected worlds censored at game 20; false alarm = belief change on the falsealarm table (`false_alarm_belief`); exploitation = sleeper bluffs let through after G₀.
- Respond rule: threshold (primary); ev secondary if time allows. All cells reported.

---

## Post-run record (appended after the runs; the text above is unchanged from commit 2167c1a)

**b01 outcome:** all 504 cells ran (24 seeds × 3 tables × 7 agents). No b01 hypothesis held; H1a held in the
opposite direction (logreg_embed had lower Brier than the Profiler on tell_relational). H4 (control) held for
Claude reflection on win rate and Brier. Full numbers: `show/bundle/boundary.json`.

**Deviations, recorded honestly:**
- The Profiler's per-claim LLM contradiction check (`llm_contradiction`) was not built in time. On
  `tell_relational` the Profiler therefore had no way to read statements against the public record, which is the
  capability H1 was meant to test. H1 as run tests the Profiler without it.
- H3 is not testable: `logreg_embed` has no discovery statistic.
- H2's "not worse" had no pre-registered margin; it was scored strictly (CI excludes 0).
- b01 ran on the laptop process pool, not box1.
- The shared reflection prompt in `blab/orchestrator/llm.py` leaked the engineered tell (`formal_register`, the
  `no_contractions` op, and a formal-speech example). b01's runner patched it in-process and asserted no leak
  (reflections tagged `raw_patched`); the permanent fix is commit f869ce6.

**lc02 outcome:** see `show/bundle/longcon2.json`. The planned Daytona VM fork was NO-GO (no VM regions for this
org; container sandboxes do not support fork). The fallback restarts from checkpoints and is called a
"checkpoint branch", never a fork (`longcon_fork.json`, `real_vm_fork: false`).
