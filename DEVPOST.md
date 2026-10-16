# Bluff Lab

<!-- TEMPLATE. Do not edit numbers by hand. Fill with:
     .venv/bin/python -m blab.orchestrator.pick --devpost --run r01   →  DEVPOST.filled.md
     "claim:" placeholders are sentences generated from results.json with their own conditionals (they say
     "no detectable difference" when a CI includes 0); "claim_if:" lines vanish when the effect is not measured. -->

**A controlled experiment in how AI agents learn through memory without retraining, and whether that memory
makes them better or more exploitable.**

## Inspiration

Agents increasingly deal with counterparts they can't trust, and they increasingly carry memory between
encounters. Memory can make an agent sharper. It can also be used against it: poisoned with a planted line
(OWASP Agentic ASI06, memory and context poisoning), or gamed by a counterpart that behaves well until it
doesn't. Real deployments rarely tell you who lied. A bluffing card game does: every challenge reveals the
cards, so we have ground truth for every belief the agent forms.

## What it does

Three players. Each claims the cards they put down face-down are the table rank. The next player either
accepts or challenges, and the challenge reveals who lied. Our agent, **the Profiler**, keeps a typed,
time-aware memory of each opponent: a base bluff rate plus hypotheses written as machine-checkable predicates
("Player 3 drops contractions when bluffing" = `formal_register == true`). Between games, a frozen Claude model
proposes hypotheses (reflection). The numeric layer scores them against later reveals:
provisional → confirmed → refuted or superseded. Only confirmed lines change decisions. Lines injected
without evidence get weight 0.

- **Mind Inspector:** freeze any decision and see the recorded numbers: card math, the memory line used and
  the revealed rounds behind it, then p(bluff) and the action. Nothing is explained after the fact.
- **Memory Surgery:** re-run the *same* recorded decision with memory erased, rolled back, minus one line,
  or with a planted false line, then replay games 11–20 on the same seeds to see whether it helped.
- **The Long Con:** an opponent plays honestly for 5–8 games, then defects. Every memory of it is true; the
  attack is assuming people don't change.
- **Strategy Forge:** the agent compiles its memory into Python, which runs as untrusted code in a
  network-blocked Daytona sandbox behind a static gate.

## What we measured (run `r01`, {{n_worlds}} worlds + {{n_branch_worlds}} ablation branches; 95% bootstrap CIs over seeds)

**The Surgery Reveal.** One recorded decision in world `{{prov:world}}` (seed {{prov:seed}}, reflection:
{{prov:reflection}}), picked by a fixed rule, not by hand (see `show/bundle/PROVENANCE.md`):
- Full memory: p(bluff) {{reveal:variants.0.result.p_final|2}} → {{reveal:variants.0.result.action}}
- Memory erased: {{reveal:variants.1.result.p_final|2}} → {{reveal:variants.1.result.action}}
- False memory planted ("Player 3 never bluffs"): {{reveal:variants.2.result.p_final|2}} → {{reveal:variants.2.result.action}}
- The same false memory, given to a provenance-blind agent: {{reveal:variants.3.result.p_final|2}} → {{reveal:variants.3.result.action}}
- The cards were revealed: bluff = {{reveal:reveal.was_bluff}}.

**Memory mattered.** {{claim:decision_flips}}

**Did it help?** {{claim:erase_hurts}}

**Better beliefs ≠ better decisions.**
- {{claim:calibration_threshold}}
- {{claim:winrate_threshold}}
- {{claim:winrate_ev}}
- {{claim_if:beliefs_not_decisions}}

The SPEC's respond rule (challenge iff p > 0.5) challenges near coin flips. Under a rule that also uses the
agent's remembered cost of playing on (`ev`), the comparison changes as shown above. We report both.

**Does memory help?** {{mem:headline}} Setting: {{mem:setting}}
observations (`{{mem:source}}`), baseline `{{mem:baseline}}`, {{mem:note}}.
{{memclaims:threshold}}

**The strongest statistical baseline (engineered-feature setting, r01).** In r01, observations included a
precomputed `formal_register` feature, which hands every learner the tell's form. The comparison below is
in that setting: {{claim:vs_logreg_threshold}}
{{claim:false_discovery_threshold}}
{{claim:false_discovery_lower_threshold}}
Our honest positioning: a bag-of-words classifier can be the better *predictor*. The Profiler's memory is explicit,
auditable and provenance-aware: every line cites the reveals behind it, you can operate on it (erase, roll
back, remove a line), and by design it gives injected lines with no evidence zero weight.

**The Long Con.** {{claim:longcon_twist}}
{{claim_if:longcon_twist}}

All metric definitions, including the §6 versions and our corrected ones, are in `runs/r01/summary.md`.

## Pre-registered follow-ups (raw observations, fresh seeds; every cell reported)

`PREREG.md` was committed (2167c1a) before any of these worlds ran. The opponent roles were designed after r01.

**Capability boundary (b01; seeds 4000–4023, 7 agents × 3 tables, Profiler with real Claude reflection).**
None of the new hypotheses held. On the relational tell, a sentence-embedding classifier had *lower* Brier than
the Profiler over the first 10 reveals (Δ +0.033, 95% CI +0.009 to +0.057), the opposite of H1. Important
deviation: the Profiler's planned per-claim check of statements against the public record was not built in time,
so H1 tested the Profiler without the capability it was meant to test. On the tell that flips mid-series, the
Profiler did not beat the embedding classifier (Δ +0.024, CI −0.003 to +0.054) or discounted Bayes. H3 was not
testable. The r01 control replicated: with real Claude reflection, `logreg_bow` beat the Profiler on win rate
(+0.067, CI +0.008 to +0.125) and Brier. Full table: `show/bundle/boundary.json`.

**Memory vs no memory, raw observations (mem01; seeds 3001–3012, real Claude reflection, threshold rule).**
No engineered tell features: every agent sees the same raw utterances. `profiler_plus` turns Claude's hypotheses
into features of a small per-opponent logistic model ("the LLM proposes, statistics weigh"). Pooled over four
tables (n = 48 paired worlds), against a bag-of-words logistic classifier: win rate +0.108 (95% CI +0.066 to
+0.150), Brier −0.011 (CI −0.020 to −0.002). Against the no-memory card counter: win rate +0.076 (CI +0.040 to
+0.111). The same agent with reflection switched off only ties the classifier (win rate +0.014, CI −0.023 to
+0.048), so the gain comes from Claude's hypotheses. Caveats: the opponents and `profiler_plus` were designed
after r01, the reflection-off control was added after pre-registration, the original `profiler` still loses to
the classifier (win rate −0.080), and this comparison is against word counts, not sentence embeddings (the
embedding classifier, built later for b01, beat the Profiler there). In a same-model test (Sonnet 5.5, only the
memory block of the prompt differs, 4 seeds per table), a hypothesis dossier improved win rate over no memory
by +0.064 (CI +0.009 to +0.127) and a raw transcript showed no detectable effect.

**The Long Con at scale (lc02; seeds 6000–6119).** Sleeper bluffs let through after the turn (threshold rule,
mean per world): counter 76.5, naive bayes 45.1, discounted bayes 30.5, `logreg_bow` 26.3, Profiler 37.9 (canned
reflection) and 32.7 (real Claude, n = 24). Naive memory let through fewer bluffs than no memory, so we do not
claim "naive memory is more exploitable". Within the same agent, though, memory of earned trust did make it more
exploitable: restarting each Profiler world from a checkpoint just before the turn with its memory erased cut
the bluffs it let through by 17.1 (CI 12.8 to 21.3; paired over 120 seeds). These are checkpoint branches;
a live Daytona VM fork was not available for our org. Full table: `show/bundle/longcon2.json`.

**The Regulars (reg01; seeds 8000–8023, four named opponents who each return 5 times; `PREREG_reg01.md` committed first).** On return encounters, identity-keyed `profiler_plus` beat a pooled classifier (win rate +0.032, CI +0.007 to +0.058) and a per-name classifier (+0.034, CI +0.008 to +0.059), but not the same agent with all regulars pooled (−0.023, CI −0.051 to +0.006), so the identity hypothesis H8 does not hold.
Surgical forgetting: `forget:Avery` changed every first post-forget decision about Avery (24/24, mean |Δp| 0.27, 6 actions flipped) and left every decision about Casey bit-identical (24/24, Δp = 0), and the forgotten regular's Brier got worse (+0.112, CI +0.086 to +0.140) while the others' did not move; the pre-registered win-rate test did not hold (Avery −0.069, CI −0.208 to +0.069; the others' CIs are wider than ±5 pts). Full table: `show/bundle/regulars.json`.

## How we built it

- **Daytona:** world boxes (4 vCPU) run many seeded worlds concurrently behind a FastAPI server. Each world
  writes an append-only event log plus memory checkpoints, and supports `/replay` and `/branch`.
  Ephemeral forge boxes run with `network_block_all` and execute AI-written strategies.
- **Anthropic Claude (Opus 5.5 via AWS Bedrock):** the reflection step between games returns structured
  hypotheses (JSON predicate DSL, never executed as code). Claude also generates the Strategy Forge code.
- **SurrealDB Agent Memory:** a mirror of the belief graph. Rewriting a belief supersedes the old one, which
  keeps its `valid_until` and history. A planted false line is flagged by the Profiler: no evidence behind it.
  (SurrealDB did not raise an `uncertainty` record for it in our spike, so we do not claim one.) Retrieval feeds reflection.
- **Arize AX:** spans for decisions, reflections, forge steps and surgery, plus two sets of experiments over
  frozen r01 decisions (telltale table, threshold rule), each with a Brier evaluator.
  - One experiment per agent, each agent scored on its own revealed decisions (n differs; 1,994 in total).
    Mean Brier: logreg_bow {{arize:logreg_bow|3}}, profiler {{arize:profiler|3}}, bayes {{arize:bayes|3}},
    counter {{arize:counter|3}}.
  - Memory Surgery: same 555 decisions, only memory changes. Each Profiler decision is replayed in-process
    (no LLM) with full memory, erased memory, a planted false memory, and the planted memory given to a
    provenance-blind agent. Mean Brier (actions flipped vs full): full {{arize_surgery:full|3}}, erased
    {{arize_surgery:erased|3}} ({{arize_flips:erased}}), false memory {{arize_surgery:false-memory|3}}
    ({{arize_flips:false-memory}}), provenance-blind {{arize_surgery:provenance-blind|3}}
    ({{arize_flips:provenance-blind}}). Ids are in `show/bundle/arize_experiment.json`.
- **ElevenLabs:** the Profiler's interrogation line ("You've gone very formal all of a sudden.") is voiced
  from the real `ask` event.

## Honesty notes

- The demo is a recorded replay of one real simulated world, picked by rule (near the median discovery game).
- Every number above comes from `runs/r01/results.json`. This file is generated from it.
- r01 is the engineered-feature control: every agent was given a precomputed `formal_register` feature, so a
  simple classifier is expected to win there, and it did (`logreg_bow` beat the Profiler on the engineered
  `telltale`). The opponent roles `tell_relational`, `tell_flipper` and `answer_tell` were designed **after**
  that result, to test where LLM memory should help. They are pre-registered in `PREREG.md` (committed before
  any of those worlds ran, on fresh seeds) and every cell is reported, including ties and losses.
- Strategy Forge: 2 of 4 telltale forges were promoted (0 of 1 on longcon); outcomes vary a lot between
  generations from the same memory. The Show displays the earliest forge for the picked world, chosen without
  looking at its result, alongside that tally.
- The tells are planted. This is a controlled experiment, and ground truth requires them. The question is
  whether the agent finds them without being told which feature matters.

## What's next

Learn the decision rule as well as the beliefs; vary tell strength to find where discovery stops; shared
dossiers between agents; collusion detection.
