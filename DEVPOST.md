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

## How we built it

- **Daytona:** world boxes (4 vCPU) run many seeded worlds concurrently behind a FastAPI server. Each world
  writes an append-only event log plus memory checkpoints, and supports `/replay` and `/branch`.
  Ephemeral forge boxes run with `network_block_all` and execute AI-written strategies.
- **Anthropic Claude (Opus 5.5 via AWS Bedrock):** the reflection step between games returns structured
  hypotheses (JSON predicate DSL, never executed as code). Claude also generates the Strategy Forge code.
- **SurrealDB Agent Memory:** a mirror of the belief graph. Rewriting a belief supersedes the old one, which
  keeps its `valid_until` and history. A planted false line is flagged by the Profiler: no evidence behind it.
  (SurrealDB did not raise an `uncertainty` record for it in our spike, so we do not claim one.) Retrieval feeds reflection.
- **Arize AX:** spans for decisions, reflections, forge steps and surgery.
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
