# Show bundle provenance (honesty record)

Built 2026-10-16T17:29:06Z by `blab pick-demo --run r01` from real event logs. No number in this bundle was written by hand.

- **Run:** `r01`
- **World:** `w0002` (`runs/r01/w0002`)
- **Seed:** 2
- **Table:** telltale
- **Respond rule:** threshold
- **Reflection:** claude
- **Decision seq:** 1026 (game 10, round 4)
- **Claim seq:** 1023, **reveal seq:** 1028

## Selection rules

- World: profiler world on the telltale table (else longcon) whose verbal_tell discovery game is closest to the median discovery game over those worlds; ties broken by lowest seed.
  - pool: claude reflection, threshold rule; candidates: 12; discovery games: [4, 7, 7, 7, 9, 9, 9, 10, 10, 10, 14, 16]; median: 9.0; picked world's discovery game: 9
- Decision: first p1 respond decision (by seq) in games 10–14 on a formal_register claim where the logged (full-memory) action is challenge, the reveal shows a bluff, and replay with as_of:0 gives play_on; decisions that used a confirmed evidence-backed verbal_tell line are tried first.
  - exact match: **True** — 8 decisions met the cheap criteria (8 used a confirmed verbal_tell line); first exact match at tier 'uses a confirmed verbal_tell line' used
  - criteria: {"games_10_14": true, "formal_register": true, "about_telltale": true, "full_challenges_and_right": true, "not_explore": true, "as_of0_flips_to_play_on": true}

## Surgery variants (real `replay_decision` outputs)

- `full`: p_final 0.7570, action challenge, flipped False
- `as_of:0`: p_final 0.4285, action play_on, flipped True
- `false_memory`: p_final 0.7570, action challenge, flipped False
- `profiler_blind`: p_final 0.2418, action play_on, flipped True

`full` replay matches the logged decision: **True**

## Files

- `world.jsonl`: events of w0002
- `twin.jsonl`: events of w0146 (bayes, same seed 2)
- `dossier.json`: memory checkpoints g00–g20 of w0002
- `ablation.json`: `blab analyze --run r01` output
- `longcon.json`: `blab analyze --run r01` output
- `results.json`: `blab analyze --run r01` output (per-world rows omitted)
- `memory.json`: 'does memory help?' comparison from runs/r01 (engineered observations)
- `stack.json`: sponsor-stack facts from run artifacts (ids/trace ids only, no secrets)
- `reveal.json`: Surgery Reveal decision + replay_decision outputs
- `cues.json`: director script generated from the picked seqs (show/FLOW.md order)
