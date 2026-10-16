# Show bundle provenance (honesty record)

Built 2026-10-16T17:05:15Z by `blab pick-demo --run local` from real event logs. No number in this bundle was written by hand.

- **Run:** `local`
- **World:** `telltale-profiler-s1` (`runs/local/telltale-profiler-s1`)
- **Seed:** 1
- **Table:** telltale
- **Decision seq:** 1043 (game 10, round 4)
- **Claim seq:** 1042, **reveal seq:** 1045

## Selection rules

- World: profiler world on the telltale table (else longcon) whose verbal_tell discovery game is closest to the median discovery game over those worlds; ties broken by lowest seed.
  - candidates: 3; discovery games: [5, 7, 7]; median: 7.0; picked world's discovery game: 7
- Decision: first p1 respond decision (by seq) in games 10–14 on a formal_register claim where the logged (full-memory) action is challenge, the reveal shows a bluff, and replay with as_of:0 gives play_on; decisions that used a confirmed evidence-backed verbal_tell line are tried first.
  - exact match: **True** — 11 decisions met the cheap criteria (11 used a confirmed verbal_tell line); first exact match at tier 'uses a confirmed verbal_tell line' used
  - criteria: {"games_10_14": true, "formal_register": true, "about_telltale": true, "full_challenges_and_right": true, "not_explore": true, "as_of0_flips_to_play_on": true}

## Surgery variants (real `replay_decision` outputs)

- `full`: p_final 0.6922, action challenge, flipped False
- `as_of:0`: p_final 0.3158, action play_on, flipped True
- `false_memory`: p_final 0.6922, action challenge, flipped False
- `profiler_blind`: p_final 0.0923, action play_on, flipped True

`full` replay matches the logged decision: **True**

## Files

- `world.jsonl`: events of telltale-profiler-s1
- `twin.jsonl`: events of telltale-bayes-s1 (bayes, same seed 1)
- `dossier.json`: memory checkpoints g00–g20 of telltale-profiler-s1
- `ablation.json`: `blab analyze --run local` output
- `longcon.json`: `blab analyze --run local` output
- `results.json`: `blab analyze --run local` output
- `reveal.json`: Surgery Reveal decision + replay_decision outputs
- `cues.json`: director script generated from the picked seqs (show/FLOW.md order)
