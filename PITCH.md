# Bluff Lab — 3-minute pitch script (read-along / rehearsal)

**How to use:**
- **Bold bracketed** lines are keypresses on the Show (`Space` / `→` = next cue). Cue numbers match the frozen
  `show/bundle/cues.json` (32 cues, 0–31).
- *Italic* lines are stage directions. Everything else is spoken.
- Target ~150 words per minute. Times on the right are cumulative.

**Numbers:**
- Every number below was read from the frozen bundle (Show world `r01/w0002`, seed 2, decision #1026; plus
  `memory.json` = mem01, `embed_exploratory.json` = mem01 vs sentence embeddings (exploratory), `longcon2.json` = lc02, `boundary.json` = b01, `forge.json`, `arize_experiment.json`).
- Never say a number that isn't on the screen or in this file. (The screen shows probabilities as decimals, 0.24; saying "24%" is the same number.)
- **Prompt caveat:** the Show world comes from r01, whose reflection prompt contained a formal-speech example.
  So on the table scenes we never say the agent found the tell on its own. The "found it from raw text" claims
  belong to mem01 (clean prompt) only.

---

## Setup (before you're called)
- Serve the Show offline: `cd show && python3 -m http.server 8790`, open `http://127.0.0.1:8790/index.html`, press `F` (fullscreen).
- Press `R` to reset, and `H` if you want the presenter HUD.
- Check audio through the room speakers: cue 4 (the Profiler's first question) and cue 8 (the ElevenLabs line).
  Cues 2, 3 and 7 also play the opponents' voices.
- Keep the backup video open in another window.

---

## Script

**[cue 0 — title]** · 0:00–0:12
> Agents are starting to carry memory from one encounter to the next. Memory can make them sharper, or it can be used against them. Bluff Lab is a controlled experiment that measures which.

**[cue 1 — the deal]** · 0:12
> Three players, five cards each. This round, the table rank is Kings.

**[cue 2 — Player 2 claims]** *(voice plays)*
> You lay cards face-down and claim they match. You're allowed to lie.

**[cue 3 — Player 3 claims]** *(voice plays)*
> The next player can accept and play on…

**[cue 4 — the Profiler asks]** *(voice plays: "Are you sure about that?")*
> …ask a question…

**[cue 5 — the call]**
> …or call it.

**[cue 6 — the reveal]** · 0:35
> The cards flip, and whoever was wrong loses a life. Every challenge gives us ground truth: we know exactly who lied.

**[cue 7 — game 10, the claim]** *(voice plays)*
> Game ten. Player 3: "The cards are one King."

**[cue 8 — the ask]** *(let the voice play: "You've gone very formal all of a sudden.")* · 0:55
> That's our agent, the Profiler. Same frozen model all day, no fine-tuning. Its memory, written between games and checked against the reveals, says this player turns formal when bluffing.

**[cue 9 — freeze]**
> Let's stop time and look inside.

**[cue 10 — Mind Inspector]**
> Card math alone says 24% this is a bluff. Its memory holds one confirmed line: Player 3 speaks formally when bluffing. Eleven reveals where it applies, a bluff rate of 68% versus 22% otherwise. That pushes it to 76%, so it challenges.

**[cue 11 — evidence]** · 1:20
> These are the numbers the decision actually used, recorded at decision time, and the rounds that proved them.

**[cue 12, cue 13 — dossier]** *(cut first if you're running long)*
> The file it built, game by game. Lines get confirmed, refuted, or superseded, and nothing is silently overwritten.

**[cue 14 — surgery: full]** · 1:35
> Now, surgery. Same state, same claim, same model. Only the memory changes.

**[cue 15 — erased]**
> Erase the memory: 43%. It plays on.

**[cue 16 — false memory]**
> Plant a false memory, "Player 3 never bluffs," with no evidence behind it. Ours gives it no weight: still 76%, still challenges.

**[cue 17 — provenance-blind]**
> An agent that trusts every line it's handed: 24%. It plays on.

**[cue 18 — flip]** *(pause one beat after the flip)* · 1:55
> It was a bluff.

**[cue 19 — ablation]** · 2:05
> One flipped decision doesn't prove memory helps. So we replayed games eleven to twenty on the same seeds, memory frozen at game ten, twelve worlds with real Claude reflection. Erase it, and the win rate falls from 23% to 9%. But memory from game four, six or eight did about as well as the full file. More memory isn't better. *Trusted* memory is.

**[cue 20 — long con]** *(press through)*

**[cue 21 — long con at scale]** *(press through; numbers are for Q&A)*

**[cue 22 — checkpoint branches]** · 2:20
> Memory can also be used against you. An opponent plays honestly for games, then turns. We restored each of 120 worlds from just before the turn and erased what the agent had learned: it let through 17 fewer of the betrayer's bluffs. Every memory was true. The attack is assuming people don't change.

**[cue 23, 24, 25 — split screen, results, beliefs vs decisions]** *(press through)*

**[cue 26 — does memory help?]** · 2:35
> The cleanest test: same model, same prompt, same raw table talk, only the memory differs. A raw transcript of past rounds: no detectable gain. A file of hypotheses: +6 points. And when Claude proposes the hypotheses and a small statistical model weighs them: +19 points over no memory. Three to four seeds per table, so treat that as early.

**[cue 27 — capability map]** · 2:48
> And where we lose: on our pre-registered boundary test, a sentence-embedding classifier beat us. The piece that checks what a player says against the record wasn't built in time. That's the next run.
> *(If c02 has landed with `holds` true, add one sentence from `runs/c02/confirm.json`; otherwise say nothing more.)*

**[cue 28, 29 — how it's built]** *(cut second if running long)*
> Daytona boxes run the worlds in parallel, Claude reflects between games, SurrealDB keeps every superseded belief, Arize traces every decision and replays memory surgery as experiments, and ElevenLabs gives the table its voices.

**[cue 30 — forge]** *(cut first if over time)*
> It can also compile its memory into code, run as untrusted in a network-blocked sandbox. On the Telltale table, two of four forges beat the agent that wrote them; the one shown won 60% to 35%. The one Long Con forge didn't.

**[cue 31 — close]** · 3:00
> The question isn't how much an agent remembers. It's which memories it should trust.

*(Stop. Smile. Wait for questions.)*

---

## If you're over time (cut in this order)
1. Forge (cue 30): press through.
2. Dossier (cues 12–13): press through.
3. "How it's built" (cues 28–29): one sentence: "Daytona, Claude, SurrealDB, Arize and ElevenLabs each run one layer."
4. Ablation second half: end on "falls from 23% to 9%."
5. Cold open: combine cues 1–5 into "Claim your cards match; you can lie; the next player plays on, asks, or calls it."

## LIVE FINALS ONLY: The Regulars (only if `runs/reg01/regulars.json` shows `holds: true`)
> *(Fill from the file; if H8 or H10 didn't hold, skip this beat entirely.)* It also remembers *who* it's playing.
> Against a returning regular it plays [+N from regulars.json] better than without identities. Tell it to forget
> Avery: against Avery it drops [Δ]; against everyone else the change stays within ±5 points.
> *(When included, cut forge and "how it's built".)*

## If something breaks
- **No audio:** read the lines yourself. Cue 8: "You've gone very formal all of a sudden."
- **The Show freezes or the projector fails:** play the backup video and narrate with this script.
- **Asked "is this live?":** "The table is a replay of one recorded world, picked by a fixed rule. Every number comes from real simulated worlds (5,916 in the main run alone), and the code is public."

---

## Q&A prep (answer in one or two sentences, then stop)

**"Didn't a classifier beat you?"**
Yes, in places, and we report it. We dropped the word-count classifier from the talk: in our first run it was handed the tell feature. On raw text we beat it anyway; full numbers are in the writeup. Against a sentence-embedding classifier on the same raw-text seeds, the Profiler-plus agent won by +19.3 points, but that's exploratory: we chose that baseline after the run, and it's the weaker of our two classifiers on those tables. On the pre-registered relational tell (b01), embeddings beat us. The confirmatory rerun against embeddings is c02 (pre-registered, fresh seeds).

**"Your agent loses to the no-memory counter."**
The original Profiler does, in r01, under the spec's rule (−10.5 points). It over-challenges near coin flips. Under a cost-aware rule it wins by +3.1 [+0.5, +5.8]. The Profiler-plus agent beats the counter on raw text: +7.6 [+4.0, +11.1].

**"Did your prompt give it the answer?"**
In the first run, yes: the reflection prompt's example was the formal-speech tell. We caught it, built a clean raw-mode prompt, made raw mode reject those features, and labelled every older world. That's why we don't claim the Show world found the tell unaided. The raw-text results (mem01, b01) use the clean prompt.

**"Your control was added after pre-registration."**
Yes, we say so. Same agent, same seeds, reflection on vs off: Claude's hypotheses add +9.5 points of win rate [+5.9, +13.2] and improve Brier by 0.017 (48 paired worlds; source `show/bundle/mem01_core_results.txt`). Treat it as exploratory.

**"Is the long con real?"**
In the paired checkpoint branches, yes: erasing the trust it learned cut the bluffs it let through by 17 [12.8, 21.3] over 120 seeds. Across agents, naive Bayes let through fewer bluffs than no memory, so we don't claim "naive memory is more exploitable". These are checkpoint restores, not live forks; VM forking wasn't available to our org.

**"Isn't the forge win just luck?"**
Partly. Two of four generations from the same memory beat the agent; two didn't. The memory-withheld control won 23%. The point is that AI-written code ran safely: statically gated, network-blocked sandbox.

**"What does Arize do?"**
It traces every decision and reflection, and runs experiments. Memory surgery on the same 555 frozen decisions: full memory Brier 0.277; erased 0.298 with 63 decisions flipped; planted false memory 0.277 with 0 flipped; a provenance-blind agent 0.284 with 146 flipped.

**"What does SurrealDB do beyond storage?"**
Beliefs are typed facts. A revision supersedes the old one and keeps its history (392 supersessions in r01), and dossiers are read as of any game. *(Don't claim uncertainty records; that spike failed.)*

**"Why not just fine-tune?"**
Memory is auditable, editable and reversible: that's what lets us do surgery on it, and forget one person without retraining.

**"What does it cost?"**
About ten reflection calls per 20-game world. The same-model memory test used 8,442 Sonnet calls with zero errors. Non-LLM worlds cost cents on Daytona.

**"What are the safety measures?"**
AI-written code never runs in our process: it's statically gated and runs in a network-blocked sandbox. Every decision is traced. A memory without evidence gets zero weight.

---

## Rehearsal checklist
- [ ] Do 3 timed run-throughs with the Show. Target 2:50.
- [ ] Do 1 run-through narrating over the backup video only.
- [ ] Say the first and last lines from memory.
- [ ] Practice aloud: "a classifier beat you", "the prompt gave it the answer", "loses to counter".
