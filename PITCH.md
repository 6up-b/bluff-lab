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

**Present from the curated deck: https://6up-b.github.io/bluff-lab/?deck=pitch** (21 cues, nothing to press through). The full deck (no `?deck`) keeps every slide for Q&A. Each table or chart ends on a spoken **Takeaway**.

**[title]** · 0:00
> Agents are starting to carry memory from one encounter to the next. Memory can make them sharper, or it can be used against them. Bluff Lab measures which.

**[cold open: deal → claims → ask → call → reveal]** (press on the beat, let the voices play) · 0:10
> Three players. You lay cards face-down and claim they match the table rank, and you're allowed to lie. The next player plays on, asks a question, or calls it. *(press to the reveal)* The cards flip and whoever was wrong loses a life, so every challenge gives us ground truth.

**[game 10: the claim]** · 0:30
> Game ten. Player 3: "The cards are one King."

**[the ask]** (let the voice play: "You've gone very formal all of a sudden.")
> That's our agent, the Profiler: the same frozen model all day, no fine-tuning. Its memory, written between games and checked against the reveals, says this player turns formal when bluffing.

**[freeze]**
> Let's stop time and look inside.

**[Mind Inspector]** · 0:50
> Card math alone says 24% this is a bluff. Memory holds one confirmed line: Player 3 speaks formally when bluffing, a 68% bluff rate across eleven reveals versus 22% otherwise. That lifts it to 76%, so it challenges.

**[surgery: full]** · 1:05
> Now, surgery. Same state, same claim, same model; only the memory changes.

**[erased]**
> Erase the memory: 43%. It plays on.

**[false memory]**
> Plant a false line, "Player 3 never bluffs," with no evidence behind it. Ours ignores it: still 76%.

**[provenance-blind]**
> An agent that trusts every line it's handed drops to 24%.

**[flip]** (pause one beat)
> It was a bluff.

**[ablation]** · 1:30
> One decision isn't proof. So we replayed games eleven to twenty on the same seeds, memory frozen at game ten. Erased, the win rate falls from 23% to 9%. But older memory did as well as the full file.
> **Takeaway:** more memory isn't better. Trusted memory is.

**[does memory help?]** · 1:50
> Same model, same raw table talk, only the memory differs. A raw transcript of past rounds: no detectable gain. A file of hypotheses: plus 6. Hypotheses weighed by a small statistical model: plus 19.
> **Takeaway:** it isn't having the history that helps. It's how the memory is structured.

**[one memory, two opponents]** · 2:10
> Then we tested it the strict way: pre-registered, fresh seeds, raw text only. It beat a word-count classifier by 8 points, and switching Claude's hypotheses off costs about 7. But against a long con, those same hypotheses let through 24 more of the betrayer's bluffs.
> **Takeaway:** Claude's hypotheses are earned trust. They win more games, and they're exactly what a long con exploits.

**[capability map]** · 2:35
> And where we lose: on prediction alone, simple classifiers do as well or better. This boundary test was pre-registered, and we lost it.
> **Takeaway:** memory doesn't make a better lie detector than a good classifier. It makes an agent you can inspect, edit and defend.

**[close]** · 2:50
> Memory made our agent better, and more exploitable by players who change. The question isn't how much an agent remembers. It's which memories it should trust, and when to stop.

*(Stop. Smile. Wait for questions.)*

## Table takeaways (Q&A, or if a skipped slide comes up)
- **Long con chart (cue 20):** the Profiler's estimate tracks the turn most closely and catches it in a median of 14 rounds. Plain Bayes lets through about as few bluffs. **Takeaway:** noticing a change isn't the same as acting on it.
- **Long con at scale (cue 21):** every memory agent catches the turn within about 10–12 rounds; with no memory it takes 55. But our Profiler also raises false alarms on honest players. **Takeaway:** sensitivity to betrayal has a cost, and our detection rule fires too easily.
- **Results grid (cue 25):** no agent wins every column. Memory agents predict and detect better, but the original Profiler over-challenges and loses on win rate under the spec's rule. **Takeaway:** prediction and decision are different scores.
- **Beliefs vs decisions (cue 26):** memory sharpens predictions under both decision rules. Only the cost-aware rule turns that into more wins. **Takeaway:** better beliefs need a decision rule that uses them.
- **Split screen (cue 24):** don't show it (6 vs 6; the games diverge after the first difference).
- **The Regulars (not in the deck):** keying memory by person split the data four ways. A pooled memory learned the shared patterns faster, and the conflicting tells were too weak to hurt it. **Takeaway:** identity needs a shared population file that each person's file starts from (partial pooling). That's the next pre-registered arm.

## If you're over time (cut in this order)
1. The checkpoint-branches beat (cue 22): press through and go straight to lc03.
2. "Early evidence, three to four seeds per table." (keep it for Q&A).
3. The ablation's "older memory" sentence. Keep the Takeaway.
4. Cold open: "Claim your cards match; you can lie; the next player plays on, asks, or calls it."

## The Regulars: no stage beat (pre-registered; H8, H10a and H10b did not hold)
Use it only in Q&A (see below).

## If something breaks
- **No audio:** read the lines yourself. Cue 8: "You've gone very formal all of a sudden."
- **The Show freezes or the projector fails:** play the backup video and narrate with this script.
- **Asked "is this live?":** "The table is a replay of one recorded world, picked by a fixed rule. Every number comes from real simulated worlds (5,916 in the main run alone), and the code is public."

---

## Q&A prep (answer in one or two sentences, then stop)

**"Beat what, exactly?"** (c01, pre-registered)
On fresh seeds 5000–5023, raw text only, the agent beat a bag-of-words classifier by 8.1 points of win rate [+5.7, +10.4], replicating mem01. That classifier wasn't handed the tell, and on raw text it's the stronger of our two classifiers. The record check we added was right 80–83% of the time but added nothing on top; the gain comes from Claude's hypotheses.

**"c02 says you beat the embedding classifier by 18 points. Why not lead with that?"**
It's a low bar on those tables: even the non-learning card counter beats that classifier by about 15 points. The c02 number we use is the replication: reflection on vs off, +7.3 [+1.9, +11.6] on fresh seeds, matching mem01's +9.5. That one wasn't pre-registered in c02, and we label it that way.


**"Did remembering *who* you play help?"**
We pre-registered it, and it didn't hold. Keying memory by person split the data four ways. A pooled memory learned the shared base rate and calibration about four times faster, and our conflicting tells were too weak to punish pooling: repeat-meeting win rate −2.3 points vs pooled [−5.1, +0.6]. It did beat a pooled classifier (+3.2 [+0.7, +5.8]). The fix is partial pooling: start each new person from the population file and let their own evidence override it.

**"Does forgetting work?"**
Mechanically, yes. Forget Avery, and the replayed decision about Avery moves from 0.31 to 0.04, while the decision about Casey stays at 0.395, exactly unchanged (`regulars.json`, decision_replays). Over a short season, though, forgetting didn't measurably change win rate (the pre-registered test didn't hold). Removing a person from memory is exact and checkable, unlike weights; the payoff just needs more data per person to show.


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
