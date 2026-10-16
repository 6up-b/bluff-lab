I would choose Agent Bluff Lab for this hackathon, but I'd make one important adjustment to the concept:

Don't pitch it as AI agents playing Liar's Deck. Pitch it as a live adversarial intelligence laboratory disguised as a bluffing game.

The game is how you make the research accessible. The real product is an environment for studying whether autonomous agents can model other agents, learn from deception, adapt their strategies, and make calibrated decisions under uncertainty.

I think that combination has a higher ceiling for creativity and presentation than the causal debugger, while retaining enough technical depth to be competitive with developer-infrastructure projects.

## The winning demo I imagine

Visual inspiration: a theatrical three-player bluffing table.

The judge walks up and sees three characters around a table. One of them is lying. The judge doesn't know which.

The presenter says:

"These aren't scripted NPCs. They're autonomous agents with different decision policies. Some calculate probabilities. Others learn your behavior. Let's see if you can fool them."

The judge plays one round, makes a bluff, and watches an agent decide whether to challenge.

Then comes the reveal.

The game freezes, the camera zooms into the agent's mind, and you see its beliefs about the human and other agents, its estimate of whether the claim was a bluff, and why it made its decision.

Finally, you press Run 100 Games.

Daytona launches isolated simulations comparing the agents' strategies, and Arize displays their outcomes, decisions, and calibration.

That's a much more memorable demonstration than a generic agent dashboard.

## Three layers of the product

1\. Play

A fast, understandable bluffing game with one human and two AI agents.

2\. Inspect

Reveal the agents' beliefs, memories, uncertainty, and strategic choices.

3\. Experiment

Fork simulations, swap policies, test counterfactual decisions, and compare learning strategies.

The key is that each layer reveals something the previous layer couldn't.

The game makes the agents' behavior tangible. The inspector makes it explainable. The experiments establish whether the behavior is actually intelligent.

## 4. The features that could make this genuinely special

I would prioritize features that aren't just visual gimmicks, but expose something interesting about agent intelligence.

Mind Inspector — highest priority

Freeze any decision and inspect what the agent knew, its estimated bluff probability, its opponent memories, and the expected utility of each action.

The crucial distinction is between the agent's actual recorded decision variables and an LLM's retrospective explanation. Only the former should be presented as the decision's underlying basis.

Fork Reality — the signature Daytona feature

Pause a game at a decision. Fork the same public state and private information into two isolated continuations.

In one, the agent challenges. In the other, it trusts.

Compare outcomes over multiple sampled hidden states consistent with the information available at that moment.

This is more technically interesting than replaying a game with different random cards because it isolates the effect of a particular decision.

Memory Graph — learning without retraining

Every revealed bluff creates evidence about an opponent's behavior.

Connect observations to hypotheses such as "bluffs more frequently under pressure" or "rarely challenges early."

As games accumulate, the agent can strengthen, weaken, or discard those hypotheses. A frozen LLM can use the retrieved memory to make different decisions without changing model weights.

This could make meaningful use of SurrealDB's agent-memory capabilities, but I would only add it after the tournament works.

Evolution Arena — a scientific tournament

Pit three policies against one another:

- A probability-based Bayesian baseline.
- A memory-conditioned opponent model.
- An LLM-based strategic agent.

Evaluate win rate, bluff-detection precision, Brier score, and performance against unseen opponents.

The particularly compelling result is whether learning an opponent's behavior actually produces an advantage over a fixed Bayesian strategy.

Psychological Warfare — optional

Let agents speak one short sentence before an opponent decides to challenge.

The language can be a strategic action: reassurance, intimidation, misdirection, or deliberate honesty.

Run an ablation with dialogue enabled and disabled. Does natural-language persuasion actually improve outcomes beyond card probabilities?

This could justify ElevenLabs voice synthesis, but it's a stretch feature rather than the core experiment.

## 5. The technical experiment I'd build around

The most defensible question is:

Does persistent opponent modeling outperform a rational, memoryless Bayesian policy?

A simple Bayesian baseline can compute the probability of a bluff using the deck composition and its private hand. The memory-based agent adds a learned estimate of how often an opponent bluffs under different circumstances.

For example, use a Beta-Bernoulli model for each opponent's bluff frequency:

\\[ p_i \sim \operatorname{Beta}(\alpha_i,\beta_i) \\]

When a challenged claim is revealed, update the parameters based on whether the opponent bluffed.

You can later extend this to context-specific statistics for hand size, prior challenges, and stakes.

The important subtlety is that not every bluff gets revealed. You only observe labels for challenged claims. If you naively treat unchallenged claims as truthful, you'll bias the model.

That itself could become an interesting experimental feature: compare naive learning against a policy that explicitly recognizes its missing observations.

### The tournament

Tournament configuration

Illustrative experiment setup

Sandbox plan

Games per policy

## 100

Include memory-based agent

Include LLM strategic agent

EXPERIMENT PLAN

Policies compared

3

Total simulated games

300

Suggested Daytona workers

4

Design this experiment&#x20;

Use identical initial game seeds and rotate player seats to reduce variance. If memory persists across games, distinguish training episodes from held-out evaluation episodes.

The tournament should also report confidence intervals. With only 100 games, a small apparent improvement in win rate may just be noise.

## 6. How the sponsors fit

The event explicitly offers awards for best use of Daytona, Arize AI, SurrealDB, ElevenLabs, and Anchor Browser, in addition to overall prizes.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://daytona-hacksprint-nyc.devpost.com\&sz=32)

Devpost



| Sponsor        | Meaningful integration                                                                 | Priority       |
| -------------- | -------------------------------------------------------------------------------------- | -------------- |
| Daytona        | Isolated tournaments, parallel policy experiments, forked counterfactual continuations | Essential      |
| Arize AI       | Trace decisions, evaluate calibration, compare policy experiments                      | Essential      |
| SurrealDB      | Persistent graph of opponent observations, hypotheses, and corrections                 | Valuable third |
| ElevenLabs     | Voiced bluffing, personality, persuasive dialogue                                      | Stretch        |
| Anchor Browser | Browser-controlled agents as actual game participants                                  | Probably skip  |

Daytona's Linux VM sandboxes can fork filesystem and memory state, whereas its container sandboxes require snapshot-based recreation for similar workflows. That's useful for the counterfactual feature, though I would start with lightweight independent simulation workers rather than depend on live VM forking on day one.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.daytona.io\&sz=32)

Daytona



Arize Phoenix supports traces, custom evaluations, datasets, and side-by-side experiments, making it a natural home for the tournament's evidence.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arize.com\&sz=32)

Phoenix



## 7. What I'd actually build before judging

The biggest danger is trying to implement five research ideas in one afternoon. I'd set a strict priority order.

P0

Playable game + inspectable agent

Three players, legal move engine, a human interface, and one Bayesian opponent. The judge can bluff and immediately see the agent's recorded probability estimate and action.

P1

Parallel tournament + Arize

Run reproducible games on Daytona, collect decisions and outcomes, and compare the memoryless baseline against a memory-based policy.

P2

Fork one decision

Choose a consequential challenge, replay two alternative actions, and display estimated outcome differences.

P3

Graph memory or voice

Only after the core game and experiment are reliable.

## 8. The three-minute pitch

I'd organize the demo like this:

0:00–0:30 — THE HOOK

"AI agents are increasingly negotiating, collaborating, and competing with one another. But can they tell when another agent is lying?"

Show the game. Let a judge challenge an agent.

0:30–1:10 — THE REVEAL

Open the Mind Inspector. Show the exact evidence, probability estimate, and action the agent used.

1:10–2:10 — THE EXPERIMENT

Run or reveal a completed Daytona tournament. Compare Bayesian, memory-based, and optionally LLM-based policies, with evaluation traces in Arize.

2:10–3:00 — THE IMPLICATION

Explain how the same infrastructure can benchmark trust, deception detection, memory, and strategic adaptation in real multi-agent systems.

The strongest closing line would be:

> "We turned a bluffing game into a reproducible testbed for whether AI agents can reason about—and learn from—other agents."

## Final decision

I would choose Agent Bluff Lab over Causal Replay for the competition, provided you keep the game rules extremely simple.

The causal debugger has a more immediate commercial application. But the bluffing lab has a distinctive combination of an understandable interactive experience, visible agent reasoning, rigorous experiments, and a memorable presentation.

One creative feature stands out above everything else: freeze a decision, reveal what the agent believed, then fork reality to see whether it should have acted differently.

That makes Daytona's isolation central to the experience, gives Arize something meaningful to evaluate, and connects the playful premise to a serious technical question.

It also makes the project more than a game. It becomes a small, observable laboratory for adversarial multi-agent decision-making.
<USER>What sort of decision policies could we have? Please brainstorm decision policies that would play to the strengths of the memory LLM agent.</USER>
I think the most interesting approach is to create opponents with latent behavioral strategies that can only be discovered through repeated interaction.

A Bayesian agent can reason about the cards. But a memory-equipped LLM agent could potentially reason about the person playing the cards: their habits, incentives, reactions to previous games, and attempts to manipulate their reputation.

That sets up a much more compelling experiment than simply comparing an LLM against a random bluffer.

The central research question: Can an agent learn another agent's hidden behavioral policy through experience, then exploit it without retraining?

There is a legitimate research foundation here. A 2026 AAMAS paper, Solving Repeated Games with Large Language Model, studies agents maintaining hypotheses about opponents' strategies and adapting their behavior. This is closely aligned with what you're proposing.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://doi.org\&sz=32)

DOI



I'd design two layers of policies: the strategies that opponents exhibit and the strategies that our learning agents use to discover and exploit them.

## 1. Opponent policies: Give the agents psychological personalities

These aren't merely dialogue personalities. Each has a distinct, testable decision policy.

The Reputation Farmer

Plays unusually honestly during early games, building a reputation for trustworthiness. As opponents become more trusting, strategically increases bluffing.

Memory opportunity: Notice that honesty was an investment in reputation rather than a fixed personality trait.

The Grudge Holder

Changes behavior toward a specific player after being challenged. It might become more aggressive against that player or deliberately stop bluffing to bait false accusations.

Memory opportunity: Track relationships and player-specific histories rather than global bluffing frequencies.

The Verbal Telltale

Has a subtle, consistent linguistic habit when bluffing: excessive reassurance, changing phrasing, or unusually detailed justifications.

Memory opportunity: Discover semantic patterns in dialogue and connect them to subsequently revealed bluffs.

The Chameleon

Switches strategies after being caught. When opponents learn one behavioral pattern, it adopts another.

Memory opportunity: Detect change points, revise outdated beliefs, and recognize that previous memories may no longer apply.

The Counter-Psychologist

Models what opponents believe about it. Once it suspects another player has identified its bluffing pattern, it deliberately reverses the pattern.

Memory opportunity: Higher-order theory of mind: "I think they think I'm bluffing."

The Pressure Player

Behaves differently after a losing streak, when holding few cards, or when the cost of a false challenge becomes high.

Memory opportunity: Identify conditional behavioral patterns rather than assigning one permanent bluff probability.

The Mimic

Copies strategies that recently succeeded against it, including bluff sizes, challenge thresholds, or verbal tactics.

Memory opportunity: Recognize that the opponent's policy is changing in response to observed players.

My favorites are Reputation Farmer, Verbal Telltale, and Chameleon. Together they test three meaningfully different capabilities:

- Long-horizon behavioral memory.
- Semantic pattern recognition.
- Adaptation to nonstationary opponents.

And critically, these policies need not be deterministic. For example, a Reputation Farmer could gradually change its bluff probability from 15% to 65%, rather than always switching after exactly five rounds.

That prevents agents from winning through trivial pattern memorization.

## 2. The actual decision-making agents

I'd compare five progressively more capable policies, rather than making every character an LLM.

| Policy                    | How it decides                                               | What it remembers                          |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------ |
| A. Card-Counting Bayesian | Exact card probabilities + expected utility                  | Current public game state                  |
| B. Adaptive Bayesian      | Card probabilities + learned opponent-specific bluff rates   | Per-opponent statistics                    |
| C. Memoryless LLM         | Reasons about the current hand, dialogue, and legal moves    | Current game only                          |
| D. Episodic Memory LLM    | Retrieves previous observations about each opponent          | Prior games and outcomes                   |
| E. Strategic Memory LLM   | Forms, tests, and revises hypotheses about opponent behavior | Structured behavioral models with evidence |

The really important comparison isn't A versus E.

It's D versus E, and C versus E.

Does simply giving an LLM previous experiences improve its performance? Or does it need a structured process for forming and updating behavioral hypotheses?

And can E outperform B—a much simpler statistical learner—on opponents with complex, context-dependent behavior?

A Bayesian model with the correct behavioral features might outperform the LLM. That's a legitimate outcome, not a failed project.

### Policy E: The Bayesian–LLM hybrid

This would be my preferred flagship agent.

Rather than asking an LLM to estimate raw probabilities from scratch, I would divide responsibilities.

Bayesian game engine

Computes legal actions, card feasibility, and a baseline probability of deception from the current hand.

LLM opponent profiler

Retrieves historical observations and proposes relevant behavioral hypotheses.

Behavioral belief updater

Uses historical evidence to adjust opponent-policy estimates, while recording uncertainty.

Expected-utility policy

Selects challenge, continue, or a particular card play based on the updated beliefs and predicted outcomes.

The LLM contributes flexible feature discovery and hypothesis generation. The numerical model contributes disciplined probability updates and decision-making.

This is stronger architecturally than making the LLM simultaneously calculate card probabilities, remember every interaction, and choose moves.

## 3. My favorite creative mechanic: Make learning an action

Here's something that would make this much more interesting.

In Liar's Deck, challenging somebody isn't just a strategic decision. It's also an experiment.

When you call a bluff, you reveal whether the opponent was lying. Otherwise, that information remains hidden.

Therefore, an agent has to balance two objectives:

- Winning the current game.
- Gathering information that will help win future games.

This is a natural Bayesian experimental-design problem.

\\[ a^\*=\arg\max_a \left[ \mathbb E[U(a)] +\lambda I(\Theta;O\mid a) \right] \\]

Here \\(\Theta\\) represents the opponent's latent strategy, and \\(O\\) is the observation generated by the action.

An agent might deliberately challenge a moderately suspicious player because the result would distinguish between two hypotheses about their behavior.

That's potentially a suboptimal decision in the current round but an intelligent one across 20 future games.

Example: The Owl is uncertain about the Fox

Two competing hypotheses:

H₁: Pressure bluffer

Bluffs more when nearly out of cards.

H₂: Reputation farmer

Bluffs more after several trusted claims.

The Fox makes a suspicious play while holding only two cards, after being trusted repeatedly.

Decision: Challenge or continue?

An information-seeking agent considers both the immediate payoff and how much the revealed cards would clarify its hypotheses.

Hypothetical scenario. The hypotheses may also overlap; a richer model need not treat them as mutually exclusive.

I'd call this policy The Scientist.

It plays the game while actively designing experiments to discover opponent strategies.

That creates a direct connection to causal experimental design, but without needing to build the full debugger.

## 4. Give the memory agent a real memory graph

Instead of storing arbitrary conversational summaries, maintain a graph of behavioral observations and hypotheses.

For example:

Opponent: The Fox

Hypothesized behavior

Bluffs more after winning a challenge

Confidence: provisional

Supporting evidence

Games 4, 6, 9: revealed bluffs after challenge wins

Contradictory evidence

Games 7, 11: truthful plays after challenge wins

Illustrative memory graph. Observations should be marked confirmed only when the relevant cards were actually revealed.

A useful twist: store not only which beliefs an agent has, but when those beliefs stopped being predictive.

That would let the Chameleon specifically test whether the memory system can recognize strategy drift.

## 5. The most spectacular demo feature: Memory Surgery

I think this may be even more valuable than a generic counterfactual replay.

Freeze a decision, then ask:

What would this exact same agent do if it had never met this opponent before?

Memory Surgery

Illustrative

Identical cards, identical observed move, different historical memory.

Memory enabled

# 81%

Estimated bluff probability

CHALLENGE

Memory erased

# 34%

Estimated bluff probability

CONTINUE

Historical memory responsible for the difference

"The Fox's bluff frequency increased following three consecutive unchallenged plays."

Numbers are hypothetical. The running system would use recorded outputs from both policies.

Using Daytona, you could run both versions from the same checkpoint, swapping only the agent's memory context.

You can go even further and perform memory ablations:

Does removing one particular memory change the decision? Does removing an entire opponent profile change the win rate? Does providing a misleading memory make performance worse?

This makes memory a manipulable, testable component rather than a black box.

A subtle but important point: a changed action establishes that the memory affected the agent's decision, but not that the memory was beneficial. For the latter, you need outcome evaluation.

## 6. A tournament specifically designed to expose memory advantages

Rather than 100 independent games with random opponents, I would run repeated series against the same opponents.

For example, split games into early and late encounters.

Example learning-curve experiment

Hypothetical win rates for illustration only, not measured performance.

Bayesian baselineMemoryless LLMStrategic-memory LLM

20%35%50%65%80%1–56–1011–1516–20

What you'd hope to investigate: whether sustained exposure to an opponent produces increasing strategic advantage. Real results may not show this pattern.

For a fair comparison, include an adaptive Bayesian baseline that also gets access to past revealed behavior. Otherwise, you'll only prove that remembering history is better than forgetting it—not that LLM-based strategic reasoning adds value.

And here's an especially interesting evaluation: introduce a new opponent with a related but not identical behavioral policy. Does the memory agent transfer a useful principle, or does it overgeneralize from its old opponent?

This tests whether it learned something resembling a strategy rather than a collection of anecdotes.

## 7. Which policies I'd actually build for the hackathon

I would select a small but expressive set:

| Role                | Policy                   | Why it's worth including                |
| ------------------- | ------------------------ | --------------------------------------- |
| Baseline player     | Card-counting Bayesian   | Strong, interpretable foundation        |
| Statistical learner | Adaptive Bayesian        | Controls for the value of simple memory |
| LLM control         | Memoryless strategic LLM | Isolates persistent-memory effects      |
| Flagship            | Strategic-memory LLM     | Learns behavioral hypotheses            |
| Opponent 1          | Reputation Farmer        | Long-horizon adaptation                 |
| Opponent 2          | Pressure Player          | Context-sensitive behavior              |
| Opponent 3          | Chameleon                | Memory revision and change detection    |

If there is time, add the Verbal Telltale opponent, because linguistic signals give the LLM a reason to use its language understanding rather than rely exclusively on numeric statistics.

The two signature features would be Mind Inspector and Memory Surgery, with Daytona-hosted tournaments demonstrating actual differences in decision quality.

The most important guardrail is keeping the experiment honest: a strong contextual Bayesian model might discover some of these patterns faster and more accurately than an LLM. That's precisely why it's a valuable baseline.

My favorite version of this project is therefore:

Agent Bluff Lab: Can an AI learn your strategy, predict your deception, and recognize when you've started manipulating its beliefs?

That has a much more interesting research arc than merely "AI that learns to bluff." It explores opponent modeling, active learning, nonstationarity, theory of mind, and adversarial memory—all through a game that a judge can understand almost immediately.

And it gives you a fascinating possible result: the agent with the most memory might not be the best one. The best agent may be the one that knows which memories to trust, when to revise them, and when to seek new evidence.
