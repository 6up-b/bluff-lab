I think there's a way to make this substantially more compelling than either "AI agents learn without retraining" or "AI agents detect liars."

My favorite framing is:

# Agent Bluff Lab: Can we prove that memory makes an AI agent more intelligent?

More specifically:

Can an agent develop a useful model of another agent's behavior through experience, exploit that knowledge, and recognize when its memories become misleading—all without updating its model weights?

The key word is prove.

You already have two excellent demo features:

- Mind Inspector: What did the agent believe, what evidence supported it, and how did those beliefs influence its decision?
- Memory Surgery: What happens to the same decision when we remove, modify, or replace selected memories?

I'd make those the center of the product, rather than the game itself.

The bluffing game is a controlled experimental environment with hidden information, adversarial incentives, repeated interactions, and objectively verifiable outcomes.

That gives you something rare in LLM agent demos: an environment where you can distinguish plausible-sounding explanations from measurable improvements in decision quality.

## 1. Six possible headline experiments

Here's how I would rank them for creativity, technical rigor, and demo appeal.

| Idea                  | Principal demo moment                                                                             | Assessment                                           |
| --------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| 1. Memory Surgery     | Remove an agent's memories at a critical moment and watch its decision change                     | Best overall                                         |
| 2. The Long Con       | An opponent earns trust for ten games, then betrays it. Does the agent anticipate the shift?      | Most cinematic                                       |
| 3. Memory Poisoning   | Implant misleading memories and see whether the agent becomes exploitable                         | Most provocative                                     |
| 4. The Chameleon Test | An opponent changes strategy; the agent must realize its old memories are outdated                | Most scientifically interesting                      |
| 5. The Interrogator   | An agent deliberately challenges a claim to learn about its opponent, even at some immediate risk | Best decision-theoretic depth                        |
| 6. Strategic Transfer | An agent uses something learned about one opponent to predict a different opponent                | Strongest generalization claim, hardest to establish |

I'd combine 1 + 2 for the presentation, with 4 as an especially interesting second experiment.

The central visual would be the same agent confronted with the same situation, but equipped with different memories.

That is a cleaner experiment than comparing a model's performance before and after playing more games, since many other factors change as the games progress.

## 2. My favorite principal demo: The Memory Surgery Reveal

Imagine this sequence.

The Raven has played 12 games against our learning agent. Over time, the Raven has exhibited a subtle linguistic tell: it tends to avoid contractions when bluffing, although the relationship is noisy.

The learning agent has collected revealed examples and formed the hypothesis that the Raven's unusually formal language is predictive of deception.

Then the Raven claims to have played two Kings.

AGENT BLUFF LAB

Demo concept

## "I have placed two Kings. You may challenge if you wish."

The Raven · Round 13

Freeze decision. Open Mind Inspector.

Current Bayesian card baseline

39%

Memory-conditioned belief

76%

Retrieved memory

"The Raven used similarly formal language in several previously revealed bluffs. This pattern has also appeared in some truthful claims."

Illustrative evidence, not actual learned results.

Agent chooses: CHALLENGE

Now you press Memory Surgery.

The agent's entire game state stays fixed: its cards, the Raven's claim, the conversation, the stakes, and the legal actions.

You change only the available long-term memory.

Counterfactual decision comparison

Full memory

# 76%

Bluff belief

Challenge

No memory

# 39%

Bluff belief

Trust

Misleading memory

# 22%

Bluff belief

Trust

Hypothetical outputs showing the intended experiment, not measured probabilities.

Then you flip over the Raven's cards.

It was bluffing.

In one moment, the judge sees a behavioral hypothesis learned from experience, a corresponding change in strategy, and the actual outcome.

But this alone only demonstrates one successful use of memory. You then show tournament results to establish whether the same memory mechanism improves performance across many encounters.

That's the difference between a compelling demonstration and a defensible technical claim.

## 3. Make the deeper claim about causal memory attribution

Here's the technical framing I would use:

A conventional agent evaluation compares:

\\[ \operatorname{Performance}(\pi\_{\text{memory}}) \quad\text{vs.}\quad \operatorname{Performance}(\pi\_{\text{baseline}}) \\]

But the policies might differ in model, prompt, information, reasoning budget, or experimental conditions.

Instead, define:

\\[ a\_{\text{full}}=\pi\_\theta(s,M) \\]

\\[ a\_{\text{ablated}}=\pi\_\theta(s,M\setminus M_i) \\]

where:

- \\(s\\) is the exact information available at the decision.
- \\(M\\) is the retrieved historical memory.
- \\(M_i\\) is one memory or a related cluster.
- \\(\theta\\) is the same frozen model and decision-policy configuration.

This isolates the effect of removing a particular memory on the decision.

You can measure the change in estimated bluff probability, chosen action, and predicted expected utility.

The distinction matters:

A memory changing a decision is not the same as a memory improving a decision.

For actual improvement, compare the two policies across multiple matched game states or counterfactual continuations.

I'd also use three baselines:

| Policy                              | Purpose                                                                 |
| ----------------------------------- | ----------------------------------------------------------------------- |
| Frozen LLM, no persistent memory    | Measures the benefit of historical information                          |
| Frozen LLM + raw episodic history   | Measures whether structured memory adds value beyond recalling examples |
| Adaptive Bayesian opponent model    | Tests whether the LLM adds anything beyond simple statistical learning  |
| Frozen LLM + SurrealDB graph memory | Experimental policy                                                     |

The adaptive Bayesian baseline is especially important. If the Raven has a measurable bluffing tell, a sufficiently expressive statistical classifier may learn it too.

The strongest result would be that the LLM-memory agent discovers a useful behavioral feature without having that feature hand-specified, then uses a numerical policy to exploit it.

That's a more interesting achievement than simply counting past bluffs.

## 4. The Long Con might be the most dramatic demonstration

This uses your Reputation Farmer opponent.

Imagine a graph showing an agent's evolving trust in the Fox.

The Fox's reputation across games

Illustrative strategy schedule and opponent trust, not actual measured behavior.

Fox's true honesty rateLearner's estimated honesty

0%25%50%75%100%13579111315

The Fox behaves honestly for several games, then switches to deception. A memory system needs evidence to detect and revise its earlier belief.

This presents two distinct intelligence challenges.

First: Can the agent recognize that its opponent has been deliberately cultivating trust?

Second: When that opponent changes behavior, can the agent update its beliefs quickly without overreacting to one unusual event?

The second is easier to measure convincingly. It's online change-point detection.

One simple version compares a stationary opponent model against a recency-weighted model or Bayesian change-point detector, then lets the LLM reason about the behavioral shift.

This would exploit a particularly relevant strength of SurrealDB Agent Memory: it preserves source provenance and historical belief changes rather than simply overwriting old beliefs.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://surrealdb.com\&sz=32)

Agent Memory

+1



The surprising finding might be that memory can make an agent more exploitable when the opponent knows how to manipulate its reputation.

That is an excellent research narrative.

## 5. An even more provocative alternative: Can an adversary hack an agent through its own memories?

I think this is worth considering as the overarching safety framing.

It's not traditional prompt injection. The adversary isn't necessarily saying "ignore your instructions."

Instead, it's engaging in behavioral memory poisoning.

The Fox behaves honestly for a long period to create reliable-looking evidence that it is trustworthy. The learning agent internalizes that pattern, and the Fox subsequently exploits it.

This is especially interesting because the memories aren't false. The recorded history is genuinely accurate.

The failure occurs because the agent incorrectly assumes the opponent's behavior is stationary.

The experiment becomes:

Can an adversary manipulate another agent's future decisions by controlling the experiences from which it learns?

That connects your game to AI agent safety, adversarial adaptation, and trust calibration.

You could show three outcomes: an agent without memory that cannot personalize its predictions, an agent with naive memory that becomes overconfident, and an agent with uncertainty-aware memory that detects a change in behavior.

I wouldn't claim the third will always win. It's an experimental hypothesis you can actually test.

This framing may be more compelling to technical judges than simply claiming that memory improves win rates.

## 6. Three more creative features worth considering

Memory Transplant

Transfer a learned opponent profile from one agent to another. Does an agent inherit useful strategic knowledge from experiences it never had?

This tests whether memory is genuinely transferable or too dependent on an individual's playing history.

The False Memory Test

Replace one genuine behavioral hypothesis with an unsupported or contradictory one, without changing the game state.

Does the agent blindly trust the retrieved graph, or appropriately discount an unreliable memory?

The Information-Seeking Challenge

The agent challenges a claim partly to discover whether an opponent's suspected tell is real.

You can display the expected information gain and immediate risk side by side, demonstrating strategic exploration rather than only exploitation.

Of these, the False Memory Test is probably the easiest addition to Memory Surgery. A transplant or information-gain policy would require more evaluation work.

## 7. The evaluation that would convince me

I'd report four metrics, rather than relying on win rate alone.

| Metric                        | What it establishes                                              |
| ----------------------------- | ---------------------------------------------------------------- |
| Bluff detection / Brier score | Whether predictions are accurate and calibrated                  |
| Win rate or cumulative reward | Whether better predictions translate into better decisions       |
| Memory ablation effect        | Whether retrieved memories actually affect behavior              |
| Adaptation after policy shift | Whether the agent can recover when its opponent changes strategy |

The simulator should retain the actual hidden cards for evaluation, but never expose them to the agents before they're legitimately revealed. This also addresses an important statistical problem: an agent only learns the true outcome of a claim when somebody challenges it.

For the experiments, I'd use matched game seeds, randomized seats, and separate learning/evaluation episodes. Any tournament improvement should be accompanied by uncertainty estimates.

One especially important ablation: Compare the memory agent with an adaptive Bayesian learner that receives exactly the same revealed observations. Otherwise, the experiment only demonstrates the advantage of having history over having none.

## 8. How I'd use the sponsors to make the results credible

[Introducing SurrealDB 3.0 - the future of AI agent memory | Blog | SurrealDB](https://images.openai.com/static-rsc-4/B-yXjRTWnvQ166lZq02rWeMgRe0oZlZ7U7P3GVF9evID8fkz_2nCawcyogi6CLANyFTG_0ZoxcjdIu30WVDgy7hofsAn8KUswEF3YT9CSYAypMO6x1fNrvVFqS_ZmfgqYL1rapcDxeE2pWOzHKKQZKXc2szPP_-tYDmxEgwrG9Q?purpose=inline)

SurrealDB — What the agent learned

Store opponent identities, behavioral hypotheses, supporting and contradictory observations, timestamps, and provenance.

During Memory Surgery, retrieve the exact evidence responsible for a belief, then selectively ablate it. Its documented graph, provenance, and temporal-belief features make this a strong fit.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://surrealdb.com\&sz=32)

Agent Memory



[Daytona: Open-source dev env that makes you 2x more productive! | Product Hunt](https://images.openai.com/static-rsc-4/qRhsoFMPXCAaN9gVP8sx4W6aq4Z2qqe5mZXkBC5f0598i1veIo9gco1u98jW4SuB4mCg-HPYp4hYzXgqIgrNNbPIMyzcqL9z0coiF3z5O-FPQG1Ar1lDDMP55tA6RrQZzDUEdFlOkVnvZEjGuV_VExyWVJ2usKGFxUIrPu4Urqk?purpose=inline)

Daytona — What would happen otherwise

Execute isolated matches and counterfactual policy comparisons from matched game states.

For the MVP, parallel sandboxes running deterministic simulations are sufficient. Daytona also documents memory-and-filesystem forking for Linux VM sandboxes if you want to demonstrate live runtime forks.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.daytona.io\&sz=32)

Daytona

+1



[Arize Phoenix – Open-Source AI Observability & Debugging Platform – Daidu.ai](https://images.openai.com/static-rsc-4/M9ODlt353iUqkAqO9ixdRpW4pf6FDN0wyTR4H8SzEp5iGXWl51jPHMAD5S7aqL58HK8L-pdNyCRR6esSGf0klNnMLGyZ-z2N8QxyZVPL-aP_CZWpt2VEJgnEtgAp8bq2-jU93uKkwGEBg-RGNNnF0aTOJ_x28kRCpU03Lwq5RQQ?purpose=inline)

Arize — Whether learning actually worked

Record the context, retrieved memories, predictions, chosen actions, and eventual outcomes.

Use Phoenix's tracing and experiment features to compare matched decisions and policy versions with deterministic evaluators.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



Mind Inspector should show recorded beliefs and evidence, not claim to expose the LLM's private internal reasoning.

## 9. The exact three-minute presentation I'd use

1. 0:00–0:30 — The Hook

   "What happens when AI agents remember other agents? Can they become better strategists—or can that memory be weaponized against them?"

   Show the Raven making a suspicious claim.
2. 0:30–1:15 — Mind Inspector

   Freeze the game. Show the memory-backed behavioral hypothesis, the supporting observations, and the resulting bluff prediction.

   Reveal the cards.
3. 1:15–1:50 — Memory Surgery

   Rerun the exact decision without the historical memory. Show how the prediction and action change.

   Highlight the specific historical observations responsible for the difference.
4. 1:50–2:30 — Tournament Evidence

   Show the completed Daytona experiments and Arize comparisons: memoryless, adaptive Bayesian, and structured-memory agents.

   Present actual measured results, including where memory fails.
5. 2:30–3:00 — The Twist

   Introduce the Chameleon or Reputation Farmer: an adversary that deliberately changes behavior.

   "Memory helps agents learn. But can they recognize when their experience is being used against them?"

## My recommendation

I would build around one flagship claim:

> Agent Bluff Lab is a controlled experiment in how AI agents learn and adapt through persistent memory, without retraining—and whether that memory makes them more capable or more exploitable.

That is a much stronger framing than "we made an AI good at bluffing."

For the time remaining today, I'd make Mind Inspector + Memory Surgery on the Verbal Telltale opponent the first completed vertical slice. It's the clearest way to show a learned, semantically meaningful behavioral pattern influencing an actual decision.

Then add the Reputation Farmer as the adversarial stress test. If you have time for one striking result, show that memory improves play against one opponent but becomes a liability against another.

The nuance is what makes the project feel rigorous: you're not trying to prove that more memory is always better. You're investigating what makes a memory worth trusting.

That's a research question with meaningful implications well beyond a card game.
