"""Utterance banks (SPEC §5.5). Templated on {n} (number word) and {ranks} (rank name, pluralized).

Invariant (tested): every CASUAL line contains a contraction (formal_register=False) and every
FORMAL line has no contraction and >= 4 words (formal_register=True), for every n in 1..3 and rank.
"""
from __future__ import annotations

from blab.engine import RANK_NAME

NUM_WORD = {1: "one", 2: "two", 3: "three"}

CASUAL = [
    "That's {n} {ranks}.",
    "It's {n} {ranks}, don't worry.",
    "Here's {n} {ranks} for ya.",
    "There's {n} {ranks} down.",
    "I've got {n} {ranks} here.",
    "That'd be {n} {ranks}.",
    "Don't sweat it, {n} {ranks}.",
    "We're looking at {n} {ranks}.",
    "You'll see {n} {ranks}.",
    "Y'know, {n} {ranks}.",
]

FORMAL = [
    "I have placed {n} {ranks}.",
    "These are {n} {ranks}, as stated.",
    "I am playing {n} {ranks}.",
    "I place {n} {ranks} on the table.",
    "Please accept {n} {ranks} from me.",
    "I declare {n} {ranks} now.",
    "Here are {n} {ranks}, as required.",
    "I am laying down {n} {ranks}.",
    "The cards are {n} {ranks}.",
    "I submit {n} {ranks} to the pile.",
]

OVERT_TELL = "Trust me, it's {n} {ranks}."
OVERT_PLAIN = "{n} {ranks}."

# answers to an `ask` (§7.5)
ANSWER_CASUAL = [
    "Nah, it's legit.",
    "C'mon, I'm not bluffing.",
    "It's fine, really.",
    "Don't overthink it.",
    "You're paranoid, it's real.",
    "I'd never lie to you.",
    "Relax, they're good.",
    "What's the problem?",
]
ANSWER_FORMAL = [
    "I assure you the claim is accurate.",
    "I am telling you the truth.",
    "Those cards are exactly as I stated.",
    "There is no reason to doubt me.",
    "I would not mislead you about this.",
    "The claim stands as I made it.",
    "You may check the cards if you wish.",
    "I have nothing further to add.",
]


def ranks_word(rank: str, n: int) -> str:
    name = RANK_NAME[rank]
    return name if n == 1 else name + "s"


def render(template: str, n: int, rank: str, digits: bool = False) -> str:
    return template.format(n=(str(n) if digits else NUM_WORD[n]), ranks=ranks_word(rank, n))


def pick(bank: list[str], rng, n: int, rank: str) -> str:
    return render(bank[int(rng.integers(0, len(bank)))], n, rank)

# ---------------------------------------------------------------- b01 (capability boundary)
# tell_relational: one factual statement per claim whose truth is computable from the PUBLIC log only.
# Wording (casual or formal) is chosen independently of bluffing; only the statement's truth carries the tell.
REL_COUNT = [                       # {N}: total cards claimed this round incl. this play; {rank_pl}: plural
    "That makes {N} {rank_pl} claimed this round.",
    "So that's {N} {rank_pl} this round.",
    "Running total this round: {N} {rank_pl}.",
    "We're at {N} {rank_pl} this round now.",
]
REL_HAND = [                        # {h}: my public hand size after the play
    "I've got {h} cards left.",
    "I have {h} cards remaining.",
    "That leaves me with {h} cards.",
    "Down to {h} cards in my hand.",
]
REL_SAME = [                        # true iff table rank == previous round's table rank
    "Same rank as last round.",
    "The rank is the same as the previous round.",
    "Same rank again, like last round.",
    "This rank matches the last round.",
]
REL_DIFF = [
    "Different rank from last round.",
    "The rank has changed since the previous round.",
    "New rank, not like last round.",
    "This rank differs from the last round.",
]
# c01 rebuilt relational tell (PREREG_c01.md): identical wording for true and false statements; only the rank
# word differs. {R}: plural rank name of the rank the claimant says the PREVIOUS round had.
REL2_PREV = [
    "Last round's rank was {R}.",
    "The previous round was {R}.",
    "Last round we played {R}.",
    "The round before this one was {R}.",
]
REL_N_RANGE = range(0, 17)          # 15 cards max in play (3 x 5), perturbed by +-1
REL_H_RANGE = range(0, 6)

# answer_tell (b01): semantic tell in the answer to the engine's neutral "Are you sure?" — evasive /
# over-assuring when bluffing; both banks mix registers (no lexical/register tell).
NEUTRAL_QUESTION = "Are you sure?"
ANSWER_EVASIVE = [
    "Why would you even ask me that?",
    "I mean, does it really matter right now?",
    "I swear on everything, it is completely real.",
    "Look, just trust me on this one.",
    "You're asking a lot of questions all of a sudden.",
    "Absolutely, one hundred percent, no doubt whatsoever.",
    "Let's just keep the game moving, shall we?",
    "I would never, ever lie about something like this.",
]
ANSWER_PLAIN = [
    "Yes.",
    "Yeah, that's what I played.",
    "Yes, those are the cards I put down.",
    "Sure, call it if you want.",
    "I am. Challenge if you like.",
    "Yep, that's my claim.",
    "Yes, I'm fine with that claim.",
    "That is what I played, yes.",
]
