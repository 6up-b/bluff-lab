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
