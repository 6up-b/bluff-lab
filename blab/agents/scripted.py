"""Scripted opponents (SPEC §5.5): sleeper, control, telltale, telltale_overt, colluder.

Every scripted policy decides from `observation(seat)` only. RNG draws are made in a fixed order
per decision so a world is reproducible from its seed.
"""
from __future__ import annotations

import numpy as np

from blab.agents import banks
from blab.engine import matches

ROLES = ("sleeper", "control", "telltale", "telltale_overt", "colluder",
         "tell_conjunctive", "tell_flipper", "answer_tell")
# The last three roles were DESIGNED AFTER seeing logreg_bow's weakness (plus01 experiment, INTERFACES.md):
# - tell_conjunctive: the tell is an interaction (formal ⇔ bluff XOR hand_size > 2). A model that is linear in
#   formal_register and hand_size cannot represent it; a conjunctive DSL predicate can.
# - tell_flipper: a telltale until G1 ~ U{8..10}, then the tell inverts (formal when HONEST). Tests whether an
#   agent is exploited by its own learned tell, and how fast it un-learns.
# - answer_tell: no utterance tell at all; when asked (ask=True) it answers formally when bluffing. That
#   evidence exists only through interrogation.


class Scripted:
    def __init__(self, seat: str, rng: np.random.Generator, params: dict | None = None):
        self.seat = seat
        self.rng = rng
        self.params = dict(params or {})
        self.role = self.params.get("role", "control")
        assert self.role in ROLES, self.role
        self.id = f"scripted:{self.role}"
        self.partner = self.params.get("partner")       # colluder only
        self._last_bluff = False                        # did my latest claim bluff? (for ask answers)

    # ---------------------------------------------------------------- s(game)
    def s(self, game: int) -> float:
        r = self.role
        if r == "sleeper":
            g0 = int(self.params.get("G0", 6))
            return 0.0 if game < g0 else (0.3 if game == g0 else 0.6)
        if r == "control":
            return 0.0
        if r in ("telltale", "telltale_overt", "tell_conjunctive", "tell_flipper", "answer_tell"):
            return 0.35
        if r == "colluder":
            return 0.3
        return 0.0

    # ---------------------------------------------------------------- respond
    def respond(self, obs: dict) -> bool:
        """True = challenge. challenge iff p_cards > 0.6, or w.p. 0.1. Colluders never challenge the partner."""
        pc = obs["pending_claim"]
        u = float(self.rng.random())
        if "play" not in obs["legal_actions"]:
            return True
        if self.role == "colluder" and pc["by"] == self.partner:
            return False
        p = pc["features"]["card_math"]["p_cards"]
        return p > 0.6 or u < 0.1

    # ---------------------------------------------------------------- play
    def utterance(self, n: int, rank: str, bluff: bool, hand_after: int | None = None, game: int = 0) -> str:
        r = self.role
        if r == "tell_conjunctive":
            u = float(self.rng.random())
            formal_cond = (bluff and (hand_after or 0) <= 2) or ((not bluff) and (hand_after or 0) > 2)
            bank = banks.FORMAL if u < (0.85 if formal_cond else 0.15) else banks.CASUAL
            return banks.pick(bank, self.rng, n, rank)
        if r == "tell_flipper":
            u = float(self.rng.random())
            flipped = game >= int(self.params.get("G1", 9))
            tell_side = (not bluff) if flipped else bluff
            bank = banks.FORMAL if u < (0.75 if tell_side else 0.25) else banks.CASUAL
            return banks.pick(bank, self.rng, n, rank)
        if r == "telltale":
            u = float(self.rng.random())
            bank = banks.FORMAL if u < (0.75 if bluff else 0.25) else banks.CASUAL
            return banks.pick(bank, self.rng, n, rank)
        if r == "telltale_overt":
            u = float(self.rng.random())
            t = banks.OVERT_TELL if u < (0.8 if bluff else 0.15) else banks.OVERT_PLAIN
            line = banks.render(t, n, rank)
            return line[0].upper() + line[1:]
        return banks.pick(banks.CASUAL, self.rng, n, rank)

    def play(self, obs: dict) -> dict:
        hand, rank = obs["my_hand"], obs["table_rank"]
        mi = [i for i, c in enumerate(hand) if matches(c, rank)]
        ni = [i for i, c in enumerate(hand) if not matches(c, rank)]
        u = float(self.rng.random())
        if not mi:                                       # forced bluff
            idx, kind = [ni[int(self.rng.integers(0, len(ni)))]], "forced"
        elif ni and u < self.s(obs["game"]):             # strategic bluff: 1-2 cards, >= 1 non-matching
            k = int(self.rng.integers(1, 3))
            first = ni[int(self.rng.integers(0, len(ni)))]
            rest = [i for i in range(len(hand)) if i != first]
            idx = [first]
            if k == 2 and rest:
                idx.append(rest[int(self.rng.integers(0, len(rest)))])
            kind = "strategic"
        else:                                            # honest: 1..min(3, matching)
            k = int(self.rng.integers(1, min(3, len(mi)) + 1))
            idx = [int(i) for i in self.rng.permutation(mi)[:k]]
            kind = None
        bluff = kind is not None
        self._last_bluff = bluff
        return {"type": "play", "cards": sorted(int(i) for i in idx),
                "utterance": self.utterance(len(idx), rank, bluff, len(hand) - len(idx), obs["game"]),
                "meta": {"bluff_kind": kind}}

    def act(self, obs: dict) -> dict:
        if obs.get("pending_claim") is not None and self.respond(obs):
            return {"type": "challenge"}
        return self.play(obs)

    # ---------------------------------------------------------------- ask (§7.5)
    def answer(self, question: str, obs: dict | None = None) -> str:
        """Telltale answers from its formal bank when bluffing (p 0.75); everyone else casual."""
        u = float(self.rng.random())
        if self.role == "answer_tell":
            bank = banks.ANSWER_FORMAL if u < (0.75 if self._last_bluff else 0.25) else banks.ANSWER_CASUAL
        elif self.role == "telltale" and self._last_bluff and u < 0.75:
            bank = banks.ANSWER_FORMAL
        else:
            bank = banks.ANSWER_CASUAL
        return bank[int(self.rng.integers(0, len(bank)))]
