"""Scripted opponents (SPEC §5.5): sleeper, control, telltale, telltale_overt, colluder.

Every scripted policy decides from `observation(seat)` only. RNG draws are made in a fixed order
per decision so a world is reproducible from its seed.
"""
from __future__ import annotations

import numpy as np

from blab.agents import banks
from blab.engine import matches

ROLES = ("sleeper", "control", "telltale", "telltale_overt", "colluder",
         "tell_conjunctive", "tell_flipper", "answer_tell", "tell_relational", "tell_relational_v2")
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
        self._ranks: dict = {}                          # tell_relational: (game, round) -> table rank seen

    # ---------------------------------------------------------------- s(game)
    def s(self, game: int) -> float:
        r = self.role
        if r == "sleeper":
            g0 = int(self.params.get("G0", 6))
            return 0.0 if game < g0 else (0.3 if game == g0 else 0.6)
        if r == "control":
            return 0.0
        if r in ("telltale", "telltale_overt", "tell_conjunctive", "tell_flipper", "answer_tell", "tell_relational", "tell_relational_v2"):
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

    def relational_utterance(self, obs: dict, n: int, bluff: bool) -> str:
        """b01 tell_relational: claim line (casual or formal, independent of bluffing) + ONE factual statement
        computable from the public log. Bluff -> statement false w.p. 0.7; honest -> false w.p. 0.1.
        RNG order: claim bank, claim line, statement kind, template, truth, perturbation."""
        rank, g, r = obs["table_rank"], obs["game"], obs["round"]
        self._ranks[(g, r)] = rank
        bank = banks.FORMAL if float(self.rng.random()) < 0.5 else banks.CASUAL
        claim = banks.pick(bank, self.rng, n, rank)
        prev = self._ranks.get((g, r - 1))
        kinds = ["count", "hand"] + (["same"] if prev is not None else [])
        kind = kinds[int(self.rng.integers(0, len(kinds)))]
        ti = int(self.rng.integers(0, 4))
        false = float(self.rng.random()) < (0.7 if bluff else 0.1)
        d = 1 if float(self.rng.random()) < 0.5 else -1
        pend = obs.get("pending_claim")
        if kind == "count":
            N = sum(int(c["count"]) for c in obs.get("round_claims", [])) + (int(pend["count"]) if pend else 0) + n
            if false:
                N = N + d if N + d >= 1 else N + 1
            stmt = banks.REL_COUNT[ti].format(N=N, rank_pl=banks.ranks_word(rank, 2))
        elif kind == "hand":
            h = len(obs["my_hand"]) - n
            if false:
                h = h + d if h + d >= 0 else h + 1
            stmt = banks.REL_HAND[ti].format(h=h)
        else:
            same = (rank == prev) != false
            stmt = (banks.REL_SAME if same else banks.REL_DIFF)[ti]
        self.rel_last = {"kind": kind, "false": bool(false), "bluff": bool(bluff)}     # audit only (no RNG use)
        return f"{claim} {stmt}"

    def relational_v2_utterance(self, obs: dict, n: int, bluff: bool) -> str:
        """c01 (PREREG_c01.md) rebuilt relational tell. Same mechanism as tell_relational (statement false w.p. 0.7
        when bluffing, 0.1 when honest) but true and false statements use IDENTICAL wording: the only statement
        names the previous round's table rank, and a false one names one of the two other ranks (uniformly).
        Table ranks are iid uniform, so the named rank is uniform and independent of the current rank whether the
        statement is true or false. Round 1 of a game has no previous round in the game: claim line only.
        RNG order: claim bank, claim line, [template, truth, alternative] (bracket only when round >= 2)."""
        rank, g, r = obs["table_rank"], obs["game"], obs["round"]
        self._ranks[(g, r)] = rank
        bank = banks.FORMAL if float(self.rng.random()) < 0.5 else banks.CASUAL
        claim = banks.pick(bank, self.rng, n, rank)
        prev = self._ranks.get((g, r - 1))
        if prev is None:
            self.rel_last = {"kind": "none", "false": None, "bluff": bool(bluff)}
            return claim
        ti = int(self.rng.integers(0, len(banks.REL2_PREV)))
        false = float(self.rng.random()) < (0.7 if bluff else 0.1)
        others = [x for x in ("K", "Q", "A") if x != prev]
        alt = others[int(self.rng.integers(0, 2))]
        named = alt if false else prev
        self.rel_last = {"kind": "prev", "false": bool(false), "bluff": bool(bluff)}
        return f"{claim} {banks.REL2_PREV[ti].format(R=banks.ranks_word(named, 2))}"

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
                "utterance": (self.relational_utterance(obs, len(idx), bluff) if self.role == "tell_relational"
                              else self.relational_v2_utterance(obs, len(idx), bluff) if self.role == "tell_relational_v2"
                              else self.utterance(len(idx), rank, bluff, len(hand) - len(idx), obs["game"])),
                "meta": {"bluff_kind": kind}}

    def act(self, obs: dict) -> dict:
        if obs.get("pending_claim") is not None and self.respond(obs):
            return {"type": "challenge"}
        return self.play(obs)

    # ---------------------------------------------------------------- ask (§7.5)
    def answer(self, question: str, obs: dict | None = None) -> str:
        """Telltale answers from its formal bank when bluffing (p 0.75); everyone else casual."""
        u = float(self.rng.random())
        if self.role == "answer_tell":       # b01: semantic tell (evasive / over-assuring when bluffing)
            bank = banks.ANSWER_EVASIVE if u < (0.75 if self._last_bluff else 0.25) else banks.ANSWER_PLAIN
        elif self.role == "telltale" and self._last_bluff and u < 0.75:
            bank = banks.ANSWER_FORMAL
        else:
            bank = banks.ANSWER_CASUAL
        return bank[int(self.rng.integers(0, len(bank)))]
