"""Game engine (SPEC §3): rules, features, card math, snapshot/restore, observation().

The engine is pure: `apply()` returns a list of (type, payload, actor) tuples that the World
emits through the EventWriter. Policies only ever see `observation(seat)`.
"""
from __future__ import annotations

import copy
import math
import re
from typing import Any

import numpy as np

SEATS = ("p1", "p2", "p3")
RANKS = ("K", "Q", "A")
DECK = ["K"] * 6 + ["Q"] * 6 + ["A"] * 6 + ["J"] * 2
DECK_SIZE = len(DECK)
RANK_NAME = {"K": "King", "Q": "Queen", "A": "Ace"}
HAND = 5

_CONTRACTION = re.compile(r"[A-Za-z]+['’][A-Za-z]+")


def has_contraction(text: str) -> bool:
    return bool(_CONTRACTION.search(text or ""))


def formal_register(text: str) -> bool:
    """§3.3: no contractions and >= 4 words."""
    return (not has_contraction(text)) and len((text or "").split()) >= 4


def matches(card: str, rank: str) -> bool:
    return card == rank or card == "J"


def is_bluff(cards: list[str], rank: str) -> bool:
    """§3.1: a bluff iff any card != table_rank and != J."""
    return any(not matches(c, rank) for c in cards)


def hypergeom_sf(U: int, m: int, h: int, k: int) -> float:
    """P(X >= k), X ~ Hypergeometric(population U, successes m, draws h)."""
    h = max(0, min(h, U))
    m = max(0, min(m, U))
    if k <= 0:
        return 1.0
    tot = math.comb(U, h)
    if tot == 0:
        return 0.0
    s = 0
    for x in range(k, min(h, m) + 1):
        s += math.comb(m, x) * math.comb(U - m, h - x)
    return s / tot


def card_math(deal_hand: list[str], rank: str, h: int, k: int) -> dict:
    """§3.5: probability the claimant could NOT have been truthful, from the observer's deal hand."""
    U = DECK_SIZE - len(deal_hand)
    m = 8 - sum(1 for c in deal_hand if matches(c, rank))
    p = 1.0 - hypergeom_sf(U, m, h, k)
    return {"U": U, "m": m, "h": h, "k": k, "p_cards": round(max(0.0, min(1.0, p)), 6)}


def _rng_state(g: np.random.Generator) -> dict:
    return copy.deepcopy(g.bit_generator.state)


def _rng_from_state(st: dict) -> np.random.Generator:
    bg = np.random.PCG64()
    bg.state = copy.deepcopy(st)
    return np.random.Generator(bg)


class Engine:
    def __init__(self, seed: int, wid: str = "w000", seats=SEATS, lives: int = 3, max_rounds: int = 30):
        self.seed = int(seed)
        self.wid = wid
        self.seats = list(seats)
        self.start_lives = lives
        self.max_rounds = max_rounds
        ch = np.random.SeedSequence(self.seed).spawn(4)
        self.deal_rng = np.random.Generator(np.random.PCG64(ch[0]))
        self.seat_rngs = {s: np.random.Generator(np.random.PCG64(ch[i + 1])) for i, s in enumerate(self.seats)}
        self.game = 0
        self.round = 0
        self.global_round = 0
        self.lives = {s: lives for s in self.seats}
        self.seat_order = list(self.seats)
        self.hands: dict[str, list[str]] = {s: [] for s in self.seats}
        self.deal_hands: dict[str, list[str]] = {s: [] for s in self.seats}
        self.table_rank: str | None = None
        self.pile: list[dict] = []          # claims this round (god view)
        self.pending: dict | None = None    # the claim awaiting a response
        self.to_act: str | None = None
        self.round_active = False
        self.game_over = True
        self.winner: str | None = None
        self.next_starter: str | None = None
        self.revealed: list[dict] = []      # all revealed claims in this world (public + per-seat p_cards)
        self.trusted_streak = {s: 0 for s in self.seats}
        self.challenged_last_round = {s: False for s in self.seats}

    # ------------------------------------------------------------ helpers
    def alive(self, s: str) -> bool:
        return self.lives[s] > 0

    def living(self) -> list[str]:
        return [s for s in self.seat_order if self.alive(s)]

    def next_living(self, s: str) -> str:
        order = self.seat_order
        i = order.index(s)
        for d in range(1, len(order) + 1):
            t = order[(i + d) % len(order)]
            if self.alive(t):
                return t
        return s

    # ------------------------------------------------------------ game / round
    def start_game(self, g: int) -> dict:
        self.game = g
        self.round = 0
        self.lives = {s: self.start_lives for s in self.seats}
        k = g % len(self.seats)
        self.seat_order = self.seats[k:] + self.seats[:k]
        self.next_starter = self.seat_order[0]
        self.game_over = False
        self.winner = None
        self.round_active = False
        self.pile, self.pending = [], None
        return {"seat_order": list(self.seat_order), "lives": dict(self.lives)}

    def start_round(self) -> dict:
        assert not self.game_over
        self.round += 1
        self.global_round += 1
        deck = list(DECK)
        perm = self.deal_rng.permutation(len(deck))
        deck = [deck[i] for i in perm]
        self.hands = {s: [] for s in self.seats}
        for s in self.living():
            self.hands[s] = [deck.pop() for _ in range(HAND)]
        self.deal_hands = {s: list(h) for s, h in self.hands.items()}
        self.table_rank = RANKS[int(self.deal_rng.integers(0, 3))]
        self.pile, self.pending = [], None
        self.round_active = True
        st = self.next_starter if (self.next_starter and self.alive(self.next_starter)) else self.living()[0]
        self.to_act = st
        evs = {"table_rank": self.table_rank, "hands": {s: list(self.hands[s]) for s in self.seats}}
        self._skip_empty()
        return evs

    def _skip_empty(self) -> list[tuple]:
        """With no pending claim, seats with 0 cards are skipped; all empty -> round ends, no loss."""
        if self.pending is not None or not self.round_active:
            return []
        seen = 0
        while self.round_active and not self.hands[self.to_act]:
            seen += 1
            self.to_act = self.next_living(self.to_act)
            if seen > len(self.seats):
                self._end_round(None)
                return []
        return []

    def legal_actions(self, seat: str) -> list[str]:
        if not self.round_active or seat != self.to_act:
            return []
        if self.pending is None:
            return ["play"] if self.hands[seat] else []
        if not self.hands[seat]:
            return ["challenge"]
        return ["challenge", "play"]

    # ------------------------------------------------------------ features
    def _features(self, claim: dict, observer: str) -> dict:
        f = dict(claim["features_public"])
        f["lives_self"] = int(claim["lives_at_claim"][observer])
        return f

    def _claim_card_math(self, claim: dict, observer: str) -> dict:
        return card_math(self.deal_hands[observer], claim["rank"], claim["h_before"], claim["count"])

    # ------------------------------------------------------------ actions
    def apply(self, seat: str, action: dict) -> list[tuple[str, dict, str | None]]:
        assert self.round_active and seat == self.to_act, (seat, self.to_act)
        legal = self.legal_actions(seat)
        typ = action.get("type")
        cards_idx = action.get("cards") or []
        hand = self.hands[seat]
        ok_cards = (isinstance(cards_idx, list) and 1 <= len(cards_idx) <= 3
                    and len(set(cards_idx)) == len(cards_idx)
                    and all(isinstance(i, int) and 0 <= i < len(hand) for i in cards_idx))
        if typ not in legal or (typ == "play" and not ok_cards):
            # legal fallback (counted by caller via the 'fallback' flag)
            if "challenge" in legal and (typ != "play" or not hand):
                action = {"type": "challenge", "fallback": True}
            else:
                action = {"type": "play", "cards": [0], "utterance": action.get("utterance", ""), "fallback": True}
            typ = action["type"]
            cards_idx = action.get("cards") or []
        out: list[tuple] = []
        if typ == "challenge":
            out += self._challenge(seat)
        else:
            if self.pending is not None:
                out.append(("play_on", {}, seat))
                self.trusted_streak[self.pending["by"]] += 1
            out += self._play(seat, cards_idx, action.get("utterance", "") or "", action.get("meta") or {})
        return out

    def _play(self, seat: str, idx: list[int], utterance: str, meta: dict) -> list[tuple]:
        hand = self.hands[seat]
        h_before = len(hand)
        cards = [hand[i] for i in idx]
        for i in sorted(idx, reverse=True):
            hand.pop(i)
        nxt = self.next_living(seat)
        fp = {
            "hand_size": len(hand), "claim_count": len(cards),
            "trusted_streak": int(self.trusted_streak[seat]),
            "lives_target": int(self.lives[seat]),
            "game_index": self.game, "round_index": self.round,
            "was_challenged_last_round": bool(self.challenged_last_round[seat]),
            "table_rank": self.table_rank, "utterance": utterance,
            "formal_register": formal_register(utterance),
        }
        claim = {"by": seat, "count": len(cards), "cards": cards, "rank": self.table_rank,
                 "utterance": utterance, "h_before": h_before, "features_public": fp,
                 "lives_at_claim": dict(self.lives), "meta": dict(meta)}
        self.pile.append(claim)
        self.pending = claim
        self.to_act = nxt
        feats = self._features(claim, nxt)
        feats["card_math"] = self._claim_card_math(claim, nxt)
        payload = {"count": len(cards), "claimed_rank": self.table_rank, "cards": list(cards),
                   "utterance": utterance, "features": feats}
        # god view: "strategic" | "forced" | None (honest) — used for RTD (§6)
        payload["bluff_kind"] = (meta.get("bluff_kind") or "forced") if is_bluff(cards, self.table_rank) else None
        return [("claim", payload, seat)]

    def annotate_pending(self, feats: dict) -> None:
        """Attach public answer_* features (§7.5) to the pending claim."""
        if self.pending is not None:
            self.pending["features_public"].update(feats)

    def _challenge(self, seat: str) -> list[tuple]:
        c = self.pending
        claimant = c["by"]
        bluff = is_bluff(c["cards"], c["rank"])
        loser = claimant if bluff else seat
        oid = f"{self.wid}:g{self.game}r{self.round}"
        entry = {"obs_id": oid, "game": self.game, "round": self.round, "by": claimant, "challenger": seat,
                 "count": c["count"], "was_bluff": bluff, "cards": list(c["cards"]), "utterance": c["utterance"],
                 "features_public": dict(c["features_public"]), "lives_at_claim": dict(c["lives_at_claim"]),
                 "card_math_by_seat": {s: self._claim_card_math(c, s) for s in self.seats},
                 "bluff_kind": (c["meta"].get("bluff_kind") or "forced") if bluff else None}
        self.revealed.append(entry)
        self.lives[loser] -= 1
        self.trusted_streak[claimant] = 0
        out = [("challenge", {"target": claimant}, seat),
               ("reveal", {"cards": list(c["cards"]), "was_bluff": bluff, "loser": loser, "claimant": claimant,
                           "challenger": seat, "obs_id": oid}, None),
               ("life_lost", {"seat": loser, "lives_left": self.lives[loser]}, loser)]
        self.challenged_last_round = {s: (s == claimant) for s in self.seats}
        self._end_round(loser)
        return out

    def _end_round(self, loser: str | None) -> None:
        self.round_active = False
        self.pending = None
        self.to_act = None
        if loser is not None:
            self.next_starter = loser if self.alive(loser) else self.next_living(loser)
        else:
            self.challenged_last_round = {s: False for s in self.seats}
            self.next_starter = self.next_living(self.next_starter or self.seat_order[0])
        liv = [s for s in self.seats if self.alive(s)]
        if len(liv) <= 1:
            self.game_over, self.winner = True, (liv[0] if liv else None)
        elif self.round >= self.max_rounds:
            best = max(self.lives.values())
            top = [s for s in self.seats if self.lives[s] == best]
            self.game_over, self.winner = True, (top[0] if len(top) == 1 else None)

    def game_end_payload(self) -> dict:
        return {"winner": self.winner, "lives": dict(self.lives)}

    # ------------------------------------------------------------ observation (§3.4)
    def revealed_entry_for(self, e: dict, seat: str) -> dict:
        f = dict(e["features_public"])
        f["lives_self"] = int(e["lives_at_claim"][seat])
        cm = e["card_math_by_seat"][seat]
        return {"obs_id": e["obs_id"], "game": e["game"], "round": e["round"], "by": e["by"],
                "challenger": e["challenger"], "count": e["count"], "was_bluff": e["was_bluff"],
                "cards": list(e["cards"]), "utterance": e["utterance"], "features": f,
                "p_cards": cm["p_cards"], "card_math": dict(cm)}

    def observation(self, seat: str) -> dict:
        pend = None
        if self.pending is not None and self.round_active:
            c = self.pending
            f = self._features(c, seat)
            f["card_math"] = self._claim_card_math(c, seat)
            pend = {"by": c["by"], "count": c["count"], "utterance": c["utterance"], "features": f}
        rc = [{"by": c["by"], "count": c["count"], "utterance": c["utterance"]}
              for c in self.pile if c is not self.pending]
        return {
            "seat": seat, "world": self.wid, "game": self.game, "round": self.round,
            "table_rank": self.table_rank, "my_hand": list(self.hands[seat]),
            "lives": dict(self.lives),
            "hand_sizes": {s: len(self.hands[s]) for s in self.seats if s != seat and self.alive(s)},
            "pending_claim": pend, "round_claims": rc,
            "revealed_history": [self.revealed_entry_for(e, seat) for e in self.revealed],
            "legal_actions": self.legal_actions(seat),
        }

    # ------------------------------------------------------------ snapshot / restore (§3.2)
    _FIELDS = ("seed", "wid", "seats", "start_lives", "max_rounds", "game", "round", "global_round", "lives",
               "seat_order", "hands", "deal_hands", "table_rank", "pile", "to_act", "round_active", "game_over",
               "winner", "next_starter", "revealed", "trusted_streak", "challenged_last_round")

    def snapshot(self) -> dict:
        d = {k: copy.deepcopy(getattr(self, k)) for k in self._FIELDS}
        d["pending_idx"] = (self.pile.index(self.pending) if self.pending is not None else None)
        d["rng"] = {"deal": _rng_state(self.deal_rng), **{s: _rng_state(g) for s, g in self.seat_rngs.items()}}
        return d

    @classmethod
    def restore(cls, d: dict, wid: str | None = None) -> "Engine":
        e = cls(d["seed"], wid=d["wid"], seats=d["seats"], lives=d["start_lives"], max_rounds=d["max_rounds"])
        for k in cls._FIELDS:
            setattr(e, k, copy.deepcopy(d[k]))
        e.pending = e.pile[d["pending_idx"]] if d.get("pending_idx") is not None else None
        e.deal_rng = _rng_from_state(d["rng"]["deal"])
        e.seat_rngs = {s: _rng_from_state(d["rng"][s]) for s in e.seats}
        if wid:
            e.wid = wid
        return e
