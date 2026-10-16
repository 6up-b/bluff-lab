"""Shared, neutral helpers for every agent arm (core-owned).

Raw-information mode (`info: "raw"`, a p1 seat param): every learner sees the SAME raw public information and
nothing hand-engineered toward the telltale's mechanism. In raw mode:
- `formal_register` and the answer_* tell-derived flags are stripped from every feature dict a policy sees
  (pending_claim, revealed_history); the god-view `claim` event keeps them for evaluation.
- a generic, neutral text statistic `word_count` (number of whitespace-separated words in the utterance)
  is added. It is not tell-specific: it is the most basic length statistic of any string.
- the predicate DSL rejects `formal_register` and the `no_contractions` op (they encode our knowledge of the
  tell); `contains` / `regex` stay as generic text ops.

`public_obs_text(obs)` renders an observation into the plain text every LLM arm receives, byte-identical
across arms.
"""
from __future__ import annotations

import copy
import json

TELL_DERIVED = ("formal_register", "answer_formal_register", "answer_has_contraction")
RAW_FORBIDDEN_FEATURES = frozenset(TELL_DERIVED)
RAW_FORBIDDEN_OPS = frozenset({"no_contractions"})
RANK_NAME = {"K": "King", "Q": "Queen", "A": "Ace"}


def word_count(text: str | None) -> int:
    return len((text or "").split())


def raw_features(f: dict | None) -> dict:
    """Copy of a features dict with tell-derived fields removed and word_count (+ answer_word_count) added."""
    out = {k: copy.deepcopy(v) for k, v in (f or {}).items() if k not in TELL_DERIVED}
    if "utterance" in out:
        out["word_count"] = word_count(out["utterance"])
    if "answer_utterance" in out:
        out["answer_word_count"] = word_count(out["answer_utterance"])
    return out


def strip_obs(obs: dict) -> dict:
    """The raw-mode observation: same structure, tell-derived precomputation removed everywhere."""
    o = dict(obs)
    if o.get("pending_claim"):
        pc = dict(o["pending_claim"])
        pc["features"] = raw_features(pc.get("features"))
        o["pending_claim"] = pc
    o["revealed_history"] = [dict(e, features=raw_features(e.get("features"))) for e in obs.get("revealed_history", [])]
    o["round_claims"] = [dict(c) for c in obs.get("round_claims", [])]
    o["info"] = "raw"
    return o


def _claim_line(e: dict) -> str:
    f = e.get("features") or {}
    rank = f.get("table_rank", "?")
    out = "bluff" if e.get("was_bluff") else "honest"
    return (f"g{e.get('game')}r{e.get('round')} {e.get('by')} claimed {e.get('count')}x{rank} "
            f"said {json.dumps(e.get('utterance', ''))} | hand_size_after={f.get('hand_size')} "
            f"trusted_streak={f.get('trusted_streak')} lives_target={f.get('lives_target')} "
            f"challenged_last_round={f.get('was_challenged_last_round')} p_cards={e.get('p_cards')} "
            f"| challenged by {e.get('challenger')} -> {out}, cards {''.join(e.get('cards') or [])}")


def public_obs_text(obs: dict, *, about: str | None = None, max_history: int | None = None) -> str:
    """Plain-text rendering of an observation (raw or engineered — call strip_obs first for raw arms).
    `about`: only revealed claims by that seat; `max_history`: keep the last N revealed claims.
    Deterministic: the same obs gives byte-identical text."""
    hist = [e for e in obs.get("revealed_history", []) if about is None or e.get("by") == about]
    if max_history is not None:
        hist = hist[-max_history:]
    lines = [f"You are {obs.get('seat')}. Game {obs.get('game')}, round {obs.get('round')}. "
             f"Table rank: {obs.get('table_rank')} ({RANK_NAME.get(obs.get('table_rank'), '?')}; Jokers are wild).",
             f"Your hand: {' '.join(obs.get('my_hand') or []) or '(empty)'}",
             "Lives: " + ", ".join(f"{s}={v}" for s, v in sorted((obs.get("lives") or {}).items())),
             "Opponent hand sizes: " + ", ".join(f"{s}={v}" for s, v in sorted((obs.get("hand_sizes") or {}).items()))]
    for c in obs.get("round_claims", []):
        lines.append(f"Earlier this round: {c['by']} claimed {c['count']} and said {json.dumps(c.get('utterance', ''))}")
    pc = obs.get("pending_claim")
    if pc:
        f = pc.get("features") or {}
        cm = f.get("card_math") or {}
        lines.append(f"PENDING CLAIM by {pc['by']}: {pc['count']}x{obs.get('table_rank')}, said "
                     f"{json.dumps(pc.get('utterance', ''))} | hand_size_after={f.get('hand_size')} "
                     f"trusted_streak={f.get('trusted_streak')} lives_target={f.get('lives_target')} "
                     f"challenged_last_round={f.get('was_challenged_last_round')} "
                     f"p_cards={cm.get('p_cards')} (prob. the claim is impossible given your hand)")
        if obs.get("ask"):
            lines.append(f"You asked: {json.dumps(obs['ask'].get('text', ''))} They answered: "
                         f"{json.dumps(obs['ask'].get('answer', ''))}")
    lines.append(f"Legal actions: {', '.join(obs.get('legal_actions') or [])}")
    lines.append(f"Revealed claims ({len(hist)}):")
    lines += [_claim_line(e) for e in hist]
    return "\n".join(lines)
