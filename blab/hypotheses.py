"""Hypothesis graph: predicate DSL, scoring, supersession, drift (SPEC §5.2, §5.3).

Pure functions over plain dicts (JSON-serializable memory). Numeric posteriors come from
`blab.grid`; nothing here touches the network or the event log directly — functions return
`(event_type, payload)` tuples that the caller emits.

Predicate DSL
-------------
leaf        {"f": <feature>, "op": <op>, "v": <value>}
combinators {"all": [...]}, {"any": [...]}, {"not": {...}}
`{"all": []}` is the empty conjunction: valid, always true (used for `baseline_bluff`).

The evaluator is a small recursive interpreter over the dict — never `eval`/`exec`. Values
are compared as data; a value such as `"__import__('os')"` is just a string.

String ops: `contains` and `regex` are case-insensitive; `==`/`!=` are exact.
`no_contractions` is true when the string has no apostrophe contraction (ASCII ' or ’ between
letters, e.g. "That's", "don’t"); `v` (default true) is the expected truth value.

Regex guard (the "50 ms timeout")
---------------------------------
CPython's `re` has no timeout and holds the GIL, so a runaway match cannot be interrupted.
Instead we make runaway matches impossible:
1. Patterns > 100 chars are rejected.
2. At validation the pattern is parsed (`re._parser`) and rejected if any unbounded/multi
   repeat contains another repeat or an alternation (`(a+)+`, `(\\w+ )*`, `(a|aa)+`: the
   exponential-backtracking shapes), if it uses backreferences, or has > 8 repeats.
3. Matching runs on input truncated to 300 chars (utterances are ≤ 300 chars).
4. As a final check, validation times the compiled pattern on adversarial probe strings of
   300 chars and rejects it if any single search takes > 50 ms.
With (2)+(3) the worst case is polynomial on 300 chars, and (4) bounds that empirically.
`eval_predicate` only ever runs regexes that passed this guard (unvalidated ones → no match).
"""
from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Any

import numpy as np

from blab import grid

ASPECTS = ("baseline_bluff", "trigger", "verbal_tell", "pressure", "exploits_trust", "collusion")

FEATURES: dict[str, type] = {
    "hand_size": int,
    "claim_count": int,
    "trusted_streak": int,
    "lives_self": int,
    "lives_target": int,
    "game_index": int,
    "round_index": int,
    "was_challenged_last_round": bool,
    "table_rank": str,
    "utterance": str,
    "formal_register": bool,
    "word_count": int,          # generic text statistic (raw-information mode; see blab/agents/common.py)
    "llm_record_check": str,    # c01: generic per-claim LLM record check, "true"|"false"|"none" (blab/agents/c01.py)
}
# raw-information mode: features/ops that encode our knowledge of the telltale are not allowed
RAW_FORBIDDEN_FEATURES = frozenset({"formal_register", "answer_formal_register", "answer_has_contraction"})
RAW_FORBIDDEN_OPS = frozenset({"no_contractions"})

OPS = ("==", "!=", "<", "<=", ">", ">=", "contains", "regex", "no_contractions")
_CMP_OPS = ("<", "<=", ">", ">=")
MAX_DEPTH = 6
MAX_LEAVES = 32
MAX_REGEX_LEN = 100
MAX_INPUT_LEN = 300
REGEX_TIMEOUT_S = 0.050
N_SAMPLES = 2000

CONFIRM_N, CONFIRM_P = 4, 0.9
REFUTE_N, REFUTE_P = 6, 0.2
GAP = 0.1
# core deviation from SPEC §5.2 (tuned on seeds 1001-1030, see INTERFACES "Status: core"): the recent-window
# refute needs >= 4 matching claims in the window and P < 0.2 (SPEC: P < 0.3, no minimum) — the spec rule refuted
# the true telltale tell on noise in 11/30 fresh worlds (seeds 2001-2030); this rule: 5/30.
WINDOW_GAMES, WINDOW_P, WINDOW_MIN_TRUE = 2, 0.2, 4
# §5.3 drift detection knobs (module-level so core can tune them; see INTERFACES "Status: core")
DRIFT_W, DRIFT_DELTA, DRIFT_THRESH, DRIFT_MIN_OLD = 8, 0.2, 0.9, 4

_CONTRACTION = re.compile(r"[A-Za-z][’'][A-Za-z]")


def has_contraction(s: str) -> bool:
    return bool(_CONTRACTION.search(s or ""))


def formal_register(utterance: str) -> bool:
    """§3.3: no contractions and ≥ 4 words."""
    return (not has_contraction(utterance)) and len((utterance or "").split()) >= 4


# ---------------------------------------------------------------- regex guard

try:  # Python 3.11+
    from re import _constants as _sc  # type: ignore[attr-defined]
    from re import _parser as _sp  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    import sre_constants as _sc  # type: ignore
    import sre_parse as _sp  # type: ignore

_REPEATS = {_sc.MAX_REPEAT, _sc.MIN_REPEAT} | ({_sc.POSSESSIVE_REPEAT} if hasattr(_sc, "POSSESSIVE_REPEAT") else set())


def _children(op, av) -> list:
    """Sub-patterns of one parsed regex node."""
    if op in _REPEATS:
        return [av[2]]
    if op is _sc.SUBPATTERN:
        return [av[-1]]
    if op is _sc.BRANCH:
        return list(av[1])
    if op in (_sc.ASSERT, _sc.ASSERT_NOT):
        return [av[1]]
    if op is _sc.GROUPREF_EXISTS:
        return [x for x in av[1:] if x is not None]
    if hasattr(_sc, "ATOMIC_GROUP") and op is _sc.ATOMIC_GROUP:
        return [av]
    return []


def _walk(p, inside_repeat: bool, stats: dict) -> str | None:
    for op, av in p:
        if op in (_sc.GROUPREF, _sc.GROUPREF_EXISTS):
            return "backreferences are not allowed"
        is_rep = op in _REPEATS and av[1] > 1
        if is_rep:
            stats["repeats"] += 1
            if inside_repeat:
                return "nested quantifier (catastrophic backtracking)"
        if op is _sc.BRANCH and inside_repeat:
            return "alternation inside a quantifier (catastrophic backtracking)"
        for ch in _children(op, av):
            r = _walk(ch, inside_repeat or is_rep, stats)
            if r:
                return r
    return None


_PROBES = ("a" * MAX_INPUT_LEN, " " * MAX_INPUT_LEN, "ab " * (MAX_INPUT_LEN // 3),
           ("I have placed two Kings. " * 12)[:MAX_INPUT_LEN], "a" * (MAX_INPUT_LEN - 1) + "!")


@lru_cache(maxsize=512)
def _check_regex(pattern: str) -> tuple[bool, str]:
    if len(pattern) > MAX_REGEX_LEN:
        return False, f"regex longer than {MAX_REGEX_LEN} chars"
    try:
        parsed = _sp.parse(pattern, re.IGNORECASE)
        rx = re.compile(pattern, re.IGNORECASE)
    except (re.error, RecursionError, OverflowError) as e:
        return False, f"invalid regex: {e}"
    stats = {"repeats": 0}
    bad = _walk(parsed, False, stats)
    if bad:
        return False, bad
    if stats["repeats"] > 8:
        return False, "too many quantifiers"
    for s in _PROBES:
        t0 = time.perf_counter()
        rx.search(s)
        if time.perf_counter() - t0 > REGEX_TIMEOUT_S:
            return False, "regex exceeded 50 ms on a probe string"
    return True, "ok"


@lru_cache(maxsize=512)
def _compiled(pattern: str) -> re.Pattern | None:
    ok, _ = _check_regex(pattern)
    return re.compile(pattern, re.IGNORECASE) if ok else None


# ---------------------------------------------------------------- DSL

def _feature_type(f: str) -> type | None:
    """Declared type; `answer_<feature>` inherits it; other `answer_*` → None (any scalar)."""
    if f in FEATURES:
        return FEATURES[f]
    return FEATURES.get(f[len("answer_"):])


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _validate_leaf(p: dict) -> tuple[bool, str]:
    extra = set(p) - {"f", "op", "v"}
    if extra:
        return False, f"unknown leaf keys {sorted(extra)}"
    f, op = p.get("f"), p.get("op")
    if not isinstance(f, str):
        return False, "leaf 'f' must be a feature name"
    if f not in FEATURES and not (f.startswith("answer_") and re.fullmatch(r"answer_[a-z0-9_]{1,40}", f)):
        return False, f"unknown feature {f!r}"
    if op not in OPS:
        return False, f"unknown op {op!r}"
    if op != "no_contractions" and "v" not in p:
        return False, f"op {op!r} needs a value 'v'"
    v = p.get("v", True)
    t = _feature_type(f)
    if t is None:  # free answer_* feature: infer from the value
        t = bool if isinstance(v, bool) else int if _is_num(v) else str if isinstance(v, str) else None
        if t is None:
            return False, f"value for {f!r} must be a scalar"
    if op in ("contains", "regex", "no_contractions") and t is not str:
        return False, f"op {op!r} needs a string feature, {f!r} is {t.__name__}"
    if op == "no_contractions":
        return (True, "ok") if isinstance(v, bool) else (False, "no_contractions takes a bool 'v'")
    if t is bool:
        if op not in ("==", "!="):
            return False, f"op {op!r} not valid for bool feature {f!r}"
        return (True, "ok") if isinstance(v, bool) else (False, f"{f!r} needs a bool value")
    if t is int:
        if op not in ("==", "!=") + _CMP_OPS:
            return False, f"op {op!r} not valid for numeric feature {f!r}"
        return (True, "ok") if _is_num(v) else (False, f"{f!r} needs a numeric value")
    # str
    if not isinstance(v, str):
        return False, f"{f!r} needs a string value"
    if op in _CMP_OPS:
        return False, f"op {op!r} not valid for string feature {f!r}"
    if len(v) > MAX_INPUT_LEN:
        return False, "string value too long"
    if op == "regex":
        return _check_regex(v)
    return True, "ok"


def _validate(p: Any, depth: int, counter: list[int]) -> tuple[bool, str]:
    if depth > MAX_DEPTH:
        return False, f"nesting depth > {MAX_DEPTH}"
    if not isinstance(p, dict):
        return False, "predicate node must be an object"
    combs = [k for k in ("all", "any", "not") if k in p]
    if combs:
        if len(p) != 1:
            return False, "combinator node must have exactly one key"
        k = combs[0]
        if k == "not":
            return _validate(p["not"], depth + 1, counter)
        kids = p[k]
        if not isinstance(kids, list):
            return False, f"'{k}' takes a list"
        for c in kids:
            ok, why = _validate(c, depth + 1, counter)
            if not ok:
                return ok, why
        return True, "ok"
    counter[0] += 1
    if counter[0] > MAX_LEAVES:
        return False, f"more than {MAX_LEAVES} leaves"
    return _validate_leaf(p)


def _raw_violation(p: Any) -> str | None:
    if isinstance(p, dict):
        if p.get("f") in RAW_FORBIDDEN_FEATURES:
            return f"feature {p['f']!r} is not available in raw-information mode"
        if p.get("op") in RAW_FORBIDDEN_OPS:
            return f"op {p['op']!r} is not available in raw-information mode"
        for k in ("all", "any"):
            for c in p.get(k) or []:
                r = _raw_violation(c)
                if r:
                    return r
        if "not" in p:
            return _raw_violation(p["not"])
    return None


def validate_predicate(pred: dict, mode: str = "engineered") -> tuple[bool, str]:
    """(ok, reason). Rejects unknown features/ops, type mismatches, unsafe/long regex, depth > 6.
    mode="raw" additionally rejects formal_register (and answer_ tell flags) and the no_contractions op."""
    try:
        ok, why = _validate(pred, 1, [0])
    except RecursionError:
        return False, f"nesting depth > {MAX_DEPTH}"
    if ok and mode == "raw":
        bad = _raw_violation(pred)
        if bad:
            return False, bad
    return ok, why


def _eval_leaf(p: dict, features: dict) -> bool:
    f, op = p["f"], p["op"]
    v = p.get("v", True)
    x = features.get(f)
    if x is None:
        return False
    if op == "no_contractions":
        return isinstance(x, str) and ((not has_contraction(x[:MAX_INPUT_LEN])) == bool(v))
    if op == "contains":
        return isinstance(x, str) and isinstance(v, str) and v.lower() in x[:MAX_INPUT_LEN].lower()
    if op == "regex":
        rx = _compiled(v) if isinstance(v, str) else None
        return bool(isinstance(x, str) and rx is not None and rx.search(x[:MAX_INPUT_LEN]))
    if isinstance(v, bool) or isinstance(x, bool):
        if not (isinstance(v, bool) and isinstance(x, bool)):
            return False
    elif _is_num(v) != _is_num(x):
        return False
    try:
        if op == "==":
            return x == v
        if op == "!=":
            return x != v
        if not (_is_num(x) and _is_num(v)):
            return False
        return {"<": x < v, "<=": x <= v, ">": x > v, ">=": x >= v}[op]
    except (TypeError, KeyError):
        return False


def eval_predicate(pred: dict, features: dict) -> bool:
    """Interpret `pred` against a feature dict. Never raises on bad data; malformed → False."""
    if not isinstance(pred, dict):
        return False
    if "all" in pred:
        return all(eval_predicate(c, features) for c in pred["all"] or [])
    if "any" in pred:
        return any(eval_predicate(c, features) for c in pred["any"] or [])
    if "not" in pred:
        return not eval_predicate(pred["not"], features)
    if "f" in pred and "op" in pred:
        try:
            return _eval_leaf(pred, features or {})
        except Exception:
            return False
    return False


def specificity(pred: dict) -> int:
    """Number of leaves (the 'most specific' tie-break)."""
    if not isinstance(pred, dict):
        return 0
    if "all" in pred or "any" in pred:
        return sum(specificity(c) for c in (pred.get("all") or pred.get("any") or []))
    if "not" in pred:
        return specificity(pred["not"])
    return 1 if "f" in pred else 0


def is_always_true(pred: dict) -> bool:
    return specificity(pred) == 0 and eval_predicate(pred, {})


# ---------------------------------------------------------------- hypotheses

def _uniform_iv() -> list[float]:
    return list(grid.interval(grid.uniform()))


def new_hypothesis(*, wid: str, about: str, aspect: str, text: str, predicate: dict,
                   created_game: int, provenance: str = "evidence", idx: int,
                   supersedes: str | None = None, score_from_game: int | None = None,
                   origin: str = "llm", mode: str = "engineered", **extra: Any) -> dict:
    """§5.2 hypothesis dict + scoring fields. Raises ValueError on a bad aspect/provenance/predicate.

    `score_from_game` (default `created_game`): score on reveals with game > this.
    `origin` ∈ {"llm","drift","surgery","stub"} — informational; "drift" sets the supersession reason.
    `extra` is copied in (e.g. `q` override for a false_memory line, `rationale`, `surreal`).
    """
    if aspect not in ASPECTS:
        raise ValueError(f"unknown aspect {aspect!r}")
    if provenance not in ("evidence", "untrusted"):
        raise ValueError(f"bad provenance {provenance!r}")
    ok, why = validate_predicate(predicate, mode)
    if not ok:
        raise ValueError(f"invalid predicate: {why}")
    iv = _uniform_iv()
    h = {"id": f"h_{wid}_{about}_{idx:03d}", "idx": idx, "about": about, "aspect": aspect,
         "text": text, "predicate": predicate, "status": "provisional", "provenance": provenance,
         "created_game": created_game, "valid_from_game": created_game, "valid_until_game": None,
         "supersedes": supersedes, "superseded_by": None, "origin": origin,
         "score_from_game": created_game if score_from_game is None else score_from_game,
         "n_true": 0, "n_false": 0, "q_true": iv, "q_false": [] if is_always_true(predicate) else list(iv),
         "p_gap": None, "evidence_obs": [], "confirmed_by": None}
    h.update(extra)
    return h


def _p_of(e: dict) -> float:
    p = e.get("p_cards")
    if p is None:
        p = ((e.get("features") or {}).get("card_math") or {}).get("p_cards", 0.0)
    return float(p)


def _feats(e: dict) -> dict:
    f = dict(e.get("features") or {})
    if "utterance" not in f and e.get("utterance") is not None:
        f["utterance"] = e["utterance"]
    return f


def _relevant(h: dict, revealed: list[dict]) -> list[dict]:
    g0 = h.get("score_from_game", h.get("created_game", 0))
    out = [e for e in revealed if e.get("by") == h.get("about") and e.get("game", 0) > g0]
    return sorted(out, key=lambda e: (e.get("game", 0), e.get("round", 0)))


def _split(pred: dict, entries: list[dict]) -> tuple[list[dict], list[dict]]:
    t, f = [], []
    for e in entries:
        (t if eval_predicate(pred, _feats(e)) else f).append(e)
    return t, f


def _post(entries: list[dict]) -> np.ndarray:
    return grid.posterior((_p_of(e), bool(e.get("was_bluff"))) for e in entries)


def _gap(t: list[dict], f: list[dict], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    pt, pf = _post(t), _post(f)
    return pt, pf, grid.p_diff_greater(pt, pf, GAP, rng, N_SAMPLES)


def score_hypothesis(h: dict, revealed: list[dict], rng: np.random.Generator,
                     now_game: int | None = None) -> dict:
    """Re-score `h` in place (and return it) on reveals about h["about"] after score_from_game.

    Status (§5.2): confirmed (n_true ≥ 4, P > 0.9) / refuted (n_true ≥ 6, P < 0.2) / provisional,
    with P = P(q_true − q_false > 0.1) from 2,000 grid samples drawn from `rng`.
    - `refuted` is absorbing (a refuted line is never revived; reflection proposes a new one).
    - If `now_game` is given and h was confirmed, the recent-window rule (`recent_window_check`)
      is applied and can flip it to refuted.
    - Always-true predicates (`{"all":[]}`, baseline_bluff lines) have no false split: q_true is the
      posterior over all claims, q_false = [], p_gap = None, and status is confirmed iff n ≥ 4
      (else provisional) — they are never refuted by scoring, only superseded (drift).
    """
    entries = _relevant(h, revealed)
    pred = h["predicate"]
    prev = h.get("status", "provisional")
    if is_always_true(pred):
        h.update(n_true=len(entries), n_false=0, q_true=list(grid.interval(_post(entries))),
                 q_false=[], p_gap=None, evidence_obs=[e.get("obs_id") for e in entries if e.get("obs_id")])
        if prev != "refuted":
            h["status"] = "confirmed" if len(entries) >= CONFIRM_N else "provisional"
            h["confirmed_by"] = "scoring" if h["status"] == "confirmed" else h.get("confirmed_by")
        return h
    t, f = _split(pred, entries)
    pt, pf, p = _gap(t, f, rng)
    h.update(n_true=len(t), n_false=len(f), q_true=list(grid.interval(pt)), q_false=list(grid.interval(pf)),
             p_gap=round(p, 4), evidence_obs=[e.get("obs_id") for e in t if e.get("obs_id")])
    if prev == "refuted":
        return h
    if len(t) >= CONFIRM_N and p > CONFIRM_P:
        status = "confirmed"
    elif len(t) >= REFUTE_N and p < REFUTE_P:
        status = "refuted"
    else:
        status = "provisional"
    if prev == "confirmed" and now_game is not None and recent_window_check(h, revealed, now_game, rng):
        status = "refuted"
        h["refuted_reason"] = "recent_window"
    elif status == "refuted":
        h["refuted_reason"] = "scoring"
    h["status"] = status
    if status == "confirmed":
        h["confirmed_by"] = "scoring"
    return h


def recent_window_check(h: dict, revealed: list[dict], now_game: int, rng: np.random.Generator) -> bool:
    """True iff confirmed `h` FAILS over the last 2 games (games now_game-1, now_game):
    P(q_true − q_false > 0.1) < 0.3 on that window, with at least 2 matching claims in it
    (fewer is no evidence either way). Does not mutate h. Always-true predicates never fail."""
    if h.get("status") != "confirmed" or is_always_true(h["predicate"]):
        return False
    win = [e for e in _relevant(h, revealed) if e.get("game", 0) > now_game - WINDOW_GAMES]
    t, f = _split(h["predicate"], win)
    if len(t) < WINDOW_MIN_TRUE:
        return False
    _, _, p = _gap(t, f, rng)
    return p < WINDOW_P


def _order(h: dict) -> tuple:
    return (h.get("created_game", 0), h.get("idx", 0))


def supersede(hyps: list[dict], now_game: int, now_round: int) -> list[tuple[str, dict]]:
    """Newest confirmed line per (about, aspect) closes every older open confirmed line
    (valid_until_game = now_game, superseded_by = new id). Mutates `hyps`; returns
    ("belief_superseded", BeliefSuperseded payload) tuples. Reason is "drift" when the new line came
    from drift detection, else "llm". An open *refuted* line explicitly named in the new line's
    `supersedes` is also closed (reason "refuted")."""
    events: list[tuple[str, dict]] = []
    groups: dict[tuple, list[dict]] = {}
    for h in hyps:
        if h.get("valid_until_game") is None:
            groups.setdefault((h.get("about"), h.get("aspect")), []).append(h)
    for (about, aspect), hs in groups.items():
        conf = sorted([h for h in hs if h.get("status") == "confirmed"], key=_order)
        if not conf:
            continue
        new = conf[-1]
        for old in hs:
            if old is new or _order(old) >= _order(new):
                continue
            if old.get("status") == "confirmed":
                reason = "drift" if new.get("origin") == "drift" else "llm"
            elif old.get("status") == "refuted" and new.get("supersedes") == old["id"]:
                reason = "refuted"
            else:
                continue
            old["valid_until_game"] = now_game
            old["superseded_by"] = new["id"]
            events.append(("belief_superseded", {
                "aspect": aspect, "old_id": old["id"], "new_id": new["id"], "reason": reason,
                "at_game": now_game, "at_round": now_round, "about": about}))
    return events


# ---------------------------------------------------------------- drift

def detect_drift(revealed_about_opp: list[dict], rng: np.random.Generator, W: int | None = None,
                 delta: float | None = None, thresh: float | None = None,
                 min_old: int | None = None) -> tuple[bool, np.ndarray | None, float]:
    """§5.3: last W revealed claims vs all earlier ones. Returns (fired, recent_posterior, p)
    with p = P(q_recent − q_old > delta). Needs ≥ min_old older claims, else (False, None, 0.0).
    Defaults come from the module constants DRIFT_*."""
    W = DRIFT_W if W is None else W
    delta = DRIFT_DELTA if delta is None else delta
    thresh = DRIFT_THRESH if thresh is None else thresh
    min_old = DRIFT_MIN_OLD if min_old is None else min_old
    es = sorted(revealed_about_opp, key=lambda e: (e.get("game", 0), e.get("round", 0)))
    if len(es) < W + max(1, min_old):
        return False, None, 0.0
    recent, old = es[-W:], es[:-W]
    pr, po = _post(recent), _post(old)
    p = grid.p_diff_greater(pr, po, delta, rng, N_SAMPLES)
    return p > thresh, pr, p


def apply_drift(hyps: list[dict], revealed: list[dict], *, wid: str, about: str, idx: int,
                now_game: int, now_round: int, rng: np.random.Generator, W: int | None = None
                ) -> tuple[dict | None, list[tuple[str, dict]], float]:
    """Convenience for the profiler at game end: run detect_drift on `about`'s reveals and, if it
    fires, append a confirmed always-true `baseline_bluff` line scored on the recent window only
    (origin "drift") and supersede the previous baseline. If there was no confirmed baseline to
    close, a belief_superseded event with old_id=None is still emitted.
    After a drift line exists, detection only looks at reveals inside the new regime (from its
    window start), so it does not re-fire every game on the same change point.
    Returns (new_hypothesis | None, events, p)."""
    mine = sorted([e for e in revealed if e.get("by") == about], key=lambda e: (e.get("game", 0), e.get("round", 0)))
    cur = [h for h in hyps if h.get("about") == about and h.get("aspect") == "baseline_bluff"
           and h.get("origin") == "drift" and h.get("valid_until_game") is None]
    if cur:
        g0 = max(h["score_from_game"] for h in cur)
        mine = [e for e in mine if e.get("game", 0) > g0]
    W = DRIFT_W if W is None else W
    fired, _, p = detect_drift(mine, rng, W=W)
    if not fired:
        return None, [], p
    window = mine[-W:]
    start = window[0].get("game", now_game) - 1
    h = new_hypothesis(wid=wid, about=about, aspect="baseline_bluff",
                       text=f"{about}'s bluff rate shifted upward (drift over last {W} reveals)",
                       predicate={"all": []}, created_game=now_game, idx=idx, origin="drift",
                       score_from_game=start, drift_p=round(p, 4))
    h["supersedes"] = next((o["id"] for o in sorted(hyps, key=_order, reverse=True)
                            if o.get("about") == about and o.get("aspect") == "baseline_bluff"
                            and o.get("status") == "confirmed" and o.get("valid_until_game") is None), None)
    score_hypothesis(h, window, rng)
    h["status"], h["confirmed_by"] = "confirmed", "drift"
    hyps.append(h)
    events = supersede(hyps, now_game, now_round)
    if not any(ev[1]["new_id"] == h["id"] for ev in events):
        events.append(("belief_superseded", {"aspect": "baseline_bluff", "old_id": None, "new_id": h["id"],
                                             "reason": "drift", "at_game": now_game, "at_round": now_round,
                                             "about": about}))
    return h, events, p


# ---------------------------------------------------------------- decision-time lookup

def _mem_entry(h: dict, used_as: str, matched: bool) -> dict:
    e = {"hyp_id": h["id"], "aspect": h["aspect"], "text": h["text"], "status": h.get("status", "provisional"),
         "provenance": h.get("provenance", "evidence"), "n_true": int(h.get("n_true", 0)),
         "n_false": int(h.get("n_false", 0)), "q_true": list(h.get("q_true") or []),
         "q_false": list(h.get("q_false") or []), "evidence_obs": list(h.get("evidence_obs") or []),
         "used_as": used_as, "matched": matched}
    if h.get("surreal") is not None:
        e["surreal"] = h["surreal"]
    if h.get("q") is not None:
        e["q_override"] = float(h["q"])
    return e


def _independently_confirmed(h: dict) -> bool:
    return h.get("status") == "confirmed" and h.get("confirmed_by") in ("scoring", "drift")


def matching_q(hyps: list[dict], features: dict, about: str, base_post: np.ndarray | None, *,
               trust_untrusted: bool = False) -> tuple[float, str | None, list[dict]]:
    """E[q] for a pending claim by `about` with `features`. Returns (q_used, q_source, memory_used).

    Candidates = open (valid_until_game is None) lines about `about` that are usable:
    - profiler: status confirmed; an `untrusted` line only if confirmed by scoring itself
      (`confirmed_by` set by score_hypothesis/apply_drift, not by whoever injected it).
    - profiler_blind (trust_untrusted=True): every open non-refuted line counts as confirmed.
    Priority: (1) most specific matching line with specificity ≥ 1 → its q_true mean;
    (2) else the most specific non-matching line → its q_false mean; (3) else an always-true
    (baseline_bluff) line → its q_true mean; (4) else mean of `base_post` (q_source "base:<about>").
    Ties: more leaves, then newer (created_game, idx), then more evidence.
    An untrusted line carrying a `q` override (false_memory) supplies that q directly when it is
    trusted only because of trust_untrusted; once independently confirmed its scored q is used.

    memory_used lists only consulted lines, §7.1 MemoryUsed shape plus `used_as` ∈
    {"q_source","considered","ignored_untrusted"}, `matched`, and `q_override` when present.
    """
    open_ = [h for h in hyps if h.get("about") == about and h.get("valid_until_game") is None]
    usable, ignored = [], []
    for h in open_:
        indep = _independently_confirmed(h)
        if h.get("provenance") == "untrusted":
            if indep or (trust_untrusted and h.get("status") != "refuted"):
                usable.append(h)
            else:
                ignored.append(h)
        elif h.get("status") == "confirmed" or (trust_untrusted and h.get("status") != "refuted"):
            usable.append(h)

    def key(h):
        return (specificity(h["predicate"]), _order(h), h.get("n_true", 0) + h.get("n_false", 0))

    matched = {h["id"]: eval_predicate(h["predicate"], features or {}) for h in usable + ignored}
    spec = [h for h in usable if not is_always_true(h["predicate"])]
    base_lines = [h for h in usable if is_always_true(h["predicate"])]
    m = [h for h in spec if matched[h["id"]]]
    nm = [h for h in spec if not matched[h["id"]] and h.get("q_false")]

    chosen, q = None, None

    def q_of(h, field):
        if h.get("q") is not None and h.get("provenance") == "untrusted" and not _independently_confirmed(h):
            return float(h["q"])
        iv = h.get(field) or []
        return float(iv[0]) if iv else None

    for pool, field in ((m, "q_true"), (nm, "q_false"), (base_lines, "q_true")):
        for h in sorted(pool, key=key, reverse=True):
            q = q_of(h, field)
            if q is not None:
                chosen = h
                break
        if chosen:
            break
    if chosen is None:
        q, src = grid.mean(base_post if base_post is not None else grid.uniform()), f"base:{about}"
    else:
        src = chosen["id"]
    mem = [_mem_entry(h, "q_source" if h is chosen else "considered", matched[h["id"]]) for h in usable]
    mem.sort(key=lambda e: e["used_as"] != "q_source")
    mem += [_mem_entry(h, "ignored_untrusted", matched[h["id"]]) for h in ignored]
    return float(q), src, mem


def hypothesis_event(h: dict) -> dict:
    """Payload for the "hypothesis" event (blab.events.HypothesisEv + extras)."""
    return {"id": h["id"], "about": h.get("about"), "aspect": h["aspect"], "text": h["text"],
            "predicate": h["predicate"], "status": h.get("status", "provisional"),
            "n_true": int(h.get("n_true", 0)), "n_false": int(h.get("n_false", 0)),
            "q_true": list(h.get("q_true") or []), "q_false": list(h.get("q_false") or []),
            "provenance": h.get("provenance", "evidence"), "valid_from_game": int(h.get("valid_from_game", 0)),
            "valid_until_game": h.get("valid_until_game"), "superseded_by": h.get("superseded_by"),
            "created_game": h.get("created_game"), "supersedes": h.get("supersedes"),
            "p_gap": h.get("p_gap"), "evidence_obs": list(h.get("evidence_obs") or []),
            "origin": h.get("origin"), "confirmed_by": h.get("confirmed_by")}
