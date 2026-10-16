"""c01 agents (PREREG_c01.md): per-claim record check, population prior, profiler_hybrid.

- `llm_record_check` (raw feature, "true" | "false" | "none"): before each respond decision Claude Sonnet is asked
  a GENERIC question, "Does any factual statement in this utterance contradict this public record?", given the
  pending utterance and the PUBLIC record only (core's `public_obs_text` with the private lines removed: our hand,
  legal actions and the hand-derived p_cards). It is never told about any tell. Cached by sha256(utterance,
  public text) in a run-wide jsonl cache shared by all arms, so identical states get identical answers and
  replay is exact. Pre-registered skip rule: an utterance with a single sentence is the claim itself (the
  cards are hidden, so the public record cannot contradict it beyond card math, which p_cards already
  carries) -> "none" without a call. A failed call -> "none" (counted).
- Population prior: the pooled grid posterior over q from CONTROL opponents' revealed claims on dev seeds
  7000-7099, tempered to N0 = 10 effective claims (blab/data/c01_pop_prior.json). Grid agents start from it
  instead of uniform; logistic calibrators use it as their offset logit(p_cards + (1 - p_cards) * E[q]).
- Arms: bayes_discount_pp, profiler_plus_c01, profiler_hybrid, profiler_plus_noreflect (see PREREG_c01.md).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from pathlib import Path

import numpy as np

from blab import grid
from blab import hypotheses as H
from blab.agents import ours
from blab.agents.common import public_obs_text
from blab.agents.embed import sentences

POP_PRIOR_PATH = Path(__file__).resolve().parent.parent / "data" / "c01_pop_prior.json"
RC_VALUES = ("true", "false", "none")
RC_MODEL = os.environ.get("BLAB_RC_BEDROCK_MODEL", "us.anthropic.claude-sonnet-5-5")
RC_TIMEOUT_S = 30.0
C01_REFLECT_GAMES = (2, 4, 8, 12)          # compute-driven, a priori (PREREG_c01.md)

RC_SYSTEM = """You check spoken statements against a public game record. The game is a 3-player bluffing card
game (a research simulation). Seats p1, p2, p3. Each round a table rank (K = Kings, Q = Queens, A = Aces) is drawn;
players take turns placing 1-3 cards face down while claiming they are all the table rank, and say something out
loud. Every round ends with a challenge that reveals one claim. The public record lists, for the current game: the
current game and round number, the current table rank, lives, hand sizes, earlier claims this round, the pending
claim with its exact utterance, and the claims revealed by challenges so far (format: g<game>r<round> <seat>
claimed <count>x<table rank of that round> ...).

Question: does any factual statement in the pending utterance contradict the public record?
- "true": at least one statement in the utterance is contradicted by the record.
- "false": the utterance contains at least one statement the record can check, and none is contradicted.
- "none": the utterance contains no statement the record can confirm or contradict (the claimed cards
  themselves are hidden and cannot be checked).
Return JSON {"answer": "true" | "false" | "none"}."""

_P_LINE = re.compile(r" p_cards=[^ ]+ \(prob\. the claim is impossible given your hand\)")


def public_record_text(obs: dict) -> str:
    """public_obs_text over the CURRENT game's revealed claims, with private lines removed (our hand, legal
    actions, the hand-derived p_cards). Deterministic; identical from a live obs and its trimmed copy."""
    o = dict(obs)
    o["revealed_history"] = [e for e in obs.get("revealed_history", []) if e.get("game") == obs.get("game")]
    o.pop("ask", None)
    txt = public_obs_text(o)
    out = []
    for ln in txt.split("\n"):
        if ln.startswith("Your hand:") or ln.startswith("Legal actions:"):
            continue
        out.append(_P_LINE.sub("", ln))
    return "\n".join(out)


def _rc_key(utt: str, text: str) -> str:
    return hashlib.sha256((utt + "\x00" + text).encode()).hexdigest()


class RecordChecker:
    """Process-wide cache + Sonnet calls. Cache/log files are jsonl, appended by every worker process."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cache: dict[str, str] = {}
        self._pos = 0
        self._client = None
        self.stats = {"calls": 0, "errors": 0, "cache_hits": 0, "skipped_single_sentence": 0}

    @property
    def cache_path(self) -> Path | None:
        p = os.environ.get("BLAB_RC_CACHE")
        return Path(p) if p else None

    def _refresh(self):
        p = self.cache_path
        if not p or not p.exists():
            return
        with open(p) as f:
            f.seek(self._pos)
            for ln in f:
                if not ln.endswith("\n"):
                    break
                self._pos += len(ln.encode())
                try:
                    r = json.loads(ln)
                    self.cache.setdefault(r["key"], r["value"])
                except Exception:  # noqa: BLE001
                    pass

    def _append(self, path: Path | None, rec: dict):
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _call(self, utt: str, text: str) -> dict:
        import anthropic
        from pydantic import BaseModel
        from typing import Literal

        class RC(BaseModel):
            answer: Literal["true", "false", "none"]

        if self._client is None:
            region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-west-2"
            self._client = anthropic.AnthropicBedrock(api_key=os.environ["AWS_SONNET_API_KEY"], aws_region=region,
                                                      timeout=RC_TIMEOUT_S, max_retries=0)
        user = f"## Public record\n{text}\n\n## Pending utterance\n{json.dumps(utt)}"
        t0 = time.monotonic()
        err = None
        for attempt in range(3):
            try:
                resp = self._client.messages.parse(
                    model=RC_MODEL, max_tokens=2000,
                    system=[{"type": "text", "text": RC_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": user}], output_format=RC,
                    output_config={"effort": "low"})
                parsed = getattr(resp, "parsed_output", None)
                if parsed is None:
                    err = f"no parsed_output (stop_reason={resp.stop_reason})"
                    break
                u = resp.usage
                return {"value": parsed.answer, "latency_ms": int((time.monotonic() - t0) * 1000),
                        "usage": {"in": getattr(u, "input_tokens", None), "out": getattr(u, "output_tokens", None),
                                  "cache_read": getattr(u, "cache_read_input_tokens", None),
                                  "cache_write": getattr(u, "cache_creation_input_tokens", None)},
                        "model": RC_MODEL, "request_id": getattr(resp, "_request_id", None)}
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:160]}"
                if time.monotonic() - t0 > RC_TIMEOUT_S:
                    break
                time.sleep(1.0 * (attempt + 1))
        return {"value": "none", "error": err, "latency_ms": int((time.monotonic() - t0) * 1000), "model": RC_MODEL}

    def check(self, obs: dict) -> dict:
        utt = str((obs.get("pending_claim") or {}).get("utterance") or "")
        if len(sentences(utt)) <= 1:
            with self.lock:
                self.stats["skipped_single_sentence"] += 1
            return {"value": "none", "source": "skip_single_sentence"}
        text = public_record_text(obs)
        key = _rc_key(utt, text)
        with self.lock:
            if key not in self.cache:
                self._refresh()
            if key in self.cache:
                self.stats["cache_hits"] += 1
                self._append(_log_path(), {"key": key, "hit": True, "t": time.time()})
                return {"value": self.cache[key], "source": "cache", "key": key}
            r = self._call(utt, text)
            self.stats["calls"] += 1
            if r.get("error"):
                self.stats["errors"] += 1
            else:                                 # failures are not cached (a replay may retry)
                self.cache[key] = r["value"]
                self._append(self.cache_path, {"key": key, "value": r["value"]})
            self._append(_log_path(), {"key": key, "hit": False, "t": time.time(), **r})
            return {"value": r["value"], "source": "error" if r.get("error") else "call", "key": key,
                    "error": r.get("error")}


def _log_path() -> Path | None:
    p = os.environ.get("BLAB_RC_LOG")
    return Path(p) if p else None


CHECKER = RecordChecker()


# ------------------------------------------------------------------ population prior

_POP = None


def pop_prior() -> np.ndarray:
    global _POP
    if _POP is None:
        _POP = grid.from_list(json.loads(POP_PRIOR_PATH.read_text())["grid"])
    return _POP


def pop_q0() -> float:
    return grid.mean(pop_prior())


class BayesDiscountPP(ours.BayesDiscount):
    """bayes_discount with the population prior in place of the uniform grid prior."""
    id = "bayes_discount_pp"

    def base_post(self, obs, about):
        g = obs.get("game", 0)
        es = [e for e in self.memory.get("revealed", []) if e.get("by") == about] + self.in_game(obs, about)
        post = pop_prior().copy()
        for e in es:
            post = grid.update(post, e["p_cards"], e["was_bluff"], ours.GAMMA ** max(0, g - e["game"]))
        return post


# ------------------------------------------------------------------ calibrator family

def offset_logit(p_c: float) -> float:
    p = float(p_c) + (1.0 - float(p_c)) * pop_q0()
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


class _C01Plus(ours.ProfilerPlus):
    """profiler_plus with (a) the population-prior offset in place of logit(p_cards) (prior coef 1, as before),
    (b) llm_record_check attached to every judged claim (pending features and, at game end, revealed features),
    so DSL predicates may use it, (c) reflection after games C01_REFLECT_GAMES. Built-in record-check features
    in the calibrator only when RC_BUILTIN."""
    RC_BUILTIN = False

    def __init__(self, seat, rng=None, params=None):
        super().__init__(seat, rng, params)
        self._rc_game: dict = {}            # in-game working memory: (game, round, by, utterance) -> value

    # names / design ------------------------------------------------------------
    def base_names(self):
        n = ["bias", "offset_logit"] + ours.BASE_NAMES[2:]
        return n + (["rc_contradiction", "rc_consistent"] if self.RC_BUILTIN else [])

    def bx(self, features: dict, p_c: float) -> list[float]:
        x = ours.base_x(features, p_c)
        x[1] = offset_logit(p_c)
        if self.RC_BUILTIN:
            v = features.get("llm_record_check")
            x += [float(v == "true"), float(v == "false")]
        return x

    def wants_reflection(self, game: int) -> bool:
        return self.reflects and self.learning and game in C01_REFLECT_GAMES

    def reflect_payload(self) -> dict:
        p = super().reflect_payload()
        p["record_check"] = True
        return p

    # record check ----------------------------------------------------------------
    def _with_rc(self, obs: dict, features: dict) -> tuple[dict, dict]:
        rc = CHECKER.check(obs)
        f = dict(features)
        f["llm_record_check"] = rc["value"]
        pc = obs.get("pending_claim") or {}
        self._rc_game[(obs.get("game"), obs.get("round"), pc.get("by"), pc.get("utterance"))] = rc["value"]
        return f, rc

    def decide_respond(self, obs: dict, rng_u: float) -> dict:
        pc = obs["pending_claim"]
        f, rc = self._with_rc(obs, pc["features"])
        o2 = dict(obs)
        o2["pending_claim"] = dict(pc, features=f)
        d = super().decide_respond(o2, rng_u)
        d["obs"] = ours.trim_obs(obs)                 # the recorded obs is the agent's input (rc is recomputed)
        d["record_check"] = rc
        return d

    def _absorb(self, obs, game):
        new = super()._absorb(obs, game)
        for e in new:
            v = self._rc_game.get((e.get("game"), e.get("round"), e.get("by"), e.get("utterance")))
            if v is not None:
                e.setdefault("features", {})["llm_record_check"] = v
        self._rc_game = {k: v for k, v in self._rc_game.items() if k[0] != game}
        return new

    # calibrator (copy of ProfilerPlus.predict/_fit with the c01 design) --------------
    def predict(self, obs, about, features, p_c):
        cal = (self.memory.get("plus") or {}).get(about) or {}
        coefs = dict(zip(cal.get("names", []), cal.get("w", [])))
        ivs = dict(zip(cal.get("names", []), cal.get("iv", [])))
        names = self.base_names()
        xb = self.bx(features, p_c)
        w0 = [0.0, 1.0] + [0.0] * (len(names) - 2)
        z, terms = 0.0, {}
        for i, n in enumerate(names):
            c = coefs.get(n, w0[i])
            z += c * xb[i]
            terms[n] = c * xb[i]
        mem = []
        for h in self.usable(about):
            on = bool(H.eval_predicate(h["predicate"], features))
            c = coefs.get(h["id"])
            contrib = (c if c is not None else 0.0) * float(on)
            z += contrib
            e = H._mem_entry(h, "feature", on)
            e.update(coef=c, coef_iv=ivs.get(h["id"]), contribution=contrib)
            mem.append(e)
        p = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))
        return None, f"plus:{about}", mem, p, {"plus": {"logit": z, "base_terms": terms, "n_fit": int(cal.get("n", 0)),
                                                       "fit_game": cal.get("game"), "q0_pop": pop_q0()}}

    def _fit(self, about: str, game: int) -> None:
        es = [e for e in self.memory["revealed"] if e["by"] == about]
        hs = self.usable(about)
        bn = self.base_names()
        names = bn + [h["id"] for h in hs]
        d = len(names)
        w0 = np.zeros(d)
        w0[1] = 1.0
        if es:
            X = np.array([self.bx(e["features"], e["p_cards"]) +
                          [float(bool(H.eval_predicate(h["predicate"], e["features"]))) for h in hs] for e in es])
            y = np.array([float(e["was_bluff"]) for e in es])
            sw = np.array([ours.PLUS_GAMMA ** max(0, game - e["game"]) for e in es])
        else:
            X, y, sw = np.zeros((0, d)), np.zeros(0), np.zeros(0)
        w = w0.copy()
        P = np.eye(d) / ours.PLUS_PRIOR_VAR

        def obj(v):
            z = X @ v
            return float(np.sum(sw * (np.logaddexp(0.0, z) - y * z)) + 0.5 * (v - w0) @ P @ (v - w0))

        f = obj(w)
        for _ in range(50):
            pr = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -35, 35)))
            Hm = (X * (sw * pr * (1 - pr))[:, None]).T @ X + P
            g = X.T @ (sw * (pr - y)) + P @ (w - w0)
            step = np.linalg.solve(Hm, g)
            t = 1.0
            while t > 1e-6:
                wn = w - t * step
                fn = obj(wn)
                if fn <= f - 1e-4 * t * float(g @ step):
                    break
                t *= 0.5
            if t <= 1e-6:
                break
            w, f = wn, fn
            if float(np.abs(t * step).max()) < 1e-9:
                break
        pr = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -35, 35)))
        Hm = (X * (sw * pr * (1 - pr))[:, None]).T @ X + P
        se = np.sqrt(np.clip(np.diag(np.linalg.inv(Hm)), 0, None))
        self.memory.setdefault("plus", {})[about] = {
            "names": names, "w": [float(v) for v in w],
            "iv": [[float(w[i] - ours.Z95 * se[i]), float(w[i] + ours.Z95 * se[i])] for i in range(d)],
            "n": len(es), "game": game}


class ProfilerPlusC01(_C01Plus):
    """c01 profiler_plus: mem01's profiler_plus + population-prior offset + llm_record_check as a DSL feature
    (reflection may propose predicates on it; not a built-in calibrator feature) + the c01 reflection schedule."""
    id = "profiler_plus_c01"


class ProfilerHybrid(_C01Plus):
    """profiler_hybrid = profiler_plus_c01 + llm_record_check as built-in calibrator features (rc_contradiction,
    rc_consistent; prior N(0,1) like every non-offset coefficient) + full memory semantics: hypotheses with
    provenance, untrusted lines weight 0 unless independently confirmed (inherited), SUPERSESSION (newest
    confirmed line per (about, aspect) closes older ones: H.supersede, as in the Profiler), replay-exact."""
    id = "profiler_hybrid"
    RC_BUILTIN = True

    def on_game_end(self, obs, game):
        events = super().on_game_end(obs, game)
        if not self.learning:
            return events
        sup = H.supersede(self.memory["hypotheses"], game, int(obs.get("round", 0)))
        if sup:
            self._refit_all(game)
            self.memory = ours._jsonable(self.memory)
        return events + sup

    def apply_reflection(self, proposed, game):
        events = super().apply_reflection(proposed, game)
        sup = H.supersede(self.memory["hypotheses"], game, 0)
        if sup:
            self._refit_all(game)
            self.memory = ours._jsonable(self.memory)
        return events + sup


class ProfilerPlusNoReflect(ProfilerHybrid):
    """profiler_hybrid without reflection: no hypotheses, so the calibrator has only the base features, the
    population-prior offset and the built-in record-check features. Isolates what Claude reflection adds."""
    id = "profiler_plus_noreflect"
    reflects = False


for _c in (BayesDiscountPP, ProfilerPlusC01, ProfilerHybrid, ProfilerPlusNoReflect):
    ours.AGENTS[_c.id] = _c
