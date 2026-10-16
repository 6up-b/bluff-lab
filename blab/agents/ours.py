"""Our agents for seat p1 (SPEC §5.1): counter, bayes, bayes_discount, profiler, profiler_amnesic,
profiler_blind.

Respond rule: challenge iff p_final > 0.5, else explore-challenge w.p. ε = 0.1 (rng_u < ε).
Play rule: holding matching cards (incl. J) → play up to 2 of them honestly; else forced 1-card bluff.

EV convention (recorded in every respond decision): a challenge swings one life — the claimant loses
one if it was a bluff (+1 for us), we lose one otherwise (−1). So ev.challenge = 2p − 1 and
ev.play_on = 1 − 2p (the mirror: accepting forgoes that swing). Only the sign matters for the rule.

Memory timing rule (INTERFACES.md): checkpoint memory (`self.memory`) changes only in
`on_game_end`/`apply_reflection`. At decision time the agent uses `self.memory` plus the CURRENT
game's entries of `obs.revealed_history` (in-game working memory). A decision is therefore a pure
function of (trimmed obs, memory, rng_u) — `decide_respond` — which is what /replay re-runs.
"""
from __future__ import annotations

import copy
import json
import math
import re
import zlib

import numpy as np

from blab import grid
from blab import hypotheses as H
from blab.agents import banks
from blab.engine import RANK_NAME, matches

EPS = 0.1
GAMMA = 0.85
SEAT_IDX = {"p1": 1, "p2": 2, "p3": 3}


def _jsonable(x):
    return json.loads(json.dumps(x))


def empty_memory(agent: str, seat: str) -> dict:
    return {"memory_version": 0, "agent": agent, "seat": seat, "base": {}, "hypotheses": [],
            "observations": [], "revealed": [], "next_idx": 1, "reflected_upto": 0, "drift_notes": []}


def trim_obs(obs: dict) -> dict:
    """The obs recorded in a decision: everything, but revealed_history limited to the current game
    (earlier games are in the memory checkpoint; per the timing rule that is all a decision depends on)."""
    o = dict(obs)
    o["revealed_history"] = [e for e in obs.get("revealed_history", []) if e.get("game") == obs.get("game")]
    o["revealed_history_scope"] = "current_game"
    return o


def _keep(e: dict) -> dict:
    return {k: e[k] for k in ("obs_id", "game", "round", "by", "challenger", "count", "was_bluff", "cards",
                              "utterance", "features", "p_cards") if k in e}


class Counter:
    """No memory: p_final = p_cards."""
    id = "counter"
    has_memory = False

    def __init__(self, seat: str, rng: np.random.Generator | None = None, params: dict | None = None):
        self.seat = seat
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.params = dict(params or {})
        self.learning = bool(self.params.get("learning", True))
        self.memory = empty_memory(self.id, seat)
        self._playon_log: list[tuple[int, int]] = []     # in-game only; folded into memory at game end

    # ------------------------------------------------------------ estimator (override)
    def estimate(self, obs: dict, about: str, features: dict) -> tuple[float, str | None, list[dict]]:
        return 0.0, None, []

    # ------------------------------------------------------------ decisions
    def decide_respond(self, obs: dict, rng_u: float) -> dict:
        pc = obs["pending_claim"]
        about = pc["by"]
        f = dict(pc["features"])
        cm = dict(f.get("card_math") or {})
        p_c = float(cm["p_cards"])
        q, src, mem, p_final, extra = self.predict(obs, about, f, p_c)
        forced = "play" not in obs["legal_actions"]
        rule = self.params.get("respond_rule", "threshold")
        if rule == "ev":
            # P(I lose a life this round): challenge → 1 − p_final; play_on → L (own empirical rate).
            # Exactly one life is lost per round, so an opponent's loss is the complement and the value
            # of an opponent life (β) cancels: challenge iff p_final > 1 − L.
            L, L_n, L_k = self.playon_loss_rate()
            thr = 1.0 - L
            ev = {"challenge": -(1.0 - p_final), "play_on": -L}
        else:
            thr = float(self.params.get("threshold", 0.5))       # §5.1: 0.5
            ev = {"challenge": 2 * p_final - 1, "play_on": 1 - 2 * p_final}
        if forced or p_final > thr:
            action, explore, prop = "challenge", False, 1.0
        elif rng_u < EPS:
            action, explore, prop = "challenge", True, EPS
        else:
            action, explore, prop = "play_on", False, 1.0 - EPS
        claim_f = {k: v for k, v in f.items() if k != "card_math"}
        return {"kind": "respond", "target": about,
                "claim": {"count": pc["count"], "rank": obs["table_rank"], "utterance": pc["utterance"],
                          "features": claim_f},
                "card_math": cm, "memory_used": mem, "q_used": (float(q) if q is not None else None), "q_source": src,
                "p_final": float(p_final),
                "ev": ev,
                "action": action, "explore": explore, "propensity": prop, "rng_u": float(rng_u),
                "forced": forced, "memory_version": int(self.memory.get("memory_version", 0)),
                "obs": trim_obs(obs), "agent": self.id,
                "ask": copy.deepcopy(obs.get("ask")) if obs.get("ask") else None,
                **({"respond_rule": "ev", "ev_detail": {"L": L, "L_n": L_n, "L_k": L_k, "threshold": thr,
                                                         "formula": "challenge iff p_final > 1 - L; L=(k+1)/(n+3)"}}
                   if rule == "ev" else {}),
                **extra}

    def predict(self, obs: dict, about: str, features: dict, p_c: float):
        """(q_used, q_source, memory_used, p_final, extra payload fields). §5.1: p_final = p_c + (1−p_c)·E[q]."""
        q, src, mem = self.estimate(obs, about, features)
        return q, src, mem, p_c + (1.0 - p_c) * q, {}

    # P(lose a life later in the round | I played on): prior Beta(1,2) (mean 1/3: one of three
    # players loses each round). Memoryless agents use the prior; memory agents use their checkpoint.
    def playon_loss_rate(self) -> tuple[float, int, int]:
        st = (self.memory or {}).get("self") or {}
        n, k = int(st.get("playon_n", 0)), int(st.get("playon_lost", 0))
        return (k + 1) / (n + 3), n, k

    def decide_play(self, obs: dict) -> tuple[dict, dict]:
        hand, rank = obs["my_hand"], obs["table_rank"]
        mi = [i for i, c in enumerate(hand) if matches(c, rank)]
        if mi:
            # prefer real rank cards over jokers; play up to 2
            mi.sort(key=lambda i: (hand[i] == "J", i))
            idx, honest = sorted(mi[:2]), True
        else:
            idx, honest = [0], False
        n = len(idx)
        utt = f"{n} {RANK_NAME[rank]}{'s' if n > 1 else ''}."
        act = {"type": "play", "cards": idx, "utterance": utt,
               "meta": {"bluff_kind": None if honest else "forced"}}
        dec = {"kind": "play", "action": "play", "claim": {"count": n, "rank": rank, "utterance": utt},
               "cards": idx, "honest": honest, "memory_version": int(self.memory.get("memory_version", 0)),
               "obs": trim_obs(obs), "agent": self.id}
        return act, dec

    def act(self, obs: dict) -> dict:
        decisions = []
        if obs.get("pending_claim") is not None:
            u = float(self.rng.random())
            d = self.decide_respond(obs, u)
            decisions.append(d)
            if d["action"] == "challenge":
                return {"type": "challenge", "decisions": decisions}
            if not d.get("forced"):
                self._playon_log.append((obs["game"], obs["round"]))
        act, dec = self.decide_play(obs)
        decisions.append(dec)
        act["decisions"] = decisions
        return act

    def answer(self, question: str, obs: dict | None = None) -> str:
        return "Just playing my cards."


class Bayes(Counter):
    """One grid posterior per opponent over all revealed claims, forever."""
    id = "bayes"
    has_memory = True

    def __init__(self, seat, rng=None, params=None):
        super().__init__(seat, rng, params)
        self.memory = empty_memory(self.id, seat)

    # memory API ---------------------------------------------------------
    def get_memory(self) -> dict:
        return copy.deepcopy(self.memory)

    def set_memory(self, mem: dict | None) -> None:
        m = empty_memory(self.id, self.seat)
        if mem:
            m.update(copy.deepcopy(mem))
        m["agent"], m["seat"] = self.id, self.seat
        self.memory = _jsonable(m)

    def wants_reflection(self, game: int) -> bool:
        return False

    # estimator ------------------------------------------------------------
    def in_game(self, obs: dict, about: str) -> list[dict]:
        return [e for e in obs.get("revealed_history", []) if e.get("game") == obs.get("game") and e.get("by") == about]

    def base_post(self, obs: dict, about: str) -> np.ndarray:
        post = grid.from_list((self.memory.get("base", {}).get(about) or {}).get("grid"))
        for e in self.in_game(obs, about):
            post = grid.update(post, e["p_cards"], e["was_bluff"])
        return post

    def estimate(self, obs, about, features):
        return grid.mean(self.base_post(obs, about)), f"base:{about}", []

    # learning ---------------------------------------------------------------
    def _absorb(self, obs: dict, game: int) -> list[dict]:
        new = [_keep(e) for e in obs.get("revealed_history", []) if e.get("game") == game and e.get("by") != self.seat]
        self.memory["revealed"].extend(new)
        self.memory["observations"].extend(e["obs_id"] for e in new)
        for e in new:
            b = self.memory["base"].setdefault(e["by"], {"grid": None, "n": 0})
            b["grid"] = grid.to_list(grid.update(grid.from_list(b["grid"]), e["p_cards"], e["was_bluff"]))
            b["n"] += 1
        # self-stats for respond_rule "ev": did I lose the round after playing on?
        loser = {}
        for e in obs.get("revealed_history", []):
            if e.get("game") == game:
                loser[e["round"]] = e["by"] if e["was_bluff"] else e["challenger"]
        st = self.memory.setdefault("self", {"playon_n": 0, "playon_lost": 0})
        for (g, r) in self._playon_log:
            if g == game:
                st["playon_n"] += 1
                st["playon_lost"] += int(loser.get(r) == self.seat)
        self._playon_log = []
        return new

    def on_game_end(self, obs: dict, game: int) -> list[tuple[str, dict]]:
        if not self.learning:
            return []
        self._absorb(obs, game)
        self.memory["memory_version"] += 1
        self.memory = _jsonable(self.memory)
        return []


class BayesDiscount(Bayes):
    """Same, with likelihood weights γ^(games ago), γ = 0.85."""
    id = "bayes_discount"

    def base_post(self, obs, about):
        g = obs.get("game", 0)
        es = [e for e in self.memory.get("revealed", []) if e.get("by") == about] + self.in_game(obs, about)
        return grid.posterior((e["p_cards"], e["was_bluff"], GAMMA ** max(0, g - e["game"])) for e in es)


class Profiler(Bayes):
    """bayes base rate + hypothesis graph (§5.2) + drift (§5.3) + provenance."""
    id = "profiler"
    trust_untrusted = False
    reflects = True

    def estimate(self, obs, about, features):
        base = self.base_post(obs, about)
        return H.matching_q(self.memory.get("hypotheses", []), features, about, base,
                            trust_untrusted=self.trust_untrusted)

    # interrogation (§7.5) -------------------------------------------------------
    MAX_ASKS_PER_GAME = 2

    def want_ask(self, obs: dict) -> dict | None:
        """Pure function of (obs, memory) + a per-game counter. Probe when p_final ∈ [0.35, 0.65],
        call-out when p_final > 0.65; ≤ 2 per game. Template filled from recorded decision variables only."""
        pc = obs.get("pending_claim")
        if not pc or "play" not in obs.get("legal_actions", []):
            return None
        g = obs.get("game")
        if getattr(self, "_ask_game", None) != g:
            self._ask_game, self._asks = g, 0
        if self._asks >= self.MAX_ASKS_PER_GAME:
            return None
        f = pc["features"]
        p_c = float(f["card_math"]["p_cards"])
        q, src, mem = self.estimate(obs, pc["by"], f)
        p = p_c + (1 - p_c) * q
        if p < 0.35:
            return None
        kind = "probe" if p <= 0.65 else "callout"
        top = next((m for m in mem if m.get("used_as") == "q_source"), None)
        rank = obs["table_rank"]
        k = pc["count"]
        if top and top["aspect"] == "verbal_tell":
            tid, text, aspect = "formal_callout", "You've gone very formal all of a sudden.", "verbal_tell"
        elif top and top["aspect"] == "exploits_trust":
            tid, text, aspect = "trust_probe", "You've been very honest so far… still?", "exploits_trust"
        elif f.get("hand_size", 9) <= 2:
            tid, aspect = "pressure_hand", "pressure"
            text = f"Two cards left and you're claiming {k} {RANK_NAME[rank]}{'s' if k > 1 else ''}?"
        else:
            tid, text, aspect = "generic_probe", "Are you sure about that?", None
        self._asks += 1
        return {"target": pc["by"], "template_id": tid, "text": text, "p_final": float(p),
                "reason_aspect": aspect, "kind": kind}

    def _score_rng(self, game: int) -> np.random.Generator:
        # independent of the decision stream, so learning on/off never shifts decision RNG
        return np.random.default_rng([int(self.params.get("seed", 0)), int(game), SEAT_IDX.get(self.seat, 0), 17])

    @staticmethod
    def _sig(h: dict) -> tuple:
        return (h.get("status"), h.get("n_true"), h.get("n_false"), h.get("valid_until_game"), h.get("superseded_by"))

    def on_game_end(self, obs: dict, game: int) -> list[tuple[str, dict]]:
        if not self.learning:
            return []
        self._absorb(obs, game)
        mem = self.memory
        hyps = mem["hypotheses"]
        rng = self._score_rng(game)
        rnd = int(obs.get("round", 0))
        before = {h["id"]: self._sig(h) for h in hyps}
        for h in hyps:
            if h.get("valid_until_game") is None and h.get("origin") != "drift":
                H.score_hypothesis(h, mem["revealed"], rng, now_game=game)
            elif h.get("valid_until_game") is None and h.get("origin") == "drift":
                # drift baselines: rescore on their own regime (always-true ⇒ stays confirmed)
                H.score_hypothesis(h, mem["revealed"], rng)
                h["status"], h["confirmed_by"] = "confirmed", "drift"
        sup = H.supersede(hyps, game, rnd)
        opps = sorted({e["by"] for e in mem["revealed"]})
        new_ids = []
        for opp in (opps if self.params.get("drift", True) else []):     # drift=False: detector off (lc02 fork)
            nh, evs, p = H.apply_drift(hyps, mem["revealed"], wid=self.params.get("wid", "w000"), about=opp,
                                      idx=mem["next_idx"], now_game=game, now_round=rnd, rng=rng)
            if nh is not None:
                mem["next_idx"] += 1
                new_ids.append(nh["id"])
                mem["drift_notes"].append({"about": opp, "game": game, "p": round(p, 4), "hyp_id": nh["id"],
                                           "q_recent": nh.get("q_true")})
                sup += evs
        events = [("hypothesis", H.hypothesis_event(h)) for h in hyps
                  if h["id"] in new_ids or before.get(h["id"]) != self._sig(h)]
        events += sup
        mem["memory_version"] += 1
        self.memory = _jsonable(mem)
        return events

    # reflection --------------------------------------------------------------
    def wants_reflection(self, game: int) -> bool:
        k = int(self.params.get("reflect_every", 2) or 0)
        return self.reflects and self.learning and k > 0 and game % k == 0

    def reflect_payload(self) -> dict:
        mem = self.memory
        new = mem["revealed"][mem.get("reflected_upto", 0):][-40:]
        return {"seat": self.seat, "agent": self.id, "wid": self.params.get("wid"),
                "info": self.params.get("info", "engineered"),
                "memory_version": mem["memory_version"], "new_obs": copy.deepcopy(new),
                "hypotheses": [H.hypothesis_event(h) for h in mem["hypotheses"]],
                "drift_notes": copy.deepcopy(mem.get("drift_notes", [])[-4:]),
                "opponents": sorted({e["by"] for e in mem["revealed"]} | {s for s in ("p2", "p3") if s != self.seat})}

    def allowed_about(self) -> tuple[str, ...]:
        """Keys a reflection proposal may be about (seats; reg01 identity-keyed agents add names)."""
        return ("p1", "p2", "p3")

    def apply_reflection(self, proposed: list[dict] | None, game: int) -> list[tuple[str, dict]]:
        mem = self.memory
        mem["reflected_upto"] = len(mem["revealed"])
        self.last_dropped = []
        events = []
        # duplicates = same about+predicate as an OPEN, non-refuted line (refuted is absorbing, so a
        # re-proposal of a refuted predicate becomes a new line that is scored afresh)
        existing = {(h.get("about"), json.dumps(h.get("predicate"), sort_keys=True)) for h in mem["hypotheses"]
                    if h.get("valid_until_game") is None and h.get("status") != "refuted"}
        for p in (proposed or [])[:3]:
            try:
                key = (p.get("about"), json.dumps(p.get("predicate"), sort_keys=True))
                if key in existing:
                    self.last_dropped.append({"proposal": p, "reason": "duplicate"})
                    continue
                if p.get("about") not in self.allowed_about() or p.get("about") == self.seat:
                    raise ValueError("bad 'about'")
                h = H.new_hypothesis(wid=self.params.get("wid", "w000"), about=p["about"], aspect=p["aspect"],
                                     text=str(p.get("text", ""))[:300], predicate=p["predicate"],
                                     created_game=game, idx=mem["next_idx"],
                                     supersedes=p.get("supersedes_id"), origin=p.get("origin", "llm"),
                                     rationale=str(p.get("rationale", ""))[:500],
                                     mode=self.params.get("info", "engineered"))
            except (ValueError, KeyError, TypeError) as e:
                self.last_dropped.append({"proposal": p, "reason": str(e)})
                continue
            mem["next_idx"] += 1
            mem["hypotheses"].append(h)
            existing.add(key)
            events.append(("hypothesis", H.hypothesis_event(h)))
        mem["memory_version"] += 1
        self.memory = _jsonable(mem)
        return events


class ProfilerBlind(Profiler):
    """Treats every memory line as confirmed regardless of provenance or evidence."""
    id = "profiler_blind"
    trust_untrusted = True


class ProfilerAmnesic(Profiler):
    """Profiler with memory reset at each game start: only in-game working memory, never reflects."""
    id = "profiler_amnesic"
    reflects = False

    def on_game_end(self, obs, game):
        return []

    def set_memory(self, mem):
        super().set_memory(None)


# ------------------------------------------------------------------ logreg_bow (§5.1, P1)

LR, L2 = 0.1, 1e-3
N_HASH = 256
NUMERIC = (("hand_size", 5.0), ("claim_count", 3.0), ("trusted_streak", 10.0), ("lives_self", 3.0),
           ("lives_target", 3.0), ("game_index", 20.0), ("round_index", 30.0))
FEAT_NAMES = (["bias", "logit_p_cards"] + [n for n, _ in NUMERIC]
              + ["was_challenged_last_round", "formal_register"] + [f"h{i}" for i in range(N_HASH)]
              + ["word_count"])        # raw mode only (0 in engineered mode ⇒ engineered fits unchanged)
DIM = len(FEAT_NAMES)
I_FORMAL = FEAT_NAMES.index("formal_register")
Z80 = 1.2815515655446004
_TOK = re.compile(r"[a-z0-9']+")


def stable_bucket(tok: str) -> int:
    """Stable across processes (crc32), unlike Python's salted hash()."""
    return zlib.crc32(tok.encode("utf-8")) % N_HASH


def featurize(features: dict, p_cards: float) -> np.ndarray:
    """[1, logit(p_cards), numeric features scaled to ~[0,1], was_challenged_last_round, formal_register,
    256-dim hashed unigram+bigram presence (lowercased, apostrophes kept)]."""
    x = np.zeros(DIM)
    pc = min(max(float(p_cards), 1e-3), 1 - 1e-3)
    x[0], x[1] = 1.0, math.log(pc / (1 - pc))
    for i, (n, scale) in enumerate(NUMERIC):
        x[2 + i] = float(features.get(n, 0) or 0) / scale
    x[2 + len(NUMERIC)] = float(bool(features.get("was_challenged_last_round")))
    x[I_FORMAL] = float(bool(features.get("formal_register")))      # absent in raw mode ⇒ 0
    if "word_count" in features:                                      # present only in raw mode
        x[DIM - 1] = float(features["word_count"]) / 10.0
    toks = _TOK.findall(str(features.get("utterance", "")).lower())
    grams = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    for g in grams:
        x[I_FORMAL + 1 + stable_bucket(g)] = 1.0
    return x


def _padw(w: list[float]) -> np.ndarray:
    """Weights saved before the word_count column existed are zero-padded (engineered fits unchanged)."""
    a = np.asarray(w, dtype=float)
    return np.concatenate([a, np.zeros(DIM - len(a))]) if len(a) < DIM else a


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))


class LogregBow(Bayes):
    """Online L2 logistic regression per opponent (lr 0.1, λ 1e-3) on revealed claims; fit only at game end
    (memory timing rule). Predicts P(bluff) directly: p_final = σ(w·x); q_used = None, q_source "logreg".
    Tell coefficient (discovery statistic, memory.logreg[opp].tell_coef = [coef, lo80, hi80]): formal_register
    coefficient of a reduced L2-MAP logistic model with a Laplace interval — see on_game_end for why."""
    id = "logreg_bow"

    def predict(self, obs, about, features, p_c):
        m = (self.memory.get("logreg") or {}).get(about)
        w = _padw(m["w"]) if m else np.zeros(DIM)
        p = _sig(float(w @ featurize(features, p_c)))
        tell = (list(m["tell_coef"]) if m.get("tell_coef") is not None else None) if m else None
        return None, "logreg", [], p, {"logreg": {"p_bluff": p, "n_fit": int(m["n"]) if m else 0,
                                                 "tell_coef": tell, "tell_feature": "formal_register",
                                                 "discovery_game": (m or {}).get("discovery_game")}}

    def estimate(self, obs, about, features):     # used only by generic helpers
        p_c = float(features["card_math"]["p_cards"])
        p = self.predict(obs, about, features, p_c)[3]
        return (max(0.0, (p - p_c) / (1 - p_c)) if p_c < 1 else 0.0), "logreg", []

    def on_game_end(self, obs: dict, game: int) -> list[tuple[str, dict]]:
        if not self.learning:
            return []
        new = self._absorb(obs, game)
        lr_mem = self.memory.setdefault("logreg", {})
        for e in sorted(new, key=lambda e: (e["game"], e["round"])):
            m = lr_mem.setdefault(e["by"], {"w": [0.0] * DIM, "n": 0, "tell_coef": [0.0, 0.0, 0.0],
                                            "tell_history": [], "discovery_game": None})
            w = _padw(m["w"])
            x = featurize(e["features"], e["p_cards"])
            g = (_sig(float(w @ x)) - float(e["was_bluff"])) * x
            reg = L2 * w
            reg[0] = 0.0
            m["w"] = list(map(float, w - LR * (g + reg)))
            m["n"] += 1
        for about, m in lr_mem.items():
            es = sorted([e for e in self.memory["revealed"] if e["by"] == about], key=lambda e: (e["game"], e["round"]))
            if not es:
                continue
            if self.params.get("info") == "raw":
                m["tell_coef"] = None            # n/a: no formal_register input in raw mode (BoW only)
                m["tell_coef_bow"] = None
                m["tell_history"].append({"game": game, "coef": None})
                continue
            # Discovery statistic. The BoW weight w[formal_register] is not identifiable on its own (it is
            # collinear with the hashed tokens of formal lines): a Laplace interval is ~1/λ wide, and a
            # bootstrap of the online fit gave 16/20 false discoveries in a null test (formal independent of
            # bluffing). So the reported tell coefficient comes from a reduced logistic model
            # [bias, logit(p_cards), formal_register], fitted to the L2-MAP (λ=1e-3) by Newton's method on
            # the same revealed claims, with an 80% Wald/Laplace interval. The BoW weight is kept as
            # tell_coef_bow.
            Xr = np.stack([[1.0, featurize(e["features"], e["p_cards"])[1], float(bool(e["features"].get("formal_register")))]
                           for e in es])
            y = np.array([float(e["was_bluff"]) for e in es])
            wr = np.zeros(3)
            for _ in range(25):
                pr = 1.0 / (1.0 + np.exp(-np.clip(Xr @ wr, -35, 35)))
                Hm = (Xr * (pr * (1 - pr))[:, None]).T @ Xr + L2 * np.eye(3)
                gr = Xr.T @ (pr - y) + L2 * wr
                wr = wr - np.linalg.solve(Hm, gr)
            pr = 1.0 / (1.0 + np.exp(-np.clip(Xr @ wr, -35, 35)))
            Hm = (Xr * (pr * (1 - pr))[:, None]).T @ Xr + L2 * np.eye(3)
            se = math.sqrt(max(float(np.linalg.inv(Hm)[2, 2]), 0.0))
            c = float(wr[2])
            lo, hi = c - Z80 * se, c + Z80 * se
            m["tell_coef_bow"] = float(m["w"][I_FORMAL])
            m["tell_coef"] = [c, lo, hi]
            m["tell_history"].append({"game": game, "coef": m["tell_coef"]})
            if m.get("discovery_game") is None and c > 0 and m["tell_coef"][1] > 0:
                m["discovery_game"] = game
        self.memory["memory_version"] += 1
        self.memory = _jsonable(self.memory)
        return []


# ------------------------------------------------------------------ profiler_plus (plus01)

PLUS_GAMMA = 0.9          # likelihood weight 0.9^(games ago)
PLUS_PRIOR_VAR = 1.0      # prior N(w0, I)
Z95 = 1.959963984540054
BASE_NAMES = ["bias", "logit_p_cards"] + [n for n, _ in NUMERIC] + ["was_challenged_last_round"]


def base_x(features: dict, p_cards: float) -> list[float]:
    pc = min(max(float(p_cards), 1e-3), 1 - 1e-3)
    return ([1.0, math.log(pc / (1 - pc))] + [float(features.get(n, 0) or 0) / sc for n, sc in NUMERIC]
            + [float(bool(features.get("was_challenged_last_round")))])


class ProfilerPlus(Profiler):
    """"LLM proposes, statistics disposes." Every usable hypothesis (open, not refuted, not always-true,
    evidence provenance — untrusted only if independently confirmed) is a binary feature in a per-opponent
    L2-MAP logistic calibrator with logit(p_cards) and the base numeric features (no formal_register, no BoW:
    the LLM's predicates are the only lexical channel). Prior N(w0, I), w0 = [0, 1, 0…] ⇒ with no data
    p_final = p_cards (counter). Likelihood weights 0.9^(games ago). Fit at game end (and right after a
    reflection adds lines) — memory timing rule, so replay is exact.
    False-discovery guard: a line is scored (status) on ALL reveals about its opponent EXCEPT the batch the
    LLM was shown when it proposed the line (h["holdout_obs"]); refuted lines drop out of the calibrator.
    Caveat: SurrealDB retrieval notes may show the LLM summaries of older reveals; those are not held out.
    No supersession between non-drift lines: correlated lines simply share weight in the regression."""
    id = "profiler_plus"

    def wants_reflection(self, game: int) -> bool:
        k = int(self.params.get("plus_reflect_every", 1) or 0)
        return self.reflects and self.learning and k > 0 and game % k == 0

    # ----- feature set
    def usable(self, about: str) -> list[dict]:
        out = []
        for h in self.memory.get("hypotheses", []):
            if h.get("about") != about or h.get("valid_until_game") is not None or h.get("status") == "refuted":
                continue
            if H.is_always_true(h["predicate"]):
                continue
            if h.get("provenance") == "untrusted" and not H._independently_confirmed(h):
                continue
            out.append(h)
        return out

    def predict(self, obs, about, features, p_c):
        cal = (self.memory.get("plus") or {}).get(about) or {}
        coefs = dict(zip(cal.get("names", []), cal.get("w", [])))
        ivs = dict(zip(cal.get("names", []), cal.get("iv", [])))
        xb = base_x(features, p_c)
        w0 = [0.0, 1.0] + [0.0] * (len(BASE_NAMES) - 2)
        z, terms = 0.0, {}
        for i, n in enumerate(BASE_NAMES):
            c = coefs.get(n, w0[i])
            z += c * xb[i]
            terms[n] = c * xb[i]
        mem = []
        for h in self.usable(about):
            on = bool(H.eval_predicate(h["predicate"], features))
            c = coefs.get(h["id"])           # None: proposed after the last fit (or removed) → no weight yet
            contrib = (c if c is not None else 0.0) * float(on)
            z += contrib
            e = H._mem_entry(h, "feature", on)
            e.update(coef=c, coef_iv=ivs.get(h["id"]), contribution=contrib)
            mem.append(e)
        p = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))
        return None, f"plus:{about}", mem, p, {"plus": {"logit": z, "base_terms": terms, "n_fit": int(cal.get("n", 0)),
                                                       "fit_game": cal.get("game")}}

    def estimate(self, obs, about, features):      # for want_ask
        p_c = float(features["card_math"]["p_cards"])
        p = self.predict(obs, about, features, p_c)[3]
        return (max(0.0, (p - p_c) / (1 - p_c)) if p_c < 1 else 0.0), f"plus:{about}", \
            self.predict(obs, about, features, p_c)[2]

    # ----- fitting
    def _fit(self, about: str, game: int) -> None:
        es = [e for e in self.memory["revealed"] if e["by"] == about]
        hs = self.usable(about)
        names = BASE_NAMES + [h["id"] for h in hs]
        d = len(names)
        w0 = np.zeros(d)
        w0[1] = 1.0
        if es:
            X = np.array([base_x(e["features"], e["p_cards"]) + [float(bool(H.eval_predicate(h["predicate"], e["features"])))
                                                               for h in hs] for e in es])
            y = np.array([float(e["was_bluff"]) for e in es])
            sw = np.array([PLUS_GAMMA ** max(0, game - e["game"]) for e in es])
        else:
            X, y, sw = np.zeros((0, d)), np.zeros(0), np.zeros(0)
        w = w0.copy()
        P = np.eye(d) / PLUS_PRIOR_VAR

        def obj(v):
            z = X @ v
            return float(np.sum(sw * (np.logaddexp(0.0, z) - y * z)) + 0.5 * (v - w0) @ P @ (v - w0))

        f = obj(w)
        for _ in range(50):                      # damped Newton (backtracking) on the convex MAP objective
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
            "iv": [[float(w[i] - Z95 * se[i]), float(w[i] + Z95 * se[i])] for i in range(d)],
            "n": len(es), "game": game}

    def _refit_all(self, game: int) -> None:
        opps = sorted({e["by"] for e in self.memory["revealed"]} | {h["about"] for h in self.memory["hypotheses"]})
        for o in opps:
            self._fit(o, game)

    def _score_all(self, game: int) -> None:
        rng = self._score_rng(game)
        for h in self.memory["hypotheses"]:
            if h.get("valid_until_game") is not None or h.get("origin") == "drift":
                continue
            hold = set(h.get("holdout_obs") or [])
            H.score_hypothesis(h, [e for e in self.memory["revealed"] if e.get("obs_id") not in hold], rng,
                               now_game=game)

    def on_game_end(self, obs: dict, game: int) -> list[tuple[str, dict]]:
        if not self.learning:
            return []
        self._absorb(obs, game)
        mem = self.memory
        before = {h["id"]: self._sig(h) for h in mem["hypotheses"]}
        self._score_all(game)
        rng = self._score_rng(game)
        rnd = int(obs.get("round", 0))
        sup, new_ids = [], []
        for opp in sorted({e["by"] for e in mem["revealed"]}):
            nh, evs, p = H.apply_drift(mem["hypotheses"], mem["revealed"], wid=self.params.get("wid", "w000"),
                                      about=opp, idx=mem["next_idx"], now_game=game, now_round=rnd, rng=rng)
            if nh is not None:
                mem["next_idx"] += 1
                new_ids.append(nh["id"])
                mem["drift_notes"].append({"about": opp, "game": game, "p": round(p, 4), "hyp_id": nh["id"]})
                sup += evs
        self._refit_all(game)
        events = [("hypothesis", H.hypothesis_event(h)) for h in mem["hypotheses"]
                  if h["id"] in new_ids or before.get(h["id"]) != self._sig(h)]
        mem["memory_version"] += 1
        self.memory = _jsonable(mem)
        return events + sup

    def apply_reflection(self, proposed, game):
        start = self.memory.get("reflected_upto", 0)
        batch = [e["obs_id"] for e in self.memory["revealed"][start:][-40:]]
        n0 = len(self.memory["hypotheses"])
        events = super().apply_reflection(proposed, game)
        for h in self.memory["hypotheses"][n0:]:
            h["holdout_obs"] = batch
            h["score_from_game"] = 0
        if len(self.memory["hypotheses"]) > n0:
            self._score_all(game)
            self._refit_all(game)
            events = [("hypothesis", H.hypothesis_event(h)) for h in self.memory["hypotheses"][n0:]]
        self.memory = _jsonable(self.memory)
        return events


AGENTS = {c.id: c for c in (Counter, Bayes, BayesDiscount, Profiler, ProfilerBlind, ProfilerAmnesic, LogregBow,
                            ProfilerPlus)}


def make_agent(agent_id: str, seat: str, rng, params: dict | None = None):
    if agent_id not in AGENTS:
        raise KeyError(f"unknown agent {agent_id!r}; have {sorted(AGENTS)}")
    return AGENTS[agent_id](seat, rng, params)

import blab.agents.llm_agents  # noqa: E402,F401  (llm-arms registry hook: registers llm_* agents in AGENTS)


# ------------------------------------------------------------------ logreg_embed (b01, PREREG.md)

EMB_NUMERIC = ["bias", "logit_p_cards"] + [n for n, _ in NUMERIC] + ["was_challenged_last_round", "word_count",
                                                                     "answer_word_count"]
EMB_LAMBDA, EMB_LR = 1.0, 0.1          # fixed a priori (PREREG); no tuning


def featurize_embed(features: dict, p_cards: float) -> np.ndarray:
    """[64-dim fixed random projection (seed 0) of embed(utterance)] ⊕ [same for embed(answer), zeros if no
    answer] ⊕ raw numeric features. Embeddings come only from blab/data/embed_cache.json (no network)."""
    from blab.agents.embed import projected
    pc = min(max(float(p_cards), 1e-3), 1 - 1e-3)
    num = [1.0, math.log(pc / (1 - pc))] + [float(features.get(n, 0) or 0) / s for n, s in NUMERIC]
    num += [float(bool(features.get("was_challenged_last_round"))),
            float(features.get("word_count", len(str(features.get("utterance", "")).split()))) / 10.0,
            float(features.get("answer_word_count", len(str(features.get("answer_utterance", "") or "").split()))) / 10.0]
    return np.concatenate([np.asarray(num), np.asarray(projected(str(features.get("utterance", "") or ""))),
                           np.asarray(projected(str(features.get("answer_utterance", "") or "")))])


class LogregEmbed(LogregBow):
    """b01: online L2 logistic regression (λ 1.0, lr 0.1, per-step penalty as in logreg_bow) on sentence
    embeddings + numeric features; trained at game end on the same revealed claims as everyone else."""
    id = "logreg_embed"

    def predict(self, obs, about, features, p_c):
        m = (self.memory.get("logreg_embed") or {}).get(about)
        x = featurize_embed(features, p_c)
        w = np.asarray(m["w"]) if m else np.zeros(len(x))
        p = _sig(float(w @ x))
        return None, "logreg", [], p, {"logreg": {"p_bluff": p, "n_fit": int(m["n"]) if m else 0,
                                                 "tell_coef": None, "tell_feature": None, "discovery_game": None,
                                                 "model": "logreg_embed"}}

    def on_game_end(self, obs: dict, game: int) -> list[tuple[str, dict]]:
        if not self.learning:
            return []
        new = self._absorb(obs, game)
        mem = self.memory.setdefault("logreg_embed", {})
        for e in sorted(new, key=lambda e: (e["game"], e["round"])):
            x = featurize_embed(e["features"], e["p_cards"])
            m = mem.setdefault(e["by"], {"w": [0.0] * len(x), "n": 0})
            w = np.asarray(m["w"])
            g = (_sig(float(w @ x)) - float(e["was_bluff"])) * x
            reg = EMB_LAMBDA * w
            reg[0] = 0.0
            m["w"] = list(map(float, w - EMB_LR * (g + reg)))
            m["n"] += 1
        self.memory["memory_version"] = int(self.memory.get("memory_version", 0)) + 1
        return []


AGENTS[LogregEmbed.id] = LogregEmbed

import blab.agents.c01  # noqa: E402,F401  (c01 registry hook: bayes_discount_pp, profiler_plus_c01, profiler_hybrid, profiler_plus_noreflect)
