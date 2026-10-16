"""World runner (SPEC §8, INTERFACES.md): games loop, checkpoints, reflection handshake, metrics.

`blab local` → `run_local`. Replay/branch live in blab/replay.py and are re-exported here.
"""
from __future__ import annotations

import asyncio
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from blab.agents.ours import AGENTS, make_agent
from blab.agents.scripted import Scripted
from blab.engine import Engine
from blab.events import EventWriter

Reflector = Callable[[str, str, dict], Awaitable[dict | None]]

# opponents for seats p2, p3 (§6). NOTE: longcon puts the sleeper at p3 (SPEC lists [sleeper, telltale]):
# only the immediate next actor can challenge, and p1 always follows p3, so p1 only ever *judges* p3's
# claims. RTD is defined on p1's p_final about the sleeper, so the sleeper must sit at p3.
TABLES: dict[str, list[str]] = {
    "telltale": ["control", "telltale"],
    "telltale_overt": ["control", "telltale_overt"],
    "longcon": ["telltale", "sleeper"],
    "falsealarm": ["control", "control"],
    "collude": ["colluder", "colluder"],
}

REFLECT_TIMEOUT_S = 60.0


def make_seats(table: str, agent: str, seed: int) -> list[dict]:
    roles = TABLES[table]
    seats = [{"seat": "p1", "policy": agent, "role": None, "params": {}}]
    for s, role in zip(("p2", "p3"), roles):
        params: dict[str, Any] = {}
        if role == "sleeper":
            params["G0"] = int(np.random.default_rng([int(seed), 5150]).integers(5, 9))   # G0 ~ U{5..8}
        if role == "colluder":
            params["partner"] = "p3" if s == "p2" else "p2"
        seats.append({"seat": s, "policy": "scripted", "role": role, "params": params})
    return seats


# ------------------------------------------------------------------ stub reflection (no network)

CANNED = [
    ("verbal_tell", "{P} speaks formally (no contractions, 4+ words) when bluffing",
     {"f": "formal_register", "op": "==", "v": True}),
    ("verbal_tell", "{P} says 'trust me' when bluffing", {"f": "utterance", "op": "contains", "v": "trust me"}),
    ("pressure", "{P} bluffs more when down to two cards or fewer", {"f": "hand_size", "op": "<=", "v": 2}),
    ("exploits_trust", "{P} cashes in after a long unchallenged streak", {"f": "trusted_streak", "op": ">=", "v": 4}),
    ("baseline_bluff", "{P} bluffs more on multi-card claims", {"f": "claim_count", "op": ">=", "v": 2}),
    ("trigger", "{P} bluffs right after being challenged", {"f": "was_challenged_last_round", "op": "==", "v": True}),
]
CANNED_ASK = [("verbal_tell", "{P} answers questions formally when bluffing",
               {"f": "answer_formal_register", "op": "==", "v": True})]


def _pname(seat: str) -> str:
    return f"Player {seat[1:]}"


async def stub_reflector(wid: str, seat: str, payload: dict) -> dict:
    """Deterministic canned proposals (≤ 3 per call), skipping lines already in memory. No network.
    Opponent order p3 → p2 (p1 judges p3's claims directly)."""
    hs = payload.get("hypotheses", [])
    k = lambda h: (h.get("about"), json.dumps(h.get("predicate"), sort_keys=True))  # noqa: E731
    have = {k(h) for h in hs if h.get("status") != "refuted" and h.get("valid_until_game") is None}
    # a refuted line is re-proposed only if its overall evidence is still strong (p_gap > 0.9), i.e. it
    # was refuted by the 2-game recent-window rule rather than by the bulk of the evidence
    revive = {k(h): h["id"] for h in hs if h.get("status") == "refuted" and h.get("valid_until_game") is None
              and (h.get("p_gap") or 0) > 0.9}
    seen = {k(h) for h in hs}
    opps = sorted(payload.get("opponents") or [s for s in ("p2", "p3") if s != seat], reverse=True)
    out = []
    canned = CANNED + (CANNED_ASK if payload.get("ask") else [])
    for aspect, text, pred in canned:
        for o in opps:
            key = (o, json.dumps(pred, sort_keys=True))
            if key in have or (key in seen and key not in revive):
                continue
            out.append({"about": o, "aspect": aspect, "text": text.format(P=_pname(o)), "predicate": pred,
                        "rationale": "stub reflector: canned hypothesis list (no LLM)"
                                     + ("; re-proposed: refuted by recent window but overall P>0.9" if key in revive else ""),
                        "supersedes_id": revive.get(key), "origin": "stub"})
            if len(out) == 3:
                return {"proposed": out, "source": "stub", "llm_trace_id": None, "surreal_trace_id": None}
    return {"proposed": out, "source": "stub", "llm_trace_id": None, "surreal_trace_id": None}


# ------------------------------------------------------------------ World

class World:
    def __init__(self, *, wid: str, seats: list[dict], seed: int, games: int, reflect_every: int = 2,
                 ask: bool = False, out_dir: str | Path | None = None, run: str = "local", box: str = "local",
                 variant: dict | None = None, learning: bool = True, reflector: Reflector | None = None,
                 parent_world: str | None = None, from_game: int | None = None,
                 start_ckpt: dict | None = None, table: str | None = None,
                 reflect_timeout: float = REFLECT_TIMEOUT_S, fresh: bool = True):
        self.wid, self.seats, self.seed, self.games = wid, copy.deepcopy(seats), int(seed), int(games)
        self.reflect_every, self.ask, self.run_id, self.box = reflect_every, ask, run, box
        self.variant, self.learning = variant, learning
        self.reflector = reflector or stub_reflector
        self.parent_world, self.from_game, self.table = parent_world, from_game, table
        self.reflect_timeout = reflect_timeout
        self.dir = Path(out_dir) / wid if out_dir else None
        if self.dir and fresh and self.dir.exists():
            shutil.rmtree(self.dir)
        self.events = EventWriter(self.dir / "events.jsonl" if self.dir else None, run, wid, box)
        if start_ckpt:
            self.engine = Engine.restore(start_ckpt["engine"], wid=wid)
            self.start_game = int(start_ckpt["game"])
        else:
            self.engine = Engine(self.seed, wid=wid)
            self.start_game = 0
        self.policies: dict[str, Any] = {}
        for sp in self.seats:
            s = sp["seat"]
            rng = self.engine.seat_rngs[s]
            if sp["policy"] == "scripted":
                self.policies[s] = Scripted(s, rng, {**sp.get("params", {}), "role": sp["role"]})
            else:
                params = {**sp.get("params", {}), "wid": wid, "seed": self.seed,
                          "reflect_every": reflect_every, "learning": learning}
                self.policies[s] = make_agent(sp["policy"], s, rng, params)
                if start_ckpt and hasattr(self.policies[s], "set_memory"):
                    self.policies[s].set_memory((start_ckpt.get("memory") or {}).get(s))
        self.status = "created"
        self._paused = False
        self.pending_reflection: dict | None = None    # payload while awaiting_reflection (for worldd)
        self.ckpts: dict[int, dict] = {}
        self.round_index: dict[tuple[int, int], int] = {}
        self.result: dict | None = None
        self.fallbacks = 0

    # ------------------------------------------------------------ control
    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    async def _gate(self) -> None:
        while self._paused:
            self.status = "paused"
            await asyncio.sleep(0.05)
        self.status = "running"

    def memory_agents(self) -> dict[str, Any]:
        return {s: p for s, p in self.policies.items() if hasattr(p, "get_memory")}

    # ------------------------------------------------------------ disk
    def meta(self) -> dict:
        return {"wid": self.wid, "table": self.table, "seed": self.seed, "games": self.games,
                "reflect_every": self.reflect_every, "seats": self.seats, "status": self.status,
                "parent_world": self.parent_world, "variant": self.variant, "from_game": self.from_game,
                "learning": self.learning, "run": self.run_id, "box": self.box, "ask": self.ask,
                "agent": next((sp["policy"] for sp in self.seats if sp["seat"] == "p1"), None),
                "start_game": self.start_game}

    def _write_meta(self) -> None:
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "meta.json").write_text(json.dumps(self.meta(), indent=1))

    def checkpoint(self, g: int) -> dict:
        ck = {"game": g, "wid": self.wid, "engine": self.engine.snapshot(),
              "memory": {s: p.get_memory() for s, p in self.memory_agents().items()}}
        ck = json.loads(json.dumps(ck))
        self.ckpts[g] = ck
        if self.dir:
            (self.dir / "ckpt").mkdir(parents=True, exist_ok=True)
            (self.dir / "ckpt" / f"g{g:02d}.json").write_text(json.dumps(ck))
            (self.dir / "memory").mkdir(parents=True, exist_ok=True)
            for s, m in ck["memory"].items():
                (self.dir / "memory" / f"{s}.json").write_text(json.dumps(m, indent=1))
        return ck

    def emit(self, typ: str, payload: dict, *, game: int = 0, round: int = 0, actor: str | None = None) -> dict:
        return self.events.emit(typ, payload, game=game, round=round, actor=actor)

    # ------------------------------------------------------------ run
    async def run(self) -> dict:
        self.status = "running"
        self._write_meta()
        self.emit("world_start", {"seats": self.seats, "seed": self.seed, "games": self.games, "box": self.box,
                                  "snapshot_id": None, "variant": self.variant, "parent_world": self.parent_world,
                                  "from_game": self.from_game, "learning": self.learning,
                                  "reflect_every": self.reflect_every, "table": self.table},
                  game=self.start_game)
        self.checkpoint(self.start_game)
        try:
            for g in range(self.start_game + 1, self.games + 1):
                await self._gate()
                await self.play_game(g)
            self.result = self.metrics()
            self.emit("world_end", self.result, game=self.games)
            self.status = "done"
        except BaseException:
            self.status = "error"
            raise
        finally:
            self._write_meta()
            self.events.close()
        return self.result

    async def play_game(self, g: int) -> None:
        eng = self.engine
        self.emit("game_start", eng.start_game(g), game=g)
        while not eng.game_over:
            deal = eng.start_round()
            r = eng.round
            self.round_index[(g, r)] = eng.global_round
            self.emit("deal", deal, game=g, round=r)
            while eng.round_active:
                seat = eng.to_act
                obs = eng.observation(seat)
                if self.ask and hasattr(self.policies[seat], "want_ask"):
                    await self._interrogate(seat, obs, g, r)
                act = self.policies[seat].act(obs)
                decs = act.get("decisions") or ([act["decision"]] if act.get("decision") else [])
                for d in decs:
                    d.setdefault("trace", {})
                    self.emit("decision", d, game=g, round=r, actor=seat)
                for typ, payload, actor in eng.apply(seat, act):
                    self.emit(typ, payload, game=g, round=r, actor=actor)
            await asyncio.sleep(0)
        rnd = eng.round
        self.emit("game_end", eng.game_end_payload(), game=g, round=rnd)
        for seat, pol in self.memory_agents().items():
            if not self.learning or not getattr(pol, "learning", True):
                continue
            obs = eng.observation(seat)
            for typ, payload in pol.on_game_end(obs, g):
                self.emit(typ, payload, game=g, round=rnd, actor=seat)
            if pol.wants_reflection(g):
                await self._reflect(seat, pol, g, rnd)
        self.checkpoint(g)
        self._write_meta()

    async def _interrogate(self, seat: str, obs: dict, g: int, r: int) -> None:
        """§7.5 (sim: text only). The answer's features are added to the pending claim's features as
        answer_* (usable by predicates) and the ask record goes into obs["ask"] (so /replay sees it)."""
        q = self.policies[seat].want_ask(obs)
        if not q:
            return
        tgt = q["target"]
        self.emit("ask", dict(q), game=g, round=r, actor=seat)
        pol = self.policies[tgt]
        text = pol.answer(q["text"], self.engine.observation(tgt)) if hasattr(pol, "answer") else ""
        from blab.engine import formal_register, has_contraction
        feats = {"answer_formal_register": formal_register(text), "answer_utterance": text,
                 "answer_has_contraction": has_contraction(text)}
        self.emit("answer", {"text": text, "latency_ms": 0, "source": "scripted", "features": feats},
                  game=g, round=r, actor=tgt)
        obs["pending_claim"]["features"].update(feats)
        self.engine.annotate_pending(feats)      # public: lands in revealed_history if challenged
        obs["ask"] = {"asked": True, "template_id": q["template_id"], "text": q["text"], "kind": q["kind"],
                      "answer": text, "reason_aspect": q["reason_aspect"]}

    async def _reflect(self, seat: str, pol, g: int, rnd: int) -> None:
        payload = pol.reflect_payload()
        payload["ask"] = bool(self.ask)
        self.emit("reflect_request", {"seat": seat, "memory_version": payload["memory_version"],
                                      "new_obs": [e["obs_id"] for e in payload["new_obs"]]},
                  game=g, round=rnd, actor=seat)
        self.status, self.pending_reflection = "awaiting_reflection", {**payload, "game": g}
        t0 = time.monotonic()
        res, err, source = None, None, None
        try:
            res = await asyncio.wait_for(self.reflector(self.wid, seat, payload), self.reflect_timeout)
            if res is None:
                err, source = "no reply", "timeout"
        except asyncio.TimeoutError:
            err, source = f"timeout after {self.reflect_timeout:.0f}s; kept old memory", "timeout"
        except Exception as e:  # noqa: BLE001 — a box never blocks/crashes on the LLM
            err, source = f"{type(e).__name__}: {e}", "error"
        latency = int((time.monotonic() - t0) * 1000)
        self.status, self.pending_reflection = "running", None
        evs = []
        proposed = []
        if res is not None and not res.get("error"):
            proposed = list(res.get("proposed") or [])
            evs = pol.apply_reflection(proposed, g)
        elif res is not None:
            err, source = str(res.get("error")), res.get("source") or "error"
        self.emit("reflect_result", {
            "seat": seat, "memory_version": pol.get_memory()["memory_version"], "proposed": proposed,
            "llm_trace_id": (res or {}).get("llm_trace_id"), "surreal_trace_id": (res or {}).get("surreal_trace_id"),
            "latency_ms": latency, "error": err, "source": (res or {}).get("source") or source,
            "dropped": getattr(pol, "last_dropped", []) if res is not None else []},
            game=g, round=rnd, actor=seat)
        for typ, p in evs:
            self.emit(typ, p, game=g, round=rnd, actor=seat)

    # ------------------------------------------------------------ metrics (§6)
    def metrics(self) -> dict:
        return compute_metrics(self.events.events, self.seats, start_game=self.start_game)


def compute_metrics(events: list[dict], seats: list[dict], start_game: int = 0) -> dict:
    ends = [e for e in events if e["type"] == "game_end" and e["game"] > start_game]
    n = len(ends)
    won = sum(1 for e in ends if e["payload"]["winner"] == "p1")
    late = [e for e in ends if 11 <= e["game"] <= 20]
    reveals = {(e["game"], e["round"]): e["payload"] for e in events if e["type"] == "reveal"}
    resp = [e for e in events if e["type"] == "decision" and e["actor"] == "p1" and e["payload"].get("kind") == "respond"]
    bs = [(e["payload"]["p_final"] - float(reveals[(e["game"], e["round"])]["was_bluff"])) ** 2
          for e in resp if e["payload"]["action"] == "challenge" and (e["game"], e["round"]) in reveals]
    out: dict[str, Any] = {"win_rate": (won / n) if n else 0.0, "games_won": won, "games": n,
                           "win_rate_11_20": (sum(1 for e in late if e["payload"]["winner"] == "p1") / len(late)) if late else None,
                           "brier": (sum(bs) / len(bs)) if bs else None, "brier_n": len(bs)}
    role = {sp["seat"]: sp.get("role") for sp in seats}
    # global round index
    gr, idx = {}, 0
    for e in events:
        if e["type"] == "deal":
            idx += 1
            gr[(e["game"], e["round"])] = idx
    last_round_of_game = {}
    for (g, r), i in gr.items():
        last_round_of_game[g] = max(i, last_round_of_game.get(g, 0))
    # verbal-tell discovery (telltale seats)
    for s, ro in role.items():
        if ro in ("telltale", "telltale_overt"):
            conf = [e for e in events if e["type"] == "hypothesis" and e["payload"].get("about") == s
                    and e["payload"]["aspect"] == "verbal_tell" and e["payload"]["status"] == "confirmed"]
            dg = conf[0]["game"] if conf else None
            out["discovery_game"] = dg
            out["discovery_hyp"] = conf[0]["payload"]["id"] if conf else None
            out["revealed_to_discovery"] = (sum(1 for (g, r), p in reveals.items() if p["claimant"] == s and g <= dg)
                                            if dg is not None else None)
            out["revealed_about_telltale"] = sum(1 for p in reveals.values() if p["claimant"] == s)
    # long con (§6/§7.3)
    sl = next((s for s, ro in role.items() if ro == "sleeper"), None)
    if sl:
        claims = [e for e in events if e["type"] == "claim" and e["actor"] == sl]
        strat = [e for e in claims if e["payload"].get("bluff_kind") == "strategic"]
        R_b = gr[(strat[0]["game"], strat[0]["round"])] if strat else None
        R_d, det_how = None, None
        if R_b is not None:
            prev_hi = False
            for e in resp:
                if e["payload"].get("target") != sl:
                    continue
                i = gr[(e["game"], e["round"])]
                hi = e["payload"]["p_final"] > 0.5
                if i >= R_b and hi and prev_hi:
                    R_d, det_how = i, "p_final>0.5 x2"
                    break
                prev_hi = hi if i >= R_b else False
            sup = [e for e in events if e["type"] == "belief_superseded" and e["payload"].get("about") == sl
                   and last_round_of_game.get(e["game"], 0) >= R_b]
            if sup:
                i = last_round_of_game[sup[0]["game"]]
                if R_d is None or i < R_d:
                    R_d, det_how = i, "belief_superseded"
        out["R_b"], out["R_d"], out["detected_by"] = R_b, R_d, det_how
        out["rtd"] = (R_d - R_b) if (R_b is not None and R_d is not None) else None
        hi = R_d if R_d is not None else idx + 1
        out["revealed_to_detection"] = (sum(1 for (g, r), p in reveals.items() if p["claimant"] == sl and R_b <= gr[(g, r)] < hi)
                                        if R_b is not None else None)
        # exploitation cost: sleeper strategic bluffs p1 accepted (played on) in [R_b, R_d)
        if R_b is not None:
            acc = [e for e in resp if e["payload"].get("target") == sl and e["payload"]["action"] == "play_on"]
            strat_rounds = {}
            for e in strat:
                strat_rounds.setdefault((e["game"], e["round"]), []).append(e["seq"])
            def swallowed(e):
                # the claim p1 responded to is the latest sleeper claim before this decision
                prev = [c for c in claims if c["seq"] < e["seq"] and (c["game"], c["round"]) == (e["game"], e["round"])]
                return bool(prev) and prev[-1]["payload"].get("bluff_kind") == "strategic"
            out["exploitation_cost"] = float(sum(1 for e in acc if R_b <= gr[(e["game"], e["round"])] < hi and swallowed(e)))
            out["exploitation_cost_after_G0"] = float(sum(1 for e in acc if gr[(e["game"], e["round"])] >= R_b and swallowed(e)))
            out["exploitation_cost_def"] = "sleeper strategic bluffs p1 accepted (play_on) between R_b and R_d"
    # false alarm: drift fired about a control opponent
    ctrl = {s for s, ro in role.items() if ro == "control"}
    if ctrl:
        out["false_alarm"] = any(e["type"] == "belief_superseded" and e["payload"].get("about") in ctrl
                                 and e["payload"].get("reason") == "drift" for e in events)
    return out


# ------------------------------------------------------------------ entry points

def _reflector_for(llm: bool) -> tuple[Reflector, str]:
    if llm:
        try:
            from blab.orchestrator.reflect import local_reflector  # integrations-owned
            return local_reflector, "claude"
        except Exception as e:  # noqa: BLE001
            print(f"[blab local] --llm requested but local_reflector unavailable ({e}); using stub")
    return stub_reflector, "stub"


async def run_local(table: str = "telltale", agent: str = "profiler", seed: int = 7, games: int = 20,
                    reflect_every: int = 2, out: str = "runs/local", llm: bool = False, ask: bool = False,
                    wid: str | None = None, quiet: bool = False, learning: bool = True) -> dict:
    if agent not in AGENTS:
        raise SystemExit(f"unknown agent {agent!r}; have {sorted(AGENTS)}")
    if table not in TABLES:
        raise SystemExit(f"unknown table {table!r}; have {sorted(TABLES)}")
    wid = wid or f"{table}-{agent}-s{seed}"
    refl, _ = _reflector_for(llm)
    w = World(wid=wid, seats=make_seats(table, agent, seed), seed=seed, games=games, reflect_every=reflect_every,
              ask=ask, out_dir=out, run="local", box="local", reflector=refl, table=table, learning=learning)
    res = await w.run()
    if not quiet:
        print(json.dumps({"wid": wid, "dir": str(w.dir), **res}, indent=1))
    return res


from blab.replay import (apply_variant, branch, branch_cli, parse_variant, replay_cli,  # noqa: E402
                         replay_decision)

__all__ = ["World", "TABLES", "make_seats", "stub_reflector", "run_local", "compute_metrics",
           "replay_decision", "replay_cli", "branch", "branch_cli", "parse_variant", "apply_variant"]
