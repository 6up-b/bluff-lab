"""reg01 — "The Regulars" (identity-keyed memory) + surgical forgetting. Pre-registered in PREREG_reg01.md.

Season world: 60 games = 20 matches x 3 games, one engine/RNG stream per seed. p2 = control (fixed); p3 = one
regular per match from a seeded schedule (5 blocks, each a permutation of the 4 regulars, no back-to-back).
Identity is public: in p1's observation `by`/`challenger` = "p3" becomes the current regular's name.

Run:  .venv/bin/python -m blab.experiments.regulars --seeds 8000-8023 --arms cheap --workers 8
      .venv/bin/python -m blab.experiments.regulars --seeds 8000-8023 --arms llm --seed-par 4
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from blab.env import load_env

load_env()
os.environ.setdefault("BLAB_BEDROCK_MODEL", "us.anthropic.claude-sonnet-5-5")   # PREREG_reg01: Sonnet 5.5 low
os.environ.setdefault("BLAB_LLM_CONCURRENCY", "6")

from blab.agents import ours  # noqa: E402
from blab.agents.scripted import Scripted  # noqa: E402
from blab.world import World  # noqa: E402

REGULARS = {"Avery": "telltale", "Blake": "reverse_telltale", "Casey": "telltale_overt", "Drew": "tell_conjunctive"}
NAMES = tuple(REGULARS)
MATCHES, GAMES_PER_MATCH = 20, 3
GAMES = MATCHES * GAMES_PER_MATCH
FORGET_AFTER_MATCH = 16                       # PREREG amendment 2: forget at the END of match 16
FORGET_GAME = FORGET_AFTER_MATCH * GAMES_PER_MATCH
RUN_DIR = Path("runs/reg01")
LLM_SEM: asyncio.Semaphore | None = None
MAX_LLM = 6


def schedule(seed: int) -> list[str]:
    """5 blocks, each a permutation of the 4 regulars; a block starting with the previous block's last
    regular is re-drawn from the same generator."""
    rng = np.random.default_rng([int(seed), 8080])
    out: list[str] = []
    for _ in range(5):
        while True:
            blk = [NAMES[i] for i in rng.permutation(len(NAMES))]
            if not out or blk[0] != out[-1]:
                break
        out += blk
    return out


def match_of(game: int) -> int:
    return (int(game) - 1) // GAMES_PER_MATCH + 1


# ------------------------------------------------------------------ arms (p1)

class _IdMixin:
    """Identity-keyed agents may hold beliefs about a regular's name."""

    def allowed_about(self):
        return ("p1", "p2", "p3") + NAMES


class LogregReg(_IdMixin, ours.LogregBow):
    id = "reg_logreg"


class ProfilerPlusReg(_IdMixin, ours.ProfilerPlus):
    id = "reg_profiler_plus"


class ProfilerPlusRegNoReflect(ProfilerPlusReg):
    id = "reg_profiler_plus_noreflect"
    reflects = False


for _c in (LogregReg, ProfilerPlusReg, ProfilerPlusRegNoReflect):
    ours.AGENTS[_c.id] = _c

# arm -> (policy, id_mode "name"|"seat", reset_each_match, uses_llm)
ARMS = {
    "counter": ("counter", "seat", False, False),
    "logreg_match": ("reg_logreg", "name", True, False),
    "logreg_pooled": ("reg_logreg", "seat", False, False),
    "logreg_identity": ("reg_logreg", "name", False, False),
    "profiler_plus": ("reg_profiler_plus", "name", False, True),
    "profiler_plus_noid": ("reg_profiler_plus", "seat", False, True),
    "profiler_plus_noreflect": ("reg_profiler_plus_noreflect", "name", False, False),
}
CHEAP = [a for a, v in ARMS.items() if not v[3]]
LLM_ARMS = [a for a, v in ARMS.items() if v[3]]


# ------------------------------------------------------------------ forget:<name>

def forget(mem: dict | None, name: str) -> tuple[dict | None, dict]:
    """forget:<name> — remove every hypothesis, base-rate cell, calibrator/logreg model, revealed observation
    (and obs id) and drift note about `name` from the box memory. Returns (memory', counts)."""
    if mem is None:
        return None, {}
    m = copy.deepcopy(mem)
    rev = m.get("revealed", [])
    upto = int(m.get("reflected_upto", 0))
    gone_ids = {e.get("obs_id") for e in rev if e.get("by") == name}
    keep_before = sum(1 for e in rev[:upto] if e.get("by") != name)
    counts = {"hypotheses": sum(1 for h in m.get("hypotheses", []) if h.get("about") == name),
              "revealed": len(gone_ids), "base": int(name in (m.get("base") or {})),
              "plus": int(name in (m.get("plus") or {})), "logreg": int(name in (m.get("logreg") or {})),
              "drift_notes": sum(1 for d in m.get("drift_notes", []) if d.get("about") == name)}
    m["hypotheses"] = [h for h in m.get("hypotheses", []) if h.get("about") != name]
    m["revealed"] = [e for e in rev if e.get("by") != name]
    m["observations"] = [o for o in m.get("observations", []) if o not in gone_ids]
    m["reflected_upto"] = keep_before
    for k in ("base", "plus", "logreg"):
        if isinstance(m.get(k), dict):
            m[k].pop(name, None)
    m["drift_notes"] = [d for d in m.get("drift_notes", []) if d.get("about") != name]
    m["memory_version"] = int(m.get("memory_version", 0)) + 1
    return m, counts


# ------------------------------------------------------------------ season world

def _relabel_entry(e: dict, name_of) -> dict:
    e = dict(e)
    nm = name_of(e.get("game"))
    for k in ("by", "challenger"):
        if e.get(k) == "p3" and nm:
            e[k] = nm
    return e


class SeasonWorld(World):
    def __init__(self, *, arm: str, sched: list[str], **kw):
        super().__init__(**kw)
        self.arm, self.sched = arm, sched
        self.id_mode, self.reset_each_match = ARMS[arm][1], ARMS[arm][2]
        self.reflect_log: list[dict] = []

    def name_of_game(self, g) -> str | None:
        try:
            return self.sched[match_of(int(g)) - 1]
        except (TypeError, ValueError, IndexError):
            return None

    def current_name(self) -> str:
        return self.sched[match_of(max(1, self.engine_game)) - 1]

    @property
    def engine_game(self) -> int:
        return int(getattr(self, "_cur_game", 1))

    def observe(self, seat: str) -> dict:
        obs = super().observe(seat)
        if seat != "p1" or self.id_mode != "name":
            return obs
        o = dict(obs)
        nm = self.name_of_game(obs.get("game"))
        if o.get("pending_claim") and o["pending_claim"].get("by") == "p3":
            o["pending_claim"] = dict(o["pending_claim"], by=nm)
        o["revealed_history"] = [_relabel_entry(e, self.name_of_game) for e in obs.get("revealed_history", [])]
        o["round_claims"] = [dict(c, by=nm) if c.get("by") == "p3" else c for c in obs.get("round_claims", [])]
        o["speaker_names"] = {"p3": nm}
        return o

    def emit(self, typ, payload, *, game=0, round=0, actor=None):
        # Event schema is frozen (about/target are seats): names map back to "p3", the name rides along.
        if typ in ("decision", "hypothesis", "belief_superseded") and isinstance(payload, dict):
            payload = dict(payload)
            for k in ("target", "about"):
                if payload.get(k) in NAMES:
                    payload[f"{k}_name"] = payload[k]
                    payload[k] = "p3"
            if payload.get("memory_used"):
                mu = []
                for e in payload["memory_used"]:
                    e = dict(e)
                    if e.get("about") in NAMES:
                        e["about_name"], e["about"] = e["about"], "p3"
                    mu.append(e)
                payload["memory_used"] = mu
        return super().emit(typ, payload, game=game, round=round, actor=actor)

    async def play_game(self, g: int) -> None:
        self._cur_game = g
        if getattr(self, "_pending_surgery", None):
            self.emit("surgery", self._pending_surgery, game=g - 1)
            self._pending_surgery = None
        if (g - 1) % GAMES_PER_MATCH == 0:
            nm = self.sched[match_of(g) - 1]
            role = REGULARS[nm]
            self.seats[2]["role"] = role
            self.seats[2]["params"] = {"name": nm}
            self.policies["p3"] = Scripted("p3", self.engine.seat_rngs["p3"], {"role": role})
            if self.reset_each_match:
                self.policies["p1"].set_memory(None)
        await super().play_game(g)


def make_reflector(world: SeasonWorld):
    """Identity-keyed retrieval: the payload is filtered to the current regular (+ p2); the regular is shown
    to Claude as `p3` (names appear only as speaker labels; here not at all), proposals map back."""
    from blab.orchestrator import llm

    async def refl(wid: str, seat: str, payload: dict) -> dict | None:
        g = int(payload.get("game") or world.engine_game)
        nm = world.name_of_game(g)
        key = nm if world.id_mode == "name" else "p3"
        p = copy.deepcopy(payload)
        p["new_obs"] = [dict(e, by=("p3" if e.get("by") == key else e.get("by")),
                             challenger=("p3" if e.get("challenger") == key else e.get("challenger")))
                        for e in p.get("new_obs", []) if e.get("by") in (key, "p2")]
        tag, ptag = f"_{key}_", "_p3_"
        idmap = {}
        hy = []
        for h in p.get("hypotheses", []):
            if h.get("about") not in (key, "p2"):
                continue
            h = dict(h)
            if h.get("about") == key:
                h["about"] = "p3"
            for k in ("id", "superseded_by"):
                if isinstance(h.get(k), str) and tag in h[k]:
                    masked = h[k].replace(tag, ptag)
                    idmap[masked] = h[k]
                    h[k] = masked
            hy.append(h)
        p["hypotheses"] = hy
        p["drift_notes"] = [dict(d, about="p3") if d.get("about") == key else d
                            for d in p.get("drift_notes", []) if d.get("about") in (key, "p2")]
        p["opponents"] = ["p2", "p3"]
        p["info"] = "raw"
        t0 = time.time()
        async with LLM_SEM:
            t1 = time.time()
            try:
                out = await asyncio.wait_for(llm.reflect(p, wid=wid, seat=seat), 120)
            except Exception as e:  # noqa: BLE001
                out = {"error": f"{type(e).__name__}: {e}"}
        rec = {"game": g, "name": nm, "key": key, "queue_s": round(t1 - t0, 2), "call_s": round(time.time() - t1, 2),
               "error": (out or {}).get("error") if out else "no backend", "n_new_obs": len(p["new_obs"])}
        if not out or out.get("proposed") is None:
            rec["ok"] = False
            world.reflect_log.append(rec)
            return None
        prop = []
        for h in out.get("proposed") or []:
            h = dict(h)
            if h.get("about") == "p3":
                h["about"] = key
            if h.get("supersedes_id") in idmap:
                h["supersedes_id"] = idmap[h["supersedes_id"]]
            prop.append(h)
        rec.update(ok=True, n_proposed=len(prop), backend=out.get("backend"))
        world.reflect_log.append(rec)
        return {"proposed": prop, "source": "claude", "llm_trace_id": out.get("llm_trace_id"),
                "surreal_trace_id": None}

    return refl


def seats_for(arm: str, seed: int) -> list[dict]:
    pol = ARMS[arm][0]
    p1 = {"info": "raw", "respond_rule": "threshold", "plus_reflect_every": GAMES_PER_MATCH}
    return [{"seat": "p1", "policy": pol, "role": None, "params": p1},
            {"seat": "p2", "policy": "scripted", "role": "control", "params": {}},
            {"seat": "p3", "policy": "scripted", "role": REGULARS[schedule(seed)[0]], "params": {}}]


def _write_season(w: SeasonWorld, extra: dict) -> None:
    if w.dir:
        w.dir.mkdir(parents=True, exist_ok=True)
        (w.dir / "season.json").write_text(json.dumps({
            "arm": w.arm, "seed": w.seed, "schedule": w.sched, "regulars": REGULARS, "id_mode": w.id_mode,
            "reset_each_match": w.reset_each_match, "reflect_log": w.reflect_log,
            "model": os.environ.get("BLAB_BEDROCK_MODEL"), **extra}, indent=1))


async def run_season(arm: str, seed: int, out: Path = RUN_DIR) -> dict:
    wid = f"s{seed}-{arm}"
    w = SeasonWorld(arm=arm, sched=schedule(seed), wid=wid, seats=seats_for(arm, seed), seed=seed, games=GAMES,
                    reflect_every=GAMES_PER_MATCH, out_dir=out, run="reg01", box="local", table="regulars",
                    reflect_timeout=180.0)
    w.reflector = make_reflector(w)
    res = await w.run()
    _write_season(w, {"variant": None, "parent_world": None})
    return {"wid": wid, **{k: res.get(k) for k in ("win_rate",) if isinstance(res, dict)}}


async def run_branch(seed: int, name: str, out: Path = RUN_DIR, parent_arm: str = "profiler_plus",
                     surreal: dict | None = None, variant_name: str | None = None) -> dict:
    """Checkpoint branch at the end of match 16: memory' = forget:<name> (or 'full'), learning ON."""
    pdir = out / f"s{seed}-{parent_arm}"
    ck = json.loads((pdir / "ckpt" / f"g{FORGET_GAME:02d}.json").read_text())
    vname = variant_name or (f"forget:{name}" if name else "full")
    counts = {}
    if name:
        ck["memory"]["p1"], counts = forget(ck["memory"].get("p1"), name)
    wid = f"s{seed}-{parent_arm}-{vname.replace(':', '_')}"
    w = SeasonWorld(arm=parent_arm, sched=schedule(seed), wid=wid, seats=seats_for(parent_arm, seed), seed=seed,
                    games=GAMES, reflect_every=GAMES_PER_MATCH, out_dir=out, run="reg01", box="local",
                    table="regulars", reflect_timeout=180.0, start_ckpt=ck, parent_world=f"s{seed}-{parent_arm}",
                    from_game=FORGET_GAME, variant={"type": vname})
    w.reflector = make_reflector(w)
    # surgery event (frozen schema: op, variant, by, surreal_uncertainty_id; everything else optional extras)
    w._pending_surgery = {"op": "forget", "variant": {"type": "forget", "name": name} if name else {"type": "full"},
                       "by": "operator", "surreal_uncertainty_id": None, "removed": counts,
                       **({"surreal_forget": surreal} if surreal else {})}
    res = await w.run()
    _write_season(w, {"variant": vname, "parent_world": f"s{seed}-{parent_arm}", "from_game": FORGET_GAME,
                      "removed": counts})
    return {"wid": wid, "removed": counts}


# ------------------------------------------------------------------ SurrealDB: person/<name> + forget

async def surreal_forget_demo(seed: int = 8000, out: Path = RUN_DIR, names=("Avery", "Blake")) -> dict:
    """Mirror the no-forget profiler_plus memory at game 48 into SurrealDB as entities person/<name>
    (scope lab/reg01/<wid>), then for each forgotten name: POST /forget {dryRun:true} (semantic preview),
    then the precise soft-delete the API supports (DELETE /entities/person/<name>), and verify that the
    other regulars' entities are untouched. The semantic /forget is never run for real: it matches by
    similarity across the whole context and could delete other experiments' memories."""
    from blab.orchestrator import memory as M
    wid = f"s{seed}-profiler_plus"
    ck = json.loads((out / wid / "ckpt" / f"g{FORGET_GAME:02d}.json").read_text())
    mem = ck["memory"]["p1"]
    scope = f"lab/reg01/{wid}"
    rec: dict = {"seed": seed, "world": wid, "scope": scope, "at_game": FORGET_GAME, "written": {}, "forget": {}}
    async with M.Surreal() as s:
        if not s.ok:
            return {"error": "SurrealDB not configured", "missing": M.missing()}
        for p in ("lab", "lab/reg01", scope):
            await s._req("POST", "/scopes", json={"path": p})
        for nm in NAMES:
            hs = [h for h in mem["hypotheses"] if h.get("about") == nm]
            rev = [e for e in mem["revealed"] if e.get("by") == nm]
            triples = [{"entity": {"type": "person", "name": nm}, "key": f"{h['aspect']}_{h['idx']:03d}",
                        "value": f"{h['text']} [status {h['status']}, n_true {h.get('n_true')}, n_false {h.get('n_false')}]"}
                       for h in hs]
            triples.append({"entity": {"type": "person", "name": nm}, "key": "base_rate",
                            "value": f"{nm}: {len(rev)} revealed claims, {sum(bool(e['was_bluff']) for e in rev)} bluffs"})
            r = await s.write_triples(triples, scope, observed_at=M.game_ts(FORGET_GAME))
            rec["written"][nm] = {"triples": len(triples), "ok": r is not None, "error": s.last_error if r is None else None}
        before = {nm: await s._req("GET", f"/entities/person/{nm}") for nm in NAMES}
        rec["attributes_before"] = {nm: len((b or {}).get("attributes") or []) for nm, b in before.items()}
        for nm in names:
            s.last_error = None
            dry = await s._req("POST", "/forget", json={"query": f"everything about {nm}", "dryRun": True})
            dry_err = s.last_error
            s.last_error = None
            dele = await s._req("DELETE", f"/entities/person/{nm}")
            del_err = s.last_error
            after = {o: await s._req("GET", f"/entities/person/{o}") for o in NAMES}
            rec["forget"][nm] = {
                "dry_run_request": {"method": "POST", "path": "/forget", "body": {"query": f"everything about {nm}", "dryRun": True}},
                "dry_run_response": dry, "dry_run_error": dry_err,
                "real_request": {"method": "DELETE", "path": f"/entities/person/{nm}",
                                 "note": "soft-delete of the entity and all its current attributes/relations"},
                "real_response": dele, "real_error": del_err,
                "attributes_after": {o: (None if a is None else len(a.get("attributes") or [])) for o, a in after.items()},
                "entity_after_status": {o: ("gone/404" if a is None else "present") for o, a in after.items()}}
    (out / "surreal_forget.json").write_text(json.dumps(rec, indent=1))
    return rec


def _cheap_job(args) -> str:
    arm, seed = args
    asyncio.run(run_season(arm, seed))
    return f"s{seed}-{arm}"


async def _llm_seed(seed: int, seed_sem: asyncio.Semaphore, deadline: float | None) -> None:
    async with seed_sem:
        if deadline and time.time() > deadline:
            print(f"[reg01] seed {seed} skipped: launch deadline passed", flush=True)
            return
        t0 = time.time()
        await asyncio.gather(run_season("profiler_plus", seed), run_season("profiler_plus_noid", seed))
        await asyncio.gather(run_branch(seed, "Avery"), run_branch(seed, "Blake"))
        print(f"[reg01] seed {seed} llm done in {time.time() - t0:.0f}s", flush=True)


async def _llm_main(seeds: list[int], seed_par: int, deadline: float | None) -> None:
    global LLM_SEM
    LLM_SEM = asyncio.Semaphore(MAX_LLM)
    sem = asyncio.Semaphore(seed_par)
    await asyncio.gather(*[_llm_seed(s, sem, deadline) for s in seeds])


def _parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="8000-8023")
    ap.add_argument("--arms", default="cheap", help="cheap | llm | comma list of cheap arms")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-par", type=int, default=4)
    ap.add_argument("--launch-deadline", default=None, help="HH:MM local; seeds not started by then are skipped")
    ap.add_argument("--surreal-demo", type=int, default=None)
    a = ap.parse_args(argv)
    if a.surreal_demo is not None:
        print(json.dumps(asyncio.run(surreal_forget_demo(a.surreal_demo)), indent=1)[:4000])
        return
    seeds = _parse_seeds(a.seeds)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if a.arms == "llm":
        dl = None
        if a.launch_deadline:
            hh, mm = map(int, a.launch_deadline.split(":"))
            lt = time.localtime()
            dl = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
        asyncio.run(_llm_main(seeds, a.seed_par, dl))
        return
    arms = CHEAP if a.arms == "cheap" else a.arms.split(",")
    jobs = [(arm, s) for s in seeds for arm in arms]
    with ProcessPoolExecutor(a.workers) as ex:
        for wid in ex.map(_cheap_job, jobs):
            print("[reg01] done", wid, flush=True)


if __name__ == "__main__":
    sys.exit(main())
