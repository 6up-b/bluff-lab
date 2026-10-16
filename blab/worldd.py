"""World box server (SPEC §8). FastAPI on 0.0.0.0:8000:

    uvicorn blab.worldd:app --host 0.0.0.0 --port 8000

Wraps core's `blab.world.World` (imported lazily). Disk layout (§8.2) under $BLAB_WORLDS
(default /home/daytona/worlds on a box, ./runs/box/worlds elsewhere):
    <wid>/events.jsonl  ckpt/gNN.json  memory/<seat>.json  meta.json

Reflection handshake (§8.3): with reflector="orchestrator" (default) the world's reflector parks a Future,
the world reports `awaiting_reflection`, the orchestrator GETs /reflect_payload and PUTs /memory, which
resolves the Future. No reply within 58 s → None → World keeps the old memory. A box never blocks on the LLM.

Auth: if env BLAB_KEY is set every route except /health requires header `X-Blab-Key: $BLAB_KEY`
(Daytona's preview proxy separately checks `x-daytona-preview-token`).
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

REFLECT_WAIT_S = 58.0


def _default_root() -> Path:
    if os.environ.get("BLAB_WORLDS"):
        return Path(os.environ["BLAB_WORLDS"])
    if Path("/home/daytona").exists():
        return Path("/home/daytona/worlds")
    return Path("runs/box/worlds")


ROOT = _default_root()
BOX_ID = os.environ.get("BLAB_BOX_ID") or socket.gethostname()


def cpu_quota() -> int:
    """quota/period from cgroup v2 cpu.max ('max 100000' → os.cpu_count())."""
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if q != "max":
            return max(1, int(int(q) // int(p)))
    except Exception:  # noqa: BLE001
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return max(1, q // p)
    except Exception:  # noqa: BLE001
        pass
    return os.cpu_count() or 1


CPU_QUOTA = cpu_quota()
STARTED = time.time()
_pool: ProcessPoolExecutor | None = None


def pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=CPU_QUOTA)
    return _pool


def _world_mod():
    import blab.world as W  # lazy: core owns it
    return W


# ------------------------------------------------------------------ registry

@dataclass
class Entry:
    wid: str
    spec: dict
    world: Any = None                 # World (in-process) or None (process-pool world)
    task: Optional[asyncio.Task] = None
    proc: Any = None                  # concurrent Future for pool worlds
    status: str = "created"
    awaiting: Optional[dict] = None   # {"seat", "payload", "future"}
    result: Optional[dict] = None
    error: Optional[str] = None
    created: float = field(default_factory=time.time)

    @property
    def dir(self) -> Path:
        run = self.spec.get("run")
        return ROOT / run / self.wid if run else ROOT / self.wid

    def cur_status(self) -> str:
        if self.awaiting is not None:
            return "awaiting_reflection"
        if self.proc is not None:
            if self.proc.done():
                return "error" if self.proc.exception() else "done"
            return "running"
        if self.world is not None:
            st = getattr(self.world, "status", None)
            if self.task is not None and self.task.done():
                return "error" if (self.task.exception() if not self.task.cancelled() else True) else "done"
            if self.task is not None and st in (None, "created"):
                return "running"
            return st or self.status
        return self.status


WORLDS: dict[str, Entry] = {}


def _get(wid: str) -> Entry:
    e = WORLDS.get(wid)
    if e is None:
        for d in [ROOT / wid, *ROOT.glob(f"*/{wid}")]:      # a world from a previous process: read-only
            if (d / "events.jsonl").exists():
                meta = _read_json(d / "meta.json") or {}
                spec = meta.get("spec") or {"wid": wid, "run": d.parent.name if d.parent != ROOT else None}
                spec.setdefault("run", d.parent.name if d.parent != ROOT else None)
                e = Entry(wid=wid, spec=spec, status="done")
                WORLDS[wid] = e
                return e
        raise HTTPException(404, f"unknown world {wid}")
    return e


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ reflectors

def make_box_reflector(wid: str):
    async def reflector(wid_: str, seat: str, payload: dict) -> dict | None:
        e = WORLDS.get(wid)
        if e is None:
            return None
        fut = asyncio.get_running_loop().create_future()
        pend = getattr(e.world, "pending_reflection", None) or {}
        e.awaiting = {"seat": seat, "payload": {**payload, "game": pend.get("game", payload.get("game")),
                                                "wid": wid, "seat": seat}, "future": fut,
                      "since": time.time()}
        try:
            return await asyncio.wait_for(fut, REFLECT_WAIT_S)
        except asyncio.TimeoutError:
            return None
        finally:
            e.awaiting = None
    return reflector


def _pick_reflector(kind: str, wid: str):
    W = _world_mod()
    if kind == "orchestrator":
        return make_box_reflector(wid)
    if kind == "stub":
        return getattr(W, "stub_reflector", None)
    if kind == "claude":            # box calls Claude directly (anthropic/openrouter reachable from boxes)
        from blab.orchestrator.reflect import local_reflector
        return local_reflector
    return None


def _world_kwargs(spec: dict) -> dict:
    kw = dict(wid=spec["wid"], seats=spec["seats"], seed=int(spec["seed"]), games=int(spec.get("games", 20)),
              reflect_every=int(spec.get("reflect_every", 2)), ask=bool(spec.get("ask", False)),
              out_dir=str(ROOT / spec.get("run", "r01")), run=spec.get("run", "r01"), box=BOX_ID)
    for k in ("variant", "learning", "parent_world", "from_game", "table"):
        if k in spec and spec[k] is not None:
            kw[k] = spec[k]
    return kw


def instrument(world, wid: str) -> None:
    """Box-side Arize `decision` spans (§10.3): wrap the world's EventWriter so each decision event gets a span
    and carries trace.arize_trace_id (canonical() strips decision.trace, so determinism is unaffected)."""
    try:
        from blab.orchestrator import tracing
    except Exception:  # noqa: BLE001
        return
    if not tracing.enabled() or getattr(world.events, "_blab_traced", False):
        return
    orig = world.events.emit

    def emit(type, payload, *, game=0, round=0, actor=None, trace_id=None):  # noqa: A002
        if type == "decision" and isinstance(payload, dict):
            try:
                tid = tracing.decision_span(wid, {**payload, "game": game, "round": round, "actor": actor})
                if tid:
                    payload = {**payload, "trace": {**(payload.get("trace") or {}), "arize_trace_id": tid}}
                    trace_id = trace_id or tid
            except Exception:  # noqa: BLE001
                pass
        return orig(type, payload, game=game, round=round, actor=actor, trace_id=trace_id)

    world.events.emit = emit
    world.events._blab_traced = True


def _run_world_proc(spec: dict) -> dict:
    """Process-pool entry point: CPU-only world (stub or no reflection)."""
    import asyncio as _a
    import blab.world as W
    kw = _world_kwargs(spec)
    kind = spec.get("reflector", "stub")
    kw["reflector"] = getattr(W, "stub_reflector", None) if kind == "stub" else None
    w = W.World(**kw)
    if os.environ.get("BLAB_TRACE_DECISIONS") == "all":
        instrument(w, spec["wid"])
    try:
        return _a.run(w.run())
    finally:
        try:
            from blab.orchestrator import tracing
            tracing.flush()
        except Exception:  # noqa: BLE001
            pass


def _run_branch_proc(world_dir: str, new_wid: str, from_game: int, games: int, variant: dict) -> dict:
    import asyncio as _a
    import blab.world as W
    w = W.branch(Path(world_dir), new_wid, from_game, games, variant, learning=False)
    return _a.run(w.run())


def _write_meta(e: Entry) -> None:
    try:
        e.dir.mkdir(parents=True, exist_ok=True)
        meta = _read_json(e.dir / "meta.json") or {}
        meta.update({"spec": e.spec, "box": BOX_ID, "status_box": e.cur_status()})
        (e.dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------ app

app = FastAPI(title="blufflab worldd")


@app.middleware("http")
async def auth(request: Request, call_next):
    key = os.environ.get("BLAB_KEY")
    if key and request.url.path != "/health" and request.headers.get("x-blab-key") != key:
        return JSONResponse({"detail": "bad X-Blab-Key"}, status_code=401)
    return await call_next(request)


@app.get("/health")
def health():
    counts = {"running": 0, "waiting": 0, "done": 0, "created": 0, "paused": 0, "error": 0}
    for e in WORLDS.values():
        st = e.cur_status()
        k = "waiting" if st == "awaiting_reflection" else st
        counts[k] = counts.get(k, 0) + 1
    try:
        import blab.world  # noqa: F401
        world_ok = True
    except Exception as ex:  # noqa: BLE001
        world_ok = f"{type(ex).__name__}: {ex}"
    return {"box_id": BOX_ID, "cpu_quota": CPU_QUOTA, "started": STARTED, "worlds": counts, "root": str(ROOT), "world_module": world_ok}


@app.post("/worlds")
async def create_world(spec: dict = Body(...)):
    """{wid, seats:[{seat,policy,role,params}], seed, games, reflect_every, ask, run?, learning?,
        reflector?: "orchestrator"|"stub"|"claude"|"none", mode?: "async"|"process", start?: bool}"""
    wid = spec.get("wid")
    if not wid or "seats" not in spec or "seed" not in spec:
        raise HTTPException(400, "need wid, seats, seed")
    if wid in WORLDS and WORLDS[wid].cur_status() not in ("done", "error", "created"):
        raise HTTPException(409, f"{wid} exists and is {WORLDS[wid].cur_status()}")
    e = Entry(wid=wid, spec=spec)
    WORLDS[wid] = e
    mode = spec.get("mode") or ("process" if spec.get("reflector", "orchestrator") in ("stub", "none") else "async")
    spec["mode"] = mode
    spec.setdefault("run", "r01")
    if mode == "async":
        W = _world_mod()
        e.world = W.World(**_world_kwargs(spec), reflector=_pick_reflector(spec.get("reflector", "orchestrator"), wid))
        if os.environ.get("BLAB_TRACE_DECISIONS", "llm") != "off":
            instrument(e.world, wid)
    _write_meta(e)
    if spec.get("start"):
        await run_world(wid)
    return {"wid": wid, "status": e.cur_status(), "mode": mode}


async def _run_task(e: Entry):
    try:
        e.result = await e.world.run()
        e.status = "done"
    except Exception as ex:  # noqa: BLE001
        e.error = f"{type(ex).__name__}: {ex}"
        e.status = "error"
        raise
    finally:
        _write_meta(e)
        e.world = None            # free the in-memory event log; disk is authoritative from here on
        try:
            from blab.orchestrator import tracing
            tracing.flush()
        except Exception:  # noqa: BLE001
            pass


@app.post("/worlds/{wid}/run")
async def run_world(wid: str):
    e = _get(wid)
    if e.spec.get("mode") == "process":
        if e.proc is None:
            loop = asyncio.get_running_loop()
            e.proc = loop.run_in_executor(pool(), _run_world_proc, e.spec)
            e.proc.add_done_callback(lambda f: setattr(e, "result", None if f.exception() else f.result()))
        return {"wid": wid, "status": e.cur_status()}
    if e.world is None:
        raise HTTPException(409, "world not runnable (restored read-only)")
    if e.task is None:
        e.task = asyncio.create_task(_run_task(e))
        e.status = "running"
    elif getattr(e.world, "status", "") == "paused" and hasattr(e.world, "resume"):
        e.world.resume()
    return {"wid": wid, "status": e.cur_status()}


@app.post("/worlds/{wid}/pause")
async def pause_world(wid: str):
    e = _get(wid)
    if e.world is None or not hasattr(e.world, "pause"):
        raise HTTPException(409, "pause not supported for this world")
    e.world.pause()
    return {"wid": wid, "status": e.cur_status()}


@app.post("/worlds/{wid}/step")
async def step_world(wid: str, n: int = 1):
    e = _get(wid)
    w = e.world
    if w is None:
        raise HTTPException(409, "not steppable")
    if hasattr(w, "step"):
        out = w.step(n)
        if asyncio.iscoroutine(out):
            out = await out
        return {"wid": wid, "status": e.cur_status(), "result": out}
    raise HTTPException(501, "World.step not implemented by core; use run/pause")


@app.get("/worlds")
def list_worlds(status: Optional[str] = None):
    out = []
    for e in WORLDS.values():
        st = e.cur_status()
        if status and st != status:
            continue
        out.append({"wid": e.wid, "status": st, "run": e.spec.get("run"), "mode": e.spec.get("mode"),
                    "error": e.error or (str(e.proc.exception()) if e.proc is not None and e.proc.done()
                                         and e.proc.exception() else None),
                    "awaiting_seat": e.awaiting["seat"] if e.awaiting else None})
    return out


def _events(e: Entry, since: int) -> list[dict]:
    p = e.dir / "events.jsonl"
    if not p.exists():
        return []
    from blab.events import read_events
    try:
        return read_events(p, since=since)
    except json.JSONDecodeError:          # a half-written last line
        out = []
        for line in p.read_text().splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev["seq"] > since:
                out.append(ev)
        return out


@app.get("/worlds/{wid}/events")
def get_events(wid: str, since: int = 0, limit: int = 0):
    evs = _events(_get(wid), since)
    return evs[:limit] if limit else evs


@app.get("/worlds/{wid}/stream")
async def stream(wid: str, since: int = 0):
    e = _get(wid)

    async def gen():
        last = since
        idle = 0
        while True:
            evs = _events(e, last)
            for ev in evs:
                last = ev["seq"]
                yield f"id: {ev['seq']}\nevent: {ev['type']}\ndata: {json.dumps(ev, separators=(',', ':'))}\n\n"
            if evs:
                idle = 0
            elif e.cur_status() in ("done", "error"):
                idle += 1
                if idle > 3:
                    yield "event: end\ndata: {}\n\n"
                    return
            await asyncio.sleep(0.25)

    return StreamingResponse(gen(), media_type="text/event-stream")


def _ckpt_memory(ck: dict, seat: str) -> dict | None:
    for path in (("memory", seat), ("agents", seat, "memory"), ("memories", seat), ("agent_memory", seat),
                 ("agents", seat)):
        cur: Any = ck
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
        if isinstance(cur, dict):
            return cur
    return None


@app.get("/worlds/{wid}/agents/{seat}/memory")
def get_memory(wid: str, seat: str, as_of_game: Optional[int] = None):
    e = _get(wid)
    if as_of_game is not None:
        ck = _read_json(e.dir / "ckpt" / f"g{as_of_game:02d}.json")
        if ck is None:
            raise HTTPException(404, f"no checkpoint g{as_of_game:02d}")
        mem = _ckpt_memory(ck, seat)
        if mem is None:
            raise HTTPException(404, f"checkpoint has no memory for {seat}")
        return mem
    pol = _policy(e, seat)
    if pol is not None and hasattr(pol, "get_memory"):
        return pol.get_memory()
    mem = _read_json(e.dir / "memory" / f"{seat}.json")
    if mem is None:
        raise HTTPException(404, "no memory")
    return mem


@app.get("/worlds/{wid}/reflect_payload")
def reflect_payload(wid: str):
    e = _get(wid)
    if e.awaiting is None:
        raise HTTPException(409, "world is not awaiting reflection")
    return e.awaiting["payload"]


@app.put("/worlds/{wid}/agents/{seat}/memory")
async def put_memory(wid: str, seat: str, body: dict = Body(...)):
    """{proposed:[hyp]|null, reflect_meta:{llm_trace_id, surreal_trace_id, error, ...}} → resumes the world."""
    e = _get(wid)
    aw = e.awaiting
    if aw is None or aw["seat"] != seat:
        raise HTTPException(409, "world is not awaiting reflection for this seat")
    meta = body.get("reflect_meta") or {}
    proposed = body.get("proposed")
    result = None if proposed is None else {"proposed": proposed, "source": meta.get("source", "claude"),
                                            **{k: v for k, v in meta.items() if k != "source"}}
    if not aw["future"].done():
        aw["future"].set_result(result)
    for _ in range(40):                # let the world apply it, report the new version
        await asyncio.sleep(0.05)
        if e.awaiting is None:
            break
    pol = _policy(e, seat)
    mv = pol.get_memory().get("memory_version") if pol is not None and hasattr(pol, "get_memory") else None
    return {"wid": wid, "seat": seat, "memory_version": mv, "accepted": result is not None}


def _policy(e: Entry, seat: str):
    w = e.world
    if w is None:
        return None
    for attr in ("policies", "agents", "players"):
        d = getattr(w, attr, None)
        if isinstance(d, dict) and seat in d:
            return d[seat]
    getter = getattr(w, "policy", None)
    if callable(getter):
        try:
            return getter(seat)
        except Exception:  # noqa: BLE001
            return None
    return None


@app.post("/worlds/{wid}/decisions/{seq}/replay")
async def replay(wid: str, seq: int, body: dict = Body(default={})):
    e = _get(wid)
    W = _world_mod()
    variant = body.get("memory_variant") or body.get("variant") or {"type": "full"}
    out = await asyncio.to_thread(W.replay_decision, e.dir, seq, variant)
    return out


@app.post("/worlds/{wid}/branch")
async def branch(wid: str, body: dict = Body(...)):
    """{new_wid, from_game, games, learning:false, memory_variant, start?:true}"""
    e = _get(wid)
    W = _world_mod()
    new_wid = body.get("new_wid") or f"{wid}_b{int(time.time()) % 100000}"
    variant = body.get("memory_variant") or {"type": "full"}
    if not body.get("learning", False):          # frozen-learning branch: CPU only → process pool
        spec = {**e.spec, "wid": new_wid, "parent_world": wid, "from_game": body.get("from_game", 10),
                "variant": variant, "learning": False, "mode": "process", "run": body.get("run") or e.spec.get("run")}
        ne = Entry(wid=new_wid, spec=spec)
        WORLDS[new_wid] = ne
        loop = asyncio.get_running_loop()
        ne.proc = loop.run_in_executor(pool(), _run_branch_proc, str(e.dir), new_wid, int(body.get("from_game", 10)),
                                       int(body.get("games", e.spec.get("games", 20))), variant)
        return {"wid": new_wid, "status": ne.cur_status()}
    w = W.branch(e.dir, new_wid, int(body.get("from_game", 10)), int(body.get("games", e.spec.get("games", 20))),
                 variant, learning=bool(body.get("learning", False)))
    spec = {**e.spec, "wid": new_wid, "parent_world": wid, "from_game": body.get("from_game", 10),
            "variant": variant, "learning": bool(body.get("learning", False)), "mode": "async"}
    if body.get("run"):
        spec["run"] = body["run"]
    ne = Entry(wid=new_wid, spec=spec, world=w)
    if os.environ.get("BLAB_TRACE_DECISIONS") == "all":
        instrument(w, new_wid)
    WORLDS[new_wid] = ne
    _write_meta(ne)
    if body.get("start", True):
        await run_world(new_wid)
    return {"wid": new_wid, "status": ne.cur_status()}


@app.post("/worlds/{wid}/agents/{seat}/surgery")
async def surgery(wid: str, seat: str, body: dict = Body(...)):
    """Live edit for the console: {op:"inject", hyp:{...}} | {op:"remove", aspect:A | hyp_id:H}."""
    e = _get(wid)
    pol = _policy(e, seat)
    if pol is None or not hasattr(pol, "get_memory"):
        raise HTTPException(409, "no live memory-bearing policy for that seat")
    mem = json.loads(json.dumps(pol.get_memory()))
    op = body.get("op")
    if op == "inject":
        h = dict(body.get("hyp") or {})
        if not h:
            raise HTTPException(400, "inject needs hyp")
        try:
            from blab.hypotheses import validate_predicate
            ok, why = validate_predicate(h.get("predicate") or {})
            if not ok:
                raise HTTPException(400, f"bad predicate: {why}")
        except ImportError:
            pass
        h.setdefault("id", f"h_{wid}_{h.get('about', 'px')}_surg{len(mem.get('hypotheses', []))}")
        h.setdefault("provenance", "untrusted")
        h.setdefault("status", "provisional")
        mem.setdefault("hypotheses", []).append(h)
    elif op == "remove":
        hs = mem.get("hypotheses", [])
        mem["hypotheses"] = [h for h in hs if not (h.get("aspect") == body.get("aspect")
                                                   or h.get("id") == body.get("hyp_id"))]
    else:
        raise HTTPException(400, "op must be inject|remove")
    mem["memory_version"] = int(mem.get("memory_version", 0)) + 1
    pol.set_memory(mem)
    try:
        ev = getattr(e.world, "events", None)
        if ev is not None:
            ev.emit("surgery", {"op": op, "variant": {k: v for k, v in body.items() if k != "op"}, "by": "operator"},
                    actor=seat)
    except Exception:  # noqa: BLE001
        pass
    return {"wid": wid, "seat": seat, "memory_version": mem["memory_version"]}


@app.get("/worlds/{wid}/results")
def results(wid: str):
    e = _get(wid)
    if e.result:
        return e.result
    for ev in reversed(_events(e, 0)):
        if ev["type"] == "world_end":
            return ev["payload"]
    raise HTTPException(404, "world not finished")


@app.get("/worlds/{wid}/checkpoints")
def checkpoints(wid: str):
    d = _get(wid).dir / "ckpt"
    if not d.exists():
        return []
    return sorted(p.name for p in d.glob("g*.json"))


@app.get("/worlds/{wid}/files")
def files(wid: str):
    """For `blab collect`: every file of the world dir (relative paths)."""
    d = _get(wid).dir
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())


@app.get("/worlds/{wid}/file")
def file(wid: str, path: str):
    d = _get(wid).dir.resolve()
    p = (d / path).resolve()
    if d not in p.parents or not p.is_file():
        raise HTTPException(404, "no such file")
    return StreamingResponse(open(p, "rb"), media_type="application/octet-stream")


@app.get("/export")
def export(run: Optional[str] = None, wids: Optional[str] = None):
    """tar.gz (built on disk, streamed) of the world dirs of `run` or the comma-separated `wids`."""
    import tarfile
    import tempfile
    from fastapi.responses import FileResponse
    from starlette.background import BackgroundTask
    want = set(wids.split(",")) if wids else None
    fd, tmp = tempfile.mkstemp(suffix=".tgz", dir=str(ROOT) if ROOT.exists() else None)
    os.close(fd)
    with tarfile.open(tmp, mode="w:gz", compresslevel=3) as tf:
        for e in list(WORLDS.values()):
            if want is not None and e.wid not in want:
                continue
            if want is None and run and e.spec.get("run") not in (None, run):
                continue
            if e.dir.exists():
                tf.add(e.dir, arcname=e.wid)
    return FileResponse(tmp, media_type="application/gzip", background=BackgroundTask(os.unlink, tmp))
