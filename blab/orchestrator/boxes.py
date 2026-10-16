"""Daytona world boxes (SPEC §2, §8, §9, §10.1): provision · run_matrix · collect · ablate · teardown.

runs/boxes.json (gitignored) = {"blab_key", "boxes":[{box_id, sandbox_id, url, token, cpu}]}.
`run_matrix(..., local_only=True)` is the no-Daytona fallback: every world runs in-process via core's
`run_local` in ProcessPoolExecutors (LLM worlds ≤ 8 at a time), writing runs/<run>/<wid>/.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from blab.env import load_env

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
BOXES_JSON = RUNS / "boxes.json"
LABELS = {"project": "blufflab", "role": "world"}
APP_DIR = "/home/daytona/app"
LLM_AGENTS = {"profiler", "profiler_blind", "llm_episodic"}
BOX_PIP = ["anthropic[bedrock]", "boto3", "fastapi", "uvicorn", "numpy", "pydantic>=2", "httpx", "pyyaml", "typer", "anthropic",
           "arize-otel", "openinference-instrumentation", "openinference-instrumentation-anthropic"]
BOX_ENV_KEYS = ("ARIZE_SPACE_ID", "ARIZE_API_KEY", "ARIZE_PROJECT_NAME",
                # in-box Claude (user preference: Bedrock in boxes; Anthropic direct as fallback)
                "AWS_DEFAULT_REGION", "AWS_OPUS_API_KEY", "AWS_SONNET_API_KEY",
                "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                "ANTHROPIC_API_KEY", "BLAB_LLM_PROVIDER")


# ------------------------------------------------------------------ box registry / HTTP helpers

def load_boxes(run: str | None = None) -> list[dict]:
    if not BOXES_JSON.exists():
        return []
    d = json.loads(BOXES_JSON.read_text())
    key = d.get("blab_key")
    return [{**b, "blab_key": key} for b in d.get("boxes", [])]


def _save_boxes(boxes: list[dict], key: str) -> None:
    RUNS.mkdir(exist_ok=True)
    BOXES_JSON.write_text(json.dumps({"blab_key": key, "boxes": [{k: v for k, v in b.items() if k != "blab_key"}
                                                                  for b in boxes]}, indent=1))


def _headers(b: dict) -> dict:
    h = {}
    if b.get("token"):
        h["x-daytona-preview-token"] = b["token"]
    if b.get("blab_key"):
        h["X-Blab-Key"] = b["blab_key"]
    return h


def box_client(b: dict, timeout: float = 30.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=b["url"].rstrip("/"), headers=_headers(b), timeout=timeout)


async def box_get(c: httpx.AsyncClient, b: dict, path: str, **params):
    try:
        r = await c.get(path, params=params or None)
        return r.json() if r.status_code == 200 else None
    except Exception:  # noqa: BLE001
        return None


async def box_put(c: httpx.AsyncClient, b: dict, path: str, body: dict):
    try:
        r = await c.put(path, json=body)
        return r.json() if r.status_code == 200 else {"status": r.status_code, "text": r.text[:200]}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


async def box_post(c: httpx.AsyncClient, b: dict, path: str, body: dict | None = None):
    try:
        r = await c.post(path, json=body or {})
        return r.json() if r.status_code == 200 else {"status": r.status_code, "text": r.text[:300]}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# ------------------------------------------------------------------ provision

KILL_WORLDD = """
import os, signal, socket, time
me = os.getpid()
for p in os.listdir('/proc'):
    if not p.isdigit() or int(p) == me:
        continue
    try:
        c = open(f'/proc/{p}/cmdline', 'rb').read().replace(bytes(1), b' ')
    except Exception:
        continue
    if (b'uvicorn' in c and b'blab.worldd' in c) or b'multiprocessing' in c:
        try:
            os.kill(int(p), signal.SIGKILL)
        except Exception:
            pass
for _ in range(40):
    s = socket.socket()
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', 8000)); s.close(); break
    except OSError:
        s.close(); time.sleep(0.25)
"""


def _start_worldd(sb, box_id: str) -> None:
    from daytona import SessionExecuteRequest
    from blab.orchestrator.spike import repo_tarball
    sb.fs.upload_file(repo_tarball(), "/home/daytona/app.tgz")
    sb.process.exec(f"rm -rf {APP_DIR} && mkdir -p {APP_DIR} /home/daytona/worlds && "
                    f"tar xzf /home/daytona/app.tgz -C {APP_DIR}", timeout=120)
    killer = KILL_WORLDD
    try:
        sb.fs.upload_file(killer.encode(), "/home/daytona/kill_worldd.py")
        sb.process.exec("python3 /home/daytona/kill_worldd.py", timeout=30)
    except Exception:  # noqa: BLE001
        pass
    sess = f"worldd-{int(time.time())}"
    sb.process.create_session(sess)
    sb.process.execute_session_command(sess, SessionExecuteRequest(
        command=(f"cd {APP_DIR} && BLAB_BOX_ID={box_id} python3 -m uvicorn blab.worldd:app --host 0.0.0.0 "
                 f"--port 8000 > /home/daytona/worldd.log 2>&1"), run_async=True))


def _wait_health(b: dict, tries: int = 60, after: float = 0.0) -> dict | None:
    for _ in range(tries):
        try:
            r = httpx.get(b["url"].rstrip("/") + "/health", headers=_headers(b), timeout=10)
            if r.status_code == 200 and r.json().get("started", after + 1) > after:
                return r.json()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    return None


def _create_box(i: int, cpu: int, mem: int, key: str) -> dict:
    from daytona import (CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams, Daytona, Image,
                         Resources)
    d = Daytona()
    box_id = f"box{i + 1}"
    env = {k: os.environ[k] for k in BOX_ENV_KEYS if os.environ.get(k)}
    env.update(BLAB_KEY=key, BLAB_BOX_ID=box_id)
    labels = {**LABELS, "box": box_id}
    t0 = time.time()
    try:
        img = Image.debian_slim("3.12").pip_install(BOX_PIP).workdir("/home/daytona")
        sb = d.create(CreateSandboxFromImageParams(image=img, resources=Resources(cpu=cpu, memory=mem, disk=8),
                                                   auto_stop_interval=0, labels=labels, env_vars=env),
                      timeout=0, on_snapshot_create_logs=lambda s: None)
        how = "image"
    except Exception as e:  # noqa: BLE001
        print(f"[provision] {box_id}: image create failed ({type(e).__name__}: {str(e)[:120]}); default snapshot")
        sb = d.create(CreateSandboxFromSnapshotParams(language="python", auto_stop_interval=0, labels=labels,
                                                      env_vars=env), timeout=180)
        sb.process.exec("pip install -q " + " ".join(f"'{p}'" for p in BOX_PIP), timeout=600)
        how = "snapshot"
    _start_worldd(sb, box_id)
    link = sb.get_preview_link(8000)
    b = {"box_id": box_id, "sandbox_id": sb.id, "url": link.url, "token": link.token, "cpu": cpu, "how": how,
         "blab_key": key}
    h = _wait_health(b)
    b["health"] = h
    b["cpu"] = (h or {}).get("cpu_quota", cpu)
    print(f"[provision] {box_id} sandbox={sb.id} via {how} in {time.time() - t0:.0f}s health={h}")
    return b


def provision(boxes: int = 2, cpu: int = 4, mem: int = 8) -> list[dict]:
    load_env()
    if not os.environ.get("DAYTONA_API_KEY"):
        print("SKIP provision: missing DAYTONA_API_KEY")
        return []
    if boxes * mem > 8:          # Tier-1 org limit: 10 GiB RAM total; keep ~2 GiB for forge boxes
        mem = max(1, 8 // boxes)
        print(f"[provision] clamping memory to {mem} GiB/box (org limit 10 GiB total)")
    old = load_boxes()
    key = os.environ.get("BLAB_KEY") or (old[0].get("blab_key") if old else None) or secrets.token_hex(16)
    start = len(old)
    out = list(old)
    with ThreadPoolExecutor(max_workers=boxes) as ex:
        futs = [ex.submit(_create_box, start + i, cpu, mem, key) for i in range(boxes)]
        for f in as_completed(futs):
            try:
                out.append(f.result())
            except Exception as e:  # noqa: BLE001
                print(f"[provision] box failed: {type(e).__name__}: {e}")
    _save_boxes(out, key)
    print(f"[provision] {len(out)} boxes → {BOXES_JSON}")
    return out


def redeploy() -> None:
    """Re-upload the repo tarball to every box and restart worldd (after core changes code). Kills running worlds."""
    load_env()
    from daytona import Daytona
    d = Daytona()
    for b in load_boxes():
        sb = d.get(b["sandbox_id"])
        t = time.time() - 30          # clock skew margin; stale servers report a much older start
        _start_worldd(sb, b["box_id"])
        print(b["box_id"], _wait_health(b, tries=45, after=t))


# ------------------------------------------------------------------ matrix

def _load_matrix(path: str) -> dict:
    import yaml
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return yaml.safe_load(p.read_text())


def expand_matrix(m: dict, run: str) -> list[dict]:
    """matrix.yaml (§6) → world specs. Seeds 1..N shared across agents (paired by seed).
    Caps: env BLAB_SEEDS_LLM / BLAB_SEEDS_NON_LLM, or matrix `limits: {llm: n, non_llm: n}`."""
    games = int(m.get("games", 20))
    reflect_every = int(m.get("reflect_every", 2))
    tables = m.get("tables") or {"telltale": ["control", "telltale"]}
    agents = list(m.get("agents") or ["profiler"])
    seeds = m.get("seeds") or {}
    lim = m.get("limits") or {}
    n_llm = int(os.environ.get("BLAB_SEEDS_LLM") or lim.get("llm") or seeds.get("llm", 30))
    n_non = int(os.environ.get("BLAB_SEEDS_NON_LLM") or lim.get("non_llm") or seeds.get("non_llm", 300))
    extra = m.get("llm_agents_extra") or {}
    specs = []
    i = 0

    try:
        from blab.world import TABLES, make_seats
    except Exception:  # noqa: BLE001
        TABLES, make_seats = {}, None

    def add(agent, table, seed):
        nonlocal i
        i += 1
        if make_seats is not None and table in TABLES:
            seats = make_seats(table, agent, seed)          # core's canonical seating (G0 etc.)
        else:
            roles = tables[table]
            seats = [{"seat": "p1", "policy": agent},
                     {"seat": "p2", "policy": "scripted", "role": roles[0]},
                     {"seat": "p3", "policy": "scripted", "role": roles[1]}]
        specs.append({"wid": f"w{i:03d}", "run": run, "agent": agent, "table": table, "seed": seed,
                      "games": games, "reflect_every": reflect_every, "ask": bool(m.get("ask", False)),
                      "llm": agent in LLM_AGENTS, "seats": seats})

    for table in tables:
        for agent in agents:
            n = n_llm if agent in LLM_AGENTS else n_non
            for s in range(1, n + 1):
                add(agent, table, s)
        for agent, n in extra.items():
            for s in range(1, min(int(n), n_llm) + 1):
                add(agent, table, s)
    return specs


def _run_local_one(spec: dict, out: str, use_llm: bool) -> dict:
    import asyncio as _a
    import inspect as _i
    from blab.env import load_env as _le
    _le()
    from blab.world import run_local
    r = run_local(table=spec["table"], agent=spec["agent"], seed=spec["seed"], games=spec["games"],
                  reflect_every=spec["reflect_every"], out=out, llm=use_llm, ask=spec.get("ask", False),
                  wid=spec["wid"], quiet=True)
    if _i.iscoroutine(r):
        r = _a.run(r)
    return {"wid": spec["wid"], "result": r}


def _run_specs_local(specs: list[dict], out_dir: Path, *, llm_ok: bool) -> list[dict]:
    llm_specs = [s for s in specs if s["llm"]]
    cpu_specs = [s for s in specs if not s["llm"]]
    done: list[dict] = []
    t0 = time.time()
    n_cpu = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=n_cpu) as cpu_pool, ProcessPoolExecutor(max_workers=8) as llm_pool:
        futs = {cpu_pool.submit(_run_local_one, s, str(out_dir), False): s for s in cpu_specs}
        futs.update({llm_pool.submit(_run_local_one, s, str(out_dir), llm_ok): s for s in llm_specs})
        for k, f in enumerate(as_completed(futs), 1):
            s = futs[f]
            try:
                done.append(f.result())
            except Exception as e:  # noqa: BLE001
                done.append({"wid": s["wid"], "error": f"{type(e).__name__}: {e}"})
            if k % 25 == 0 or k == len(futs):
                print(f"[run] {k}/{len(futs)} worlds done ({time.time() - t0:.0f}s)")
    return done


def _assign(specs: list[dict], boxes: list[dict]) -> dict[str, list[dict]]:
    """Greedy bin-pack: LLM world ≈ 0.05 vCPU, non-LLM ≈ 1 vCPU, onto the box with least load/cpu."""
    load = {b["box_id"]: 0.0 for b in boxes}
    cpu = {b["box_id"]: float(b.get("cpu") or 4) for b in boxes}
    out: dict[str, list[dict]] = {b["box_id"]: [] for b in boxes}
    for s in sorted(specs, key=lambda s: s["llm"]):
        bid = min(load, key=lambda k: load[k] / cpu[k])
        load[bid] += 0.05 if s["llm"] else 1.0
        out[bid].append(s)
    return out


async def _schedule(specs: list[dict], boxes: list[dict]) -> dict:
    plan = _assign(specs, boxes)
    placed = {}
    for b in boxes:
        async with box_client(b, timeout=60) as c:
            for s in plan[b["box_id"]]:
                body = {k: s[k] for k in ("wid", "run", "seats", "seed", "games", "reflect_every", "ask", "table")}
                body["reflector"] = "orchestrator" if s["llm"] else "none"
                body["start"] = True
                r = await box_post(c, b, "/worlds", body)
                placed[s["wid"]] = {"box": b["box_id"], **({"error": r} if "wid" not in (r or {}) else {})}
    return placed


def run_matrix(matrix: str = "matrix.yaml", run: str = "r01", local_only: bool = False) -> dict:
    load_env()
    from blab.orchestrator import llm
    m = _load_matrix(matrix)
    specs = expand_matrix(m, run)
    run_dir = RUNS / run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "worlds.json").write_text(json.dumps(specs, indent=1))
    n_llm = sum(s["llm"] for s in specs)
    print(f"[run] {run}: {len(specs)} worlds ({n_llm} LLM) from {matrix}; llm backend={llm.backend()}")
    boxes = [] if local_only else load_boxes()
    if not boxes:
        if not local_only:
            print("[run] no boxes provisioned → running locally (fallback)")
        res = _run_specs_local(specs, run_dir, llm_ok=llm.available())
        (run_dir / "local_results.json").write_text(json.dumps(res, indent=1, default=str))
        bad = [r for r in res if r.get("error")]
        print(f"[run] local done: {len(res) - len(bad)} ok, {len(bad)} errors")
        return {"worlds": len(specs), "errors": len(bad), "mode": "local"}
    placed = asyncio.run(_schedule(specs, boxes))
    (run_dir / "placement.json").write_text(json.dumps(placed, indent=1))
    errs = {w: p for w, p in placed.items() if p.get("error")}
    print(f"[run] scheduled {len(placed) - len(errs)} worlds on {len(boxes)} boxes ({len(errs)} errors)")
    if errs:
        print("[run] first error:", next(iter(errs.items())))
    if os.environ.get("BLAB_NO_WAIT"):
        return {"worlds": len(specs), "scheduled": len(placed) - len(errs)}
    from blab.orchestrator.reflect import reflect_worker
    asyncio.run(reflect_worker(run))
    collect(run)
    return {"worlds": len(specs), "scheduled": len(placed) - len(errs), "mode": "boxes"}


# ------------------------------------------------------------------ collect

async def _collect_box(b: dict, run: str) -> int:
    import io
    import tarfile
    async with box_client(b, timeout=600) as c:
        try:
            r = await c.get("/export", params={"run": run})
            if r.status_code == 200:
                dst = RUNS / run
                dst.mkdir(parents=True, exist_ok=True)
                with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
                    names = {m.name.split("/")[0] for m in tf.getmembers()}
                    tf.extractall(dst, filter="data")
                return len(names)
            print(f"[collect] {b['box_id']}: /export → {r.status_code}; falling back to per-file")
        except Exception as e:  # noqa: BLE001
            print(f"[collect] {b['box_id']}: /export failed ({type(e).__name__}: {e}); per-file fallback")
        n = 0
        for w in await box_get(c, b, "/worlds") or []:
            if w.get("run") not in (None, run):
                continue
            wid = w["wid"]
            for f in await box_get(c, b, f"/worlds/{wid}/files") or []:
                try:
                    r = await c.get(f"/worlds/{wid}/file", params={"path": f})
                    if r.status_code == 200:
                        p = RUNS / run / wid / f
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(r.content)
                except Exception:  # noqa: BLE001
                    pass
            n += 1
        return n


async def _collect(run: str) -> int:
    res = await asyncio.gather(*[_collect_box(b, run) for b in load_boxes()], return_exceptions=True)
    for b, r in zip(load_boxes(), res):
        print(f"[collect] {b['box_id']}: {r}")
    return sum(r for r in res if isinstance(r, int))


def collect(run: str = "r01") -> int:
    load_env()
    if not load_boxes():
        print(f"[collect] no boxes; local runs are already in runs/{run}/")
        return 0
    n = asyncio.run(_collect(run))
    print(f"[collect] pulled {n} worlds → runs/{run}/")
    return n


# ------------------------------------------------------------------ ablate

def parse_variant(v: str, m: dict) -> dict:
    if not v.startswith(("transplant:", "false_memory")):
        try:
            from blab.replay import parse_variant as core_parse
            return core_parse(v)
        except Exception:  # noqa: BLE001
            pass
    if v == "full":
        return {"type": "full"}
    if v.startswith("as_of:"):
        return {"type": "as_of", "game": int(v.split(":", 1)[1])}
    if v.startswith("minus:"):
        return {"type": "minus", "aspect": v.split(":", 1)[1]}
    if v == "false_memory":
        return {"type": "false_memory", **(m.get("false_memory_text") or
                                           {"about": "p3", "aspect": "baseline_bluff", "text": "Player 3 never bluffs",
                                            "q": 0.0})}
    if v.startswith("transplant:"):
        return {"type": "transplant", "which": v.split(":", 1)[1]}
    raise ValueError(f"unknown variant {v}")


def _slug(v: str) -> str:
    return v.replace(":", "").replace("_", "")


def _transplant_source(spec: dict, specs: list[dict], which: str) -> dict | None:
    """same_role: same agent+table, next seed. other_role: same agent+seed, a different table."""
    if which == "same_role":
        c = [s for s in specs if s["agent"] == spec["agent"] and s["table"] == spec["table"]
             and s["seed"] != spec["seed"]]
        c.sort(key=lambda s: (s["seed"] - spec["seed"]) % 10**6)
    else:
        c = [s for s in specs if s["agent"] == spec["agent"] and s["table"] != spec["table"]
             and s["seed"] == spec["seed"]]
    return c[0] if c else None


def _ckpt_memory(world_dir: Path, game: int, seat: str = "p1") -> dict | None:
    from blab.worldd import _ckpt_memory as cm
    try:
        return cm(json.loads((world_dir / "ckpt" / f"g{game:02d}.json").read_text()), seat)
    except Exception:  # noqa: BLE001
        return None


def _branch_local_one(world_dir: str, new_wid: str, from_game: int, games: int, variant: dict) -> dict:
    import asyncio as _a
    import inspect as _i
    from blab.world import branch
    w = branch(Path(world_dir), new_wid, from_game, games, variant, learning=False)
    if _i.iscoroutine(w):
        w = _a.run(w)
    return {"wid": new_wid, "result": _a.run(w.run())}


def ablate(run: str = "r01", matrix: str = "matrix.yaml") -> dict:
    load_env()
    m = _load_matrix(matrix)
    ab = m.get("ablate") or {}
    run_dir = RUNS / run
    specs = json.loads((run_dir / "worlds.json").read_text())
    at = int(ab.get("at_game", 10))
    agents = set(ab.get("agents") or ["profiler"])
    variants = ab.get("variants") or ["full", "as_of:0"]
    targets = [s for s in specs if s["agent"] in agents]
    jobs = []
    for s in targets:
        for v in variants:
            var = parse_variant(v, m)
            if var["type"] == "transplant":
                src = _transplant_source(s, specs, var["which"])
                if src is None:
                    continue
                mem = _ckpt_memory(run_dir / src["wid"], at)
                var = {"type": "transplant", "from_world": src["wid"], "from_seat": "p1",
                       "about_map": {"p2": "p2", "p3": "p3"}, "which": var["which"],
                       **({"memory": mem} if mem else {})}
            jobs.append((s, v, var))
    print(f"[ablate] {len(jobs)} branch worlds ({len(targets)} parents × {len(variants)} variants) at g{at}")
    branches = []
    boxes = load_boxes()
    placement = json.loads((run_dir / "placement.json").read_text()) if (run_dir / "placement.json").exists() else {}
    if boxes and placement:
        async def go():
            byid = {b["box_id"]: b for b in boxes}
            for s, v, var in jobs:
                b = byid.get(placement.get(s["wid"], {}).get("box"))
                if b is None:
                    continue
                async with box_client(b, timeout=60) as c:
                    if var.get("type") == "transplant" and "memory" not in var:
                        sb_ = byid.get(placement.get(var["from_world"], {}).get("box"))
                        if sb_ is not None:
                            async with box_client(sb_) as c2:
                                mem = await box_get(c2, sb_, f"/worlds/{var['from_world']}/agents/p1/memory",
                                                    as_of_game=at)
                            if mem:
                                var = {**var, "memory": mem}
                    nw = f"{s['wid']}_{_slug(v)}"
                    r = await box_post(c, b, f"/worlds/{s['wid']}/branch",
                                       {"new_wid": nw, "from_game": at, "games": s["games"], "learning": False,
                                        "memory_variant": var, "run": run, "start": True})
                    branches.append({"wid": nw, "parent": s["wid"], "variant": v, "box": b["box_id"], "resp": r})
        asyncio.run(go())
        (run_dir / "ablate.json").write_text(json.dumps(branches, indent=1, default=str))
        print(f"[ablate] scheduled {len(branches)} on boxes; `blab collect --run {run}` when done")
        return {"branches": len(branches), "mode": "boxes"}
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        futs = {}
        for s, v, var in jobs:
            nw = f"{s['wid']}_{_slug(v)}"
            futs[ex.submit(_branch_local_one, str(run_dir / s["wid"]), nw, at, s["games"], var)] = (s, v, nw)
        for f in as_completed(futs):
            s, v, nw = futs[f]
            try:
                f.result()
                branches.append({"wid": nw, "parent": s["wid"], "variant": v, "box": "local"})
            except Exception as e:  # noqa: BLE001
                branches.append({"wid": nw, "parent": s["wid"], "variant": v, "error": f"{type(e).__name__}: {e}"})
    (run_dir / "ablate.json").write_text(json.dumps(branches, indent=1, default=str))
    bad = sum(1 for b in branches if b.get("error"))
    print(f"[ablate] local: {len(branches) - bad} ok, {bad} errors → {run_dir}/ablate.json")
    return {"branches": len(branches), "errors": bad, "mode": "local"}


# ------------------------------------------------------------------ teardown

def teardown(run: str = "r01", yes: bool = False) -> int:
    load_env()
    if not yes:
        print("Refusing: pass --yes to delete every sandbox labelled project=blufflab (NOT before the awards).")
        return 0
    from daytona import Daytona, ListSandboxesQuery
    d = Daytona()
    n = 0
    for sb in d.list(ListSandboxesQuery(labels={"project": "blufflab"})):
        if (getattr(sb, "labels", None) or {}).get("project") != "blufflab":
            continue
        try:
            d.delete(sb)
            n += 1
        except Exception as e:  # noqa: BLE001
            print(f"[teardown] {sb.id}: {e}")
    if BOXES_JSON.exists():
        BOXES_JSON.rename(BOXES_JSON.with_suffix(f".deleted.{int(time.time())}.json"))
    print(f"[teardown] deleted {n} sandboxes")
    return n
