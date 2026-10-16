"""Reflection (SPEC §5.4, §8.3 handshake).

- `local_reflector(wid, seat, payload)` — the Reflector used in-process by core's `run_local(llm=True)`:
  SurrealDB write + retrieve (if configured) → Claude → returns
  {"proposed":[...], "llm_trace_id", "surreal_trace_id", "source":"claude", ...} or None (keep old memory).
- `reflect_worker(run)` — orchestrator loop for Daytona boxes: polls every box for awaiting_reflection
  worlds, GET reflect_payload → SurrealDB → Claude → PUT memory. Runs until every world of the run is done.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from blab.orchestrator import llm, memory

BUDGET_S = 55.0     # World wraps the reflector in wait_for(…, 60)


async def _surreal_pre(wid: str, seat: str, payload: dict) -> dict:
    if not memory.enabled():
        return {}
    try:
        return await asyncio.wait_for(memory.reflect_sync(wid, seat, payload), 12)
    except Exception as e:  # noqa: BLE001  (timeout or anything: Surreal is best-effort)
        return {"error": f"{type(e).__name__}: {e}"}


async def reflect_once(wid: str, seat: str, payload: dict) -> dict:
    """Full round trip; always returns a dict. `proposed is None` means: keep the old memory."""
    t0 = time.time()
    pre = await _surreal_pre(wid, seat, payload)
    if pre.get("hits"):
        payload = {**payload, "notes": {"surreal_retrieved": [h.get("summary") or h.get("text") or h.get("value")
                                                              for h in pre["hits"][:10]]}}
    out = await llm.reflect(payload, wid=wid, seat=seat)
    res = {"proposed": None, "llm_trace_id": None, "surreal_trace_id": pre.get("surreal_trace_id"),
           "source": "claude", "latency_ms": 0}
    if out is None:
        res.update(error="no LLM backend (set ANTHROPIC_API_KEY or OPENROUTER_API_KEY)", source="stub")
    else:
        res.update(proposed=out.get("proposed"), llm_trace_id=out.get("llm_trace_id"),
                   backend=out.get("backend"), dropped=out.get("dropped") or [])
        if out.get("arize_trace_id"):
            res["arize_trace_id"] = out["arize_trace_id"]
        if out.get("error"):
            res["error"] = out["error"]
    if pre.get("refs"):
        res["surreal_refs"] = pre["refs"]
    res["latency_ms"] = int((time.time() - t0) * 1000)
    return res


async def local_reflector(wid: str, seat: str, payload: dict) -> dict | None:
    """Reflector for core's World / run_local(llm=True). None → World keeps old memory and logs the error."""
    try:
        res = await asyncio.wait_for(reflect_once(wid, seat, payload), BUDGET_S)
    except asyncio.TimeoutError:
        print(f"[reflect] {wid}/{seat} timeout")
        return None
    except Exception as e:  # noqa: BLE001
        print(f"[reflect] {wid}/{seat} error {type(e).__name__}: {e}")
        return None
    if res.get("proposed") is None:
        print(f"[reflect] {wid}/{seat} g{payload.get('game')} no proposal: {res.get('error')}")
        return None
    return res


# ------------------------------------------------------------------ orchestrator worker (boxes)

async def _handle(client, box: dict, wid: str, log: list) -> None:
    from blab.orchestrator.boxes import box_get, box_put
    payload = await box_get(client, box, f"/worlds/{wid}/reflect_payload")
    if not payload or not payload.get("seat"):
        return
    seat = payload["seat"]
    res = await reflect_once(wid, seat, payload)
    body = {"proposed": res.get("proposed"),
            "reflect_meta": {k: v for k, v in res.items() if k != "proposed"},
            "memory_version": payload.get("memory_version")}
    put = await box_put(client, box, f"/worlds/{wid}/agents/{seat}/memory", body)
    log.append({"wid": wid, "seat": seat, "game": payload.get("game"), "n": len(res.get("proposed") or []),
                "error": res.get("error"), "latency_ms": res.get("latency_ms"), "put": put})
    # mirror the post-reflection belief state into SurrealDB (supersessions)
    if memory.enabled() and put:
        mem = await box_get(client, box, f"/worlds/{wid}/agents/{seat}/memory")
        if mem:
            try:
                await asyncio.wait_for(memory.after_reflection(wid, seat, mem.get("hypotheses") or [],
                                                               int(payload.get("game") or 0)), 15)
            except Exception:  # noqa: BLE001
                pass


async def reflect_worker(run: str = "r01", poll_s: float = 0.5, max_idle_s: float = 3600 * 3) -> list[dict]:
    from blab.orchestrator.boxes import box_client, box_get, load_boxes
    boxes = load_boxes(run)
    if not boxes:
        print("[reflect] no boxes in runs/boxes.json — run `blab provision` (or `blab run --local-only`, which "
              "reflects in-process)")
        return []
    inflight: dict[str, asyncio.Task] = {}
    log: list[dict] = []
    t_start = time.time()
    clients = {b["box_id"]: box_client(b) for b in boxes}
    seen_any = False
    last_check, last_print = 0.0, 0.0
    try:
        while time.time() - t_start < max_idle_s:
            for b in boxes:
                c = clients[b["box_id"]]
                worlds = await box_get(c, b, "/worlds", status="awaiting_reflection") or []
                for w in worlds:
                    if run and w.get("run") not in (None, run):
                        continue
                    key = f"{b['box_id']}:{w['wid']}"
                    if key not in inflight or inflight[key].done():
                        inflight[key] = asyncio.create_task(_handle(c, b, w["wid"], log))
            for k in [k for k, t in inflight.items() if t.done()]:
                t = inflight.pop(k)
                if t.exception():
                    print(f"[reflect] {k} failed: {t.exception()!r}")
            if time.time() - last_check > 5:
                last_check = time.time()
                active = 0
                for b in boxes:
                    h = await box_get(clients[b["box_id"]], b, "/health") or {}
                    cnt = h.get("worlds") or {}
                    active += sum(cnt.get(k, 0) for k in ("running", "waiting", "created", "paused"))
                    seen_any = seen_any or sum(cnt.values()) > 0
                if time.time() - last_print > 60:
                    last_print = time.time()
                    print(f"[reflect] t+{time.time() - t_start:.0f}s active worlds={active} "
                          f"reflections={len(log)} inflight={len(inflight)}", flush=True)
                if seen_any and active == 0 and not inflight:
                    break
            await asyncio.sleep(poll_s)
    finally:
        for c in clients.values():
            await c.aclose()
    out = Path("runs") / run / "reflect_log.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log, indent=1, default=str))
    print(f"[reflect] {len(log)} reflections handled; log → {out}")
    return log
