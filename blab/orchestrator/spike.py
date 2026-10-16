"""`blab spike` — SPEC §15 checks. Prints a PASS/FAIL/SKIP table and writes runs/spike.json.

Every check degrades to SKIP (with the missing env var names) when its keys are absent.
Spike sandboxes are labelled {project:"blufflab", role:"spike"} and deleted afterwards.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import tarfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from blab.env import load_env

ROOT = Path(__file__).resolve().parents[2]
SPIKE_LABELS = {"project": "blufflab", "role": "spike"}

TRIVIAL_APP = '''
from fastapi import FastAPI
app = FastAPI()
@app.get("/health")
def health():
    return {"ok": True, "box_id": "spike"}
'''

EXCLUDE_DIRS = {".git", ".venv", "runs", "__pycache__", ".pytest_cache", "node_modules", "show"}
EXCLUDE_FILES = {"settings.env"}


def repo_tarball(include_show: bool = False) -> bytes:
    """Tar.gz of the repo for upload. NEVER includes settings.env / *.env / runs / .venv."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted(ROOT.rglob("*")):
            rel = p.relative_to(ROOT)
            parts = set(rel.parts)
            excl = EXCLUDE_DIRS - ({"show"} if include_show else set())
            if parts & excl:
                continue
            if p.is_file():
                if p.name in EXCLUDE_FILES or p.suffix == ".env" or p.suffix == ".pyc":
                    continue
                tf.add(p, arcname=str(rel))
    return buf.getvalue()


def _row(check, status, detail=""):
    return {"check": check, "status": status, "detail": str(detail)[:300]}


def _missing(*keys):
    return [k for k in keys if not os.environ.get(k)]


# ------------------------------------------------------------------ Daytona

def _daytona():
    from daytona import Daytona
    return Daytona()


def _curl(sb, url: str, timeout: int = 12) -> tuple[bool, str]:
    cmd = (f"curl -sS -o /dev/null -m {timeout} -w '%{{http_code}}' {url} "
           f"|| python3 -c \"import urllib.request,sys;"
           f"print(urllib.request.urlopen('{url}',timeout={timeout}).status)\"")
    try:
        r = sb.process.exec(cmd, timeout=timeout * 2 + 10)
        out = (r.result or "").strip()
        code = out.splitlines()[-1] if out else ""
        ok = r.exit_code == 0 and code[:1] in "12345" and code != "000"
        return ok, f"exit={r.exit_code} out={out[-120:]}"
    except Exception as e:  # noqa: BLE001
        return False, f"exc {type(e).__name__}: {e}"


def check_d1_d2() -> list[dict]:
    rows = []
    if _missing("DAYTONA_API_KEY"):
        return [_row("D1", "SKIP", "missing DAYTONA_API_KEY"), _row("D2", "SKIP", "missing DAYTONA_API_KEY")]
    from daytona import CreateSandboxFromSnapshotParams, SessionExecuteRequest
    import httpx
    d = _daytona()
    sb = None
    t0 = time.time()
    try:
        aws = {k: os.environ[k] for k in ("AWS_DEFAULT_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                                         "AWS_SESSION_TOKEN") if os.environ.get(k)}
        sb = d.create(CreateSandboxFromSnapshotParams(
            language="python", auto_stop_interval=0, labels=SPIKE_LABELS, env_vars=aws), timeout=180)
        t_create = time.time() - t0
        sb.fs.upload_file(repo_tarball(), "/home/daytona/app.tgz")
        sb.fs.upload_file(TRIVIAL_APP.encode(), "/home/daytona/spike_app.py")
        r = sb.process.exec("mkdir -p /home/daytona/app && tar xzf /home/daytona/app.tgz -C /home/daytona/app "
                            "&& ls /home/daytona/app | head -20", timeout=60)
        ok_tar = r.exit_code == 0 and "blab" in (r.result or "")
        r = sb.process.exec("python3 -c 'import fastapi,uvicorn,numpy,pydantic,httpx' 2>&1 || "
                            "pip install -q fastapi uvicorn numpy pydantic httpx pyyaml 2>&1 | tail -2",
                            timeout=240)
        sb.process.create_session("spike")
        sb.process.execute_session_command("spike", SessionExecuteRequest(
            command="cd /home/daytona && python3 -m uvicorn spike_app:app --host 0.0.0.0 --port 8000",
            run_async=True))
        link = sb.get_preview_link(8000)
        status, body = None, ""
        for _ in range(30):
            try:
                resp = httpx.get(link.url.rstrip("/") + "/health",
                                 headers={"x-daytona-preview-token": link.token or ""}, timeout=10)
                status, body = resp.status_code, resp.text[:80]
                if status == 200:
                    break
            except Exception as e:  # noqa: BLE001
                body = str(e)[:80]
            time.sleep(2)
        rows.append(_row("D1", "PASS" if status == 200 and ok_tar else "FAIL",
                         f"sandbox={sb.id} create={t_create:.0f}s tar={'ok' if ok_tar else 'fail'} "
                         f"GET /health via preview → {status} {body}"))
        # D2: egress from inside the box
        region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
        targets = [f"https://bedrock-runtime.{region}.amazonaws.com", "https://api.anthropic.com", "https://openrouter.ai/api/v1/models",
                   "https://otlp.arize.com", "https://app.arize.com", "https://api.elevenlabs.io",
                   "https://example.com", "https://pypi.org/simple/", "https://github.com"]
        from blab.orchestrator.memory import conf as _sconf
        sc = _sconf()
        surreal = sc[0].split("/api/v1/")[0] if sc else None
        targets.append(surreal or "https://srv1.spectron.aws-usw2.surreal.cloud")
        res = {}
        for u in targets:
            ok, det = _curl(sb, u)
            res[u] = "ok" if ok else "blocked/fail"
        anth = res["https://api.anthropic.com"] == "ok"
        arize = res["https://otlp.arize.com"] == "ok"
        detail = "; ".join(f"{u.split('//')[-1].split('/')[0]}={v}" for u, v in res.items())
        if not surreal:
            detail += " (surreal host is a guess; SPECTRON_URL unset)"
        else:
            detail = detail.replace(surreal.split("//")[-1], "SURREAL_HOST")
        rows.append(_row("D2", "PASS" if anth and arize else "FAIL", detail))
        rows[-1]["egress"] = res
        rows.append(_bedrock_in_box(sb))
    except Exception as e:  # noqa: BLE001
        rows.append(_row("D1", "FAIL", f"{type(e).__name__}: {e}"))
        rows.append(_row("D2", "SKIP", "D1 failed"))
        traceback.print_exc()
    finally:
        if sb is not None:
            try:
                d.delete(sb)
            except Exception:  # noqa: BLE001
                pass
    return rows


BEDROCK_PROBE = '''
import os, anthropic
c = anthropic.AnthropicBedrock(aws_region=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
r = c.messages.create(model=os.environ.get("BLAB_BEDROCK_MODEL", "us.anthropic.claude-opus-5-5"), max_tokens=64,
                      messages=[{"role": "user", "content": "Reply with exactly: OK"}])
print("BEDROCK", r.stop_reason, "".join(b.text for b in r.content if b.type == "text")[:20])
'''


def _bedrock_in_box(sb) -> dict:
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        return _row("D2b", "SKIP", "missing AWS_ACCESS_KEY_ID")
    try:
        sb.fs.upload_file(BEDROCK_PROBE.encode(), "/home/daytona/bedrock_probe.py")
        sb.process.exec("pip install -q 'anthropic[bedrock]' boto3 2>&1 | tail -1", timeout=300)
        r = sb.process.exec("python3 /home/daytona/bedrock_probe.py 2>&1 | tail -3", timeout=120)
        out = (r.result or "").strip()
        return _row("D2b", "PASS" if "BEDROCK" in out and "OK" in out else "FAIL",
                    f"in-box Bedrock call: {out[-200:]}")
    except Exception as e:  # noqa: BLE001
        return _row("D2b", "FAIL", f"{type(e).__name__}: {e}")


def check_d3() -> list[dict]:
    if _missing("DAYTONA_API_KEY"):
        return [_row("D3", "SKIP", "missing DAYTONA_API_KEY")]
    from daytona import CreateSandboxFromSnapshotParams
    d = _daytona()
    sb = None
    try:
        sb = d.create(CreateSandboxFromSnapshotParams(
            language="python", network_block_all=True, ephemeral=True, auto_stop_interval=10,
            labels={**SPIKE_LABELS, "kind": "forge"}), timeout=180)
        ok_a, det_a = _curl(sb, "https://api.anthropic.com", timeout=6)
        ok_e, det_e = _curl(sb, "https://example.com", timeout=6)
        blocked = not ok_a and not ok_e
        return [_row("D3", "PASS" if blocked else "FAIL",
                     f"sandbox={sb.id} network_block_all=True: anthropic={'reachable' if ok_a else 'blocked'} "
                     f"example.com={'reachable' if ok_e else 'blocked'}")]
    except Exception as e:  # noqa: BLE001
        return [_row("D3", "FAIL", f"{type(e).__name__}: {e}")]
    finally:
        if sb is not None:
            try:
                d.delete(sb)
            except Exception:  # noqa: BLE001
                pass


# ------------------------------------------------------------------ Arize / LLM / Surreal / Eleven

def check_a1() -> list[dict]:
    miss = _missing("ARIZE_SPACE_ID", "ARIZE_API_KEY")
    if miss:
        return [_row("A1", "SKIP", f"missing {', '.join(miss)}"), _row("A2", "SKIP", f"missing {', '.join(miss)}")]
    try:
        from blab.orchestrator import tracing
        tid = tracing.spike_span()
        return [_row("A1", "PASS" if tid else "FAIL", f"laptop span sent trace_id={tid} (check AX UI; box span not tested)"),
                _row("A2", "SKIP", "experiments not run in spike")]
    except Exception as e:  # noqa: BLE001
        return [_row("A1", "FAIL", f"{type(e).__name__}: {e}"), _row("A2", "SKIP", "")]


CANNED_PAYLOAD = {
    "seat": "p1", "memory_version": 3, "game": 4,
    "new_obs": [
        {"obs_id": f"spike:g{g}r{r}", "game": g, "round": r, "by": "p3", "challenger": "p1", "count": c,
         "was_bluff": b, "utterance": u, "cards": [],
         "features": {"hand_size": hs, "claim_count": c, "formal_register": fr, "trusted_streak": 1,
                      "lives_self": 3, "lives_target": 2, "game_index": g, "round_index": r,
                      "was_challenged_last_round": False, "table_rank": "K", "utterance": u},
         "p_cards": 0.3}
        for (g, r, c, b, u, hs, fr) in [
            (1, 1, 2, True, "I have placed two Kings.", 3, True),
            (1, 2, 1, False, "That's a King.", 4, False),
            (2, 1, 2, True, "I am placing two Kings.", 2, True),
            (2, 3, 1, False, "One King, don't worry.", 3, False),
            (3, 2, 2, True, "I have placed two Kings on the table.", 2, True),
            (4, 1, 1, False, "It's a King.", 4, False),
        ]],
    "hypotheses": [],
}


def check_c1() -> list[dict]:
    """C1 on every configured provider (bedrock / anthropic / openrouter)."""
    from blab.orchestrator import llm
    if not llm.available():
        return [_row("C1", "SKIP", "missing AWS_* / ANTHROPIC_API_KEY / OPENROUTER_API_KEY")]

    async def one(p):
        t0 = time.time()
        try:
            out = await llm.reflect(CANNED_PAYLOAD, wid="spike", only=p)
        except Exception as e:  # noqa: BLE001
            return _row(f"C1.{p}", "FAIL", f"{type(e).__name__}: {e}")
        if out and out.get("proposed"):
            h = out["proposed"]
            return _row(f"C1.{p}", "PASS", f"{len(h)} hyps in {time.time()-t0:.1f}s; first: {h[0].get('aspect')} "
                        f"{json.dumps(h[0].get('predicate'))}")
        return _row(f"C1.{p}", "FAIL", f"{(out or {}).get('error')!r}")

    async def go():
        return await asyncio.gather(*[one(p) for p in llm.providers()])
    return list(asyncio.run(go()))


def check_surreal() -> list[dict]:
    miss = _missing("SPECTRON_URL", "SPECTRON_CONTEXT_ID", "SPECTRON_API_KEY")
    if miss:
        return [_row(c, "SKIP", f"missing {', '.join(miss)}") for c in ("S1", "S2", "S3")]
    try:
        from blab.orchestrator import memory
        return memory.spike_checks()
    except Exception as e:  # noqa: BLE001
        return [_row("S1", "FAIL", f"{type(e).__name__}: {e}"), _row("S2", "SKIP", ""), _row("S3", "SKIP", "")]


def check_e1() -> list[dict]:
    miss = _missing("ELEVENLABS_API_KEY")
    if miss:
        return [_row("E1", "SKIP", "missing ELEVENLABS_API_KEY (ELEVENLABS_VOICE_ID optional)")]
    try:
        from blab.orchestrator import voice
        p = voice.tts("You've gone very formal all of a sudden.", ROOT / "runs" / "spike_e1.mp3")
        return [_row("E1", "PASS" if p else "FAIL", f"mp3 → {p}")]
    except Exception as e:  # noqa: BLE001
        return [_row("E1", "FAIL", f"{type(e).__name__}: {e}")]


ORDER = ["D1", "D2", "D2b", "D3", "S1", "S2", "S3", "A1", "A2", "C1", "E1"]


def cleanup_spike_boxes() -> int:
    if _missing("DAYTONA_API_KEY"):
        return 0
    n = 0
    try:
        from daytona import ListSandboxesQuery
        d = _daytona()
        for sb in d.list(ListSandboxesQuery(labels=SPIKE_LABELS)):
            try:
                d.delete(sb)
                n += 1
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return n


def print_table(rows: list[dict]) -> None:
    print(f"\n{'check':<6} {'status':<6} detail")
    print("-" * 100)
    for r in rows:
        print(f"{r['check']:<6} {r['status']:<6} {r['detail']}")


def run_spike(checks: list[str] | None = None) -> list[dict]:
    load_env()
    want = set(checks or ORDER)
    jobs = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        if want & {"D1", "D2"}:
            jobs.append(ex.submit(check_d1_d2))
        if "D3" in want:
            jobs.append(ex.submit(check_d3))
        if want & {"A1", "A2"}:
            jobs.append(ex.submit(check_a1))
        if "C1" in want:
            jobs.append(ex.submit(check_c1))
        if want & {"S1", "S2", "S3"}:
            jobs.append(ex.submit(check_surreal))
        if "E1" in want:
            jobs.append(ex.submit(check_e1))
        rows = [r for j in jobs for r in j.result()]
    rows = [r for r in rows if r["check"].split(".")[0] in want or r["check"] == "D2b"]
    key = lambda r: ORDER.index(r["check"].split(".")[0]) if r["check"].split(".")[0] in ORDER else 99  # noqa: E731
    rows.sort(key=key)
    left = cleanup_spike_boxes()
    print_table(rows)
    print(f"(spike sandboxes cleaned up: {left} leftover deleted)")
    out = ROOT / "runs" / "spike.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = {}
    if out.exists() and checks:
        try:
            prev = {r["check"]: r for r in json.loads(out.read_text())["rows"]}
        except Exception:  # noqa: BLE001
            prev = {}
    for r in rows:
        prev[r["check"]] = r
    if any(r["check"].startswith("C1.") for r in rows):
        prev.pop("C1", None)
    merged = sorted(prev.values(), key=key)
    out.write_text(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": merged}, indent=2))
    return rows


if __name__ == "__main__":
    import sys
    run_spike(sys.argv[1:] or None)
