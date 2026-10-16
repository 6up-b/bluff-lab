"""Fork Reality (lc02): a REAL Daytona VM fork of a live Long Con world at the moment of betrayal (sleeper's G0).

    python -m blab.orchestrator.fork            → runs/lc02/longcon_fork.json + runs/lc02/fork/<child>/...

Parent VM (snapshot daytona-vm-small) runs worldd + one longcon profiler world paused right before game G0
(ckpt g<G0-1> on disk). For each intervention we `sb.fork()` the parent (sequentially: one fork at a time),
restart worldd in the child, verify the child serves the same world with the same last seq on its OWN preview
link, then POST /continue (restore ckpt, RNG copied ⇒ paired by construction) with one intervention, run to
game 20, pull the child's events, delete the child. If VM create or fork fails we say so (real_vm_fork=false);
the checkpoint-branch fallback is `blab ablate`, which must be called a "checkpoint branch", never a fork.
"""
from __future__ import annotations

import io
import json
import os
import tarfile
import time
from pathlib import Path

import httpx

from blab.env import load_env

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "lc02"
PIP = "fastapi uvicorn numpy 'pydantic>=2' httpx pyyaml"
INTERVENTIONS = ["none", "erase", "false_memory", "drift_off"]


def _h(link, key):
    return {"x-daytona-preview-token": link.token or "", "X-Blab-Key": key}


def _wait(url, headers, tries=90):
    for _ in range(tries):
        try:
            r = httpx.get(url + "/health", headers=headers, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    return None


def _start(sb, key, box_id):
    from daytona import SessionExecuteRequest
    from blab.orchestrator.boxes import KILL_WORLDD
    sb.fs.upload_file(KILL_WORLDD.encode(), "/home/daytona/kill_worldd.py")
    sb.process.exec("python3 /home/daytona/kill_worldd.py", timeout=30)
    sess = f"worldd-{int(time.time() * 1000) % 10**8}"
    sb.process.create_session(sess)
    sb.process.execute_session_command(sess, SessionExecuteRequest(
        command=(f"cd /home/daytona/app && BLAB_KEY={key} BLAB_BOX_ID={box_id} BLAB_WORLDS=/home/daytona/worlds "
                 f"python3 -m uvicorn blab.worldd:app --host 0.0.0.0 --port 8000 > /home/daytona/worldd.log 2>&1"),
        run_async=True))


def run_fork(seed: int = 6000, table: str = "longcon") -> dict:
    load_env()
    from daytona import CreateSandboxFromSnapshotParams, Daytona
    from blab.orchestrator.boxes import git_tarball
    from blab.world import make_seats
    d = Daytona()
    key = os.environ.get("BLAB_KEY") or "lc02"
    rec: dict = {"real_vm_fork": False, "seed": seed, "table": table, "children": [], "errors": []}
    seats = make_seats(table, "profiler", seed)
    sleeper = next(s for s in seats if s.get("role") == "sleeper")
    g0 = int(sleeper["params"]["G0"])
    seats[0]["params"] = {"respond_rule": "threshold", "reflection": "stub"}
    rec.update(sleeper_seat=sleeper["seat"], G0=g0, pause_before_game=g0)
    wid = f"lcf{seed}"
    t0 = time.time()
    try:
        parent = d.create(CreateSandboxFromSnapshotParams(snapshot="daytona-vm-small", auto_stop_interval=0,
                                                          labels={"project": "blufflab", "role": "vm-parent"}),
                          timeout=300)
    except Exception as e:  # noqa: BLE001
        rec["errors"].append(f"VM create failed: {type(e).__name__}: {e}")
        return _save(rec)
    rec["parent_sandbox_id"] = parent.id
    rec["vm_create_s"] = round(time.time() - t0, 1)
    print(f"[fork] parent VM {parent.id} in {rec['vm_create_s']}s")
    r = parent.process.exec(f"python3 -c 'import fastapi,uvicorn,numpy,pydantic,httpx,yaml' 2>/dev/null || "
                            f"(pip install -q {PIP} 2>&1 || pip3 install -q --break-system-packages {PIP} 2>&1) | tail -2",
                            timeout=600)
    parent.fs.upload_file(git_tarball(), "/home/daytona/app.tgz")
    parent.process.exec("mkdir -p /home/daytona/app /home/daytona/worlds && tar xzf /home/daytona/app.tgz -C /home/daytona/app",
                        timeout=120)
    _start(parent, key, "vm-parent")
    plink = parent.get_preview_link(8000)
    ph = _h(plink, key)
    if not _wait(plink.url, ph):
        rec["errors"].append("parent worldd did not come up: " +
                             (parent.process.exec("tail -5 /home/daytona/worldd.log", timeout=20).result or "")[-300:])
        return _save(rec)
    body = {"wid": wid, "run": "lc02", "seats": seats, "seed": seed, "games": 20, "reflect_every": 2,
            "table": table, "reflector": "stub", "mode": "async", "pause_before_game": g0, "start": True}
    rr = httpx.post(plink.url + "/worlds", json=body, headers=ph, timeout=60)
    rec["parent_create"] = rr.json()
    for _ in range(120):
        st = httpx.get(plink.url + "/worlds", headers=ph, timeout=30).json()
        s = next((w for w in st if w["wid"] == wid), {})
        if s.get("status") == "paused":
            break
        time.sleep(0.5)
    evs = httpx.get(plink.url + f"/worlds/{wid}/events", headers=ph, timeout=60).json()
    rec["parent_status"] = s.get("status")
    rec["parent_last_seq"] = evs[-1]["seq"] if evs else None
    rec["parent_last_game"] = evs[-1]["game"] if evs else None
    print(f"[fork] parent world {wid} status={s.get('status')} last seq={rec['parent_last_seq']} "
          f"(paused before G0={g0})")
    for iv in INTERVENTIONS:
        ch: dict = {"intervention": iv}
        t1 = time.time()
        try:
            child = parent.fork(name=f"lc-fork-{iv.replace('_', '-')}-{int(time.time()) % 10000}", timeout=300)
        except Exception as e:  # noqa: BLE001
            ch["error"] = f"fork failed: {type(e).__name__}: {e}"
            rec["children"].append(ch)
            rec["errors"].append(ch["error"])
            print(f"[fork] {ch['error']}")
            break
        ch["fork_s"] = round(time.time() - t1, 1)
        ch["sandbox_id"] = child.id
        rec["real_vm_fork"] = True
        try:
            _start(child, key, f"vm-{iv}")
            clink = child.get_preview_link(8000)
            ch["preview_host_differs"] = clink.url != plink.url
            chh = _h(clink, key)
            ch["child_health"] = _wait(clink.url, chh)
            cev = httpx.get(clink.url + f"/worlds/{wid}/events", headers=chh, timeout=60).json()
            ch["child_last_seq"] = cev[-1]["seq"] if cev else None
            ch["same_last_seq"] = ch["child_last_seq"] == rec["parent_last_seq"]
            cb = {"new_wid": f"{wid}_{iv}", "from_game": g0 - 1, "games": 20, "intervention": iv,
                  "about": sleeper["seat"], "text": f"Player {sleeper['seat'][-1]} never bluffs"}
            ch["continue"] = httpx.post(clink.url + f"/worlds/{wid}/continue", json=cb, headers=chh, timeout=120).json()
            for _ in range(240):
                st = httpx.get(clink.url + "/worlds", headers=chh, timeout=30).json()
                cs = next((w for w in st if w["wid"] == cb["new_wid"]), {})
                if cs.get("status") in ("done", "error"):
                    break
                time.sleep(0.5)
            ch["status"] = cs.get("status")
            ch["results"] = httpx.get(clink.url + f"/worlds/{cb['new_wid']}/results", headers=chh, timeout=60).json()
            r = httpx.get(clink.url + "/export", params={"wids": cb["new_wid"]}, headers=chh, timeout=300)
            dst = OUT / "fork"
            dst.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
                tf.extractall(dst, filter="data")
            ch["events_dir"] = str((dst / cb["new_wid"]).relative_to(ROOT))
            ch["total_s"] = round(time.time() - t1, 1)
            print(f"[fork] {iv}: child {child.id} fork={ch['fork_s']}s same_seq={ch['same_last_seq']} "
                  f"→ {ch['status']} win_rate={ch['results'].get('win_rate')}")
        except Exception as e:  # noqa: BLE001
            ch["error"] = f"{type(e).__name__}: {e}"
            print(f"[fork] {iv}: {ch['error']}")
        finally:
            try:
                ch["fork_tree"] = httpx.get(f"{os.environ.get('DAYTONA_API_URL', 'https://app.daytona.io/api')}"
                                            f"/sandbox/{parent.id}/forks",
                                            headers={"Authorization": f"Bearer {os.environ['DAYTONA_API_KEY']}"},
                                            timeout=30).json() if iv == INTERVENTIONS[-1] else None
            except Exception as e:  # noqa: BLE001
                ch["fork_tree"] = f"unavailable: {e}"
            try:
                d.delete(child)
                ch["deleted"] = True
            except Exception:  # noqa: BLE001
                ch["deleted"] = False
        rec["children"].append(ch)
    # parent world copy for the Show
    try:
        r = httpx.get(plink.url + "/export", params={"wids": wid}, headers=ph, timeout=300)
        with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
            tf.extractall(OUT / "fork", filter="data")
    except Exception as e:  # noqa: BLE001
        rec["errors"].append(f"parent export: {e}")
    rec["total_s"] = round(time.time() - t0, 1)
    return _save(rec)


def _save(rec: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "longcon_fork.json").write_text(json.dumps(rec, indent=1, default=str))
    print(f"[fork] real_vm_fork={rec['real_vm_fork']} → {OUT / 'longcon_fork.json'}")
    return rec


if __name__ == "__main__":
    run_fork()
