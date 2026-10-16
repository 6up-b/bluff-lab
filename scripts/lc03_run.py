# usage: PYTHONPATH=. .venv/bin/python scripts/lc03_run.py   (resumes: skips worlds whose meta.json status == done)
"""lc03 (PREREG_lc03.md): Long Con with profiler_plus. Tables longcon, falsealarm. Seeds 10000-10011.
Fast arms (counter, bayes_discount, logreg_embed, profiler_plus_noreflect) first, then profiler_plus (real Claude
reflection, Bedrock Opus 5.5, effort low, raw prompt llm.STABLE_SYSTEM_RAW) seed-major, 6 at a time.
No world launched after LAUNCH_DEADLINE; everything stopped at HARD_STOP. Prints no results."""
import asyncio, json, os, time
os.environ["BLAB_LLM_CONCURRENCY"] = "6"
from blab.env import load_env
load_env()
from blab.orchestrator import llm
from blab.orchestrator.reflect import local_reflector
from blab.world import World, make_seats, stub_reflector, TABLES as WT
from blab.agents import embed

assert "formal" not in llm.STABLE_SYSTEM_RAW.lower() and "contraction" not in llm.STABLE_SYSTEM_RAW.lower()
assert WT["longcon"] == ["telltale", "sleeper"] and WT["falsealarm"] == ["control", "control"]
assert llm.BEDROCK_MODEL == "us.anthropic.claude-opus-5-5", llm.BEDROCK_MODEL
# embedding cache must cover every sentence the scripted roles can say (a miss is an error)
_c = embed._cache()
_miss = [s for s in embed.enumerate_texts() if embed.sha1(s) not in _c]
assert not _miss, f"embed cache misses: {len(_miss)} e.g. {_miss[:3]}"
print(f"[check] embed cache covers {len(embed.enumerate_texts())} sentences, 0 misses", flush=True)

OUT = "runs/lc03"
TABLES = ["longcon", "falsealarm"]
SEEDS = list(range(10000, 10012))           # PREREG_lc03: n = 12
LAUNCH_DEADLINE = "15:30"
HARD_STOP = "15:38"
FAST = ("counter", "bayes_discount", "logreg_embed", "profiler_plus_noreflect")
POLICY = {"profiler_plus_noreflect": "profiler_plus"}
sem = asyncio.Semaphore(6)


def past(hhmm):
    return time.strftime("%H:%M") >= hhmm


async def one(table, arm, seed):
    wid = f"{table}-{arm}-raw-s{seed}"
    mp = f"{OUT}/{wid}/meta.json"
    if os.path.exists(mp) and json.load(open(mp)).get("status") == "done":
        return
    async with sem:
        if arm == "profiler_plus" and past(LAUNCH_DEADLINE):
            print(f"[not-launched] {wid} (launch deadline {LAUNCH_DEADLINE})", flush=True)
            return
        seats = make_seats(table, POLICY.get(arm, arm), seed)
        seats[0]["params"].update(info="raw", respond_rule="threshold")
        refl = local_reflector if arm == "profiler_plus" else stub_reflector
        w = World(wid=wid, seats=seats, seed=seed, games=20, reflect_every=2, out_dir=OUT, run="lc03", box="laptop",
                  reflector=refl, table=table)
        t = time.time()
        try:
            await w.run()
            print(f"[done] {time.strftime('%H:%M:%S')} {wid} {time.time()-t:.0f}s", flush=True)
        except Exception as e:                                                           # noqa: BLE001
            print(f"[FAIL] {wid} {e!r}", flush=True)


async def main():
    os.makedirs(OUT, exist_ok=True)
    specs = [{"wid": f"{t}-{a}-raw-s{s}", "table": t, "arm": a, "seed": s}
             for s in SEEDS for t in TABLES for a in FAST + ("profiler_plus",)]
    json.dump(specs, open(f"{OUT}/worlds.json", "w"), indent=1)
    for s in SEEDS:
        for t in TABLES:
            for a in FAST:
                await one(t, a, s)
    print(f"[fast-done] {time.strftime('%H:%M:%S')}", flush=True)
    jobs = [asyncio.create_task(one(t, "profiler_plus", s)) for s in SEEDS for t in TABLES]
    while not all(j.done() for j in jobs):
        if past(HARD_STOP):
            print(f"[hard-stop] {HARD_STOP}: cancelling {sum(not j.done() for j in jobs)} worlds", flush=True)
            for j in jobs:
                j.cancel()
            break
        await asyncio.sleep(5)
    await asyncio.gather(*jobs, return_exceptions=True)
    print("[runner-exit]", time.strftime("%H:%M:%S"), flush=True)

asyncio.run(main())
