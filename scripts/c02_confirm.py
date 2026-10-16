# usage: PYTHONPATH=. .venv/bin/python scripts/c02_confirm.py   (resumes: skips worlds whose meta.json status == done)
"""c02 confirmatory run (PREREG_c02.md): memory agent vs the sentence-embedding baseline.
Arms: counter, logreg_embed, profiler_plus_noreflect (no network), then profiler_plus (real Claude reflection,
Bedrock Opus 5.5, effort low, raw prompt llm.STABLE_SYSTEM_RAW, after every game as in mem01).
Queue: non-LLM arms for all seeds first, then profiler_plus seed-major (seed -> table). At most 6 worlds and
6 concurrent Claude calls. No world launched after LAUNCH_DEADLINE; everything stopped at HARD_STOP."""
import asyncio, json, os, sys, time
os.environ["BLAB_LLM_CONCURRENCY"] = "6"
from blab.env import load_env
load_env()
from blab.orchestrator import llm
from blab.orchestrator.reflect import local_reflector
from blab.world import World, make_seats, stub_reflector

assert "formal" not in llm.STABLE_SYSTEM_RAW.lower() and "contraction" not in llm.STABLE_SYSTEM_RAW.lower()
OUT = "runs/c02"
TABLES = ["telltale", "tell_conjunctive", "tell_flipper", "longcon"]
SEEDS = list(range(9000, 9012))           # PREREG_c02: n = 12
LAUNCH_DEADLINE = "15:30"
HARD_STOP = "15:40"
FAST = ("counter", "logreg_embed", "profiler_plus_noreflect")
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
        # profiler_plus_noreflect: profiler_plus whose reflector refuses (stub_reflector raises on raw payloads),
        # exactly mem01's plus_nohyp control: memory keeps no hypotheses, no network.
        refl = local_reflector if arm == "profiler_plus" else stub_reflector
        w = World(wid=wid, seats=seats, seed=seed, games=20, reflect_every=2, out_dir=OUT, run="c02", box="laptop",
                  reflector=refl, table=table)
        t = time.time()
        try:
            r = await w.run()
            print(f"[done] {time.strftime('%H:%M:%S')} {wid} {time.time()-t:.0f}s", flush=True)   # no results printed
        except Exception as e:                                                           # noqa: BLE001
            print(f"[FAIL] {wid} {e!r}", flush=True)


async def main():
    for s in SEEDS:
        for t in TABLES:
            for a in FAST:
                await one(t, a, s)
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
