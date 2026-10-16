"""c01 confirmatory run (PREREG_c01.md). Fresh test seeds 5000-5023, raw info, threshold rule, 20 games.

Phase A (no LLM calls, in-process): logreg_bow, logreg_embed, bayes_discount_pp on all 4 tables, and
profiler_plus_noreflect on the 3 tables whose utterances are single sentences (no record-check call is made).
Phase B (LLM: Opus reflection, Sonnet record check): 8 worker processes, one world each at a time, so at most
8 concurrent Claude calls. Queue order (pre-registered): for seed 5000..5023: for table: profiler_hybrid,
profiler_plus_c01; then profiler_plus_noreflect on tell_relational_v2 for every seed. No world is launched after
LAUNCH_DEADLINE; the analysis uses only seeds whose cells are all complete (PREREG_c01.md stopping rule).
Resumes: skips worlds whose meta.json status == done.
Usage: PYTHONPATH=. .venv/bin/python scripts/c01_run.py [A|B|AB]
"""
import asyncio, json, os, sys, time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["BLAB_RC_CACHE"] = "runs/c01/rc_cache.jsonl"
os.environ["BLAB_RC_LOG"] = "runs/c01/rc_log.jsonl"
OUT = "runs/c01"
TABLES = ["tell_relational_v2", "tell_flipper", "tell_conjunctive", "longcon"]
SEEDS = list(range(5000, 5024))
LAUNCH_DEADLINE = os.environ.get("C01_LAUNCH_DEADLINE", "15:30")
FAST = ["logreg_bow", "logreg_embed", "bayes_discount_pp"]
LLM = ["profiler_hybrid", "profiler_plus_c01"]


def done(wid):
    mp = f"{OUT}/{wid}/meta.json"
    return os.path.exists(mp) and json.load(open(mp)).get("status") == "done"


def run_world(job):
    table, agent, seed = job
    wid = f"{table}-{agent}-raw-s{seed}"
    if done(wid):
        return wid, "skip", None
    if job_llm(job) and time.strftime("%H:%M") >= LAUNCH_DEADLINE:
        return wid, "not_launched", None
    from blab.env import load_env
    load_env()
    from blab.orchestrator.reflect import local_reflector
    from blab.world import World, make_seats, stub_reflector
    seats = make_seats(table, agent, seed)
    seats[0]["params"].update(info="raw", respond_rule="threshold")
    refl = local_reflector if agent in LLM else stub_reflector
    w = World(wid=wid, seats=seats, seed=seed, games=20, out_dir=OUT, run="c01", box="laptop", reflector=refl,
              table=table)
    t = time.time()
    try:
        r = asyncio.run(w.run())
        print(f"[done] {time.strftime('%H:%M:%S')} {wid} {time.time()-t:.0f}s win={r['win_rate']:.2f} brier={r['brier']}",
              flush=True)
        return wid, "done", r["win_rate"]
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {wid} {e!r}", flush=True)
        return wid, "fail", repr(e)


def job_llm(job):
    t, a, _ = job
    return a in LLM or (a == "profiler_plus_noreflect" and t == "tell_relational_v2")


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "AB"
    os.makedirs(OUT, exist_ok=True)
    if "A" in phase:
        jobs = [(t, a, s) for s in SEEDS for t in TABLES for a in FAST]
        jobs += [(t, "profiler_plus_noreflect", s) for s in SEEDS for t in TABLES if t != "tell_relational_v2"]
        with Pool(8) as p:
            for r in p.imap(run_world, jobs):
                pass
        print(f"[phase A] {time.strftime('%H:%M:%S')} done", flush=True)
    if "B" in phase:
        jobs = [(t, a, s) for s in SEEDS for t in TABLES for a in LLM]
        jobs += [("tell_relational_v2", "profiler_plus_noreflect", s) for s in SEEDS]
        with Pool(8, maxtasksperchild=1) as p:
            for r in p.imap(run_world, jobs):
                if r[1] == "not_launched":
                    print(f"[not_launched] {r[0]}", flush=True)
        print(f"[phase B] {time.strftime('%H:%M:%S')} done", flush=True)
