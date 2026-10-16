# usage: PYTHONPATH=. .venv/bin/python scripts/c02_embed_mem01.py   (resumes: skips worlds whose meta.json status == done)
"""c02 (A), EXPLORATORY (baseline chosen after mem01 was run): logreg_embed on the exact mem01 core tables and
seeds, raw info, threshold rule, 20 games. Same wid scheme as mem01: <table>-logreg_embed-raw-s<seed> in runs/mem01.
logreg_embed never calls the network: embeddings come only from blab/data/embed_cache.json (a miss raises)."""
import asyncio, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

OUT = "runs/mem01"
TABLES = ["telltale", "tell_conjunctive", "tell_flipper", "longcon"]
SEEDS = list(range(3001, 3013))


def check_cache():
    from blab.agents import embed
    d = json.load(open(embed.CACHE))
    miss = [s for s in embed.enumerate_texts() if embed.sha1(s) not in d["vectors"]]
    print(f"embed cache: {d['n']} vectors ({d['model']}), enumeration misses: {len(miss)}", flush=True)
    assert not miss, miss[:5]


def one(args):
    table, seed = args
    from blab.world import World, make_seats, stub_reflector
    agent = "logreg_embed"
    wid = f"{table}-{agent}-raw-s{seed}"
    mp = f"{OUT}/{wid}/meta.json"
    if os.path.exists(mp) and json.load(open(mp)).get("status") == "done":
        return f"[skip] {wid}"
    seats = make_seats(table, agent, seed)
    seats[0]["params"].update(info="raw", respond_rule="threshold")
    w = World(wid=wid, seats=seats, seed=seed, games=20, reflect_every=2, out_dir=OUT, run="mem01_embed_explore",
              box="laptop", reflector=stub_reflector, table=table)
    t = time.time()
    r = asyncio.run(w.run())
    return f"[done] {wid} {time.time()-t:.0f}s win={r['win_rate']:.3f} brier={r['brier']}"


if __name__ == "__main__":
    check_cache()
    jobs = [(t, s) for s in SEEDS for t in TABLES]
    with ProcessPoolExecutor(int(sys.argv[1]) if sys.argv[1:] else 12) as ex:
        for msg in ex.map(one, jobs):
            print(msg, flush=True)
