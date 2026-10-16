"""b01 runner (capability boundary, PREREG.md): CPU-only arms on the laptop (process pool).

Profiler with real Claude reflection is NOT run by this runner (see runs/b01/NOT_RUN.md)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import yaml

OUT = "runs/b01"


def one(args):
    table, agent, seed, games = args
    from blab.world import World, make_seats, stub_reflector
    wid = f"{table}-{agent}-raw-s{seed}"
    mp = f"{OUT}/{wid}/meta.json"
    if os.path.exists(mp) and json.load(open(mp)).get("status") == "done":
        return wid, "skip"
    seats = make_seats(table, agent, seed)
    seats[0]["params"].update(info="raw", respond_rule="threshold")
    w = World(wid=wid, seats=seats, seed=seed, games=games, reflect_every=2, out_dir=OUT, run="b01", box="laptop",
              reflector=stub_reflector, table=table)
    t = time.time()
    r = asyncio.run(w.run())
    return wid, f"{time.time()-t:.1f}s win={r['win_rate']:.2f} brier={r['brier']}"


def main(matrix="matrix_boundary.yaml", only=None):
    m = yaml.safe_load(open(matrix))
    s0, s1 = m["seeds"]
    agents = [a for a in m["agents"] if a not in m.get("not_run", {})]
    if only:
        agents = [a for a in agents if a in only]
    jobs = [(t, a, s, m["games"]) for s in range(s0, s1 + 1) for t in m["tables"] for a in agents]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("B01_WORKERS", "24"))) as ex:
        for wid, msg in ex.map(one, jobs):
            print(f"[done] {time.strftime('%H:%M:%S')} {wid} {msg}", flush=True)


if __name__ == "__main__":
    main(only=sys.argv[1:] or None)
