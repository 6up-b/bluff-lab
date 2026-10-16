"""c01 smoke test on a DEV seed (7000): one short world per given arm/table, real Claude. Not part of the analysis."""
import asyncio, os, sys, time, json
os.environ.setdefault("BLAB_RC_CACHE", "runs/c01dev/rc_cache.jsonl")
os.environ.setdefault("BLAB_RC_LOG", "runs/c01dev/rc_log.jsonl")
from blab.env import load_env
load_env()
from blab.orchestrator.reflect import local_reflector
from blab.world import World, make_seats
from blab.agents.c01 import CHECKER
table, agent, games = sys.argv[1], sys.argv[2], int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 7000
seats = make_seats(table, agent, seed)
seats[0]["params"].update(info="raw", respond_rule="threshold")
t = time.time()
w = World(wid=f"{table}-{agent}-raw-s{seed}", seats=seats, seed=seed, games=games, out_dir="runs/c01dev", run="c01dev",
          box="laptop", reflector=local_reflector, table=table)
r = asyncio.run(w.run())
print(f"{time.time()-t:.0f}s win={r['win_rate']} brier={r['brier']}", CHECKER.stats)
