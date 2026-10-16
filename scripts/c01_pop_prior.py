"""c01 step 3: population prior (DEV seeds 7000-7099 only; PREREG_c01.md).

Pooled grid posterior over strategic bluff propensity q from the revealed claims of CONTROL opponents (role
"control", seat p2) on the c01 tables that have one (tell_relational_v2, tell_flipper, tell_conjunctive),
p1 = counter (no learning; it only generates the record), raw info, 20 games, seeds 7000-7099.
Tempered to N0 = 10 effective claims (each likelihood raised to N0 / n_claims; a priori), so the prior carries
about one opponent's first two games of evidence and the first reveals can still move it.
Writes blab/data/c01_pop_prior.json.
"""
import asyncio, json
from pathlib import Path
import numpy as np
from blab import grid
from blab.world import World, make_seats, stub_reflector

N0 = 10.0
TABLES = ("tell_relational_v2", "tell_flipper", "tell_conjunctive")
claims = []


async def main():
    for t in TABLES:
        for s in range(7000, 7100):
            seats = make_seats(t, "counter", s)
            seats[0]["params"].update(info="raw")
            w = World(wid=f"pp-{t}-{s}", seats=seats, seed=s, games=20, reflector=stub_reflector, table=t)
            await w.run()
            obs = w.observe("p1")
            claims.extend((e["p_cards"], e["was_bluff"]) for e in obs["revealed_history"] if e["by"] == "p2")


asyncio.run(main())
wt = N0 / len(claims)
post = grid.posterior((pc, b, wt) for pc, b in claims)
m, lo, hi = grid.interval(post)
out = {"source": "c01 dev seeds 7000-7099, control opponents (p2) on " + ", ".join(TABLES) + ", p1=counter, raw, 20 games",
       "n_claims": len(claims), "n_bluffs": int(sum(b for _, b in claims)), "N0": N0, "weight_per_claim": wt,
       "mean": m, "ci95": [lo, hi], "grid": grid.to_list(post)}
Path("blab/data/c01_pop_prior.json").write_text(json.dumps(out))
print({k: v for k, v in out.items() if k != "grid"})
