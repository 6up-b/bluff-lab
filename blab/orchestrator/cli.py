"""`blab` CLI (SPEC §9). Thin dispatcher: each subcommand lazily imports its module so a
missing cloud dependency never breaks `blab local`.

Module owners implement the target functions with these exact signatures.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Optional

import typer

from blab.env import load_env

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _call(fn, **kw):
    out = fn(**kw)
    if inspect.iscoroutine(out):
        out = asyncio.run(out)
    return out


@app.callback()
def _root():
    load_env()


@app.command()
def spike():
    """§15 checks; prints a PASS/FAIL table."""
    from blab.orchestrator.spike import run_spike
    _call(run_spike)


@app.command()
def local(table: str = "telltale", agent: str = "profiler", seed: int = 7, games: int = 20,
          reflect_every: int = 2, out: str = "runs/local", llm: bool = False,
          ask: bool = False, wid: Optional[str] = None):
    """Run one world in-process (no Daytona). Stub reflection unless --llm."""
    from blab.world import run_local
    _call(run_local, table=table, agent=agent, seed=seed, games=games,
          reflect_every=reflect_every, out=out, llm=llm, ask=ask, wid=wid)


@app.command()
def replay(world: str, seq: int, variants: str = "full,as_of:0,false_memory",
           root: str = "runs/local"):
    """Re-run one recorded decision under memory variants (§7.2 decision-level)."""
    from blab.world import replay_cli
    _call(replay_cli, world=world, seq=seq, variants=variants.split(","), root=root)


@app.command()
def branch(world: str, from_game: int = 10, variant: str = "full", root: str = "runs/local",
           new_wid: Optional[str] = None):
    """Restore checkpoint g<from_game> and play the rest on the same seeds with memory variant (learning frozen)."""
    from blab.world import branch_cli
    _call(branch_cli, world=world, from_game=from_game, variant=variant, root=root, new_wid=new_wid)


@app.command()
def provision(boxes: int = 2, cpu: int = 4, mem: int = 8):
    from blab.orchestrator.boxes import provision as fn
    _call(fn, boxes=boxes, cpu=cpu, mem=mem)


@app.command()
def run(matrix: str = "matrix.yaml", run: str = "r01", local_only: bool = False):
    from blab.orchestrator.boxes import run_matrix
    _call(run_matrix, matrix=matrix, run=run, local_only=local_only)


@app.command()
def reflect(run: str = "r01"):
    from blab.orchestrator.reflect import reflect_worker
    _call(reflect_worker, run=run)


@app.command()
def ablate(run: str = "r01", matrix: str = "matrix.yaml"):
    from blab.orchestrator.boxes import ablate as fn
    _call(fn, run=run, matrix=matrix)


@app.command()
def forge(world: str, seat: str = "p1", run: str = "r01"):
    from blab.orchestrator.forge import forge as fn
    _call(fn, world=world, seat=seat, run=run)


@app.command()
def voice(run: str = "r01", world: str = ""):
    from blab.orchestrator.voice import voice_for_world
    _call(voice_for_world, run=run, world=world)


@app.command()
def collect(run: str = "r01"):
    from blab.orchestrator.boxes import collect as fn
    _call(fn, run=run)


@app.command()
def analyze(run: str = "r01"):
    from blab.orchestrator.analyze import analyze as fn
    _call(fn, run=run)


@app.command("pick-demo")
def pick_demo(run: str = "r01", out: str = "show/bundle"):
    from blab.orchestrator.pick import pick_demo as fn
    _call(fn, run=run, out=out)


@app.command()
def teardown(run: str = "r01", yes: bool = False):
    """Delete boxes. NOT before the awards ceremony."""
    from blab.orchestrator.boxes import teardown as fn
    _call(fn, run=run, yes=yes)


def main():
    app()


if __name__ == "__main__":
    main()
