"""Strategy Forge: static gate (runs on the orchestrator) + in-sandbox evaluator (SPEC §7.4).

Gate:  `gate(source) -> {"ok": bool, "reason": str}`  — never executes the source.
Eval (inside a network_block_all Daytona forge box):
    python -m blab.forge_eval --policy policy.py --table telltale --seeds 1,2,... --games 20 --decide-timeout-ms 50
The engine runs in THIS (parent) process; the untrusted policy runs in a spawned child process that receives
only observation JSON over a pipe and returns action JSON. Illegal action / exception / timeout → legal fallback
(challenge if a claim is pending, else play 1 card), counted. stdout = one JSON line
{win_rate, ci, crashes, timeouts, illegal, mean_ms, games}.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time

ALLOWED_IMPORTS = {"math", "random", "statistics", "collections", "itertools", "functools"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                   "setattr", "delattr", "input", "breakpoint"}
MAX_SOURCE = 20 * 1024


def gate(source: str) -> dict:
    if len(source.encode()) > MAX_SOURCE:
        return {"ok": False, "reason": f"source > {MAX_SOURCE} bytes"}
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {"ok": False, "reason": f"syntax error: {e}"}
    has_decide = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return {"ok": False, "reason": f"import {a.name} not allowed (line {node.lineno})"}
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if node.level or mod not in ALLOWED_IMPORTS:
                return {"ok": False, "reason": f"from {node.module} import not allowed (line {node.lineno})"}
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return {"ok": False, "reason": f"forbidden name {node.id} (line {node.lineno})"}
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            return {"ok": False, "reason": f"forbidden dunder name {node.id} (line {node.lineno})"}
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in FORBIDDEN_NAMES):
            return {"ok": False, "reason": f"forbidden attribute .{node.attr} (line {node.lineno})"}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "decide":
            if isinstance(node, ast.FunctionDef) and tree.body.count(node):
                has_decide = True
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            return {"ok": False, "reason": f"global/nonlocal not allowed (line {node.lineno})"}
    if not has_decide:
        return {"ok": False, "reason": "must define top-level def decide(obs)"}
    return {"ok": True, "reason": "ok"}


# ------------------------------------------------------------------ child process (untrusted policy)

def _child(source: str, conn) -> None:
    import builtins
    safe = {k: getattr(builtins, k) for k in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list",
        "map", "max", "min", "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
        "ValueError", "KeyError", "IndexError", "Exception", "print")}

    def _imp(name, *a, **k):
        if name.split(".")[0] not in ALLOWED_IMPORTS:
            raise ImportError(name)
        return __import__(name, *a, **k)
    safe["__import__"] = _imp
    g = {"__builtins__": safe, "__name__": "policy"}
    try:
        exec(compile(source, "policy.py", "exec"), g)   # sandboxed child in a network-blocked forge box only
        decide = g["decide"]
    except Exception as e:  # noqa: BLE001
        conn.send({"fatal": f"{type(e).__name__}: {e}"})
        return
    conn.send({"ready": True})
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        try:
            conn.send({"action": decide(json.loads(msg))})
        except Exception as e:  # noqa: BLE001
            conn.send({"error": f"{type(e).__name__}: {e}"})


class ChildPolicy:
    """Policy adapter: obs → child over a pipe; enforces the per-decision timeout and legality."""
    id = "forge"

    def __init__(self, source: str, timeout_ms: int = 50):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.parent, child = ctx.Pipe()
        self.proc = ctx.Process(target=_child, args=(source, child), daemon=True)
        self.proc.start()
        self.timeout = timeout_ms / 1000.0
        self.crashes = self.timeouts = self.illegal = 0
        self.ms: list[float] = []
        first = self.parent.recv() if self.parent.poll(10) else {"fatal": "child did not start"}
        if "fatal" in first:
            raise RuntimeError(first["fatal"])

    @staticmethod
    def fallback(obs: dict) -> dict:
        if obs.get("pending_claim") and "challenge" in obs.get("legal_actions", ["challenge"]):
            return {"type": "challenge"}
        return {"type": "play", "cards": [0], "utterance": "One card."}

    def _legal(self, a, obs) -> bool:
        if not isinstance(a, dict):
            return False
        legal = obs.get("legal_actions") or []
        if a.get("type") == "challenge":
            return "challenge" in legal
        if a.get("type") == "play" and "play" in legal:
            cards = a.get("cards")
            n = len(obs.get("my_hand") or [])
            return (isinstance(cards, list) and 1 <= len(cards) <= 3 and len(set(cards)) == len(cards)
                    and all(isinstance(c, int) and 0 <= c < n for c in cards))
        return False

    def act(self, obs: dict) -> dict:
        t0 = time.perf_counter()
        if not self.proc.is_alive():
            self.crashes += 1
            return self.fallback(obs)
        self.parent.send(json.dumps(obs))
        if not self.parent.poll(self.timeout):
            self.timeouts += 1
            self.proc.kill()
            return self.fallback(obs)
        r = self.parent.recv()
        self.ms.append((time.perf_counter() - t0) * 1000)
        if "error" in r:
            self.crashes += 1
            return self.fallback(obs)
        a = r.get("action")
        if not self._legal(a, obs):
            self.illegal += 1
            return self.fallback(obs)
        if a["type"] == "play":
            a = {"type": "play", "cards": a["cards"], "utterance": str(a.get("utterance", ""))[:120]}
        return a

    def close(self):
        try:
            self.parent.send(None)
        except Exception:  # noqa: BLE001
            pass
        self.proc.kill()


def evaluate(source: str, table: str, seeds: list[int], games: int, timeout_ms: int) -> dict:
    """Plays `games` games per seed with the forge policy in seat p1 (core's World, stub reflection off)."""
    import asyncio
    import numpy as np
    from blab import world as W

    totals = {"won": 0, "games": 0, "crashes": 0, "timeouts": 0, "illegal": 0, "ms": []}
    per_seed = []
    for seed in seeds:
        pol = ChildPolicy(source, timeout_ms)
        try:
            w = W.World(wid=f"forge-s{seed}", seats=W.make_seats(table, "counter", seed), seed=seed, games=games,
                        out_dir=None, run="forge", box="forge", learning=False, table=table)
            w.policies["p1"] = pol
            res = asyncio.run(w.run())
            won = int(res.get("games_won", round(res.get("win_rate", 0) * games)))
        finally:
            pol.close()
        per_seed.append(won / games)
        totals["won"] += won
        totals["games"] += games
        for k in ("crashes", "timeouts", "illegal"):
            totals[k] += getattr(pol, k)
        totals["ms"] += pol.ms
    wr = totals["won"] / max(1, totals["games"])
    rng = np.random.default_rng(0)
    arr = np.array(per_seed)
    boots = [rng.choice(arr, len(arr)).mean() for _ in range(2000)] if len(arr) else [0]
    return {"win_rate": wr, "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "crashes": totals["crashes"], "timeouts": totals["timeouts"], "illegal": totals["illegal"],
            "mean_ms": float(np.mean(totals["ms"])) if totals["ms"] else None, "games": totals["games"],
            "per_seed": per_seed}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--table", default="telltale")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--decide-timeout-ms", type=int, default=50)
    a = ap.parse_args(argv)
    src = open(a.policy).read()
    g = gate(src)
    if not g["ok"]:
        print(json.dumps({"gate": g}))
        sys.exit(2)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    print(json.dumps(evaluate(src, a.table, seeds, a.games, a.decide_timeout_ms)))


if __name__ == "__main__":
    main()
