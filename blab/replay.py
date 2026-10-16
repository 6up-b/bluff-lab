"""Memory Surgery (SPEC §7.2): decision-level replay and outcome-level branches.

memory_variant dicts (§8.3):
  {"type":"full"} | {"type":"as_of","game":N} | {"type":"minus","aspect":A}
  {"type":"false_memory","about":"p3","aspect":"baseline_bluff","text":"…","q":0.0[, "predicate":…]}
  {"type":"transplant","memory":{…inline…} | "from_world":W[, "root":…], "from_seat":"p1", "about_map":{"p3":"p3"}}
Optional on any variant: "agent": "<policy id>" re-runs the same state with a different agent (e.g. profiler_blind).
String forms: full | as_of:N | minus:A | false_memory | transplant:<wid> | agent:<id> | <any>@<agent id>
"""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

from blab import hypotheses as H
from blab.agents.ours import make_agent
from blab.events import read_events

FALSE_MEMORY_DEFAULT = {"about": "p3", "aspect": "baseline_bluff", "text": "Player 3 never bluffs", "q": 0.0}
# One leaf that matches every claim (claim_count is 1..3). A one-leaf line competes with other one-leaf
# lines on recency in matching_q, so the injected (newest) line actually takes effect for an agent that
# trusts it — `{"all":[]}` would always lose to any confirmed specific line and test nothing.
FALSE_MEMORY_PREDICATE = {"f": "claim_count", "op": ">=", "v": 1}


def parse_variant(s: str | dict) -> dict:
    if isinstance(s, dict):
        return s
    s = s.strip()
    if "@" in s and not s.startswith("{"):        # "<variant>@<agent>", e.g. "false_memory@profiler_blind"
        base, agent = s.rsplit("@", 1)
        return {**parse_variant(base), "agent": agent}
    if s == "full":
        return {"type": "full"}
    if s.startswith("as_of:"):
        return {"type": "as_of", "game": int(s.split(":", 1)[1])}
    if s.startswith("minus:"):
        return {"type": "minus", "aspect": s.split(":", 1)[1]}
    if s == "false_memory":
        return {"type": "false_memory", **FALSE_MEMORY_DEFAULT}
    if s.startswith("transplant:"):
        return {"type": "transplant", "from_world": s.split(":", 1)[1], "from_seat": "p1"}
    if s.startswith("agent:"):           # same memory, different agent: "agent:profiler_blind"
        return {"type": "full", "agent": s.split(":", 1)[1]}
    if s.startswith("{"):
        return json.loads(s)
    raise ValueError(f"unknown memory variant {s!r}")


def _load_ckpt(world_dir: Path, g: int) -> dict:
    return json.loads((Path(world_dir) / "ckpt" / f"g{g:02d}.json").read_text())


def _ckpt_games(world_dir: Path) -> list[int]:
    return sorted(int(p.stem[1:]) for p in (Path(world_dir) / "ckpt").glob("g*.json"))


def apply_variant(memory: dict | None, variant: dict, *, world_dir: Path, seat: str, at_game: int,
                  wid: str) -> dict | None:
    """Memory M' for `seat` given the memory M at checkpoint `at_game` of world_dir."""
    v = parse_variant(variant)
    t = v.get("type", "full")
    if memory is None:
        return None
    m = copy.deepcopy(memory)
    if t == "full":
        return m
    if t == "as_of":
        games = [g for g in _ckpt_games(world_dir) if g <= min(int(v["game"]), at_game)]
        g = max(games) if games else 0
        mm = _load_ckpt(world_dir, g)["memory"].get(seat)
        return copy.deepcopy(mm) if mm is not None else None
    if t == "minus":
        m["hypotheses"] = [h for h in m.get("hypotheses", []) if h.get("aspect") != v["aspect"]]
        m["memory_version"] = m.get("memory_version", 0) + 1
        return m
    if t == "false_memory":
        idx = max([h.get("idx", 0) for h in m.get("hypotheses", [])] + [m.get("next_idx", 1) - 1, 0]) + 1
        h = H.new_hypothesis(wid=wid, about=v.get("about", "p3"), aspect=v.get("aspect", "baseline_bluff"),
                             text=v.get("text", FALSE_MEMORY_DEFAULT["text"]),
                             predicate=v.get("predicate") or FALSE_MEMORY_PREDICATE, created_game=at_game,
                             provenance="untrusted", idx=idx, origin="surgery", q=float(v.get("q", 0.0)))
        h["status"] = "confirmed"          # asserted by the injector, NOT by scoring (confirmed_by stays None)
        h["confirmed_by"] = None
        m.setdefault("hypotheses", []).append(h)
        m["next_idx"] = idx + 1
        m["memory_version"] = m.get("memory_version", 0) + 1
        return m
    if t == "transplant":
        src = v.get("memory")
        if src is None:
            root = Path(v.get("root") or Path(world_dir).parent)
            src = json.loads((root / v["from_world"] / "memory" / f"{v.get('from_seat', 'p1')}.json").read_text())
        amap = v.get("about_map") or {"p2": "p2", "p3": "p3"}
        src = copy.deepcopy(src)
        out = copy.deepcopy(m)
        out["hypotheses"] = []
        for h in src.get("hypotheses", []):
            if h.get("about") in amap:
                h["about"] = amap[h["about"]]
                h["id"] = h["id"] + "_tx"
                h["transplanted_from"] = v.get("from_world")
                out["hypotheses"].append(h)
        out["base"] = {amap[k]: b for k, b in (src.get("base") or {}).items() if k in amap}
        out["revealed"] = [dict(e, by=amap[e["by"]]) for e in src.get("revealed", []) if e.get("by") in amap]
        out["memory_version"] = m.get("memory_version", 0) + 1
        return out
    raise ValueError(f"unknown variant type {t!r}")


def _seat_spec(meta: dict, seat: str) -> dict:
    return next(sp for sp in meta["seats"] if sp["seat"] == seat)


def replay_decision(world_dir: str | Path, seq: int, variant: dict | str) -> dict:
    """Re-run one recorded decision under memory variant M' with the SAME obs and rng_u."""
    world_dir = Path(world_dir)
    v = parse_variant(variant)
    meta = json.loads((world_dir / "meta.json").read_text())
    ev = next((e for e in read_events(world_dir / "events.jsonl") if e["seq"] == int(seq)), None)
    if ev is None or ev["type"] != "decision":
        raise ValueError(f"seq {seq} is not a decision event in {world_dir}")
    d, seat, game = ev["payload"], ev["actor"], ev["game"]
    agent_id = v.get("agent") or d.get("agent") or _seat_spec(meta, seat)["policy"]
    mem = _load_ckpt(world_dir, game - 1)["memory"].get(seat)
    m2 = apply_variant(mem, v, world_dir=world_dir, seat=seat, at_game=game - 1, wid=meta["wid"])
    params = {**_seat_spec(meta, seat).get("params", {}), "wid": meta["wid"], "seed": meta["seed"],
              "reflect_every": meta.get("reflect_every", 2), "learning": False}
    pol = make_agent(agent_id, seat, None, params)
    if hasattr(pol, "set_memory"):
        pol.set_memory(m2)
    if d["kind"] == "respond":
        out = pol.decide_respond(d["obs"], d["rng_u"])
    else:
        _, out = pol.decide_play(d["obs"])
    out = json.loads(json.dumps(out))
    out["trace"] = {}
    res = dict(out)
    res.update({"replay_of": int(seq), "variant": v,
                "logged": {"p_final": d.get("p_final"), "action": d.get("action"), "q_used": d.get("q_used"),
                           "q_source": d.get("q_source")},
                "delta_p": (out["p_final"] - d["p_final"]) if d.get("p_final") is not None else None,
                "flipped": out["action"] != d["action"],
                "delta_ev": ((out["ev"]["challenge"] - d["ev"]["challenge"]) if d.get("ev") else None)})
    return res


def replay_cli(world: str, seq: int, variants: list[str], root: str = "runs/local") -> list[dict]:
    wd = Path(world) if Path(world).exists() else Path(root) / world
    rows = []
    print(f"{'variant':<28} {'agent':<17} {'q_used':>7} {'p_final':>8} {'action':<10} {'flip':<5} {'dp':>7}  q_source")
    for s in variants:
        r = replay_decision(wd, seq, s)
        rows.append(r)
        print(f"{s:<28} {r['agent']:<17} {r['q_used']:>7.3f} {r['p_final']:>8.3f} {r['action']:<10} "
              f"{str(r['flipped']):<5} {r['delta_p']:>+7.3f}  {r['q_source']}")
    lg = rows[0]["logged"] if rows else {}
    print(f"{'(logged)':<28} {'':<17} {lg.get('q_used') or 0:>7.3f} {lg.get('p_final') or 0:>8.3f} {lg.get('action')}")
    return rows


def branch(world_dir: str | Path, new_wid: str, from_game: int, games: int | None = None,
           variant: dict | str = "full", learning: bool = False, out_dir: str | Path | None = None,
           reflector=None):
    """Restore ckpt g<from_game> (engine + RNG + memory), apply the variant to memory-bearing seats,
    and return a World that plays games from_game+1..games on the same seeds."""
    from blab.world import World
    world_dir = Path(world_dir)
    v = parse_variant(variant)
    meta = json.loads((world_dir / "meta.json").read_text())
    ck = _load_ckpt(world_dir, from_game)
    ck = copy.deepcopy(ck)
    seats = copy.deepcopy(meta["seats"])
    for s, mem in list(ck.get("memory", {}).items()):
        ck["memory"][s] = apply_variant(mem, v, world_dir=world_dir, seat=s, at_game=from_game, wid=new_wid)
    if v.get("agent"):
        for sp in seats:
            if sp["seat"] == "p1":
                sp["policy"] = v["agent"]
    return World(wid=new_wid, seats=seats, seed=meta["seed"], games=games or meta["games"],
                 reflect_every=meta.get("reflect_every", 2), ask=meta.get("ask", False),
                 out_dir=out_dir if out_dir is not None else world_dir.parent, run=meta.get("run", "local"),
                 box=meta.get("box", "local"), variant=v, learning=learning, parent_world=meta["wid"],
                 from_game=from_game, start_ckpt=ck, table=meta.get("table"), reflector=reflector)


def branch_cli(world: str, from_game: int = 10, variant: str = "full", root: str = "runs/local",
               new_wid: str | None = None) -> dict:
    wd = Path(world) if Path(world).exists() else Path(root) / world
    tag = variant.replace(":", "-")
    nw = new_wid or f"{wd.name}.b{from_game}.{tag}"
    w = branch(wd, nw, from_game, variant=variant)
    res = asyncio.run(w.run()) if not _in_loop() else None
    print(json.dumps({"wid": nw, "dir": str(w.dir), **(res or {})}, indent=1))
    return res


def _in_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False
