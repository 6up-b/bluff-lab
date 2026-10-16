import asyncio
import json

import pytest

from blab.agents.ours import make_agent
from blab.events import canonical, read_events
from blab.world import branch, replay_decision, run_local

SKIP_KEYS = {"replay_of", "variant", "logged", "delta_p", "flipped", "delta_ev", "trace"}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    asyncio.run(run_local(table="telltale", agent="profiler", seed=7, games=14, reflect_every=2, out=str(out),
                          llm=False, ask=False, wid="wr", quiet=True))
    return out / "wr"


def _decisions(wd, kind="respond", min_game=1):
    return [e for e in read_events(wd / "events.jsonl")
            if e["type"] == "decision" and e["payload"]["kind"] == kind and e["game"] >= min_game]


def test_replay_full_equals_logged(world):
    ds = _decisions(world)
    assert len(ds) > 50
    for e in ds:
        r = replay_decision(world, e["seq"], "full")
        logged = {k: v for k, v in e["payload"].items() if k != "trace"}
        got = {k: v for k, v in r.items() if k not in SKIP_KEYS}
        assert got == logged, e["seq"]
        assert r["delta_p"] == 0 and not r["flipped"]
    for e in _decisions(world, "play")[:20]:
        r = replay_decision(world, e["seq"], "full")
        assert {k: v for k, v in r.items() if k not in SKIP_KEYS} == {k: v for k, v in e["payload"].items() if k != "trace"}


def test_as_of_0_equals_amnesic(world):
    ds = _decisions(world, min_game=8)
    changed = 0
    for e in ds:
        r = replay_decision(world, e["seq"], "as_of:0")
        am = make_agent("profiler_amnesic", e["actor"], None, {"wid": "wr", "seed": 7})
        a = am.decide_respond(e["payload"]["obs"], e["payload"]["rng_u"])
        assert (r["p_final"], r["q_used"], r["q_source"], r["action"]) == (a["p_final"], a["q_used"], a["q_source"], a["action"])
        changed += abs(r["delta_p"]) > 1e-9
    assert changed > 0          # memory matters: as_of:0 changes p


def test_false_memory_ignored_by_profiler_exploits_blind(world):
    ds = [e for e in _decisions(world, min_game=8) if e["payload"]["target"] == "p3"]
    blind_moved = 0
    for e in ds:
        r = replay_decision(world, e["seq"], "false_memory")
        assert r["p_final"] == e["payload"]["p_final"]          # untrusted line has weight 0
        assert any(m["used_as"] == "ignored_untrusted" for m in r["memory_used"])
        b = replay_decision(world, e["seq"], "false_memory@profiler_blind")
        blind_moved += b["q_used"] == 0.0
    assert blind_moved == len(ds)


def _norm(evs, wid_from, wid_to, min_game, parent=None):
    out = []
    for e in canonical(evs):
        if e["game"] < min_game or e["type"] in ("world_start", "world_end"):
            continue
        e = {k: v for k, v in e.items() if k not in ("seq", "world")}
        t = json.dumps(e).replace(f'"{wid_from}', f'"{wid_to}')
        if parent:   # branch memory keeps evidence ids that point into the parent world
            t = t.replace(f'"{parent}:', f'"{wid_to}:')
        out.append(json.loads(t))
    return out


def test_branch_full_reproduces_when_frozen(tmp_path):
    asyncio.run(run_local(table="telltale", agent="profiler", seed=11, games=14, out=str(tmp_path), llm=False,
                          ask=False, wid="wf", quiet=True, learning=False))
    w = branch(tmp_path / "wf", "wf_b", 10, variant="full", learning=False)
    asyncio.run(w.run())
    a = _norm(read_events(tmp_path / "wf" / "events.jsonl"), "wf", "X", 11)
    b = _norm(read_events(tmp_path / "wf_b" / "events.jsonl"), "wf_b", "X", 11, parent="wf")
    assert a and a == b
    ws = read_events(tmp_path / "wf_b" / "events.jsonl")[0]
    assert ws["payload"]["parent_world"] == "wf" and ws["payload"]["from_game"] == 10
    assert ws["payload"]["variant"] == {"type": "full"}


def test_branch_from_learning_world_first_game_identical(world, tmp_path):
    w = branch(world, "wr_b", 10, variant="full", learning=False, out_dir=tmp_path)
    asyncio.run(w.run())
    a = _norm(read_events(world / "events.jsonl"), "wr", "X", 11)
    b = _norm(read_events(tmp_path / "wr_b" / "events.jsonl"), "wr_b", "X", 11, parent="wr")
    g11a = [e for e in a if e["game"] == 11 and e["type"] not in ("hypothesis", "belief_superseded",
                                                                  "reflect_request", "reflect_result")]
    g11b = [e for e in b if e["game"] == 11]
    assert g11a == g11b
    # variants run too
    for v in ("as_of:0", "minus:verbal_tell", "false_memory"):
        bw = branch(world, f"wr_{v.replace(':', '-')}", 10, variant=v, out_dir=tmp_path)
        res = asyncio.run(bw.run())
        assert res["games"] == 4


def test_ask_world_replays_exactly(tmp_path):
    asyncio.run(run_local(table="telltale", agent="profiler", seed=3, games=8, out=str(tmp_path), llm=False,
                          ask=True, wid="wa", quiet=True))
    evs = read_events(tmp_path / "wa" / "events.jsonl")
    asks = [e for e in evs if e["type"] == "ask"]
    assert asks and all(e["type"] == "answer" for e in evs if e["seq"] - 1 in {a["seq"] for a in asks})
    asked = [e for e in _decisions(tmp_path / "wa") if e["payload"].get("ask")]
    assert asked
    for e in asked:
        r = replay_decision(tmp_path / "wa", e["seq"], "full")
        assert {k: v for k, v in r.items() if k not in SKIP_KEYS} == {k: v for k, v in e["payload"].items() if k != "trace"}
