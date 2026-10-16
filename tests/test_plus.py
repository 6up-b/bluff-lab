import asyncio

import pytest

from blab.events import read_events, validate_event
from blab.world import World, make_seats, replay_decision, run_local

SKIP = {"replay_of", "variant", "logged", "delta_p", "flipped", "delta_ev", "trace"}


@pytest.fixture(scope="module")
def pw(tmp_path_factory):
    out = tmp_path_factory.mktemp("plus")
    asyncio.run(run_local(table="telltale", agent="profiler_plus", seed=101, games=14, out=str(out), llm=False,
                          ask=False, wid="wp", quiet=True))
    return out / "wp"


def _resp(wd, g0=1):
    return [e for e in read_events(wd / "events.jsonl")
            if e["type"] == "decision" and e["payload"]["kind"] == "respond" and e["game"] >= g0]


def test_plus_valid_and_replay_exact(pw):
    evs = read_events(pw / "events.jsonl")
    for e in evs:
        validate_event(e)
    assert sum(e["type"] == "reflect_request" for e in evs) == 14          # reflects every game
    ds = _resp(pw)
    for d in ds[::5]:
        r = replay_decision(pw, d["seq"], "full")
        assert {k: v for k, v in r.items() if k not in SKIP} == {k: v for k, v in d["payload"].items() if k != "trace"}
    late = _resp(pw, 10)
    assert any(m.get("coef") is not None for d in late for m in d["payload"]["memory_used"])


def test_plus_variants(pw):
    d = [e for e in _resp(pw, 10) if e["payload"]["target"] == "p3"
         and any(m["matched"] and m.get("coef") for m in e["payload"]["memory_used"])][0]
    r0 = replay_decision(pw, d["seq"], "as_of:0")
    assert abs(r0["p_final"] - d["payload"]["card_math"]["p_cards"]) < 1e-3 or r0["memory_used"] == []
    fm = replay_decision(pw, d["seq"], "false_memory")
    assert fm["p_final"] == d["payload"]["p_final"]                            # untrusted line: no feature
    asp = next(m["aspect"] for m in d["payload"]["memory_used"] if m["matched"] and m.get("coef"))
    mi = replay_decision(pw, d["seq"], f"minus:{asp}")
    assert all(m["aspect"] != asp for m in mi["memory_used"])


def test_plus_raw_and_new_tables():
    for t in ("tell_conjunctive", "tell_flipper"):
        seats = make_seats(t, "profiler_plus", 2)
        w = World(wid="x", seats=seats, seed=2, games=12, out_dir=None, table=t)
        r = asyncio.run(w.run())
        assert r["games"] == 12
