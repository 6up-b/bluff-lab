"""llm-arms: prompts differ only in the memory block; replay-from-cache is exact; errors fall back to p_cards.
No network: `estimate_sync` is monkeypatched."""
import asyncio
import json

import numpy as np
import pytest

from blab.agents import llm_agents as L
from blab.agents.ours import AGENTS
from blab.world import World, make_seats

CALLS = []


def fake_estimate(user: str) -> dict:
    CALLS.append(user)
    h = int(L.prompt_hash(L.SYSTEM, user)[:8], 16)
    return {"p_bluff": (h % 1000) / 1000, "reason": "fake", "provider": "fake", "model": "fake",
            "request_id": f"r{h}", "latency_ms": 1}


async def _fake_reflector(wid, seat, payload):
    return {"proposed": [{"about": "p3", "aspect": "baseline_bluff", "text": "p3 bluffs more on multi-card claims",
                          "predicate": {"f": "claim_count", "op": ">=", "v": 2}, "rationale": "t",
                          "supersedes_id": None}], "source": "test", "llm_trace_id": None, "surreal_trace_id": None}


def _run(arm, tmp_path, games=3, seed=11):
    seats = make_seats("telltale", arm, seed)
    seats[0]["params"] = {"info": "raw", "llm_cache": str(tmp_path / f"{arm}-s{seed}" / "llm_cache.jsonl")}
    w = World(wid=f"{arm}-s{seed}", seats=seats, seed=seed, games=games, out_dir=tmp_path, reflector=_fake_reflector,
              table="telltale")
    L.cache_reset()
    asyncio.run(w.run())
    return w


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(L, "estimate_sync", fake_estimate)
    CALLS.clear()


def test_registered():
    for a in ("llm_nomem", "llm_episodic", "llm_dossier"):
        assert a in AGENTS


def test_prompt_differs_only_in_memory_block(tmp_path):
    w = _run("llm_dossier", tmp_path)
    ev = [e for e in w.events.events if e["type"] == "decision" and e["payload"]["kind"] == "respond"
          and not e["payload"]["forced"] and e["game"] == 3]
    d = ev[0]["payload"]
    mem = w.ckpts[2]["memory"]["p1"]
    users = {}
    for arm in ("llm_nomem", "llm_episodic", "llm_dossier"):
        pol = AGENTS[arm]("p1", np.random.default_rng(0), {"info": "raw"})
        if hasattr(pol, "set_memory"):
            pol.set_memory(mem)
        CALLS.clear()
        pol.decide_respond(d["obs"], d["rng_u"])
        users[arm] = CALLS[-1]
    heads = {a: u.split("## Memory about")[0] for a, u in users.items()}
    assert len(set(heads.values())) == 1                       # identical observation text
    assert len(set(users.values())) == 3                       # memory blocks differ
    assert "formal_register" not in users["llm_dossier"]       # raw information only


def test_replay_full_is_exact_from_cache(tmp_path):
    from blab.replay import replay_decision
    w = _run("llm_episodic", tmp_path)
    ev = [e for e in w.events.events if e["type"] == "decision" and e["payload"]["kind"] == "respond"
          and not e["payload"]["forced"] and e["game"] >= 2]
    n_before = len(CALLS)
    for e in ev[:5]:
        r = replay_decision(w.dir, e["seq"], "full")
        assert r["p_final"] == e["payload"]["p_final"] and not r["flipped"]
        assert r["llm"]["cached"] is True
    assert len(CALLS) == n_before                              # no fresh calls on replay(full)


def test_error_falls_back_to_p_cards(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "estimate_sync", lambda u: {"error": "timeout after 30s", "provider": "x", "latency_ms": 30000})
    w = _run("llm_nomem", tmp_path, games=1)
    ds = [e["payload"] for e in w.events.events if e["type"] == "decision" and e["payload"]["kind"] == "respond"
          and not e["payload"]["forced"]]
    assert ds and all(d["llm"]["error"] and d["p_final"] == d["card_math"]["p_cards"] for d in ds)
    assert all(d["q_used"] is None for d in ds)
