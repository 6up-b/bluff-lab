import asyncio
import json
import math

import numpy as np
import pytest

from blab import hypotheses as H
from blab.agents import banks
from blab.agents.scripted import Scripted
from blab.engine import Engine, card_math, hypergeom_sf, is_bluff
from blab.events import canonical, read_events, validate_event
from blab.world import World, make_seats, run_local


def _start(seed=1):
    e = Engine(seed, wid="wt")
    e.start_game(1)
    e.start_round()
    return e


# ---------------------------------------------------------------- rules
def test_bluff_rule():
    assert not is_bluff(["K", "J"], "K")
    assert not is_bluff(["J"], "Q")
    assert is_bluff(["K", "Q"], "K")
    assert is_bluff(["A"], "K")


def test_round_and_challenge_resolution():
    e = _start(3)
    a = e.to_act
    assert e.legal_actions(a) == ["play"]
    rank = e.table_rank
    hand = e.hands[a]
    bad = [i for i, c in enumerate(hand) if c not in (rank, "J")]
    idx = [bad[0]] if bad else [0]
    evs = e.apply(a, {"type": "play", "cards": idx, "utterance": "That's one King."})
    assert [t for t, _, _ in evs] == ["claim"]
    b = e.to_act
    assert b == e.next_living(a)
    assert set(e.legal_actions(b)) == {"challenge", "play"}
    # only the immediate next actor may act
    with pytest.raises(AssertionError):
        e.apply(e.next_living(b), {"type": "challenge"})
    evs = e.apply(b, {"type": "challenge"})
    types = [t for t, _, _ in evs]
    assert types == ["challenge", "reveal", "life_lost"]
    rv = evs[1][1]
    assert rv["was_bluff"] == is_bluff(rv["cards"], rank)
    assert rv["loser"] == (a if rv["was_bluff"] else b)
    assert e.lives[rv["loser"]] == 2
    assert not e.round_active


def test_empty_hand_must_challenge():
    e = _start(5)
    a = e.to_act
    b = e.next_living(a)
    e.apply(a, {"type": "play", "cards": [0], "utterance": "x"})
    e.hands[b] = []
    assert e.legal_actions(b) == ["challenge"]
    evs = e.apply(b, {"type": "play", "cards": [0], "utterance": "y"})   # illegal → fallback challenge
    assert evs[0][0] == "challenge"


def test_game_ends_with_one_survivor():
    e = Engine(11, wid="wt")
    e.start_game(1)
    pols = {s: Scripted(s, e.seat_rngs[s], {"role": "control"}) for s in e.seats}
    n = 0
    while not e.game_over:
        e.start_round()
        while e.round_active:
            s = e.to_act
            e.apply(s, pols[s].act(e.observation(s)))
        n += 1
    assert e.winner is None or sum(v > 0 for v in e.lives.values()) == 1 or n == 30


# ---------------------------------------------------------------- card math
def test_hypergeom_and_p_cards():
    # direct formula check
    U, m, h, k = 15, 6, 4, 2
    p = sum(math.comb(m, x) * math.comb(U - m, h - x) for x in range(k, min(h, m) + 1)) / math.comb(U, h)
    assert abs(hypergeom_sf(U, m, h, k) - p) < 1e-12
    # I hold all 8 matching cards -> claimant cannot be truthful
    cm = card_math(["K"] * 5, "K", 5, 1)
    assert cm["U"] == 15 and cm["m"] == 3
    cm0 = card_math(["K", "K", "K", "K", "K"], "K", 4, 1)
    assert 0 <= cm0["p_cards"] <= 1
    full = card_math(["K", "K", "K", "J", "J"], "K", 5, 4)   # m=3 < k=4 -> impossible
    assert full["p_cards"] == 1.0
    # more claimed cards -> less likely truthful
    hand = ["Q", "A", "A", "Q", "K"]
    ps = [card_math(hand, "K", 5, k)["p_cards"] for k in (1, 2, 3)]
    assert ps[0] < ps[1] < ps[2]


# ---------------------------------------------------------------- hidden info
def test_observation_never_leaks_other_hands():
    w = World(wid="wleak", seats=make_seats("telltale", "profiler", 3), seed=3, games=3, out_dir=None)
    eng = w.engine
    leaks = []
    orig = eng.observation

    def spy(seat):
        o = orig(seat)
        # other seats' current hands must not appear; revealed cards are public and allowed
        allowed = {"my_hand", "revealed_history"}
        s = json.dumps({k: v for k, v in o.items() if k not in allowed})
        for other in eng.seats:
            if other != seat and eng.hands[other]:
                assert "hands" not in o
                # the concrete hand list must not be serialized anywhere outside revealed_history
                if json.dumps(eng.hands[other]) in s and len(eng.hands[other]) > 1:
                    leaks.append((seat, other))
        assert set(o) == {"seat", "world", "game", "round", "table_rank", "my_hand", "lives", "hand_sizes",
                          "pending_claim", "round_claims", "revealed_history", "legal_actions"}
        if o["pending_claim"]:
            assert "cards" not in o["pending_claim"]
        for rc in o["round_claims"]:
            assert set(rc) == {"by", "count", "utterance"}
        for r in o["revealed_history"]:
            assert "card_math_by_seat" not in r and "lives_at_claim" not in r
        return o

    eng.observation = spy
    asyncio.run(w.run())
    assert not leaks


# ---------------------------------------------------------------- banks
def test_banks_register():
    for t in banks.CASUAL:
        for n in (1, 2, 3):
            for r in ("K", "Q", "A"):
                u = banks.render(t, n, r)
                assert H.has_contraction(u) and not H.formal_register(u), u
    for t in banks.FORMAL:
        for n in (1, 2, 3):
            for r in ("K", "Q", "A"):
                u = banks.render(t, n, r)
                assert not H.has_contraction(u) and H.formal_register(u), u
    assert len(banks.CASUAL) >= 8 and len(banks.FORMAL) >= 8
    for u in banks.ANSWER_FORMAL:
        assert H.formal_register(u)
    for u in banks.ANSWER_CASUAL:
        assert H.has_contraction(u)


# ---------------------------------------------------------------- determinism
def test_snapshot_restore_roundtrip():
    e = _start(9)
    a = e.to_act
    e.apply(a, {"type": "play", "cards": [0, 1], "utterance": "I am playing two Kings."})
    snap = json.loads(json.dumps(e.snapshot()))
    e2 = Engine.restore(snap)
    assert json.dumps(e2.snapshot(), sort_keys=True) == json.dumps(e.snapshot(), sort_keys=True)
    assert e2.observation(e2.to_act) == e.observation(e.to_act)
    # RNG streams continue identically
    assert e.deal_rng.random() == e2.deal_rng.random()
    assert e.seat_rngs["p2"].random() == e2.seat_rngs["p2"].random()


def _run(seed, agent="profiler", table="telltale", games=6):
    w = World(wid="wd", seats=make_seats(table, agent, seed), seed=seed, games=games, out_dir=None, table=table)
    asyncio.run(w.run())
    return w


def test_same_seed_identical_log():
    a = canonical(_run(4).events.events)
    b = canonical(_run(4).events.events)
    assert a == b
    c = canonical(_run(5).events.events)
    assert a != c


def test_local_world_20_games_valid(tmp_path):
    res = asyncio.run(run_local(table="telltale", agent="profiler", seed=7, games=20, reflect_every=2,
                                out=str(tmp_path), llm=False, ask=False, wid="w_t", quiet=True))
    evs = read_events(tmp_path / "w_t" / "events.jsonl")
    for e in evs:
        validate_event(e)
    types = {e["type"] for e in evs}
    for t in ("world_start", "game_start", "deal", "claim", "decision", "challenge", "play_on", "reveal",
              "life_lost", "game_end", "reflect_request", "reflect_result", "hypothesis", "world_end"):
        assert t in types, t
    assert [e["seq"] for e in evs] == list(range(1, len(evs) + 1))
    assert sum(e["type"] == "game_end" for e in evs) == 20
    assert (tmp_path / "w_t" / "ckpt" / "g00.json").exists() and (tmp_path / "w_t" / "ckpt" / "g20.json").exists()
    assert (tmp_path / "w_t" / "memory" / "p1.json").exists() and (tmp_path / "w_t" / "meta.json").exists()
    assert 0 <= res["win_rate"] <= 1
    # every respond decision of p1 has the §7.1 core fields
    for e in evs:
        if e["type"] == "decision" and e["payload"]["kind"] == "respond":
            p = e["payload"]
            assert p["card_math"] and p["rng_u"] is not None and p["obs"] and p["agent"] == "profiler"
            assert abs(p["p_final"] - (p["card_math"]["p_cards"] + (1 - p["card_math"]["p_cards"]) * p["q_used"])) < 1e-9


@pytest.mark.parametrize("agent", ["counter", "bayes", "bayes_discount", "profiler_amnesic", "profiler_blind"])
def test_all_agents_run(agent):
    w = _run(2, agent=agent, games=4)
    assert w.result["games"] == 4


@pytest.mark.parametrize("table", ["longcon", "falsealarm", "telltale_overt", "collude"])
def test_tables_run(table):
    w = _run(2, table=table, games=9)
    if table == "longcon":
        assert "rtd" in w.result


def test_reflection_timeout_keeps_memory():
    async def slow(wid, seat, payload):
        await asyncio.sleep(5)

    w = World(wid="wto", seats=make_seats("telltale", "profiler", 1), seed=1, games=2, out_dir=None,
              reflector=slow, reflect_timeout=0.05)
    asyncio.run(w.run())
    rr = [e for e in w.events.events if e["type"] == "reflect_result"]
    assert rr and rr[0]["payload"]["source"] == "timeout" and rr[0]["payload"]["error"]
    assert w.policies["p1"].memory["hypotheses"] == []
