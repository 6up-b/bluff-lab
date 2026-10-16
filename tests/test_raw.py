import asyncio
import json

from blab import hypotheses as H
from blab.agents.common import public_obs_text, strip_obs
from blab.events import read_events, validate_event
from blab.world import World, make_seats, replay_decision, stub_reflector

SKIP = {"replay_of", "variant", "logged", "delta_p", "flipped", "delta_ev", "trace"}


def _raw_world(agent, tmp_path, wid, table="telltale", games=6, reflector=None, ask=False):
    seats = make_seats(table, agent, 3)
    seats[0]["params"]["info"] = "raw"
    w = World(wid=wid, seats=seats, seed=3, games=games, out_dir=tmp_path, table=table, reflector=reflector, ask=ask)
    asyncio.run(w.run())
    return w


def test_dsl_raw_mode():
    assert H.validate_predicate({"f": "formal_register", "op": "==", "v": True})[0]
    assert not H.validate_predicate({"f": "formal_register", "op": "==", "v": True}, "raw")[0]
    assert not H.validate_predicate({"f": "utterance", "op": "no_contractions", "v": True}, "raw")[0]
    assert not H.validate_predicate({"all": [{"f": "hand_size", "op": "<=", "v": 2},
                                             {"not": {"f": "formal_register", "op": "==", "v": True}}]}, "raw")[0]
    assert H.validate_predicate({"f": "word_count", "op": ">=", "v": 4}, "raw")[0]
    assert H.validate_predicate({"f": "utterance", "op": "contains", "v": "'"}, "raw")[0]


def test_stub_refuses_raw():
    import pytest
    with pytest.raises(RuntimeError):
        asyncio.run(stub_reflector("w", "p1", {"info": "raw", "hypotheses": []}))


def test_raw_world_hides_tell_features(tmp_path):
    w = _raw_world("profiler", tmp_path, "wr")
    evs = read_events(tmp_path / "wr" / "events.jsonl")
    for e in evs:
        validate_event(e)
    decs = [e for e in evs if e["type"] == "decision" and e["actor"] == "p1"]
    assert decs
    for d in decs:
        s = json.dumps(d["payload"])
        assert "formal_register" not in s and "answer_has_contraction" not in s
    resp = [d for d in decs if d["payload"]["kind"] == "respond"]
    assert all("word_count" in d["payload"]["claim"]["features"] for d in resp)
    # god view keeps ground truth
    assert any("formal_register" in e["payload"]["features"] for e in evs if e["type"] == "claim")
    # stub reflection refused -> no hypotheses learned, errors logged
    rr = [e for e in evs if e["type"] == "reflect_result"]
    assert rr and all(e["payload"]["error"] for e in rr)
    assert w.policies["p1"].memory["hypotheses"] == []
    for d in resp[::4]:
        r = replay_decision(tmp_path / "wr", d["seq"], "full")
        assert {k: v for k, v in r.items() if k not in SKIP} == {k: v for k, v in d["payload"].items() if k != "trace"}


def test_raw_logreg_and_ask(tmp_path):
    w = _raw_world("logreg_bow", tmp_path, "wl", games=5)
    m = w.policies["p1"].memory["logreg"]["p3"]
    assert m["tell_coef"] is None
    evs = read_events(tmp_path / "wl" / "events.jsonl")
    d = [e for e in evs if e["type"] == "decision" and e["payload"]["kind"] == "respond"][-1]
    r = replay_decision(tmp_path / "wl", d["seq"], "full")
    assert {k: v for k, v in r.items() if k not in SKIP} == {k: v for k, v in d["payload"].items() if k != "trace"}
    # interrogation in raw mode: answer text is visible, tell flags are not
    w2 = _raw_world("profiler", tmp_path, "wa", table="answer_tell", games=4, ask=True)
    decs = [e for e in w2.events.events if e["type"] == "decision" and e["payload"].get("ask")]
    assert decs
    for e in decs:
        f = e["payload"]["claim"]["features"]
        assert "answer_utterance" in f and "answer_word_count" in f and "answer_formal_register" not in f


def test_public_obs_text_deterministic():
    from blab.engine import Engine
    e = Engine(5, wid="wt")
    e.start_game(1)
    e.start_round()
    a = e.to_act
    e.apply(a, {"type": "play", "cards": [0], "utterance": "I am playing one King."})
    o = strip_obs(e.observation(e.to_act))
    t1, t2 = public_obs_text(o), public_obs_text(json.loads(json.dumps(o)))
    assert t1 == t2 and "PENDING CLAIM" in t1 and "formal" not in t1.lower()
