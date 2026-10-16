import asyncio
import subprocess
import sys

import numpy as np

from blab.agents.ours import DIM, I_FORMAL, LogregBow, featurize, stable_bucket
from blab.events import read_events, validate_event
from blab.world import World, make_seats, replay_decision, run_local

SKIP = {"replay_of", "variant", "logged", "delta_p", "flipped", "delta_ev", "trace"}


def test_hash_is_stable_across_processes():
    here = [stable_bucket(t) for t in ("i am", "that's", "kings")]
    code = "from blab.agents.ours import stable_bucket as b; print([b(t) for t in ('i am', \"that's\", 'kings')])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == str(here)


def test_featurize_shape_and_tokens():
    x = featurize({"utterance": "I am playing two Kings.", "formal_register": True, "hand_size": 2}, 0.3)
    assert x.shape == (DIM,) and x[0] == 1.0 and x[I_FORMAL] == 1.0
    assert 6 <= x[I_FORMAL + 1:].sum() <= 9          # 5 unigrams + 4 bigrams (collisions allowed)
    assert abs(x[1] - np.log(0.3 / 0.7)) < 1e-12


def _entry(i, formal, bluff, p=0.2):
    utt = "I have placed two Kings." if formal else "That's two Kings."
    return {"obs_id": f"w:g{i // 4 + 1}r{i % 4 + 1}", "game": i // 4 + 1, "round": i % 4 + 1, "by": "p3",
            "challenger": "p1", "count": 2, "was_bluff": bluff, "cards": [], "utterance": utt, "p_cards": p,
            "features": {"utterance": utt, "formal_register": formal, "hand_size": 3, "claim_count": 2}}


def test_learns_formal_tell_coefficient():
    rng = np.random.default_rng(0)
    ag = LogregBow("p1", None, {"wid": "w", "seed": 0})
    entries = []
    for i in range(160):
        formal = bool(rng.random() < 0.5)
        entries.append(_entry(i, formal, bool(rng.random() < (0.7 if formal else 0.1))))
    for g in range(1, 41):
        obs = {"revealed_history": [e for e in entries if e["game"] == g]}
        ag.on_game_end(obs, g)
    m = ag.memory["logreg"]["p3"]
    c, lo, hi = m["tell_coef"]
    assert c > 0 and lo > 0 and lo < c < hi
    assert m["discovery_game"] is not None
    f = {"utterance": "I have placed two Kings.", "formal_register": True, "hand_size": 3, "claim_count": 2,
         "card_math": {"p_cards": 0.2}}
    f2 = {"utterance": "That's two Kings.", "formal_register": False, "hand_size": 3, "claim_count": 2,
          "card_math": {"p_cards": 0.2}}
    assert ag.predict({}, "p3", f, 0.2)[3] > ag.predict({}, "p3", f2, 0.2)[3]


def test_world_runs_validates_and_replays_exactly(tmp_path):
    asyncio.run(run_local(table="telltale", agent="logreg_bow", seed=5, games=10, out=str(tmp_path), llm=False,
                          ask=False, wid="wl", quiet=True))
    evs = read_events(tmp_path / "wl" / "events.jsonl")
    for e in evs:
        validate_event(e)
    ds = [e for e in evs if e["type"] == "decision" and e["payload"]["kind"] == "respond"]
    assert ds and all(d["payload"]["q_source"] == "logreg" and "tell_coef" in d["payload"]["logreg"] for d in ds)
    for e in ds[::3]:
        r = replay_decision(tmp_path / "wl", e["seq"], "full")
        assert {k: v for k, v in r.items() if k not in SKIP} == {k: v for k, v in e["payload"].items() if k != "trace"}
    late = [d for d in ds if d["game"] >= 5]
    assert any(d["payload"]["logreg"]["n_fit"] > 0 for d in late)


def test_ev_rule_runs():
    seats = make_seats("telltale", "logreg_bow", 2)
    seats[0]["params"]["respond_rule"] = "ev"
    w = World(wid="we", seats=seats, seed=2, games=4, out_dir=None)
    asyncio.run(w.run())
    d = next(e for e in w.events.events if e["type"] == "decision" and e["payload"]["kind"] == "respond")
    assert d["payload"]["respond_rule"] == "ev"
