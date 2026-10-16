"""b01: tell_relational wording independent of bluffing, statement truth rates; engine auto-ask; logreg_embed."""
import asyncio
import re

import numpy as np

from blab.agents import banks
from blab.agents.scripted import Scripted


def _truth(obs, n, utt):
    rank = obs["table_rank"]
    m = re.search(r"(\d+) (King|Queen|Ace)s", utt)
    if m and "round" in utt and "last round" not in utt and "previous round" not in utt:
        N = sum(c["count"] for c in obs["round_claims"]) + n
        return int(m.group(1)) == N
    m = re.search(r"(\d+) cards", utt)
    if m:
        return int(m.group(1)) == len(obs["my_hand"]) - n
    return None


def test_relational_truth_rates_and_wording_independent():
    rng = np.random.default_rng(0)
    sc = Scripted("p3", rng, {"role": "tell_relational"})
    stats = {True: [0, 0], False: [0, 0]}
    formal = {True: [0, 0], False: [0, 0]}
    for i in range(6000):
        bluff = bool(i % 2)
        obs = {"table_rank": "K", "game": 1, "round": i + 1, "my_hand": ["K"] * 5,
               "round_claims": [{"by": "p2", "count": 2, "utterance": ""}], "pending_claim": None}
        utt = sc.relational_utterance(obs, 1, bluff)
        claim = utt.split(". ")[0] + "."
        is_formal = any(banks.render(t, 1, "K") == claim for t in banks.FORMAL)
        formal[bluff][0] += is_formal
        formal[bluff][1] += 1
        t = _truth(obs, 1, utt)
        if t is not None:
            stats[bluff][0] += (not t)
            stats[bluff][1] += 1
    assert abs(stats[True][0] / stats[True][1] - 0.7) < 0.04
    assert abs(stats[False][0] / stats[False][1] - 0.1) < 0.03
    assert abs(formal[True][0] / formal[True][1] - formal[False][0] / formal[False][1]) < 0.04


def test_engine_asks_every_answer_tell_claim(tmp_path):
    from blab.world import World, make_seats, stub_reflector
    seats = make_seats("answer_tell", "logreg_embed", 4)
    seats[0]["params"].update(info="raw")
    w = World(wid="t", seats=seats, seed=4, games=2, out_dir=tmp_path, reflector=stub_reflector, table="answer_tell")
    asyncio.run(w.run())
    import json
    evs = [json.loads(l) for l in open(tmp_path / "t" / "events.jsonl")]
    claims = [e for e in evs if e["type"] == "claim" and e["actor"] == "p3"]
    asks = [e for e in evs if e["type"] == "ask"]
    answers = [e for e in evs if e["type"] == "answer"]
    assert claims and len(asks) == len(claims) == len(answers)
    assert all(a["payload"]["text"] in banks.ANSWER_EVASIVE + banks.ANSWER_PLAIN for a in answers)


def test_embed_cache_covers_enumeration():
    from blab.agents.embed import enumerate_texts, projected
    for s in enumerate_texts():
        assert len(projected(s)) == 64
