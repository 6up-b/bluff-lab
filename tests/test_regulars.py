"""reg01 (PREREG_reg01.md): schedule, Blake's role, forget:<name>, checkpoint-branch exactness."""
import asyncio
import json

import numpy as np

from blab.agents.scripted import Scripted
from blab.engine import formal_register
from blab.experiments import regulars as R


def test_schedule_blocks_and_no_back_to_back():
    for seed in range(8000, 8024):
        s = R.schedule(seed)
        assert len(s) == 20
        for b in range(5):
            assert sorted(s[4 * b:4 * b + 4]) == sorted(R.NAMES)
        assert all(a != b for a, b in zip(s, s[1:]))
        assert s == R.schedule(seed)


def test_reverse_telltale_is_avery_mirrored():
    def rate(role, bluff):
        sc = Scripted("p3", np.random.default_rng(1), {"role": role})
        return np.mean([formal_register(sc.utterance(1, "K", bluff)) for _ in range(4000)])
    assert abs(rate("reverse_telltale", False) - 0.75) < 0.03 and abs(rate("reverse_telltale", True) - 0.25) < 0.03
    assert abs(rate("telltale", True) - 0.75) < 0.03 and abs(rate("telltale", False) - 0.25) < 0.03


def test_forget_removes_only_that_name():
    mem = {"memory_version": 3, "hypotheses": [{"about": "Avery", "id": "a"}, {"about": "Casey", "id": "c"}],
           "revealed": [{"by": "Avery", "obs_id": 1}, {"by": "Casey", "obs_id": 2}, {"by": "Avery", "obs_id": 3}],
           "observations": [1, 2, 3], "reflected_upto": 2, "base": {"Avery": {}, "Casey": {}},
           "plus": {"Avery": {}, "Casey": {}}, "drift_notes": [{"about": "Avery"}]}
    m, c = R.forget(mem, "Avery")
    assert [h["id"] for h in m["hypotheses"]] == ["c"] and m["observations"] == [2]
    assert m["revealed"] == [{"by": "Casey", "obs_id": 2}] and m["reflected_upto"] == 1
    assert "Avery" not in m["base"] and "Avery" not in m["plus"] and "Casey" in m["plus"] and m["drift_notes"] == []
    assert c["hypotheses"] == 1 and c["revealed"] == 2


def test_branch_full_is_exact(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "GAMES", 9)
    monkeypatch.setattr(R, "FORGET_GAME", 6)
    asyncio.run(R.run_season("logreg_identity", 9901, tmp_path))
    asyncio.run(R.run_branch(9901, None, tmp_path, parent_arm="logreg_identity"))

    def decs(w):
        out = []
        for line in open(tmp_path / w / "events.jsonl"):
            e = json.loads(line)
            if e["type"] == "decision" and e["actor"] == "p1" and e["game"] > 6:
                out.append((e["game"], e["round"], e["payload"].get("p_final"), e["payload"]["action"]))
        return out
    a, b = decs("s9901-logreg_identity"), decs("s9901-logreg_identity-full")
    assert a and a == b
