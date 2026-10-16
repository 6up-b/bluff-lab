from blab.forge_eval import gate

GOOD = '''
import math
from collections import Counter

def decide(obs):
    if obs.get("pending_claim"):
        p = obs["pending_claim"]["features"].get("card_math", {}).get("p_cards", 0.5)
        if p < 0.4:
            return {"type": "challenge"}
    return {"type": "play", "cards": [0], "utterance": "One card."}
'''


def test_good_passes():
    assert gate(GOOD) == {"ok": True, "reason": "ok"}


def test_import_os_rejected():
    g = gate("import os\n" + GOOD)
    assert not g["ok"] and "os" in g["reason"]


def test_from_import_rejected():
    assert not gate("from subprocess import run\n" + GOOD)["ok"]
    assert not gate("from . import x\n" + GOOD)["ok"]


def test_forbidden_names_and_dunders():
    for bad in ("x = open('f')", "eval('1')", "getattr(obj, 'a')", "y = ().__class__", "__import__('os')",
                "exec('1')", "z = globals()"):
        assert not gate(bad + "\n" + GOOD)["ok"], bad


def test_must_define_decide():
    assert not gate("def act(obs):\n    return {}\n")["ok"]


def test_syntax_and_size():
    assert not gate("def decide(:\n")["ok"]
    assert not gate(GOOD + "#" * 21000)["ok"]
