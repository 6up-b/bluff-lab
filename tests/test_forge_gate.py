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


def test_class_bases_escape_rejected():
    g = gate("x = ().__class__.__bases__[0].__subclasses__()\n" + GOOD)
    assert not g["ok"] and "__" in g["reason"]


def test_private_and_module_internal_attributes_rejected():
    for bad in ("import random\nrandom._os.system('id')", "import statistics\nstatistics.sys.modules",
                "s = '{0.__class__}'.format(1)", "def g():\n    yield 1\nf = g().gi_frame"):
        assert not gate(bad + "\n" + GOOD)["ok"], bad


def test_child_policy_fallbacks_counted():
    from blab.forge_eval import ChildPolicy
    obs = {"my_hand": ["K", "Q"], "legal_actions": ["challenge", "play"], "pending_claim": {"by": "p3"}}
    slow = "def decide(obs):\n    x = 0\n    for i in range(10**9):\n        x += i\n    return {'type': 'challenge'}\n"
    pol = ChildPolicy(slow, timeout_ms=50)
    try:
        assert pol.act(obs) == {"type": "challenge"} and pol.timeouts == 1
    finally:
        pol.close()
    bad = "def decide(obs):\n    return {'type': 'play', 'cards': [7]}\n"
    pol = ChildPolicy(bad, timeout_ms=50)
    try:
        assert pol.act(obs)["type"] == "challenge" and pol.illegal == 1
    finally:
        pol.close()
    boom = "def decide(obs):\n    return 1 / 0\n"
    pol = ChildPolicy(boom, timeout_ms=50)
    try:
        assert pol.act(obs)["type"] == "challenge" and pol.crashes == 1
    finally:
        pol.close()
