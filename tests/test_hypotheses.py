import numpy as np
import pytest

from blab import hypotheses as H
from blab.events import BeliefSuperseded, HypothesisEv, MemoryUsed

FORMAL = {"f": "formal_register", "op": "==", "v": True}


# ---------------------------------------------------------------- synthetic data

def telltale_reveals(n, seed, about="p3", start_game=1, per_game=3, s=0.35, pf_bluff=0.75, pf_honest=0.25,
                     flip=False):
    """Revealed claims from a telltale-like opponent. flip=True inverts the tell."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        game, rnd = start_game + i // per_game, 1 + i % per_game
        p = float(rng.uniform(0.1, 0.3))
        bluff = bool(rng.random() < p or rng.random() < s)
        pf = pf_bluff if bluff else pf_honest
        if flip:
            pf = 1 - pf
        formal = bool(rng.random() < pf)
        utt = "I have placed two Kings." if formal else "That's two Kings."
        out.append({"obs_id": f"w001:g{game}r{rnd}", "game": game, "round": rnd, "by": about, "count": 2,
                    "was_bluff": bluff, "utterance": utt, "p_cards": p,
                    "features": {"formal_register": formal, "hand_size": 3, "claim_count": 2,
                                 "table_rank": "K"}})
    return out


def rate_reveals(n, q, seed, about="p2", start_game=1, per_game=3):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        game, rnd = start_game + i // per_game, 1 + i % per_game
        p = float(rng.uniform(0.1, 0.3))
        bluff = bool(rng.random() < p + (1 - p) * q)
        out.append({"obs_id": f"w001:g{game}r{rnd}", "game": game, "round": rnd, "by": about, "count": 1,
                    "was_bluff": bluff, "utterance": "That's a King.", "p_cards": p,
                    "features": {"formal_register": False, "hand_size": 3}})
    return out


def mk(pred=FORMAL, aspect="verbal_tell", about="p3", idx=1, created_game=0, **kw):
    return H.new_hypothesis(wid="w001", about=about, aspect=aspect, text="t", predicate=pred,
                            created_game=created_game, idx=idx, **kw)


# ---------------------------------------------------------------- DSL

F = {"hand_size": 2, "claim_count": 2, "trusted_streak": 5, "table_rank": "K", "formal_register": True,
     "was_challenged_last_round": False, "utterance": "I have placed two Kings.", "answer_utterance": "I am sure."}


@pytest.mark.parametrize("pred,exp", [
    ({"f": "hand_size", "op": "==", "v": 2}, True),
    ({"f": "hand_size", "op": "!=", "v": 2}, False),
    ({"f": "hand_size", "op": "<", "v": 3}, True),
    ({"f": "hand_size", "op": "<=", "v": 1}, False),
    ({"f": "trusted_streak", "op": ">", "v": 4}, True),
    ({"f": "trusted_streak", "op": ">=", "v": 6}, False),
    ({"f": "table_rank", "op": "==", "v": "K"}, True),
    ({"f": "utterance", "op": "contains", "v": "placed"}, True),
    ({"f": "utterance", "op": "contains", "v": "PLACED"}, True),
    ({"f": "utterance", "op": "regex", "v": r"^i (have|am)\b"}, True),
    ({"f": "utterance", "op": "no_contractions", "v": True}, True),
    ({"f": "utterance", "op": "no_contractions"}, True),
    ({"f": "answer_utterance", "op": "contains", "v": "sure"}, True),
    ({"f": "was_challenged_last_round", "op": "==", "v": False}, True),
    ({"all": [FORMAL, {"f": "hand_size", "op": "<=", "v": 2}]}, True),
    ({"all": [FORMAL, {"f": "hand_size", "op": ">", "v": 2}]}, False),
    ({"any": [{"f": "hand_size", "op": ">", "v": 9}, FORMAL]}, True),
    ({"any": []}, False),
    ({"not": FORMAL}, False),
    ({"not": {"any": [{"f": "table_rank", "op": "==", "v": "Q"}]}}, True),
    ({"all": []}, True),
])
def test_dsl_ops(pred, exp):
    assert H.validate_predicate(pred) == (True, "ok"), H.validate_predicate(pred)
    assert H.eval_predicate(pred, F) is exp


def test_no_contractions_curly_and_straight():
    p = {"f": "utterance", "op": "no_contractions", "v": True}
    assert not H.eval_predicate(p, {"utterance": "That's two Kings."})
    assert not H.eval_predicate(p, {"utterance": "It’s a King, don’t worry."})
    assert H.eval_predicate(p, {"utterance": "These are Kings."})
    assert H.eval_predicate({"f": "utterance", "op": "no_contractions", "v": False}, {"utterance": "That's it"})


def test_missing_feature_is_false():
    assert H.eval_predicate(FORMAL, {}) is False
    assert H.eval_predicate({"not": FORMAL}, {}) is True


@pytest.mark.parametrize("pred,why", [
    ({"f": "secret_cards", "op": "==", "v": 1}, "unknown feature"),
    ({"f": "hand_size", "op": "~=", "v": 1}, "unknown op"),
    ({"f": "hand_size", "op": "==", "v": "2"}, "numeric"),
    ({"f": "hand_size", "op": "==", "v": True}, "numeric"),
    ({"f": "formal_register", "op": ">", "v": True}, "not valid"),
    ({"f": "formal_register", "op": "==", "v": 1}, "bool"),
    ({"f": "table_rank", "op": "<", "v": "K"}, "not valid"),
    ({"f": "hand_size", "op": "contains", "v": "2"}, "string feature"),
    ({"f": "utterance", "op": "regex", "v": "a" * 101}, "longer than 100"),
    ({"f": "utterance", "op": "regex", "v": "(a+)+$"}, "catastrophic"),
    ({"f": "utterance", "op": "regex", "v": r"(\w+\s?)*$"}, "catastrophic"),
    ({"f": "utterance", "op": "regex", "v": "(a|aa)+b"}, "catastrophic"),
    ({"f": "utterance", "op": "regex", "v": "(unclosed"}, "invalid regex"),
    ({"f": "hand_size", "op": "=="}, "needs a value"),
    ({"f": "hand_size", "op": "==", "v": 1, "x": 2}, "unknown leaf keys"),
    ({"all": [FORMAL], "any": []}, "exactly one key"),
    ({"all": FORMAL}, "takes a list"),
    ("formal_register == True", "must be an object"),
])
def test_rejections(pred, why):
    ok, reason = H.validate_predicate(pred)
    assert not ok and why in reason, reason


def test_depth_limit():
    p = FORMAL
    for _ in range(5):
        p = {"not": p}
    assert H.validate_predicate(p)[0]          # 6 levels
    assert not H.validate_predicate({"not": p})[0]
    assert "depth" in H.validate_predicate({"not": p})[1]


def test_code_strings_are_just_data():
    evil = "__import__('os').system('echo pwned')"
    p = {"f": "utterance", "op": "==", "v": evil}
    assert H.validate_predicate(p)[0]
    assert H.eval_predicate(p, {"utterance": evil}) is True
    assert H.eval_predicate(p, {"utterance": "x"}) is False
    assert H.eval_predicate({"f": "utterance", "op": "contains", "v": "__import__"}, {"utterance": "hi"}) is False


def test_unvalidated_bad_regex_never_runs():
    # eval on a predicate that skipped validation: unsafe regex → no match, no hang
    assert H.eval_predicate({"f": "utterance", "op": "regex", "v": "(a+)+$"}, {"utterance": "a" * 5000 + "!"}) is False


def test_specificity():
    assert H.specificity({"all": []}) == 0
    assert H.specificity(FORMAL) == 1
    assert H.specificity({"all": [FORMAL, {"not": {"f": "hand_size", "op": "<", "v": 2}}]}) == 2


def test_new_hypothesis_rejects_bad():
    with pytest.raises(ValueError):
        mk(aspect="vibes")
    with pytest.raises(ValueError):
        mk(pred={"f": "nope", "op": "==", "v": 1})
    h = mk(idx=4)
    assert h["id"] == "h_w001_p3_004" and h["status"] == "provisional" and h["valid_until_game"] is None


# ---------------------------------------------------------------- scoring

def test_strong_tell_confirms():
    rev = telltale_reveals(60, seed=1)
    h = H.score_hypothesis(mk(), rev, np.random.default_rng(0))
    assert h["status"] == "confirmed", h
    assert h["q_true"][0] > h["q_false"][0] + 0.2
    assert h["p_gap"] > 0.9 and h["n_true"] + h["n_false"] == 60
    assert len(h["evidence_obs"]) == h["n_true"] and h["evidence_obs"][0].startswith("w001:g")


def test_no_signal_predicate_not_confirmed():
    rev = telltale_reveals(60, seed=2)
    h = H.score_hypothesis(mk(pred={"f": "table_rank", "op": "==", "v": "K"}), rev, np.random.default_rng(0))
    # always matches → n_false = 0, q_false uniform → no confirmation
    assert h["status"] != "confirmed"
    h2 = H.score_hypothesis(mk(pred={"f": "hand_size", "op": ">=", "v": 3}), rev, np.random.default_rng(0))
    assert h2["status"] != "confirmed"
    # a random split of the data has no signal either
    for i, e in enumerate(rev):
        e["features"]["claim_count"] = 1 + (i * 7919) % 3
    h3 = H.score_hypothesis(mk(pred={"f": "claim_count", "op": "==", "v": 2}), rev, np.random.default_rng(0))
    assert h3["status"] != "confirmed"


def test_flipped_tell_refutes():
    rev = telltale_reveals(60, seed=3, flip=True)
    h = H.score_hypothesis(mk(), rev, np.random.default_rng(0))
    assert h["status"] == "refuted" and h["p_gap"] < 0.2


def test_only_after_created_game_and_about():
    rev = telltale_reveals(30, seed=1) + rate_reveals(30, 0.3, seed=1, about="p2")
    h = H.score_hypothesis(mk(created_game=5), rev, np.random.default_rng(0))
    assert h["n_true"] + h["n_false"] == sum(1 for e in rev if e["by"] == "p3" and e["game"] > 5)


def test_status_transitions_and_recent_window():
    good = telltale_reveals(45, seed=1)                      # games 1..15
    h = H.score_hypothesis(mk(), good, np.random.default_rng(0), now_game=15)
    assert h["status"] == "confirmed"
    bad = telltale_reveals(12, seed=9, start_game=16, per_game=6, flip=True)   # games 16, 17
    allrev = good + bad
    assert H.recent_window_check(h, allrev, 17, np.random.default_rng(0)) is True
    assert H.recent_window_check(h, good, 15, np.random.default_rng(0)) is False
    H.score_hypothesis(h, allrev, np.random.default_rng(0), now_game=17)
    assert h["status"] == "refuted" and h["refuted_reason"] == "recent_window"
    # refuted is absorbing
    H.score_hypothesis(h, good, np.random.default_rng(0), now_game=15)
    assert h["status"] == "refuted"


def test_provisional_with_little_data():
    h = H.score_hypothesis(mk(), telltale_reveals(4, seed=1), np.random.default_rng(0))
    assert h["status"] == "provisional"


def test_baseline_always_true():
    h = mk(pred={"all": []}, aspect="baseline_bluff", about="p2")
    H.score_hypothesis(h, rate_reveals(3, 0.3, seed=0), np.random.default_rng(0))
    assert h["status"] == "provisional" and h["n_false"] == 0 and h["q_false"] == [] and h["p_gap"] is None
    H.score_hypothesis(h, rate_reveals(20, 0.3, seed=0), np.random.default_rng(0))
    assert h["status"] == "confirmed" and h["n_true"] == 20
    HypothesisEv.model_validate(H.hypothesis_event(h))


def test_determinism():
    rev = telltale_reveals(40, seed=5)
    a = H.score_hypothesis(mk(), rev, np.random.default_rng(42))
    b = H.score_hypothesis(mk(), rev, np.random.default_rng(42))
    assert a == b
    assert H.detect_drift(rev, np.random.default_rng(7))[2] == H.detect_drift(rev, np.random.default_rng(7))[2]


def test_hypothesis_event_validates():
    h = H.score_hypothesis(mk(), telltale_reveals(30, seed=1), np.random.default_rng(0))
    HypothesisEv.model_validate(H.hypothesis_event(h))
    HypothesisEv.model_validate(H.hypothesis_event(mk(provenance="untrusted", q=0.0)))


# ---------------------------------------------------------------- supersession

def test_supersede_event_shape():
    old = mk(idx=1, created_game=2)
    new = mk(idx=2, created_game=6, supersedes=old["id"])
    other = mk(idx=3, created_game=7, aspect="pressure")
    for h in (old, new, other):
        h["status"] = "confirmed"
    hyps = [old, new, other]
    evs = H.supersede(hyps, 8, 0)
    assert len(evs) == 1
    typ, payload = evs[0]
    assert typ == "belief_superseded"
    BeliefSuperseded.model_validate(payload)
    assert payload == {"aspect": "verbal_tell", "old_id": old["id"], "new_id": new["id"], "reason": "llm",
                       "at_game": 8, "at_round": 0, "about": "p3"}
    assert old["valid_until_game"] == 8 and old["superseded_by"] == new["id"]
    assert new["valid_until_game"] is None and other["valid_until_game"] is None
    assert H.supersede(hyps, 9, 0) == []   # idempotent


def test_supersede_needs_newer_confirmed():
    old, new = mk(idx=1, created_game=2), mk(idx=2, created_game=6)
    old["status"] = "confirmed"            # new still provisional
    assert H.supersede([old, new], 8, 0) == []


# ---------------------------------------------------------------- drift

def test_drift_fires_on_honest_to_bluffing():
    rev = rate_reveals(18, 0.0, seed=1, about="p2") + rate_reveals(9, 0.65, seed=2, about="p2", start_game=7)
    fired, post, p = H.detect_drift(rev, np.random.default_rng(0))
    assert fired and p > 0.9 and post is not None and abs(post.sum() - 1) < 1e-9


@pytest.mark.parametrize("q,seed", [(0.0, 3), (0.3, 4), (0.6, 5)])
def test_drift_quiet_when_stationary(q, seed):
    fired, _, p = H.detect_drift(rate_reveals(30, q, seed=seed), np.random.default_rng(0))
    assert not fired, p


def test_drift_too_little_data():
    assert H.detect_drift(rate_reveals(9, 0.9, seed=0), np.random.default_rng(0)) == (False, None, 0.0)


def test_apply_drift_supersedes_baseline():
    rev = rate_reveals(18, 0.0, seed=1, about="p2") + rate_reveals(9, 0.65, seed=2, about="p2", start_game=7)
    base = mk(pred={"all": []}, aspect="baseline_bluff", about="p2", idx=1)
    H.score_hypothesis(base, rev[:18], np.random.default_rng(0))
    assert base["status"] == "confirmed"
    hyps = [base]
    new, evs, p = H.apply_drift(hyps, rev, wid="w001", about="p2", idx=2, now_game=9, now_round=0,
                                rng=np.random.default_rng(0))
    assert new is not None and new["status"] == "confirmed" and new["origin"] == "drift"
    assert new["q_true"][0] > 0.4 and new["n_true"] == 8
    assert len(evs) == 1 and evs[0][1]["reason"] == "drift" and evs[0][1]["old_id"] == base["id"]
    BeliefSuperseded.model_validate(evs[0][1])
    assert base["valid_until_game"] == 9
    # does not re-fire on the same change point next game
    new2, evs2, _ = H.apply_drift(hyps, rev, wid="w001", about="p2", idx=3, now_game=10, now_round=0,
                                  rng=np.random.default_rng(0))
    assert new2 is None and evs2 == []
    q, src, mem = H.matching_q(hyps, {"formal_register": False}, "p2", None)
    assert src == new["id"] and q == new["q_true"][0]


# ---------------------------------------------------------------- matching_q

def _confirmed_tell():
    return H.score_hypothesis(mk(idx=1), telltale_reveals(60, seed=1), np.random.default_rng(0))


def test_matching_q_priority():
    tell = _confirmed_tell()
    base = np.full(101, 1 / 101)
    q, src, mem = H.matching_q([tell], {"formal_register": True}, "p3", base)
    assert src == tell["id"] and q == tell["q_true"][0]
    assert mem[0]["used_as"] == "q_source" and mem[0]["matched"]
    for m in mem:
        MemoryUsed.model_validate(m)
    q, src, _ = H.matching_q([tell], {"formal_register": False}, "p3", base)
    assert src == tell["id"] and q == tell["q_false"][0]
    q, src, mem = H.matching_q([tell], {"formal_register": True}, "p2", base)
    assert src == "base:p2" and abs(q - 0.5) < 1e-9 and mem == []


def test_more_specific_wins():
    tell = _confirmed_tell()
    spec = mk(idx=2, pred={"all": [FORMAL, {"f": "hand_size", "op": "<=", "v": 2}]}, aspect="pressure")
    spec.update(status="confirmed", confirmed_by="scoring", q_true=[0.9, 0.8, 0.95], q_false=[0.2, 0.1, 0.3])
    q, src, mem = H.matching_q([tell, spec], {"formal_register": True, "hand_size": 2}, "p3", None)
    assert src == spec["id"] and q == 0.9
    assert {m["used_as"] for m in mem} == {"q_source", "considered"}
    q, src, _ = H.matching_q([tell, spec], {"formal_register": True, "hand_size": 4}, "p3", None)
    assert src == tell["id"]


def test_false_memory_ignored_by_profiler_used_by_blind():
    tell = _confirmed_tell()
    fake = mk(idx=9, created_game=10, provenance="untrusted", origin="surgery", q=0.0)
    fake["status"] = "confirmed"            # injected claim of confidence — not from scoring
    feats = {"formal_register": True}
    q, src, mem = H.matching_q([tell, fake], feats, "p3", None)
    assert src == tell["id"] and q == tell["q_true"][0]
    ign = [m for m in mem if m["used_as"] == "ignored_untrusted"]
    assert len(ign) == 1 and ign[0]["hyp_id"] == fake["id"] and ign[0]["q_override"] == 0.0
    MemoryUsed.model_validate(ign[0])
    q, src, mem = H.matching_q([tell, fake], feats, "p3", None, trust_untrusted=True)
    assert src == fake["id"] and q == 0.0
    assert [m["used_as"] for m in mem][0] == "q_source"


def test_untrusted_confirmed_by_scoring_counts():
    h = mk(idx=5, provenance="untrusted")
    H.score_hypothesis(h, telltale_reveals(60, seed=1), np.random.default_rng(0))
    assert h["status"] == "confirmed" and h["confirmed_by"] == "scoring"
    q, src, _ = H.matching_q([h], {"formal_register": True}, "p3", None)
    assert src == h["id"] and q == h["q_true"][0]


def test_superseded_and_provisional_skipped():
    tell = _confirmed_tell()
    tell["valid_until_game"] = 5
    prov = mk(idx=2)
    q, src, mem = H.matching_q([tell, prov], {"formal_register": True}, "p3", None)
    assert src == "base:p3" and mem == []
