"""b01: profiler with REAL Claude reflection (Opus via Bedrock, reflection: "claude"), info raw, threshold rule.

Raw-mode reflection prompt + raw DSL validation: the same runner-side patch as runs/mem01/core_runner.py
(llm.py has no raw variant yet). BLAB_LLM_CONCURRENCY=8 caps concurrent Claude calls (Bedrock is shared).
Usage: python -m blab.experiments.b01_claude <table> [<table> ...]"""
import asyncio, json, os, sys, time
os.environ.setdefault("BLAB_LLM_CONCURRENCY", "8")
from blab.env import load_env
load_env()
from blab.orchestrator import llm
from blab.orchestrator.reflect import local_reflector
from blab.world import World, make_seats

S = llm.STABLE_SYSTEM
REPL = [
    ("| formal_register | bool | utterance has no contractions and >= 4 words |\n",
     "| word_count | int | number of whitespace-separated words in the utterance |\n"),
    ('(e.g. answer_formal_register)', '(e.g. answer_utterance, answer_word_count)'),
    ("== != < <= > >= contains regex no_contractions", "== != < <= > >= contains regex"),
    ("(contains/regex/no_contractions apply to string features", "(contains/regex apply to string features"),
    ('{"all": [{"f":"formal_register","op":"==","v":true}, {"f":"hand_size","op":"<=","v":2}]}',
     '{"all": [{"f":"hand_size","op":"<=","v":2}, {"f":"claim_count","op":">=","v":2}]}'),
    ('e.g. "Player 3 speaks formally (no contractions) when bluffing"', 'e.g. "Player 3 bluffs more on multi-card claims"'),
]
if "formal_register" in S:
    for a, b in REPL:
        assert a in S, a
        S = S.replace(a, b)
assert "formal" not in S.lower() and "no_contractions" not in S
llm.STABLE_SYSTEM = S
def _raw_validate(pred):
    from blab.hypotheses import validate_predicate
    return validate_predicate(pred, "raw")
llm._validate_pred = _raw_validate

OUT = "runs/b01"
SEEDS = list(range(4000, 4024))
sem = asyncio.Semaphore(int(os.environ.get("B01_WORLDS", "24")))

async def one(table, seed):
    wid = f"{table}-profiler-raw-s{seed}"
    mp = f"{OUT}/{wid}/meta.json"
    if os.path.exists(mp) and json.load(open(mp)).get("status") == "done":
        return
    seats = make_seats(table, "profiler", seed)
    seats[0]["params"].update(info="raw", respond_rule="threshold", reflection="claude")
    async with sem:
        w = World(wid=wid, seats=seats, seed=seed, games=20, reflect_every=2, out_dir=OUT, run="b01", box="laptop",
                  reflector=local_reflector, table=table)
        t = time.time()
        try:
            r = await w.run()
            print(f"[done] {time.strftime('%H:%M:%S')} {wid} {time.time()-t:.0f}s win={r['win_rate']:.2f} brier={r['brier']}", flush=True)
        except Exception as e:
            print(f"[FAIL] {wid} {e!r}", flush=True)

async def main(tables):
    await asyncio.gather(*[one(t, s) for s in SEEDS for t in tables])

asyncio.run(main(sys.argv[1:]))
