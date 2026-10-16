# usage: PYTHONPATH=. .venv/bin/python scripts/mem01_core.py   (resumes: skips worlds whose meta.json status == done)
"""mem01 (core arms): counter, logreg_bow, profiler, profiler_plus in RAW mode, real Claude reflection."""
import asyncio, json, os, time
os.environ["BLAB_LLM_CONCURRENCY"] = "6"   # lead cap (shared Bedrock)
from blab.env import load_env
load_env()
from blab.orchestrator import llm
from blab.orchestrator.reflect import local_reflector
from blab.world import World, make_seats, stub_reflector

# Raw-mode system prompt (core runner-side patch; llm.py has no raw variant yet — see INTERFACES Requests)
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
if os.environ.get("BLAB_RAW_PROMPT", "1") == "1" and "formal_register" in S:
    for a, b in REPL:
        assert a in S, a
        S = S.replace(a, b)
    assert "formal" not in S.lower() and "no_contractions" not in S
    llm.STABLE_SYSTEM = S
_orig_validate = llm._validate_pred
def _raw_validate(pred):
    from blab.hypotheses import validate_predicate
    return validate_predicate(pred, "raw")
llm._validate_pred = _raw_validate

OUT = "runs/mem01"
TABLES = ["telltale", "tell_conjunctive", "tell_flipper", "longcon"]
SEEDS = list(range(3001, 3013))
sem_worlds = asyncio.Semaphore(8)

async def one(table, agent, seed):
    wid = f"{table}-{agent}-raw-s{seed}"
    mp = f"{OUT}/{wid}/meta.json"
    if os.path.exists(mp) and json.load(open(mp)).get("status") == "done":
        return
    seats = make_seats(table, agent, seed)
    seats[0]["params"].update(info="raw", respond_rule="threshold")
    llm_arm = agent in ("profiler", "profiler_plus")
    async with sem_worlds:
        w = World(wid=wid, seats=seats, seed=seed, games=20, reflect_every=2, out_dir=OUT, run="mem01", box="laptop",
                  reflector=local_reflector if llm_arm else stub_reflector, table=table)
        t = time.time()
        try:
            r = await w.run()
            print(f"[done] {time.strftime('%H:%M:%S')} {wid} {time.time()-t:.0f}s win={r['win_rate']:.2f} brier={r['brier']}", flush=True)
        except Exception as e:
            print(f"[FAIL] {wid} {e!r}", flush=True)

async def main():
    for s in SEEDS:                         # fast arms first (all seeds)
        for t in TABLES:
            for a in ("counter", "logreg_bow"):
                await one(t, a, s)
    jobs = [one(t, a, s) for s in SEEDS for t in TABLES for a in ("profiler_plus", "profiler")]
    await asyncio.gather(*jobs)
asyncio.run(main())
