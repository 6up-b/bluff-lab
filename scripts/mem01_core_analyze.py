import json, glob, os, numpy as np, collections
from blab.events import read_events
from blab.hypotheses import eval_predicate
R = "runs/mem01"
rows = {}
for mp in glob.glob(f"{R}/*-raw-s*/meta.json") + glob.glob("runs/mem01_ctrl/*-raw-s*/meta.json"):
    m = json.load(open(mp))
    if m.get("status") != "done": continue
    d = os.path.dirname(mp); evs = read_events(f"{d}/events.jsonl")
    we = [e for e in evs if e["type"] == "world_end"][0]["payload"]
    disc = None; right = None
    if m["table"] in ("telltale", "tell_flipper", "tell_conjunctive"):
        claims = [e["payload"] for e in evs if e["type"] == "claim" and e["actor"] == "p3"]
        for e in evs:
            p = e["payload"]
            if e["type"] == "hypothesis" and p.get("about") == "p3" and p["aspect"] == "verbal_tell" and p["status"] == "confirmed":
                feats = [dict(c["features"], word_count=len(c["utterance"].split())) for c in claims]
                agree = np.mean([eval_predicate(p["predicate"], f) == bool(f["formal_register"]) for f in feats])
                agree = max(agree, 1 - agree)
                disc = e["game"]; right = round(float(agree), 2); break
    if "plus_nohyp" in m["wid"]: m["agent"] = "plus_nohyp"
    rows[(m["table"], m["agent"], m["seed"])] = dict(win=we["win_rate"], late=we.get("win_rate_11_20"), brier=we.get("brier"), disc=disc, agree=right)
def boot(x, B=2000, rng=np.random.default_rng(0)):
    x = np.asarray(x, float)
    if len(x) < 2: return (float(x.mean()) if len(x) else None, None, None)
    bs = rng.choice(x, (B, len(x))).mean(1)
    return (round(float(x.mean()), 3), round(float(np.quantile(bs, .025)), 3), round(float(np.quantile(bs, .975)), 3))
tables = ["telltale", "tell_conjunctive", "tell_flipper", "longcon"]; agents = ["counter", "logreg_bow", "plus_nohyp", "profiler", "profiler_plus"]
for t in tables:
    print(f"== {t}")
    for a in agents:
        rs = [v for (tt, aa, s), v in rows.items() if tt == t and aa == a]
        if not rs: continue
        ds = [r["disc"] for r in rs if r["disc"]]
        print(f"  {a:14} n={len(rs):2} win {np.mean([r['win'] for r in rs]):.3f} late {np.mean([r['late'] for r in rs]):.3f} brier {np.mean([r['brier'] for r in rs if r['brier'] is not None]):.3f}"
              + (f" verbal_tell confirmed {len(ds)}/{len(rs)} median g{np.median(ds)} agree-with-formal {[r['agree'] for r in rs if r['agree'] is not None]}" if a.startswith("profiler") else ""))
    for a, b in (("profiler_plus", "logreg_bow"), ("profiler", "logreg_bow"), ("profiler_plus", "counter"), ("profiler_plus", "profiler"), ("profiler_plus", "plus_nohyp"), ("plus_nohyp", "logreg_bow")):
        seeds = sorted(s for (tt, aa, s) in rows if tt == t and aa == a and (t, b, s) in rows)
        if len(seeds) < 2: continue
        dw = [rows[(t, a, s)]["win"] - rows[(t, b, s)]["win"] for s in seeds]
        db = [rows[(t, a, s)]["brier"] - rows[(t, b, s)]["brier"] for s in seeds]
        print(f"  Δ {a}−{b} (n={len(seeds)}): win {boot(dw)} brier {boot(db)}")
# pooled
print("== pooled over tables")
for a, b in (("profiler_plus", "logreg_bow"), ("profiler", "logreg_bow"), ("profiler_plus", "counter"), ("logreg_bow", "counter"), ("profiler_plus", "profiler"), ("profiler_plus", "plus_nohyp"), ("plus_nohyp", "logreg_bow")):
    keys = sorted((t, s) for (t, aa, s) in rows if aa == a and (t, b, s) in rows)
    if len(keys) < 2: continue
    print(f"  Δ {a}−{b} (n={len(keys)}): win {boot([rows[(t,a,s)]['win']-rows[(t,b,s)]['win'] for t,s in keys])} brier {boot([rows[(t,a,s)]['brier']-rows[(t,b,s)]['brier'] for t,s in keys])}")
