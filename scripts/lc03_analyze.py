# usage: PYTHONPATH=. .venv/bin/python scripts/lc03_analyze.py   → runs/lc03/longcon3.json, runs/lc03/longcon3.md
"""lc03 analysis (PREREG_lc03.md). Per-world metrics are lc02's (analyze_longcon2._world, by import); arms are
taken from the wid (profiler_plus and profiler_plus_noreflect share the policy name). Paired-by-seed hypotheses,
2000-resample bootstrap over seeds (numpy default_rng(0)), 95% percentile CI."""
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from blab.orchestrator import analyze as A
from blab.orchestrator.analyze_longcon2 import _world, km_boot

RD = Path("runs/lc03")
SEEDS = list(range(10000, 10012))
ARMS = ["counter", "bayes_discount", "logreg_embed", "profiler_plus_noreflect", "profiler_plus"]
B = 2000


def extra(d: str) -> dict:
    """win rate + reflection bookkeeping for one world (read straight from events.jsonl)."""
    out = {"win_rate": None, "n_reflect": 0, "n_reflect_err": 0, "bedrock": 0, "anthropic": 0, "other": 0, "lat": []}
    for line in open(Path(d) / "events.jsonl"):
        if '"world_end"' in line or '"reflect_result"' in line:
            e = json.loads(line)
            if e["type"] == "world_end":
                out["win_rate"] = e["payload"].get("win_rate")
            elif e["type"] == "reflect_result":
                p = e["payload"]
                out["n_reflect"] += 1
                if p.get("error"):
                    out["n_reflect_err"] += 1
                tid = p.get("llm_trace_id") or ""
                out["bedrock" if tid.startswith("msg_bdrk") else "anthropic" if tid.startswith("msg_") else "other"] += 1
                if p.get("latency_ms") is not None:
                    out["lat"].append(p["latency_ms"])
    return out


def one(d: str):
    w = _world(d)
    if not w:
        return None
    wid = w["wid"]
    w["arm"] = wid[len(w["table"]) + 1: wid.rindex("-raw-s")]
    w.update(extra(d))
    return w


def paired(a: dict, b: dict) -> dict:
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds], float)
    if len(d) < 2:
        return {"mean": float(d.mean()) if len(d) else None, "lo": None, "hi": None, "n": len(d), "seeds": seeds}
    rng = np.random.default_rng(0)
    ms = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(ms, 2.5)), "hi": float(np.percentile(ms, 97.5)),
            "n": len(d), "seeds": seeds}


def main():
    specs = json.loads((RD / "worlds.json").read_text())
    dirs = [str(RD / s["wid"]) for s in specs if (RD / s["wid"] / "events.jsonl").exists()]
    with ProcessPoolExecutor() as ex:
        ws = [w for w in ex.map(one, dirs, chunksize=8) if w]
    done = {(w["table"], w["arm"], w["seed"]) for w in ws}
    not_run = [s["wid"] for s in specs if (s["table"], s["arm"], s["seed"]) not in done]
    rng = np.random.default_rng(0)
    by = defaultdict(lambda: {"longcon": [], "falsealarm": []})
    for w in ws:
        by[w["arm"]][w["table"]].append(w)
    res = {"run": "lc03", "n_worlds": len(ws), "prereg": "PREREG_lc03.md (commit 8bee6f3)", "seeds": "10000-10011",
           "seeds_reserved_unused": "10012-10023", "info": "raw", "respond_rule": "threshold", "games": 20,
           "tables": {"longcon": ["telltale", "sleeper (p3)"], "falsealarm": ["control", "control"]},
           "reflection": "profiler_plus only: real Claude, Opus 5.5 (Bedrock us.anthropic.claude-opus-5-5 / Anthropic "
                         "claude-opus-5-5, alternated by llm.py), effort low, llm.STABLE_SYSTEM_RAW, after every game",
           "definitions": "lc02's (blab/orchestrator/analyze_longcon2.py docstring), plus win_rate = p1 world_end win_rate",
           "not_run": not_run, "arms": {}}
    for arm in ARMS:
        lc, fa = by[arm]["longcon"], by[arm]["falsealarm"]
        lcb = [w for w in lc if w.get("r_b") is not None]
        rtds = [w["rtd"] for w in lcb if w["rtd"] is not None]
        a = {"label": arm, "rule": "threshold", "n_longcon": len(lc), "n_with_strategic_bluff": len(lcb),
             "n_falsealarm": len(fa),
             "detection_rate": A.boot(w["detected"] for w in lcb),
             "rtd_median_detected": A.med(rtds) | {"values": None},
             "rtd_km_median": km_boot([w["censor_t"] for w in lcb], [int(w["detected"]) for w in lcb], rng),
             "exploitation": A.boot(w["exploitation"] for w in lc),
             "lives_lost_after_g0": A.boot(w["lives_lost_after_g0"] for w in lc),
             "false_alarm_belief": A.boot(w["false_alarm_belief"] for w in fa),
             "false_alarm_any": A.boot(w["false_alarm_any"] for w in fa),
             "false_alarm_drift_only": A.boot(w["false_alarm_drift"] for w in fa),
             "win_rate_longcon": A.boot(w["win_rate"] for w in lc),
             "win_rate_falsealarm": A.boot(w["win_rate"] for w in fa),
             "g0_values": sorted({w["g0"] for w in lc if w.get("g0") is not None})}
        tb, hb = defaultdict(list), defaultdict(list)
        for w in lc:
            for g, v in (w.get("trust_by_game") or {}).items():
                tb[int(g)].append(v)
            for g, v in (w.get("honesty_by_game") or {}).items():
                if v is not None:
                    hb[int(g)].append(v)
        a["trust_curve"] = [{"game": g, **A.boot(tb[g])} for g in sorted(tb)]
        a["true_honesty_curve"] = [{"game": g, **A.boot(hb[g])} for g in sorted(hb)]
        if arm == "profiler_plus":
            allw = lc + fa
            lat = [x for w in allw for x in w["lat"]]
            a["reflections"] = {"n": sum(w["n_reflect"] for w in allw), "errors": sum(w["n_reflect_err"] for w in allw),
                                "bedrock": sum(w["bedrock"] for w in allw), "anthropic": sum(w["anthropic"] for w in allw),
                                "other": sum(w["other"] for w in allw),
                                "latency_ms_median": float(np.median(lat)) if lat else None}
        res["arms"][arm] = a

    def per_seed(arm, table, key):
        return {w["seed"]: float(w[key]) for w in by[arm][table] if w.get(key) is not None}

    ex = {a: per_seed(a, "longcon", "exploitation") for a in ARMS}
    h31, h32 = paired(ex["profiler_plus"], ex["bayes_discount"]), paired(ex["profiler_plus"], ex["profiler_plus_noreflect"])
    fa = res["arms"]["profiler_plus"]["false_alarm_belief"]
    res["hypotheses"] = {
        "H31": {"text": "profiler_plus lets through fewer sleeper bluffs after G0 than bayes_discount",
                "delta": "exploitation(profiler_plus) - exploitation(bayes_discount), paired by seed", **h31,
                "holds": h31["hi"] is not None and h31["hi"] < 0},
        "H32": {"text": "profiler_plus lets through fewer sleeper bluffs after G0 than profiler_plus_noreflect",
                "delta": "exploitation(profiler_plus) - exploitation(profiler_plus_noreflect), paired by seed", **h32,
                "holds": h32["hi"] is not None and h32["hi"] < 0},
        "H33": {"text": "profiler_plus false-alarm rate (false_alarm_belief) on falsealarm (report only)", **fa,
                "holds": None, "supplementary_false_alarm_any": res["arms"]["profiler_plus"]["false_alarm_any"]},
    }
    res["paired_vs_bayes_discount"] = {
        arm: {f"{t}.{k}": paired(per_seed(arm, t, k), per_seed("bayes_discount", t, k))
              for t, k in (("longcon", "exploitation"), ("longcon", "lives_lost_after_g0"), ("longcon", "detected"),
                           ("longcon", "win_rate"), ("falsealarm", "win_rate"), ("falsealarm", "false_alarm_belief"))}
        for arm in ARMS if arm != "bayes_discount"}
    n_pp = len(by["profiler_plus"]["longcon"])
    res["deviations"] = [
        "n = 12 seeds (10000-10011), pre-registered in PREREG_lc03.md before launch from a wall-time estimate; "
        "10012-10023 reserved and unused.",
        "Laptop process pool, not box2 (Claude concurrency, not CPU, was the bottleneck).",
        "Claude reflection alternated between Bedrock and the Anthropic API (same model Opus 5.5, effort low), as llm.py "
        "does when both keys are present; counts in arms.profiler_plus.reflections.",
    ] + ([f"only {n_pp}/12 profiler_plus longcon worlds completed; not run: {not_run}"] if not_run else [])
    (RD / "longcon3.json").write_text(json.dumps(res, indent=1, default=float))
    (RD / "longcon3.md").write_text(summary(res))
    import sys
    if "--quiet" not in sys.argv:
        print(summary(res))


def _f(x, k=1, dp=2):
    if not x or x.get("mean") is None:
        return "—"
    s = f"{x['mean'] * k:.{dp}f}"
    if x.get("lo") is not None:
        s += f" [{x['lo'] * k:.{dp}f}, {x['hi'] * k:.{dp}f}]"
    return s


def summary(res):
    L = [f"# lc03 Long Con with profiler_plus: {res['n_worlds']} worlds, seeds {res['seeds']}", "",
         "| arm | n lc/fa | detect % | KM median RTD [CI] | exploitation | lives lost ≥G0 | false alarm % (belief) | FA % (§6 rule, suppl.) | win lc | win fa |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for a in res["arms"].values():
        km = a["rtd_km_median"]
        kms = "—" if km["median"] is None else f"{km['median']:.0f}"
        if km.get("lo") is not None or km.get("hi") is not None:
            kms += f" [{km['lo'] if km['lo'] is not None else '∞'}, {km['hi'] if km['hi'] is not None else '∞'}]"
        L.append(f"| {a['label']} | {a['n_longcon']}/{a['n_falsealarm']} | {_f(a['detection_rate'], 100, 0)} | {kms} | "
                 f"{_f(a['exploitation'])} | {_f(a['lives_lost_after_g0'])} | {_f(a['false_alarm_belief'], 100, 0)} | "
                 f"{_f(a['false_alarm_any'], 100, 0)} | {_f(a['win_rate_longcon'])} | {_f(a['win_rate_falsealarm'])} |")
    L.append("")
    for k, h in res["hypotheses"].items():
        L.append(f"- **{k}** ({h['text']}): {_f(h)} n={h['n']} holds={h['holds']}")
    L.append("")
    L += [f"- deviation: {d}" for d in res["deviations"]]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
