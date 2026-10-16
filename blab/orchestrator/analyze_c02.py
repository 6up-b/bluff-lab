"""c02 analysis: profiler_plus (memory agent) vs logreg_embed (sentence-embedding baseline).

  python -m blab.orchestrator.analyze_c02 explore   → runs/mem01/embed_exploratory.json (+ .md)
      EXPLORATORY: baseline chosen after mem01 was run (mem01 seeds 3001–3012, core tables).
  python -m blab.orchestrator.analyze_c02 confirm   → runs/c02/confirm.json (+ .md)  (PREREG_c02.md, H21–H23)

Paired by seed; 2000-resample bootstrap (default_rng(0)), 95% percentile CI. Primary pooling = per seed, the mean
over the 4 tables of (arm − baseline), bootstrap over seeds; secondary = mem01's pooling over table × seed pairs.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np

TABLES = ["telltale", "tell_conjunctive", "tell_flipper", "longcon"]
BASE = "logreg_embed"
EXPLORE_LABEL = "exploratory: baseline chosen after mem01 was run"


def load(dirs: list[str], rename: dict[str, str] | None = None) -> dict:
    rows = {}
    for d in dirs:
        for mp in glob.glob(f"{d}/*-raw-s*/meta.json"):
            m = json.load(open(mp))
            if m.get("status") != "done":
                continue
            wid = m["wid"]
            arm = wid[len(m["table"]) + 1: wid.rindex("-raw-s")]
            arm = (rename or {}).get(arm, arm)
            we = None
            with open(os.path.join(os.path.dirname(mp), "events.jsonl")) as f:
                for line in f:
                    if '"world_end"' in line:
                        e = json.loads(line)
                        if e["type"] == "world_end":
                            we = e["payload"]
            if we is None:
                continue
            rows[(m["table"], arm, int(m["seed"]))] = {"win": we["win_rate"], "brier": we.get("brier")}
    return rows


def boot(x, B: int = 2000) -> dict:
    x = np.asarray(x, float)
    if len(x) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    if len(x) < 2:
        return {"mean": round(float(x.mean()), 4), "lo": None, "hi": None, "n": len(x)}
    bs = np.random.default_rng(0).choice(x, (B, len(x))).mean(1)
    return {"mean": round(float(x.mean()), 4), "lo": round(float(np.quantile(bs, .025)), 4),
            "hi": round(float(np.quantile(bs, .975)), 4), "n": int(len(x))}


def cells(rows: dict, arms: list[str]) -> dict:
    out = {}
    for t in TABLES:
        for a in arms:
            rs = [v for (tt, aa, s), v in rows.items() if tt == t and aa == a]
            if rs:
                out[f"{t}|{a}"] = {"n": len(rs), "win": round(float(np.mean([r["win"] for r in rs])), 4),
                                   "brier": round(float(np.mean([r["brier"] for r in rs])), 4)}
    return out


def deltas(rows: dict, a: str, b: str) -> dict:
    res = {"per_table": {}}
    for t in TABLES:
        seeds = sorted(s for (tt, aa, s) in rows if tt == t and aa == a and (t, b, s) in rows)
        res["per_table"][t] = {m: boot([rows[(t, a, s)][m] - rows[(t, b, s)][m] for s in seeds]) for m in ("win", "brier")}
    # primary pooling: seeds complete on all 4 tables for both arms; per-seed mean over tables
    seeds = sorted({s for (_, aa, s) in rows if aa == a})
    full = [s for s in seeds if all((t, a, s) in rows and (t, b, s) in rows for t in TABLES)]
    res["pooled_by_seed"] = {m: boot([np.mean([rows[(t, a, s)][m] - rows[(t, b, s)][m] for t in TABLES]) for s in full])
                             for m in ("win", "brier")}
    res["pooled_by_seed"]["seeds"] = full
    keys = sorted((t, s) for (t, aa, s) in rows if aa == a and (t, b, s) in rows)
    res["pooled_table_seed_pairs"] = {m: boot([rows[(t, a, s)][m] - rows[(t, b, s)][m] for t, s in keys])
                                      for m in ("win", "brier")}
    return res


def fmt(c: dict) -> str:
    if c["mean"] is None:
        return "n/a"
    if c["lo"] is None:
        return f"{c['mean']:+.3f} (n={c['n']})"
    return f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] (n={c['n']})"


def md_table(out: dict, pairs: list[tuple[str, str]]) -> list[str]:
    L = ["| Δ | table | win rate | Brier |", "|---|---|---|---|"]
    for a, b in pairs:
        d = out["deltas"][f"{a}-{b}"]
        for t in TABLES:
            L.append(f"| {a} − {b} | {t} | {fmt(d['per_table'][t]['win'])} | {fmt(d['per_table'][t]['brier'])} |")
        L.append(f"| {a} − {b} | **pooled (per-seed mean)** | {fmt(d['pooled_by_seed']['win'])} | {fmt(d['pooled_by_seed']['brier'])} |")
        L.append(f"| {a} − {b} | pooled (table×seed pairs) | {fmt(d['pooled_table_seed_pairs']['win'])} | {fmt(d['pooled_table_seed_pairs']['brier'])} |")
    return L


def explore() -> dict:
    rows = load(["runs/mem01", "runs/mem01_ctrl"], rename={"plus_nohyp": "plus_nohyp"})
    arms = ["counter", "logreg_bow", "logreg_embed", "plus_nohyp", "profiler", "profiler_plus"]
    pairs = [("profiler_plus", BASE), ("profiler", BASE), ("plus_nohyp", BASE), ("logreg_bow", BASE)]
    out = {"label": EXPLORE_LABEL, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "source": "runs/mem01 (+ runs/mem01_ctrl plus_nohyp); logreg_embed worlds by scripts/c02_embed_mem01.py",
           "seeds": "3001-3012", "info": "raw", "respond_rule": "threshold", "games": 20,
           "cells": cells(rows, arms), "deltas": {f"{a}-{b}": deltas(rows, a, b) for a, b in pairs}}
    os.makedirs("runs/mem01", exist_ok=True)
    json.dump(out, open("runs/mem01/embed_exploratory.json", "w"), indent=1)
    L = [f"# mem01 vs logreg_embed — {EXPLORE_LABEL}", "",
         "Paired by seed, 2000-resample bootstrap 95% CI. Δ win > 0 and Δ Brier < 0 favour the first arm.", ""]
    L += md_table(out, pairs)
    open("runs/mem01/embed_exploratory.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return out


def confirm() -> dict:
    rows = load(["runs/c02"])
    arms = ["counter", "logreg_embed", "profiler_plus_noreflect", "profiler_plus"]
    pairs = [("profiler_plus", BASE), ("profiler_plus_noreflect", BASE), ("counter", BASE),
             ("profiler_plus", "profiler_plus_noreflect")]
    D = {f"{a}-{b}": deltas(rows, a, b) for a, b in pairs}
    pp, nr = D[f"profiler_plus-{BASE}"]["pooled_by_seed"], D[f"profiler_plus_noreflect-{BASE}"]["pooled_by_seed"]
    ok = lambda c: c["lo"] is not None  # noqa: E731
    H = {
        "H21": {"text": "profiler_plus win rate > logreg_embed, pooled", "ci": pp["win"],
                "holds": bool(ok(pp["win"]) and pp["win"]["lo"] > 0), "seeds": pp["seeds"]},
        "H22": {"text": "profiler_plus Brier < logreg_embed, pooled", "ci": pp["brier"],
                "holds": bool(ok(pp["brier"]) and pp["brier"]["hi"] < 0), "seeds": pp["seeds"]},
        "H23": {"text": "profiler_plus_noreflect win rate does not beat logreg_embed, pooled (holds iff CI lo <= 0)",
                "ci": nr["win"], "holds": bool(ok(nr["win"]) and nr["win"]["lo"] <= 0), "seeds": nr["seeds"]},
    }
    planned = list(range(9000, 9012))
    not_run = {a: [f"{t}-s{s}" for s in planned for t in TABLES if (t, a, s) not in rows] for a in arms}
    # reflection accounting for profiler_plus
    n_ok = n_fail = 0
    for mp in glob.glob("runs/c02/*-profiler_plus-raw-s*/events.jsonl"):
        for line in open(mp):
            if '"reflect_result"' in line:
                p = json.loads(line)["payload"]
                n_ok += p.get("source") == "claude"
                n_fail += p.get("source") != "claude"
    devs = []
    done_seeds = pp["seeds"]
    if len(done_seeds) < len(planned):
        devs.append(f"only {len(done_seeds)}/{len(planned)} seeds complete for profiler_plus on all 4 tables "
                    f"(compute-driven stopping rule); not run: {not_run['profiler_plus']}")
    out = {"prereg": "PREREG_c02.md", "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds_planned": "9000-9011",
           "info": "raw", "respond_rule": "threshold", "games": 20,
           "reflection": {"model": "us.anthropic.claude-opus-5-5 (Bedrock), effort low", "ok": n_ok, "failed_or_timeout": n_fail},
           "hypotheses": H, "holds": {k: v["holds"] for k, v in H.items()}, "cells": cells(rows, arms),
           "deltas": D, "not_run": not_run, "deviations": devs}
    json.dump(out, open("runs/c02/confirm.json", "w"), indent=1)
    L = ["# c02 confirmatory (PREREG_c02.md)", ""]
    for k, v in H.items():
        L.append(f"- **{k}** {v['text']}: {fmt(v['ci'])} → holds={v['holds']}")
    L += ["", f"Reflections: {n_ok} ok, {n_fail} failed/timeout. Deviations: {devs or 'none'}", ""]
    L += md_table(out, pairs)
    open("runs/c02/confirm.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return out


if __name__ == "__main__":
    {"explore": explore, "confirm": confirm}[sys.argv[1]]()
