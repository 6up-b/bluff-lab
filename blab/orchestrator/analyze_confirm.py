"""c01 confirmatory analysis (PREREG_c01.md) -> runs/c01/confirm.json.

H11: profiler_hybrid's Brier on the first 10 reveals (p1's first 10 revealed claims by p3) < logreg_embed's, on
     tell_relational_v2 and on tell_flipper (two tests).
H12: profiler_hybrid's win rate > logreg_bow's, pooled (per seed: mean over the 4 tables of the paired Δ).
H13: profiler_plus (c01 version, profiler_plus_c01) win rate > logreg_bow's, pooled.
Paired by seed, 2000-resample bootstrap 95% CI, holds iff the CI excludes 0 in the predicted direction. A
hypothesis uses every seed for which all of its cells are complete (pre-registered stopping rule); no exclusions.
Usage: python -m blab.orchestrator.analyze_confirm [runs_root]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from blab.orchestrator.analyze import _load_one, boot, ensure_events, world_dirs
from blab.orchestrator.analyze_boundary import _brier_window, _clean

TABLES = ("tell_relational_v2", "tell_flipper", "tell_conjunctive", "longcon")
ARMS = ("logreg_bow", "logreg_embed", "bayes_discount_pp", "profiler_plus_c01", "profiler_hybrid",
        "profiler_plus_noreflect")
LABEL = {"bayes_discount_pp": "bayes_discount (+population prior)", "profiler_plus_c01": "profiler_plus (c01)"}
SEEDS = list(range(5000, 5024))
METRICS = ("win_rate", "brier", "brier_first10")


def _parse(wid: str):
    for t in TABLES:
        if wid.startswith(t + "-"):
            rest = wid[len(t) + 1:]
            a, s = rest.rsplit("-raw-s", 1)
            return t, a, int(s)
    return None


def _paired(xa: dict, xb: dict, seeds=None) -> dict:
    ks = sorted(set(xa) & set(xb) if seeds is None else [s for s in seeds if s in xa and s in xb])
    return {**boot([xa[s] - xb[s] for s in ks], np.random.default_rng(0)), "seeds": ks}


def _holds(d, direction):
    if d.get("lo") is None:
        return None
    return bool(d["hi"] < 0) if direction == "neg" else bool(d["lo"] > 0)


def confirm(root: str | Path = "runs") -> dict:
    root = Path(root)
    cells: dict = defaultdict(lambda: defaultdict(dict))       # table -> arm -> seed -> metrics
    status: dict = defaultdict(lambda: defaultdict(dict))
    reflect = defaultdict(lambda: {"calls": 0, "errors": 0, "proposed": 0})
    for d in world_dirs(root / "c01"):
        p = _parse(d.name)
        if not p:
            continue
        t, a, s = p
        if s not in SEEDS:
            continue
        w = _load_one(d)
        if not w or not w.get("done"):
            status[t][a][s] = "incomplete"
            continue
        ensure_events(w)
        m = w["m"]
        cells[t][a][s] = {"win_rate": m["win_rate"], "brier": m["brier"], "brier_first10": _brier_window(w)}
        status[t][a][s] = "done"
        for e in w["events"]:
            if e["type"] == "reflect_result":
                r = reflect[a]
                r["calls"] += 1
                r["errors"] += int(bool(e["payload"].get("error")))
                r["proposed"] += len(e["payload"].get("proposed") or [])
        w.pop("events", None)

    per_cell, deltas = {}, {}
    for t in TABLES:
        per_cell[t], deltas[t] = {}, {}
        for a in ARMS:
            c = cells[t].get(a) or {}
            n_inc = sum(1 for v in status[t].get(a, {}).values() if v == "incomplete")
            per_cell[t][a] = ({"status": "not run", "n_seeds": 0, "n_incomplete": n_inc} if not c else
                              {"status": "run", "n_seeds": len(c), "n_incomplete": n_inc,
                               "n_not_run": len(SEEDS) - len(c) - n_inc, "label": LABEL.get(a, a),
                               **{k: boot([v[k] for v in c.values()], np.random.default_rng(0)) for k in METRICS}})
            deltas[t][a] = {}
            for b in ("logreg_bow", "logreg_embed"):
                if a == b or not c or not cells[t].get(b):
                    continue
                deltas[t][a][f"vs_{b}"] = {k: {kk: vv for kk, vv in _paired(
                    {s: v[k] for s, v in c.items() if v[k] is not None},
                    {s: v[k] for s, v in cells[t][b].items() if v[k] is not None}).items() if kk != "seeds"}
                    for k in METRICS}

    def pooled(a, b, key="win_rate"):
        """Per seed: mean over the 4 tables of (a − b); seeds with all 8 cells complete."""
        per_seed = {}
        for s in SEEDS:
            if all(s in cells[t].get(a, {}) and s in cells[t].get(b, {}) for t in TABLES):
                per_seed[s] = float(np.mean([cells[t][a][s][key] - cells[t][b][s][key] for t in TABLES]))
        return {**boot(per_seed.values(), np.random.default_rng(0)), "seeds": sorted(per_seed)}

    def one(t, a, b, key):
        xa = {s: v[key] for s, v in (cells[t].get(a) or {}).items() if v[key] is not None}
        xb = {s: v[key] for s, v in (cells[t].get(b) or {}).items() if v[key] is not None}
        return _paired(xa, xb)

    hyps = []
    for name, t in (("H11a", "tell_relational_v2"), ("H11b", "tell_flipper")):
        d = one(t, "profiler_hybrid", "logreg_embed", "brier_first10")
        hyps.append({"name": name, "text": f"{t}: profiler_hybrid Brier (first 10 reveals by p3) < logreg_embed",
                     "metric": "brier_first10", "table": t, "a": "profiler_hybrid", "b": "logreg_embed",
                     "direction": "neg", **d, "n_seeds": len(d["seeds"]), "holds": _holds(d, "neg")})
    for name, a in (("H12", "profiler_hybrid"), ("H13", "profiler_plus_c01")):
        d = pooled(a, "logreg_bow")
        hyps.append({"name": name, "text": f"{LABEL.get(a, a)} win rate > logreg_bow, pooled over the 4 tables",
                     "metric": "win_rate (pooled: per-seed mean over tables of the paired Δ)", "table": "pooled",
                     "a": a, "b": "logreg_bow", "direction": "pos", **d, "n_seeds": len(d["seeds"]),
                     "holds": _holds(d, "pos")})
    for h in hyps:
        h["status"] = "run" if h["n_seeds"] else "not run"
        h["winner"] = (None if h.get("lo") is None else
                       (h["a"] if h["holds"] else
                        (h["b"] if (h["lo"] > 0 if h["direction"] == "neg" else h["hi"] < 0) else "tie (CI spans 0)")))
    pooled_all = {a: {"vs_logreg_bow": pooled(a, "logreg_bow")} for a in ARMS if a != "logreg_bow"}
    # Lead's request (14:3x), received AFTER test-seed worlds had run -> exploratory, not pre-registered.
    expl = []
    for name, a in (("H14", "profiler_hybrid"), ("H15", "profiler_plus_c01")):
        d = pooled(a, "logreg_embed")
        expl.append({"name": name, "label": "exploratory, added after the run started",
                     "text": f"{LABEL.get(a, a)} win rate > logreg_embed, pooled over the 4 tables", "a": a,
                     "b": "logreg_embed", "direction": "pos", **d, "n_seeds": len(d["seeds"]),
                     "ci_excludes_0_in_direction": _holds(d, "pos")})
    expl_brier = {t: {a: one(t, a, "logreg_embed", "brier") for a in ("profiler_hybrid", "profiler_plus_c01")}
                  for t in TABLES}

    rc = {"calls": 0, "errors": 0, "cache_hits": 0, "in": 0, "out": 0, "cache_read": 0, "cache_write": 0,
          "values": defaultdict(int), "latency_ms_sum": 0}
    lp = root / "c01" / "rc_log.jsonl"
    if lp.exists():
        for ln in open(lp):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("hit"):
                rc["cache_hits"] += 1
                continue
            rc["calls"] += 1
            rc["errors"] += int(bool(r.get("error")))
            rc["values"][r.get("value")] += 1
            rc["latency_ms_sum"] += int(r.get("latency_ms") or 0)
            for k in ("in", "out", "cache_read", "cache_write"):
                rc[k] += int((r.get("usage") or {}).get(k) or 0)
    rc["values"] = dict(rc["values"])
    return {"run": "c01", "prereg": "PREREG_c01.md", "info": "raw", "respond_rule": "threshold", "games": 20,
            "seeds": [5000, 5023], "resamples": 2000, "tables": list(TABLES), "arms": list(ARMS), "labels": LABEL,
            "definitions": {"brier_first10": "mean (p_final - was_bluff)^2 over p1's first 10 revealed claims by p3",
                            "pooled": "per seed, mean over the 4 tables of the paired (a - b) world win rate",
                            "delta": "a - b, paired by seed", "holds": "95% CI excludes 0 in the predicted direction"},
            "hypotheses": hyps, "per_cell": per_cell, "paired_deltas": deltas, "pooled_win_rate": pooled_all,
            "exploratory": {"label": "exploratory, added after the run started (lead request after test-seed "
                                     "worlds had run; not in PREREG_c01.md)", "hypotheses": expl,
                            "brier_vs_logreg_embed_per_table": expl_brier},
            "reflection": dict(reflect), "record_check": rc, "deviations": []}


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "runs"
    out_path = Path(root) / "c01" / "confirm.json"
    res = _clean(confirm(root))
    if out_path.exists():                                   # keep hand-written deviations across re-analyses
        old = json.loads(out_path.read_text())
        res["deviations"] = old.get("deviations", [])
    out_path.write_text(json.dumps(res, indent=1))
    for h in res["hypotheses"]:
        print(f"{h['name']:5} n={h['n_seeds']:2} mean={h['mean']} CI=[{h['lo']}, {h['hi']}] holds={h['holds']}")
