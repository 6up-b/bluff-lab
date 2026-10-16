"""b01 capability-boundary analysis (PREREG.md, committed 2167c1a) -> runs/b01/boundary.json.

H1 (relational): tell_relational, Brier over p1's first 10 REVEALED claims by the tell opponent (p3);
    profiler < logreg_embed and profiler < logreg_bow.
H2 (shifting): tell_flipper, Brier over the first 10 reveals of p3's claims in games >= G1 (after the flip);
    profiler < logreg_embed, and profiler not worse than bayes_discount.
H3 (answer): answer_tell, revealed claims to discovery; profiler < logreg_embed.
H4 (control): r01 telltale, logreg_bow >= profiler (Claude and stub reported separately, never pooled).
"Revealed" = the claim was challenged and its cards shown (the project's Brier definition, analyze.world_metrics).
Paired by seed, 2000-resample bootstrap 95% CI; holds = CI excludes 0 in the predicted direction. All cells reported.
Usage: python -m blab.orchestrator.analyze_boundary [runs_root]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from blab.orchestrator.analyze import boot, ensure_events, load_worlds, respond_decisions

TABLES = ("tell_relational", "tell_flipper", "answer_tell")
AGENTS = ("counter", "bayes", "bayes_discount", "logreg_bow", "logreg_embed", "profiler+claude", "profiler_amnesic")
NOT_RUN_NOTE = "cell not run (deadline GO/NO-GO 14:15: real-Claude reflection wall time)"


def _brier_window(w: dict, from_game: int = 0, k: int = 10) -> float | None:
    out = []
    for d in respond_decisions(w["events"]):
        e = d["ev"]
        if d["reveal"] is None or e["payload"].get("target") != "p3" or e["game"] < from_game:
            continue
        if e["payload"].get("p_final") is None:
            continue
        out.append((e["payload"]["p_final"] - (1.0 if d["reveal"]["payload"]["was_bluff"] else 0.0)) ** 2)
        if len(out) == k:
            break
    return float(np.mean(out)) if len(out) == k else (float(np.mean(out)) if out else None)


def _metrics(w: dict) -> dict:
    m = w["m"]
    ensure_events(w)
    g1 = int((w["seats"].get("p3", {}).get("params") or {}).get("G1", 0) or 0)
    return {"win_rate": m["win_rate"], "brier": m["brier"], "n_brier": m["n_brier"],
            "brier_first10": _brier_window(w), "brier_after_flip10": _brier_window(w, from_game=g1) if g1 else None,
            "revealed_to_discovery": m.get("revealed_to_discovery")}


def _paired(cells: dict, a: str, b: str, key: str) -> dict:
    xa = {s: v[key] for s, v in cells.get(a, {}).items() if v.get(key) is not None}
    xb = {s: v[key] for s, v in cells.get(b, {}).items() if v.get(key) is not None}
    ks = sorted(set(xa) & set(xb))
    return {**boot([xa[s] - xb[s] for s in ks], np.random.default_rng(0)), "a": a, "b": b, "metric": key}


def _holds(d: dict, direction: str) -> bool | None:
    if d.get("lo") is None:
        return None
    return bool(d["hi"] < 0) if direction == "neg" else bool(d["lo"] > 0)


def _hyp(name, cells, a, b, key, direction, text, also_not_worse=None):
    if not cells.get(a) or not cells.get(b):
        return {"name": name, "text": text, "status": "not run", "a": a, "b": b, "metric": key,
                "note": NOT_RUN_NOTE if not cells.get(a) else "cell not run"}
    d = _paired(cells, a, b, key)
    return {"name": name, "text": text, "status": "run", **d, "direction": direction, "holds": _holds(d, direction)}


def boundary(root: str | Path = "runs") -> dict:
    root = Path(root)
    ws = [w for w in load_worlds(root / "b01") if w["done"]]
    cells: dict = defaultdict(lambda: defaultdict(dict))         # table -> label -> seed -> metrics
    for w in ws:
        cells[w["table"]][w["label"]][w["seed"]] = _metrics(w)
        w.pop("events", None)
    per_cell = {}
    for t in TABLES:
        per_cell[t] = {}
        for a in AGENTS:
            c = cells[t].get(a) or {}
            per_cell[t][a] = ({"status": "not run", "note": NOT_RUN_NOTE} if not c else
                              {"status": "run", "n_seeds": len(c),
                               **{k: boot([v[k] for v in c.values()], np.random.default_rng(0))
                                  for k in ("win_rate", "brier", "brier_first10", "brier_after_flip10",
                                            "revealed_to_discovery")}})
    P = "profiler+claude"
    hyps = {
        "H1": [_hyp("H1a", cells["tell_relational"], P, "logreg_embed", "brier_first10", "neg",
                    "tell_relational: profiler Brier (first 10 reveals by p3) < logreg_embed"),
               _hyp("H1b", cells["tell_relational"], P, "logreg_bow", "brier_first10", "neg",
                    "tell_relational: profiler Brier (first 10 reveals by p3) < logreg_bow")],
        "H2": [_hyp("H2a", cells["tell_flipper"], P, "logreg_embed", "brier_after_flip10", "neg",
                    "tell_flipper: profiler Brier (10 reveals after the flip) < logreg_embed"),
               _hyp("H2b", cells["tell_flipper"], P, "bayes_discount", "brier_after_flip10", "not_worse",
                    "tell_flipper: profiler Brier (10 reveals after the flip) not worse than bayes_discount")],
        "H3": [_hyp("H3", cells["answer_tell"], P, "logreg_embed", "revealed_to_discovery", "neg",
                    "answer_tell: profiler revealed claims to discovery < logreg_embed")],
    }
    for h in hyps["H2"]:
        if h["name"] == "H2b" and h["status"] == "run":
            h["holds"] = None if h.get("hi") is None else bool(h["hi"] < 0)
            h["note"] = ("'not worse' has no pre-registered margin; reported as holds iff the CI excludes 0 "
                         "in profiler's favour (strict), with the CI itself shown")
    # H4: r01 engineered telltale (control replication), Claude and stub separate
    r01 = [w for w in load_worlds(root / "r01") if w["done"] and w["table"] == "telltale" and not w["vkey"]
           and w["rule"] == "threshold"]
    r01c: dict = defaultdict(dict)
    for w in r01:
        m = w["m"]
        r01c[w["label"]][w["seed"]] = {"win_rate": m["win_rate"], "brier": m["brier"]}
    hyps["H4"] = []
    for prof in ("profiler", "profiler+claude"):
        for key, direction in (("win_rate", "pos"), ("brier", "neg")):
            h = _hyp(f"H4[{prof}|{key}]", r01c, "logreg_bow", prof, key, direction,
                     f"r01 telltale (engineered, lexical): logreg_bow {key} better than {prof} "
                     f"({'stub' if prof == 'profiler' else 'Claude'} reflection)")
            h["source"] = "runs/r01"
            hyps["H4"].append(h)

    def row(cue, h, winner_if):
        if h["status"] != "run":
            return {"cue": cue, "hypothesis": h["name"], "status": "not run", "winner": None, "delta": None,
                    "ci": None, "holds": None, "note": h.get("note")}
        win = winner_if(h)
        return {"cue": cue, "hypothesis": h["name"], "status": "run", "metric": h["metric"], "a": h["a"], "b": h["b"],
                "winner": win, "delta": h["mean"], "ci": [h["lo"], h["hi"]], "n": h["n"], "holds": h["holds"]}

    def lower_wins(h):
        if h["lo"] is None:
            return None
        return h["a"] if h["hi"] < 0 else (h["b"] if h["lo"] > 0 else "tie (CI spans 0)")

    def higher_wins(h):
        if h["lo"] is None:
            return None
        return h["a"] if h["lo"] > 0 else (h["b"] if h["hi"] < 0 else "tie (CI spans 0)")

    h4c = next(h for h in hyps["H4"] if h["name"] == "H4[profiler+claude|win_rate]")
    h4s = next(h for h in hyps["H4"] if h["name"] == "H4[profiler|win_rate]")
    cmap = [row("lexical / engineered (r01 telltale, Claude reflection)", h4c, higher_wins),
            row("lexical / engineered (r01 telltale, stub reflection)", h4s, higher_wins),
            row("relational (tell_relational)", hyps["H1"][0], lower_wins),
            row("shifting (tell_flipper)", hyps["H2"][0], lower_wins),
            row("shifting vs bayes_discount (tell_flipper)", hyps["H2"][1], lower_wins),
            row("answer (answer_tell)", hyps["H3"][0], lower_wins)]
    return {"run": "b01", "prereg": "PREREG.md @ 2167c1a", "info": "raw", "respond_rule": "threshold",
            "seeds": [4000, 4023], "games": 20, "resamples": 2000,
            "definitions": {"brier_first10": "mean (p_final - was_bluff)^2 over p1's first 10 revealed claims by p3",
                            "brier_after_flip10": "same, over the first 10 reveals of p3 claims in games >= G1",
                            "revealed_to_discovery": "analyze.tell_discovery (telltale-family only); None elsewhere",
                            "delta": "a - b, paired by seed", "holds": "95% CI excludes 0 in the predicted direction"},
            "per_cell": per_cell, "hypotheses": hyps, "capability_map": {
                "columns": ["cue", "winner", "delta", "ci", "holds"], "rows": cmap,
                "note": "Δ = profiler − baseline (H1–H3, lower Brier/claims is better) or logreg_bow − profiler (r01 "
                        "win rate). Roles designed after r01 (PREREG disclosure). 'not run' cells were not run."}}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "runs"
    res = _clean(boundary(root))
    out = Path(root) / "b01" / "boundary.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["capability_map"], indent=1))
