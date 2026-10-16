"""reg01 analysis (PREREG_reg01.md): The Regulars + surgical forgetting.

  .venv/bin/python -m blab.orchestrator.analyze_regulars [--run reg01]
→ runs/<run>/regulars.json (+ regulars.md)

Win rate = fraction of games p1 wins. Encounter index k = the regular's k-th match (1 = first meeting,
2–5 = returns). Brier = p1's non-forced respond decisions on p3's claims vs the god-view bluff.
Paired bootstrap over seeds (2000 resamples, 95% CI). Holds only if the CI excludes 0 (H10 others: CI ⊂ [−5, +5] pts).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from blab.experiments import regulars as R

ARMS = list(R.ARMS)
NB = 2000
EQ = 0.05


def boot(vals: list[float]) -> dict:
    a = np.asarray([v for v in vals if v is not None and not np.isnan(v)], float)
    if len(a) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(0)
    bs = a[rng.integers(0, len(a), (NB, len(a)))].mean(1)
    return {"mean": float(a.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)),
            "n": int(len(a))}


def load_world(d: Path) -> dict | None:
    ev = d / "events.jsonl"
    se = d / "season.json"
    if not ev.exists() or not se.exists():
        return None
    season = json.loads(se.read_text())
    wins, decs, last_claim, refl = {}, [], {}, []
    for line in ev.open():
        e = json.loads(line)
        t = e["type"]
        if t == "game_end":
            wins[e["game"]] = float(e["payload"].get("winner") == "p1")
        elif t == "claim":
            last_claim[(e["game"], e["round"])] = e
        elif t == "decision" and e["actor"] == "p1" and e["payload"].get("kind") == "respond":
            p = e["payload"]
            c = last_claim.get((e["game"], e["round"]))
            if p.get("forced") or c is None or c["actor"] != "p3":
                continue
            decs.append({"game": e["game"], "p": float(p["p_final"]), "y": float(c["payload"].get("bluff_kind") is not None),
                         "seq": e["seq"], "action": p["action"]})
        elif t == "reflect_result":
            refl.append({"game": e["game"], "error": e["payload"].get("error"),
                         "n": len(e["payload"].get("proposed") or [])})
    return {"season": season, "wins": wins, "decs": decs, "refl": refl}


def per_match(w: dict) -> list[dict]:
    sched = w["season"]["schedule"]
    out = []
    for m in range(1, R.MATCHES + 1):
        gs = range((m - 1) * R.GAMES_PER_MATCH + 1, m * R.GAMES_PER_MATCH + 1)
        if not all(g in w["wins"] for g in gs):
            continue
        nm = sched[m - 1]
        k = sched[:m].count(nm)
        ds = [d for d in w["decs"] if R.match_of(d["game"]) == m]
        out.append({"match": m, "name": nm, "k": k, "win": float(np.mean([w["wins"][g] for g in gs])),
                    "brier_sum": float(sum((d["p"] - d["y"]) ** 2 for d in ds)), "brier_n": len(ds)})
    return out


def brier(ms: list[dict]) -> float | None:
    n = sum(m["brier_n"] for m in ms)
    return sum(m["brier_sum"] for m in ms) / n if n else None


def win(ms: list[dict]) -> float | None:
    return float(np.mean([m["win"] for m in ms])) if ms else None


def paired(a: dict, b: dict, seeds: list[int]) -> dict:
    return boot([a[s] - b[s] for s in seeds if a.get(s) is not None and b.get(s) is not None])


def replay(seed: int, run_dir: Path, name_forgot: str, target: str) -> dict | None:
    """Parent's first non-forced respond decision against `target` after game 48, replayed with the
    checkpoint memory at the start of its game: unchanged, and with forget:<name_forgot>."""
    from blab.agents.ours import make_agent
    d = run_dir / f"s{seed}-profiler_plus"
    if not (d / "events.jsonl").exists():
        return None
    for line in (d / "events.jsonl").open():
        e = json.loads(line)
        if (e["type"] == "decision" and e["actor"] == "p1" and e["game"] > R.FORGET_GAME
                and e["payload"].get("kind") == "respond" and not e["payload"].get("forced")
                and e["payload"].get("target_name") == target):
            p = e["payload"]
            ckp = d / "ckpt" / f"g{e['game'] - 1:02d}.json"
            if not ckp.exists():
                return None
            mem = json.loads(ckp.read_text())["memory"]["p1"]
            params = {"info": "raw", "respond_rule": "threshold", "plus_reflect_every": 3,
                      "wid": f"s{seed}-profiler_plus", "seed": seed, "reflect_every": 3, "learning": True}
            ag = make_agent("reg_profiler_plus", "p1", np.random.default_rng(0), params)
            ag.set_memory(mem)
            full = ag.decide_respond(p["obs"], p["rng_u"])
            ag.set_memory(R.forget(mem, name_forgot)[0])
            fg = ag.decide_respond(p["obs"], p["rng_u"])
            return {"seed": seed, "world": d.name, "seq": e["seq"], "game": e["game"], "round": e["round"],
                    "target": target, "forgot": name_forgot, "utterance": p["claim"]["utterance"],
                    "p_logged": p["p_final"], "p_full_replay": full["p_final"],
                    "reproduces_log": abs(full["p_final"] - p["p_final"]) < 1e-12,
                    "p_forget": fg["p_final"], "delta_p": fg["p_final"] - full["p_final"],
                    "action_full": full["action"], "action_forget": fg["action"],
                    "flipped": full["action"] != fg["action"], "rng_u": p["rng_u"],
                    "n_memory_lines_full": len(full["memory_used"]), "n_memory_lines_forget": len(fg["memory_used"])}
    return None


def relearn(seed: int, run_dir: Path, name: str) -> dict | None:
    par = run_dir / f"s{seed}-profiler_plus" / "ckpt" / f"g{R.FORGET_GAME:02d}.json"
    br = run_dir / f"s{seed}-profiler_plus-forget_{name}" / "memory" / "p1.json"
    if not par.exists() or not br.exists():
        return None
    before = [h for h in json.loads(par.read_text())["memory"]["p1"]["hypotheses"] if h.get("about") == name]
    after = [h for h in json.loads(br.read_text())["hypotheses"] if h.get("about") == name]
    key = lambda h: json.dumps(h["predicate"], sort_keys=True)  # noqa: E731
    bk = {key(h) for h in before}
    return {"before": len(before), "after": len(after),
            "before_confirmed": sum(h.get("status") == "confirmed" for h in before),
            "after_confirmed": sum(h.get("status") == "confirmed" for h in after),
            "repropose_same_predicate": sum(key(h) in bk for h in after),
            "relearned_any": len(after) > 0}


def analyze(run: str = "reg01") -> dict:
    rd = Path("runs") / run
    W: dict[str, dict[int, list[dict]]] = {}
    refl_stats = {}
    seeds_all = sorted({int(p.name.split("-")[0][1:]) for p in rd.glob("s*-*") if p.is_dir()})
    raw: dict[str, dict[int, dict]] = {}
    for p in sorted(rd.glob("s*-*")):
        if not p.is_dir():
            continue
        w = load_world(p)
        if w is None:
            continue
        seed = int(p.name.split("-")[0][1:])
        label = p.name.split("-", 1)[1]
        raw.setdefault(label, {})[seed] = w
        W.setdefault(label, {})[seed] = per_match(w)
        rs = refl_stats.setdefault(label, {"calls": 0, "errors": 0})
        rs["calls"] += len(w["refl"])
        rs["errors"] += sum(1 for r in w["refl"] if r["error"])
    complete = lambda lab: {s for s, ms in W.get(lab, {}).items() if len(ms) == R.MATCHES}  # noqa: E731
    seeds = sorted(set.intersection(*[complete(a) for a in ARMS])) if all(a in W for a in ARMS) else []
    out = {"run": run, "prereg": "PREREG_reg01.md", "seeds": seeds, "n_seeds": len(seeds),
           "seeds_started": seeds_all, "regulars": R.REGULARS, "matches": R.MATCHES,
           "games_per_match": R.GAMES_PER_MATCH, "forget_after_match": R.FORGET_AFTER_MATCH,
           "model": "us.anthropic.claude-sonnet-5-5 (Bedrock, effort low), raw prompt", "respond_rule": "threshold",
           "reflections": refl_stats}
    # ---- per arm, by encounter index
    arms = {}
    for a in ARMS:
        enc = {}
        for k in range(1, 6):
            enc[str(k)] = {"win_rate": boot([win([m for m in W[a][s] if m["k"] == k]) for s in seeds]),
                           "brier": boot([brier([m for m in W[a][s] if m["k"] == k]) for s in seeds])}
        per_reg = {nm: {"first": boot([win([m for m in W[a][s] if m["name"] == nm and m["k"] == 1]) for s in seeds]),
                        "returns": boot([win([m for m in W[a][s] if m["name"] == nm and m["k"] >= 2]) for s in seeds])}
                   for nm in R.NAMES}
        arms[a] = {"win_rate_first": boot([win([m for m in W[a][s] if m["k"] == 1]) for s in seeds]),
                   "win_rate_returns": boot([win([m for m in W[a][s] if m["k"] >= 2]) for s in seeds]),
                   "brier_first": boot([brier([m for m in W[a][s] if m["k"] == 1]) for s in seeds]),
                   "brier_returns": boot([brier([m for m in W[a][s] if m["k"] >= 2]) for s in seeds]),
                   "win_rate_all": boot([win(W[a][s]) for s in seeds]),
                   "by_encounter": enc, "by_regular_win": per_reg}
    out["arms"] = arms
    ret = {a: {s: win([m for m in W[a][s] if m["k"] >= 2]) for s in seeds} for a in ARMS}
    retb = {a: {s: brier([m for m in W[a][s] if m["k"] >= 2]) for s in seeds} for a in ARMS}
    # ---- H8 / H9
    h8a = paired(ret["profiler_plus"], ret["profiler_plus_noid"], seeds)
    h8b = paired(ret["profiler_plus"], ret["logreg_pooled"], seeds)
    h9 = paired(ret["profiler_plus"], ret["logreg_identity"], seeds)
    lo_ok = lambda c: c["lo"] is not None and c["lo"] > 0  # noqa: E731
    hyps = {
        "H8": {"text": "returns: win(profiler_plus) > win(profiler_plus_noid) AND > win(logreg_pooled)",
               "vs_noid": h8a, "vs_logreg_pooled": h8b, "holds": bool(lo_ok(h8a) and lo_ok(h8b))},
        "H9": {"text": "returns: profiler_plus − logreg_identity (report only, no directional claim)",
               "win_rate": h9, "brier": paired(retb["profiler_plus"], retb["logreg_identity"], seeds),
               "holds": None},
    }
    # supplementary deltas vs every arm on returns (no claims)
    out["deltas_returns_vs_profiler_plus"] = {a: {"win_rate": paired(ret["profiler_plus"], ret[a], seeds),
                                                  "brier": paired(retb["profiler_plus"], retb[a], seeds)}
                                              for a in ARMS if a != "profiler_plus"}
    # ---- H10 forgetting
    forget = {}
    for F in ("Avery", "Blake"):
        lab = f"profiler_plus-forget_{F}"
        fseeds = [s for s in seeds if s in W.get(lab, {})          # branch logs hold games 49-60 only
                  and sum(m["match"] > R.FORGET_AFTER_MATCH for m in W[lab][s]) == R.MATCHES - R.FORGET_AFTER_MATCH]
        per = {}
        for nm in R.NAMES:
            def b5(lst, nm=nm):
                ms = [m for m in lst if m["match"] > R.FORGET_AFTER_MATCH and m["name"] == nm]
                return ms
            dw = [win(b5(W[lab][s])) - win(b5(W["profiler_plus"][s])) for s in fseeds]
            db = [(brier(b5(W[lab][s])) or 0) - (brier(b5(W["profiler_plus"][s])) or 0) for s in fseeds
                  if brier(b5(W[lab][s])) is not None and brier(b5(W["profiler_plus"][s])) is not None]
            c = boot(dw)
            per[nm] = {"delta_win": c, "delta_brier": boot(db), "forgotten": nm == F,
                       "branch_win": boot([win(b5(W[lab][s])) for s in fseeds]),
                       "parent_win": boot([win(b5(W["profiler_plus"][s])) for s in fseeds]),
                       "holds": (bool(c["hi"] is not None and c["hi"] < 0) if nm == F else
                                 bool(c["lo"] is not None and c["lo"] >= -EQ and c["hi"] <= EQ))}
        rl = [relearn(s, rd, F) for s in fseeds]
        rl = [r for r in rl if r]
        forget[F] = {"n_seeds": len(fseeds), "per_regular": per,
                     "holds": bool(per[F]["holds"] and all(per[o]["holds"] for o in R.NAMES if o != F)),
                     "forgotten_part_holds": per[F]["holds"],
                     "others_equivalence_holds": {o: per[o]["holds"] for o in R.NAMES if o != F},
                     "relearning": {"seeds_with_new_hyps_about_forgotten": sum(r["relearned_any"] for r in rl),
                                    "seeds": len(rl),
                                    "mean_hyps_before": float(np.mean([r["before"] for r in rl])) if rl else None,
                                    "mean_hyps_after": float(np.mean([r["after"] for r in rl])) if rl else None,
                                    "seeds_reproposing_same_predicate": sum(r["repropose_same_predicate"] > 0 for r in rl),
                                    "per_seed": rl}}
    hyps["H10a"] = {"text": "forget:Avery after match 16; block-5 Δwin(forgotten) < 0 and others' CI ⊂ [−5,+5] pts",
                    **{k: forget["Avery"][k] for k in ("holds", "forgotten_part_holds", "others_equivalence_holds")}}
    hyps["H10b"] = {"text": "forget:Blake, same rule",
                    **{k: forget["Blake"][k] for k in ("holds", "forgotten_part_holds", "others_equivalence_holds")}}
    out["hypotheses"] = hyps
    out["forgetting"] = forget
    # ---- decision-level replays
    reps = {"Avery": [r for r in (replay(s, rd, "Avery", "Avery") for s in seeds) if r],
            "Casey": [r for r in (replay(s, rd, "Avery", "Casey") for s in seeds) if r]}
    out["decision_replays"] = {
        "rule": "parent's first non-forced respond decision vs target after game 48; ckpt memory at its game start; "
                "unchanged vs forget:Avery; same rng_u",
        "primary_seed": seeds[0] if seeds else None,
        "avery_first": reps["Avery"][0] if reps["Avery"] else None,
        "casey_first": reps["Casey"][0] if reps["Casey"] else None,
        "summary": {t: {"n": len(v), "all_reproduce_log": all(r["reproduces_log"] for r in v),
                        "n_shifted": sum(abs(r["delta_p"]) > 1e-12 for r in v),
                        "n_exactly_zero": sum(r["delta_p"] == 0.0 for r in v),
                        "n_flipped": sum(r["flipped"] for r in v),
                        "mean_abs_delta_p": float(np.mean([abs(r["delta_p"]) for r in v])) if v else None}
                    for t, v in reps.items()},
        "all": reps}
    sf = rd / "surreal_forget.json"
    out["surreal_forget"] = json.loads(sf.read_text()) if sf.exists() else None
    dv = rd / "deviations.json"
    out["deviations"] = json.loads(dv.read_text()) if dv.exists() else []
    (rd / "regulars.json").write_text(json.dumps(out, indent=1))
    return out


def fmt(c):
    return "n/a" if c.get("mean") is None else f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] n={c['n']}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="reg01")
    a = ap.parse_args(argv)
    o = analyze(a.run)
    print("seeds", o["n_seeds"], o["seeds"])
    for a_, v in o["arms"].items():
        print(f"{a_:26s} first {v['win_rate_first']['mean']:.3f} returns {v['win_rate_returns']['mean']:.3f} "
              f"brier_ret {v['brier_returns']['mean']:.3f} | k: " +
              " ".join(f"{v['by_encounter'][str(k)]['win_rate']['mean']:.3f}" for k in range(1, 6)))
    h = o["hypotheses"]
    print("H8 vs noid", fmt(h["H8"]["vs_noid"]), "vs pooled", fmt(h["H8"]["vs_logreg_pooled"]), "holds", h["H8"]["holds"])
    print("H9 win", fmt(h["H9"]["win_rate"]), "brier", fmt(h["H9"]["brier"]))
    for F in ("Avery", "Blake"):
        f = o["forgetting"][F]
        print(f"H10 forget:{F} holds={f['holds']} n={f['n_seeds']}")
        for nm, v in f["per_regular"].items():
            print(f"   {nm:6s} Δwin {fmt(v['delta_win'])} Δbrier {fmt(v['delta_brier'])} holds={v['holds']}")
        print("   relearning", {k: v for k, v in f["relearning"].items() if k != "per_seed"})
    print("replays", json.dumps(o["decision_replays"]["summary"]))


if __name__ == "__main__":
    main()
