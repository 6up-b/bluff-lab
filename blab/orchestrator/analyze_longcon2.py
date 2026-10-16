"""lc02 — Long Con at scale (PREREG.md). Metrics defined before the run:

- detection rate: fraction of longcon worlds with a detection (R_d, §6 rule via analyze.detect_round) after R_b
- median RTD among detected worlds (rounds, R_d − R_b)
- Kaplan–Meier median RTD, undetected worlds censored at the end of game 20 (time = rounds from R_b to the end)
- false alarm = `false_alarm_belief` (a belief_superseded fires) on the falsealarm table [control, control]
- exploitation = sleeper bluffs p1 let through (played on) in games ≥ G0
- trust curves: per game, the sleeper's true honesty vs each agent's estimate (1 − mean q_used about the sleeper)

Arms are (agent label, respond rule); Claude reflection (`profiler+claude`) and stub (`profiler`) are never pooled.
Bootstrap 95% CIs over worlds (seeds), 2000 resamples.

    python -m blab.orchestrator.analyze_longcon2 --run lc02   → runs/lc02/longcon2.json, runs/lc02/longcon2.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from blab.orchestrator import analyze as A

B = 2000


def _world(d: str, branches: bool = False) -> dict | None:
    w = A.load_world(Path(d))
    if not w or not w["done"] or (bool(w["vkey"]) != branches):
        return None
    m = A.world_metrics(w)
    out = {"wid": w["wid"], "intervention": (w["variant"] or {}).get("intervention"), "label": w["label"], "rule": w["rule"], "reflection": w["reflection"],
           "table": w["table"], "seed": w["seed"]}
    if w["table"] == "falsealarm":
        out["false_alarm_belief"] = m.get("false_alarm_belief")
        out["false_alarm_any"] = m.get("false_alarm")
    sl = A.seat_with_role(w, "sleeper")
    if sl and w["table"] == "longcon":
        evs = w["events"]
        n_rounds = sum(1 for e in evs if e["type"] == "deal")
        g0, r_b, rtd = m.get("g0"), m.get("r_b"), m.get("rtd")
        rds = A.respond_decisions(evs)
        exploit = sum(1 for d in rds if d["ev"]["payload"].get("target") == sl and g0 is not None
                      and d["ev"]["game"] >= g0 and d["truth_bluff"] and d["ev"]["payload"].get("action") == "play_on")
        lost = sum(1 for e in evs if e["type"] == "reveal" and e["payload"]["claimant"] == sl
                   and e["payload"]["loser"] == "p1" and g0 is not None and e["game"] >= g0)
        out.update(g0=g0, r_b=r_b, rtd=rtd, detected=None if r_b is None else float(rtd is not None),
                   censor_t=None if r_b is None else (rtd if rtd is not None else max(0, n_rounds - r_b)),
                   detect_how=m.get("detect_how"), detection_game=m.get("detection_game"),
                   exploitation=exploit if g0 is not None else None, lives_lost_after_g0=lost if g0 is not None else None,
                   trust_by_game=m.get("trust_by_game"), honesty_by_game=m.get("honesty_by_game"))
    return out


def km_median(times: list[float], events: list[int]) -> float | None:
    """Kaplan–Meier median survival time (time to detection); None if S never drops to ≤ 0.5."""
    if not times:
        return None
    t = np.asarray(times, float)
    e = np.asarray(events, int)
    s = 1.0
    for u in np.unique(t[e == 1]):
        at_risk = np.sum(t >= u)
        d = np.sum((t == u) & (e == 1))
        if at_risk:
            s *= 1 - d / at_risk
        if s <= 0.5:
            return float(u)
    return None


def km_boot(times, events, rng) -> dict:
    est = km_median(times, events)
    if len(times) < 2:
        return {"median": est, "lo": None, "hi": None, "n": len(times)}
    t, e = np.asarray(times), np.asarray(events)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(t), len(t))
        v = km_median(list(t[i]), list(e[i]))
        bs.append(np.inf if v is None else v)
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    return {"median": est, "lo": None if np.isinf(lo) else float(lo), "hi": None if np.isinf(hi) else float(hi),
            "n": len(times), "n_events": int(np.sum(e)), "undefined_frac_boot": float(np.mean(np.isinf(bs)))}


def analyze_lc02(run: str = "lc02", root: str = "runs") -> dict:
    rd = Path(root) / run
    specs = json.loads((rd / "worlds.json").read_text())
    dirs = [str(rd / s["wid"]) for s in specs if (rd / s["wid"] / "events.jsonl").exists()]
    with ProcessPoolExecutor() as ex:
        ws = [w for w in ex.map(_world, dirs, chunksize=16) if w]
    rng = np.random.default_rng(0)
    arms: dict = defaultdict(lambda: {"longcon": [], "falsealarm": []})
    for w in ws:
        if w["table"] in ("longcon", "falsealarm"):
            arms[(w["label"], w["rule"])][w["table"]].append(w)
    res = {"run": run, "n_worlds": len(ws), "prereg": "PREREG.md (commit 2167c1a)", "seeds": "6000-6119",
           "definitions": __doc__, "arms": {}}
    for (label, rule), t in sorted(arms.items()):
        lc, fa = t["longcon"], t["falsealarm"]
        lcb = [w for w in lc if w.get("r_b") is not None]
        rtds = [w["rtd"] for w in lcb if w["rtd"] is not None]
        arm = {
            "label": label, "rule": rule, "reflection": (lc or fa or [{}])[0].get("reflection"),
            "n_longcon": len(lc), "n_with_strategic_bluff": len(lcb), "n_falsealarm": len(fa),
            "detection_rate": A.boot(w["detected"] for w in lcb),
            "rtd_median_detected": A.med(rtds) | {"values": None},
            "rtd_km_median": km_boot([w["censor_t"] for w in lcb], [int(w["detected"]) for w in lcb], rng),
            "exploitation": A.boot(w["exploitation"] for w in lc),
            "lives_lost_after_g0": A.boot(w["lives_lost_after_g0"] for w in lc),
            "false_alarm_belief": A.boot(w["false_alarm_belief"] for w in fa),
            "false_alarm_any": A.boot(w["false_alarm_any"] for w in fa),
        }
        tb, hb = defaultdict(list), defaultdict(list)
        for w in lc:
            for g, v in (w.get("trust_by_game") or {}).items():
                tb[int(g)].append(v)
            for g, v in (w.get("honesty_by_game") or {}).items():
                if v is not None:
                    hb[int(g)].append(v)
        arm["trust_curve"] = [{"game": g, **A.boot(tb[g])} for g in sorted(tb)]
        arm["true_honesty_curve"] = [{"game": g, **A.boot(hb[g])} for g in sorted(hb)]
        arm["g0_values"] = sorted({w["g0"] for w in lc if w.get("g0") is not None})
        res["arms"][f"{label}|{rule}"] = arm
    cbf = rd / "checkpoint_branches.json"
    if cbf.exists():
        cb = [b for b in json.loads(cbf.read_text()) if "wid" in b]
        from functools import partial
        with ProcessPoolExecutor() as ex:
            bws = [w for w in ex.map(partial(_world, branches=True), [str(rd / b["wid"]) for b in cb], chunksize=8) if w]
        by = defaultdict(list)
        for w in bws:
            by[w["intervention"]].append(w)
        res["checkpoint_branches"] = {
            "kind": "checkpoint branch (restore ckpt g<G0-1>, same seeds/RNG, learning ON, stub reflection) — NOT a VM fork",
            "parent_arm": "profiler|threshold (stub)",
            "interventions": {iv: {"n": len(v), "detection_rate": A.boot(w["detected"] for w in v if w.get("r_b") is not None),
                                   "rtd_km_median": km_boot([w["censor_t"] for w in v if w.get("r_b") is not None],
                                                            [int(w["detected"]) for w in v if w.get("r_b") is not None], rng),
                                   "exploitation": A.boot(w["exploitation"] for w in v),
                                   "lives_lost_after_g0": A.boot(w["lives_lost_after_g0"] for w in v)}
                              for iv, v in sorted(by.items())}}
    (rd / "longcon2.json").write_text(json.dumps(res, indent=1, default=float))
    (rd / "longcon2.md").write_text(summary(res))
    print(summary(res))
    return res


def _f(x, pct=False):
    if not x or x.get("mean") is None:
        return "—"
    k = 100 if pct else 1
    s = f"{x['mean'] * k:.1f}" if pct else f"{x['mean']:.2f}"
    if x.get("lo") is not None:
        s += f" [{x['lo'] * k:.1f}, {x['hi'] * k:.1f}]" if pct else f" [{x['lo']:.2f}, {x['hi']:.2f}]"
    return s + ("%" if pct else "")


def summary(res: dict) -> str:
    lines = [f"# lc02 Long Con at scale — {res['n_worlds']} worlds, seeds {res['seeds']}", "",
             "| arm | rule | n | detect rate | RTD median (detected) | KM median RTD [95% CI] | exploitation (bluffs let through ≥G0) | false alarm (belief) |",
             "|---|---|---|---|---|---|---|---|"]
    for a in res["arms"].values():
        km = a["rtd_km_median"]
        kms = "—" if km["median"] is None else f"{km['median']:.0f}"
        if km.get("lo") is not None or km.get("hi") is not None:
            kms += f" [{km['lo'] if km['lo'] is not None else '∞'}, {km['hi'] if km['hi'] is not None else '∞'}]"
        md = a["rtd_median_detected"]
        lines.append(f"| {a['label']} | {a['rule']} | {a['n_longcon']}/{a['n_falsealarm']} | "
                     f"{_f(a['detection_rate'], True)} | {md['median'] if md['median'] is not None else '—'} (n={md['n']}) | "
                     f"{kms} | {_f(a['exploitation'])} | {_f(a['false_alarm_belief'], True)} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="lc02")
    ap.add_argument("--root", default="runs")
    a = ap.parse_args()
    analyze_lc02(a.run, a.root)
