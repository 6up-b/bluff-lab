"""Experiment mem01 — a fair test of memory (owner: llm-arms; pre-registered in INTERFACES.md "Status: llm-arms").

Same model (Sonnet 5.5), same system prompt, same raw information (info: raw); only the memory block differs:
llm_nomem vs llm_episodic vs llm_dossier. Reference arms: counter, logreg_bow, profiler, profiler_plus (all raw).

    .venv/bin/python -m blab.experiments.mem01 estimate
    .venv/bin/python -m blab.experiments.mem01 run [--seeds 3001-3008] [--tables ...] [--launch-deadline HH:MM]
    .venv/bin/python -m blab.experiments.mem01 analyze

Output: runs/mem01/<wid>/ (standard world layout), runs/mem01/results.json, runs/mem01/summary.md.
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures as cf
import datetime as dt
import json
import threading
import time
from pathlib import Path

import numpy as np

OUT = Path("runs/mem01")
ARMS = ["counter", "logreg_bow", "llm_nomem", "llm_episodic", "llm_dossier", "profiler", "profiler_plus"]
LLM_ARMS = ["llm_nomem", "llm_episodic", "llm_dossier"]
REFLECTIVE = {"llm_dossier", "profiler", "profiler_plus"}
TABLES = ["telltale", "longcon", "tell_flipper", "tell_conjunctive"]
GAMES = 20
BASELINE = "llm_nomem"
# the seat whose tell / change point is under study, per table
TELL_SEAT = {"telltale": "p3", "longcon": "p2", "tell_flipper": "p3", "tell_conjunctive": "p3"}
CHANGE_SEAT = {"longcon": "p3", "tell_flipper": "p3"}


def wid_of(table: str, arm: str, seed: int) -> str:
    return f"{table}-{arm}-raw-s{seed}"          # same naming as core's mem01 reference-arm runner


# Raw-mode reflection prompt: IDENTICAL to core's mem01 runner (runs/mem01/core_runner.py), so llm_dossier's
# reflection is the same machinery as the raw profiler / profiler_plus arms.
RAW_REPL = [
    ("| formal_register | bool | utterance has no contractions and >= 4 words |\n",
     "| word_count | int | number of whitespace-separated words in the utterance |\n"),
    ('(e.g. answer_formal_register)', '(e.g. answer_utterance, answer_word_count)'),
    ("== != < <= > >= contains regex no_contractions", "== != < <= > >= contains regex"),
    ("(contains/regex/no_contractions apply to string features", "(contains/regex apply to string features"),
    ('{"all": [{"f":"formal_register","op":"==","v":true}, {"f":"hand_size","op":"<=","v":2}]}',
     '{"all": [{"f":"hand_size","op":"<=","v":2}, {"f":"claim_count","op":">=","v":2}]}'),
    ('e.g. "Player 3 speaks formally (no contractions) when bluffing"', 'e.g. "Player 3 bluffs more on multi-card claims"'),
]


def apply_raw_reflection_prompt() -> None:
    from blab.orchestrator import llm
    S = llm.STABLE_SYSTEM
    if "formal_register" in S:
        for a, b in RAW_REPL:
            assert a in S, a
            S = S.replace(a, b)
        assert "formal" not in S.lower() and "no_contractions" not in S
        llm.STABLE_SYSTEM = S

    def _raw_validate(pred):
        from blab.hypotheses import validate_predicate
        return validate_predicate(pred, "raw")
    llm._validate_pred = _raw_validate


def specs(seeds: list[int], tables: list[str], arms: list[str]) -> list[dict]:
    """Priority order: seed-major (complete paired blocks first), then table, then the slowest arms first."""
    slow = {"llm_dossier": 0, "llm_episodic": 1, "llm_nomem": 2, "profiler_plus": 3, "profiler": 4}
    out = []
    for s in seeds:
        for t in tables:
            for a in sorted(arms, key=lambda a: slow.get(a, 9)):
                out.append({"seed": s, "table": t, "arm": a, "wid": wid_of(t, a, s)})
    return out


def run_world(sp: dict, rule: str = "threshold") -> dict:
    from blab.agents import llm_agents as L
    from blab.world import World, make_seats
    seats = make_seats(sp["table"], sp["arm"], sp["seed"])
    params = {"info": "raw", "respond_rule": rule}
    if sp["arm"] in LLM_ARMS:
        params["llm_cache"] = str(OUT / sp["wid"] / "llm_cache.jsonl")
    seats[0]["params"] = {**seats[0].get("params", {}), **params}
    refl = L.guarded_reflector if sp["arm"] in REFLECTIVE else None
    t0 = time.time()
    w = World(wid=sp["wid"], seats=seats, seed=sp["seed"], games=GAMES, reflect_every=2, out_dir=OUT, run="mem01",
              box="local", reflector=refl, table=sp["table"], learning=True, reflect_timeout=150.0)
    L.cache_reset(params.get("llm_cache"))
    res = asyncio.run(w.run())
    return {**sp, "secs": round(time.time() - t0, 1), "win_rate": res.get("win_rate")}


def _parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _at(hhmm: str) -> float:
    h, m = map(int, hhmm.split(":"))
    now = dt.datetime.now()
    return now.replace(hour=h, minute=m, second=0, microsecond=0).timestamp()


def run(seeds: list[int], tables: list[str], arms: list[str], launch_deadline: str | None, hard_stop: str | None,
        workers: int = 48) -> None:
    from blab.agents import llm_agents as L
    from blab.env import load_env
    load_env()
    apply_raw_reflection_prompt()
    OUT.mkdir(parents=True, exist_ok=True)
    todo = specs(seeds, tables, arms)
    t_launch = _at(launch_deadline) if launch_deadline else float("inf")
    t_stop = _at(hard_stop) if hard_stop else float("inf")
    done, skipped, failed = [], [], []
    lock = threading.Lock()
    log = open(OUT / "run_log.jsonl", "a")
    t0 = time.time()

    def job(sp):
        if time.time() > t_launch:
            with lock:
                skipped.append(sp)
            return None
        try:
            r = run_world(sp)
        except Exception as e:  # noqa: BLE001
            with lock:
                failed.append({**sp, "error": f"{type(e).__name__}: {e}"})
                log.write(json.dumps({**sp, "error": str(e)}) + "\n")
                log.flush()
            return None
        with lock:
            done.append(r)
            log.write(json.dumps(r) + "\n")
            log.flush()
        return r

    ex = cf.ThreadPoolExecutor(max_workers=workers)
    futs = [ex.submit(job, sp) for sp in todo]
    last = 0.0
    while not all(f.done() for f in futs):
        time.sleep(2)
        if time.time() - last > 60:
            last = time.time()
            st = dict(L.STATS)
            print(f"[mem01] t+{time.time() - t0:.0f}s done={len(done)}/{len(todo)} failed={len(failed)} "
                  f"skipped={len(skipped)} calls={st['calls']} errors={st['errors']} "
                  f"mean_lat={st['latency_ms_sum'] / max(1, st['calls'] - st['errors']):.0f}ms "
                  f"reflections={st.get('reflections', 0)} refl_err={st.get('reflection_errors', 0)} "
                  f"providers={st['by_provider']}", flush=True)
        if time.time() > t_stop:
            print("[mem01] HARD STOP reached; abandoning running worlds (they will be excluded as incomplete)")
            break
    ex.shutdown(wait=False, cancel_futures=True)
    st = dict(L.STATS)
    (OUT / "run_stats.json").write_text(json.dumps({"done": len(done), "failed": failed, "skipped": len(skipped),
                                                    "stats": st, "secs": round(time.time() - t0)}, indent=1))
    print(f"[mem01] finished: done={len(done)} failed={len(failed)} skipped={len(skipped)} stats={st}")


# ------------------------------------------------------------------ estimate

def estimate(seeds, tables, arms, decisions_per_world=None, lat_s=None, refl_s=20.0, conc=16):
    """Call count + wall time, measured from a pilot world when available (runs/mem01/pilot.json)."""
    pilot = OUT / "pilot.json"
    pm = json.loads(pilot.read_text()) if pilot.exists() else {}
    dpw = decisions_per_world or pm.get("calls_per_world", 200)
    lat = lat_s or pm.get("mean_latency_s", 2.2)
    n_llm = len(seeds) * len(tables) * sum(1 for a in arms if a in LLM_ARMS)
    calls = n_llm * dpw
    refl = len(seeds) * len(tables) * (10 * ("llm_dossier" in arms) + 10 * ("profiler" in arms)
                                        + 20 * ("profiler_plus" in arms))
    slot_s = calls * lat + refl * refl_s
    wall = slot_s / conc
    print(f"[mem01 estimate] seeds={len(seeds)} tables={len(tables)} arms={len(arms)}: LLM worlds={n_llm}, "
          f"per-decision calls≈{calls} (≈{dpw}/world, measured={bool(pm)}), reflection calls≈{refl}; "
          f"latency {lat:.1f}s/call, reflection {refl_s:.0f}s; ≤{conc} concurrent → wall ≈ {wall / 60:.1f} min")
    return wall


# ------------------------------------------------------------------ analysis

def _load(wdir: Path) -> list[dict] | None:
    p = wdir / "events.jsonl"
    if not p.exists():
        return None
    ev = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if not ev or ev[-1]["type"] != "world_end":
        return None
    return ev


def world_metrics(ev: list[dict], table: str) -> dict:
    meta_end = ev[-1]["payload"]
    ends = {e["game"]: e["payload"]["winner"] for e in ev if e["type"] == "game_end"}
    reveals = {(e["game"], e["round"]): e["payload"] for e in ev if e["type"] == "reveal"}
    resp = [e for e in ev if e["type"] == "decision" and e["actor"] == "p1" and e["payload"].get("kind") == "respond"]
    brier_g: dict[int, list[float]] = {}
    for e in resp:
        p = e["payload"]
        if p["action"] == "challenge" and (e["game"], e["round"]) in reveals:
            brier_g.setdefault(e["game"], []).append((p["p_final"] - float(reveals[(e["game"], e["round"])]["was_bluff"])) ** 2)
    allb = [x for v in brier_g.values() for x in v]
    buckets = [(1, 5), (6, 10), (11, 15), (16, 20)]
    out = {"win_rate": float(np.mean([w == "p1" for w in ends.values()])),
           "win_rate_11_20": float(np.mean([ends[g] == "p1" for g in ends if 11 <= g <= 20])),
           "brier": float(np.mean(allb)) if allb else None, "brier_n": len(allb),
           "brier_buckets": [(float(np.mean(v)) if (v := [x for g in range(a, b + 1) for x in brier_g.get(g, [])]) else None)
                             for a, b in buckets],
           "win_buckets": [float(np.mean([ends[g] == "p1" for g in range(a, b + 1) if g in ends])) for a, b in buckets]}
    llm = [e["payload"].get("llm") for e in resp if e["payload"].get("llm")]
    if llm:
        out["llm_calls"] = sum(1 for x in llm if not x.get("skipped"))
        out["llm_errors"] = sum(1 for x in llm if x.get("error"))
        lat = [x["latency_ms"] for x in llm if x.get("latency_ms") and not x.get("error")]
        out["llm_latency_ms"] = float(np.mean(lat)) if lat else None
    out["reflections"] = sum(1 for e in ev if e["type"] == "reflect_result")
    out["reflection_errors"] = sum(1 for e in ev if e["type"] == "reflect_result" and e["payload"].get("error"))
    # tell discovery (hypothesis-bearing arms): first confirmed verbal_tell line about the tell seat
    ts = TELL_SEAT.get(table)
    conf = [e for e in ev if e["type"] == "hypothesis" and e["payload"].get("about") == ts
            and e["payload"].get("aspect") == "verbal_tell" and e["payload"].get("status") == "confirmed"]
    out["discovery_game"] = conf[0]["game"] if conf else None
    out["discovery_text"] = conf[0]["payload"].get("text") if conf else None
    out["discovery_predicate"] = conf[0]["payload"].get("predicate") if conf else None
    # exploitation + recovery after the change point (longcon sleeper / tell_flipper inversion)
    cs = CHANGE_SEAT.get(table)
    if cs:
        claims = [e for e in ev if e["type"] == "claim" and e["actor"] == cs]
        gr, i = {}, 0
        for e in ev:
            if e["type"] == "deal":
                i += 1
                gr[(e["game"], e["round"])] = i
        if table == "longcon":
            strat = [c for c in claims if c["payload"].get("bluff_kind") == "strategic"]
            cp_idx = gr[(strat[0]["game"], strat[0]["round"])] if strat else None
        else:
            g1 = next((sp.get("params", {}).get("G1") for sp in ev[0]["payload"]["seats"] if sp["seat"] == cs), None)
            cp_idx = min((v for (g, r), v in gr.items() if g >= int(g1 or 99)), default=None)
        out["change_round"] = cp_idx

        def judged(e):
            prev = [c for c in claims if c["seq"] < e["seq"] and (c["game"], c["round"]) == (e["game"], e["round"])]
            return prev[-1]["payload"] if prev else None
        after = [e for e in resp if e["payload"].get("target") == cs and cp_idx is not None
                 and gr[(e["game"], e["round"])] >= cp_idx]
        lt = 0
        rec = None
        streak = 0
        for e in after:
            c = judged(e)
            if not c or c.get("bluff_kind") != "strategic":
                continue
            if e["payload"]["action"] == "play_on":
                lt += 1
            hi = e["payload"]["p_final"] > 0.5
            streak = streak + 1 if hi else 0
            if streak >= 2 and rec is None:
                rec = gr[(e["game"], e["round"])] - cp_idx
        out["let_through_after_change"] = lt if cp_idx is not None else None
        out["recovery_rounds"] = rec
        out["rtd_world"] = meta_end.get("rtd")
    return out


def _boot(x: np.ndarray, rng, n=2000):
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)
    if len(x) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    if len(x) == 1:
        return {"mean": float(x[0]), "lo": None, "hi": None, "n": 1}
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return {"mean": float(x.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)),
            "n": int(len(x))}


def analyze() -> dict:
    rng = np.random.default_rng(20261016)
    W: dict[tuple[str, str, int], dict] = {}
    for d in sorted(OUT.iterdir()) if OUT.exists() else []:
        if not d.is_dir() or d.name.startswith("pilot-"):
            continue
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else None
        if not meta:
            continue
        ev = _load(d)
        if ev is None:
            continue
        W[(meta["table"], meta["agent"], int(meta["seed"]))] = world_metrics(ev, meta["table"])
    tables = [t for t in TABLES if any(k[0] == t for k in W)]
    arms = [a for a in ARMS if any(k[1] == a for k in W)]
    res = {"experiment": "mem01", "baseline": BASELINE, "setting": "raw", "respond_rule": "threshold",
           "model": "Sonnet 5.5 (us.anthropic.claude-sonnet-5-5 on Bedrock / claude-sonnet-5-5 direct)",
           "tables": {}, "worlds": {f"{t}|{a}|{s}": m for (t, a, s), m in W.items()}}
    metrics = ["win_rate", "win_rate_11_20", "brier"]
    for t in tables:
        seeds_all = sorted({k[2] for k in W if k[0] == t})
        # paired set: seeds where every LLM arm present on this table finished
        tr: dict = {"arms": {}, "delta_vs_" + BASELINE: {}, "seeds": seeds_all}
        for a in arms:
            ws = {s: W[(t, a, s)] for s in seeds_all if (t, a, s) in W}
            if not ws:
                continue
            row = {m: _boot([w[m] for w in ws.values()], rng) for m in metrics}
            row["n_worlds"] = len(ws)
            row["brier_curve"] = [_boot([w["brier_buckets"][i] for w in ws.values()], rng) for i in range(4)]
            row["win_curve"] = [_boot([w["win_buckets"][i] for w in ws.values()], rng) for i in range(4)]
            if any(w.get("llm_calls") is not None for w in ws.values()):
                row["llm_calls"] = int(sum(w.get("llm_calls", 0) for w in ws.values()))
                row["llm_errors"] = int(sum(w.get("llm_errors", 0) for w in ws.values()))
            row["reflections"] = int(sum(w["reflections"] for w in ws.values()))
            row["reflection_errors"] = int(sum(w["reflection_errors"] for w in ws.values()))
            dg = [w["discovery_game"] for w in ws.values()]
            if a in REFLECTIVE:
                row["discovery"] = {"found": sum(1 for g in dg if g is not None), "of": len(dg),
                                    "median_game": (float(np.median([g for g in dg if g is not None]))
                                                    if any(g is not None for g in dg) else None),
                                    "texts": [{"seed": s, "game": w["discovery_game"], "text": w["discovery_text"],
                                               "predicate": w["discovery_predicate"]}
                                              for s, w in ws.items() if w["discovery_game"] is not None]}
            if t in CHANGE_SEAT:
                row["let_through_after_change"] = _boot([w.get("let_through_after_change") for w in ws.values()], rng)
                rr = [w.get("recovery_rounds") for w in ws.values()]
                row["recovery_rounds"] = {"recovered": sum(1 for r in rr if r is not None), "of": len(rr),
                                          "median": (float(np.median([r for r in rr if r is not None]))
                                                     if any(r is not None for r in rr) else None)}
            tr["arms"][a] = row
            if a != BASELINE and any((t, BASELINE, s) in W for s in ws):
                pair = [s for s in ws if (t, BASELINE, s) in W]
                dd = {}
                for m in metrics + (["let_through_after_change"] if t in CHANGE_SEAT else []):
                    diffs = [ws[s][m] - W[(t, BASELINE, s)][m] for s in pair
                             if ws[s].get(m) is not None and W[(t, BASELINE, s)].get(m) is not None]
                    dd[m] = _boot(diffs, rng)
                dd["paired_seeds"] = pair
                tr["delta_vs_" + BASELINE][a] = dd
        res["tables"][t] = tr
    # pooled across tables (paired by table×seed)
    pooled = {}
    for a in arms:
        if a == BASELINE:
            continue
        dd = {}
        for m in metrics:
            diffs = [W[k][m] - W[(k[0], BASELINE, k[2])][m] for k in W if k[1] == a and (k[0], BASELINE, k[2]) in W
                     and W[k].get(m) is not None and W[(k[0], BASELINE, k[2])].get(m) is not None]
            dd[m] = _boot(diffs, rng)
        pooled[a] = dd
    res["pooled_delta_vs_" + BASELINE] = pooled
    # raw transcript vs structured: llm_dossier − llm_episodic, paired by table×seed
    dd = {}
    for m in metrics:
        diffs = [W[k][m] - W[(k[0], "llm_episodic", k[2])][m] for k in W if k[1] == "llm_dossier"
                 and (k[0], "llm_episodic", k[2]) in W and W[k].get(m) is not None
                 and W[(k[0], "llm_episodic", k[2])].get(m) is not None]
        dd[m] = _boot(diffs, rng)
    res["pooled_dossier_minus_episodic"] = dd
    res["llm_totals"] = {a: {"calls": int(sum(w.get("llm_calls", 0) for k, w in W.items() if k[1] == a)),
                             "errors": int(sum(w.get("llm_errors", 0) for k, w in W.items() if k[1] == a)),
                             "reflections": int(sum(w["reflections"] for k, w in W.items() if k[1] == a)),
                             "reflection_errors": int(sum(w["reflection_errors"] for k, w in W.items() if k[1] == a)),
                             "worlds": sum(1 for k in W if k[1] == a)} for a in LLM_ARMS}

    def verdict(d, m):
        if d["n"] < 2 or d["lo"] is None:
            return "insufficient"
        better = (d["lo"] > 0) if m.startswith("win") else (d["hi"] < 0)
        worse = (d["hi"] < 0) if m.startswith("win") else (d["lo"] > 0)
        return "helps" if better else ("hurts" if worse else "no detectable difference")
    res["verdicts"] = {a: {m: verdict(pooled[a][m], m) for m in metrics} for a in pooled}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    (OUT / "summary.md").write_text(summary_md(res))
    print(summary_md(res))
    return res


def _f(d, k=3):
    if not d or d.get("mean") is None:
        return "–"
    if d.get("lo") is None:
        return f"{d['mean']:.{k}f}"
    return f"{d['mean']:.{k}f} [{d['lo']:.{k}f}, {d['hi']:.{k}f}]"


def summary_md(res: dict) -> str:
    L = ["# mem01 — does memory help? (same model, same prompt, same raw information)", "",
         f"Setting: raw information (no formal_register). Respond rule: threshold. Model: {res['model']}. "
         f"Baseline for Δ: {res['baseline']} (paired by seed; bootstrap 95% CI, 2000 resamples).", ""]
    for t, tr in res["tables"].items():
        L += [f"## {t} (seeds {tr['seeds'][0]}–{tr['seeds'][-1]}, n per arm below)", "",
              "| arm | n | win rate | win g11–20 | Brier | Δwin vs nomem | ΔBrier vs nomem | LLM calls/errors |",
              "|---|---|---|---|---|---|---|---|"]
        for a, r in tr["arms"].items():
            d = tr["delta_vs_" + res["baseline"]].get(a, {})
            L.append(f"| {a} | {r['n_worlds']} | {_f(r['win_rate'])} | {_f(r['win_rate_11_20'])} | {_f(r['brier'])} | "
                     f"{_f(d.get('win_rate'))} | {_f(d.get('brier'))} | "
                     f"{(str(r.get('llm_calls')) + '/' + str(r.get('llm_errors'))) if 'llm_calls' in r else '–'} |")
        L.append("")
        L.append("Brier by game bucket (1–5 / 6–10 / 11–15 / 16–20):")
        for a, r in tr["arms"].items():
            L.append(f"- {a}: " + " / ".join(_f(b, 3).split(" [")[0] for b in r["brier_curve"]))
        disc = [(a, r["discovery"]) for a, r in tr["arms"].items() if "discovery" in r]
        if disc:
            L.append("")
            L.append("Tell discovery (first confirmed verbal_tell line about the tell seat):")
            for a, dd in disc:
                ex = dd["texts"][0] if dd["texts"] else None
                L.append(f"- {a}: {dd['found']}/{dd['of']} worlds, median game {dd['median_game']}"
                         + (f"; e.g. s{ex['seed']} g{ex['game']}: \"{ex['text']}\" {json.dumps(ex['predicate'])}" if ex else ""))
        if any("let_through_after_change" in r for r in tr["arms"].values()):
            L.append("")
            L.append("After the change point (strategic bluffs by the changing seat let through; recovery = rounds "
                     "until p_final>0.5 on 2 consecutive strategic bluffs):")
            for a, r in tr["arms"].items():
                if "let_through_after_change" in r:
                    rr = r["recovery_rounds"]
                    L.append(f"- {a}: let through {_f(r['let_through_after_change'], 1)}; recovered {rr['recovered']}/{rr['of']}"
                             f", median {rr['median']} rounds")
        L.append("")
    L += ["## Pooled over tables (paired by table×seed), Δ vs llm_nomem", "",
          "| arm | Δ win rate | Δ win g11–20 | Δ Brier | verdict (win / late / Brier) |", "|---|---|---|---|---|"]
    for a, d in res["pooled_delta_vs_" + res["baseline"]].items():
        v = res["verdicts"][a]
        L.append(f"| {a} | {_f(d['win_rate'])} | {_f(d['win_rate_11_20'])} | {_f(d['brier'])} | "
                 f"{v['win_rate']} / {v['win_rate_11_20']} / {v['brier']} |")
    d = res.get("pooled_dossier_minus_episodic") or {}
    L += ["", f"Structured vs raw transcript (llm_dossier − llm_episodic, pooled paired): Δ win {_f(d.get('win_rate'))}, "
          f"Δ win g11–20 {_f(d.get('win_rate_11_20'))}, Δ Brier {_f(d.get('brier'))}", "",
          "LLM calls / errors (fallback p_bluff = p_cards) / reflections (errors): "
          + "; ".join(f"{a} {v['calls']}/{v['errors']}/{v['reflections']}({v['reflection_errors']}) over {v['worlds']} worlds"
                      for a, v in (res.get("llm_totals") or {}).items())]
    L += ["", "\"Helps\" = paired 95% CI excludes 0 in the better direction (pre-registered criterion)."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ pilot

def pilot(table="telltale", seed=2999, arm="llm_nomem", games=4):
    """One short world to measure calls/game and latency (seed outside the experiment's range)."""
    from blab.agents import llm_agents as L
    from blab.env import load_env
    global GAMES
    load_env()
    apply_raw_reflection_prompt()
    g0, GAMES = GAMES, games
    t0 = time.time()
    run_world({"seed": seed, "table": table, "arm": arm, "wid": f"pilot-{table}-{arm}-s{seed}"})
    GAMES = g0
    st = dict(L.STATS)
    ok = max(1, st["calls"] - st["errors"])
    pm = {"calls_per_world": round(st["calls"] / games * 20), "mean_latency_s": st["latency_ms_sum"] / ok / 1000,
          "pilot_secs": round(time.time() - t0, 1), "pilot_games": games, "stats": st}
    (OUT / "pilot.json").write_text(json.dumps(pm, indent=1))
    print(json.dumps(pm, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["estimate", "run", "analyze", "pilot"])
    ap.add_argument("--seeds", default="3001-3008")
    ap.add_argument("--tables", default=",".join(TABLES))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--launch-deadline", default=None)
    ap.add_argument("--hard-stop", default=None)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--pilot-arm", default="llm_nomem")
    ap.add_argument("--pilot-games", type=int, default=4)
    a = ap.parse_args()
    seeds, tables, arms = _parse_seeds(a.seeds), a.tables.split(","), a.arms.split(",")
    if a.cmd == "estimate":
        estimate(seeds, tables, arms)
    elif a.cmd == "pilot":
        pilot(arm=a.pilot_arm, games=a.pilot_games)
    elif a.cmd == "run":
        estimate(seeds, tables, arms)
        run(seeds, tables, arms, a.launch_deadline, a.hard_stop, a.workers)
        analyze()
    else:
        analyze()


if __name__ == "__main__":
    main()
