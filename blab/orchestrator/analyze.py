"""`blab analyze --run R` — SPEC §6 metrics from real event logs, with bootstrap 95% CIs over seeds.

Reads every `events.jsonl` under `runs/<run>/` (any depth; `--run local` reads `runs/local/`).
Writes `runs/<run>/results.json`, `runs/<run>/summary.md`, and the Show inputs
`runs/<run>/ablation.json` + `runs/<run>/longcon.json`. Nothing here invents a number: a metric that
cannot be computed from the logs is `None` and renders as "—" / "awaiting data".
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from blab.events import read_events

TABLES = {("control", "telltale"): "telltale", ("sleeper", "telltale"): "longcon",
          ("control", "control"): "falsealarm", ("control", "telltale_overt"): "telltale_overt"}
AGENT_ORDER = ["counter", "bayes", "bayes_discount", "logreg_bow", "llm_episodic",
               "profiler_amnesic", "profiler", "profiler_blind"]
ABL_ORDER = ["as_of:0", "as_of:4", "as_of:6", "as_of:8", "full", "minus:verbal_tell",
             "false_memory", "transplant:same_role", "transplant:other_role"]
ABL_LABEL = {"as_of:0": "erased", "as_of:4": "as of g4", "as_of:6": "as of g6", "as_of:8": "as of g8",
             "full": "full", "minus:verbal_tell": "minus tell", "false_memory": "false memory",
             "transplant:same_role": "transplant, same role", "transplant:other_role": "transplant, other role"}
LATE_FROM = 11
B_RESAMPLES = 2000


# ------------------------------------------------------------------ stats
def boot(values: Iterable[float | None], rng: np.random.Generator | None = None) -> dict:
    """Mean with a percentile-bootstrap 95% CI over worlds (seeds). None values are dropped."""
    xs = np.array([float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))])
    if len(xs) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    if len(xs) == 1:
        return {"mean": float(xs[0]), "lo": None, "hi": None, "n": 1}
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(xs), size=(B_RESAMPLES, len(xs)))
    ms = xs[idx].mean(axis=1)
    return {"mean": float(xs.mean()), "lo": float(np.percentile(ms, 2.5)), "hi": float(np.percentile(ms, 97.5)),
            "n": int(len(xs))}


def med(values: Iterable[float | None]) -> dict:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {"median": None, "q1": None, "q3": None, "n": 0, "values": []}
    return {"median": float(np.median(xs)), "q1": float(np.percentile(xs, 25)), "q3": float(np.percentile(xs, 75)),
            "n": len(xs), "values": xs}


# ------------------------------------------------------------------ loading
def variant_key(v: dict | None) -> str | None:
    if not v:
        return None
    t = v.get("type")
    if t == "as_of":
        return f"as_of:{v.get('game')}"
    if t == "minus":
        return f"minus:{v.get('aspect')}"
    if t == "transplant":
        return v.get("label") or f"transplant:{v.get('role', 'same_role')}"
    return t


def world_dirs(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("events.jsonl")})


def load_world(d: Path) -> dict | None:
    try:
        evs = read_events(d / "events.jsonl")
    except Exception:
        return None
    if not evs or evs[0]["type"] != "world_start":
        return None
    meta = {}
    if (d / "meta.json").exists():
        try:
            meta = json.loads((d / "meta.json").read_text())
        except Exception:
            meta = {}
    ws = evs[0]["payload"]
    seats = {s["seat"]: s for s in ws["seats"]}
    roles = (seats.get("p2", {}).get("role"), seats.get("p3", {}).get("role"))
    table = meta.get("table") or TABLES.get(roles) or "+".join(str(r) for r in roles)
    variant = ws.get("variant") or meta.get("variant")
    return {"dir": d, "wid": evs[0]["world"], "run": evs[0]["run"], "events": evs, "seed": ws["seed"],
            "agent": seats.get("p1", {}).get("policy"), "seats": seats, "roles": {"p2": roles[0], "p3": roles[1]},
            "table": table, "variant": variant, "vkey": variant_key(variant),
            "parent_world": ws.get("parent_world") or meta.get("parent_world"),
            "from_game": ws.get("from_game") or meta.get("from_game"),
            "done": any(e["type"] == "world_end" for e in evs)}


def load_worlds(root: Path) -> list[dict]:
    out = []
    for d in world_dirs(root):
        w = load_world(d)
        if w:
            w["m"] = world_metrics(w)
            out.append(w)
    return out


def seat_with_role(w: dict, role: str) -> str | None:
    for s, r in w["roles"].items():
        if r == role:
            return s
    return None


# ------------------------------------------------------------------ per-world metrics
def respond_decisions(evs: list[dict], seat: str = "p1") -> list[dict]:
    """p1 respond decisions, each annotated with the claim it answered, the reveal (if any), god-view truth."""
    out, claim, rnd = [], None, 0
    for i, e in enumerate(evs):
        if e["type"] == "deal":
            claim, rnd = None, rnd + 1
        elif e["type"] == "claim":
            claim = e
        elif e["type"] == "decision" and e["actor"] == seat and e["payload"].get("kind") == "respond":
            rev = None
            for f in evs[i + 1:]:
                if f["type"] in ("claim", "deal", "game_end"):
                    break
                if f["type"] == "reveal":
                    rev = f
                    break
            truth = None
            if claim is not None:
                truth = claim["payload"].get("bluff_kind") is not None if "bluff_kind" in claim["payload"] else None
                if truth is None:
                    tr = claim["payload"]["claimed_rank"]
                    truth = any(c != tr and c != "J" for c in claim["payload"]["cards"])
            out.append({"ev": e, "claim": claim, "reveal": rev, "truth_bluff": truth, "ord": rnd})
    return out


def strategic(claim_ev: dict, hands: dict) -> tuple[bool, bool]:
    """(had_choice, strategic_bluff) for a claim. Uses the engine's god-view `bluff_kind` when present."""
    p = claim_ev["payload"]
    tr = p["claimed_rank"]
    bk = p.get("bluff_kind", "missing")
    hand = hands.get(claim_ev["actor"], [])
    had_match = any(c == tr or c == "J" for c in hand)
    for c in p["cards"]:
        if c in hand:
            hand.remove(c)
    if bk != "missing":
        if bk == "strategic":
            return True, True
        if bk == "forced":
            return False, False
        return had_match, False
    bluff = any(c != tr and c != "J" for c in p["cards"])
    return had_match, bluff and had_match


def world_metrics(w: dict) -> dict:
    evs = w["events"]
    ends = [e for e in evs if e["type"] == "game_end"]
    start_g = (w["from_game"] or 0) if w["vkey"] else 0
    played = [e for e in ends if e["game"] > start_g]
    wins = [1.0 if e["payload"]["winner"] == "p1" else 0.0 for e in played]
    late = [1.0 if e["payload"]["winner"] == "p1" else 0.0 for e in played if e["game"] >= LATE_FROM]
    rds = respond_decisions(evs)
    br = [(d["ev"]["payload"]["p_final"] - (1.0 if d["reveal"]["payload"]["was_bluff"] else 0.0)) ** 2
          for d in rds if d["reveal"] is not None and d["ev"]["payload"].get("p_final") is not None]
    m: dict[str, Any] = {
        "games": len(played), "win_rate": float(np.mean(wins)) if wins else None,
        "win_rate_late": float(np.mean(late)) if late else None,
        "brier": float(np.mean(br)) if br else None, "n_brier": len(br),
    }
    # discovery (telltale tables)
    tt = seat_with_role(w, "telltale") or seat_with_role(w, "telltale_overt")
    m["discovery_game"] = m["revealed_to_discovery"] = None
    if tt:
        n_rev = 0
        for e in evs:
            if e["type"] == "reveal" and e["payload"]["claimant"] == tt:
                n_rev += 1
            if (e["type"] == "hypothesis" and e["payload"].get("aspect") == "verbal_tell"
                    and e["payload"].get("status") == "confirmed" and e["payload"].get("provenance") == "evidence"
                    and e["payload"].get("about") in (None, tt)):
                m["discovery_game"] = e["game"]
                m["revealed_to_discovery"] = n_rev
                break
    # long con
    sl = seat_with_role(w, "sleeper")
    if sl:
        m.update(longcon_metrics(w, sl, rds))
    # false alarm: detection fires against a control opponent
    if w["table"] == "falsealarm":
        m["false_alarm"] = float(any(detect_round(w, s, rds, after=0)[0] is not None for s in ("p2", "p3")))
        # stricter variant: only a belief change (belief_superseded) counts, not two high-p_final decisions
        m["false_alarm_belief"] = float(any(e["type"] == "belief_superseded" for e in w["events"]))
    we = next((e["payload"] for e in evs if e["type"] == "world_end"), None)
    m["world_end"] = we
    return m


def detect_round(w: dict, seat: str, rds: list[dict], after: int) -> tuple[int | None, str | None]:
    """R_d (§6): first round ≥ `after` with p_final(seat) > 0.5 on 2 consecutive respond decisions about
    `seat`, or a belief_superseded about `seat`. Returns (round ordinal, how)."""
    cand = []
    prev_hi = False
    for d in rds:
        if d["ev"]["payload"].get("target") != seat or d["ord"] < after:
            continue
        hi = (d["ev"]["payload"].get("p_final") or 0) > 0.5
        if hi and prev_hi:
            cand.append((d["ord"], "p_final"))
            break
        prev_hi = hi
    ords = round_ordinals(w["events"])
    for e in w["events"]:
        if e["type"] == "belief_superseded" and e["payload"].get("about") in (seat, None):
            if e["payload"].get("about") is None and e["actor"] not in (None, "p1"):
                continue
            o = ords.get((e["payload"].get("at_game", e["game"]), e["payload"].get("at_round", e["round"])))
            if o is None:   # game-end supersession: count it at the last round of that game
                o = max((v for (g, r), v in ords.items() if g == e["payload"].get("at_game", e["game"])), default=None)
            if o is not None and o >= after:
                cand.append((o, "superseded"))
                break
    if not cand:
        return None, None
    return min(cand)


def round_ordinals(evs: list[dict]) -> dict[tuple[int, int], int]:
    out, n = {}, 0
    for e in evs:
        if e["type"] == "deal":
            n += 1
            out[(e["game"], e["round"])] = n
    return out


def longcon_metrics(w: dict, sl: str, rds: list[dict]) -> dict:
    evs = w["events"]
    hands: dict[str, list[str]] = {}
    rnd = 0
    per_game = defaultdict(lambda: [0, 0])          # game → [had_choice, strategic]
    r_b = g_first = None
    for e in evs:
        if e["type"] == "deal":
            rnd += 1
            hands = {k: list(v) for k, v in e["payload"]["hands"].items()}
        elif e["type"] == "claim":
            had, strat = strategic(e, hands)
            if e["actor"] == sl:
                if had:
                    per_game[e["game"]][0] += 1
                if strat:
                    per_game[e["game"]][1] += 1
                    if r_b is None:
                        r_b, g_first = rnd, e["game"]
    honesty = {g: (1 - s / c) if c else None for g, (c, s) in per_game.items()}
    # agent's trust estimate: 1 − mean q_used about the sleeper (q_used None ⇒ no memory ⇒ q = 0)
    qs = defaultdict(list)
    for d in rds:
        p = d["ev"]["payload"]
        if p.get("target") == sl:
            qs[d["ev"]["game"]].append(p.get("q_used") or 0.0)
    trust = {g: 1 - float(np.mean(v)) for g, v in qs.items()}
    g0 = None
    params = w["seats"].get(sl, {}).get("params") or {}
    for k in ("G0", "g0", "g_0", "switch_game"):
        if params.get(k) is not None:
            g0 = params[k]
            break
    g0_src = "params" if g0 is not None else ("first strategic bluff" if g_first else None)
    g0 = g0 if g0 is not None else g_first
    out = {"sleeper": sl, "g0": g0, "g0_source": g0_src, "r_b": r_b, "honesty_by_game": honesty,
           "trust_by_game": trust, "r_d": None, "rtd": None, "detect_how": None, "detection_game": None,
           "missed_bluffs": None, "lives_lost_to_sleeper": None}
    if r_b is None:
        return out
    r_d, how = detect_round(w, sl, rds, after=r_b)
    out["r_d"], out["detect_how"] = r_d, how
    out["rtd"] = (r_d - r_b) if r_d is not None else None
    ords = round_ordinals(evs)
    inv = {v: k for k, v in ords.items()}
    out["detection_game"] = inv[r_d][0] if r_d is not None else None
    hi = r_d if r_d is not None else 10 ** 9
    # sleeper bluffs p1 let through (played on) between R_b and R_d; lives p1 lost on sleeper claims in that window
    missed = sum(1 for d in rds if d["ev"]["payload"].get("target") == sl and r_b <= d["ord"] <= hi
                 and d["truth_bluff"] and d["ev"]["payload"].get("action") == "play_on")
    lost = 0
    for e in evs:
        if e["type"] == "reveal" and e["payload"]["claimant"] == sl and e["payload"]["loser"] == "p1":
            o = ords.get((e["game"], e["round"]))
            if o is not None and r_b <= o <= hi:
                lost += 1
    out["missed_bluffs"], out["lives_lost_to_sleeper"] = missed, lost
    return out


# ------------------------------------------------------------------ aggregation
def by_seed(ws: list[dict], key: str) -> dict[int, float]:
    return {w["seed"]: w["m"].get(key) for w in ws if w["m"].get(key) is not None}


def paired_delta(a: dict[int, float], b: dict[int, float]) -> dict:
    seeds = sorted(set(a) & set(b))
    return boot([a[s] - b[s] for s in seeds])


def aggregate(worlds: list[dict]) -> dict:
    base = [w for w in worlds if not w["vkey"]]
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    keyed = defaultdict(list)
    for w in base:
        keyed[(w["table"], w["agent"])].append(w)
    for (t, a), ws in keyed.items():
        g = {k: boot(w["m"].get(k) for w in ws)
             for k in ("win_rate", "win_rate_late", "brier", "discovery_game", "revealed_to_discovery",
                       "rtd", "missed_bluffs", "lives_lost_to_sleeper", "false_alarm", "false_alarm_belief")}
        g["discovered_frac"] = boot(1.0 if w["m"].get("discovery_game") is not None else 0.0 for w in ws) \
            if t.startswith("telltale") else None
        g["rtd_median"] = med(w["m"].get("rtd") for w in ws)
        g["n_worlds"] = len(ws)
        g["seeds"] = sorted(w["seed"] for w in ws)
        groups[t][a] = g
    return groups


def ablation(worlds: list[dict], table: str = "telltale", agent: str = "profiler") -> dict:
    br = [w for w in worlds if w["vkey"] and w["table"] == table and w["agent"] == agent]
    if not br:   # fall back to any table that has branches
        br = [w for w in worlds if w["vkey"] and w["agent"] == agent]
        table = br[0]["table"] if br else table
    by_v = defaultdict(list)
    for w in br:
        by_v[w["vkey"]].append(w)
    full = by_seed(by_v.get("full", []), "win_rate")
    bars = []
    for k in sorted(by_v, key=lambda k: ABL_ORDER.index(k) if k in ABL_ORDER else 99):
        b = boot(w["m"]["win_rate"] for w in by_v[k])
        b.update({"variant": k, "label": ABL_LABEL.get(k, k),
                  "delta_vs_full": paired_delta(by_seed(by_v[k], "win_rate"), full) if full and k != "full" else None})
        bars.append(b)
    at = sorted({w["from_game"] for w in br if w["from_game"] is not None})
    return {"table": table, "agent": agent, "at_game": at[0] if len(at) == 1 else at, "metric": "win_rate",
            "window": f"{(at[0] + 1) if len(at) == 1 else '?'}–20", "bars": bars, "flips": None,
            "n_branch_worlds": len(br)}


def decision_flips(worlds: list[dict], table: str = "telltale", agent: str = "profiler", max_worlds: int = 10) -> dict:
    """§6 decision flips: identical recorded respond states (games 11–20) whose action differs between the
    logged full-memory decision and replay under as_of:0. Real replays; capped at `max_worlds` worlds."""
    try:
        from blab.replay import replay_decision
    except Exception as e:
        return {"n": 0, "flipped": None, "error": str(e)}
    ws = sorted([w for w in worlds if not w["vkey"] and w["table"] == table and w["agent"] == agent and w["done"]],
                key=lambda w: w["seed"])[:max_worlds]
    n = fl = 0
    errs = 0
    for w in ws:
        for d in respond_decisions(w["events"]):
            if d["ev"]["game"] < LATE_FROM:
                continue
            try:
                r = replay_decision(w["dir"], d["ev"]["seq"], "as_of:0")
            except Exception:
                errs += 1
                continue
            n += 1
            fl += int(bool(r.get("flipped")))
    return {"n": n, "flipped": fl, "worlds": len(ws), "errors": errs, "variant": "as_of:0", "games": f"{LATE_FROM}–20"}


def longcon(worlds: list[dict], agents=("counter", "bayes", "profiler")) -> dict:
    lc = [w for w in worlds if not w["vkey"] and w["table"] == "longcon"]
    if not lc:
        return {"games": [], "n_worlds": 0}
    games = sorted({e["game"] for w in lc for e in w["events"] if e["type"] == "game_end"})
    honesty = [dict(game=g, **boot(w["m"]["honesty_by_game"].get(g) for w in lc)) for g in games]
    out_agents, rtd, ec, ll = {}, {}, {}, {}
    for a in agents:
        ws = [w for w in lc if w["agent"] == a]
        if not ws:
            continue
        out_agents[a] = [dict(game=g, **boot(w["m"]["trust_by_game"].get(g) for w in ws)) for g in games]
        rtd[a] = {k: v for k, v in med(w["m"].get("rtd") for w in ws).items() if k != "values"}
        rtd[a]["detected"] = sum(1 for w in ws if w["m"].get("rtd") is not None)
        rtd[a]["worlds"] = len(ws)
        ec[a] = boot(w["m"].get("missed_bluffs") for w in ws)
        ll[a] = boot(w["m"].get("lives_lost_to_sleeper") for w in ws)
    twist = None
    if "bayes" in out_agents and "counter" in out_agents:
        b = by_seed([w for w in lc if w["agent"] == "bayes"], "missed_bluffs")
        c = by_seed([w for w in lc if w["agent"] == "counter"], "missed_bluffs")
        diff = paired_delta(b, c)
        twist = {"diff": diff, "holds": bool(diff["lo"] is not None and diff["lo"] > 0),
                 "metric": "missed_bluffs (bayes − counter, paired by seed)"}
    pw = [w for w in lc if w["agent"] == "profiler"]
    sl_seats = sorted({w["m"].get("sleeper") for w in lc if w["m"].get("sleeper")})
    return {
        "games": games, "n_worlds": len(lc), "true_honesty": honesty, "agents": out_agents,
        "sleeper_seat": sl_seats[0] if len(sl_seats) == 1 else None,
        "g0": med(w["m"].get("g0") for w in lc),
        "detection": {"profiler": med(w["m"].get("detection_game") for w in pw)} if pw else {},
        "rtd": rtd, "exploitation_cost": ec, "exploitation_label": "sleeper bluffs let through before detection",
        "lives_lost_to_sleeper": ll, "twist": twist,
        "estimate_note": "Estimate = 1 − mean q_used about the sleeper per game; counter has no memory (q = 0).",
    }


def show_grid(groups: dict) -> dict:
    cols = [
        {"key": "tt_win", "label": "Win rate, telltale", "table": "telltale", "metric": "win_rate", "fmt": "pct"},
        {"key": "tt_late", "label": "Win rate g11–20", "table": "telltale", "metric": "win_rate_late", "fmt": "pct"},
        {"key": "tt_brier", "label": "Brier (lower is better)", "table": "telltale", "metric": "brier", "fmt": "num"},
        {"key": "tt_disc", "label": "Tell found at game", "table": "telltale", "metric": "discovery_game", "fmt": "num"},
        {"key": "lc_rtd", "label": "Rounds to detection", "table": "longcon", "metric": "rtd", "fmt": "num"},
        {"key": "lc_missed", "label": "Bluffs let through", "table": "longcon", "metric": "missed_bluffs", "fmt": "num"},
        {"key": "fa", "label": "False alarms", "table": "falsealarm", "metric": "false_alarm", "fmt": "pct"},
    ]
    agents = sorted({a for t in groups.values() for a in t},
                    key=lambda a: AGENT_ORDER.index(a) if a in AGENT_ORDER else 99)
    rows = []
    for a in agents:
        cells = {}
        for c in cols:
            g = groups.get(c["table"], {}).get(a)
            if g and g.get(c["metric"]) and g[c["metric"]].get("n"):
                cells[c["key"]] = g[c["metric"]]
        rows.append({"agent": a, "cells": cells})
    return {"columns": cols, "rows": rows}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def summary_md(res: dict) -> str:
    L = [f"# Results — run `{res['run']}`", "", f"Generated {res['generated']} from {res['n_worlds']} worlds "
         f"({res['n_branch_worlds']} ablation branches). Means with 95% bootstrap CIs over seeds.", ""]
    for t, ag in res["groups"].items():
        L += [f"## table `{t}`", "", "| agent | n | win rate | win g11–20 | Brier | tell found (game) | RTD median | bluffs let through | false alarm |",
              "|---|---|---|---|---|---|---|---|---|"]
        for a, g in sorted(ag.items(), key=lambda kv: AGENT_ORDER.index(kv[0]) if kv[0] in AGENT_ORDER else 99):
            def c(m):
                x = g.get(m) or {}
                if x.get("mean") is None:
                    return "—"
                return f"{x['mean']:.3f}" + (f" [{x['lo']:.3f}, {x['hi']:.3f}]" if x.get("lo") is not None else "")
            rm = g["rtd_median"]["median"]
            L.append(f"| {a} | {g['n_worlds']} | {c('win_rate')} | {c('win_rate_late')} | {c('brier')} | "
                     f"{c('discovery_game')} | {'—' if rm is None else rm} | {c('missed_bluffs')} | {c('false_alarm')} |")
        L.append("")
    ab = res["ablation"]
    if ab["bars"]:
        L += [f"## ablation ({ab['table']}, {ab['agent']}, from game {ab['at_game']})", "",
              "| variant | n | win rate | Δ vs full |", "|---|---|---|---|"]
        for b in ab["bars"]:
            d = b.get("delta_vs_full") or {}
            ds = "—" if d.get("mean") is None else f"{d['mean']:+.3f}" + (f" [{d['lo']:+.3f}, {d['hi']:+.3f}]" if d.get("lo") is not None else "")
            L.append(f"| {b['variant']} | {b['n']} | {b['mean']:.3f} | {ds} |")
        L.append("")
    tw = (res["longcon"] or {}).get("twist")
    if tw:
        d = tw["diff"]
        L += ["## long con twist", "", f"{tw['metric']}: mean {d['mean']}, CI [{d['lo']}, {d['hi']}], "
              f"n {d['n']} → holds = {tw['holds']}", ""]
    return "\n".join(L)


def analyze(run: str = "r01", root: str | Path = "runs") -> dict:
    rdir = Path(root) / run
    worlds = load_worlds(rdir)
    groups = aggregate(worlds)
    res = {
        "run": run, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_worlds": sum(1 for w in worlds if not w["vkey"]), "n_branch_worlds": sum(1 for w in worlds if w["vkey"]),
        "groups": groups, "ablation": ablation(worlds), "longcon": longcon(worlds),
        "worlds": [{"wid": w["wid"], "dir": str(w["dir"]), "seed": w["seed"], "agent": w["agent"],
                    "table": w["table"], "variant": w["vkey"], "parent_world": w["parent_world"], "done": w["done"],
                    **{k: v for k, v in w["m"].items() if k not in ("honesty_by_game", "trust_by_game", "world_end")}}
                   for w in worlds],
    }
    res["show_grid"] = show_grid(groups)
    res["ablation"]["flips"] = _jsonable(decision_flips(worlds))
    res["ablation"]["source"] = f"runs/{run}"
    res["longcon"]["source"] = f"runs/{run}"
    res = _jsonable(res)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "results.json").write_text(json.dumps(res, indent=1))
    (rdir / "ablation.json").write_text(json.dumps(res["ablation"], indent=1))
    (rdir / "longcon.json").write_text(json.dumps(res["longcon"], indent=1))
    (rdir / "summary.md").write_text(summary_md(res))
    print(f"analyze: {res['n_worlds']} worlds + {res['n_branch_worlds']} branches → {rdir}/results.json, summary.md")
    return res
