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
AGENT_ORDER = ["llm_nomem", "llm_episodic", "llm_dossier", "counter", "bayes", "bayes_discount", "logreg_bow",
               "profiler_amnesic", "profiler", "profiler_plus", "profiler_blind"]
ABL_ORDER = ["as_of:0", "as_of:4", "as_of:6", "as_of:8", "full", "minus:verbal_tell",
             "false_memory", "transplant:same_role", "transplant:other_role"]
ABL_LABEL = {"as_of:0": "erased", "as_of:4": "as of g4", "as_of:6": "as of g6", "as_of:8": "as of g8",
             "full": "full", "minus:verbal_tell": "minus tell", "false_memory": "false memory",
             "transplant:same_role": "transplant, same role", "transplant:other_role": "transplant, other role"}
LATE_FROM = 11
REFLECTING = {"profiler", "profiler_blind"}
BASELINES = ("counter", "bayes", "bayes_discount", "logreg_bow")
METRICS = ("win_rate", "win_rate_late", "brier", "discovery_game", "revealed_to_discovery", "false_discovery", "rtd",
           "missed_bluffs", "lives_lost_to_sleeper", "false_alarm", "false_alarm_belief")
METRIC_DEFS = {
    "win_rate": "§6: games won / games played",
    "win_rate_late": "§6: win rate over games 11–20",
    "brier": "§6: mean (p_final − was_bluff)² over p1 respond decisions whose claim was revealed",
    "discovery_game": "§6: first game a verbal_tell hypothesis (evidence) about the telltale is confirmed",
    "revealed_to_discovery": "§6: revealed telltale claims observed before discovery",
    "rtd": "§6: R_d − R_b in rounds (R_b = sleeper's first strategic bluff; R_d = 2 consecutive p_final>0.5 "
           "about the sleeper, or a belief_superseded about it); undetected worlds excluded",
    "false_discovery": "null-tell false discovery: the same discovery test fires in a falsealarm world (no opponent "
                       "has a tell); profiler family = confirmed verbal_tell, logreg_bow = tell_coef interval > 0",
    "lives_lost_to_sleeper": "§6 exploitation cost: lives p1 lost on the sleeper's claims between R_b and R_d",
    "missed_bluffs": "corrected exploitation cost: sleeper bluffs p1 played on between R_b and R_d "
                     "(through game 20 if never detected)",
    "false_alarm": "§6: the RTD detection rule fires against a control opponent",
    "false_alarm_belief": "corrected: a belief_superseded fires in a falsealarm world (belief change only)",
}
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
    agent = seats.get("p1", {}).get("policy")
    prm = seats.get("p1", {}).get("params") or {}
    rule = prm.get("respond_rule") or meta.get("respond_rule") or "threshold"
    refl = prm.get("reflection") or meta.get("reflection") or \
        ((("claude" if meta.get("llm") else "stub")) if agent in REFLECTING else "none")
    label = agent if refl in ("stub", "none") else f"{agent}+{refl}"
    return {"rule": rule, "reflection": refl, "label": label, "dir": d, "wid": evs[0]["world"], "run": evs[0]["run"], "events": evs, "seed": ws["seed"],
            "agent": seats.get("p1", {}).get("policy"), "seats": seats, "roles": {"p2": roles[0], "p3": roles[1]},
            "table": table, "variant": variant, "vkey": variant_key(variant),
            "parent_world": ws.get("parent_world") or meta.get("parent_world"),
            "from_game": ws.get("from_game") or meta.get("from_game"),
            "done": any(e["type"] == "world_end" for e in evs)}


def _load_one(d: Path) -> dict | None:
    w = load_world(d)
    if not w:
        return None
    w["m"] = world_metrics(w)
    w["m"]["game_list"] = sorted({e["game"] for e in w["events"] if e["type"] == "game_end"})
    del w["events"]                      # keep memory flat for thousands of worlds; ensure_events() reloads
    return w


def ensure_events(w: dict) -> dict:
    if "events" not in w:
        w["events"] = read_events(Path(w["dir"]) / "events.jsonl")
    return w


def load_worlds(root: Path, workers: int | None = None) -> list[dict]:
    dirs = world_dirs(root)
    if len(dirs) < 50:
        out = [_load_one(d) for d in dirs]
    else:
        import os
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 4)) as ex:
            out = list(ex.map(_load_one, dirs, chunksize=16))
    return [w for w in out if w]


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


BUCKETS = [(1, 5), (6, 10), (11, 15), (16, 20)]


def bucket(g: int) -> int:
    return min(len(BUCKETS) - 1, max(0, (int(g) - 1) // 5))


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
    # learning curves: per game bucket (1–5, 6–10, 11–15, 16–20)
    wb, bb = defaultdict(list), defaultdict(list)
    for e in played:
        wb[bucket(e["game"])].append(1.0 if e["payload"]["winner"] == "p1" else 0.0)
    for d in rds:
        if d["reveal"] is not None and d["ev"]["payload"].get("p_final") is not None:
            bb[bucket(d["ev"]["game"])].append(
                (d["ev"]["payload"]["p_final"] - (1.0 if d["reveal"]["payload"]["was_bluff"] else 0.0)) ** 2)
    m["win_by_bucket"] = {b: float(np.mean(v)) for b, v in wb.items()}
    m["brier_by_bucket"] = {b: float(np.mean(v)) for b, v in bb.items()}
    # discovery (telltale tables) and null-tell false discovery (falsealarm table: no opponent has a tell)
    tt = seat_with_role(w, "telltale") or seat_with_role(w, "telltale_overt")
    m["discovery_game"] = m["revealed_to_discovery"] = m["false_discovery"] = None
    g_disc, n_rev = tell_discovery(w, tt)
    if tt:
        m["discovery_game"], m["revealed_to_discovery"] = g_disc, (n_rev if g_disc is not None else None)
    if w["table"] == "falsealarm" and can_discover(w):
        m["false_discovery"] = 1.0 if g_disc is not None else 0.0
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


def can_discover(w: dict) -> bool:
    return w["agent"] in REFLECTING or str(w["agent"]).startswith("logreg")


def tell_discovery(w: dict, tt: str | None) -> tuple[int | None, int]:
    """First game a verbal tell is 'discovered' (§6) and revealed claims by `tt` observed before it.
    Profiler family: a verbal_tell hypothesis (provenance evidence) becomes confirmed — about `tt`, or about
    anyone when there is no telltale (null table). logreg_bow: its recorded tell coefficient's interval
    `logreg.tell_coef = [coef, lo, hi]` first excludes 0 on the positive side."""
    n_rev = 0
    lr = str(w["agent"]).startswith("logreg")
    mf = Path(w["dir"]) / "memory" / "p1.json"
    if lr and mf.exists():               # the agent's own record (game-end), covers every opponent
        try:
            L = (json.loads(mf.read_text()).get("logreg") or {})
            gs = [v.get("discovery_game") for k, v in L.items() if (tt is None or k == tt) and v.get("discovery_game") is not None]
            g = min(gs) if gs else None
            if g is None:
                return None, 0
            n = sum(1 for e in w["events"] if e["type"] == "reveal" and tt and e["payload"]["claimant"] == tt and e["game"] <= g)
            return g, n
        except Exception:
            pass
    for e in w["events"]:
        if e["type"] == "reveal" and tt and e["payload"]["claimant"] == tt:
            n_rev += 1
        if lr:
            if e["type"] == "decision" and e["actor"] == "p1":
                tc = (e["payload"].get("logreg") or {}).get("tell_coef") or e["payload"].get("tell_coef")
                if tc and len(tc) == 3 and tc[1] is not None and tc[1] > 0:
                    return e["game"], n_rev
        elif (e["type"] == "hypothesis" and e["payload"].get("aspect") == "verbal_tell"
              and e["payload"].get("status") == "confirmed" and e["payload"].get("provenance") == "evidence"
              and (tt is None or e["payload"].get("about") in (None, tt))):
            return e["game"], n_rev
    return None, n_rev


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
def by_seed(ws: list[dict], key: str) -> dict:
    """Per-world metric keyed by (table, seed) so pooled comparisons stay paired."""
    return {(w["table"], w["seed"]): w["m"].get(key) for w in ws if w["m"].get(key) is not None}


def paired_delta(a: dict, b: dict) -> dict:
    keys = sorted(set(a) & set(b), key=str)
    return boot([a[k] - b[k] for k in keys])


def label_order(a: str) -> tuple:
    base = a.split("+")[0]
    return (AGENT_ORDER.index(base) if base in AGENT_ORDER else 99, a)


def base_worlds(worlds: list[dict]) -> list[dict]:
    return [w for w in worlds if not w["vkey"] and w["done"]]


def aggregate(worlds: list[dict]) -> dict:
    """groups[respond_rule][table][agent label] = metrics with bootstrap CIs over seeds."""
    groups: dict = defaultdict(lambda: defaultdict(dict))
    keyed = defaultdict(list)
    for w in base_worlds(worlds):
        keyed[(w["rule"], w["table"], w["label"])].append(w)
    for (r, t, a), ws in keyed.items():
        g = {k: boot(w["m"].get(k) for w in ws) for k in METRICS}
        g["win_by_bucket"] = [boot(w["m"].get("win_by_bucket", {}).get(b) for w in ws) for b in range(len(BUCKETS))]
        g["brier_by_bucket"] = [boot(w["m"].get("brier_by_bucket", {}).get(b) for w in ws) for b in range(len(BUCKETS))]
        g["discovered_frac"] = boot(1.0 if w["m"].get("discovery_game") is not None else 0.0 for w in ws) \
            if t.startswith("telltale") else None
        g["rtd_median"] = med(w["m"].get("rtd") for w in ws)
        g.update({"n_worlds": len(ws), "seeds": sorted(w["seed"] for w in ws), "agent": ws[0]["agent"],
                  "reflection": ws[0]["reflection"], "respond_rule": r})
        groups[r][t][a] = g
    return groups


DELTA_METRICS = ("win_rate", "win_rate_late", "brier", "missed_bluffs", "lives_lost_to_sleeper", "rtd", "false_discovery")


def deltas(worlds: list[dict]) -> dict:
    """Paired-by-seed Δ (profiler-family − baseline) per respond rule, per table and pooled ("all")."""
    out: dict = defaultdict(dict)
    bw = base_worlds(worlds)
    for r in sorted({w["rule"] for w in bw}):
        ws_r = [w for w in bw if w["rule"] == r]
        ours = sorted({w["label"] for w in ws_r if w["agent"] in ("profiler",)}, key=label_order)
        bases = [b for b in BASELINES if any(w["label"] == b for w in ws_r)]
        for t in sorted({w["table"] for w in ws_r}) + ["all"]:
            tw = [w for w in ws_r if t == "all" or w["table"] == t]
            d = {}
            for a in ours:
                for b in bases:
                    A_ = [w for w in tw if w["label"] == a]
                    B_ = [w for w in tw if w["label"] == b]
                    if not A_ or not B_:
                        continue
                    d[f"{a} − {b}"] = {m: paired_delta(by_seed(A_, m), by_seed(B_, m)) for m in DELTA_METRICS}
            if d:
                out[r][t] = d
    return out


def ablation(worlds: list[dict], table: str = "telltale", agent: str = "profiler", rule: str | None = None) -> dict:
    br = [w for w in worlds if w["vkey"] and w["agent"] == agent and (rule is None or w["rule"] == rule)]
    if any(w["table"] == table for w in br):
        br = [w for w in br if w["table"] == table]
    elif br:
        table = br[0]["table"]
        br = [w for w in br if w["table"] == table]
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
    return {"table": table, "agent": agent, "respond_rule": rule, "at_game": at[0] if len(at) == 1 else at,
            "metric": "win_rate", "window": f"{(at[0] + 1) if len(at) == 1 else '?'}–20", "bars": bars,
            "flips": None, "n_branch_worlds": len(br)}


def decision_flips(worlds: list[dict], table: str = "telltale", agent: str = "profiler", rule: str | None = None,
                   max_worlds: int = 10) -> dict:
    """§6 decision flips: recorded respond states (games 11–20) whose action differs between the logged
    full-memory decision and replay under as_of:0. Real replays; capped at `max_worlds` worlds."""
    try:
        from blab.replay import replay_decision
    except Exception as e:
        return {"n": 0, "flipped": None, "error": str(e)}
    ws = sorted([w for w in base_worlds(worlds) if w["table"] == table and w["label"] == agent
                 and (rule is None or w["rule"] == rule)], key=lambda w: w["seed"])[:max_worlds]
    n = fl = errs = 0
    for w in ws:
        for d in respond_decisions(ensure_events(w)["events"]):
            if d["ev"]["game"] < LATE_FROM:
                continue
            try:
                r = replay_decision(w["dir"], d["ev"]["seq"], "as_of:0")
            except Exception:
                errs += 1
                continue
            n += 1
            fl += int(bool(r.get("flipped")))
        w.pop("events", None)
    return {"n": n, "flipped": fl, "worlds": len(ws), "errors": errs, "variant": "as_of:0",
            "games": f"{LATE_FROM}–20", "respond_rule": rule}


def longcon(worlds: list[dict], agents=("counter", "bayes", "profiler"), rule: str | None = None) -> dict:
    lc = [w for w in base_worlds(worlds) if w["table"] == "longcon" and (rule is None or w["rule"] == rule)]
    if not lc:
        return {"games": [], "n_worlds": 0, "respond_rule": rule}
    games = sorted({g for w in lc for g in w["m"].get("game_list", [])})
    honesty = [dict(game=g, **boot(w["m"]["honesty_by_game"].get(g) for w in lc)) for g in games]
    out_agents, rtd, ec, ll = {}, {}, {}, {}
    for a in agents:
        ws = [w for w in lc if w["label"] == a]
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
        diff = paired_delta(by_seed([w for w in lc if w["label"] == "bayes"], "missed_bluffs"),
                            by_seed([w for w in lc if w["label"] == "counter"], "missed_bluffs"))
        diff_spec = paired_delta(by_seed([w for w in lc if w["label"] == "bayes"], "lives_lost_to_sleeper"),
                                 by_seed([w for w in lc if w["label"] == "counter"], "lives_lost_to_sleeper"))
        twist = {"diff": diff, "holds": bool(diff["lo"] is not None and diff["lo"] > 0),
                 "metric": "sleeper bluffs let through, bayes − counter (paired by seed)",
                 "diff_spec": diff_spec, "holds_spec": bool(diff_spec["lo"] is not None and diff_spec["lo"] > 0),
                 "metric_spec": "§6 lives lost to the sleeper, bayes − counter (paired by seed)"}
    pw = [w for w in lc if w["label"] == "profiler"]
    sl_seats = sorted({w["m"].get("sleeper") for w in lc if w["m"].get("sleeper")})
    return {
        "games": games, "n_worlds": len(lc), "respond_rule": rule, "true_honesty": honesty, "agents": out_agents,
        "sleeper_seat": sl_seats[0] if len(sl_seats) == 1 else None,
        "g0": med(w["m"].get("g0") for w in lc),
        "detection": {"profiler": med(w["m"].get("detection_game") for w in pw)} if pw else {},
        "rtd": rtd, "exploitation_cost": ec, "exploitation_label": "sleeper bluffs let through before detection",
        "lives_lost_to_sleeper": ll, "twist": twist,
        "estimate_note": "Estimate = 1 − mean q_used about the sleeper per game; counter has no memory (q = 0).",
    }


# ------------------------------------------------------------------ "does memory help?" (memory.json)
def setting_of(worlds: list[dict], run: str) -> str:
    """'raw' (no engineered tell features in observations) vs 'engineered' (precomputed formal_register etc.)."""
    for w in worlds:
        prm = (w["seats"].get("p1", {}).get("params") or {})
        for k in ("obs_mode", "features", "setting", "observation"):
            if prm.get(k):
                return str(prm[k])
    return "raw" if run.startswith("mem") else "engineered"


def memory_compare(worlds: list[dict], run: str, baseline: str | None = None) -> dict:
    """Arms × tables × respond rules: per-arm metrics, paired-by-seed Δ vs the no-memory arm, learning curves."""
    bw = base_worlds(worlds)
    labels = {w["label"] for w in bw}
    base = baseline or next((b for b in ("llm_nomem", "counter") if b in labels), None)
    out = {"run": run, "setting": setting_of(bw, run), "baseline": base, "rules": {},
           "buckets": [f"g{a}–{b}" for a, b in BUCKETS],
           "headline": ("Same model, same prompt, same observation, different memory." if base == "llm_nomem"
                        else "Memory vs no memory (the no-memory arm is the card counter, a different agent)."),
           "note": ("same model, prompt and observation; only memory differs" if base == "llm_nomem"
                    else "baseline is the no-memory card counter (no LLM no-memory arm in this run)")}
    for r in sorted({w["rule"] for w in bw}):
        R = {}
        for t in sorted({w["table"] for w in bw if w["rule"] == r}) + ["all"]:
            tw = [w for w in bw if w["rule"] == r and (t == "all" or w["table"] == t)]
            arms = sorted({w["label"] for w in tw}, key=label_order)
            T = {"arms": arms, "metrics": {}, "delta_vs_baseline": {}, "curves": {}}
            for a in arms:
                aw = [w for w in tw if w["label"] == a]
                T["metrics"][a] = {m: boot(w["m"].get(m) for w in aw) for m in ("win_rate", "win_rate_late", "brier")}
                T["metrics"][a]["n_worlds"] = len(aw)
                T["curves"][a] = {
                    "win_rate": [boot(w["m"].get("win_by_bucket", {}).get(b) for w in aw) for b in range(len(BUCKETS))],
                    "brier": [boot(w["m"].get("brier_by_bucket", {}).get(b) for w in aw) for b in range(len(BUCKETS))]}
                if base and a != base:
                    bwl = [w for w in tw if w["label"] == base]
                    T["delta_vs_baseline"][a] = {m: paired_delta(by_seed(aw, m), by_seed(bwl, m))
                                                 for m in ("win_rate", "win_rate_late", "brier")}
            R[t] = T
        out["rules"][r] = R
    # claims, generated from the data
    cl = {}
    for r, R in out["rules"].items():
        for a, d in (R.get("all", {}).get("delta_vs_baseline") or {}).items():
            for m, better in (("win_rate", "higher"), ("brier", "lower")):
                x = d.get(m) or {}
                if x.get("lo") is None:
                    continue
                sg = _sig(x)
                pct_ = m.startswith("win")
                verdict = ("better than" if sg == better else "worse than" if sg in ("lower", "higher") else
                           "not detectably different from")
                cl[f"{a}|{m}|{r}"] = {"holds": sg == better, "sign": sg, "delta": x,
                                      "text": f"{a} is {verdict} {base} on {m.replace('_', ' ')}: {_ci(x, pct_)} "
                                              f"({r} rule, {out['setting']} observations)."}
    out["claims"] = cl
    return out


# ------------------------------------------------------------------ claims (text generated from data only)
def _sig(d: dict | None) -> str | None:
    """'lower' / 'higher' if the 95% CI excludes 0, else 'none'; None if not computable."""
    if not d or d.get("lo") is None:
        return None
    if d["hi"] < 0:
        return "lower"
    if d["lo"] > 0:
        return "higher"
    return "none"


def _ci(d: dict, pct: bool = False) -> str:
    f = (lambda x: f"{x * 100:+.1f} pts") if pct else (lambda x: f"{x:+.3f}")
    return f"{f(d['mean'])} (95% CI {f(d['lo'])} to {f(d['hi'])}, n = {d['n']} paired seeds)"


def claims(res: dict) -> dict:
    """Every sentence the Show or Devpost may say about outcomes, each with `holds` computed from the data."""
    out: dict = {}
    D = res.get("deltas") or {}
    for rule, tabs in D.items():
        dd = (tabs.get("telltale") or tabs.get("all") or {}).get("profiler − counter") or {}
        da = (tabs.get("all") or {}).get("profiler − counter") or {}
        b, wr = dd.get("brier"), da.get("win_rate")
        if b and b.get("lo") is not None:
            out[f"calibration_{rule}"] = {
                "holds": _sig(b) == "lower", "delta": b,
                "text": f"Memory improves calibration: Brier {_ci(b)} vs the no-memory counter ({rule} rule)."
                if _sig(b) == "lower" else f"No calibration gain detected: Brier Δ {_ci(b)} ({rule} rule)."}
        if wr and wr.get("lo") is not None:
            s = _sig(wr)
            out[f"winrate_{rule}"] = {
                "holds": s == "higher", "sign": s, "delta": wr,
                "text": {"higher": f"The Profiler wins more than the counter: {_ci(wr, True)} ({rule} rule).",
                         "lower": f"The Profiler wins less than the counter: {_ci(wr, True)} ({rule} rule).",
                         "none": f"No detectable win-rate difference vs the counter: {_ci(wr, True)} ({rule} rule)."}[s]}
    for rule, tabs in D.items():
        dl = (tabs.get("telltale") or {}).get("profiler − logreg_bow") or {}
        b, wr = dl.get("brier"), dl.get("win_rate")
        if b and b.get("lo") is not None and wr and wr.get("lo") is not None:
            sb, sw = _sig(b), _sig(wr)
            stronger = sb == "higher" or sw == "lower"
            out[f"vs_logreg_{rule}"] = {
                "holds": stronger, "brier": b, "win_rate": wr,
                "text": (("A bag-of-words logistic regression is the stronger predictor on the telltale table: "
                          if stronger else "Profiler vs bag-of-words logistic regression, telltale table: ")
                         + f"Brier Δ (profiler − logreg) {_ci(b)}; win-rate Δ {_ci(wr, True)} ({rule} rule).")}
    for rule, tabs in (res.get("groups") or {}).items():
        fa = tabs.get("falsealarm") or {}
        rates = {a: g["false_discovery"] for a, g in fa.items() if (g.get("false_discovery") or {}).get("n")}
        if rates:
            out[f"false_discovery_{rule}"] = {
                "holds": True, "rates": rates,
                "text": "On tables where no opponent has a tell, the discovery test still fired in: " + "; ".join(
                    f"{a} {r['mean'] * 100:.0f}% of {r['n']} worlds" for a, r in sorted(rates.items(), key=lambda kv: label_order(kv[0])))
                        + f" ({rule} rule)."}
    for rule, tabs in D.items():
        fd = ((tabs.get("falsealarm") or {}).get("profiler − logreg_bow") or {}).get("false_discovery")
        if fd and fd.get("lo") is not None:
            s_ = _sig(fd)
            out[f"false_discovery_lower_{rule}"] = {
                "holds": s_ == "lower", "delta": fd,
                "text": {"lower": "The Profiler 'finds' fewer non-existent tells than the bag-of-words baseline: ",
                         "higher": "The Profiler 'finds' more non-existent tells than the bag-of-words baseline: ",
                         "none": "No detectable difference in false tell discoveries vs the bag-of-words baseline: "}[s_]
                        + f"{_ci(fd)} ({rule} rule)."}
    cal = [k for k, v in out.items() if k.startswith("calibration_") and v["holds"]]
    wins = [k for k, v in out.items() if k.startswith("winrate_") and v["holds"]]
    if cal and not wins:
        out["beliefs_not_decisions"] = {
            "holds": True, "text": "Better beliefs ≠ better decisions: the decision rule must also be learned.",
            "support": cal + [k for k in out if k.startswith("winrate_")]}
    else:
        out["beliefs_not_decisions"] = {"holds": False, "text": None}
    ab = res.get("ablation") or {}
    e = next((x for x in ab.get("bars", []) if x["variant"] == "as_of:0"), None)
    if e and e.get("delta_vs_full") and e["delta_vs_full"].get("lo") is not None:
        d = e["delta_vs_full"]
        out["erase_hurts"] = {"holds": _sig(d) == "lower", "delta": d,
                              "text": f"Erasing memory at game {ab.get('at_game')} changes win rate in games "
                                      f"{ab.get('window')} by {_ci(d, True)}."}
    fl = ab.get("flips") or {}
    if fl.get("n"):
        out["decision_flips"] = {"holds": fl["flipped"] > 0, "text": f"Erasing memory flips {fl['flipped']} of "
                                 f"{fl['n']} recorded decisions (games {fl.get('games')}, same states, real replays)."}
    tw = (res.get("longcon") or {}).get("twist")
    if tw and tw["diff"].get("lo") is not None:
        out["longcon_twist"] = {"holds": tw["holds"], "delta": tw["diff"],
                                "text": ("Naive memory is more exploitable than no memory: " if tw["holds"] else
                                         "Naive memory was less exploitable than no memory: " if _sig(tw["diff"]) == "lower"
                                         else "Naive memory was not measurably more exploitable than no memory: ")
                                + f"{tw['metric']} {_ci(tw['diff'])}."}
    return out


def decisions_view(res: dict) -> dict:
    """Rows for the Show's 'better beliefs ≠ better decisions' beat: paired Δ profiler − counter per rule."""
    rows = []
    for rule, tabs in (res.get("deltas") or {}).items():
        dd = (tabs.get("telltale") or {}).get("profiler − counter") or {}
        da = (tabs.get("all") or {}).get("profiler − counter") or {}
        if dd.get("brier", {}).get("n"):
            rows.append({"rule": rule, "metric": "brier", "label": "Brier score, telltale table",
                         "better": "lower", "fmt": "num", "delta": dd["brier"]})
        if da.get("win_rate", {}).get("n"):
            rows.append({"rule": rule, "metric": "win_rate", "label": "Win rate, all tables",
                         "better": "higher", "fmt": "pct", "delta": da["win_rate"]})
    c = res.get("claims") or {}
    return {"rows": rows, "headline": c.get("beliefs_not_decisions"),
            "texts": [v["text"] for k, v in c.items() if k.startswith(("calibration_", "winrate_")) and v.get("text")]}


def show_grid(groups: dict, rule: str) -> dict:
    cols = [
        {"key": "tt_win", "label": "Win rate, telltale", "table": "telltale", "metric": "win_rate", "fmt": "pct"},
        {"key": "tt_late", "label": "Win rate g11–20", "table": "telltale", "metric": "win_rate_late", "fmt": "pct"},
        {"key": "tt_brier", "label": "Brier (lower is better)", "table": "telltale", "metric": "brier", "fmt": "num"},
        {"key": "tt_disc", "label": "Tell found at game", "table": "telltale", "metric": "discovery_game", "fmt": "num"},
        {"key": "fd", "label": "False tell found (null table)", "table": "falsealarm", "metric": "false_discovery", "fmt": "pct"},
        {"key": "lc_rtd", "label": "Rounds to detection", "table": "longcon", "metric": "rtd", "fmt": "num"},
        {"key": "lc_missed", "label": "Sleeper bluffs let through", "table": "longcon", "metric": "missed_bluffs", "fmt": "num"},
        {"key": "fa", "label": "False alarms (belief change)", "table": "falsealarm", "metric": "false_alarm_belief", "fmt": "pct"},
    ]
    G = groups.get(rule, {})
    agents = sorted({a for t in G.values() for a in t}, key=label_order)
    rows = []
    for a in agents:
        cells = {}
        for c in cols:
            g = G.get(c["table"], {}).get(a)
            if g and g.get(c["metric"]) and g[c["metric"]].get("n"):
                cells[c["key"]] = g[c["metric"]]
        rows.append({"agent": a, "cells": cells})
    return {"columns": cols, "rows": rows, "respond_rule": rule}


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


def _cell(x: dict | None) -> str:
    x = x or {}
    if x.get("mean") is None:
        return "—"
    return f"{x['mean']:.3f}" + (f" [{x['lo']:.3f}, {x['hi']:.3f}]" if x.get("lo") is not None else "")


def summary_md(res: dict) -> str:
    L = [f"# Results — run `{res['run']}`", "", f"Generated {res['generated']} from {res['n_worlds']} worlds "
         f"({res['n_branch_worlds']} ablation branches). Means with 95% bootstrap CIs over seeds. "
         f"Primary respond rule: `{res['primary_rule']}`.", "", "## Claims (generated from the data)", ""]
    for k, c in res["claims"].items():
        if c.get("text"):
            L.append(f"- **{k}** (holds: {c['holds']}): {c['text']}")
    L += ["", "## Metric definitions", ""] + [f"- `{k}`: {v}" for k, v in METRIC_DEFS.items()] + [""]
    for r, tabs in res["groups"].items():
        for t, ag in sorted(tabs.items()):
            L += [f"## rule `{r}`, table `{t}`", "",
                  "| agent | n | win rate | win g11–20 | Brier | tell found | RTD median | bluffs let through | §6 lives lost to sleeper | §6 false alarm | false alarm (belief) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for a, g in sorted(ag.items(), key=lambda kv: label_order(kv[0])):
                rm = g["rtd_median"]["median"]
                L.append(f"| {a} | {g['n_worlds']} | {_cell(g['win_rate'])} | {_cell(g['win_rate_late'])} | "
                         f"{_cell(g['brier'])} | {_cell(g['discovery_game'])} | {'—' if rm is None else rm} | "
                         f"{_cell(g['missed_bluffs'])} | {_cell(g['lives_lost_to_sleeper'])} | "
                         f"{_cell(g['false_alarm'])} | {_cell(g['false_alarm_belief'])} |")
            L.append("")
    for r, tabs in res["deltas"].items():
        L += [f"## paired Δ, rule `{r}`", "", "| table | comparison | win rate | win g11–20 | Brier | bluffs let through |",
              "|---|---|---|---|---|---|"]
        for t, comps in tabs.items():
            for k, d in comps.items():
                L.append(f"| {t} | {k} | {_cell(d['win_rate'])} | {_cell(d['win_rate_late'])} | {_cell(d['brier'])} | "
                         f"{_cell(d['missed_bluffs'])} |")
        L.append("")
    for r, ab in res["ablation_by_rule"].items():
        if ab["bars"]:
            L += [f"## ablation (rule `{r}`, {ab['table']}, {ab['agent']}, from game {ab['at_game']})", "",
                  "| variant | n | win rate | Δ vs full |", "|---|---|---|---|"]
            for b in ab["bars"]:
                L.append(f"| {b['variant']} | {b['n']} | {b['mean']:.3f} | {_cell(b.get('delta_vs_full'))} |")
            L.append("")
    return "\n".join(L)


def analyze(run: str = "r01", root: str | Path = "runs", worlds: list[dict] | None = None) -> dict:
    rdir = Path(root) / run
    worlds = load_worlds(rdir) if worlds is None else worlds
    groups = aggregate(worlds)
    rules = sorted({w["rule"] for w in worlds}) or ["threshold"]
    primary = "threshold" if "threshold" in rules else rules[0]
    res = {
        "run": run, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_worlds": len(base_worlds(worlds)), "n_branch_worlds": sum(1 for w in worlds if w["vkey"]),
        "rules": rules, "primary_rule": primary, "metric_defs": METRIC_DEFS, "setting": setting_of(worlds, run),
        "groups": groups, "deltas": deltas(worlds),
        "ablation_by_rule": {r: ablation(worlds, rule=r) for r in rules},
        "longcon_by_rule": {r: longcon(worlds, rule=r) for r in rules},
        "worlds": [{"wid": w["wid"], "dir": str(w["dir"]), "seed": w["seed"], "agent": w["agent"], "label": w["label"],
                    "respond_rule": w["rule"], "reflection": w["reflection"],
                    "table": w["table"], "variant": w["vkey"], "parent_world": w["parent_world"], "done": w["done"],
                    **{k: v for k, v in w["m"].items() if k not in ("honesty_by_game", "trust_by_game", "world_end")}}
                   for w in worlds],
    }
    res["ablation_by_rule"][primary]["flips"] = decision_flips(worlds, rule=primary)
    res["ablation"] = res["ablation_by_rule"][primary]
    res["longcon"] = res["longcon_by_rule"][primary]
    res["show_grid"] = show_grid(groups, primary)
    res["show_grids"] = {r: show_grid(groups, r) for r in rules}
    res["ablation"]["source"] = res["longcon"]["source"] = f"runs/{run}"
    res = _jsonable(res)
    res["claims"] = _jsonable(claims(res))
    res["decisions"] = _jsonable(decisions_view(res))
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "results.json").write_text(json.dumps(res, indent=1))
    (rdir / "ablation.json").write_text(json.dumps(res["ablation"], indent=1))
    (rdir / "longcon.json").write_text(json.dumps(res["longcon"], indent=1))
    (rdir / "summary.md").write_text(summary_md(res))
    print(f"analyze: {res['n_worlds']} worlds + {res['n_branch_worlds']} branches, rules {rules} "
          f"→ {rdir}/results.json, summary.md")
    return res
