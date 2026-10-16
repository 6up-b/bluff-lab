"""Arize AX v8 experiment from REAL r01 data (SPEC §10.3).

Rows = revealed p1 respond decisions on the telltale table, threshold rule, base (non-variant) worlds:
the real-Claude-reflection profiler worlds plus the same seeds' counter / bayes / logreg_bow worlds.
`experiments.run` has no row filter, so we create ONE DATASET PER AGENT (frozen-decisions-r01-<agent>)
and run one experiment per agent on it. The task replays the recorded p_final (no recomputation, no LLM
calls); the evaluator is Brier = (p - was_bluff)^2. Writes runs/r01/arize_experiment.json.

    python -m blab.orchestrator.arize_exp [--cap 2000]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from blab.env import load_env
from blab.orchestrator.analyze import ensure_events, load_worlds, respond_decisions

RUN = Path("runs/r01")
OTHERS = ("counter", "bayes", "logreg_bow")


def _base(w: dict) -> bool:
    return (w["table"] == "telltale" and w["rule"] == "threshold" and not w.get("variant")
            and not w.get("parent_world") and w.get("done"))


def build_rows(cap: int = 2000) -> pd.DataFrame:
    ws = [w for w in load_worlds(RUN) if _base(w)]
    prof = sorted((w for w in ws if w["agent"] == "profiler" and w["reflection"] == "claude"), key=lambda w: int(w["seed"]))
    seeds = [w["seed"] for w in prof]
    by = {(w["agent"], w["seed"]): w for w in ws if w["agent"] in OTHERS}
    rows: list[dict] = []
    for s in seeds:
        group = [w for w in [next(p for p in prof if p["seed"] == s)] + [by.get((a, s)) for a in OTHERS] if w]
        chunk = []
        for w in group:
            ensure_events(w)
            label = "profiler" if w["agent"] == "profiler" else w["agent"]
            for d in respond_decisions(w["events"]):
                rev = d["reveal"]
                if not rev or "was_bluff" not in rev["payload"]:
                    continue
                p = d["ev"]["payload"]
                if p.get("p_final") is None:
                    continue
                chunk.append({
                    "state_id": f"{w['wid']}:{d['ev']['seq']}",
                    "memory_snapshot_id": f"{w['wid']}:mv{p.get('memory_version', 0)}",
                    "agent": label, "reflection": w["reflection"], "respond_rule": w["rule"],
                    "seed": str(s), "p_final": float(p["p_final"]),
                    "was_bluff": int(bool(rev["payload"]["was_bluff"])),
                    "world_dir": str(w["dir"]), "seq": int(d["ev"]["seq"]),
                    "tell_seat": next((st for st, r in w["roles"].items() if r == "telltale"), "p3"),
                })
            w.pop("events", None)
        if rows and len(rows) + len(chunk) > cap:
            break
        rows += chunk
    df = pd.DataFrame(rows)
    df["input"] = df["state_id"] + " | " + df["memory_snapshot_id"]
    return df


def task(dataset_row) -> float:
    return float(dataset_row["p_final"])


def brier(output, dataset_row):
    from arize.experiments import EvaluationResult
    p, y = float(output), float(dataset_row["was_bluff"])
    return EvaluationResult(score=(p - y) ** 2, label="brier", explanation=f"p={p:.4f} was_bluff={int(y)}")


SURGERY = {"surgery-full": "full", "surgery-erased": "as_of:0",
           "surgery-false-memory": "false_memory", "surgery-provenance-blind": "false_memory@profiler_blind"}


def _variant(name: str, tell_seat: str):
    """String variant → dict; false_memory targets the telltale seat, same logic as pick.false_memory_variant."""
    from blab.orchestrator.pick import false_memory_variant
    from blab.replay import parse_variant
    if not name.startswith("false_memory"):
        return name
    v = false_memory_variant(tell_seat)
    if "@" in name:
        v = {**v, "agent": parse_variant(name).get("agent")}
    return v


def _num(o, k):
    if isinstance(o, str):
        try:
            o = json.loads(o)
        except Exception:
            return float(o) if k == "p_final" else o
    return o[k] if isinstance(o, dict) else (float(o) if k == "p_final" else None)


def surgery(c, space: str, stamp: str) -> dict:
    from arize.experiments import EvaluationResult
    from blab.replay import replay_decision
    df = build_rows(2000)
    df = df[df["agent"] == "profiler"].reset_index(drop=True)
    local = {}
    for ename, vname in SURGERY.items():
        rs = [replay_decision(r.world_dir, int(r.seq), _variant(vname, r.tell_seat)) for r in df.itertuples()]
        local[ename] = pd.DataFrame({"p": [x["p_final"] for x in rs], "a": [x["action"] for x in rs]})
    recorded = float(((df["p_final"] - df["was_bluff"]) ** 2).mean())
    full_b = float(((local["surgery-full"]["p"] - df["was_bluff"]) ** 2).mean())
    print(f"surgery sanity: recorded={recorded:.6f} full-replay={full_b:.6f} n={len(df)}")
    if abs(full_b - recorded) > 1e-9:
        raise SystemExit("surgery: full replay Brier != recorded Brier; not publishing")
    full_action = dict(zip(df["state_id"], local["surgery-full"]["a"]))
    ds_name = f"surgery-r01-{stamp}"
    ds = c.datasets.create(space=space, name=ds_name, examples=df)
    out = {"dataset_id": ds.id, "dataset_name": ds_name, "n_rows": len(df), "recorded_brier": recorded,
           "sanity_full_equals_recorded": True, "experiments": []}

    def b_eval(output, dataset_row):
        p, y = _num(output, "p_final"), float(dataset_row["was_bluff"])
        return EvaluationResult(score=(p - y) ** 2, label="brier", explanation=f"p={p:.4f} was_bluff={int(y)}")

    def f_eval(output, dataset_row):
        a, a0 = _num(output, "action"), full_action[dataset_row["state_id"]]
        return EvaluationResult(score=float(a != a0), label="flipped" if a != a0 else "same",
                                explanation=f"{a0}->{a}")

    for ename, vname in SURGERY.items():
        def task(dataset_row, _v=vname):
            r = replay_decision(dataset_row["world_dir"], int(dataset_row["seq"]), _variant(_v, dataset_row["tell_seat"]))
            return {"p_final": float(r["p_final"]), "action": r["action"]}
        exp, res = c.experiments.run(name=f"{ename}-{stamp}", dataset=ds.id, space=space, task=task,
                                     evaluators={"brier": b_eval, "flipped": f_eval}, concurrency=10)
        col = lambda n: next((k for k in res.columns if n in k and k.endswith("score")), None)
        lb = float(((local[ename]["p"] - df["was_bluff"]) ** 2).mean())
        lf = int((local[ename]["a"].values != local["surgery-full"]["a"].values).sum())
        out["experiments"].append({"name": ename, "variant": vname, "experiment_id": getattr(exp, "id", None),
                                   "mean_brier_arize": float(res[col("brier")].mean()), "mean_brier_local": lb,
                                   "flips_arize": int(res[col("flipped")].sum()), "flips_local": lf})
        print(out["experiments"][-1])
    time.sleep(5)
    ids = [getattr(e, "id", None) for e in (getattr(c.experiments.list(dataset=ds.id, space=space), "experiments", None) or [])]
    for e in out["experiments"]:
        e["visible_in_listing"] = e["experiment_id"] in ids
    return out


def main_surgery() -> None:
    load_env()
    from arize import ArizeClient
    space = os.environ["ARIZE_SPACE_ID"]
    c = ArizeClient(api_key=os.environ["ARIZE_API_KEY"])
    out = surgery(c, space, datetime.now().strftime("%H%M%S"))
    p = RUN / "arize_experiment.json"
    j = json.loads(p.read_text())
    j["surgery"] = out
    p.write_text(json.dumps(j, indent=2))
    Path("show/bundle/arize_experiment.json").write_text(json.dumps(j, indent=2))
    print(json.dumps(out, indent=2))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--surgery", action="store_true")
    args = ap.parse_args(argv)
    if args.surgery:
        return main_surgery()
    load_env()
    from arize import ArizeClient
    space = os.environ["ARIZE_SPACE_ID"]
    c = ArizeClient(api_key=os.environ["ARIZE_API_KEY"])
    df = build_rows(args.cap)
    stamp = datetime.now().strftime("%H%M%S")
    print(f"rows={len(df)} seeds={df['seed'].nunique()} per-agent={df['agent'].value_counts().to_dict()}")
    out = {"dataset_name": f"frozen-decisions-r01", "datasets": [], "experiments": [],
           "selection": "r01 base worlds, telltale table, threshold rule; profiler+claude seeds and same-seed counter/bayes/logreg_bow; revealed p1 respond decisions",
           "dataset_layout": "one dataset per agent (experiments.run has no row filter)"}
    for agent, sub in df.groupby("agent"):
        sub = sub.reset_index(drop=True)
        name = f"frozen-decisions-r01-{agent}-{stamp}"
        ds = c.datasets.create(space=space, name=name, examples=sub)
        local = float(((sub["p_final"] - sub["was_bluff"]) ** 2).mean())
        exp, res = c.experiments.run(name=f"{agent}-r01-{stamp}", dataset=ds.id, space=space, task=task,
                                     evaluators=[brier], concurrency=10)
        score_col = next((k for k in res.columns if "brier" in k and k.endswith("score")), None)
        ax = float(res[score_col].mean()) if score_col else None
        out["datasets"].append({"agent": agent, "dataset_id": ds.id, "dataset_name": name})
        out["experiments"].append({"agent": agent, "dataset_id": ds.id, "experiment_id": getattr(exp, "id", None),
                                   "n_rows": len(sub), "n_scored": int(res[score_col].notna().sum()) if score_col else 0,
                                   "mean_brier_arize": ax, "mean_brier_local": local})
        print(agent, out["experiments"][-1])
    time.sleep(5)
    listed = {}
    for d in out["datasets"]:
        lr = c.experiments.list(dataset=d["dataset_id"], space=space)
        listed[d["agent"]] = [getattr(e, "id", None) for e in getattr(lr, "experiments", None) or getattr(lr, "data", None) or []]
    for e in out["experiments"]:
        e["visible_in_listing"] = e["experiment_id"] in listed.get(e["agent"], [])
    out["dataset_id"] = out["datasets"][0]["dataset_id"] if out["datasets"] else None
    out["ax_url"] = None
    out["created"] = datetime.now().isoformat(timespec="seconds")
    (RUN / "arize_experiment.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
