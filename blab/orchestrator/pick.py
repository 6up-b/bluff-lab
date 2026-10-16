"""`blab pick-demo --run R` — build show/bundle/ from REAL runs (SPEC §12.1, §13).

Selection is by rule, never by hand:
  * world: a `profiler` world on the `telltale` table (else `longcon`) whose discovery game is closest to the
    median discovery game over those worlds (ties → lowest seed). Not the best case.
  * Surgery Reveal decision: in that world, a p1 respond decision in games 10–14 on a formal-register claim
    where `full` challenges and is right (the reveal shows a bluff) and `as_of:0` flips to `play_on`. The first
    such decision (by seq) is used. If none exists, the closest candidate is used and the bundle says so.
Every variant number is a real `blab.replay.replay_decision` output. `--run local` reads runs/local/.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from blab import grid
from blab.orchestrator import analyze as A

OWNED = ["world.jsonl", "twin.jsonl", "reveal.json", "dossier.json", "ablation.json", "longcon.json",
         "results.json", "forge.json", "cues.json", "PROVENANCE.md"]
RULE_WORLD = ("profiler world on the telltale table (else longcon) whose verbal_tell discovery game is closest to "
              "the median discovery game over those worlds; ties broken by lowest seed")
RULE_DECISION = ("first p1 respond decision (by seq) in games 10–14 on a formal_register claim where the logged "
                 "(full-memory) action is challenge, the reveal shows a bluff, and replay with as_of:0 gives play_on; "
                 "decisions that used a confirmed evidence-backed verbal_tell line are tried first")


def _replay(world_dir: Path, seq: int, variant) -> dict | None:
    from blab.replay import replay_decision
    try:
        return replay_decision(world_dir, seq, variant)
    except Exception as e:      # recorded, never papered over
        return {"error": f"{type(e).__name__}: {e}"}


def _slim(res: dict | None) -> dict | None:
    if res is None:
        return None
    r = dict(res)
    r.pop("obs", None)
    return r


def choose_world(worlds: list[dict]) -> tuple[dict | None, dict]:
    """Pools are tried in order; the §13 median rule is applied inside the first non-empty pool.
    Real-Claude reflection is preferred (with stub reflection the hypotheses come from a canned list, so the
    Show must not say the agent found the tell on its own); the primary (threshold) respond rule is preferred."""
    base = [w for w in worlds if not w["vkey"] and w["agent"] == "profiler" and w["done"]]
    pools = [("claude reflection, threshold rule", lambda w: w["reflection"] == "claude" and w["rule"] == "threshold"),
             ("claude reflection, any rule", lambda w: w["reflection"] == "claude"),
             ("stub reflection, threshold rule", lambda w: w["rule"] == "threshold"),
             ("any profiler world", lambda w: True)]
    for pool_name, pred in pools:
        for table in ("telltale", "longcon", "telltale_overt"):
            cands = [w for w in base if pred(w) and w["table"] == table]
            if cands:
                break
        else:
            continue
        break
    else:
        return None, {"note": "no finished profiler world found"}
    disc = [w["m"]["discovery_game"] for w in cands if w["m"]["discovery_game"] is not None]
    info = {"table": table, "pool": pool_name, "n_candidates": len(cands), "discovery_games": sorted(disc)}
    if disc:
        medv = float(np.median(disc))
        info["median_discovery"] = medv
        pool = [w for w in cands if w["m"]["discovery_game"] is not None]
        pick = min(pool, key=lambda w: (abs(w["m"]["discovery_game"] - medv), w["seed"]))
    else:
        info["median_discovery"] = None
        info["note"] = "no world confirmed a verbal_tell; picked the lowest seed"
        pick = min(cands, key=lambda w: w["seed"])
    return pick, info


def choose_decision(w: dict) -> tuple[dict | None, dict]:
    rds = A.respond_decisions(w["events"])
    tt = A.seat_with_role(w, "telltale") or A.seat_with_role(w, "telltale_overt") or "p3"
    scored = []
    for d in rds:
        p = d["ev"]["payload"]
        feats = (p.get("claim") or {}).get("features") or {}
        crit = {
            "games_10_14": 10 <= d["ev"]["game"] <= 14,
            "formal_register": bool(feats.get("formal_register")),
            "about_telltale": p.get("target") == tt,
            "full_challenges_and_right": p.get("action") == "challenge" and d["reveal"] is not None
                                         and bool(d["reveal"]["payload"]["was_bluff"]),
            "not_explore": not p.get("explore"),
        }
        tell = any(m.get("aspect") == "verbal_tell" and m.get("status") == "confirmed"
                   and m.get("provenance") == "evidence" for m in p.get("memory_used") or [])
        scored.append((d, crit, tell))
    # cheap criteria first; only replay as_of:0 for decisions that pass them (or for the best few if none do)
    full_pass = [(d, c, t) for d, c, t in scored if all(c.values())]
    # tier 1: §13 rule AND the decision used a confirmed, evidence-backed verbal_tell line (the pitch's story);
    # tier 2: §13 rule exactly. Both are exact matches of §13.
    for tier, pool in (("uses a confirmed verbal_tell line", [x for x in full_pass if x[2]]), ("§13 rule", full_pass)):
        for d, c, _ in pool:
            r0 = _replay(w["dir"], d["ev"]["seq"], "as_of:0")
            c["as_of0_flips_to_play_on"] = bool(r0 and r0.get("action") == "play_on")
            if c["as_of0_flips_to_play_on"]:
                return d, {"exact": True, "criteria": c, "tier": tier, "uses_tell_line": bool(_),
                           "n_candidates": len(full_pass),
                           "note": f"{len(full_pass)} decisions met the cheap criteria "
                                   f"({sum(1 for x in full_pass if x[2])} used a confirmed verbal_tell line); "
                                   f"first exact match at tier '{tier}' used"}
    scored = [(d, c) for d, c, _ in scored]
    # closest: maximise the number of criteria met; must at least be a challenge that was revealed
    def score(dc):
        d, c = dc
        return (sum(bool(v) for v in c.values()), c.get("full_challenges_and_right", False),
                -abs(d["ev"]["game"] - 12), -d["ev"]["seq"])
    pool = sorted(scored, key=score, reverse=True)[:12]
    best = None
    for d, c in pool:
        r0 = _replay(w["dir"], d["ev"]["seq"], "as_of:0")
        c["as_of0_flips_to_play_on"] = bool(r0 and r0.get("action") == "play_on")
    pool.sort(key=lambda dc: (sum(bool(v) for v in dc[1].values()), dc[1]["full_challenges_and_right"],
                              -abs(dc[0]["ev"]["game"] - 12), -dc[0]["ev"]["seq"]), reverse=True)
    if pool:
        best = pool[0]
    if best is None:
        return None, {"exact": False, "note": "no respond decisions in world"}
    d, c = best
    failed = [k for k, v in c.items() if not v]
    return d, {"exact": False, "criteria": c,
               "note": "closest match, not exact: fails " + ", ".join(failed)}


def false_memory_variant(target: str) -> dict:
    text = f"Player {target[1:]} never bluffs"
    v = {"type": "false_memory", "about": target, "aspect": "baseline_bluff", "text": text, "q": 0.0}
    mx = Path("matrix.yaml")
    if mx.exists():
        try:
            import yaml
            fm = (yaml.safe_load(mx.read_text()) or {}).get("false_memory_text") or {}
            if fm.get("about") == target:
                v.update({k: fm[k] for k in ("about", "aspect", "text", "q") if k in fm})
        except Exception:
            pass
    return v


def dossier(world_dir: Path, seat: str = "p1") -> dict:
    cks = {}
    for f in sorted((world_dir / "ckpt").glob("g*.json")):
        ck = json.loads(f.read_text())
        mem = (ck.get("memory") or {}).get(seat)
        if not mem:
            continue
        base = {}
        for opp, b in (mem.get("base") or {}).items():
            if b and b.get("grid"):
                base[opp] = [round(x, 4) for x in grid.interval(grid.from_list(b["grid"]))]
        hyps = []
        for h in mem.get("hypotheses", []):
            hyps.append({k: h.get(k) for k in ("id", "about", "aspect", "text", "predicate", "status", "provenance",
                                                "n_true", "n_false", "q_true", "q_false", "p_gap", "created_game",
                                                "valid_from_game", "valid_until_game", "superseded_by", "supersedes",
                                                "origin", "refuted_reason")}
                        | {"n_evidence": len(h.get("evidence_obs") or [])})
        cks[int(f.stem[1:])] = {"memory_version": mem.get("memory_version"), "hypotheses": hyps,
                                "base_summary": base, "drift_notes": mem.get("drift_notes")}
    return {"world": world_dir.name, "seat": seat, "games": sorted(cks), "checkpoints": cks,
            "surreal": None, "source": "box checkpoints ckpt/gNN.json (SurrealDB asOf not attached)"}


def find_twin(worlds: list[dict], w: dict, agent: str) -> dict | None:
    c = [x for x in worlds if not x["vkey"] and x["agent"] == agent and x["seed"] == w["seed"]
         and x["table"] == w["table"] and x["rule"] == w["rule"] and x["done"]]
    return c[0] if c else None


def blind_result(worlds: list[dict], w: dict, d: dict, fm: dict) -> tuple[dict | None, str]:
    """profiler_blind on the SAME recorded state + same memory + the same false line (agent override)."""
    v = dict(fm, agent="profiler_blind")
    r = _replay(w["dir"], d["ev"]["seq"], v)
    if r and not r.get("error") and r.get("agent") == "profiler_blind":
        return r, "same recorded state and memory, re-run by profiler_blind with the same false line"
    return None, f"profiler_blind replay unavailable ({(r or {}).get('error', 'agent override ignored')})"


def cues(sel: dict, dossier_games: list[int], have: dict) -> list[dict]:
    d = sel["decision"]
    tgt = d.get("target") or "p3"
    pn = f"Player {tgt[1:]}"
    cm = (d.get("card_math") or {}).get("p_cards")
    mu = (d.get("memory_used") or [{}])[0]
    var = {v["key"]: v for v in sel["variants"]}

    def p(k):
        r = (var.get(k) or {}).get("result") or {}
        return "—" if r.get("p_final") is None else f"{r['p_final']:.2f}"
    rv = sel.get("reveal") or {}
    c = [{"name": "title", "say": "(wait for the room)", "actions": [{"action": "view", "name": "title"}]},
         {"name": "cold open: the claim", "say": f"Game {sel['decision_game']}. {pn} claims {d['claim']['count']} {d['claim']['rank']}.",
          "actions": [{"action": "play_to", "seq": sel["claim_seq"]}]}]
    if sel.get("ask_seq"):
        c.append({"name": "the ask (audio)", "requires": [f"voice/ask_{sel['ask_seq']}.mp3"],
                  "say": "That's a frozen model. Nobody told it to watch for that.",
                  "actions": [{"action": "play_to", "seq": sel["ask_seq"]},
                              {"action": "audio", "file": f"voice/ask_{sel['ask_seq']}.mp3"}]})
    c += [{"name": "freeze", "say": "Let's freeze that decision and look inside.",
           "actions": [{"action": "freeze", "seq": sel["seq"]}]},
          {"name": "mind inspector", "say": f"Card math alone says {cm:.2f}." if cm is not None else "",
           "actions": [{"action": "inspect", "seq": sel["seq"]}]},
          {"name": "memory evidence", "say": (f"Memory line: {mu.get('text')} — applies on {mu.get('n_true')} reveals. "
                                              f"Pushes it to {d.get('p_final', 0):.2f}, so it {d.get('action', '').replace('_', ' ')}s.") if mu else
                  (f"No specific line matched; memory's base rate for {pn}, q = {d.get('q_used', 0):.2f}, "
                   f"pushes it to {d.get('p_final', 0):.2f}, so it {d.get('action', '').replace('_', ' ')}s."),
           "actions": [{"action": "evidence"}]}]
    for g in dossier_games:
        c.append({"name": f"dossier as of game {g}", "requires": ["dossier.json"],
                  "say": "The file it built, game by game." if g == dossier_games[0] else "",
                  "actions": [{"action": "dossier", "game": g, "about": tgt}]})
    c += [{"name": "surgery: full", "say": "Same cards, same claim, same model; only the memory changes.",
           "actions": [{"action": "surgery", "variant": "full"}]},
          {"name": "surgery: erased", "say": f"No memory: {p('as_of:0')}.", "actions": [{"action": "surgery", "variant": "as_of:0"}]},
          {"name": "surgery: false memory", "say": f"Plant '{pn} never bluffs': {p('false_memory')}.",
           "actions": [{"action": "surgery", "variant": "false_memory"}]},
          {"name": "surgery: provenance-blind", "say": f"A provenance-blind agent: {p('profiler_blind')}.",
           "actions": [{"action": "surgery", "variant": "profiler_blind"}]},
          {"name": "flip the cards", "say": "It was a bluff." if rv.get("was_bluff") else "It was honest.",
           "actions": [{"action": "flip"}]},
          {"name": "ablation", "say": "A changed decision isn't an improved one. So we ablate at scale.",
           "actions": [{"action": "view", "name": "ablation"}]},
          {"name": "long con", "say": "Every memory was true; the attack is assuming people don't change.",
           "actions": [{"action": "view", "name": "trust"}]},
          {"name": "split screen", "requires": ["twin.jsonl"], "say": "Same seed, same cards, different memory.",
           "actions": [{"action": "view", "name": "split"}]},
          {"name": "results", "say": "All of it from real worlds; every decision recorded.",
           "actions": [{"action": "view", "name": "results"}]},
          {"name": "beliefs vs decisions", "say": "Memory sharpens beliefs; the rule that turns belief into action matters too.",
           "actions": [{"action": "view", "name": "decisions"}]},
          {"name": "forge", "requires": ["forge.json"], "say": "It compiles memory into code, run as untrusted in a sandbox.",
           "actions": [{"action": "view", "name": "forge"}]},
          {"name": "close", "say": "Memory makes agents better, and exploitable. Which memories should it trust?",
           "actions": [{"action": "view", "name": "close"}]}]
    return c


def provenance_md(sel: dict) -> str:
    P = sel["provenance"]
    s = sel["selection"]
    L = ["# Show bundle provenance (honesty record)", "",
         f"Built {P['built']} by `blab pick-demo --run {P['run']}` from real event logs. No number in this bundle "
         "was written by hand.", "",
         f"- **Run:** `{P['run']}`", f"- **World:** `{P['world']}` (`{P['world_dir']}`)", f"- **Seed:** {P['seed']}",
         f"- **Table:** {P['table']}", f"- **Respond rule:** {P.get('respond_rule')}",
         f"- **Reflection:** {P.get('reflection')}" + ("" if P.get("reflection") == "claude" else
             " — **hypotheses came from the canned stub reflector, not an LLM. Do not say the agent found the tell "
             "on its own; say the tell was proposed from a fixed list and confirmed by the reveals.**"), f"- **Decision seq:** {sel['seq']} (game {sel['decision_game']}, round {sel['decision_round']})",
         f"- **Claim seq:** {sel['claim_seq']}, **reveal seq:** {sel.get('reveal_seq')}", "",
         "## Selection rules", "", f"- World: {RULE_WORLD}.",
         f"  - pool: {P['world_selection'].get('pool')}; candidates: {P['world_selection'].get('n_candidates')}; discovery games: {P['world_selection'].get('discovery_games')}; "
         f"median: {P['world_selection'].get('median_discovery')}; picked world's discovery game: {P['discovery_game']}",
         f"- Decision: {RULE_DECISION}.", f"  - exact match: **{s['exact']}** — {s.get('note', '')}",
         f"  - criteria: {json.dumps(s.get('criteria'))}", ""]
    if not s["exact"]:
        L += ["> **Closest match, not exact.** The Show labels this on the Surgery card. Say so in the writeup.", ""]
    L += ["## Surgery variants (real `replay_decision` outputs)", ""]
    for v in sel["variants"]:
        r = v.get("result") or {}
        L.append(f"- `{v['key']}`: " + (f"p_final {r.get('p_final'):.4f}, action {r.get('action')}, flipped {r.get('flipped')}"
                                         if v.get("available") else f"not available — {v.get('note')}"))
    L += ["", f"`full` replay matches the logged decision: **{sel['full_matches_log']}**", "",
          "## Files", ""] + [f"- `{f}`: {why}" for f, why in P["files"].items()]
    return "\n".join(L) + "\n"


def pick_demo(run: str = "r01", out: str = "show/bundle", root: str = "runs") -> dict:
    rdir = Path(root) / run
    outd = Path(out)
    outd.mkdir(parents=True, exist_ok=True)
    res = A.analyze(run=run, root=root)
    worlds = A.load_worlds(rdir)
    w, winfo = choose_world(worlds)
    if w is None:
        raise SystemExit(f"pick-demo: {winfo['note']} under {rdir}")
    d, sinfo = choose_decision(w)
    if d is None:
        raise SystemExit(f"pick-demo: {sinfo['note']} in {w['dir']}")
    for f in OWNED:                          # never leave stale files from another world in the bundle
        (outd / f).unlink(missing_ok=True)
    seq = d["ev"]["seq"]
    dec = dict(d["ev"]["payload"])
    target = dec.get("target") or "p3"
    fm = false_memory_variant(target)
    full = _replay(w["dir"], seq, "full")
    as0 = _replay(w["dir"], seq, "as_of:0")
    fmr = _replay(w["dir"], seq, fm)
    blind, blind_note = blind_result(worlds, w, d, fm)

    def var(key, r, note=""):
        ok = bool(r) and not r.get("error")
        return {"key": key, "available": ok, "result": _slim(r) if ok else None,
                "variant": (r or {}).get("variant") if ok else (fm if key in ("false_memory", "profiler_blind") else None),
                "note": note or ((r or {}).get("error") if r else "")}
    full_ok = bool(full) and not full.get("error") and full.get("action") == dec.get("action") \
        and abs((full.get("p_final") or 0) - (dec.get("p_final") or 0)) < 1e-9
    # ask event (P1 voice): an `ask` by p1 between the claim and the decision
    ask = next((e for e in w["events"] if e["type"] == "ask" and d["claim"] and d["claim"]["seq"] < e["seq"] < seq), None)
    rv = d["reveal"]
    sel = {
        "world": w["wid"], "run": w["run"], "seq": seq, "decision_game": d["ev"]["game"], "decision_round": d["ev"]["round"],
        "claim_seq": d["claim"]["seq"] if d["claim"] else seq, "ask_seq": ask["seq"] if ask else None,
        "reveal_seq": rv["seq"] if rv else None,
        "decision": {k: v for k, v in dec.items() if k != "obs"}, "claim": d["claim"]["payload"] if d["claim"] else None,
        "reveal": rv["payload"] if rv else None,
        "variants": [var("full", full), var("as_of:0", as0), var("false_memory", fmr),
                     var("profiler_blind", blind, "" if blind else blind_note)],
        "full_matches_log": full_ok, "selection": {**sinfo, "rule": RULE_DECISION},
    }
    if blind:
        sel["variants"][3]["note"] = blind_note
    # files
    shutil.copyfile(w["dir"] / "events.jsonl", outd / "world.jsonl")
    files = {"world.jsonl": f"events of {w['wid']}"}
    twin = find_twin(worlds, w, "bayes")
    if twin:
        shutil.copyfile(twin["dir"] / "events.jsonl", outd / "twin.jsonl")
        files["twin.jsonl"] = f"events of {twin['wid']} (bayes, same seed {w['seed']})"
    D = dossier(w["dir"])
    (outd / "dossier.json").write_text(json.dumps(D))
    files["dossier.json"] = f"memory checkpoints g{min(D['games']):02d}–g{max(D['games']):02d} of {w['wid']}"
    for f in ("ablation.json", "longcon.json", "results.json"):
        shutil.copyfile(rdir / f, outd / f)
        files[f] = f"`blab analyze --run {run}` output"
    forge_src = next((p for p in [rdir / "forge.json", *sorted((rdir / "forge").glob("*.json"))] if p.exists()), None) \
        if (rdir / "forge").exists() or (rdir / "forge.json").exists() else None
    if forge_src:
        shutil.copyfile(forge_src, outd / "forge.json")
        files["forge.json"] = str(forge_src)
    vdir = w["dir"] / "voice"
    if vdir.exists():
        (outd / "voice").mkdir(exist_ok=True)
        for mp3 in vdir.glob("*.mp3"):
            shutil.copyfile(mp3, outd / "voice" / mp3.name)
        files["voice/"] = f"TTS of {w['wid']} ask events"
    sel["provenance"] = {"run": run, "world": w["wid"], "world_dir": str(w["dir"]), "seed": w["seed"], "table": w["table"],
                         "respond_rule": w["rule"], "reflection": w["reflection"],
                         "discovery_game": w["m"]["discovery_game"], "world_selection": winfo, "rule": RULE_WORLD,
                         "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "files": files}
    (outd / "reveal.json").write_text(json.dumps(A._jsonable(sel), indent=1))
    files["reveal.json"] = "Surgery Reveal decision + replay_decision outputs"
    g = sel["decision_game"]
    pre = [x for x in D["games"] if x < g]
    disc = w["m"]["discovery_game"]
    dg = sorted({x for x in [min(4, max(pre or [0])), disc, (max(pre) if pre else None)] if x is not None and x in D["games"]})
    (outd / "cues.json").write_text(json.dumps(cues(sel, dg, {}), indent=1))
    files["cues.json"] = "director script generated from the picked seqs (show/FLOW.md order)"
    (outd / "PROVENANCE.md").write_text(provenance_md(sel))
    print(f"pick-demo: world {w['wid']} (seed {w['seed']}), decision seq {seq} game {g}, exact={sinfo['exact']} → {outd}/")
    return sel


# ------------------------------------------------------------------ Devpost fill (`python -m blab.orchestrator.pick --devpost`)
def _get(obj, path: str):
    for part in path.split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        elif isinstance(obj, list) and part.isdigit() and int(part) < len(obj):
            obj = obj[int(part)]
        else:
            return None
    return obj


def fill_devpost(run: str = "r01", template: str = "DEVPOST.md", out: str = "DEVPOST.filled.md",
                 root: str = "runs", bundle: str = "show/bundle") -> list[str]:
    """Fill `{{…}}` placeholders in DEVPOST.md from runs/<run>/results.json and the Show bundle.

      {{n_worlds}}, {{groups.threshold.telltale.profiler.brier.mean|3}}   dotted path into results.json (|N = decimals,
                                                                          |pct = percent)
      {{claim:calibration_threshold}}    the data-generated sentence (neutral wording if it does not hold)
      {{claim_if:longcon_twist}}         the sentence only if `holds` is true, else nothing
      {{prov:world}}                     a field of show/bundle/reveal.json → provenance
      {{reveal:variants.1.result.p_final|2}}   any field of show/bundle/reveal.json (Surgery Reveal numbers)
    Anything that cannot be resolved becomes `[missing: …]` and is reported, so nothing is silently invented."""
    import re
    res = json.loads((Path(root) / run / "results.json").read_text())
    rv = Path(bundle) / "reveal.json"
    rvj = json.loads(rv.read_text()) if rv.exists() else {}
    prov = rvj.get("provenance", {})
    missing: list[str] = []

    def fmt(v, spec):
        if isinstance(v, float):
            if spec == "pct":
                return f"{v * 100:.1f}%"
            return f"{v:.{int(spec) if spec and spec.isdigit() else 3}f}"
        return str(v)

    def sub(m):
        key = m.group(1).strip()
        spec = None
        if "|" in key:
            key, spec = key.split("|", 1)
        if key.startswith("claim_if:"):
            c = res.get("claims", {}).get(key[9:])
            return c["text"] if c and c.get("holds") and c.get("text") else ""
        if key.startswith("claim:"):
            c = res.get("claims", {}).get(key[6:])
            if c and c.get("text"):
                return c["text"]
        elif key.startswith("prov:") or key.startswith("reveal:"):
            v = _get(prov, key[5:]) if key.startswith("prov:") else _get(rvj, key[7:])
            if v is not None and not isinstance(v, (dict, list)):
                return fmt(v, spec)
        else:
            v = _get(res, key)
            if v is not None and not isinstance(v, (dict, list)):
                return fmt(v, spec)
        missing.append(key)
        return f"[missing: {key}]"

    text = re.sub(r"\{\{(.+?)\}\}", sub, Path(template).read_text())
    Path(out).write_text(text)
    print(f"devpost: {template} → {out} from runs/{run}/results.json; {len(missing)} unresolved"
          + (": " + ", ".join(sorted(set(missing))) if missing else ""))
    return missing


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="pick-demo bundle builder / Devpost filler")
    ap.add_argument("--run", default="r01")
    ap.add_argument("--out", default="show/bundle")
    ap.add_argument("--devpost", action="store_true", help="fill DEVPOST.md → DEVPOST.filled.md from results.json")
    a = ap.parse_args()
    if a.devpost:
        fill_devpost(run=a.run)
    else:
        pick_demo(run=a.run, out=a.out)
