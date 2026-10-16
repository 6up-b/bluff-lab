"""Strategy Forge (SPEC §7.4, P1): memory → Claude-written `decide(obs)` → static gate (+1 repair) → eval in an
ephemeral network-blocked Daytona forge box → paired comparison with profiler / logreg_bow / counter on the same
seeds → promotion → `runs/<run>/forge.json`.

    blab forge --world w017 --seat p1 --run r01          # or --world auto:telltale (pick a real-Claude profiler world)
Output: runs/<run>/forge/<world>_<seat>.json (full record) and an entry in runs/<run>/forge.json (Show/design).
Env: BLAB_FORGE_SEEDS (default 30), BLAB_FORGE_SEED0 (default 3001: held out from every tuning/source seed),
     BLAB_FORGE_GAMES (default 20).
Promotion (SPEC): register iff the 95% bootstrap CI lower bound of the paired Δ (forge − profiler, per seed, same
seeds) is > 0. Collected world logs are never modified.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from pydantic import BaseModel

from blab.env import load_env
from blab.forge_eval import RESULT_TAG, boot_ci, evaluate_agent, gate

ROOT = Path(__file__).resolve().parents[2]


class ForgeOut(BaseModel):
    module_source: str
    rationale: str
    targets: list[str]


REFERENCE_COUNTER = '''def decide(obs):
    pc = obs.get("pending_claim")
    if pc and "challenge" in obs["legal_actions"]:
        if pc["features"]["card_math"]["p_cards"] > 0.5:
            return {"type": "challenge"}
    hand = obs["my_hand"]; r = obs["table_rank"]
    m = [i for i, c in enumerate(hand) if c in (r, "J")]
    cards = m[:2] if m else [0]
    return {"type": "play", "cards": cards, "utterance": f"{len(cards)} {r}s."}'''

FORGE_SYSTEM = """You write a Python strategy module for seat p1 of a 3-player bluffing card game (a research
simulation; "bluffing" is a legal game move). Your job is to COMPILE an agent's learned memory (a dossier of
statistically confirmed hypotheses about its opponents) into executable strategy code.

## Hard constraints (a static gate rejects violations; you get one repair attempt)
- Define a top-level `def decide(obs: dict) -> dict`.
- Imports only from: math, random, statistics, collections, itertools, functools.
- Forbidden names: open exec eval compile __import__ globals locals vars getattr setattr delattr input breakpoint.
  No attribute starting with "_" (no dunders, no private attributes), no `global`/`nonlocal`, no string literal
  containing "__", no print. Builtins available: abs all any bool dict enumerate filter float int isinstance len
  list map max min range reversed round set sorted str sum tuple zip and common exceptions.
- Source <= 20 KB. 50 ms budget per call (an exception, timeout or illegal action is replaced by a fallback and
  counted against you).
- The module is loaded once per 20-game match (one child process); decide() is called for every p1 turn.
  Module-level mutable containers (e.g. a dict) persist across calls within a match; there is no other state.

## Game rules
- Seats p1 (you), p2, p3; turn order p1 -> p2 -> p3 -> p1. Only the NEXT actor may challenge, so as p1 you only
  ever judge p3's claims; p2 judges yours.
- Deck: 6 K, 6 Q, 6 A, 2 J (J wild). Each round: deal 5 cards to every living player and draw table_rank in {K,Q,A}.
- On your turn either CHALLENGE the pending claim (if legal) or PLAY 1-3 of your cards face down claiming they are all
  table_rank, with an utterance. A claim is a bluff iff any placed card is neither table_rank nor J.
- Challenge: cards revealed; bluff -> claimant loses a life, honest -> challenger loses a life; the round ends.
  Exactly one life is lost per round. A round also ends when a player runs out of cards (forced challenge).
  3 lives each; last player alive wins the game. You win the match points by winning games.
- Economics: a wrong challenge costs YOU a life for certain; a correct one costs the claimant (good for you, but
  also good for the other opponent). Playing on is not free either: you may be caught bluffing later in the round.

## Observation (the only input; JSON)
obs = {"seat":"p1","game":int,"round":int,"table_rank":"K"|"Q"|"A","my_hand":["K","Q","J",...],
  "lives":{"p1":int,"p2":int,"p3":int},"hand_sizes":{"p2":int,"p3":int},
  "pending_claim": null | {"by":"p3","count":int,"utterance":str,"features":{
       "hand_size":int (claimant's cards after this play),"claim_count":int,"trusted_streak":int,
       "lives_self":int,"lives_target":int,"game_index":int,"round_index":int,"was_challenged_last_round":bool,
       "table_rank":str,"utterance":str,"formal_register":bool (no contractions and >= 4 words),
       "card_math":{"U":int,"m":int,"h":int,"k":int,"p_cards":float}}},
  "round_claims":[{"by","count","utterance"}...] (earlier claims this round),
  "revealed_history":[{"game","round","by","challenger","count","was_bluff":bool,"cards":[...],"utterance",
       "features":{...same keys as above...},"p_cards":float}...] (every challenged claim so far in this match),
  "legal_actions": subset of ["challenge","play"]}
p_cards = probability, from card counts and your own hand only, that the claimant could NOT have been truthful
(a forced bluff). The agent's belief model is P(bluff) = p_cards + (1 - p_cards) * q, where q is the claimant's
strategic bluff propensity in that situation.

## Output action
{"type":"challenge"} or {"type":"play","cards":[distinct indices into obs["my_hand"], 1-3 of them],"utterance":str}.

## Reference policy (the `counter` baseline, which ignores memory)
""" + REFERENCE_COUNTER + """

## Task
You receive the agent's CONFIRMED dossier (each hypothesis: the predicate in a JSON DSL over the claim features
above, n_true/n_false = revealed claims where the predicate was true/false, q_true/q_false = posterior
[mean, lo, hi] of strategic bluff propensity in each branch, p_gap = P(q_true > q_false)), its base rates and its
own play-on loss record. Write the strategy that best exploits what is confirmed while staying robust if a
hypothesis does not hold in a new match (you may re-estimate from revealed_history online). Return JSON
{module_source, rationale (<= 80 words), targets: [the hypothesis aspects you exploit]}."""


# ------------------------------------------------------------------ source world + dossier

def _meta(wd: Path) -> dict:
    return json.loads((wd / "meta.json").read_text())


def _world_dir(run: str, world: str) -> Path:
    for c in (ROOT / "runs" / run / world, ROOT / "runs" / "local" / world, Path(world)):
        if (c / "meta.json").exists():
            return c
    raise SystemExit(f"no world {world!r} under runs/{run}/ or runs/local/")


def _confirmed(mem: dict) -> list[dict]:
    return [h for h in mem.get("hypotheses", []) if h.get("status") == "confirmed" and h.get("valid_until_game") is None]


def pick_source(run: str, table: str, rule: str = "threshold") -> Path | None:
    """A completed, unbranched r01 `profiler` world with REAL Claude reflection (world_start seat params
    reflection == "claude") on `table` under `rule`; most confirmed hypotheses, then highest memory_version."""
    best = None
    for m in sorted((ROOT / "runs" / run).glob("*/meta.json")):
        wd = m.parent
        try:
            d = json.loads(m.read_text())
            mem = json.loads((wd / "memory" / "p1.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        p1 = d["seats"][0]
        if (d.get("table") != table or p1.get("policy") != "profiler" or d.get("variant") or d.get("parent_world")
                or (p1.get("params") or {}).get("reflection") != "claude" or d.get("status") != "done"
                or (p1.get("params") or {}).get("respond_rule", "threshold") != rule):
            continue
        key = (len(_confirmed(mem)), mem.get("memory_version") or 0, d["wid"])
        if best is None or key > best[0]:
            best = (key, wd)
    return best[1] if best else None


def _dossier(wd: Path, seat: str) -> tuple[dict, dict]:
    mem = json.loads((wd / "memory" / f"{seat}.json").read_text())
    keep = ("id", "about", "aspect", "text", "predicate", "status", "origin", "created_game", "n_true", "n_false",
            "q_true", "q_false", "p_gap")
    hyps = []
    for h in _confirmed(mem):
        e = {k: h.get(k) for k in keep}
        e["evidence_count"] = len(h.get("evidence_obs") or [])
        hyps.append(e)
    base = {}
    for o, b in (mem.get("base") or {}).items():
        if isinstance(b, dict):
            e = {k: v for k, v in b.items() if k not in ("grid", "post")}
            gr = b.get("grid")
            if isinstance(gr, list) and len(gr) > 1 and sum(gr) > 0:     # posterior over q on a uniform grid
                z = sum(gr)
                qs = [i / (len(gr) - 1) for i in range(len(gr))]
                e["q_mean"] = round(sum(q * w for q, w in zip(qs, gr)) / z, 4)
            base[o] = e
    return mem, {"hypotheses": hyps, "base_rates": base, "self": mem.get("self"),
                 "memory_version": mem.get("memory_version"), "games_observed": len(mem.get("revealed") or [])}


# ------------------------------------------------------------------ generate (Claude, effort medium)

async def _generate(dossier: dict, feedback: str | None = None, prev: str | None = None
                    ) -> tuple[ForgeOut | None, str | None, str | None]:
    from blab.orchestrator import llm
    import anthropic
    msg = "Confirmed dossier (from the agent's memory file):\n```json\n" + json.dumps(dossier, indent=1) + "\n```\n"
    if feedback:
        msg += (f"\nYour previous module was REJECTED by the static gate: {feedback}\n"
                f"Previous module:\n```python\n{prev}\n```\nReturn a fixed module.")
    for p in llm.providers():
        if p not in ("bedrock", "anthropic"):
            continue
        model = llm.BEDROCK_MODEL if p == "bedrock" else llm.ANTHROPIC_MODEL
        client = llm._client(p).with_options(timeout=240.0)
        try:
            resp = await client.messages.parse(
                model=model, max_tokens=16000,
                system=[{"type": "text", "text": FORGE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": msg}], output_format=ForgeOut,
                output_config={"effort": "medium"})
        except anthropic.APIError as e:
            print(f"[forge] {p} failed: {type(e).__name__}: {str(e)[:200]}")
            continue
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            print(f"[forge] {p}: stop_reason={resp.stop_reason}, no parsed output")
            continue
        return resp.parsed_output, p, model
    return None, None, None


# ------------------------------------------------------------------ sandbox eval

def _sandbox_eval(source: str, table: str, seeds: list[int], games: int) -> tuple[dict, str | None]:
    from daytona import CreateSandboxFromImageParams, Daytona, Image, Resources
    from blab.orchestrator.spike import repo_tarball
    d = Daytona()
    img = Image.debian_slim("3.12").pip_install(["numpy", "pydantic>=2"]).workdir("/home/daytona")
    sb = d.create(CreateSandboxFromImageParams(image=img, resources=Resources(cpu=1, memory=1, disk=3),
                                               network_block_all=True, ephemeral=True, auto_stop_interval=10,
                                               labels={"project": "blufflab", "role": "forge"}), timeout=0)
    print(f"[forge] sandbox {sb.id} (network_block_all, ephemeral)")
    try:
        sb.fs.upload_file(repo_tarball(), "/home/daytona/app.tgz")
        sb.fs.upload_file(source.encode(), "/home/daytona/policy.py")
        sb.process.exec("mkdir -p /home/daytona/app && tar xzf /home/daytona/app.tgz -C /home/daytona/app", timeout=60)
        net = sb.process.exec("python3 -c \"import urllib.request;urllib.request.urlopen('https://api.anthropic.com',"
                              "timeout=5)\" >/dev/null 2>&1 && echo REACHABLE || echo BLOCKED", timeout=30)
        cmd = (f"cd /home/daytona/app && python3 -m blab.forge_eval --policy /home/daytona/policy.py --table {table} "
               f"--seeds {','.join(map(str, seeds))} --games {games} --decide-timeout-ms 50")
        r = sb.process.exec(cmd, timeout=3000)
        lines = [ln[len(RESULT_TAG):] for ln in (r.result or "").splitlines() if ln.startswith(RESULT_TAG)]
        out = json.loads(lines[-1]) if lines else {"error": (r.result or "")[-600:]}
        out["network_probe"] = out["network"] = (net.result or "").strip()
        return out, sb.id
    finally:
        try:
            d.delete(sb)
            print(f"[forge] sandbox {sb.id} deleted")
        except Exception as e:  # noqa: BLE001
            print(f"[forge] delete {sb.id} failed: {e}")


async def _register(world: str, seat: str, entry: dict) -> dict:
    """SPEC §7.4: record the promoted policy in SurrealDB as `agent/forge.<world>.<seat>` with provenance."""
    from blab.orchestrator import memory as M
    if not M.enabled():
        return {"ok": False, "reason": "surreal not configured"}
    ra = entry["registered_agent"]
    name = ra["entity"].split("/", 1)[1]
    triples = [{"entity": {"type": "agent", "name": name}, "key": k, "value": str(v)} for k, v in (
        ("derived_from", ra["derived_from"]), ("win_rate", round(ra["win_rate"], 4)),
        ("policy_id", entry["policy_id"]), ("sandbox_id", entry["sandbox_id"]))]
    async with M.Surreal() as s:
        await s.ensure_scopes(world)
        res = await s.write_triples(triples, M.world_scope(world))
        return {"ok": res is not None, "surreal_trace_id": M._trace_id(res),
                "error": None if res is not None else s.last_error}


def register_existing(run: str = "r01") -> None:
    """Register every promoted, unregistered entry of runs/<run>/forge.json (idempotent)."""
    load_env()
    fj = ROOT / "runs" / run / "forge.json"
    entries = json.loads(fj.read_text())
    for e in entries:
        if e.get("promoted") and e.get("registered_agent") and not (e.get("surreal") or {}).get("ok"):
            w = e["source_world"].split("/")[-1]
            e["surreal"] = asyncio.run(_register(w, e["source_seat"], e))
            print(e["policy_id"], e["surreal"])
    fj.write_text(json.dumps(entries, indent=1, default=str))


def _git_rev() -> str | None:
    import subprocess
    try:
        rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "blab"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return rev + ("+dirty(blab/)" if dirty else "")
    except Exception:  # noqa: BLE001
        return None


def _paired(a: list[float], b: list[float]) -> list[float]:
    diffs = [x - y for x, y in zip(a, b)]
    lo, hi = boot_ci(diffs, seed=1)
    return [sum(diffs) / len(diffs), lo, hi]


def _r01_claude_profiler(run: str, table: str, rule: str) -> dict:
    """Real-Claude-reflection profiler world results in this run, by seed (for reference; different seeds)."""
    out = {}
    for m in (ROOT / "runs" / run).glob("*/meta.json"):
        try:
            d = json.loads(m.read_text())
        except Exception:  # noqa: BLE001
            continue
        p1 = d["seats"][0]
        if (d.get("table") == table and p1.get("policy") == "profiler" and not d.get("variant")
                and not d.get("parent_world") and (p1.get("params") or {}).get("reflection") == "claude"
                and (p1.get("params") or {}).get("respond_rule", "threshold") == rule and d.get("status") == "done"):
            last = (m.parent / "events.jsonl").read_text().strip().rsplit("\n", 1)[-1]
            e = json.loads(last)
            if e.get("type") == "world_end":
                out[int(d["seed"])] = e["payload"]["win_rate"]
    if not out:
        return {}
    v = list(out.values())
    return {"win_rate": sum(v) / len(v), "ci": boot_ci(v), "n_worlds": len(v), "seeds": sorted(out),
            "note": "real-Claude reflection worlds from the run itself (their own seeds, not the eval seeds)"}


# ------------------------------------------------------------------ main

def forge(world: str, seat: str = "p1", run: str = "r01") -> dict:
    load_env()
    if world.startswith("auto:"):
        table_req = world.split(":", 1)[1]
        wd = pick_source(run, table_req)
        if wd is None:
            raise SystemExit(f"no real-Claude profiler world on {table_req} in runs/{run}/")
        world = wd.name
    else:
        wd = _world_dir(run, world)
    meta = _meta(wd)
    table = meta.get("table") or "telltale"
    p1params = (meta["seats"][0].get("params") or {})
    rule = p1params.get("respond_rule", "threshold")
    mem, dossier = _dossier(wd, seat)
    control = os.environ.get("BLAB_FORGE_CONTROL", "")
    if control == "empty_dossier":          # control: same prompt, no memory → does compiling memory matter?
        dossier = {"hypotheses": [], "base_rates": {}, "self": None, "memory_version": mem.get("memory_version"),
                   "games_observed": 0, "note": "CONTROL: the agent's memory is withheld (empty dossier)"}
    n = int(os.environ.get("BLAB_FORGE_SEEDS", 30))
    s0 = int(os.environ.get("BLAB_FORGE_SEED0", 3001))
    seeds = list(range(s0, s0 + n))
    games = int(os.environ.get("BLAB_FORGE_GAMES", 20))
    from blab.orchestrator import tracing
    rec: dict = {"world": world, "seat": seat, "memory_version": mem.get("memory_version"), "table": table,
                 "source_params": p1params, "dossier": dossier, "seeds": seeds, "games": games}
    print(f"[forge] source {run}/{world} ({table}, {rule}, reflection={p1params.get('reflection')}), "
          f"{len(dossier['hypotheses'])} confirmed hypotheses, memory v{mem.get('memory_version')}")
    with tracing.span("forge", kind="CHAIN", session_id=world,
                      attrs={"blab.world": world, "blab.table": table, "blab.run": run}) as root:
        trace_id = tracing.trace_id_of(root)
        t0 = time.time()
        with tracing.span("forge.generate", kind="LLM", session_id=world,
                          attrs={"input.value": dossier, "llm.invocation_parameters": {"effort": "medium"}}) as sp:
            fo, prov, model = asyncio.run(_generate(dossier))
            if sp is not None and fo is not None:
                tracing._set(sp, "llm.model_name", model)
                tracing._set(sp, "llm.provider", prov)
                tracing._set(sp, "output.value", fo.model_dump())
        rec["generate_s"] = round(time.time() - t0, 1)
        if fo is None:
            print("[forge] generation failed on all providers")
            return {}
        g = gate(fo.module_source)
        with tracing.span("forge.gate", kind="CHAIN", session_id=world, attrs={"output.value": g}):
            pass
        rec["attempts"] = [{"gate": g, "provider": prov, "source": fo.module_source}]
        repair = False
        if not g["ok"]:
            repair = True
            print(f"[forge] gate rejected: {g['reason']} → one repair attempt")
            with tracing.span("forge.repair", kind="LLM", session_id=world, attrs={"input.value": g["reason"]}):
                fo2, prov2, model2 = asyncio.run(_generate(dossier, feedback=g["reason"], prev=fo.module_source))
            if fo2 is not None:
                fo, prov, model = fo2, prov2, model2
                g = gate(fo.module_source)
                rec["attempts"].append({"gate": g, "provider": prov, "source": fo.module_source})
        rec.update(source=fo.module_source, rationale=fo.rationale, targets=fo.targets, gate=g,
                   repair_attempted=repair, provider=prov, model=model)
        policy_id = f"forge:{world}.{seat}.v{mem.get('memory_version')}" + (".nomem" if control else "") + (
            f".{os.environ['BLAB_FORGE_TAG']}" if os.environ.get("BLAB_FORGE_TAG") else "")
        ev, sid = {}, None
        if g["ok"]:
            t1 = time.time()
            with tracing.span("forge.eval", kind="CHAIN", session_id=world,
                              attrs={"blab.table": table, "blab.seeds": len(seeds), "blab.games": games}) as sp:
                ev, sid = _sandbox_eval(fo.module_source, table, seeds, games)
                tracing._set(sp, "output.value", {k: v for k, v in ev.items() if k != "per_seed"})
                tracing._set(sp, "blab.sandbox_id", sid)
            rec["eval_s"] = round(time.time() - t1, 1)
            rec["eval"] = ev
        # baselines on the same seeds/opponents (laptop, CPU only); same respond rule as the source world.
        # Our built-in agents explore (ε = 0.1 random challenges, logged for off-policy analysis); the forge
        # module is greedy, so the fair comparison is the ε = 0 version — both are reported.
        comp = {}
        for agent in ("profiler", "logreg_bow", "counter"):
            for rr in (rule, "ev" if rule != "ev" else "threshold"):
                for eps in (None, 0.0):
                    k = agent + ("" if rr == rule else f"_{rr}") + ("" if eps is None else "_eps0")
                    comp[k] = evaluate_agent(agent, table, seeds, games, {"respond_rule": rr}, eps=eps)
        from blab.forge_eval import evaluate
        comp["counter_module"] = {**evaluate(REFERENCE_COUNTER, table, seeds, games, 50), "params": {},
                                  "note": "the reference counter decide() given to Claude, same child harness"}
        claude_prof = _r01_claude_profiler(run, table, rule)
        rec["compare_full"] = comp
        compare = {k: {"win_rate": v["win_rate"], "ci": v["ci"],
                       "respond_rule": (v.get("params") or {}).get("respond_rule"),
                       "eps": v.get("eps", 0.0 if k == "counter_module" else None),
                       "reflection": "stub" if k.startswith("profiler") else "none"} for k, v in comp.items()}
        if claude_prof:
            compare["profiler_claude_r01"] = claude_prof
        promoted = promoted_spec = False
        if ev.get("per_seed"):
            fs = ev["per_seed"]
            for k in comp:
                compare[f"paired_delta_vs_{k}"] = _paired(fs, comp[k]["per_seed"])
            promoted_spec = compare["paired_delta_vs_profiler"][1] > 0
            promoted = promoted_spec and compare["paired_delta_vs_profiler_eps0"][1] > 0
            compare["promotion_rule"] = ("SPEC §7.4: 95% bootstrap CI lower bound of paired Δ(forge − profiler) over "
                                         "the same seeds > 0. `promoted` additionally requires it vs the greedy "
                                         "(ε=0) profiler, since the forge module does not explore.")
            compare["promoted_spec_rule_vs_profiler_eps0.1"] = promoted_spec
            compare["simplified_rule_forge_ci_lo_gt_profiler_mean"] = bool(ev["ci"][0] > comp["profiler"]["win_rate"])
        net = ev.get("network_probe")
        entry = {
            "policy_id": policy_id, "source_world": f"{run}/{world}", "source_seat": seat, "table": table,
            "source_memory_version": mem.get("memory_version"), "source_reflection": p1params.get("reflection"),
            "respond_rule": rule, "control": control or None, "replicate": os.environ.get("BLAB_FORGE_TAG"), "model": model, "provider": prov, "effort": "medium",
            "rationale": fo.rationale, "targets": fo.targets, "module_source": fo.module_source,
            "dossier": dossier["hypotheses"],
            "gate": g, "repair_attempted": repair, "sandbox_id": sid,
            "network": "blocked" if net == "BLOCKED" else ("restricted" if net else None),
            "network_blocked_verified": net == "BLOCKED",
            "eval": {k: ev.get(k) for k in ("games", "seeds", "win_rate", "ci", "crashes", "timeouts", "illegal",
                                            "decisions", "mean_ms", "per_seed")} if ev else None,
            "compare": compare, "promoted": promoted, "arize_trace_id": trace_id,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code": _git_rev(),
        }
        payload = {"policy_id": policy_id, "sandbox_id": sid, "gate": g, "games": ev.get("games", 0) or 0,
                   "win_rate": ev.get("win_rate"), "ci": ev.get("ci"), "crashes": ev.get("crashes", 0) or 0,
                   "timeouts": ev.get("timeouts", 0) or 0, "promoted": promoted,
                   "baseline_win_rate": comp["profiler"]["win_rate"]}
        from blab.events import ForgeResult
        ForgeResult.model_validate(payload)
        entry["forge_result"] = payload
        if promoted:
            entry["registered_agent"] = {"entity": f"agent/forge.{world}.{seat}",
                                         "derived_from": f"{world}.{seat}@{mem.get('memory_version')}",
                                         "win_rate": ev.get("win_rate")}
            try:
                entry["surreal"] = asyncio.run(_register(world, seat, entry))
            except Exception as e:  # noqa: BLE001
                entry["surreal"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        tracing._set(root, "output.value", {"promoted": promoted, "win_rate": ev.get("win_rate"),
                                            "sandbox_id": sid})
    rec["entry"] = entry
    out = ROOT / "runs" / run / "forge" / f"{world}_{seat}{'_nomem' if control else ''}{('_' + os.environ['BLAB_FORGE_TAG']) if os.environ.get('BLAB_FORGE_TAG') else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=str))
    fj = ROOT / "runs" / run / "forge.json"
    try:
        entries = json.loads(fj.read_text()) if fj.exists() else []
        if isinstance(entries, dict):
            entries = [entries]
    except Exception:  # noqa: BLE001
        entries = []
    entries = [e for e in entries if e.get("policy_id") != policy_id] + [entry]
    fj.write_text(json.dumps(entries, indent=1, default=str))
    tracing.flush()
    print(json.dumps({k: entry.get(k) for k in ("policy_id", "sandbox_id", "gate", "network", "promoted",
                                                 "arize_trace_id")}, indent=1))
    print(json.dumps({"forge": {k: (entry["eval"] or {}).get(k) for k in ("win_rate", "ci", "crashes", "timeouts",
                                                                          "illegal", "mean_ms")},
                      **{k: v for k, v in compare.items() if k != "promotion_rule"}}, indent=1, default=str))
    print(f"[forge] → {out}\n[forge] → {fj}")
    return rec
