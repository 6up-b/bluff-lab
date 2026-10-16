"""Strategy Forge (SPEC §7.4, P1): memory → Claude-written `decide(obs)` → static gate → eval in an ephemeral
network-blocked Daytona forge box → compare with the Profiler on the same seeds → forge_result.

    blab forge --world w017 --seat p1 --run r01
Output: runs/<run>/forge/<world>_<seat>.json (source, gate, eval, baseline, forge_result payload).
Env: BLAB_FORGE_SEEDS (default 10), BLAB_FORGE_GAMES (default 20).
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from blab.env import load_env
from blab.forge_eval import gate

ROOT = Path(__file__).resolve().parents[2]


class ForgeOut(BaseModel):
    module_source: str
    rationale: str
    targets: list[str]


FORGE_SYSTEM = """You write a Python strategy module for one seat of a 3-player bluffing card game (a research
simulation). The module must define `def decide(obs: dict) -> dict` and use only the standard modules math, random,
statistics, collections, itertools, functools (no other imports, no file/network/OS access, no dunder attributes,
no eval/exec/getattr). It runs in a sandbox with a 50 ms budget per call.

Game: deck 6K 6Q 6A 2J (J wild); each round every living player gets 5 cards and a table_rank in {K,Q,A}. On your
turn either CHALLENGE the previous player's claim or PLAY 1-3 cards face down claiming they are all table_rank.
A claim is a bluff iff any card is neither table_rank nor J. Bluff caught -> claimant loses a life, else challenger.
3 lives. Only the next actor may challenge.

obs = {"seat","game","round","table_rank","my_hand":[...],"lives":{...},"hand_sizes":{...},
       "pending_claim": null | {"by","count","utterance","features":{hand_size, claim_count, trusted_streak,
            lives_self, lives_target, game_index, round_index, was_challenged_last_round, table_rank, utterance,
            formal_register, card_math:{U,m,h,k,p_cards}}},
       "round_claims":[...], "revealed_history":[{game, round, by, count, was_bluff, utterance, features, p_cards}],
       "legal_actions":["challenge","play"] (subset)}
p_cards = probability (from card counts) that the claimant could NOT have been truthful (a forced bluff).
Return {"type":"challenge"} or {"type":"play","cards":[indices into obs["my_hand"]],"utterance":str}.

Reference policy (the `counter` baseline):
def decide(obs):
    pc = obs.get("pending_claim")
    if pc and "challenge" in obs["legal_actions"]:
        if pc["features"]["card_math"]["p_cards"] > 0.5:
            return {"type": "challenge"}
    hand = obs["my_hand"]; r = obs["table_rank"]
    m = [i for i, c in enumerate(hand) if c in (r, "J")]
    cards = m[:2] if m else [0]
    return {"type": "play", "cards": cards, "utterance": f"{len(cards)} {r}s."}

Compile the CONFIRMED dossier you are given into the strategy (e.g. raise the bluff estimate when a confirmed
predicate matches). Return JSON {module_source, rationale, targets:[aspect...]}."""


def _world_dir(run: str, world: str) -> Path:
    for c in (ROOT / "runs" / run / world, ROOT / "runs" / "local" / world, Path(world)):
        if (c / "meta.json").exists():
            return c
    raise SystemExit(f"no world {world!r} under runs/{run}/ or runs/local/")


def _dossier(wd: Path, seat: str) -> tuple[dict, list[dict]]:
    mem = json.loads((wd / "memory" / f"{seat}.json").read_text())
    hyps = [h for h in mem.get("hypotheses", []) if h.get("status") == "confirmed"
            and h.get("valid_until_game") is None]
    keep = ("id", "about", "aspect", "text", "predicate", "n_true", "n_false", "q_true", "q_false", "provenance")
    return mem, [{k: h.get(k) for k in keep} for h in hyps]


async def _generate(dossier: list[dict], feedback: str | None = None) -> tuple[ForgeOut | None, str | None]:
    from blab.orchestrator import llm
    msg = "Confirmed dossier:\n```json\n" + json.dumps(dossier, indent=1) + "\n```\n"
    if feedback:
        msg += f"\nYour previous module was rejected by the static gate: {feedback}\nFix it."
    for p in llm.providers():
        if p == "openrouter":
            continue
        import anthropic
        client = llm._client(p)
        try:
            resp = await client.messages.parse(
                model=llm.BEDROCK_MODEL if p == "bedrock" else llm.ANTHROPIC_MODEL, max_tokens=16000,
                system=[{"type": "text", "text": FORGE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": msg}], output_format=ForgeOut,
                output_config={"effort": "medium"})
        except anthropic.APIError as e:
            print(f"[forge] {p} failed: {e}")
            continue
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            continue
        return resp.parsed_output, p
    return None, None


def _sandbox_eval(source: str, table: str, seeds: list[int], games: int) -> tuple[dict, str | None]:
    from daytona import CreateSandboxFromImageParams, Daytona, Image, Resources
    from blab.orchestrator.spike import repo_tarball
    d = Daytona()
    img = Image.debian_slim("3.12").pip_install(["numpy", "pydantic>=2"]).workdir("/home/daytona")
    sb = d.create(CreateSandboxFromImageParams(image=img, resources=Resources(cpu=1, memory=1, disk=3),
                                               network_block_all=True, ephemeral=True, auto_stop_interval=10,
                                               labels={"project": "blufflab", "role": "forge"}), timeout=0)
    try:
        sb.fs.upload_file(repo_tarball(), "/home/daytona/app.tgz")
        sb.fs.upload_file(source.encode(), "/home/daytona/policy.py")
        sb.process.exec("mkdir -p /home/daytona/app && tar xzf /home/daytona/app.tgz -C /home/daytona/app", timeout=60)
        net = sb.process.exec("python3 -c \"import urllib.request;urllib.request.urlopen('https://api.anthropic.com',"
                              "timeout=5)\" >/dev/null 2>&1 && echo REACHABLE || echo BLOCKED", timeout=30)
        cmd = (f"cd /home/daytona/app && python3 -m blab.forge_eval --policy /home/daytona/policy.py --table {table} "
               f"--seeds {','.join(map(str, seeds))} --games {games} --decide-timeout-ms 50")
        r = sb.process.exec(cmd, timeout=1800)
        lines = [ln for ln in (r.result or "").splitlines() if ln.startswith("{")]
        out = json.loads(lines[-1]) if lines else {"error": (r.result or "")[-400:]}
        out["network"] = (net.result or "").strip()
        return out, sb.id
    finally:
        try:
            d.delete(sb)
        except Exception:  # noqa: BLE001
            pass


def _baseline(table: str, seeds: list[int], games: int) -> float:
    """Profiler (stub reflection, in-process) on the same seeds."""
    from blab.world import run_local
    wrs = []
    for s in seeds:
        r = asyncio.run(run_local(table=table, agent="profiler", seed=s, games=games,
                                  out=str(ROOT / "runs" / "forge_baseline"), quiet=True, wid=f"fb-{table}-s{s}"))
        wrs.append(r["win_rate"])
    return sum(wrs) / len(wrs)


def forge(world: str, seat: str = "p1", run: str = "r01") -> dict:
    load_env()
    wd = _world_dir(run, world)
    meta = json.loads((wd / "meta.json").read_text())
    table = meta.get("table") or "telltale"
    mem, dossier = _dossier(wd, seat)
    seeds = list(range(1, int(os.environ.get("BLAB_FORGE_SEEDS", 10)) + 1))
    games = int(os.environ.get("BLAB_FORGE_GAMES", 20))
    from blab.orchestrator import tracing
    rec: dict = {"world": world, "seat": seat, "memory_version": mem.get("memory_version"), "table": table,
                 "dossier": dossier, "seeds": seeds, "games": games}
    with tracing.span("forge.generate", kind="LLM", session_id=world, attrs={"input.value": dossier}):
        fo, prov = asyncio.run(_generate(dossier))
    if fo is None:
        print("[forge] generation failed on all providers")
        return {}
    g = gate(fo.module_source)
    rec["attempts"] = [{"gate": g, "provider": prov}]
    if not g["ok"]:
        with tracing.span("forge.repair", kind="LLM", session_id=world, attrs={"input.value": g["reason"]}):
            fo2, prov = asyncio.run(_generate(dossier, feedback=g["reason"]))
        if fo2 is not None:
            fo, g = fo2, gate(fo2.module_source)
            rec["attempts"].append({"gate": g, "provider": prov})
    rec.update(source=fo.module_source, rationale=fo.rationale, targets=fo.targets, gate=g)
    policy_id = f"{world}.{seat}.v{mem.get('memory_version')}"
    payload = {"policy_id": policy_id, "sandbox_id": None, "gate": g}
    if g["ok"]:
        t0 = time.time()
        with tracing.span("forge.eval", kind="CHAIN", session_id=world, attrs={"blab.table": table}) as sp:
            ev, sid = _sandbox_eval(fo.module_source, table, seeds, games)
            tracing._set(sp, "output.value", ev) if sp is not None else None
        rec["eval"] = ev
        rec["eval_s"] = round(time.time() - t0, 1)
        base = _baseline(table, seeds, games)
        rec["baseline_profiler_stub_win_rate"] = base
        wr = ev.get("win_rate")
        payload.update(sandbox_id=sid, games=ev.get("games", 0), win_rate=wr, ci=ev.get("ci"),
                       crashes=ev.get("crashes", 0), timeouts=ev.get("timeouts", 0), baseline_win_rate=base,
                       promoted=bool(wr is not None and ev.get("ci") and ev["ci"][0] > base),
                       network=ev.get("network"), illegal=ev.get("illegal"))
    from blab.events import ForgeResult
    ForgeResult.model_validate(payload)
    rec["forge_result"] = payload
    out = ROOT / "runs" / run / "forge" / f"{world}_{seat}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=str))
    tracing.flush()
    print(json.dumps({k: payload.get(k) for k in ("policy_id", "sandbox_id", "gate", "win_rate", "ci",
                                                  "baseline_win_rate", "promoted", "network")}, indent=1))
    print(f"[forge] → {out}")
    return rec
