"""ElevenLabs TTS for a world's table talk (SPEC §7.5, §10.5). SKIP without ELEVENLABS_API_KEY.

    blab voice --run r01 --world w0002  → runs/<run>/<wid>/voice/{ask,claim,answer}_<seq>.mp3 + voice/manifest.json
Profiler voice: $ELEVENLABS_VOICE_ID, else a premade voice from GET /v1/voices (George preferred).
Opponents (p2, p3): two other distinct premade voices. Model: eleven_v4, else eleven_v3 (audio tags), else
eleven_multilingual_v2 with no tags; chosen by ONE short test call.
Tags: ask lines from the RECORDED p_final only; opponent lines from the utterance TEXT only (never hidden fields).
"""
from __future__ import annotations

import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

API = "https://api.elevenlabs.io/v1"
TAG_MODELS = ("eleven_v4", "eleven_v3")
FALLBACK_MODEL = "eleven_multilingual_v2"
_voice_cache: dict = {}
_model_cache: dict = {}

ASK_TAG_RULE = "[curious] if 0.35 <= p_final <= 0.65; [suspicious] if p_final > 0.65; else none (recorded decision p_final)"
OPP_TAG_RULE = "[stiffly] if no contractions and >= 4 words, else [casually] (utterance text only)"
_CONTRACTION = re.compile(r"[A-Za-z][’']([A-Za-z])")


def _h() -> dict:
    return {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}


def _voices() -> list[dict]:
    if "all" not in _voice_cache:
        r = httpx.get(f"{API}/voices", headers=_h(), timeout=30)
        r.raise_for_status()
        vs = r.json().get("voices", [])
        _voice_cache["all"] = [v for v in vs if v.get("category") == "premade"] or vs
    return _voice_cache["all"]


def pick_voice() -> tuple[str, str]:
    """Profiler (voice_id, name)."""
    if os.environ.get("ELEVENLABS_VOICE_ID"):
        return os.environ["ELEVENLABS_VOICE_ID"], "env:ELEVENLABS_VOICE_ID"
    if "v" in _voice_cache:
        return _voice_cache["v"]
    prem = _voices()
    pref = next((v for v in prem if v.get("name", "").split()[0] in ("George", "Brian", "Daniel", "Adam")), None)
    v = pref or prem[0]
    _voice_cache["v"] = (v["voice_id"], v.get("name", ""))
    return _voice_cache["v"]


def pick_opponent_voices() -> dict[str, tuple[str, str]]:
    """{'p2': (id, name), 'p3': (id, name)}: distinct premade voices, both different from the Profiler's."""
    prof = pick_voice()[0]
    prefs = ("Charlie", "Callum", "Liam", "Will", "Eric", "Chris", "Roger", "River")
    rest = [v for v in _voices() if v["voice_id"] != prof]
    rest.sort(key=lambda v: (prefs.index(v["name"].split()[0]) if v.get("name", "").split()[0] in prefs else 99))
    assert len(rest) >= 2, "need two premade voices besides the Profiler's"
    return {"p2": (rest[0]["voice_id"], rest[0].get("name", "")), "p3": (rest[1]["voice_id"], rest[1].get("name", ""))}


def _post(text: str, vid: str, model: str) -> httpx.Response:
    return httpx.post(f"{API}/text-to-speech/{vid}", params={"output_format": "mp3_44100_128"}, headers=_h(),
                      json={"text": text, "model_id": model}, timeout=90)


def pick_model() -> str:
    """ONE test call per candidate tag model; the first the plan accepts wins, else the v2 fallback (no tags)."""
    if os.environ.get("ELEVENLABS_MODEL_ID"):
        return os.environ["ELEVENLABS_MODEL_ID"]
    if "m" not in _model_cache:
        vid = pick_voice()[0]
        _model_cache["m"] = FALLBACK_MODEL
        for m in TAG_MODELS:
            r = _post("[curious] Hm.", vid, m)
            if r.status_code == 200 and r.content:
                _model_cache["m"] = m
                break
            print(f"[voice] model {m} rejected: {r.status_code}")
    return _model_cache["m"]


def tags_ok(model: str) -> bool:
    return model in TAG_MODELS


def ask_tag(p_final: float | None) -> str:
    """Profiler tone from the recorded decision probability only."""
    if p_final is None:
        return ""
    if 0.35 <= p_final <= 0.65:
        return "[curious]"
    return "[suspicious]" if p_final > 0.65 else ""


def opponent_tag(utterance: str) -> str:
    """Opponent tone from the spoken TEXT only. Takes one string; hidden fields (was_bluff, cards) cannot reach it."""
    assert isinstance(utterance, str), "opponent_tag takes only the utterance string"
    formal = not _CONTRACTION.search(utterance) and len(utterance.split()) >= 4
    return "[stiffly]" if formal else "[casually]"


def tts(text: str, out: str | os.PathLike, voice_id: str | None = None, model: str | None = None) -> str | None:
    if not os.environ.get("ELEVENLABS_API_KEY"):
        print("SKIP voice: missing ELEVENLABS_API_KEY")
        return None
    vid = voice_id or pick_voice()[0]
    model = model or pick_model()
    from blab.orchestrator import tracing
    with tracing.span("voice.tts", kind="CHAIN", attrs={"input.value": text, "blab.voice": vid, "blab.model": model}):
        r = _post(text, vid, model)
    if r.status_code != 200 or not r.content:
        print(f"[voice] TTS failed {r.status_code}: {r.text[:200]}")
        return None
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(r.content)
    return str(p)


def _world_dir(run: str, world: str) -> Path:
    for cand in (Path("runs") / run / world, Path("runs/local") / world, Path(world)):
        if (cand / "events.jsonl").exists():
            return cand
    raise SystemExit(f"no events.jsonl for world {world!r} under runs/{run}/ or runs/local/")


def _show_rounds(events: list[dict], wd: Path) -> list[tuple[int, int]]:
    """Rounds whose claims/answers appear in the Show's cues: the cold-open round and the decision round."""
    rounds: dict[tuple, set] = {}
    for e in events:
        if e.get("game") and e.get("round"):
            rounds.setdefault((e["game"], e["round"]), set()).add(e["type"])
    out = [next((k for k in sorted(rounds) if {"deal", "challenge", "reveal"} <= rounds[k]), None)]
    for rp in (wd.parent.parent.parent / "show" / "bundle" / "reveal.json", Path("show/bundle/reveal.json")):
        if rp.exists():
            cs = json.loads(rp.read_text()).get("claim_seq")
            ce = next((e for e in events if e["seq"] == cs), None)
            if ce:
                out.append((ce["game"], ce["round"]))
            break
    return [k for k in dict.fromkeys(out) if k]


def voice_for_world(run: str = "r01", world: str = "", redo: bool = True) -> list[dict]:
    from blab.env import load_env
    load_env()
    if not os.environ.get("ELEVENLABS_API_KEY"):
        print("SKIP voice: missing ELEVENLABS_API_KEY")
        return []
    wd = _world_dir(run, world)
    from blab.events import read_events
    evs = sorted(read_events(wd / "events.jsonl"), key=lambda e: e["seq"])
    model = pick_model()
    tagged = tags_ok(model)
    vid, name = pick_voice()
    opp = pick_opponent_voices()
    dec_p = {}
    for e in evs:
        if e["type"] == "decision":
            dec_p.setdefault((e["game"], e["round"], e["payload"].get("target")), e["payload"].get("p_final"))
    vdir = wd / "voice"
    vdir.mkdir(exist_ok=True)
    jobs = []   # (kind, event, voice_id, tag, rule)
    for e in evs:
        if e["type"] == "ask":
            pf = e["payload"].get("p_final")
            if pf is None:
                pf = dec_p.get((e["game"], e["round"], e["payload"].get("target")))
            jobs.append(("ask", e, vid, ask_tag(pf) if tagged else "", pf))
    show = set(_show_rounds(evs, wd))
    for e in evs:
        if e["type"] in ("claim", "answer") and (e["game"], e["round"]) in show and e["actor"] in opp:
            text = e["payload"].get("utterance") if e["type"] == "claim" else e["payload"].get("text")
            if text:
                jobs.append((e["type"], e, opp[e["actor"]][0], opponent_tag(text) if tagged else "", None))

    def text_of(j):
        k, e = j[0], j[1]
        return e["payload"]["utterance"] if k == "claim" else e["payload"]["text"]
    # dedupe identical (voice, tag, text) synth requests
    uniq: dict[tuple, Path] = {}
    for j in jobs:
        key = (j[2], j[3], text_of(j))
        uniq.setdefault(key, vdir / f"{j[0]}_{j[1]['seq']}.mp3")

    def synth(item):
        (v, tag, text), path = item
        if redo or not path.exists():
            tts(f"{tag} {text}".strip(), path, voice_id=v, model=model)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(synth, uniq.items()))
    out = {"ask": [], "claim": [], "answer": []}
    for j in jobs:
        kind, e, v, tag, pf = j
        path = vdir / f"{kind}_{e['seq']}.mp3"
        src = uniq[(v, tag, text_of(j))]
        if src != path and src.exists():
            shutil.copyfile(src, path)
        row = {"seq": e["seq"], "game": e["game"], "round": e["round"], "actor": e["actor"], "text": text_of(j),
               "tag": tag or None, "voice_id": v, "file": str(path.relative_to(wd)) if path.exists() else None}
        if kind == "ask":
            row["p_final"] = pf
        out[kind].append(row)
    man = {"voice_id": vid, "voice_name": name, "model_id": model, "tags": tagged,
           "opponent_voices": {k: {"voice_id": a, "voice_name": b} for k, (a, b) in opp.items()},
           "tag_rules": {"ask": ASK_TAG_RULE if tagged else "none (model without audio tags)",
                         "opponent": OPP_TAG_RULE if tagged else "none (model without audio tags)",
                         "never": "no tag is derived from was_bluff, cards or any hidden field; no [nervous]/[stammers]"},
           "show_rounds": sorted(show), "asks": out["ask"], "claims": out["claim"], "answers": out["answer"]}
    (vdir / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"[voice] {len(out['ask'])} ask, {len(out['claim'])} claim, {len(out['answer'])} answer lines → {vdir} "
          f"(model {model}; profiler {name}; p2 {opp['p2'][1]}; p3 {opp['p3'][1]})")
    return out["ask"] + out["claim"] + out["answer"]
