"""ElevenLabs TTS for a world's `ask` events (SPEC §7.5, §10.5). SKIP without ELEVENLABS_API_KEY.

    blab voice --run r01 --world w017   → runs/<run>/<wid>/voice/ask_<seq>.mp3 + voice/manifest.json
Voice: $ELEVENLABS_VOICE_ID, else the first premade voice from GET /v1/voices (logged in the manifest).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

API = "https://api.elevenlabs.io/v1"
MODEL = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
_voice_cache: dict = {}


def _h() -> dict:
    return {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}


def pick_voice() -> tuple[str, str]:
    """(voice_id, name)."""
    if os.environ.get("ELEVENLABS_VOICE_ID"):
        return os.environ["ELEVENLABS_VOICE_ID"], "env:ELEVENLABS_VOICE_ID"
    if "v" in _voice_cache:
        return _voice_cache["v"]
    r = httpx.get(f"{API}/voices", headers=_h(), timeout=30)
    r.raise_for_status()
    voices = r.json().get("voices", [])
    prem = [v for v in voices if v.get("category") == "premade"] or voices
    pref = next((v for v in prem if v.get("name", "").split()[0] in ("George", "Brian", "Daniel", "Adam")), None)
    v = pref or prem[0]
    _voice_cache["v"] = (v["voice_id"], v.get("name", ""))
    return _voice_cache["v"]


def tts(text: str, out: str | os.PathLike) -> str | None:
    if not os.environ.get("ELEVENLABS_API_KEY"):
        print("SKIP voice: missing ELEVENLABS_API_KEY")
        return None
    vid, name = pick_voice()
    from blab.orchestrator import tracing
    with tracing.span("voice.tts", kind="CHAIN", attrs={"input.value": text, "blab.voice": name}):
        r = httpx.post(f"{API}/text-to-speech/{vid}", params={"output_format": "mp3_44100_128"}, headers=_h(),
                       json={"text": text, "model_id": MODEL}, timeout=60)
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


def voice_for_world(run: str = "r01", world: str = "") -> list[dict]:
    from blab.env import load_env
    load_env()
    if not os.environ.get("ELEVENLABS_API_KEY"):
        print("SKIP voice: missing ELEVENLABS_API_KEY")
        return []
    wd = _world_dir(run, world)
    from blab.events import read_events
    asks = [e for e in read_events(wd / "events.jsonl") if e["type"] == "ask"]
    vid, name = pick_voice()
    out = []
    for e in asks:
        path = wd / "voice" / f"ask_{e['seq']}.mp3"
        if not path.exists():
            tts(e["payload"]["text"], path)
        out.append({"seq": e["seq"], "game": e["game"], "round": e["round"], "text": e["payload"]["text"],
                    "file": str(path.relative_to(wd)) if path.exists() else None})
    (wd / "voice").mkdir(exist_ok=True)
    (wd / "voice" / "manifest.json").write_text(json.dumps({"voice_id": vid, "voice_name": name, "model_id": MODEL,
                                                            "asks": out}, indent=1))
    print(f"[voice] {len(out)} ask lines → {wd / 'voice'} (voice {name})")
    return out
