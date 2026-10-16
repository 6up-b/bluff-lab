"""Claude calls for reflection (SPEC §5.4, §10.4).

Backends (chosen at call time; BLAB_LLM_PROVIDER=bedrock|anthropic|openrouter|auto, auto = that order,
falling through on credential errors such as expired STS tokens, refusals or persistent errors):
  0. AWS Bedrock (`AWS_ACCESS_KEY_ID`…): AsyncAnthropicBedrock, model `us.anthropic.claude-opus-5-5`,
     `messages.parse` structured output (no server-side fallbacks on Bedrock).
  1. Anthropic SDK (`ANTHROPIC_API_KEY` set): `claude-opus-5-5`, `messages.parse` structured output,
     effort low, server-side refusal fallbacks, cached byte-stable system prompt.
  2. OpenRouter (`OPENROUTER_API_KEY` set): OpenAI-compatible chat completions with a strict
     JSON-schema `response_format`, model `anthropic/claude-opus-5.5`.
Output is always validated with Pydantic `ReflectionOut`; predicates with `hypotheses.validate_predicate`
when that module exists. Timeout 60 s; refusal / error → None. At most 8 concurrent calls.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

ANTHROPIC_MODEL = "claude-opus-5-5"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-opus-5.5")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_S = 60.0
MAX_CONCURRENT = int(os.environ.get("BLAB_LLM_CONCURRENCY", "8"))

ASPECTS = ("baseline_bluff", "trigger", "verbal_tell", "pressure", "exploits_trust", "collusion")

_rr = 0
_sem: asyncio.Semaphore | None = None
_sem_loop = None


def _semaphore() -> asyncio.Semaphore:
    global _sem, _sem_loop
    loop = asyncio.get_running_loop()
    if _sem is None or _sem_loop is not loop:
        _sem, _sem_loop = asyncio.Semaphore(MAX_CONCURRENT), loop
    return _sem


# ------------------------------------------------------------------ schema

class HypWire(BaseModel):
    """What the model emits. `predicate` is a JSON string (the DSL is recursive; keep the schema flat)."""
    about: Literal["p2", "p3"]
    aspect: Literal["baseline_bluff", "trigger", "verbal_tell", "pressure", "exploits_trust", "collusion"]
    text: str
    predicate: str = Field(description="JSON-encoded predicate in the DSL")
    rationale: str
    supersedes_id: Optional[str] = None


class ReflectionOut(BaseModel):
    hypotheses: list[HypWire] = Field(max_length=3)


REFLECTION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["hypotheses"],
    "properties": {
        "hypotheses": {
            "type": "array", "maxItems": 3,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["about", "aspect", "text", "predicate", "rationale", "supersedes_id"],
                "properties": {
                    "about": {"type": "string", "enum": ["p2", "p3"]},
                    "aspect": {"type": "string", "enum": list(ASPECTS)},
                    "text": {"type": "string"},
                    "predicate": {"type": "string", "description": "JSON-encoded predicate in the DSL"},
                    "rationale": {"type": "string"},
                    "supersedes_id": {"type": ["string", "null"]},
                },
            },
        }
    },
}


# ------------------------------------------------------------------ frozen system prompt (byte-stable!)

STABLE_SYSTEM = """You are the reflection module of "the Profiler", a research agent that studies opponents in a
3-player bluffing card game. You never play; you read the public record of revealed claims and propose
falsifiable hypotheses about how each opponent bluffs. Every hypothesis you propose is scored automatically
against future revealed claims, so a vague or untestable hypothesis is worthless. This is a game-theory
research simulation; "bluffing" is a legal game move.

## Rules
- Seats p1, p2, p3. p1 is the Profiler (us). Opponents are p2 and p3 ("Player 2", "Player 3").
- Deck: 6 K, 6 Q, 6 A, 2 J (Jokers wild). Each round: shuffle, deal 5 to each living player, draw table_rank from {K,Q,A}.
- On a turn a player either CHALLENGES the previous claim, or PLAYS 1-3 cards face down claiming they are all table_rank,
  with a spoken utterance. Only the next actor may challenge.
- A claim is a bluff iff any placed card is neither table_rank nor J. On challenge the cards are revealed:
  bluff -> claimant loses a life; honest -> challenger loses a life. 3 lives each.
- p_cards = probability (from card counts and the observer's hand only) that the claimant COULD NOT have been
  truthful, i.e. a forced bluff. A bluff with high p_cards was likely forced; strategic bluffing is bluffing beyond
  what card math implies (bluffs at low p_cards). The model is P(bluff) = p_cards + (1 - p_cards) * q, where q is
  the opponent's strategic bluff propensity in that situation.

## Features of a claim (from the observer's point of view)
| feature | type | meaning |
|---|---|---|
| hand_size | int | claimant's cards after this play |
| claim_count | int | cards in this claim (1-3) |
| trusted_streak | int | claimant's consecutive unchallenged claims across games (reset when challenged) |
| lives_self | int | observer's lives |
| lives_target | int | claimant's lives |
| game_index | int | 1-based game number |
| round_index | int | 1-based round number |
| was_challenged_last_round | bool | claimant was challenged in the previous round |
| table_rank | str | K/Q/A |
| utterance | str | claimant's spoken line |
| formal_register | bool | utterance has no contractions and >= 4 words |
Features of an answer to our question may be referenced with the prefix "answer_" (e.g. answer_formal_register).

## Predicate DSL (JSON; evaluated by an interpreter, never executed as code)
- leaf: {"f": <feature>, "op": <op>, "v": <value>} with op in == != < <= > >= contains regex no_contractions
  (contains/regex/no_contractions apply to string features; regex patterns <= 100 chars)
- combinators: {"all": [<pred>, ...]}, {"any": [<pred>, ...]}, {"not": <pred>}
- Example: {"all": [{"f":"formal_register","op":"==","v":true}, {"f":"hand_size","op":"<=","v":2}]}
The predicate must describe WHEN the claim is more likely to be a bluff (the "true" branch should have a higher
bluff rate than the "false" branch).

## Aspects (closed set, one per hypothesis)
baseline_bluff (overall bluff propensity), trigger (game-state condition, e.g. low hand size or lives),
verbal_tell (something in the utterance), pressure (behaviour after being challenged / when low on lives),
exploits_trust (bluffs more after building a trusted streak; long-con / defection), collusion (coordination between p2 and p3).

## Task
You receive the revealed observations since the last reflection and the current hypotheses with their scoring
statistics (n_true, n_false, q_true, q_false = posterior [mean, lo, hi] of strategic bluff propensity when the
predicate is true/false; status provisional/confirmed/refuted). Propose at most 3 NEW or REVISED hypotheses that
are most likely to be confirmed by future evidence. Do not repeat a hypothesis that already exists with the same
predicate. If a drift / baseline shift is reported, consider an exploits_trust hypothesis. To replace an existing
hypothesis set supersedes_id to its id, else null.

## Output schema
Return JSON: {"hypotheses": [{"about": "p2"|"p3", "aspect": <aspect>, "text": <one plain-English sentence,
e.g. "Player 3 speaks formally (no contractions) when bluffing">, "predicate": <the predicate as a JSON-encoded
string>, "rationale": <one sentence citing the evidence obs ids>, "supersedes_id": <id or null>}]}
Return an empty list if the evidence supports nothing new.
"""


def user_message(payload: dict) -> str:
    """Volatile content (goes in messages, after the cached system prompt)."""
    obs = payload.get("new_obs") or []
    obs = obs[-40:]
    slim = []
    for o in obs:
        if isinstance(o, str):
            slim.append({"obs_id": o})
            continue
        f = dict(o.get("features") or {})
        f.pop("card_math", None)
        slim.append({"obs_id": o.get("obs_id"), "game": o.get("game"), "round": o.get("round"), "by": o.get("by"),
                     "count": o.get("count"), "utterance": o.get("utterance"), "was_bluff": o.get("was_bluff"),
                     "p_cards": o.get("p_cards"), "features": f})
    hyps = []
    for h in payload.get("hypotheses") or []:
        hyps.append({k: h.get(k) for k in ("id", "about", "aspect", "text", "predicate", "status", "provenance",
                                             "n_true", "n_false", "q_true", "q_false", "valid_until_game")
                     if k in h})
    parts = {"seat": payload.get("seat", "p1"), "memory_version": payload.get("memory_version"),
             "game": payload.get("game"), "revealed_observations": slim, "current_hypotheses": hyps}
    for k in ("drift", "notes", "base"):
        if payload.get(k):
            parts[k] = payload[k]
    return ("Reflect on this record and propose hypotheses.\n\n```json\n"
            + json.dumps(parts, separators=(",", ":"), default=str) + "\n```")


# ------------------------------------------------------------------ validation

# Raw-information variant (info: "raw"): no engineered tell feature, no tell-shaped op, and neutral examples, so the
# prompt cannot hint at the formal-speech tell. Built from STABLE_SYSTEM so both stay byte-stable.
_RAW_REPL = [
    ("| formal_register | bool | utterance has no contractions and >= 4 words |\n",
     "| word_count | int | number of whitespace-separated words in the utterance |\n"),
    ('(e.g. answer_formal_register)', '(e.g. answer_utterance, answer_word_count)'),
    ("== != < <= > >= contains regex no_contractions", "== != < <= > >= contains regex"),
    ("(contains/regex/no_contractions apply to string features", "(contains/regex apply to string features"),
    ('{"all": [{"f":"formal_register","op":"==","v":true}, {"f":"hand_size","op":"<=","v":2}]}',
     '{"all": [{"f":"hand_size","op":"<=","v":2}, {"f":"claim_count","op":">=","v":2}]}'),
    ('e.g. "Player 3 speaks formally (no contractions) when bluffing"', 'e.g. "Player 3 bluffs more on multi-card claims"'),
]
STABLE_SYSTEM_RAW = STABLE_SYSTEM
for _a, _b in _RAW_REPL:
    assert _a in STABLE_SYSTEM_RAW, _a
    STABLE_SYSTEM_RAW = STABLE_SYSTEM_RAW.replace(_a, _b)
assert "formal" not in STABLE_SYSTEM_RAW.lower() and "contraction" not in STABLE_SYSTEM_RAW.lower()


def _validate_pred(pred, mode: str | None = None) -> tuple[bool, str]:
    try:
        from blab.hypotheses import validate_predicate
    except Exception:  # noqa: BLE001  (module not ready yet)
        return (isinstance(pred, dict) and bool(pred)), "no validator"
    try:
        return validate_predicate(pred, mode) if mode else validate_predicate(pred)
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def to_proposed(out: ReflectionOut, mode: str | None = None) -> tuple[list[dict], list[dict]]:
    """Wire → §5.4 proposed dicts. Returns (proposed, dropped). mode="raw" rejects engineered features/ops."""
    proposed, dropped = [], []
    for h in out.hypotheses[:3]:
        try:
            pred = json.loads(h.predicate) if isinstance(h.predicate, str) else h.predicate
        except json.JSONDecodeError as e:
            dropped.append({"text": h.text, "reason": f"predicate not JSON: {e}"})
            continue
        ok, why = _validate_pred(pred, mode)
        if not ok:
            dropped.append({"text": h.text, "predicate": pred, "reason": why})
            continue
        proposed.append({"about": h.about, "aspect": h.aspect, "text": h.text, "predicate": pred,
                         "rationale": h.rationale, "supersedes_id": h.supersedes_id})
    return proposed, dropped


# ------------------------------------------------------------------ backends

BEDROCK_MODEL = os.environ.get("BLAB_BEDROCK_MODEL", "us.anthropic.claude-opus-5-5")


def _bedrock_api_key() -> str | None:
    """Long-lived Bedrock API keys (bearer tokens) per model family; preferred over expiring STS creds."""
    fam = "SONNET" if "sonnet" in BEDROCK_MODEL else "OPUS"
    return os.environ.get(f"AWS_{fam}_API_KEY") or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")


def _have(p: str) -> bool:
    if p == "bedrock":
        return bool(_bedrock_api_key() or os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE"))
    if p == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if p == "openrouter":
        return bool(os.environ.get("OPENROUTER_API_KEY"))
    return False


def providers() -> list[str]:
    """BLAB_LLM_PROVIDER = bedrock|anthropic|openrouter|auto (default auto = bedrock → anthropic → openrouter).
    A named provider still falls through to the others if it fails (e.g. expired STS creds)."""
    order = ["bedrock", "anthropic", "openrouter"]
    pref = os.environ.get("BLAB_LLM_PROVIDER", "auto").strip().lower()
    if pref in order:
        order = [pref] + [o for o in order if o != pref]
    return [p for p in order if _have(p)]


def backend() -> str | None:
    ps = providers()
    return ps[0] if ps else None


def available() -> bool:
    return backend() is not None


_clients: dict = {}
_dead: set[str] = set()          # providers whose credentials failed (expired/denied) this process


def _client(p: str):
    if p not in _clients:
        import anthropic
        if p == "bedrock":
            region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-west-2"
            key = _bedrock_api_key()
            _clients[p] = (anthropic.AsyncAnthropicBedrock(api_key=key, aws_region=region, timeout=TIMEOUT_S, max_retries=1)
                           if key else
                           anthropic.AsyncAnthropicBedrock(aws_region=region, timeout=TIMEOUT_S, max_retries=1))
        else:
            _clients[p] = anthropic.AsyncAnthropic(timeout=TIMEOUT_S, max_retries=2)
    return _clients[p]


def _is_cred_error(e: Exception) -> bool:
    t = f"{type(e).__name__} {e}"
    return any(k in t for k in ("ExpiredToken", "UnrecognizedClient", "InvalidClientTokenId", "AuthenticationError",
                                "PermissionDenied", "security token", "NoCredentials", "AccessDenied", "403", "401"))


async def _call_claude(p: str, msg: str, system: str | None = None) -> tuple[ReflectionOut | None, str | None, str | None]:
    """Anthropic SDK path for both `anthropic` (direct; server-side refusal fallbacks on) and `bedrock`
    (structured outputs supported; no server-side fallbacks there). Returns (parsed, request_id, error)."""
    import anthropic
    client = _client(p)
    common = dict(model=BEDROCK_MODEL if p == "bedrock" else ANTHROPIC_MODEL, max_tokens=16000,
                  system=[{"type": "text", "text": system or STABLE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                  messages=[{"role": "user", "content": msg}],
                  output_format=ReflectionOut, output_config={"effort": "low"})
    try:
        if p == "anthropic":
            try:
                resp = await client.beta.messages.parse(betas=["server-side-fallback-2026-07-01"],
                                                        fallbacks="default", **common)
            except (TypeError, anthropic.BadRequestError):
                resp = await client.messages.parse(**common)
        else:
            resp = await client.messages.parse(**common)
    except anthropic.RateLimitError as e:
        ra = float(e.response.headers.get("retry-after", "5") or 5)
        await asyncio.sleep(min(ra, 20))
        try:
            resp = await client.messages.parse(**common)
        except Exception as e2:  # noqa: BLE001
            return None, None, f"{type(e2).__name__}: {e2}"
    except Exception as e:  # noqa: BLE001
        if _is_cred_error(e):
            _dead.add(p)
        return None, None, f"{type(e).__name__}: {str(e)[:200]}"
    rid = getattr(resp, "_request_id", None) or getattr(resp, "id", None)
    if resp.stop_reason == "refusal":
        return None, rid, "refusal"
    parsed = getattr(resp, "parsed_output", None)
    if parsed is None:
        return None, rid, f"no parsed_output (stop_reason={resp.stop_reason})"
    if not isinstance(parsed, ReflectionOut):
        parsed = ReflectionOut.model_validate(parsed if isinstance(parsed, dict) else parsed.model_dump())
    return parsed, rid, None


async def _call_openrouter(msg: str, system: str | None = None) -> tuple[ReflectionOut | None, str | None, str | None]:
    import httpx
    body = {
        "model": OPENROUTER_MODEL,
        "max_tokens": 8000,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system or STABLE_SYSTEM,
                                            "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": msg},
        ],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "ReflectionOut", "strict": True,
                                            "schema": REFLECTION_JSON_SCHEMA}},
        "reasoning": {"effort": "low"},
    }
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
               "HTTP-Referer": "https://github.com/blufflab", "X-Title": "Bluff Lab"}
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as c:
        for attempt in range(2):
            try:
                r = await c.post(OPENROUTER_URL, json=body, headers=headers)
            except Exception as e:  # noqa: BLE001
                return None, None, f"{type(e).__name__}: {e}"
            if r.status_code == 429 and attempt == 0:
                await asyncio.sleep(min(float(r.headers.get("retry-after", "5") or 5), 20))
                continue
            break
    if r.status_code != 200:
        return None, None, f"openrouter HTTP {r.status_code}: {r.text[:200]}"
    data = r.json()
    rid = data.get("id")
    try:
        choice = data["choices"][0]
        if choice.get("finish_reason") in ("content_filter",) or choice.get("native_finish_reason") == "refusal":
            return None, rid, "refusal"
        content = choice["message"].get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            content = content[content.find("{"):]
        return ReflectionOut.model_validate_json(content[: content.rfind("}") + 1]), rid, None
    except (KeyError, IndexError, ValidationError, ValueError) as e:
        return None, rid, f"bad output: {type(e).__name__}: {str(e)[:150]}"


async def reflect(payload: dict, wid: str = "", seat: str | None = None, only: str | None = None) -> dict | None:
    """One reflection call. Returns {"proposed", "dropped", "llm_trace_id", "backend", "latency_ms"} or
    {"proposed": None, "error": ...}; returns None when no LLM backend is configured."""
    chain = [p for p in providers() if only in (None, p)]
    if only is None and os.environ.get("BLAB_LLM_SPREAD") and len(chain) > 1:
        # spread load across the Claude providers (bedrock ⇄ anthropic), keep the rest as fallbacks
        global _rr
        claude = [p for p in chain if p in ("bedrock", "anthropic")]
        if len(claude) == 2:
            _rr += 1
            first = claude[_rr % 2]
            chain = [first] + [p for p in chain if p != first]
    be = chain[0] if chain else None
    if be is None:
        return None
    msg = user_message(payload)
    raw = payload.get("info") == "raw"
    sysp = STABLE_SYSTEM_RAW if raw else None      # None → STABLE_SYSTEM looked up at call time
    t0 = time.time()
    try:
        from blab.orchestrator import tracing
        span_cm = tracing.span("reflection", kind="LLM", session_id=wid,
                               attrs={"blab.seat": seat or payload.get("seat", "p1"), "blab.backend": be,
                                      "llm.model_name": {"anthropic": ANTHROPIC_MODEL, "bedrock": BEDROCK_MODEL}.get(be, OPENROUTER_MODEL),
                                      "input.value": msg[:8000]})
    except Exception:  # noqa: BLE001
        from contextlib import nullcontext
        span_cm = nullcontext(None)
    async with _semaphore():
        with span_cm as sp:
            parsed, rid, err, used = None, None, "no provider", be
            for p in [p for p in chain if p not in _dead] or chain[:1]:
                used = p
                call = ((lambda m: _call_openrouter(m, sysp)) if p == "openrouter"
                        else (lambda m, _p=p: _call_claude(_p, m, sysp)))
                for attempt in range(2):
                    remaining = TIMEOUT_S - (time.time() - t0)
                    if remaining < 5:
                        err = err or "timeout"
                        break
                    try:
                        parsed, rid, err = await asyncio.wait_for(call(msg), remaining)
                    except asyncio.TimeoutError:
                        parsed, rid, err = None, None, "timeout"
                    if parsed is not None or err in ("refusal", "timeout") or p in _dead:
                        break
                    if time.time() - t0 > 20:
                        break
                if parsed is not None or err == "timeout":
                    break
                # refusal / credential failure / persistent error → next provider
            be = used
            out: dict = {"backend": be, "llm_trace_id": rid, "latency_ms": int((time.time() - t0) * 1000)}
            if parsed is None:
                out.update(proposed=None, error=err)
            else:
                proposed, dropped = to_proposed(parsed, "raw" if raw else None)
                out.update(proposed=proposed, dropped=dropped, prompt="raw" if raw else "engineered")
            try:
                if sp is not None:
                    sp.set_attribute("output.value", json.dumps(out, default=str)[:8000])
                    tid = tracing.trace_id_of(sp)
                    if tid:
                        out["arize_trace_id"] = tid
            except Exception:  # noqa: BLE001
                pass
    return out
