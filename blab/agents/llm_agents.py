"""Per-decision LLM arms for the memory experiment (mem01; owner: llm-arms).

Three arms that differ ONLY in the memory block of the user message (same model, same byte-stable system
prompt, same rendering of the current observation, same respond/play rules as ours.py):

- `llm_nomem`    — no memory: current claim + card math + this game's revealed claims.
- `llm_episodic` — + the last 30 revealed claims about the claimant from PREVIOUS games (raw transcript).
- `llm_dossier`  — + the dossier of hypotheses about the claimant (text, predicate, n_true/n_false,
                   q_true/q_false, status, provenance), maintained between games by the profiler's own
                   reflection + scoring machinery (inherits ours.Profiler; nothing reimplemented).

Claude returns {p_bluff, reason} via structured output; p_final = p_bluff and the agent applies the SAME
respond rule (threshold|ev, ε-exploration with the recorded rng_u) as every other agent. `reason` is the
model's own PRE-decision rationale recorded at decision time (not an after-the-fact explanation).

Memory timing rule: memory blocks are rendered from the checkpoint memory (`self.memory`, changed only at
game end) plus in-game entries of obs.revealed_history, so a decision is a function of (obs, memory, rng_u,
model response). Replay: every call is cached by sha256(system + user prompt) in `params["llm_cache"]`
(a jsonl file in the world dir); a replay with the same prompt (variant `full`) reuses the recorded response
and is exact; a memory variant changes the prompt and makes a fresh call.

LLM calls run on one background asyncio loop (≤ BLAB_LLM_ARMS_CONCURRENCY = 16 concurrent across the
process, decisions AND reflections routed through `guarded_reflector`), so `act()` (sync, as World expects)
blocks on a future. Timeout 30 s → p_bluff = p_cards and `llm.error` is recorded.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
from pathlib import Path

from pydantic import BaseModel, Field

from blab.agents.ours import AGENTS, Bayes, Counter, Profiler
from blab.agents.common import _claim_line, public_obs_text

SONNET_BEDROCK = os.environ.get("BLAB_ARMS_BEDROCK_MODEL", "us.anthropic.claude-sonnet-5-5")
SONNET_DIRECT = os.environ.get("BLAB_ARMS_MODEL", "claude-sonnet-5-5")
CALL_TIMEOUT_S = 30.0
MAX_CONCURRENT = int(os.environ.get("BLAB_LLM_ARMS_CONCURRENCY", "16"))
EPISODIC_N = 30
DOSSIER_MAX = 12


class BluffEstimate(BaseModel):
    p_bluff: float = Field(description="probability in [0,1] that the current claim is a bluff")
    reason: str = Field(description="one or two sentences: the rationale for this estimate")


# ------------------------------------------------------------------ frozen system prompt (byte-stable!)

SYSTEM = """You are Player 1 in a 3-player bluffing card game (a game-theory research simulation; bluffing is a
legal move). Your only job on this turn: estimate the probability that the claim just made to you is a bluff.

## Rules
- Seats p1 (you), p2 ("Player 2"), p3 ("Player 3"). Deck: 6 K, 6 Q, 6 A, 2 J (Jokers are wild).
- Each round: shuffle, deal 5 cards to each living player, draw a table rank from {K, Q, A}.
- On a turn a player either CHALLENGES the previous claim or PLAYS 1-3 cards face down, claiming they are all
  the table rank, and says something out loud. Only the next player may challenge.
- A claim is a bluff iff any played card is neither the table rank nor a Joker. On a challenge the cards are
  revealed: bluff -> the claimant loses a life; honest -> the challenger loses a life. 3 lives each; the last
  player alive wins the game. A world is a series of games against the same two opponents.

## Card math (computed from public counts and your own hand only)
U = unseen cards (not in your hand), m = unseen cards matching the table rank (incl. Jokers),
h = cards the claimant held before playing, k = cards claimed. p_cards = probability that the claimant could
NOT have had k matching cards, i.e. that the claim is a forced bluff. The true bluff probability is at least
p_cards; opponents may also bluff strategically when they could have told the truth. Opponents can differ
from each other and may change behaviour over time. What they say (the exact utterance) is public.

## Your input
The current claim (claimant, count, exact utterance), the card math, hand sizes and lives, and the claims
revealed by challenges earlier in this game. A "Memory" section may follow with information from earlier
games against this claimant; it may be empty. Use whatever is informative; ignore what is not.

## Output
Return JSON {"p_bluff": <number in [0,1]>, "reason": <one or two sentences>}. Give your honest calibrated
probability; your decision to challenge is made from it by a fixed rule.
"""


def _pn(seat: str) -> str:
    return seat


def render_observation(obs: dict) -> str:
    """core's byte-stable public_obs_text (blab.agents.common) over the CURRENT game's revealed claims only
    (obs.revealed_history covers all games; earlier games are memory, which differs by arm), plus the full
    card-math line. The obs a raw-mode seat receives is already stripped by World (info: raw)."""
    cur = dict(obs)
    cur["revealed_history"] = [e for e in obs.get("revealed_history", []) if e.get("game") == obs.get("game")]
    cm = ((obs.get("pending_claim") or {}).get("features") or {}).get("card_math") or {}
    return ("## Current state\n" + public_obs_text(cur) +
            f"\nCard math for the pending claim: U={cm.get('U')} m={cm.get('m')} h={cm.get('h')} k={cm.get('k')} "
            f"p_cards={cm.get('p_cards')}")


def prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256((system + "\x00" + user).encode()).hexdigest()


# ------------------------------------------------------------------ background loop + calls

_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_lock = threading.Lock()
_sem: asyncio.Semaphore | None = None
_clients: dict = {}
_dead: set[str] = set()
STATS = {"calls": 0, "errors": 0, "cache_hits": 0, "latency_ms_sum": 0, "by_provider": {}}
_stats_lock = threading.Lock()


def bg_loop() -> asyncio.AbstractEventLoop:
    global _bg_loop, _sem
    with _bg_lock:
        if _bg_loop is None:
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, daemon=True, name="llm-arms-loop").start()
            _sem = asyncio.Semaphore(MAX_CONCURRENT)
            _bg_loop = loop
    return _bg_loop


def _providers() -> list[str]:
    ps = []
    if os.environ.get("AWS_SONNET_API_KEY"):
        ps.append("bedrock")
    if os.environ.get("ANTHROPIC_API_KEY"):
        ps.append("anthropic")
    pref = os.environ.get("BLAB_ARMS_PROVIDER", "").strip().lower()
    if pref in ps:
        ps = [pref] + [p for p in ps if p != pref]
    return [p for p in ps if p not in _dead] or ps


def _client(p: str):
    if p not in _clients:
        import anthropic
        if p == "bedrock":
            region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-west-2"
            _clients[p] = anthropic.AsyncAnthropicBedrock(api_key=os.environ["AWS_SONNET_API_KEY"], aws_region=region,
                                                          timeout=CALL_TIMEOUT_S, max_retries=0)
        else:
            _clients[p] = anthropic.AsyncAnthropic(timeout=CALL_TIMEOUT_S, max_retries=0)
    return _clients[p]


async def _call_once(p: str, user: str):
    client = _client(p)
    common = dict(model=SONNET_BEDROCK if p == "bedrock" else SONNET_DIRECT, max_tokens=4000,
                  system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                  messages=[{"role": "user", "content": user}],
                  output_format=BluffEstimate, output_config={"effort": "low"})
    if p == "anthropic":
        import anthropic
        try:
            return await client.beta.messages.parse(betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                                    **common)
        except (TypeError, anthropic.BadRequestError):
            return await client.messages.parse(**common)
    return await client.messages.parse(**common)


def _retryable(e: Exception) -> bool:
    import anthropic
    if isinstance(e, (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError,
                      anthropic.InternalServerError)):
        return True
    return isinstance(e, anthropic.APIStatusError) and e.status_code >= 500


def _cred_error(e: Exception) -> bool:
    import anthropic
    return isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError))


async def _estimate(user: str) -> dict:
    """{p_bluff, reason, provider, model, request_id, latency_ms} or {error, ...}. Total budget 30 s."""
    t0 = time.monotonic()
    err, provider = "no provider (AWS_SONNET_API_KEY / ANTHROPIC_API_KEY unset)", None
    async with _sem:
        t_start = time.monotonic()
        for p in _providers():
            provider = p
            delay = 1.0
            while True:
                remaining = CALL_TIMEOUT_S - (time.monotonic() - t_start)
                if remaining < 2:
                    err = err if err.startswith("timeout") else f"timeout ({err})"
                    break
                try:
                    resp = await asyncio.wait_for(_call_once(p, user), remaining)
                except asyncio.TimeoutError:
                    err = "timeout after 30s"
                    break
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {str(e)[:160]}"
                    if _cred_error(e):
                        _dead.add(p)
                        break
                    if _retryable(e):
                        await asyncio.sleep(min(delay, max(0.0, CALL_TIMEOUT_S - (time.monotonic() - t_start) - 2)))
                        delay *= 2
                        continue
                    break
                rid = getattr(resp, "_request_id", None) or getattr(resp, "id", None)
                if resp.stop_reason == "refusal":
                    err = "refusal"
                    break
                parsed = getattr(resp, "parsed_output", None)
                if parsed is None:
                    err = f"no parsed_output (stop_reason={resp.stop_reason})"
                    break
                pb = float(parsed.p_bluff)
                return {"p_bluff": min(1.0, max(0.0, pb)), "reason": parsed.reason[:600], "provider": p,
                        "model": getattr(resp, "model", None) or (SONNET_BEDROCK if p == "bedrock" else SONNET_DIRECT),
                        "request_id": rid, "latency_ms": int((time.monotonic() - t0) * 1000),
                        "queue_ms": int((t_start - t0) * 1000),
                        "usage": {"in": getattr(resp.usage, "input_tokens", None),
                                  "out": getattr(resp.usage, "output_tokens", None),
                                  "cache_read": getattr(resp.usage, "cache_read_input_tokens", None)}}
            if err.startswith("timeout"):
                break
    return {"error": err, "provider": provider, "latency_ms": int((time.monotonic() - t0) * 1000)}


def estimate_sync(user: str) -> dict:
    fut = asyncio.run_coroutine_threadsafe(_estimate(user), bg_loop())
    try:
        return fut.result(timeout=CALL_TIMEOUT_S + 600)      # queueing time is not part of the 30 s budget
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "provider": None, "latency_ms": None}


async def guarded_reflector(wid: str, seat: str, payload: dict) -> dict | None:
    """The profiler's reflector (blab.orchestrator.reflect.local_reflector: SurrealDB pre-step + Claude, same
    prompt/schema/validation) run on the shared background loop under the same ≤16-call semaphore, so
    World(reflector=...) works from any loop/thread. None → World keeps the old memory and logs the error."""
    from blab.orchestrator.reflect import local_reflector

    async def _run():
        async with _sem:
            return await local_reflector(wid, seat, payload)
    try:
        out = await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(_run(), bg_loop()))
    except Exception as e:  # noqa: BLE001
        out = None
        print(f"[llm-arms] reflection {wid}/{seat} failed: {type(e).__name__}: {e}")
    with _stats_lock:
        STATS["reflections"] = STATS.get("reflections", 0) + 1
        if out is None:
            STATS["reflection_errors"] = STATS.get("reflection_errors", 0) + 1
    return out


# ------------------------------------------------------------------ replay cache (by prompt hash)

_cache_mem: dict[str, dict[str, dict]] = {}
_cache_lock = threading.Lock()


def _cache_load(path: str) -> dict[str, dict]:
    with _cache_lock:
        if path not in _cache_mem:
            d: dict[str, dict] = {}
            p = Path(path)
            if p.exists():
                for line in p.read_text().splitlines():
                    try:
                        r = json.loads(line)
                        d[r["prompt_hash"]] = r
                    except Exception:  # noqa: BLE001
                        pass
            _cache_mem[path] = d
        return _cache_mem[path]


def _cache_put(path: str, rec: dict) -> None:
    d = _cache_load(path)
    with _cache_lock:
        d[rec["prompt_hash"]] = rec
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def cache_reset(path: str | None = None) -> None:
    with _cache_lock:
        if path is None:
            _cache_mem.clear()
        else:
            _cache_mem.pop(path, None)


# ------------------------------------------------------------------ arms

class _LLMMixin:
    memory_kind = "none"

    def memory_block(self, obs: dict, about: str) -> tuple[str, list[dict]]:
        return "(no memory: this arm sees only the current game)", []

    def predict(self, obs: dict, about: str, features: dict, p_c: float):
        forced = "play" not in obs.get("legal_actions", [])
        mem_text, mem_used = self.memory_block(obs, about)
        user = render_observation(obs) + f"\n\n## Memory about {_pn(about)} from earlier games\n" + mem_text
        h = prompt_hash(SYSTEM, user)
        llm = {"model": None, "provider": None, "request_id": None, "latency_ms": None, "reason": None,
               "reason_kind": "model's own pre-decision rationale, recorded at decision time",
               "prompt_hash": h, "memory_kind": self.memory_kind, "cached": False, "error": None,
               "memory_obs": list(getattr(self, "_episodic_obs", []) or []) if self.memory_kind == "episodic" else None}
        if forced:
            # no choice to make (must challenge); identical across arms, no call
            llm.update(skipped="forced")
            return None, "llm", mem_used, float(p_c), {"llm": llm}
        cache_path = self.params.get("llm_cache")
        rec = _cache_load(cache_path).get(h) if cache_path else None
        if rec is not None:
            r = dict(rec["response"])
            llm.update({k: r.get(k) for k in ("model", "provider", "request_id", "latency_ms", "reason")}, cached=True)
            with _stats_lock:
                STATS["cache_hits"] += 1
            return None, "llm", mem_used, float(r["p_bluff"]), {"llm": llm}
        r = estimate_sync(user)
        with _stats_lock:
            STATS["calls"] += 1
            if r.get("error"):
                STATS["errors"] += 1
            else:
                STATS["latency_ms_sum"] += int(r.get("latency_ms") or 0)
                bp = STATS["by_provider"]
                bp[r["provider"]] = bp.get(r["provider"], 0) + 1
        if r.get("error"):
            llm.update(error=r["error"], provider=r.get("provider"), latency_ms=r.get("latency_ms"),
                       fallback="p_bluff = p_cards")
            return None, "llm", mem_used, float(p_c), {"llm": llm}
        llm.update({k: r.get(k) for k in ("model", "provider", "request_id", "latency_ms", "reason", "usage",
                                          "queue_ms")})
        if cache_path:
            _cache_put(cache_path, {"prompt_hash": h, "response": {k: r.get(k) for k in (
                "p_bluff", "reason", "model", "provider", "request_id", "latency_ms")}, "user": user})
        return None, "llm", mem_used, float(r["p_bluff"]), {"llm": llm}


class LLMNoMem(_LLMMixin, Counter):
    """No memory baseline. (Counter: no checkpoint memory; respond/play rules shared with ours.py.)"""
    id = "llm_nomem"
    memory_kind = "none"


class LLMEpisodic(_LLMMixin, Bayes):
    """Raw transcript memory: the last 30 revealed claims about the claimant from previous games.
    (Bayes supplies the memory plumbing: `memory.revealed` is appended at game end only.)"""
    id = "llm_episodic"
    memory_kind = "episodic"

    def memory_block(self, obs, about):
        past = [e for e in self.memory.get("revealed", []) if e.get("by") == about
                and e.get("game") != obs.get("game")][-EPISODIC_N:]
        if not past:
            self._episodic_obs = []
            return "(no revealed claims by this player in earlier games yet)", []
        self._episodic_obs = [e.get("obs_id") for e in past]
        return (f"Last {len(past)} revealed claims by {_pn(about)} in earlier games (oldest first):\n"
                + "\n".join(_claim_line(e) for e in past)), []


def _fmt_q(q) -> str:
    if not q:
        return "n/a"
    try:
        return f"{q[0]:.2f} [{q[1]:.2f}, {q[2]:.2f}]"
    except Exception:  # noqa: BLE001
        return str(q)


class LLMDossier(_LLMMixin, Profiler):
    """Structured memory: the profiler's hypothesis dossier (Claude reflection every 2 games + automatic
    scoring + supersession/drift, all inherited from ours.Profiler) rendered into the prompt."""
    id = "llm_dossier"
    memory_kind = "dossier"

    def memory_block(self, obs, about):
        hs = [h for h in self.memory.get("hypotheses", []) if h.get("about") == about
              and h.get("valid_until_game") is None]
        order = {"confirmed": 0, "provisional": 1, "refuted": 2}
        hs = sorted(hs, key=lambda h: (order.get(h.get("status"), 3), -(h.get("n_true") or 0)))[:DOSSIER_MAX]
        if not hs:
            return "(dossier empty: no hypotheses about this player yet)", []
        lines, used = [], []
        for h in hs:
            lines.append(f"- [{h.get('id')}] {h.get('text')} | aspect {h.get('aspect')} | predicate "
                         f"{json.dumps(h.get('predicate'), separators=(',', ':'))} | status {h.get('status')} | "
                         f"n_true {h.get('n_true')} n_false {h.get('n_false')} | q_true {_fmt_q(h.get('q_true'))} "
                         f"q_false {_fmt_q(h.get('q_false'))} | provenance {h.get('provenance')}")
            used.append({"hyp_id": h.get("id"), "aspect": h.get("aspect"), "text": h.get("text") or "",
                         "status": h.get("status") or "provisional", "provenance": h.get("provenance") or "evidence",
                         "n_true": int(h.get("n_true") or 0), "n_false": int(h.get("n_false") or 0),
                         "q_true": list(h.get("q_true") or [0.5, 0.0, 1.0]),
                         "q_false": list(h.get("q_false") or [0.5, 0.0, 1.0]),
                         "evidence_obs": list(h.get("evidence_obs") or [])[-20:],
                         "predicate": h.get("predicate"), "used_as": "shown"})
        head = ("Dossier of hypotheses about this player (q = strategic bluff propensity [mean, 95% interval] "
                "when the predicate is true / false, scored automatically on later revealed claims):\n")
        return head + "\n".join(lines), used

    def want_ask(self, obs):          # interrogation is out of scope for mem01
        return None


ARMS = {c.id: c for c in (LLMNoMem, LLMEpisodic, LLMDossier)}
AGENTS.update(ARMS)
