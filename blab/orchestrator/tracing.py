"""Arize AX tracing (SPEC §10.3). A no-op unless ARIZE_SPACE_ID and ARIZE_API_KEY are set.

    from blab.orchestrator import tracing
    with tracing.span("reflection", kind="LLM", session_id=wid, attrs={...}) as sp:
        ...
        tracing.trace_id_of(sp)   # hex trace id, or None
    tracing.flush()

Also used inside boxes for `decision` spans (worldd) — same env vars, otlp.arize.com is reachable from boxes.
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from typing import Any

_lock = threading.Lock()
_tp = None
_tracer = None
_failed = False


def enabled() -> bool:
    return bool(os.environ.get("ARIZE_SPACE_ID") and os.environ.get("ARIZE_API_KEY")) and not _failed


def _init():
    global _tp, _tracer, _failed
    if _tracer is not None or not enabled():
        return _tracer
    with _lock:
        if _tracer is not None:
            return _tracer
        try:
            from arize.otel import register
            from openinference.instrumentation import OITracer, TraceConfig
            _tp = register(space_id=os.environ["ARIZE_SPACE_ID"], api_key=os.environ["ARIZE_API_KEY"],
                           project_name=os.environ.get("ARIZE_PROJECT_NAME", "bluff-lab"), verbose=False)
            try:
                from openinference.instrumentation.anthropic import AnthropicInstrumentor
                AnthropicInstrumentor().instrument(tracer_provider=_tp)
            except Exception:  # noqa: BLE001
                pass
            _tracer = OITracer(_tp.get_tracer("blab"), config=TraceConfig())
        except Exception as e:  # noqa: BLE001
            _failed = True
            print(f"[tracing] disabled: {type(e).__name__}: {e}")
    return _tracer


def _set(sp, k: str, v: Any) -> None:
    if v is None:
        return
    if not isinstance(v, (str, bool, int, float)):
        v = json.dumps(v, default=str)[:8000]
    try:
        sp.set_attribute(k, v)
    except Exception:  # noqa: BLE001
        pass


@contextmanager
def span(name: str, *, kind: str = "CHAIN", session_id: str | None = None, attrs: dict | None = None):
    tr = _init()
    if tr is None:
        yield None
        return
    with tr.start_as_current_span(name) as sp:
        _set(sp, "openinference.span.kind", kind)
        if session_id:
            _set(sp, "session.id", session_id)
        for k, v in (attrs or {}).items():
            _set(sp, k, v)
        yield sp


def trace_id_of(sp) -> str | None:
    if sp is None:
        return None
    try:
        return format(sp.get_span_context().trace_id, "032x")
    except Exception:  # noqa: BLE001
        return None


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and len(prefix) < 40:
            out.update(flat(v, key + "."))
        elif isinstance(v, (str, bool, int, float)):
            out[key] = v
        elif v is not None:
            out[key] = json.dumps(v, default=str)[:2000]
    return out


def decision_span(wid: str, decision: dict, seq: int | None = None) -> str | None:
    """One `decision` span (box side or replayed from events.jsonl on the laptop)."""
    if not enabled():
        return None
    d = {k: v for k, v in decision.items() if k not in ("obs",)}
    with span("decision", kind="CHAIN", session_id=wid,
              attrs={"input.value": d.get("claim"), "output.value": d.get("action"),
                     **{f"blab.{k}": v for k, v in flat(d).items()}, "blab.seq": seq}) as sp:
        return trace_id_of(sp)


def flush() -> None:
    if _tp is not None:
        try:
            _tp.force_flush()
        except Exception:  # noqa: BLE001
            pass


def spike_span() -> str | None:
    with span("spike.laptop", kind="CHAIN", session_id="spike",
              attrs={"input.value": "spike A1", "output.value": "ok", "blab.check": "A1"}) as sp:
        tid = trace_id_of(sp)
    flush()
    return tid
