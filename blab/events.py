"""Event log — THE CONTRACT between the Lab and the Show (SPEC §4). FROZEN at Milestone 1.

Every line of events.jsonl is one `Event`. The log is a god view (hidden cards included);
the UI is responsible for hiding them. Payload models are permissive (`extra="allow"`) so
producers may attach additional fields, but the listed fields are required and must keep
their names and meaning. Do not rename or remove fields after M1 — only add optional ones.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

EventType = Literal[
    "world_start", "game_start", "deal", "ask", "answer", "claim", "decision",
    "challenge", "play_on", "reveal", "life_lost", "game_end",
    "reflect_request", "reflect_result", "hypothesis", "belief_superseded",
    "surgery", "forge_result", "world_end",
]

Seat = Literal["p1", "p2", "p3"]
Rank = Literal["K", "Q", "A", "J"]


class _P(BaseModel):
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------- payloads

class SeatSpec(_P):
    seat: Seat
    policy: str                      # e.g. "profiler", "scripted"
    role: Optional[str] = None       # sleeper|telltale|telltale_overt|control|colluder|None (our agents)
    params: dict[str, Any] = Field(default_factory=dict)


class WorldStart(_P):
    seats: list[SeatSpec]
    seed: int
    games: int
    box: str
    snapshot_id: Optional[str] = None
    variant: Optional[dict[str, Any]] = None     # memory_variant for branch worlds
    parent_world: Optional[str] = None
    from_game: Optional[int] = None
    learning: bool = True
    reflect_every: Optional[int] = None


class GameStart(_P):
    seat_order: list[Seat]
    lives: dict[str, int]


class Deal(_P):
    table_rank: Literal["K", "Q", "A"]
    hands: dict[str, list[str]]                  # god view


class Ask(_P):
    target: Seat
    template_id: str
    text: str
    p_final: float
    reason_aspect: Optional[str] = None
    audio: Optional[str] = None


class Answer(_P):
    text: str
    latency_ms: int
    source: Literal["scripted", "stt", "typed"]
    features: dict[str, Any] = Field(default_factory=dict)


class Claim(_P):
    count: int
    claimed_rank: Literal["K", "Q", "A"]
    cards: list[str]                             # god view: actual cards placed
    utterance: str
    features: dict[str, Any]                     # §3.3, from the next actor's point of view


class CardMath(_P):
    U: int
    m: int
    h: int
    k: int
    p_cards: float


class MemoryUsed(_P):
    hyp_id: str
    aspect: str
    text: str
    status: str
    provenance: str
    n_true: int
    n_false: int
    q_true: list[float]                          # [mean, lo, hi]
    q_false: list[float]
    evidence_obs: list[str] = Field(default_factory=list)   # obs ids "w017:g4r2"
    surreal: Optional[dict[str, Any]] = None


class Decision(_P):
    """Mind Inspector record (§7.1). Written AT DECISION TIME; never recomputed for display."""
    kind: Literal["respond", "play"]
    target: Optional[Seat] = None
    claim: Optional[dict[str, Any]] = None       # {count, rank, utterance, features}
    card_math: Optional[CardMath] = None
    memory_used: list[MemoryUsed] = Field(default_factory=list)
    q_used: Optional[float] = None
    q_source: Optional[str] = None               # hyp id | "base:p3" | "drift:p3" | None
    p_final: Optional[float] = None
    ev: dict[str, float] = Field(default_factory=dict)
    action: Literal["challenge", "play_on", "play"]
    explore: bool = False
    propensity: float = 1.0
    rng_u: Optional[float] = None                # the exploration draw, so /replay is exact
    memory_version: int = 0
    ask: Optional[dict[str, Any]] = None
    trace: dict[str, Any] = Field(default_factory=dict)
    obs: Optional[dict[str, Any]] = None         # the observation the policy saw (for /replay)
    agent: Optional[str] = None                  # policy id, e.g. "profiler"


class Challenge(_P):
    target: Seat


class PlayOn(_P):
    pass


class Reveal(_P):
    cards: list[str]
    was_bluff: bool
    loser: Seat
    claimant: Seat
    challenger: Seat
    obs_id: Optional[str] = None                 # "w017:g4r2" — the id evidence lists point at


class LifeLost(_P):
    seat: Seat
    lives_left: int


class GameEnd(_P):
    winner: Optional[Seat]                       # None = draw
    lives: dict[str, int]


class ReflectRequest(_P):
    seat: Seat
    memory_version: int
    new_obs: list[str]


class ReflectResult(_P):
    seat: Seat
    memory_version: int
    proposed: list[dict[str, Any]]
    llm_trace_id: Optional[str] = None
    surreal_trace_id: Optional[str] = None
    latency_ms: int = 0
    error: Optional[str] = None
    source: Optional[str] = None                 # "claude" | "stub" | "timeout"


class HypothesisEv(_P):
    id: str
    about: Optional[Seat] = None
    aspect: str
    text: str
    predicate: dict[str, Any]
    status: Literal["provisional", "confirmed", "refuted"]
    n_true: int
    n_false: int
    q_true: list[float]
    q_false: list[float]
    provenance: Literal["evidence", "untrusted"]
    valid_from_game: int
    valid_until_game: Optional[int] = None
    superseded_by: Optional[str] = None


class BeliefSuperseded(_P):
    aspect: str
    old_id: Optional[str]
    new_id: str
    reason: Literal["drift", "llm", "refuted"]
    at_game: int
    at_round: int
    about: Optional[Seat] = None


class Surgery(_P):
    op: str
    variant: dict[str, Any]
    by: Literal["operator"] = "operator"
    surreal_uncertainty_id: Optional[str] = None


class ForgeResult(_P):
    policy_id: str
    sandbox_id: Optional[str]
    gate: dict[str, Any]                         # {ok, reason}
    games: int = 0
    win_rate: Optional[float] = None
    ci: Optional[list[float]] = None
    crashes: int = 0
    timeouts: int = 0
    promoted: bool = False
    baseline_win_rate: Optional[float] = None


class WorldEnd(_P):
    win_rate: float
    games_won: int
    brier: Optional[float] = None
    rtd: Optional[float] = None
    revealed_to_detection: Optional[int] = None
    exploitation_cost: Optional[float] = None
    false_alarm: Optional[bool] = None


PAYLOADS: dict[str, type[_P]] = {
    "world_start": WorldStart, "game_start": GameStart, "deal": Deal, "ask": Ask,
    "answer": Answer, "claim": Claim, "decision": Decision, "challenge": Challenge,
    "play_on": PlayOn, "reveal": Reveal, "life_lost": LifeLost, "game_end": GameEnd,
    "reflect_request": ReflectRequest, "reflect_result": ReflectResult,
    "hypothesis": HypothesisEv, "belief_superseded": BeliefSuperseded,
    "surgery": Surgery, "forge_result": ForgeResult, "world_end": WorldEnd,
}


# ---------------------------------------------------------------- envelope

class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")
    v: int = SCHEMA_VERSION
    run: str
    world: str
    box: str
    game: int
    round: int
    seq: int
    ts: str
    type: EventType
    actor: Optional[str] = None
    payload: dict[str, Any]
    trace_id: Optional[str] = None


def validate_event(e: dict | Event) -> Event:
    ev = e if isinstance(e, Event) else Event.model_validate(e)
    PAYLOADS[ev.type].model_validate(ev.payload)
    return ev


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def obs_id(world: str, game: int, round_: int) -> str:
    return f"{world}:g{game}r{round_}"


def _dump(payload: Any) -> Any:
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    return payload


class EventWriter:
    """Append-only JSONL writer with a per-world monotonic `seq`. Thread-safe.

    Use `emit()` everywhere; it validates the payload against the contract.
    `path=None` keeps events in memory only (used by /replay and tests).
    """

    def __init__(self, path: str | os.PathLike | None, run: str, world: str, box: str,
                 start_seq: int = 0, validate: bool = True):
        self.path = Path(path) if path else None
        self.run, self.world, self.box = run, world, box
        self.seq = start_seq
        self.validate = validate
        self.events: list[dict] = []
        self._lock = threading.Lock()
        self._fh = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", buffering=1)

    def emit(self, type: str, payload: Any, *, game: int = 0, round: int = 0,
             actor: str | None = None, trace_id: str | None = None) -> dict:
        with self._lock:
            self.seq += 1
            ev = {"v": SCHEMA_VERSION, "run": self.run, "world": self.world, "box": self.box,
                  "game": game, "round": round, "seq": self.seq, "ts": now_ts(), "type": type,
                  "actor": actor, "payload": _dump(payload), "trace_id": trace_id}
            if self.validate:
                validate_event(ev)
            self.events.append(ev)
            if self._fh:
                self._fh.write(json.dumps(ev, separators=(",", ":")) + "\n")
            return ev

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def read_events(path: str | os.PathLike, since: int = 0) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                ev = json.loads(line)
                if ev["seq"] > since:
                    out.append(ev)
    return out


VOLATILE = ("ts", "trace_id")


def canonical(events: Iterable[dict]) -> list[dict]:
    """Strip wall-clock/trace fields so two runs on the same seed compare equal (§3.2)."""
    out = []
    for e in events:
        e = {k: v for k, v in e.items() if k not in VOLATILE}
        p = dict(e["payload"])
        if e["type"] == "decision":
            p.pop("trace", None)
        if e["type"] == "reflect_result":
            for k in ("latency_ms", "llm_trace_id", "surreal_trace_id"):
                p.pop(k, None)
        e["payload"] = p
        out.append(e)
    return out
