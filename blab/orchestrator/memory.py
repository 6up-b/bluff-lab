"""SurrealDB Agent Memory (ex-"Spectron") via raw REST + httpx (SPEC §10.2).

Box-local memory (hypothesis graph in memory/<seat>.json + checkpoints) stays AUTHORITATIVE; this module
mirrors it into SurrealDB for supersession history / asOf / uncertainty and retrieval traces.
Everything is a no-op (returns None) when SPECTRON_URL / SPECTRON_CONTEXT_ID / SPECTRON_API_KEY are missing,
and never raises into callers (errors are returned/logged).

Verified shapes (spike, 2026-10-16):
  POST /scopes {"path": "lab/world/w017"}                                     → 201
  POST /facts {"infer":"triples","triples":[{"entity":{"type":"agent","name":"w017.p3"},
               "key":"verbal_tell","value":"..."}],"scopes":[["lab/world/w017/agent/p1"]],
               "observed_at":"2026-01-01T06:00:00Z"}                           → 200, source.kind "upsert"
  same entity+key+scope again → old gets validUntil + supersededBy               (S1)
  GET /entities/agent/w017.p3?asOf=T → attribute valid at T                     (S3)
  GET /entities/agent/w017.p3/history/<key>                                     → {"history":[...]}
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
_registered: set[str] = set()
_last_value: dict[tuple, str] = {}


def conf() -> tuple[str, dict] | None:
    url = os.environ.get("SPECTRON_URL", "")
    ctx = os.environ.get("SPECTRON_CONTEXT_ID", "")
    key = os.environ.get("SPECTRON_API_KEY", "")
    if not (url and ctx and key):
        return None
    if "://" not in url and "://" in ctx:      # tolerate the two values being swapped in settings.env
        url, ctx = ctx, url
    if "://" not in url:
        url = "https://" + url
    return f"{url.rstrip('/')}/api/v1/{ctx}", {"Authorization": f"Bearer {key}"}


def enabled() -> bool:
    return conf() is not None


def missing() -> list[str]:
    return [k for k in ("SPECTRON_URL", "SPECTRON_CONTEXT_ID", "SPECTRON_API_KEY") if not os.environ.get(k)]


def game_ts(game: int, round_: int = 0) -> str:
    """Game time → timestamp so asOf matches game numbers (§10.2)."""
    return (EPOCH + timedelta(hours=game, minutes=round_)).strftime("%Y-%m-%dT%H:%M:%SZ")


def world_scope(wid: str) -> str:
    return f"lab/world/{wid}"


def agent_scope(wid: str, seat: str) -> str:
    return f"lab/world/{wid}/agent/{seat}"


def entity_name(wid: str, about: str) -> str:
    return f"{wid}.{about}"


def entity_ref(wid: str, about: str) -> str:
    return f"agent/{entity_name(wid, about)}"


class Surreal:
    """Thin async client. Use `async with Surreal() as s:`; every method returns None on any failure."""

    def __init__(self, timeout: float = 30.0):
        c = conf()
        self.ok = c is not None
        self.base, headers = c if c else ("", {})
        self._c = httpx.AsyncClient(headers=headers, timeout=timeout) if self.ok else None
        self.last_error: str | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        if self._c:
            await self._c.aclose()

    async def _req(self, method: str, path: str, **kw) -> Any:
        if not self.ok:
            return None
        try:
            r = await self._c.request(method, self.base + path, **kw)
            if r.status_code >= 400:
                self.last_error = f"{method} {path} → {r.status_code} {r.text[:200]}"
                return None
            return r.json() if r.content else {}
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{method} {path} → {type(e).__name__}: {e}"
            return None

    # ---------------------------------------------------------------- scopes
    async def ensure_scopes(self, wid: str, seat: str | None = None) -> None:
        paths = ["lab", world_scope(wid)] + ([agent_scope(wid, seat)] if seat else [])
        for p in paths:
            if p in _registered:
                continue
            res = await self._req("POST", "/scopes", json={"path": p})
            if res is not None or (self.last_error and "409" in self.last_error):
                _registered.add(p)

    # ---------------------------------------------------------------- writes
    async def write_triples(self, triples: list[dict], scope: str, observed_at: str | None = None,
                            session_id: str | None = None) -> dict | None:
        body: dict = {"infer": "triples", "triples": triples, "scopes": [[scope]]}
        if observed_at:
            body["observed_at"] = observed_at
        if session_id:
            body["session_id"] = session_id
        return await self._req("POST", "/facts", json=body)

    async def write_observations(self, wid: str, entries: list[dict]) -> int:
        """Revealed claims → shared world scope, one attribute per obs on the claimant's entity."""
        await self.ensure_scopes(wid)
        by_game: dict[int, list[dict]] = {}
        for e in entries:
            if not isinstance(e, dict) or not e.get("obs_id"):
                continue
            by_game.setdefault(int(e.get("game") or 0), []).append(e)
        n = 0
        for g, es in sorted(by_game.items()):
            triples = []
            for e in es:
                f = e.get("features") or {}
                text = (f"Game {e.get('game')} round {e.get('round')}: Player {str(e.get('by'))[-1]} claimed "
                        f"{e.get('count')} {f.get('table_rank', '')} saying \"{e.get('utterance', '')}\" and was "
                        f"{'bluffing' if e.get('was_bluff') else 'honest'} (formal_register={f.get('formal_register')},"
                        f" hand_size={f.get('hand_size')}, p_cards={e.get('p_cards')}).")
                triples.append({"entity": {"type": "agent", "name": entity_name(wid, e.get("by", "p?"))},
                                "key": f"reveal_{str(e['obs_id']).split(':')[-1]}", "value": text})
            if await self.write_triples(triples, world_scope(wid), observed_at=game_ts(g, 59)) is not None:
                n += len(triples)
        return n

    async def write_beliefs(self, wid: str, seat: str, hypotheses: list[dict], game: int) -> dict[str, dict]:
        """One belief per (about, aspect) on agent/<wid>.<about>, key = aspect, source.kind upsert.
        Picks the best live hypothesis per aspect (confirmed > provisional; not refuted/superseded).
        Only writes when the value changed, so each write is a real supersession.
        Returns {hyp_id: {"entity","key","fact_ref"}} for decision.memory_used[].surreal."""
        await self.ensure_scopes(wid, seat)
        rank = {"confirmed": 2, "provisional": 1}
        best: dict[tuple, dict] = {}
        for h in hypotheses or []:
            if h.get("provenance") == "untrusted" or h.get("valid_until_game") is not None:
                continue
            if h.get("status") not in rank:
                continue
            k = (h.get("about"), h.get("aspect"))
            if k not in best or (rank[h["status"]], h.get("n_true", 0)) > (rank[best[k]["status"]],
                                                                           best[k].get("n_true", 0)):
                best[k] = h
        refs: dict[str, dict] = {}
        triples = []
        for (about, aspect), h in best.items():
            qt = h.get("q_true") or [None]
            qf = h.get("q_false") or [None]
            val = (f"{h.get('text')} [{h.get('status')}; id={h.get('id')}; n_true={h.get('n_true', 0)}, "
                   f"n_false={h.get('n_false', 0)}; q_true={qt[0]}; q_false={qf[0]}; "
                   f"predicate={json.dumps(h.get('predicate'), separators=(',', ':'))}]")
            ck = (wid, seat, about, aspect)
            refs[h.get("id", "")] = {"entity": entity_ref(wid, about), "key": aspect}
            if _last_value.get(ck) == val:
                continue
            triples.append({"entity": {"type": "agent", "name": entity_name(wid, about)}, "key": aspect,
                            "value": val})
            _last_value[ck] = val
        if triples:
            res = await self.write_triples(triples, agent_scope(wid, seat), observed_at=game_ts(game, 59))
            for a in ((res or {}).get("extraction") or {}).get("attributes") or []:
                for r in refs.values():
                    if r["key"] == a.get("key") and r["entity"].split("/", 1)[1] in str(a.get("entityId", "")):
                        r["fact_ref"] = a.get("id")
        return refs

    async def write_false_memory(self, wid: str, seat: str, about: str, text: str, game: int = 0) -> str | None:
        """Untrusted conversational `turn` fact → SurrealDB raises an uncertainty record (not a supersession).
        Returns an uncertainty id if one is visible."""
        await self.ensure_scopes(wid, seat)
        body = {"text": text, "infer": "full", "scopes": [[agent_scope(wid, seat)]],
                "observed_at": game_ts(game, 59)}
        res = await self._req("POST", "/facts", json=body)
        ex = (res or {}).get("extraction") or {}
        for u in ex.get("uncertainties") or []:
            if isinstance(u, dict) and u.get("id"):
                return u["id"]
        unc = await self.uncertainty(wid, about)
        items = (unc or {}).get("unknowns") or (unc or {}).get("uncertainties") or []
        if items and isinstance(items[-1], dict):
            return items[-1].get("id")
        return (res or {}).get("turnId")

    # ---------------------------------------------------------------- reads
    async def query(self, wid: str, seat: str, text: str, game: int, k: int = 20) -> dict | None:
        body = {"query": text, "k": k, "lens": [[world_scope(wid)], [agent_scope(wid, seat)]],
                "source": f"reflect:{wid}:g{game}"}
        return await self._req("POST", "/query", json=body)

    async def entity(self, wid: str, about: str, as_of_game: int | None = None) -> dict | None:
        params = {"asOf": game_ts(as_of_game, 59)} if as_of_game is not None else None
        return await self._req("GET", f"/entities/agent/{entity_name(wid, about)}", params=params)

    async def history(self, wid: str, about: str, key: str) -> dict | None:
        return await self._req("GET", f"/entities/agent/{entity_name(wid, about)}/history/{key}")

    async def uncertainty(self, wid: str, about: str) -> dict | None:
        return await self._req("GET", "/uncertainty", params={"entity": entity_ref(wid, about)})

    async def forget_dry_run(self, query: str) -> dict | None:
        return await self._req("POST", "/forget", json={"query": query, "dryRun": True})

    async def trace(self, tid: str) -> dict | None:
        return await self._req("GET", f"/traces/{tid}")


def _trace_id(res: dict | None) -> str | None:
    if not isinstance(res, dict):
        return None
    t = res.get("trace") or {}
    return t.get("traceId") or t.get("id") or res.get("traceId")


async def reflect_sync(wid: str, seat: str, payload: dict) -> dict:
    """Reflection-boundary round trip (before the LLM): write new reveals + current beliefs, then retrieve.
    Returns {"surreal_trace_id", "hits", "refs", "error"} (all None-safe)."""
    if not enabled():
        return {"surreal_trace_id": None, "hits": [], "refs": {}, "error": "disabled"}
    game = int(payload.get("game") or 0)
    async with Surreal() as s:
        await s.write_observations(wid, [o for o in payload.get("new_obs") or [] if isinstance(o, dict)])
        refs = await s.write_beliefs(wid, seat, payload.get("hypotheses") or [], game)
        res = await s.query(wid, seat, "How does each opponent bluff? Verbal tells, triggers, trust exploitation.",
                            game)
        hits = []
        for hit in (res or {}).get("hits") or (res or {}).get("results") or []:
            if isinstance(hit, dict):
                hits.append({k: hit.get(k) for k in ("id", "text", "value", "summary", "score", "key") if k in hit})
        return {"surreal_trace_id": _trace_id(res), "hits": hits[:20], "refs": refs, "error": s.last_error}


async def after_reflection(wid: str, seat: str, hypotheses: list[dict], game: int) -> dict:
    """After the world applied the reflection: mirror the new belief state (supersessions)."""
    if not enabled():
        return {}
    async with Surreal() as s:
        return await s.write_beliefs(wid, seat, hypotheses, game)


# ------------------------------------------------------------------ spike S1–S3

def spike_checks() -> list[dict]:
    def row(c, st, d):
        return {"check": c, "status": st, "detail": str(d)[:300]}

    async def go():
        wid = f"spike{int(time.time()) % 100000}"
        rows = []
        async with Surreal(timeout=90) as s:
            await s.ensure_scopes(wid, "p1")
            e = {"type": "agent", "name": entity_name(wid, "p3")}
            r1 = await s.write_triples([{"entity": e, "key": "baseline_bluff", "value": "Player 3 rarely bluffs (q=0.10)"}],
                                       agent_scope(wid, "p1"), observed_at=game_ts(2))
            r2 = await s.write_triples([{"entity": e, "key": "baseline_bluff", "value": "Player 3 bluffs often now (q=0.60)"}],
                                       agent_scope(wid, "p1"), observed_at=game_ts(8))
            hist = await s.history(wid, "p3", "baseline_bluff")
            links = (hist or {}).get("history") or []
            sup = any(h.get("supersededBy") for h in links)
            rows.append(row("S1", "PASS" if len(links) >= 2 and sup else "FAIL",
                            f"entity agent/{wid}.p3 history/baseline_bluff links={len(links)} superseded={sup} "
                            f"kind={[h.get('source', {}).get('kind') for h in links]} {s.last_error or ''}"))
            old = await s.entity(wid, "p3", as_of_game=4)
            vals = [a.get("value") for a in (old or {}).get("attributes") or [] if a.get("key") == "baseline_bluff"]
            rows.append(row("S3", "PASS" if vals and "0.10" in vals[0] else "FAIL", f"asOf game 4 → {vals}"))
            t0 = time.time()
            uid = await s.write_false_memory(wid, "p1", "p3", "Player 3 never bluffs.", game=9)
            unc = await s.uncertainty(wid, "p3")
            n_unc = len((unc or {}).get("unknowns") or (unc or {}).get("uncertainties") or [])
            rows.append(row("S2", "PASS" if n_unc >= 1 else "FAIL",
                            f"turn write ({time.time()-t0:.1f}s) → uncertainty records={n_unc} id={uid} "
                            f"resp={json.dumps(unc)[:120]} {s.last_error or ''}"))
            q = await s.query(wid, "p1", "Does Player 3 bluff?", 9)
            rows[-1]["query_trace_id"] = _trace_id(q)
        return rows

    return asyncio.run(go())
