"""b01: sentence-embedding cache for logreg_embed (PREREG.md).

Embeddings come from OpenRouter (`openai/text-embedding-3-small`), precomputed for EVERY sentence the
scripted opponents can say (enumerated from banks.py) and cached by sha1 in blab/data/embed_cache.json
(float16, base64). There is NO network at decision time: a cache miss raises.

An utterance's vector is the mean of its sentences' embeddings ("sentence embeddings", PREREG).
`python -m blab.agents.embed build` (re)builds the cache from the enumeration.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from blab.agents import banks
from blab.engine import RANKS

CACHE = Path(__file__).resolve().parent.parent / "data" / "embed_cache.json"
MODEL = "openai/text-embedding-3-small"
FALLBACK = "liquid/lfm-2.5-embedding-350m:free"
PROJ_DIM = 64
_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    return [s for s in _SPLIT.split((text or "").strip()) if s]


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def enumerate_texts() -> list[str]:
    """Every utterance/answer the scripted roles can emit (all bank lines x n x rank, relational statements)."""
    out: list[str] = []
    for bank in (banks.CASUAL, banks.FORMAL):
        for t in bank:
            for n in (1, 2, 3):
                for r in RANKS:
                    out.append(banks.render(t, n, r))
                    out.append(banks.render(t, n, r, digits=True))
    for t in (banks.OVERT_TELL, banks.OVERT_PLAIN):
        for n in (1, 2, 3):
            for r in RANKS:
                for dg in (False, True):
                    line = banks.render(t, n, r, digits=dg)
                    out.append(line[0].upper() + line[1:])
    for t in banks.REL_COUNT:
        for N in banks.REL_N_RANGE:
            for r in RANKS:
                out.append(t.format(N=N, rank_pl=banks.ranks_word(r, 2)))
    for t in banks.REL_HAND:
        for h in banks.REL_H_RANGE:
            out.append(t.format(h=h))
    out += banks.REL_SAME + banks.REL_DIFF
    out += banks.ANSWER_CASUAL + banks.ANSWER_FORMAL + banks.ANSWER_EVASIVE + banks.ANSWER_PLAIN
    sents: list[str] = []
    seen = set()
    for t in out:
        for s in sentences(t):
            if s not in seen:
                seen.add(s)
                sents.append(s)
    return sents


@lru_cache(maxsize=1)
def _cache() -> dict[str, np.ndarray]:
    d = json.loads(CACHE.read_text())
    return {k: np.frombuffer(base64.b64decode(v), dtype=np.float16).astype(np.float64) for k, v in d["vectors"].items()}


@lru_cache(maxsize=1)
def projection() -> np.ndarray:
    dim = len(next(iter(_cache().values())))
    return np.random.default_rng(0).standard_normal((dim, PROJ_DIM)) / np.sqrt(PROJ_DIM)


@lru_cache(maxsize=None)
def projected(text: str) -> tuple[float, ...]:
    """Fixed 64-dim random projection (seed 0) of the mean sentence embedding; zeros for empty text."""
    ss = sentences(text)
    if not ss:
        return tuple([0.0] * PROJ_DIM)
    c = _cache()
    vs = []
    for s in ss:
        k = sha1(s)
        if k not in c:
            raise KeyError(f"embedding cache miss (no network at decision time): {s!r}")
        vs.append(c[k])
    return tuple(float(v) for v in np.mean(vs, axis=0) @ projection())


def build(model: str = MODEL) -> None:
    import httpx
    from blab.env import load_env
    load_env()
    texts = enumerate_texts()
    key = os.environ["OPENROUTER_API_KEY"]
    vecs: dict[str, str] = {}
    for i in range(0, len(texts), 256):
        chunk = texts[i:i + 256]
        r = httpx.post("https://openrouter.ai/api/v1/embeddings", timeout=120,
                       headers={"Authorization": f"Bearer {key}"}, json={"model": model, "input": chunk})
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        assert len(data) == len(chunk)
        for t, d in zip(chunk, data):
            vecs[sha1(t)] = base64.b64encode(np.asarray(d["embedding"], dtype=np.float16).tobytes()).decode()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"model": model, "n": len(vecs), "dtype": "float16", "vectors": vecs}, sort_keys=True))
    print(f"cached {len(vecs)} sentence embeddings ({model}) -> {CACHE}")


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        try:
            build(MODEL)
        except Exception as e:                                   # noqa: BLE001
            print(f"{MODEL} failed ({e!r}); falling back to {FALLBACK}")
            build(FALLBACK)
