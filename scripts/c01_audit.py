"""c01 step 1: text-only leak audit of the relational tell (DEV seeds 7000-7099 only; PREREG_c01.md).

Can "this statement is false" be predicted from the utterance ALONE (no public record)? Logistic regression
(L2, lambda 1.0, a priori) on hashed unigram+bigram BoW (256) + the 64-dim projected sentence embedding
(logreg_embed's features), 5-fold CV grouped by seed; AUC of the out-of-fold scores, 2000-resample bootstrap
over seeds (95% CI). Leak iff CI lower bound > 0.55.
Usage: PYTHONPATH=. .venv/bin/python scripts/c01_audit.py tell_relational|tell_relational_v2 [out.json]
"""
import asyncio, json, sys
import numpy as np
from blab.agents import scripted
from blab.agents.embed import projected
from blab.agents.ours import featurize
from blab.world import World, make_seats, stub_reflector

ROLE = sys.argv[1]
STMT_ONLY = "--statement-only" in sys.argv          # drop the claim line: features of the statement sentence only
sys.argv = [a for a in sys.argv if a != "--statement-only"]
SEEDS = range(7000, 7100)
rows = []
_orig = scripted.Scripted.play


def play(self, obs):
    a = _orig(self, obs)
    if self.role == ROLE and getattr(self, "rel_last", None) and self.rel_last.get("false") is not None:
        rows.append({"seed": CUR[0], "utt": a["utterance"], **self.rel_last})
    return a


scripted.Scripted.play = play
CUR = [0]


async def main():
    for s in SEEDS:
        CUR[0] = s
        seats = make_seats(ROLE, "counter", s)
        seats[0]["params"].update(info="raw")
        await World(wid=f"audit-{s}", seats=seats, seed=s, games=20, reflector=stub_reflector, table=ROLE).run()


asyncio.run(main())


def feats(u):
    if STMT_ONLY:
        from blab.agents.embed import sentences
        u = " ".join(sentences(u)[1:])
    bow = featurize({"utterance": u}, 0.5)[-257:-1]          # 256 hashed uni+bigram presence features
    return np.concatenate([[1.0], bow, np.asarray(projected(u))])


X = np.stack([feats(r["utt"]) for r in rows]); y = np.array([float(r["false"]) for r in rows])
sd = np.array([r["seed"] for r in rows])


def fit(X, y, lam=1.0):
    w = np.zeros(X.shape[1])
    for _ in range(30):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -35, 35)))
        R = lam * np.eye(X.shape[1]); R[0, 0] = 0
        H = (X * (p * (1 - p))[:, None]).T @ X + R + 1e-9 * np.eye(X.shape[1])
        g = X.T @ (p - y) + R @ w
        w -= np.linalg.solve(H, g)
    return w


def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


useeds = np.array(sorted(set(sd)))
fold = {s: i % 5 for i, s in enumerate(useeds)}
f = np.array([fold[s] for s in sd])
oof = np.zeros(len(y))
for k in range(5):
    w = fit(X[f != k], y[f != k])
    oof[f == k] = X[f == k] @ w
A = auc(oof, y)
rng = np.random.default_rng(0)
idx_by_seed = {s: np.where(sd == s)[0] for s in useeds}
bs = []
for _ in range(2000):
    ss = rng.choice(useeds, len(useeds))
    ii = np.concatenate([idx_by_seed[s] for s in ss])
    if 0 < y[ii].sum() < len(ii):
        bs.append(auc(oof[ii], y[ii]))
lo, hi = np.percentile(bs, [2.5, 97.5])
by_kind = {}
for kd in sorted({r["kind"] for r in rows}):
    m = np.array([r["kind"] == kd for r in rows])
    by_kind[kd] = {"n": int(m.sum()), "false_rate": float(y[m].mean()),
                   "auc": auc(oof[m], y[m]) if 0 < y[m].sum() < m.sum() else None}
out = {"role": ROLE, "statement_only": STMT_ONLY, "dev_seeds": "7000-7099", "n_statements": len(y), "false_rate": float(y.mean()),
       "model": "L2 logreg (lambda 1.0, a priori) on 256 hashed uni+bigram BoW + 64-dim projected embedding; 5-fold CV grouped by seed",
       "auc": A, "ci": [float(lo), float(hi)], "leak": bool(lo > 0.55), "by_kind": by_kind}
print(json.dumps(out, indent=1))
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=1)
