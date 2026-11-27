"""Moves along n: a good configuration with n -/+ k points is a good start for n points.

  up   best(n-1) plus one point,          up2  best(n-2) plus two points,
  dn   best(n+1) minus one point,         dn2  best(n+2) minus two points,
  mv   best(n) with 1-3 points removed and re-inserted.
Points are added by greedy insertion: the candidate (of many random ones) whose smallest triangle with an existing pair
is largest.  Two drivers:
  gpu_cycle     every cell gets hundreds of such seeds per move, completed and soft-polished on the GPU in float32; only
                the best few cells per cycle (soft value >= gate * target, least-tried first) get the SLP polish.
  deletion_scan exhaustive on the CPU: delete every point of best(n+1) in turn, SLP-polish all, re-polish the best few.
"""
import sys

import numpy as np

from .geometry import boundary_types, normalize, value
from .polish import slp

MOVES = {'up': -1, 'dn': 1, 'up2': -2, 'dn2': 2, 'mv': 0}


def polish_job(job):
    """(P, V, iters) -> (value, P); runs in a worker process.  A failed polish (degenerate start) returns (0, None)."""
    P, V, it = job
    try:
        Q, v = slp(P, V, iters=it)
        return float(v), Q
    except Exception as e:
        print(f'polish failed ({V}, n={len(P)}): {e!r}', file=sys.stderr)
        return 0.0, None


def deletion_scan(P, V, pmap, iters=300, top=3, final=1500):
    """best (value, Q) over all one-point deletions of P; pmap(f, jobs) maps jobs over the worker pool"""
    P = normalize(P, V)
    res = pmap(polish_job, [(np.delete(P, p, 0), V, iters) for p in range(len(P))])
    order = sorted(range(len(P)), key=lambda p: -res[p][0])[:top]
    fin = pmap(polish_job, [(res[p][1], V, final) for p in order if res[p][1] is not None])
    return max(fin, key=lambda r: r[0]) if fin else (0.0, None)


class GpuScreen:
    """the GPU driver: one batch holding about `per_cell` seeds per cell (split evenly over the moves) for the cells ns
    of variant V; per cycle the best chain of at most `top` cells with soft ratio >= `gate` gets the SLP polish"""

    def __init__(self, V, ns, store, per_cell=510, moves=('up', 'dn', 'up2', 'dn2', 'mv'), top=9, gate=0.92, iters=300):
        from .gpu import Batch
        self.V, self.ns, self.store, self.moves, self.top, self.gate, self.iters = V, list(ns), store, moves, top, gate, iters
        self.per = per_cell // len(moves)
        self.cell = np.repeat(np.arange(len(self.ns)), self.per * len(moves))
        self.move = np.tile(np.repeat(np.arange(len(moves)), self.per), len(self.ns))
        self.batch = Batch(V, np.array(self.ns)[self.cell], len(self.cell))
        self.fails = np.zeros(len(self.ns), int)            # SLP polishes per cell since its last real gain
        self.cyc = 0

    def _seeds(self, rng):
        V, NM, B = self.V, self.batch.NM, len(self.cell)
        P0 = np.zeros((B, 2 * NM), np.float32); T0 = np.zeros((B, NM), np.int32)
        NN = np.array(self.ns, np.int32)[self.cell]; ok = np.zeros(B, bool)
        src = {m: self.store.get(V, m) for m in range(self.ns[0] - 2, self.ns[-1] + 3)}
        b = 0
        for n in self.ns:
            for name in self.moves:
                dk = MOVES[name]; _, S = src[n + dk]
                if S is not None:
                    S = normalize(S, V); ty = np.array(boundary_types(S, V), np.int32); m = len(S)
                    for q in range(self.per):
                        keep = np.ones(m, bool)
                        if name == 'dn': keep[(q + 7 * self.cyc) % m] = False
                        elif name == 'dn2': keep[rng.choice(m, 2, replace=False)] = False
                        elif name == 'mv': keep[rng.choice(m, 1 + q % 3, replace=False)] = False
                        k = int(keep.sum()); P0[b + q, :2 * k] = S[keep].reshape(-1); T0[b + q, :k] = ty[keep]; NN[b + q] = k
                    ok[b:b + self.per] = True
                b += self.per
        return P0, T0, NN, ok

    def cycle(self, pmap, rng=np.random):
        """one screening cycle; returns [(n, gain)] for the cells that improved"""
        self.cyc += 1; V = self.V
        P0, T0, NN, ok = self._seeds(rng)
        self.batch.load(P0, T0, NN)
        v = self.batch.fill().cpu().numpy().astype(float)
        tg = np.array([self.store.target(V, n) for n in self.ns])[self.cell]
        with np.errstate(divide='ignore', invalid='ignore'):  # soft ratio; a cell without a target has ratio 1
            r = np.where(ok & (v > 0), v / np.where(tg > 0, tg, v), 0.0)
        best = np.full(len(self.ns), -1); rc = np.zeros(len(self.ns))
        for b in np.flatnonzero(r > 0):
            c = self.cell[b]
            if r[b] > rc[c]: rc[c], best[c] = r[b], b
        cells = sorted([c for c in range(len(self.ns)) if rc[c] >= self.gate], key=lambda c: (self.fails[c], -rc[c]))[:self.top]
        res = pmap(polish_job, [(self.batch.points(best[c]), V, self.iters) for c in cells])
        out = []; self.last = (len(cells), float(rc.max()), int((rc >= 0.95).sum()))   # polished, best soft ratio, cells >= 0.95
        for c, (vq, Q) in zip(cells, res):
            n = self.ns[c]; self.fails[c] += 1
            g = self.store.offer(V, n, Q, f'GPU {self.moves[self.move[best[c]]]} + SLP') if Q is not None else None
            if g:
                if g > 1 + 1e-5: self.fails[c] = 0
                out.append((n, g))
        return out


def insert_cpu(P, V, k=4, grid=300):
    """CPU greedy insertion (no GPU): the k best grid points by the smallest triangle with an existing pair"""
    P = normalize(P, V); m = len(P)
    i, j = np.triu_indices(m, 1); A = P[i]; d = P[j] - A
    lo, hi = (P.min(0), P.max(0)) if V == 'c' else (np.zeros(2), np.ones(2))
    g = (np.arange(grid) + 0.5) / grid
    X, Y = np.meshgrid(lo[0] + g * (hi[0] - lo[0]), lo[1] + g * (hi[1] - lo[1])); C = np.c_[X.ravel(), Y.ravel()]
    if V == 't': C = C[C.sum(1) < 1]
    if V == 'c':
        from matplotlib.path import Path
        from scipy.spatial import ConvexHull
        C = C[Path(P[ConvexHull(P).vertices]).contains_points(C)]
    s = np.concatenate([np.abs(d[None, :, 0] * (c[:, 1:2] - A[None, :, 1]) - d[None, :, 1] * (c[:, 0:1] - A[None, :, 0])).min(1)
                        for c in np.array_split(C, max(1, len(C) // 1500))])
    picks, step = [], 3 * (hi - lo).max() / grid
    for o in np.argsort(-s):
        if all(np.abs(C[o] - c).max() > step for c in picks): picks.append(C[o])
        if len(picks) >= k: break
    return [np.vstack([P, c]) for c in picks]
