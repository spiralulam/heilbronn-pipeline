"""High-precision tie refinement and the exact gate.

A polished configuration is a local max-min point: a set A of active triangles share the minimum value t.  We solve
the tie system
    sigma_k * v_k(z) = t     (k in A),     v_k = normalised signed area,  sigma_k = sign at the start point,
for the free parameters z (interior points: x, y; points on a side: their position along it; corners fixed) and t,
by Gauss-Newton in mpmath with the analytic Jacobian (minimum-norm step when the system is underdetermined).  The active
set is taken within a relative window of the minimum; a ladder of windows is tried, and a solution is accepted only if
no other triangle ends below t.  The convex value is affine invariant; there the three points spanning the largest
triangle are held fixed, which removes exactly the six gauge directions.  Coordinates are then truncated toward zero to `digits` decimals and the value of the
written decimals is recomputed in exact rational arithmetic.  A refinement that loses value is discarded.
"""
from decimal import Decimal, localcontext

import mpmath as mp
import numpy as np

from . import exact
from .geometry import areas, triples, value as fvalue

WINDOWS = ('1e-9', '1e-7', '1e-6', '1e-5')


def _kinds(P, V, eps):
    """per point: 'fix' corner, 'x' / 'y' only that coordinate free, 'h' on x + y = 1 (parameter x), 'xy' interior"""
    K = []
    for x, y in P:
        if V == 's':
            fx, fy = abs(x) < eps or abs(x - 1) < eps, abs(y) < eps or abs(y - 1) < eps
            K.append('fix' if fx and fy else 'x' if fy else 'y' if fx else 'xy')
        elif V == 't':
            on = [abs(y) < eps, abs(x) < eps, abs(x + y - 1) < eps]
            K.append('fix' if sum(on) >= 2 else 'x' if on[0] else 'y' if on[1] else 'h' if on[2] else 'xy')
        else:
            K.append('xy')
    if V == 'c':                                            # affine gauge: fix the points of the largest triangle
        Pf = np.array([[float(x), float(y)] for x, y in P]); T = triples(len(Pf))
        a = np.abs(areas(Pf, T))
        for i in T[int(np.argmax(a))]:
            K[i] = 'fix'
    return K


def _snap(P, K, V):
    out = []
    for (x, y), k in zip(P, K):
        if V == 's':
            if k in ('x', 'fix'): y = mp.mpf(0) if abs(y) < 0.5 else mp.mpf(1)
            if k in ('y', 'fix'): x = mp.mpf(0) if abs(x) < 0.5 else mp.mpf(1)
        elif V == 't':
            if k == 'x': y = mp.mpf(0)
            elif k == 'y': x = mp.mpf(0)
            elif k == 'h': y = 1 - x
            elif k == 'fix': x, y = mp.nint(x), mp.nint(y)
        out.append((x, y))
    return out


def _hull(Q):
    """indices of the convex hull, counter-clockwise (monotone chain)"""
    idx = sorted(range(len(Q)), key=lambda i: (Q[i][0], Q[i][1]))
    def cr(o, a, b): return (Q[a][0] - Q[o][0]) * (Q[b][1] - Q[o][1]) - (Q[a][1] - Q[o][1]) * (Q[b][0] - Q[o][0])
    lo, up = [], []
    for i in idx:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], i) <= 0: lo.pop()
        lo.append(i)
    for i in reversed(idx):
        while len(up) >= 2 and cr(up[-2], up[-1], i) <= 0: up.pop()
        up.append(i)
    return lo[:-1] + up[:-1]


class _System:
    def __init__(self, P, V):
        self.V = V; n = len(P)
        self.K = _kinds(P, V, mp.mpf('1e-9'))
        self.P = _snap(P, self.K, V)
        self.var = [(i, c) for i, k in enumerate(self.K)
                    for c in ((0, 1) if k == 'xy' else (0,) if k in ('x', 'h') else (1,) if k == 'y' else ())]
        self.col = {}
        for v, (i, c) in enumerate(self.var):
            self.col.setdefault(i, []).append((v, c))
        self.T = triples(n)

    def z0(self):
        return [self.P[i][c] for i, c in self.var]

    def build(self, z):
        Q = [list(p) for p in self.P]
        for (i, c), val in zip(self.var, z):
            Q[i][c] = val
            if self.K[i] == 'h': Q[i][1] = 1 - val
        return Q

    def scale(self, Q):
        """v_k = area2_k * scale; for the convex variant also the hull and the gradient of its area"""
        if self.V == 's': return mp.mpf(1) / 2, None
        if self.V == 't': return mp.mpf(1), None
        h = _hull(Q); m = len(h)
        H = sum(Q[h[q]][0] * Q[h[(q + 1) % m]][1] - Q[h[(q + 1) % m]][0] * Q[h[q]][1] for q in range(m)) / 2
        gH = {h[q]: ((Q[h[(q + 1) % m]][1] - Q[h[q - 1]][1]) / 2, (Q[h[q - 1]][0] - Q[h[(q + 1) % m]][0]) / 2) for q in range(m)}
        return 1 / (2 * H), (H, gH)

    def area2(self, Q, k):
        i, j, l = self.T[k]
        return (Q[j][0] - Q[i][0]) * (Q[l][1] - Q[i][1]) - (Q[l][0] - Q[i][0]) * (Q[j][1] - Q[i][1])

    def grad(self, Q, k, sc, hd):
        """gradient of v_k with respect to the free parameters (sparse dict column -> value)"""
        i, j, l = self.T[k]
        g = {i: (Q[j][1] - Q[l][1], Q[l][0] - Q[j][0]), j: (Q[l][1] - Q[i][1], Q[i][0] - Q[l][0]),
             l: (Q[i][1] - Q[j][1], Q[j][0] - Q[i][0])}
        g = {p: (gx * sc, gy * sc) for p, (gx, gy) in g.items()}
        if hd is not None:                                  # d(a2 / 2H) = d a2 / 2H - a2 / (2 H^2) dH
            H, gH = hd; f = self.area2(Q, k) * sc / H
            for p, (hx, hy) in gH.items():
                gx, gy = g.get(p, (0, 0)); g[p] = (gx - f * hx, gy - f * hy)
        out = {}
        for p, (gx, gy) in g.items():
            for v, c in self.col.get(p, ()):
                out[v] = out.get(v, 0) + (gx if c == 0 else gy) - (gy if self.K[p] == 'h' else 0)
        return out


def _newton(S, act, sg, z, t0, digits):
    x = list(z) + [t0]; m, nv = len(act), len(z) + 1; nrm = mp.inf
    for it in range(60):
        Q = S.build(x[:-1]); sc, hd = S.scale(Q)
        F = mp.matrix([sg[a] * S.area2(Q, k) * sc - x[-1] for a, k in enumerate(act)])
        nrm = mp.norm(F)
        if nrm < mp.mpf(10) ** (-(digits + 15)):
            break
        J = mp.zeros(m, nv)
        for a, k in enumerate(act):
            for v, d in S.grad(Q, k, sc, hd).items():
                J[a, v] = sg[a] * d
            J[a, nv - 1] = -1
        try:
            dx = -(J.T * mp.lu_solve(J * J.T, F)) if m <= nv else -mp.lu_solve(J.T * J, J.T * F)
        except ZeroDivisionError:
            return None
        x = [x[c] + dx[c] for c in range(nv)]
    return x, nrm


def _trunc(v, digits):
    s = mp.nstr(v, digits + 20, strip_zeros=False, min_fixed=-mp.inf, max_fixed=mp.inf)
    neg = s.startswith('-'); s = s.lstrip('-')
    a, b = (s if '.' in s else s + '.').split('.')
    r = a + '.' + (b + '0' * digits)[:digits]
    return '-' + r if neg and any(ch not in '0.' for ch in r) else r


def refine(P, V, digits=32, windows=WINDOWS, log=None):
    """returns (lines, exact value): `digits`-decimal coordinate strings and the exact value of exactly those decimals.
    Falls back to the truncated input coordinates when no refinement is accepted or the refinement loses value."""
    P = np.asarray(P, float); v0 = fvalue(P, V)
    with mp.workdps(digits + 25):
        S = _System([(mp.mpf(repr(float(x))), mp.mpf(repr(float(y)))) for x, y in P], V)
        z = S.z0(); Q0 = S.build(z); sc0, _ = S.scale(Q0)
        vals = [abs(S.area2(Q0, k) * sc0) for k in range(len(S.T))]; vmin = min(vals)
        best, seen = None, set()
        for w in windows:
            act = [k for k, v in enumerate(vals) if v <= vmin * (1 + mp.mpf(w))]
            if tuple(act) in seen: continue
            seen.add(tuple(act))
            r = _newton(S, act, [mp.sign(S.area2(Q0, k)) for k in act], z, vmin, digits)
            if r is None: continue
            x, nrm = r
            ok = nrm < mp.mpf(10) ** (-(digits + 5))
            if ok:                                           # no other triangle may end below t (float screen)
                Qf = np.array([[float(a), float(b)] for a, b in S.build(x[:-1])])
                ok = fvalue(Qf, V) >= float(x[-1]) * (1 - 1e-11)
            if log: log(f'window {w}: {len(act)} active, residual {mp.nstr(nrm, 3)}, t = {mp.nstr(x[-1], 20)} {"ok" if ok else "rejected"}')
            if ok and (best is None or x[-1] > best[-1]):
                best = x
        Q = S.build(best[:-1]) if best is not None else S.P
        lines = [(_trunc(a, digits), _trunc(b, digits)) for a, b in Q]
    F = exact.parse(lines)
    ve = exact.value(F, V) if exact.inside(F, V) else 0
    if float(ve) < v0 * (1 - 1e-9):                         # refinement lost value: truncated doubles instead
        lines = [(_trunc(mp.mpf(repr(x)), digits), _trunc(mp.mpf(repr(y)), digits)) for x, y in P]
        if V == 't':                                         # keep hypotenuse points exactly on x + y = 1
            with localcontext() as ctx:
                ctx.prec = 80
                lines = [(a, b) if Decimal(a) + Decimal(b) <= 1 else (a, format(1 - Decimal(a), 'f')) for a, b in lines]
        F = exact.parse(lines); ve = exact.value(F, V) if exact.inside(F, V) else 0
    return lines, ve
