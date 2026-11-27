"""Trust-region sequential linear programming (SLP) for max_P min_k |a_k(P)| / H(P).

At the iterate P with trust radius delta, solve the LP (HiGHS)
    max_{d, t}  t
    s.t.        v_k + g_k . d >= t       for every triangle k in S,
                |d|_inf <= delta,  P + d in the container,
where v_k = |a_k| / H is the normalised area and g_k its gradient.  Accept the step if the true value increases and
update delta by the ratio of actual to predicted gain.

The LP keeps only triangles that can become the minimum inside the box.  Moving all points by at most delta changes
|a_k| by at most delta/2 * L_k, where L_k is the sum of |dx| + |dy| over the three edges, so a triangle with
|a_k| - delta/2 L_k > a_min + delta/2 L_min can never bind; dropping it does not change the LP optimum.  The remaining
rows are sparse (seven non-zeros, plus the hull vertices for the convex variant).
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .geometry import areas, hull_area_grad, normalize, project, triples, value


def _rows(P, T, sel, a, scale):
    """sparse rows -scale * sign(a_k) * grad a_k for k in sel, plus +1 for t (last column)"""
    n = len(P)
    i, j, k = T[sel, 0], T[sel, 1], T[sel, 2]
    s = 0.5 * np.sign(a[sel]) * scale
    pi, pj, pk = P[i], P[j], P[k]
    m = len(sel); r = np.arange(m)
    rows = np.concatenate([r] * 7)
    cols = np.concatenate([2 * i, 2 * i + 1, 2 * j, 2 * j + 1, 2 * k, 2 * k + 1, np.full(m, 2 * n)])
    vals = np.concatenate([-s * (pj[:, 1] - pk[:, 1]), -s * (pk[:, 0] - pj[:, 0]), -s * (pk[:, 1] - pi[:, 1]),
                           -s * (pi[:, 0] - pk[:, 0]), -s * (pi[:, 1] - pj[:, 1]), -s * (pj[:, 0] - pi[:, 0]), np.ones(m)])
    return rows, cols, vals


def slp(P, V, iters=300, delta=None, tol=1e-15):
    """polish P for variant V; returns (P, value).  Convex configurations are returned normalised (centroid 0, area 1)."""
    P = normalize(np.array(P, float), V)
    n = len(P); T = triples(n)
    delta = 0.02 / np.sqrt(n) if delta is None else delta
    f = 2.0 if V == 't' else 1.0                              # value = f * min area for s / t
    c = np.zeros(2 * n + 1); c[-1] = -1.0
    I, J, K = T[:, 0], T[:, 1], T[:, 2]
    if V == 't':                                              # x + y <= 1 for every moved point
        R = sp.csr_matrix((np.ones(2 * n), (np.repeat(np.arange(n), 2), np.arange(2 * n))), shape=(n, 2 * n + 1))
    for _ in range(iters):
        a = areas(P, T); aa = np.abs(a)
        L1 = np.abs(P[J] - P[K]).sum(1) + np.abs(P[K] - P[I]).sum(1) + np.abs(P[I] - P[J]).sum(1)
        if V == 'c':
            H, gH = hull_area_grad(P); gh = gH.ravel(); hc = np.nonzero(gh)[0]
            v = aa / H; q = v.argmin()
            bd = delta * (0.5 * L1 / H + v * np.abs(gh).sum() / H)
            sel = np.where(v - bd <= v[q] + bd[q] * 1.0001 + 1e-18)[0]
            rows, cols, vals = _rows(P, T, sel, a, 1.0 / H)
            r = np.arange(len(sel))                           # hull term: -v_k / H * grad H
            rows = np.concatenate([rows, np.repeat(r, len(hc))])
            cols = np.concatenate([cols, np.tile(hc, len(sel))])
            vals = np.concatenate([vals, np.outer(v[sel] / H, gh[hc]).ravel()])
            A = sp.csr_matrix((vals, (rows, cols)), shape=(len(sel), 2 * n + 1))   # duplicates are summed
            b = v[sel]; bounds = [(-delta, delta)] * (2 * n) + [(None, None)]; cur = v[q]
        else:
            q = aa.argmin()
            sel = np.where(aa <= aa[q] + 0.5 * delta * (L1 + L1[q]) * 1.0001 + 1e-18)[0]
            rows, cols, vals = _rows(P, T, sel, a, f)
            A = sp.csr_matrix((vals, (rows, cols)), shape=(len(sel), 2 * n + 1))
            b = f * aa[sel]; x = P.ravel(); cur = f * aa[q]
            if V == 's':
                bounds = [(max(-delta, -x[u]), min(delta, 1 - x[u])) for u in range(2 * n)] + [(None, None)]
            else:
                bounds = [(max(-delta, -x[u]), delta) for u in range(2 * n)] + [(None, None)]
                A = sp.vstack([A, R]).tocsr(); b = np.concatenate([b, 1 - P[:, 0] - P[:, 1]])
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status != 0:
            delta *= 0.5; continue
        Pn = project(P + res.x[:-1].reshape(n, 2), V); pred = res.x[-1]
        vn = value(Pn, V)
        if vn > cur:
            P = normalize(Pn, V)
            rho = (vn - cur) / max(pred - cur, 1e-300)
            delta = min(2 * delta, 0.1) if rho > 0.75 else delta * 0.5 if rho < 0.25 else delta
        else:
            delta *= 0.3
        if delta < tol:
            break
    return P, value(P, V)
