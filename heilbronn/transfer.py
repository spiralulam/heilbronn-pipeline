"""Transfers between the three variants (affine invariance).

A square or triangle configuration is also a convex configuration, with value at least as large (its hull has area at
most that of the container).  Conversely, a convex configuration is mapped into the square by the affine map that sends
a small enclosing parallelogram onto [0,1]^2, and into the triangle by the map that sends a small enclosing triangle onto
(0,0), (1,0), (0,1).  The value is preserved up to the ratio hull area / enclosing area; the SLP polish then adapts the
configuration to the new container.
  parallelogram: minimum over pairs of hull-edge directions (each side pair flush with a hull edge; optimal)
  triangle:      minimum over triples of supporting hull-edge lines that enclose the hull (heuristic; a minimal
                 enclosing triangle has at least one side flush with an edge [O'Rourke et al. 1986])
"""
import itertools
import numpy as np
from scipy.spatial import ConvexHull

from .geometry import project


def _edge_lines(P):
    """outer normal form n.x <= c of every hull edge"""
    H = P[ConvexHull(P).vertices]; m = len(H); out = []
    for q in range(m):
        d = H[(q + 1) % m] - H[q]; nrm = np.array([d[1], -d[0]]) / np.hypot(*d); c = nrm @ H[q]
        if (H @ nrm).max() > c + 1e-9: nrm, c = -nrm, -c
        out.append((nrm, c))
    return H, out


def to_square(P):
    P = np.asarray(P, float); H, L = _edge_lines(P); best = None
    for (n1, _), (n2, _) in itertools.combinations(L, 2):
        s = abs(n1[0] * n2[1] - n1[1] * n2[0])
        if s < 1e-6: continue
        area = np.ptp(H @ n1) * np.ptp(H @ n2) / s
        if best is None or area < best[0]: best = (area, n1, n2)
    _, n1, n2 = best
    u = (P @ n1 - (H @ n1).min()) / np.ptp(H @ n1); v = (P @ n2 - (H @ n2).min()) / np.ptp(H @ n2)
    return project(np.c_[u, v], 's')


def to_triangle(P):
    P = np.asarray(P, float); H, L = _edge_lines(P); best = None
    for e in itertools.combinations(L, 3):
        A = np.array([x[0] for x in e]); c = np.array([x[1] for x in e])
        try:
            Vt = np.array([np.linalg.solve(A[[a, b]], c[[a, b]]) for a, b in ((0, 1), (1, 2), (0, 2))])
            T = np.c_[Vt[1] - Vt[0], Vt[2] - Vt[0]]
            if abs(np.linalg.det(T)) < 1e-12: continue
            B = np.linalg.solve(T, (P - Vt[0]).T).T          # barycentric coordinates: must enclose every point
        except np.linalg.LinAlgError:                         # parallel edge lines
            continue
        if B.min() < -1e-9 or B.sum(1).max() > 1 + 1e-9: continue
        area = abs(np.linalg.det(T)) / 2
        if best is None or area < best[0]: best = (area, B)
    if best is None:                                          # no three edge lines enclose the hull (e.g. a parallelogram):
        return project(to_square(P) / 2, 't')                 # square, then (u, v) / 2 lies in the triangle
    return project(best[1], 't')


def transfer(P, src, dst):
    """configuration of variant src -> starting configuration for variant dst (square <-> triangle via the hull)"""
    if dst == src or dst == 'c':
        return np.asarray(P, float)
    return to_square(P) if dst == 's' else to_triangle(P)
