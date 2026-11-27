"""Containers, triangle areas and the Heilbronn value.

Three variants, each a letter used throughout the package:
  's'  unit square [0,1]^2                         value = min area
  't'  triangle with vertices (0,0), (1,0), (0,1)  value = min area / (1/2)
  'c'  convex hull of the points themselves        value = min area / area(conv P)
All three values are invariant under the affine maps that preserve the container; 'c' under all affine maps.
"""
import itertools
import numpy as np

VARIANTS = ('s', 't', 'c')
NAMES = {'s': 'square', 't': 'triangle', 'c': 'convex'}
_TRIPLES = {}


def triples(n):
    """all index triples i < j < k as an (n choose 3, 3) array (cached)"""
    if n not in _TRIPLES:
        _TRIPLES[n] = np.array(list(itertools.combinations(range(n), 3)))
    return _TRIPLES[n]


def areas(P, T=None):
    """signed areas of the triangles T (default: all triples)"""
    T = triples(len(P)) if T is None else T
    i, j, k = T[:, 0], T[:, 1], T[:, 2]
    return 0.5 * ((P[j, 0] - P[i, 0]) * (P[k, 1] - P[i, 1]) - (P[k, 0] - P[i, 0]) * (P[j, 1] - P[i, 1]))


def hull_area_grad(P):
    """area of conv(P) and its gradient with respect to all coordinates (non-zero on hull vertices only)"""
    from scipy.spatial import ConvexHull
    v = ConvexHull(P).vertices          # counter-clockwise
    g = np.zeros_like(P, dtype=float)
    m, A = len(v), 0.0
    for q in range(m):
        a, b, c = v[q], v[(q + 1) % m], v[q - 1]
        A += P[a, 0] * P[b, 1] - P[b, 0] * P[a, 1]
        g[a] = 0.5 * (P[b, 1] - P[c, 1]), 0.5 * (P[c, 0] - P[b, 0])
    return 0.5 * A, g


def value(P, V):
    m = np.abs(areas(P)).min()
    if V == 't':
        return 2.0 * m
    if V == 'c':
        return m / hull_area_grad(P)[0]
    return m


def normalize(P, V):
    """convex: centroid 0 and hull area 1 (keeps coordinates O(1) for trust regions and float32); s / t unchanged"""
    P = np.asarray(P, float)
    if V != 'c':
        return P
    return (P - P.mean(0)) / np.sqrt(hull_area_grad(P)[0])


def project(P, V):
    """push points back into the container (after a step that may leave it by rounding)"""
    P = np.asarray(P, float).copy()
    if V == 's':
        return np.clip(P, 0.0, 1.0)
    if V == 't':
        P = np.maximum(P, 0.0)
        s = P.sum(1)
        P[s > 1] /= s[s > 1, None]
    return P


def inside(P, V, tol=0.0):
    if V == 's':
        return bool((P >= -tol).all() and (P <= 1 + tol).all())
    if V == 't':
        return bool((P >= -tol).all() and (P.sum(1) <= 1 + tol).all())
    return True


def boundary_types(P, V, eps=1e-7):
    """0 interior, 1..NS on side s (open), 100 + c on corner c; the sides and corners are listed in gpu.GEO"""
    if V == 'c':
        return [0] * len(P)
    corners = {'s': [(0, 0), (1, 0), (0, 1), (1, 1)], 't': [(0, 0), (1, 0), (0, 1)]}[V]
    out = []
    for x, y in np.asarray(P, float):
        t = next((100 + c for c, (a, b) in enumerate(corners) if abs(x - a) < eps and abs(y - b) < eps), 0)
        if not t:
            if V == 's':
                t = 1 if y < eps else 2 if y > 1 - eps else 3 if x < eps else 4 if x > 1 - eps else 0
            else:
                t = 1 if y < eps else 2 if x < eps else 3 if x + y > 1 - eps else 0
        out.append(t)
    return out


def load(fn):
    """read a configuration: one point 'x y' per line, lines starting with # are comments"""
    pts = [l.split()[:2] for l in open(fn) if l.strip() and not l.startswith('#')]
    return np.array([(float(x), float(y)) for x, y in pts])


def save(fn, P, V, note=''):
    """write atomically with a header '# V n=.. value=.. note'"""
    import os
    P = np.asarray(P, float)
    if V == 'c':
        P = P - P.min(0)                 # first quadrant (the value is affine invariant)
    with open(fn + '.tmp', 'w') as f:
        f.write(f'# {V} n={len(P)} value={float(value(P, V))!r} {note}\n'.rstrip() + '\n')
        for x, y in P:
            f.write(f'{float(x)!r} {float(y)!r}\n')
    os.replace(fn + '.tmp', fn)
