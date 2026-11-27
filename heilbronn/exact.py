"""Exact verification in rational arithmetic: the published value is the one of the decimal coordinates as written."""
from fractions import Fraction
from itertools import combinations


def parse(lines):
    """decimal strings -> list of Fraction pairs (exact)"""
    return [(Fraction(x), Fraction(y)) for x, y in lines]


def hull(F):
    """convex hull (Andrew's monotone chain) of exact points, counter-clockwise, collinear points dropped"""
    pts = sorted(set(F))
    if len(pts) < 3:
        return pts
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def inside(F, V):
    if V == 's':
        return all(0 <= x <= 1 and 0 <= y <= 1 for x, y in F)
    if V == 't':
        return all(x >= 0 and y >= 0 and x + y <= 1 for x, y in F)
    return True


def value(F, V):
    """exact Heilbronn value of the exact points F (container membership is checked separately by inside)"""
    m = min(abs((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])) for p, q, r in combinations(F, 3))
    if V == 's':
        return m / 2
    if V == 't':
        return m
    h = hull(F)
    H = sum(h[i][0] * h[(i + 1) % len(h)][1] - h[(i + 1) % len(h)][0] * h[i][1] for i in range(len(h))) / 2
    return m / 2 / H
