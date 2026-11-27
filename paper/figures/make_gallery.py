"""Gallery of configurations with their smallest triangles: python paper/figures/make_gallery.py data/best out.pdf s:64 t:64 c:64"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from heilbronn import geometry as G

BLUE, DARK, ORANGE = '#1f77b4', '#174f7a', '#ff7f0e'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'mathtext.fontset': 'cm'})


def panel(ax, P, V, n):
    T = G.triples(len(P)); a = np.abs(G.areas(P, T)); crit = T[a <= a.min() * (1 + 1e-6)]
    box = {'s': [(0, 0), (1, 0), (1, 1), (0, 1)], 't': [(0, 0), (1, 0), (0, 1)]}.get(V)
    if box is None:
        box = P[ConvexHull(P).vertices]
    ax.add_patch(Polygon(box, closed=True, fill=False, lw=0.8, ec='0.15'))
    for t in crit:
        ax.add_patch(Polygon(P[t], closed=True, fc=BLUE, ec=BLUE, alpha=0.13, lw=0.3))
    ax.scatter(P[:, 0], P[:, 1], s=4, c=DARK, zorder=3, linewidths=0)
    ax.set_aspect('equal'); ax.axis('off')
    lo, hi = np.min(box, 0), np.max(box, 0); pad = 0.02 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
    name = {'s': 'square', 't': 'triangle', 'c': 'convex'}[V]
    m, e = f'{G.value(P, V):.4e}'.split('e')
    ax.set_title(f'{name}, $n={n}$: ${m}\\cdot 10^{{{int(e)}}}$\n{len(crit)} smallest triangles', fontsize=8)


if __name__ == '__main__':
    root, out, cells = sys.argv[1], sys.argv[2], sys.argv[3:]
    fig, axs = plt.subplots(1, len(cells), figsize=(2.15 * len(cells), 2.45))
    for ax, c in zip(np.atleast_1d(axs), cells):
        V, n = c.split(':'); P = G.load(f'{root}/{V}{n}.txt')
        if V == 'c':                       # rotate the convex picture onto its principal axes (PCA), long axis horizontal
            P = P - P.mean(0); w, U = np.linalg.eigh(np.cov(P.T)); P = P @ U[:, ::-1]
        panel(ax, P, V, int(n))
    fig.tight_layout(pad=0.3); fig.savefig(out, bbox_inches='tight')
