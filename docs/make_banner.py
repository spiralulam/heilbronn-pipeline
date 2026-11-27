"""README picture: python docs/make_banner.py data/best docs/banner.png s:22 t:22 c:22
Record configurations with every triangle of minimal area shaded, one colour per triangle (matplotlib's tab10 cycle)."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from heilbronn import geometry as G

COLOURS = plt.get_cmap('tab10').colors
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11})


def panel(ax, P, V, n, alpha):
    T = G.triples(len(P)); a = np.abs(G.areas(P, T)); crit = T[a <= a.min() * (1 + 1e-6)]
    box = {'s': [(0, 0), (1, 0), (1, 1), (0, 1)], 't': [(0, 0), (1, 0), (0, 1)]}.get(V)
    if box is None:
        box = P[ConvexHull(P).vertices]
    for k, t in enumerate(crit):
        ax.add_patch(Polygon(P[t], closed=True, fc=COLOURS[k % 10], ec='none', alpha=alpha))
    ax.add_patch(Polygon(box, closed=True, fill=False, lw=1.0, ec='0.1', zorder=2))
    ax.scatter(P[:, 0], P[:, 1], s=16, color=COLOURS[0], zorder=3, linewidths=0)
    ax.set_aspect('equal', adjustable='datalim'); ax.axis('off')   # equal panel boxes, titles in one line
    lo, hi = np.min(box, 0), np.max(box, 0); pad = 0.03 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
    name = {'s': 'square', 't': 'triangle', 'c': 'convex'}[V]
    ax.set_title(f'{name}, n = {n}: {len(crit)} smallest triangles', fontsize=11, color='0.15')


if __name__ == '__main__':
    root, out, cells = sys.argv[1], sys.argv[2], sys.argv[3:]
    alpha = float(os.environ.get('ALPHA', 0.25))
    fig, axs = plt.subplots(1, len(cells), figsize=(4.0 * len(cells), 4.3))
    for ax, c in zip(np.atleast_1d(axs), cells):
        V, n = c.split(':'); P = G.load(f'{root}/{V}{n}.txt')
        if V == 'c':                       # principal axes, long axis horizontal
            P = P - P.mean(0); w, U = np.linalg.eigh(np.cov(P.T)); P = P @ U[:, ::-1]
        panel(ax, P, V, int(n), alpha)
    fig.tight_layout(pad=0.6); fig.savefig(out, dpi=150, facecolor='white')
