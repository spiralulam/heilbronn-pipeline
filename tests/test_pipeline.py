"""Fast checks of the CPU components: python -m pytest tests"""
import numpy as np
import pytest

from heilbronn import exact, geometry as G
from heilbronn.neighbours import deletion_scan, polish_job
from heilbronn.polish import slp
from heilbronn.refine import refine
from heilbronn.transfer import transfer


def random_config(V, n, seed=0):
    P = np.random.RandomState(seed).rand(n, 2)
    if V == 't':
        f = P.sum(1) > 1; P[f] = 1 - P[f]
    return P


@pytest.mark.parametrize('V', 'stc')
def test_value_conventions(V):
    P = np.array([[0, 0], [1, 0], [0, 1], [0.3, 0.3]])
    a = abs(G.areas(P)).min()
    expected = {'s': a, 't': 2 * a, 'c': a / G.hull_area_grad(P)[0]}[V]
    assert np.isclose(G.value(P, V), expected)


@pytest.mark.parametrize('V', 'stc')
def test_slp_improves_and_stays_inside(V):
    P = random_config(V, 12)
    Q, v = slp(P, V, iters=200)
    assert v > G.value(P, V) and G.inside(Q, V, 1e-12)
    assert np.isclose(v, G.value(Q, V))


@pytest.mark.parametrize('V', 'stc')
def test_refine_exact_value(V):
    Q, v = slp(random_config(V, 10, 1), V, iters=300)
    lines, ve = refine(Q, V, digits=30)
    F = exact.parse(lines)
    assert exact.inside(F, V) and exact.value(F, V) == ve
    assert float(ve) >= v * (1 - 1e-9)


def test_affine_invariance_and_transfer():
    Q, _ = slp(random_config('c', 11, 2), 'c', iters=300)
    A = np.array([[2.0, 0.7], [-0.3, 1.1]])
    assert np.isclose(G.value(Q @ A.T + 5, 'c'), G.value(Q, 'c'))
    for W in 'st':
        R = transfer(Q, 'c', W)
        assert G.inside(R, W, 1e-12) and len(R) == len(Q)


def test_deletion_scan():
    Q, _ = slp(random_config('s', 11, 3), 's', iters=300)
    v, R = deletion_scan(Q, 's', lambda f, jobs: [f(j) for j in jobs], iters=100, final=200)
    assert len(R) == 10 and np.isclose(v, G.value(R, 's')) and polish_job((R, 's', 10))[0] >= v * (1 - 1e-12)


def test_transfer_all_stored():
    """every stored configuration maps into the other containers; the convex value is affine invariant"""
    import glob, os
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), '..', 'data', 'best', '*.txt')))
    if not files:
        pytest.skip('no data/best')
    for fn in files:
        V = os.path.basename(fn)[0]; P = G.load(fn)
        for W in 'stc':
            R = transfer(P, V, W)
            assert len(R) == len(P) and G.inside(R, W, 1e-12), (fn, W)
            assert G.value(R, 'c') >= G.value(P, 'c') * (1 - 1e-6), (fn, W)


class _FakeBatch:
    """stands in for gpu.Batch: soft values per cell size from SOFT, random configurations as chain results"""
    SOFT = {}

    def __init__(self, V, nv, B, **kw):
        self.V, self.nv, self.NM = V, np.asarray(nv), int(np.max(nv)) + 3

    def load(self, P, TY, NN):
        self.P = P

    def fill(self):
        import types
        v = np.array([self.SOFT[int(n)] for n in self.nv])
        return types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: v))

    def points(self, b):
        return random_config(self.V, int(self.nv[b]), seed=int(b))


def test_gpu_gate(tmp_path, monkeypatch):
    """cells with soft value >= gate * target are polished, best ratio first; a cell without a target always passes"""
    import sys, types
    from heilbronn.neighbours import GpuScreen
    from heilbronn.store import Store
    monkeypatch.setitem(sys.modules, 'heilbronn.gpu', types.SimpleNamespace(Batch=_FakeBatch))
    store = Store(str(tmp_path))
    for n in (10, 11, 12):                                   # weak stored configurations: random points
        G.save(store.path('s', n), random_config('s', n, seed=100 + n), 's', '')
    t = {n: store.get('s', n)[0] for n in (10, 11, 12)}
    _FakeBatch.SOFT = {10: 0.95 * t[10], 11: 0.5 * t[11], 12: 1.2 * t[12], 13: 1e-3}
    jobs = []
    def pmap(f, js):
        jobs.extend(js); return [f(j) for j in js]
    S = GpuScreen('s', [10, 11, 12, 13], store, per_cell=10, top=9, gate=0.92, iters=100)
    out = S.cycle(pmap)
    assert [len(j[0]) for j in jobs] == [12, 13, 10]       # ratio 1.2, 1 (no target), 0.95; 0.5 is gated out
    assert sorted(n for n, g in out) == [10, 12, 13] and all(g > 1 for n, g in out)
    assert store.get('s', 13)[1] is not None and S.last[0] == 3
