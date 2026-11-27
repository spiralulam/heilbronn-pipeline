"""The table of best configurations: one file <root>/<V><n>.txt per cell, plus optional reference values (for example
the published records) in <root>/targets.json as {"s59": 0.0017187, ...}.  Every component reads its sources here and
offers its results here; a result is kept only if it beats the stored value."""
import json
import os
import threading

import numpy as np

from .geometry import inside, load, save, value


class Store:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._cache = {}
        self._lock = threading.Lock()                       # offer() is called from several threads

    def path(self, V, n):
        return os.path.join(self.root, f'{V}{n}.txt')

    def get(self, V, n):
        """(value, points) of the stored configuration, (0.0, None) if there is none"""
        fn = self.path(V, n)
        try:
            mt = os.path.getmtime(fn)
        except OSError:
            return 0.0, None
        if self._cache.get((V, n), (None,))[0] != mt:
            P = load(fn)
            self._cache[(V, n)] = (mt, float(value(P, V)) if len(P) == n else 0.0, P)
        return self._cache[(V, n)][1:]

    def reference(self, V, n):
        try:
            return float(json.load(open(os.path.join(self.root, 'targets.json'))).get(f'{V}{n}', 0.0))
        except (OSError, ValueError):
            return 0.0

    def target(self, V, n):
        return max(self.get(V, n)[0], self.reference(V, n))

    def offer(self, V, n, P, note=''):
        """keep P if it beats the stored configuration (relative margin 1e-9); returns the gain factor or None"""
        if P is None or len(P) != n or not inside(np.asarray(P, float), V, 1e-12):
            return None
        P = np.asarray(P, float)
        v = float(value(P, V))
        with self._lock:
            old = self.get(V, n)[0]
            if v <= old * (1 + 1e-9):
                return None
            save(self.path(V, n), P, V, note)
        return v / old if old else float('inf')
