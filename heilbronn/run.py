"""The search driver: all generators feed one table of best configurations (store.Store), and every improvement is
picked up by the others in their next round.

  GPU screening   neighbour seeds n -/+ 1, 2 and re-insertions for every cell, soft-polished on the GPU; the best
                  few per cycle get the SLP polish (neighbours.GpuScreen).  Cycles through the variants.
  deletion scans  best(n+1) minus each point in turn, from the top cell down; a cell is rescanned when its source changed.
  transfers       an improved configuration is also tried in the other variants (transfer.py).
  C engine        `engines` slots run the C search engine (csrc/): cold symmetric runs under a point group and warm
                  ruin & recreate from the stored best; outputs are SLP-polished and offered to the table.
CPU work runs in one process pool of size cpu - engines, so the machine is never oversubscribed.
"""
import hashlib
import os
import random
import subprocess
import tempfile
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '1')                          # one BLAS thread per worker

import numpy as np

from .geometry import load
from .neighbours import deletion_scan, insert_cpu, polish_job
from .store import Store
from .transfer import transfer

# point groups for the cold symmetric runs, with sampling weights (from the yield of the campaign)
GROUPS = {'s': {'D4': 10, 'D2': 2, 'C4': 1.5, 'D2d': 1.5, 'C2': 0.3, 'D1': 0.1},
          't': {'D3': 4, 'C3': 4, 'D1': 1},
          'c': {'D3': 3, 'C3': 3, 'D6': 3, 'D7': 3, 'D8': 3, 'D9': 3, 'D11': 2.5, 'D2': 2, 'D4': 2, 'D5': 2, 'D10': 2,
                'D12': 1.5, 'C2': 1, 'C4': 1, 'C5': 1, 'C6': 1, 'D1': 0.5}}


DEFECTS = {1: ('v3', 'f1'), 2: ('v2', 'f2'), 3: ('v1',)}       # square, n mod 4 -> D4 vacancies (v) / free points (f)
DEFECT_W = {'v1': 4, 'v2': 3, 'v3': 3, 'f1': 0.7, 'f2': 0.7}


def compatible(V, n, G):
    if V == 's' and G in ('D4', 'C4'): return n % 4 == 0
    if V == 's' and G in ('D2', 'D2d', 'C2'): return n % 2 == 0
    if V == 't' and G in ('C3', 'D3'): return n % 3 != 2
    return True


class Driver:
    def __init__(self, variants, ns, root, cpu, engines, engine, gpu, hours):
        self.variants, self.ns, self.store = variants, list(ns), Store(root)
        self.stop_at = time.time() + 3600 * hours if hours else None
        self.engine, self.engines, self.use_gpu = engine, engines, gpu
        self.pool = ProcessPoolExecutor(max(1, cpu - engines))
        self.log_file = os.path.join(root, 'run.log'); self.lock = threading.Lock()
        self.pending = []                                    # improved cells waiting for transfers
        self.tmp = tempfile.mkdtemp(prefix='heil_')
        self.procs = set()                                   # running engine processes (terminated on stop)

    def running(self):
        return not os.path.exists(os.path.join(self.store.root, 'STOP')) and (self.stop_at is None or time.time() < self.stop_at)

    def pmap(self, f, jobs):
        return list(self.pool.map(f, jobs))

    def log(self, msg):
        with self.lock, open(self.log_file, 'a') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')

    def failed(self, what):
        """a generator step raised: log it and carry on (one bad configuration must not stop the run)"""
        self.log(f'{what} failed: ' + traceback.format_exc().strip().splitlines()[-1])

    def improved(self, V, n, gain, how):
        self.log(f'{V}{n} {self.store.get(V, n)[0]:.12e} ({(gain - 1) * 100 if gain != float("inf") else 0:+.4f} %) {how}')
        with self.lock:
            self.pending.append((V, n))

    # ------------------------------------------------------------------ generators
    def gpu_loop(self):
        from .neighbours import GpuScreen
        screens = [GpuScreen(V, self.ns, self.store) for V in self.variants]
        while self.running():
            for S in screens:
                try:
                    t0 = time.time(); imp = S.cycle(self.pmap)
                except Exception:
                    self.failed(f'GPU {S.V} cycle'); time.sleep(10); continue
                for n, g in imp:
                    self.improved(S.V, n, g, 'GPU neighbour screening')
                self.log(f'GPU {S.V} cycle {S.cyc}: {time.time() - t0:.0f} s, {S.last[0]} SLP polishes, {len(imp)} improved; '
                         f'best soft / target {S.last[1]:.3f}, cells >= 0.95: {S.last[2]}')

    def scan_loop(self):
        done = {}
        while self.running():
            idle = True
            for V in self.variants:
                for n in reversed(self.ns):
                    v, P = self.store.get(V, n + 1)
                    if P is None: continue
                    key = hashlib.md5(P.tobytes()).hexdigest()
                    if done.get((V, n)) == key: continue
                    idle = False; done[(V, n)] = key
                    try:
                        self.scan_cell(V, n, P)
                    except Exception:
                        self.failed(f'scan {V}{n + 1}')
                    self.do_transfers()
            self.do_transfers()
            if idle: time.sleep(30)

    def scan_cell(self, V, n, P):
        """deletion scan P (n+1 points) -> n points; without a GPU also greedy insertion n+1 -> n+2 on the CPU"""
        t0 = time.time(); vq, Q = deletion_scan(P, V, self.pmap)
        g = self.store.offer(V, n, Q, f'deletion scan from n={n + 1}') if Q is not None else None
        if g: self.improved(V, n, g, 'deletion scan')
        else: self.log(f'scan {V}{n + 1} -> {V}{n}: {vq / max(self.store.get(V, n)[0], 1e-300):.4f} of best, {time.time() - t0:.0f} s')
        if not self.use_gpu and n + 2 in self.ns:
            res = self.pmap(polish_job, [(Q0, V, 300) for Q0 in insert_cpu(P, V)])
            vq, Q = max(res, key=lambda r: r[0])
            g = self.store.offer(V, n + 2, Q, f'insertion from n={n + 1}') if Q is not None else None
            if g: self.improved(V, n + 2, g, 'insertion')

    def do_transfers(self):
        with self.lock:
            todo, self.pending = self.pending, []
        jobs = []
        for V, n in set(todo):
            _, P = self.store.get(V, n)
            for W in ('s', 't', 'c'):
                if W != V and W in self.variants and P is not None:
                    try:
                        jobs.append((W, n, (transfer(P, V, W), W, 300)))
                    except Exception:
                        self.failed(f'transfer {V}{n} -> {W}')
        for (W, n, _), (v, Q) in zip(jobs, self.pmap(polish_job, [j[2] for j in jobs])):
            g = self.store.offer(W, n, Q, 'transfer + SLP') if Q is not None else None
            if g: self.improved(W, n, g, 'transfer')

    def engine_job(self, rng):
        """one C engine run: cold symmetric (60 %) or warm ruin & recreate from the stored best (40 %)"""
        V = rng.choice(self.variants); n = rng.choice(self.ns); seed = rng.randint(1, 10**7)
        out = os.path.join(self.tmp, f'{V}{n}_{seed}')
        src = self.store.path(V, n)
        base = [self.engine, V, str(n), '--seed', str(seed), '--out', out]
        if rng.random() < 0.4 and os.path.exists(src):
            return V, n, base + ['--time', '900', '--init', src], out, 'warm RR'
        groups = [G for G in GROUPS[V] if compatible(V, n, G)]
        if V == 's' and n % 4:                               # D4 with defect orbits: n = 4k - vacancies or 4k + free points
            groups += [f'D4{d}' for d in DEFECTS[n % 4]]
        G = rng.choices(groups, weights=[GROUPS[V].get(g, DEFECT_W.get(g[2:], 1)) for g in groups])[0]
        secs = 1800 if V == 's' else 1500 if n < 60 else 2700
        if G.startswith('D4') and len(G) > 2:
            kind, k = G[2], G[3]
            return V, n, base + ['--group', 'D4', '--vacancies' if kind == 'v' else '--free', k, '--time', '3600'], out, f'symmetric D4 {G[2:]}'
        return V, n, base + ['--group', G, '--time', str(secs)], out, f'symmetric {G}'

    def engine_slot(self, k):
        rng = random.Random(k * 7919 + int(time.time()))
        while self.running():
            V, n, cmd, out, how = self.engine_job(rng)
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with self.lock:
                self.procs.add(proc)
            while proc.poll() is None and self.running():
                time.sleep(5)
            with self.lock:
                self.procs.discard(proc)
            if proc.poll() is None:                          # stopped: end the run, discard its output
                proc.terminate(); proc.wait()
                return
            if not os.path.exists(out + '.txt'): continue
            try:
                P = load(out + '.txt'); os.remove(out + '.txt')
                v, Q = self.pmap(polish_job, [(P, V, 400)])[0]
                g = self.store.offer(V, n, Q, f'C engine ({how}) + SLP') if Q is not None else None
                if g: self.improved(V, n, g, f'C engine, {how}')
            except Exception:
                self.failed(f'C engine output {V}{n}')

    def run(self):
        threads = [threading.Thread(target=self.scan_loop, daemon=True)]
        if self.use_gpu:
            threads.append(threading.Thread(target=self.gpu_loop, daemon=True))
        if self.engine and os.path.exists(self.engine):
            threads += [threading.Thread(target=self.engine_slot, args=(k,), daemon=True) for k in range(self.engines)]
        self.log(f'start: variants {",".join(self.variants)}, n = {self.ns[0]}..{self.ns[-1]}, {len(threads)} threads')
        for t in threads: t.start()
        while self.running() and any(t.is_alive() for t in threads):
            time.sleep(10)
        with self.lock:
            for p in self.procs:
                p.terminate()
        self.log('stop')


def run(variants, ns, root, cpu=None, engines=None, engine=None, gpu=True, hours=None):
    cpu = cpu or max(1, (os.cpu_count() or 2) - 1)
    engines = cpu // 3 if engines is None else engines
    engine = engine or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csrc', 'heil')
    if gpu:
        from .gpu import available
        gpu = available()
    Driver(variants, ns, root, cpu, engines, engine, gpu, hours).run()
