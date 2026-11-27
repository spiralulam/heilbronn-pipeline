"""Command line.

  python -m heilbronn run V [V ...] --n 37-100 --store best [--cpu 12] [--engines 3] [--no-gpu]   the search driver
  python -m heilbronn polish V in.txt out.txt                     SLP polish of one configuration
  python -m heilbronn transfer SRC DST in.txt out.txt             map to another variant, then polish
  python -m heilbronn scan V in.txt out.txt                       deletion scan (n+1 -> n)
  python -m heilbronn export V in.txt out.txt [--digits 32]       tie refinement + exact value of the written decimals
  python -m heilbronn verify V in.txt                             exact value of a coordinate file
"""
import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog='heilbronn', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('variants', nargs='+', choices='stc')
    r.add_argument('--n', default='37-100'); r.add_argument('--store', default='best')
    r.add_argument('--cpu', type=int, default=None); r.add_argument('--engines', type=int, default=None)
    r.add_argument('--engine', default=None, help='path of the C engine (default csrc/heil next to the package)')
    r.add_argument('--no-gpu', action='store_true'); r.add_argument('--hours', type=float, default=None)
    for name in ('polish', 'scan', 'export'):
        p = sub.add_parser(name); p.add_argument('V', choices='stc'); p.add_argument('src'); p.add_argument('out')
        if name == 'export': p.add_argument('--digits', type=int, default=32)
    p = sub.add_parser('transfer'); p.add_argument('src_v', choices='stc'); p.add_argument('dst_v', choices='stc')
    p.add_argument('src'); p.add_argument('out')
    p = sub.add_parser('verify'); p.add_argument('V', choices='stc'); p.add_argument('src')
    a = ap.parse_args(argv)

    from . import geometry as G
    if a.cmd == 'run':
        from .run import run
        lo, hi = (int(x) for x in a.n.split('-')) if '-' in a.n else (int(a.n), int(a.n))
        run(a.variants, range(lo, hi + 1), a.store, cpu=a.cpu, engines=a.engines, engine=a.engine, gpu=not a.no_gpu, hours=a.hours)
    elif a.cmd == 'polish':
        from .polish import slp
        P, v = slp(G.load(a.src), a.V); G.save(a.out, P, a.V, 'SLP polish'); print(f'{v:.15g}')
    elif a.cmd == 'transfer':
        from .polish import slp
        from .transfer import transfer
        P, v = slp(transfer(G.load(a.src), a.src_v, a.dst_v), a.dst_v); G.save(a.out, P, a.dst_v, f'transfer from {a.src_v}'); print(f'{v:.15g}')
    elif a.cmd == 'scan':
        from concurrent.futures import ProcessPoolExecutor
        from .neighbours import deletion_scan
        with ProcessPoolExecutor() as ex:
            v, P = deletion_scan(G.load(a.src), a.V, lambda f, jobs: list(ex.map(f, jobs)))
        G.save(a.out, P, a.V, 'deletion scan'); print(f'{v:.15g}')
    elif a.cmd == 'export':
        from .refine import refine
        lines, ve = refine(G.load(a.src), a.V, digits=a.digits, log=lambda m: print(m, file=sys.stderr))
        with open(a.out, 'w') as f:
            f.writelines(f'{x}\t{y}\n' for x, y in lines)
        print(f'exact value {float(ve):.17g}')
    elif a.cmd == 'verify':
        from . import exact
        F = exact.parse([l.split()[:2] for l in open(a.src) if l.strip() and not l.startswith('#')])
        ok = exact.inside(F, a.V); ve = exact.value(F, a.V)
        print(f'n={len(F)} inside={ok} exact value {float(ve):.17g}')


if __name__ == '__main__':
    main()
