"""Shared setup for the benchmarks: molecule, patching, timing."""
import argparse
import sys
import time


def parser(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--xyz', required=True, help='geometry file')
    ap.add_argument('--charge', type=int, default=0)
    ap.add_argument('--spin', type=int, default=0)
    ap.add_argument('--basis', default='6-31G*')
    ap.add_argument('--xc', default='b3lyp-d3bj')
    ap.add_argument('--grids', type=int, default=3, help='grid level')
    ap.add_argument('--conv-tol', type=float, default=1e-9)
    ap.add_argument('--patch', default='',
                    help="'all', a comma separated module list, or '' for "
                         "unpatched GPU4PySCF")
    ap.add_argument('--reps', type=int, default=3)
    return ap


def apply_patch(spec):
    """Patch before gpu4pyscf is used, and report what is active."""
    import electrolites
    if spec == 'all':
        electrolites.patch_all()
    elif spec:
        electrolites.patch(spec.split(','))
    print(f'# gpu4pyscf {electrolites.gpu4pyscf_version()}  '
          f'patched: {electrolites.patched() or "(none)"}', flush=True)
    return electrolites.patched()


def build_mol(a):
    from pyscf import gto
    return gto.M(atom=a.xyz, basis=a.basis, charge=a.charge, spin=a.spin,
                 verbose=0, max_memory=80000)


def make_timer(tag, results):
    """A timer that reports ``min`` of ``reps`` and records the number."""
    import cupy as cp

    def timeit(label, fn, reps):
        try:
            fn()
            cp.cuda.Stream.null.synchronize()
        except Exception as exc:                       # noqa: BLE001
            results[label] = f'ERR {type(exc).__name__}: {exc}'
            print(f'[{tag}] {label:16s} ERR {type(exc).__name__}: '
                  f'{str(exc)[:120]}', file=sys.stderr, flush=True)
            return None
        ts = []
        out = None
        for _ in range(reps):
            cp.cuda.Stream.null.synchronize()
            t = time.perf_counter()
            out = fn()
            cp.cuda.Stream.null.synchronize()
            ts.append(time.perf_counter() - t)
        results[label] = min(ts)
        print(f'[{tag}] {label:16s} {min(ts):9.3f} s   (min of {reps})',
              flush=True)
        return out
    return timeit
