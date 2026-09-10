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
    ap.add_argument('--cycles', type=int, default=0,
                    help='run this many SCF cycles first and time the builds '
                         'on the resulting density instead of on the `minao` '
                         'guess.  A `minao` guess is exactly zero on the '
                         'polarisation shells and screens far harder than a '
                         'converged density, so a fixed-density speedup '
                         'measured on it is not the speedup the SCF sees -- '
                         'see docs/ROUND2_1.8.1.md 2.')
    ap.add_argument('--no-grad', action='store_true',
                    help='skip the gradient timing.  Its unpatched half runs '
                         "GPU4PySCF's CPU one-electron derivative, so it is "
                         'sensitive to OMP_NUM_THREADS in a way the GPU '
                         'builds are not -- set that explicitly, or leave the '
                         'gradient out when only the builds are wanted.')
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


def needs_ecp(atoms, basis):
    """Whether this basis carries an ECP that this molecule needs.

    The def2 family replaces the core of everything past Kr with an ECP, and a
    4d or 5d transition-metal complex is wrong without it.  Nothing else here
    has one, and passing ``ecp=`` for a set that has none makes the parser
    complain once per element, so the test is made rather than caught.
    """
    from pyscf import gto
    if not str(basis).lower().startswith('def2'):
        return False
    return any(gto.charge(sym) > 36 for sym, _ in gto.format_atom(atoms))


def build_mol(a):
    """Build the molecule, attaching the basis's own ECP if the elements need it."""
    from pyscf import gto
    kw = dict(atom=a.xyz, basis=a.basis, charge=a.charge, spin=a.spin,
              verbose=0, max_memory=80000)
    if needs_ecp(a.xyz, a.basis):
        kw['ecp'] = a.basis
    return gto.M(**kw)


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
