"""`fastdfj3c` against GPU4PySCF, on the three-centre Coulomb build and on a
whole pure-functional SCF.

`docs/DF_WORK_LIST.md` **W9**.  Six variants of ``df_jk.get_j``:

* ``stock``  -- GPU4PySCF 1.8.1 unchanged;
* ``ctr``    -- the recontraction tables cached (`fastdfj3c` item 2) and
                nothing else;
* ``opt``    -- the two contractions' constants and scratch hoisted to build
                time (items 1, 3 and 4), with GPU4PySCF's launch geometry;
* ``blk``    -- ``opt`` plus the auxiliary-vector kernel's shell-pair blocks
                split (item 5);
* ``mdm``    -- the McMurchie--Davidson engine used for every density count
                (item 6) and nothing else.  Identical to ``stock`` at one
                density, which is the row that says how much of a
                multi-density margin is the engine and how much is the rest;
* ``fast``   -- all of them.

The variants are installed by swapping the bound methods rather than by
re-importing, so all six run **in one process, against one density, on one
set of option objects**, and are timed **round robin** (`docs/DF_WORK_LIST.md`
§5.2).  That removes the two traps this path has produced before: a cold first
process (§5.3) and a second tensor silently landing on the host (§5.10).

``--scf`` adds whole DF-PBE runs, which is the thing W9 is actually about: a
pure functional never builds a cderi, so `fastdf` reads zero calls on it and
this build is its entire two-electron cost.  Each variant is run twice and the
second is quoted.

    python benchmarks/df_j3c_bench.py --cases glycine,penicillinV,taxol
    python benchmarks/df_j3c_bench.py --cases PfPMT:-1 --reps 3
    python benchmarks/df_j3c_bench.py --cases penicillinV --scf --xc pbe
    python benchmarks/df_j3c_bench.py --cases penicillinV --ndm 1,2,4
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MOL = os.path.join(HERE, 'molecules')
sys.path.insert(0, os.path.dirname(HERE))

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--cases', default='glycine,alanine_dipeptide,penicillinV')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--xc', default='pbe')
ap.add_argument('--ndm', default='1')
ap.add_argument('--reps', type=int, default=8)
ap.add_argument('--cycles', type=int, default=1,
                help='SCF cycles to run before the fixed-density timing')
ap.add_argument('--variants', default='stock,ctr,opt,blk,mdm,fast')
ap.add_argument('--scf', action='store_true',
                help='also run whole SCFs, stock and fast, twice each')
ap.add_argument('--scf-max-cycle', type=int, default=50)
ap.add_argument('--patch', default='',
                help="other electrolites modules to patch underneath, so the "
                     "ratio is measured against a patched baseline")
a = ap.parse_args()

import electrolites                                           # noqa: E402
if a.patch:
    electrolites.patch(a.patch.split(','))
# imported for its originals and its replacements; the variants below install
# one set or the other, so importing it does not decide the measurement
from electrolites import fastdfj3c                            # noqa: E402

import cupy as cp                                             # noqa: E402
from pyscf import gto                                         # noqa: E402
from gpu4pyscf import dft                                      # noqa: E402
from gpu4pyscf.df import df_jk, j_engine_3c2e                  # noqa: E402
from gpu4pyscf.gto.mole import SortedGTO                       # noqa: E402

_sync = cp.cuda.Stream.null.synchronize

_OPT = j_engine_3c2e.Int3c2eOpt
_CTR_NAMES = ('CT_dot_mat', 'C_dot_mat', 'mat_dot_C', 'mat_dot_CT')
_STOCK_CTR = {n: getattr(fastdfj3c, '_orig_' + n) for n in _CTR_NAMES}
_FAST_CTR = {n: getattr(fastdfj3c, n) for n in _CTR_NAMES}
_STOCK_OPT = {'contract_dm': fastdfj3c._orig_contract_dm,
              'contract_auxvec': fastdfj3c._orig_contract_auxvec}
_FAST_OPT = {'contract_dm': fastdfj3c.contract_dm,
             'contract_auxvec': fastdfj3c.contract_auxvec}

#: The shipped auxiliary-vector block size, so `blk` and `fast` measure what
#: a user gets rather than what this script chose.
_CHUNK = fastdfj3c.SP_CHUNK_AUX

#: variant -> (recontraction tables cached, contraction constants hoisted,
#:             auxiliary-vector shell pairs per block, MD engine for every
#:             density count)
VARIANTS = {
    'stock': (False, False, 0, False),
    'ctr': (True, False, 0, False),
    'opt': (False, True, 0, False),
    'blk': (False, True, _CHUNK, False),
    'mdm': (False, False, 0, True),
    'fast': (True, True, _CHUNK, True),
}


def install(variant):
    """Bind one variant's methods.  Nothing else in the process changes."""
    ctr, opt, chunk, mdm = VARIANTS[variant]
    for n in _CTR_NAMES:
        setattr(SortedGTO, n, (_FAST_CTR if ctr else _STOCK_CTR)[n])
    for n, f in (_FAST_OPT if opt else _STOCK_OPT).items():
        setattr(_OPT, n, f)
    fastdfj3c.SP_CHUNK_AUX = chunk
    df_jk.get_j = fastdfj3c.get_j if mdm else fastdfj3c._orig_get_j


def load(spec):
    name, _, rest = spec.partition(':')
    charge, _, spin = rest.partition(':')
    mol = gto.M(atom=os.path.join(MOL, name + '.xyz'), basis=a.basis,
                charge=int(charge or 0), spin=int(spin or 0),
                verbose=0, max_memory=80000)
    return name, mol


def mk(mol):
    # an open-shell case has to be UKS, and its density is two matrices --
    # `docs/DF_WORK_LIST.md`'s generalisation rule asks for a different
    # chemical class than the organics, and every transition-metal case in
    # `benchmarks/molecules` is open shell
    cls = dft.UKS if mol.spin else dft.RKS
    mf = cls(mol, xc=a.xc).density_fit(auxbasis=a.auxbasis)
    mf.grids.level = 3
    mf.verbose = 0
    mf.conv_tol = 1e-9
    return mf


def per_call(spec, variants, ndms):
    name, mol = load(spec)
    install('stock')
    mf = mk(mol)
    if a.cycles:
        mf.max_cycle = a.cycles
        mf.kernel()
        dm0 = cp.asarray(mf.make_rdm1())
    else:
        dm0 = cp.asarray(mf.get_init_guess())
    dfobj = mf.with_df
    if dfobj.intopt is None:
        dfobj.build(build_cderi=False)
    print(f'\n## {name}  {mol.natm} atoms  nao {mol.nao}  '
          f'auxbasis nao {dfobj.auxmol.nao}', flush=True)

    nao = dm0.shape[-1]
    dm0 = dm0.reshape(-1, nao, nao)
    for ndm in ndms:
        dm = (dm0 if ndm == 1 else
              cp.repeat(dm0, ndm, axis=0)).reshape(-1, nao, nao)
        if len(dm) == 1:
            dm = dm[0]
        ref = None
        t = {v: [] for v in variants}
        # one warm pass per variant, so no row pays a first-touch allocation
        for v in variants:
            install(v)
            out = df_jk.get_j(dfobj, dm, 1)
            _sync()
            if ref is None:
                ref = cp.asnumpy(out)
                err = 0.
            else:
                err = float(abs(cp.asnumpy(out) - ref).max() / abs(ref).max())
            t[v].append(err)
        errs = {v: t[v].pop() for v in variants}

        for _ in range(a.reps):
            for v in variants:                      # round robin, §5.2
                install(v)
                _sync()
                t0 = time.perf_counter()
                df_jk.get_j(dfobj, dm, 1)
                _sync()
                t[v].append(time.perf_counter() - t0)

        base = min(t[variants[0]])
        print(f'\n  get_j, ndm = {ndm}, min of {a.reps}')
        print(f'  {"variant":10s} {"ms":>9s} {"x stock":>9s} {"rel err":>10s}')
        for v in variants:
            ms = min(t[v]) * 1e3
            print(f'  {v:10s} {ms:9.3f} {base/min(t[v]):9.3f} '
                  f'{errs[v]:10.1e}')
        sys.stdout.flush()
    install('stock')


def whole_scf(spec):
    name, mol = load(spec)
    print(f'\n## {name}  whole {a.xc} SCF, density-fitted', flush=True)
    out = {}
    for rep in range(2):
        for v in ('stock', 'fast'):
            install(v)
            mf = mk(mol)
            mf.max_cycle = a.scf_max_cycle
            _sync()
            t0 = time.perf_counter()
            e = mf.kernel()
            _sync()
            dt = time.perf_counter() - t0
            out[v] = (dt, e, mf.cycles if hasattr(mf, 'cycles') else -1)
            print(f'  rep {rep}  {v:6s} {dt:8.3f} s   E = {e:.10f}',
                  flush=True)
            mf = None
            cp.get_default_memory_pool().free_all_blocks()
    print(f'  {name}: stock {out["stock"][0]:.3f} s, '
          f'fast {out["fast"][0]:.3f} s, '
          f'{out["stock"][0]/out["fast"][0]:.3f}x, '
          f'dE = {abs(out["stock"][1]-out["fast"][1]):.2e} Eh', flush=True)
    install('stock')


def main():
    variants = [v for v in a.variants.split(',') if v in VARIANTS]
    ndms = [int(x) for x in a.ndm.split(',')]
    print(f'# gpu4pyscf {electrolites.gpu4pyscf_version()}  basis {a.basis}  '
          f'xc {a.xc}  also patched: {electrolites.patched() or "(none)"}')
    for spec in a.cases.split(','):
        per_call(spec, variants, ndms)
        if a.scf:
            whole_scf(spec)


main()
