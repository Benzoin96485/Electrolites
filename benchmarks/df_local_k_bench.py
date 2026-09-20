"""W6's acceptance test: a robust pair-atomic build against the global one.

`DF_WORK_LIST.md` **W6** wants the pair-atomic path measured on **energies**,
not only on wall clocks, and `docs/README_dflocal.md` did that at 5 to 14
atoms with an ``nao^4`` reference.  `electrolites/dflocal.py` is an
implementation that is not ``nao^4``, so this script carries the same question
up the §3.2 ladder and times the thing at the same time.

Three tables, all round robin so that a competing job's drift does not land on
one variant (`DF_WORK_LIST.md` §5.2), minimum of ``--reps``:

    build     the one-time tensor.  Global DF's is `_cholesky_eri`: three-
              centre integrals, a ``naux^3`` Cholesky of the global metric and
              a ``2 npair naux_sorted naux`` metric contraction, which
              `docs/README_fastdfcderi.md` §2 measures at **59.5 %** of a
              284-atom block loop.  PARI's is the same integrals, one small
              factorisation per *atom pair*, and a fold that is
              ``npair nlocal naux``.
    get_jk    the recurring build.  Global DF is two contractions of
              ``2 nao^2 nocc naux``; PARI is one of ``2 nao^2 naux nlambda``
              (`benchmarks/df_local_k_probe.py` measures the ratio those two
              predict).
    scf       ``--scf``: both to convergence, and the energy difference per
              atom, which is the number W6 asks for and the reason this
              script exists rather than a kernel benchmark.

Past about a hundred atoms the two tensors are the same size and will not
both fit on the card, and the loser goes to the **host** with nothing said --
which is a 6.4x measurement of a 1.2x difference.  So ``--only global|pari``
runs one side per process and ``--dm-cache`` makes the two processes drive
exactly the same converged density.  ``--anatomy`` prints where one PARI
build and one PARI pass spend their time, with a synchronising timer between
stages.

    python benchmarks/df_local_k_bench.py --cases penicillinV:0:0
    python benchmarks/df_local_k_bench.py --cases taxol:0:0 --only pari \
        --dm-cache $SCRATCH/dm --anatomy
    python benchmarks/df_local_k_bench.py --no-jk --scf --alkanes 1,2,4,8,16,20
"""
import argparse
import json
import os
import time

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--cases', default='glycine:0:0,alanine_dipeptide:0:0,'
                                   'penicillinV:0:0')
ap.add_argument('--domain', default='pair')
ap.add_argument('--lambda-tol', type=float, default=0.0)
ap.add_argument('--store', default='auto', choices=('auto', 'gpu', 'cpu'))
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--patch', default='',
                help="'all' to put global DF on `fastdf`+`fastdfcderi`, "
                     "'' for stock GPU4PySCF")
ap.add_argument('--scf', action='store_true',
                help='also run both to convergence and compare energies')
ap.add_argument('--xc', default='', help='empty for HF, else a functional')
ap.add_argument('--conv-tol', type=float, default=1e-10)
ap.add_argument('--no-jk', action='store_true', help='only the SCF table')
ap.add_argument('--only', default='both', choices=('both', 'global', 'pari'),
                help='time one side only.  Past about a hundred atoms the two '
                     'tensors are the same size and do not both fit on the '
                     'card, so holding both alive puts one of them on the '
                     'host and the comparison stops being one -- see '
                     '`docs/README_dflocal_k.md` §5.  With `--dm-cache` the '
                     'two processes drive exactly the same density')
ap.add_argument('--anatomy', action='store_true',
                help='also print where one PARI get_jk pass spends its time, '
                     'with a synchronising timer between stages')
ap.add_argument('--dm-cache', default='',
                help='converge once, save the density here, and load it in '
                     'every later process')
ap.add_argument('--alkanes', default='',
                help='instead of --cases, an idealised all-anti alkane '
                     'series: "1,2,4,8,16" is CH4 to C16H34.  This is the '
                     'size-extensivity test `docs/README_dflocal.md` §2.2 '
                     'could only run to C4H10, because its reference builds '
                     'an nao^4 tensor and this does not')
a = ap.parse_args()

if a.patch:
    import electrolites
    electrolites.patch_all() if a.patch == 'all' else \
        electrolites.patch(a.patch.split(','))

import cupy as cp                                            # noqa: E402
from pyscf import gto                                        # noqa: E402
from _common import alkane                                   # noqa: E402
from gpu4pyscf import scf as gscf, dft as gdft               # noqa: E402
import electrolites.dflocal as dl                            # noqa: E402


def timeit(fn, reps):
    fn()
    cp.cuda.Stream.null.synchronize()
    ts = []
    for _ in range(reps):
        cp.cuda.Stream.null.synchronize()
        t = time.perf_counter()
        out = fn()
        cp.cuda.Stream.null.synchronize()
        ts.append(time.perf_counter() - t)
    return min(ts), out


def make_mf(mol):
    mf = (gdft.RKS(mol, xc=a.xc) if a.xc else gscf.RHF(mol))
    mf.verbose = 0
    mf.conv_tol = a.conv_tol
    if a.xc:
        mf.grids.level = 3
    return mf


CASES = [(c.split(':') + ['0', '0'])[:3] for c in a.cases.split(',')] \
    if not a.alkanes else \
    [(f'c{n}', '0', '0') for n in a.alkanes.split(',')]
GEOM = {} if not a.alkanes else \
    {f'c{n}': alkane(int(n)) for n in a.alkanes.split(',')}

rows = []
if not a.no_jk:
    print(f'{"molecule":20s} {"natm":>5s} {"nao":>5s} {"nlocal":>7s} '
          f'{"nlambda":>8s} {"build g":>9s} {"build p":>9s} {"jk g":>8s} '
          f'{"jk p":>8s} {"ratio":>6s} {"relK":>9s}')
for name, spin, charge in CASES:
    mol = gto.M(atom=GEOM.get(name, f'benchmarks/molecules/{name}.xyz'),
                basis=a.basis, spin=int(spin), charge=int(charge), verbose=0,
                max_memory=200000)
    row = dict(name=name, natm=int(mol.natm), nao=int(mol.nao))

    if not a.no_jk:
        # a converged density, because the fixed-density speedup on a `minao`
        # guess is not the one the SCF sees -- docs/ROUND2_1.8.1.md §2
        cache = f'{a.dm_cache}.{name}.npy' if a.dm_cache else ''
        if cache and os.path.exists(cache):
            dm = cp.asarray(np.load(cache))
        else:
            mf = make_mf(mol).density_fit(auxbasis=a.auxbasis)
            mf.kernel()
            dm = cp.asarray(mf.make_rdm1())
            if cache:
                np.save(cache, cp.asnumpy(dm))
            del mf
            cp.get_default_memory_pool().free_all_blocks()

        tb = {}
        tj = {}
        vk = {}
        vj = {}
        obj = {}
        if a.only in ('both', 'global'):
            obj['g'] = make_mf(mol).density_fit(auxbasis=a.auxbasis).with_df
        if a.only in ('both', 'pari'):
            obj['p'] = dl.PARIDF(mol, a.auxbasis, domain=a.domain,
                                 store=a.store, lambda_tol=a.lambda_tol)

        def rebuild(k):
            if k == 'g':
                obj[k].reset()
            else:
                obj[k].M = None
            return obj[k].build()

        for _ in range(a.reps):                     # round robin
            for k in obj:
                t = time.perf_counter(); rebuild(k)
                cp.cuda.Stream.null.synchronize()
                tb[k] = min(tb.get(k, 1e9), time.perf_counter() - t)
        for _ in range(a.reps):
            for k in obj:
                t = time.perf_counter(); vj[k], vk[k] = obj[k].get_jk(dm)
                cp.cuda.Stream.null.synchronize()
                tj[k] = min(tj.get(k, 1e9), time.perf_counter() - t)

        relk = relj = float('nan')
        if len(obj) == 2:
            scale = float(cp.abs(vk['g']).max())
            relk = float(cp.abs(vk['p'] - vk['g']).max()) / scale
            relj = float(cp.abs(vj['p'] - vj['g']).max()) / \
                float(cp.abs(vj['g']).max())
        st = dl.stats()
        g, p = tb.get('g', float('nan')), tb.get('p', float('nan'))
        jg, jp = tj.get('g', float('nan')), tj.get('p', float('nan'))
        row.update(build_global=g, build_pari=p, jk_global=jg, jk_pari=jp,
                   relk=relk, relj=relj, nlocal=st['nlocal'],
                   nlambda=st['nlambda'], store=st['store'],
                   kmode=st['kmode'])
        print(f'{name:20s} {mol.natm:5d} {mol.nao:5d} {st["nlocal"]:7.0f} '
              f'{st["nlambda"]:8.0f} {g:9.3f} {p:9.3f} {jg:8.3f} '
              f'{jp:8.3f} {jg/jp:5.2f}x {relk:9.2e}  {st["store"]}/'
              f'{st["kmode"]}')
        if a.anatomy and 'p' in obj:
            dl.STAGE.clear()
            obj['p'].profile = True
            obj['p'].M = None
            obj['p'].build()
            obj['p'].get_jk(dm)
            cp.cuda.Stream.null.synchronize()
            tot = sum(dl.STAGE.values())
            print('#   ' + name + ' PARI get_jk stages: ' + '  '.join(
                f'{k} {v*1e3:.0f} ms ({v/tot:.0%})'
                for k, v in dl.STAGE.items()) + f'  total {tot:.3f} s')
            row['stages'] = dict(dl.STAGE)
            obj['p'].profile = False
        del obj, dm, vj, vk
        cp.get_default_memory_pool().free_all_blocks()

    if a.scf:
        e_g = t_g = float('nan')
        if a.only in ('both', 'global'):
            mf = make_mf(mol).density_fit(auxbasis=a.auxbasis)
            t = time.perf_counter(); e_g = mf.kernel()
            t_g = time.perf_counter() - t
            del mf
            cp.get_default_memory_pool().free_all_blocks()
        mf = None
        e_p = t_p = float('nan')
        if a.only in ('both', 'pari'):
            mf = dl.density_fit(make_mf(mol), auxbasis=a.auxbasis,
                                domain=a.domain, store=a.store)
            mf.with_df.lambda_tol = a.lambda_tol
            t = time.perf_counter(); e_p = mf.kernel()
            t_p = time.perf_counter() - t
        row.update(e_global=float(e_g), e_pari=float(e_p),
                   scf_global=t_g, scf_pari=t_p)
        nheavy = sum(1 for z in mol.atom_charges() if z > 1)
        print(f'#   {name}: natm {mol.natm} nao {mol.nao}  SCF  '
              f'global {e_g:.9f} ({t_g:.1f} s)   PARI {e_p:.9f} ({t_p:.1f} s)'
              f'   dE {e_p-e_g:+.3e}   per atom {(e_p-e_g)/mol.natm:+.2e}   '
              f'per heavy atom {(e_p-e_g)/nheavy:+.2e}')
        row['nheavy'] = int(nheavy)
        del mf
        cp.get_default_memory_pool().free_all_blocks()
    rows.append(row)

print('RESULT ' + json.dumps(dict(basis=a.basis, xc=a.xc, domain=a.domain,
                                  patch=a.patch, rows=rows)))
