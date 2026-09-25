"""The launch configuration of `gen_j3c.py`'s kernels, class by class.

``docs/DF_WORK_LIST.md`` **W19**.  Each ``(lij, lk)`` class of the three-centre
Coulomb build is a device function with its own auxiliary shells per block
(``kb``), and classes of similar register need share a *tier* kernel with one
thread count and one ``__launch_bounds__`` occupancy target.  All of those are
compile-time constants, so a configuration is a module: this compiles one per
``(kb, minblocks)`` pair -- the same pair for every class and tier -- times
every class alone inside its tier's kernel, and chooses per tier the
``minblocks`` whose classes' best times sum lowest and per class the ``kb``
that is best at it.  Then the whole build, every tier on the stream pool, with
the defaults and with the choice, against GPU4PySCF's kernels.

Every configuration's whole-build output is compared element-wise with
GPU4PySCF's.

    python benchmarks/df_j3c_kernel_sweep.py --cases penicillinV,taxol \\
        --json $SCRATCH/j3c_sweep.json
"""
import argparse
import ctypes
import itertools
import json
import os
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
MOL = os.path.join(HERE, 'molecules')
sys.path.insert(0, os.path.dirname(HERE))

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--cases', default='penicillinV')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--reps', type=int, default=5)
ap.add_argument('--cycles', type=int, default=2)
ap.add_argument('--max-order', type=int, default=None)
ap.add_argument('--dm-kb', default='1,2,4')
ap.add_argument('--aux-kb', default='8,16,32')
ap.add_argument('--minb', default='1,2,4')
ap.add_argument('--threads', default='128',
                help='threads per block of every tier; more than one value '
                     'is swept at the default kb and minblocks')
ap.add_argument('--ppb', default='512,1024,2048,4096',
                help='shell pairs per block of the density kernel, swept at '
                     'the best compiled configuration')
ap.add_argument('--json', default='')
a = ap.parse_args()

import cupy as cp                                             # noqa: E402
from pyscf import gto                                         # noqa: E402
from gpu4pyscf import dft                                      # noqa: E402
from gpu4pyscf.df import df_jk                                 # noqa: E402
from gpu4pyscf.df import j_engine_3c2e as J3C                  # noqa: E402

from electrolites import _j3c, _j3ck                           # noqa: E402
from electrolites.codegen import gen_j3c                       # noqa: E402

_sync = cp.cuda.Stream.null.synchronize
lib = J3C.libvhf_md


def grid(spec):
    axes = [[int(x) for x in part.split(',')] for part in spec.split(':')]
    return list(itertools.product(*axes))


DM_KB = [int(x) for x in a.dm_kb.split(',')]
AUX_KB = [int(x) for x in a.aux_kb.split(',')]
MINB = [int(x) for x in a.minb.split(',')]
THREADS = [int(x) for x in a.threads.split(',')]


def ptr(x):
    return ctypes.cast(x.data.ptr, ctypes.c_void_p)


def tmin(fn, reps):
    fn()
    _sync()
    ts = []
    for _ in range(reps):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def load(spec):
    name, _, rest = spec.partition(':')
    charge, _, spin = rest.partition(':')
    mol = gto.M(atom=os.path.join(MOL, name + '.xyz'), basis=a.basis,
                charge=int(charge or 0), spin=int(spin or 0),
                verbose=0, max_memory=80000)
    return name, mol


def run(spec):
    name, mol = load(spec)
    mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc='pbe').density_fit(
        auxbasis=a.auxbasis)
    mf.grids.level = 3
    mf.verbose = 0
    mf.max_cycle = a.cycles
    mf.kernel()
    dm = cp.asarray(mf.make_rdm1())
    dm = dm.reshape(-1, mol.nao, mol.nao).sum(0)
    dfobj = mf.with_df
    df_jk.get_j(dfobj, dm, 1)
    opt = dfobj.j_engine
    c = _j3c.constants(opt)
    dm_t = opt.mol.apply_C_mat_CT(dm)
    dm_xyz = cp.empty(c.dm_xyz_size)
    lib.dm_to_Rt(ptr(dm_xyz), ptr(dm_t), ctypes.c_int(1),
                 ctypes.byref(c.envs), ptr(c.shl_pair_idx), ptr(c.pair_loc),
                 ctypes.c_int(c.npair), ptr(c.ao_loc))
    rho = opt.contract_dm(dm_t, 1)
    rho = opt.auxmol.apply_C_dot(opt.auxmol.apply_CT_dot(rho, axis=-1),
                                 axis=-1)
    aux_xyz = cp.empty(c.aux_xyz_size)
    lib.aux_to_Rt(ptr(aux_xyz), ptr(rho), ctypes.byref(c.envs),
                  ptr(c.aux_loc), ptr(c.aux_xyz_loc), ctypes.c_int(c.auxnbas))

    # GPU4PySCF's kernels over the whole basis, the reference and the baseline
    nsp_dm, shm_dm = J3C._int3c2e_dm_scheme()
    nsp_aux, shm_aux = J3C._int3c2e_auxvec_scheme()
    off0, n0 = c.pair_offsets(0)
    offa, na = c.pair_offsets(c.auto_chunk(c.ksh_blocks, 4000, 256, 8192))
    ko, nk = c.ksh_batches(16)

    def up_dm(out):
        lib.contract_int3c2e_dm(
            ptr(out), ptr(dm_xyz), ctypes.c_int(1), ctypes.c_int(c.naux_cart),
            ctypes.byref(c.envs), ctypes.c_int(shm_dm), ctypes.c_int(n0),
            ctypes.c_int(c.auxnbas), ptr(off0), ptr(c.shl_pair_idx),
            ptr(c.pair_loc), ptr(nsp_dm), ctypes.c_double(0.))

    def up_aux(out):
        lib.contract_int3c2e_auxvec(
            ptr(out), ptr(aux_xyz), ctypes.c_int(1), ctypes.c_int(c.naux_sph),
            ctypes.byref(c.envs), ctypes.c_int(shm_aux), ctypes.c_int(na),
            ctypes.c_int(nk), ptr(offa), ptr(ko), ptr(c.shl_pair_idx),
            ptr(c.pair_loc), ptr(c.aux_xyz_loc), ptr(nsp_aux),
            ctypes.c_double(0.))

    ref_dm = cp.zeros(c.naux_cart)
    up_dm(ref_dm)
    ref_aux = cp.zeros(c.dm_xyz_size)
    up_aux(ref_aux)
    t_up_dm = tmin(lambda: up_dm(cp.zeros(c.naux_cart)), a.reps)
    t_up_aux = tmin(lambda: up_aux(cp.zeros(c.dm_xyz_size)), a.reps)

    P = _j3ck.Plan(opt, c, max_order=a.max_order or 99)
    print(f'\n## {name}  {mol.natm} atoms  nao {mol.nao}  {c.npair} pairs  '
          f'{c.auxnbas} auxiliary shells  classes {len(P.classes)}  '
          f'left to GPU4PySCF: {P.theirs}', flush=True)
    print(f'  GPU4PySCF: contract_dm {t_up_dm * 1e3:.3f} ms, contract_auxvec '
          f'(W9 split) {t_up_aux * 1e3:.3f} ms')

    ntiers = len(gen_j3c.TIERS)
    classes = list(P.classes)
    tiers_of = {cl: gen_j3c.tier_of(sum(cl)) for cl in classes}
    T = {'dm': {}, 'aux': {}}
    errs = []
    nkb = max(len(DM_KB), len(AUX_KB))
    variants = [(DM_KB[i % len(DM_KB)], AUX_KB[i % len(AUX_KB)], mb, THREADS[0])
                for i in range(nkb) for mb in MINB]
    variants += [(None, None, None, nt) for nt in THREADS[1:]]
    for v, (dkb, akb, mb, nt) in enumerate(variants):
        kb = tuple(sorted(
            ([(('dm',) + cl, dkb) for cl in classes] if dkb else [])
            + ([(('aux',) + cl, akb) for cl in classes] if akb else [])))
        tiers = tuple(sorted(
            ((kind, t), (nt, mb if mb else gen_j3c.TIER_CFG[kind][t][1]))
            for kind in ('dm', 'aux') for t in range(ntiers)))
        P.kb, P.tiers = kb, tiers
        t0 = time.perf_counter()
        mod = P.module()
        mod.get_function(gen_j3c.tier_name('dm', tiers_of[classes[0]]))
        tc = time.perf_counter() - t0
        rho_o = cp.zeros(c.naux_cart)
        vj_o = cp.zeros(c.dm_xyz_size)
        P.contract_dm(dm_xyz, rho_o)
        P.contract_auxvec(aux_xyz, vj_o)
        _sync()
        if not P.theirs:
            errs.append((float(abs(rho_o - ref_dm).max() / abs(ref_dm).max()),
                         float(abs(vj_o - ref_aux).max()
                               / abs(ref_aux).max())))
        for cl in classes:
            out = cp.zeros(c.naux_cart)
            T['dm'][cl, v] = tmin(lambda: P.contract_dm(
                dm_xyz, out, only=cl), a.reps)
            out = cp.zeros(c.dm_xyz_size)
            T['aux'][cl, v] = tmin(lambda: P.contract_auxvec(
                aux_xyz, out, only=cl), a.reps)
        print(f'  cfg {v:2d}: dm kb {dkb} aux kb {akb} minblocks {mb} '
              f'threads {nt}  compiled in {tc:.1f} s', flush=True)
    if errs:
        print(f'  worst relative difference against GPU4PySCF over every '
              f'configuration: dm {max(e[0] for e in errs):.1e}, auxvec '
              f'{max(e[1] for e in errs):.1e}')

    # per tier the minblocks, per class the kb
    kvar = [v for v, x in enumerate(variants) if x[0] is not None]
    choice_kb, choice_tier = {}, {}
    for kind, kbs in (('dm', DM_KB), ('aux', AUX_KB)):
        for t in range(ntiers):
            members = [cl for cl in classes if tiers_of[cl] == t]
            if not members:
                continue
            best_mb, best_sum = None, None
            for mb in MINB:
                vs = [v for v in kvar if variants[v][2] == mb]
                tot = sum(min(T[kind][cl, v] for v in vs) for cl in members)
                if best_sum is None or tot < best_sum:
                    best_mb, best_sum = mb, tot
            choice_tier[(kind, t)] = (THREADS[0], best_mb)
            vs = [v for v in kvar if variants[v][2] == best_mb]
            for cl in members:
                vb = min(vs, key=lambda v: T[kind][cl, v])
                choice_kb[(kind,) + cl] = (variants[vb][0] if kind == 'dm'
                                           else variants[vb][1])
    print(f'  {"lij":>3s} {"lk":>3s} {"tier":>4s} {"dm kb":>5s} {"dm ms":>8s} '
          f'{"dflt x":>7s}   {"aux kb":>6s} {"aux ms":>8s} {"dflt x":>7s}')

    def t_of(kind, cl, kbv, mb):
        for v in kvar:
            dkb, akb, m, _ = variants[v]
            if m == mb and (dkb if kind == 'dm' else akb) == kbv:
                return T[kind][cl, v]
        return float('nan')
    for cl in classes:
        t = tiers_of[cl]
        cells = []
        for kind in ('dm', 'aux'):
            kbv = choice_kb[(kind,) + cl]
            mb = choice_tier[(kind, t)][1]
            tb = t_of(kind, cl, kbv, mb)
            td = t_of(kind, cl, gen_j3c.default_kb(kind, *cl),
                      gen_j3c.TIER_CFG[kind][t][1])
            cells.append((kbv, tb, td / tb))
        print(f'  {cl[0]:3d} {cl[1]:3d} {t:4d} {cells[0][0]:5d} '
              f'{cells[0][1] * 1e3:8.3f} {cells[0][2]:7.3f}   '
              f'{cells[1][0]:6d} {cells[1][1] * 1e3:8.3f} {cells[1][2]:7.3f}')
    print('  tiers: ' + '  '.join(f'{k[0]}{k[1]}={v[1]}'
                                  for k, v in sorted(choice_tier.items())))
    for v, (dkb, akb, mb, nt) in enumerate(variants):
        if dkb is None:
            print(f'  threads {nt} at the defaults: dm '
                  f'{sum(T["dm"][cl, v] for cl in classes) * 1e3:.3f} ms, '
                  f'aux {sum(T["aux"][cl, v] for cl in classes) * 1e3:.3f} ms'
                  ' summed over classes')

    res = {}
    for label, kb, tiers in (
            ('default', (), ()),
            ('chosen', tuple(sorted(choice_kb.items())),
             tuple(sorted(choice_tier.items())))):
        P.kb, P.tiers = kb, tiers
        rho_o = cp.zeros(c.naux_cart)
        vj_o = cp.zeros(c.dm_xyz_size)
        res[label] = (tmin(lambda: P.contract_dm(dm_xyz, rho_o), a.reps),
                      tmin(lambda: P.contract_auxvec(aux_xyz, vj_o), a.reps))
    ppbs = {}
    for ppb in (int(x) for x in a.ppb.split(',')):
        ppbs[ppb] = tmin(lambda: P.contract_dm(
            dm_xyz, cp.zeros(c.naux_cart), ppb=ppb), a.reps)
    print(f'  whole build, every tier on {_j3ck.NSTREAMS} streams:')
    for label, (td, ta) in res.items():
        print(f'    {label:8s} contract_dm {td * 1e3:8.3f} ms '
              f'({t_up_dm / td:5.2f}x)   contract_auxvec {ta * 1e3:8.3f} ms '
              f'({t_up_aux / ta:5.2f}x)')
    print('    density kernel, pairs per block at the chosen cfg: '
          + '  '.join(f'{k}: {v * 1e3:.3f} ms' for k, v in ppbs.items()))
    best = {**{k: v for k, v in choice_kb.items()},
            **{k: v for k, v in choice_tier.items()}}
    tdm, taux = T['dm'], T['aux']
    sys.stdout.flush()
    return {'case': spec, 'natm': mol.natm, 'basis': a.basis,
            'upstream': [t_up_dm, t_up_aux],
            'whole': {k: list(v) for k, v in res.items()},
            'ppb': ppbs,
            'best': {'_'.join(map(str, k)): v for k, v in best.items()},
            'variants': variants,
            'dm': {f'{k[0][0]}_{k[0][1]}_{k[1]}': t for k, t in tdm.items()},
            'aux': {f'{k[0][0]}_{k[0][1]}_{k[1]}': t
                    for k, t in taux.items()}}


out = [run(s) for s in a.cases.split(',')]
if a.json:
    with open(a.json, 'w') as f:
        json.dump(out, f)
print('@@JSON ' + json.dumps([{k: v for k, v in o.items()
                               if k in ('case', 'upstream', 'whole', 'best')}
                              for o in out]))
