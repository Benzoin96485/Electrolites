"""Where the three-centre Coulomb kernels spend their time, class by class.

``docs/DF_WORK_LIST.md`` **W19**.  After W9 the Coulomb build of a pure
density-fitted functional is two integral kernels, and W19's falsification
criterion is to profile *one angular-momentum class* before writing a
generator for any of them.  GPU4PySCF launches each kernel once for the whole
basis, so a class has no time of its own until it is given one: this launches
GPU4PySCF's own kernels restricted to one class at a time, with the same
arguments ``fastdfj3c`` gives them.

* ``contract_int3c2e_dm`` is restricted to one shell-pair group by its block
  offsets.  Its grid covers the auxiliary shells ``[nbas, nbas + nksh)`` and
  the auxiliary shells are sorted by angular momentum, so a launch over the
  first ``nksh`` shells is a launch over the lowest classes; the cost of one
  auxiliary class is the difference of two such prefixes.
* ``contract_int3c2e_auxvec`` takes both block lists as arguments and is
  restricted to one ``(shell-pair group, auxiliary group)`` directly.

Each class's time is reported with its quartet count (shell pairs x auxiliary
shells) and the time per quartet, which is the number a kernel rewrite moves.
With ``--ours`` the same table is taken for ``electrolites/_j3ck.py``'s
kernels, class for class, and each class's auxiliary-vector output is
compared element-wise with GPU4PySCF's.

    python benchmarks/df_j3c_class_probe.py --cases penicillinV,taxol
"""
import argparse
import ctypes
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MOL = os.path.join(HERE, 'molecules')
sys.path.insert(0, os.path.dirname(HERE))

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--cases', default='penicillinV')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--reps', type=int, default=5)
ap.add_argument('--cycles', type=int, default=2)
ap.add_argument('--ours', action='store_true',
                help="also time electrolites/_j3ck.py's kernels")
ap.add_argument('--max-order', type=int, default=99,
                help='the ceiling --ours generates classes up to')
ap.add_argument('--whole', action='store_true',
                help='also time each kernel over the whole basis')
a = ap.parse_args()

import cupy as cp                                             # noqa: E402
from pyscf import gto                                         # noqa: E402
from gpu4pyscf import dft                                      # noqa: E402
from gpu4pyscf.df import df_jk                                 # noqa: E402
from gpu4pyscf.df import j_engine_3c2e as J3C                  # noqa: E402

from electrolites import _j3c                                  # noqa: E402

_sync = cp.cuda.Stream.null.synchronize
lib = J3C.libvhf_md


def load(spec):
    name, _, rest = spec.partition(':')
    charge, _, spin = rest.partition(':')
    mol = gto.M(atom=os.path.join(MOL, name + '.xyz'), basis=a.basis,
                charge=int(charge or 0), spin=int(spin or 0),
                verbose=0, max_memory=80000)
    return name, mol


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


def nf3(l):
    return (l + 1) * (l + 2) * (l + 3) // 6


def run(spec):
    name, mol = load(spec)
    cls = dft.UKS if mol.spin else dft.RKS
    mf = cls(mol, xc='pbe').density_fit(auxbasis=a.auxbasis)
    mf.grids.level = 3
    mf.verbose = 0
    mf.max_cycle = a.cycles
    mf.kernel()
    dm = cp.asarray(mf.make_rdm1())
    dm = dm.reshape(-1, dm.shape[-1], dm.shape[-1]).sum(0)
    dfobj = mf.with_df
    df_jk.get_j(dfobj, dm, 1)
    opt = dfobj.j_engine
    c = _j3c.constants(opt)
    smol, saux = opt.mol, opt.auxmol
    omega = float(smol.omega)

    dm_t = smol.apply_C_mat_CT(dm)
    dm_xyz = cp.empty(c.dm_xyz_size)
    lib.dm_to_Rt(ptr(dm_xyz), ptr(dm_t), ctypes.c_int(1),
                 ctypes.byref(c.envs), ptr(c.shl_pair_idx), ptr(c.pair_loc),
                 ctypes.c_int(c.npair), ptr(c.ao_loc))
    rho = opt.contract_dm(dm_t, 1)
    rho = saux.apply_C_dot(saux.apply_CT_dot(rho, axis=-1), axis=-1)
    aux_xyz = cp.empty(c.aux_xyz_size)
    lib.aux_to_Rt(ptr(aux_xyz), ptr(rho), ctypes.byref(c.envs),
                  ptr(c.aux_loc), ptr(c.aux_xyz_loc), ctypes.c_int(c.auxnbas))

    nsp_dm, shm_dm = J3C._int3c2e_dm_scheme()
    nsp_aux, shm_aux = J3C._int3c2e_auxvec_scheme()
    aux_l = np.asarray(saux._bas[:, 1])
    lks = sorted(set(aux_l.tolist()))
    aux_count = {lk: int((aux_l == lk).sum()) for lk in lks}
    # prefix of the auxiliary shells up to and including class lk
    prefix = {lk: int((aux_l <= lk).sum()) for lk in lks}

    off = c._sp_offsets_host
    groups = [(int(l), int(lo), int(hi)) for l, lo, hi
              in zip(c.group_lij, off[:-1], off[1:]) if hi > lo]
    # the pairs of one group can span several (li, lj) of the same li+lj;
    # the group is what the kernels see
    print(f'\n## {name}  {mol.natm} atoms  nao {mol.nao}  {c.npair} shell '
          f'pairs, groups (lij: pairs) '
          + ' '.join(f'{l}:{hi - lo}' for l, lo, hi in groups)
          + '  auxiliary (lk: shells) '
          + ' '.join(f'{lk}:{aux_count[lk]}' for lk in lks), flush=True)

    vj_aux = cp.zeros(c.naux_cart)
    vj_xyz = cp.zeros(c.dm_xyz_size)

    def dm_launch(lo, hi, nksh):
        offs = cp.asarray(np.array([lo, hi], dtype=np.int32))

        def go():
            lib.contract_int3c2e_dm(
                ptr(vj_aux), ptr(dm_xyz), ctypes.c_int(1),
                ctypes.c_int(c.naux_cart), ctypes.byref(c.envs),
                ctypes.c_int(shm_dm), ctypes.c_int(1), ctypes.c_int(nksh),
                ptr(offs), ptr(c.shl_pair_idx), ptr(c.pair_loc),
                ptr(nsp_dm), ctypes.c_double(omega))
        return go

    def aux_launch(lo, hi, lk):
        chunk = c.auto_chunk(c.ksh_blocks, 4000, 256, 8192)
        sp = (np.append(np.arange(lo, hi, chunk, dtype=np.int32),
                        np.int32(hi)))
        spd = cp.asarray(sp)
        k0 = c.nbas + int((aux_l < lk).sum())
        k1 = k0 + aux_count[lk]
        ko = np.append(np.arange(k0, k1, 16, dtype=np.int32), np.int32(k1))
        kod = cp.asarray(ko)

        def go():
            lib.contract_int3c2e_auxvec(
                ptr(vj_xyz), ptr(aux_xyz), ctypes.c_int(1),
                ctypes.c_int(c.naux_sph), ctypes.byref(c.envs),
                ctypes.c_int(shm_aux), ctypes.c_int(len(sp) - 1),
                ctypes.c_int(len(ko) - 1), ptr(spd), ptr(kod),
                ptr(c.shl_pair_idx), ptr(c.pair_loc), ptr(c.aux_xyz_loc),
                ptr(nsp_aux), ctypes.c_double(omega))
        return go

    rows = []
    tot_dm = tot_aux = 0.
    for lij, lo, hi in groups:
        npair = hi - lo
        prev = 0.
        for lk in lks:
            t_pre = tmin(dm_launch(lo, hi, prefix[lk]), a.reps)
            t_dm = max(t_pre - prev, 0.)
            prev = t_pre
            t_aux = tmin(aux_launch(lo, hi, lk), a.reps)
            q = npair * aux_count[lk]
            rows.append((lij, lk, npair, aux_count[lk], q, t_dm, t_aux))
            tot_dm += t_dm
            tot_aux += t_aux

    print(f'  {"lij":>3s} {"lk":>3s} {"order":>5s} {"pairs":>7s} '
          f'{"aux":>6s} {"quartets":>10s} {"dm ms":>8s} {"ns/q":>7s} '
          f'{"aux ms":>8s} {"ns/q":>7s} {"share":>6s}')
    for lij, lk, npair, nk, q, t_dm, t_aux in rows:
        print(f'  {lij:3d} {lk:3d} {lij + lk:5d} {npair:7d} {nk:6d} '
              f'{q:10.3e} {t_dm * 1e3:8.3f} {t_dm / q * 1e9:7.3f} '
              f'{t_aux * 1e3:8.3f} {t_aux / q * 1e9:7.3f} '
              f'{(t_dm + t_aux) / (tot_dm + tot_aux):6.1%}')
    print(f'  sum of classes: dm {tot_dm * 1e3:.2f} ms, auxvec '
          f'{tot_aux * 1e3:.2f} ms')
    if a.ours:
        from electrolites import _j3ck
        P = _j3ck.Plan(opt, c, max_order=a.max_order)
        up = {}
        for lij, lk, npair, nk, q, t_dm, t_aux in rows:
            u = up.setdefault((lij, lk), [0, nk, 0., 0.])
            u[0] += npair
            u[2] += t_dm
            u[3] += t_aux
        print(f'\n  this module\'s kernels, each class alone (ceiling '
              f'{a.max_order}), against the rows above summed per class')
        print(f'  {"lij":>3s} {"lk":>3s} {"order":>5s} {"dm ms":>8s} '
              f'{"x":>6s} {"aux ms":>8s} {"x":>6s}  {"rel err dm/aux":>16s}')
        tot = [0., 0., 0., 0.]
        for (lij, lk), (npair, nk, u_dm, u_aux) in sorted(up.items()):
            if (lij, lk) not in P.classes:
                continue
            vj = cp.zeros(c.dm_xyz_size)
            t_dm = tmin(lambda: P.contract_dm(dm_xyz, cp.zeros(c.naux_cart),
                                              only=(lij, lk)), a.reps)
            t_aux = tmin(lambda: P.contract_auxvec(
                aux_xyz, cp.zeros(c.dm_xyz_size), only=(lij, lk)), a.reps)
            # the class against GPU4PySCF's: its groups, its auxiliary shells
            P.contract_auxvec(aux_xyz, vj, only=(lij, lk))
            vj_xyz[:] = 0
            for glij, lo, hi in groups:
                if glij == lij:
                    aux_launch(lo, hi, lk)()
            refv = vj_xyz.copy()
            e_aux = float(abs(vj - refv).max() / max(abs(refv).max(), 1e-300))
            tot[0] += u_dm
            tot[1] += t_dm
            tot[2] += u_aux
            tot[3] += t_aux
            print(f'  {lij:3d} {lk:3d} {lij + lk:5d} {t_dm * 1e3:8.3f} '
                  f'{u_dm / t_dm:6.2f} {t_aux * 1e3:8.3f} {u_aux / t_aux:6.2f}'
                  f'  {"":>7s}{e_aux:9.1e}')
        print(f'  over these classes: dm {tot[0] * 1e3:.2f} -> '
              f'{tot[1] * 1e3:.2f} ms, auxvec {tot[2] * 1e3:.2f} -> '
              f'{tot[3] * 1e3:.2f} ms')
    if a.whole:
        n = len(off) - 1
        t_w = tmin(lambda: opt.contract_dm(dm_t, 1), a.reps)
        print(f'  whole contract_dm {t_w * 1e3:.2f} ms (upstream, {n} groups)')
    sys.stdout.flush()


for spec in a.cases.split(','):
    run(spec)
