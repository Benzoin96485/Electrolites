"""The density-fitted Hessian's ``ejk_int3c2e_ip2`` kernel, class by class.

``docs/DF_WORK_LIST.md`` **W29**.  After W24 the ip2 term of a
density-fitted Hessian (``df/hessian/rhf.py:_jk_energy_per_atom``, reached
through ``fastdfhessip._ejk_ip2``) is 46-52 % one kernel at 21-42 atoms.
GPU4PySCF launches it once per auxiliary batch over every shell pair, so a
``(li, lj, lk)`` class has no time of its own until it is given one: this
launches the kernel restricted to one shell-pair class (a slice of
``shl_pair_offsets``) and one auxiliary batch (one ``lk`` and contraction
pattern), with exactly the arguments ``_ejk_ip2`` gives it.

The kernel's work does not depend on the pseudo-density's values -- it
screens nothing -- so the probe feeds it a seeded random one, of the shape
and layout ``_ejk_ip2`` builds, and needs no SCF.

Per class it prints the time, the number of shell triples and primitive
triples, and the time per primitive triple and Rys root; ``--whole`` also
times the production launch (every pair class at once, one auxiliary batch
per launch), which is what the per-class sum is compared against.  With
``--ours`` the same table is taken for W31's kernel (``libmyip2``, what
``electrolites/_ip2launch.py`` launches; ``--lib`` another build of the same
entry point) and each class's output is compared element-wise with
GPU4PySCF's.

    python benchmarks/df_ip2_class_probe.py --cases penicillinV,tmc_ni_dithiolene:-1:1
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
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--omega', type=float, default=0.)
ap.add_argument('--ours', action='store_true',
                help="also time W31's kernel (electrolites/_ip2launch.kernel)")
ap.add_argument('--lib', default='',
                help='with --ours: a shared library exporting '
                     'ejk_int3c2e_ip2 to use as ours')
ap.add_argument('--shm', type=int, default=0,
                help='dynamic shared memory per block at least this many '
                     'bytes (an occupancy control for ablations: 90000 is one '
                     'block per SM)')
ap.add_argument('--whole', action='store_true',
                help='time the production launch (all pair classes per '
                     'auxiliary batch) as well')
ap.add_argument('--no-classes', action='store_true',
                help='skip the per-class table')
ap.add_argument('--only', default='',
                help="restrict to classes 'li,lj,lk;...' (for ncu)")
ap.add_argument('--kchunk', type=int, default=0,
                help='split every auxiliary batch into launches of this many '
                     'shells along the grid y dimension (0: one, as '
                     "upstream and fastdfhessip launch it).  Timing only: "
                     "the kernel's pseudo-density layout depends on the "
                     'split, and the random one does not care')
ap.add_argument('--pchunk', type=int, default=0,
                help='shell pairs per thread block (0: the scheme\'s '
                     'nsp_per_block at lk = 0, as fastdfhessip item 1)')
ap.add_argument('--summary', action='store_true',
                help='print only the per-(li, lj, lk) summary')
ap.add_argument('--nvtx', action='store_true',
                help='bracket the timed region with cudaProfilerStart/Stop')
a = ap.parse_args()

import cupy as cp                                             # noqa: E402
from pyscf import gto, lib                                    # noqa: E402
from gpu4pyscf import dft                                      # noqa: E402
from gpu4pyscf.df.int3c2e_bdiv import (                        # noqa: E402
    _split_l_ctr_pattern, get_ao_pair_loc, libvhf_rys, _check_rsh_factors)
from gpu4pyscf.df.hessian.rhf import int3c2e_scheme_ip2        # noqa: E402

_sync = cp.cuda.Stream.null.synchronize


def load(spec):
    name, _, rest = spec.partition(':')
    charge, _, spin = rest.partition(':')
    sys.path.insert(0, HERE)
    from _common import needs_ecp
    xyz = os.path.join(MOL, name + '.xyz')
    kw = dict(atom=xyz, basis=a.basis, charge=int(charge or 0),
              spin=int(spin or 0), verbose=0, max_memory=80000)
    if needs_ecp(xyz, a.basis):
        kw['ecp'] = a.basis
    return name, gto.M(**kw)


def ptr(x):
    return ctypes.cast(x.data.ptr, ctypes.c_void_p)


def tmin(fn, reps):
    fn()
    _sync()
    ts = []
    for _ in range(reps):
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        fn()
        e1.record()
        e1.synchronize()
        ts.append(cp.cuda.get_elapsed_time(e0, e1) * 1e-3)
    return min(ts)


class Setup:
    """``fastdfhessip._ejk_ip2``'s kernel arguments for one molecule."""

    def __init__(self, mol, omega=0.):
        cls = dft.UKS if mol.spin else dft.RKS
        mf = cls(mol, xc='b3lyp').density_fit(auxbasis=a.auxbasis)
        mf.verbose = 0
        wd = mf.with_df
        wd.build(build_cderi=False)
        opt = self.opt = wd.intopt
        smol, saux = opt.mol, opt.auxmol
        self.smol, self.saux = smol, saux
        omega, self.lr_factor, self.sr_factor = _check_rsh_factors(
            smol, omega, None, None)
        self.omega = omega
        nsp_per_block, gout_stride, shm_size = int3c2e_scheme_ip2(omega)
        self.gout_stride = cp.asarray(gout_stride, dtype=np.int32)
        lmax = smol.uniq_l_ctr[:, 0].max()
        laux = saux.uniq_l_ctr[:, 0].max()
        self.shm_size_max = int(shm_size[:laux+1, :lmax+1, :lmax+1].max())
        self.bas_ij_idx, self.shl_pair_offsets = smol.aggregate_shl_pairs(
            opt.bas_ij_cache, a.pchunk if a.pchunk > 0 else nsp_per_block[0])
        # pchunk -1: per auxiliary l, the kernel's own tile -- as many pairs
        # per block as it runs side by side (nst / aux_per_block)
        nst = 256 // np.asarray(gout_stride.get())
        self.nsp_tile = nst // np.minimum(nst, 16)
        self._offsets_lk = {}
        self.ao_pair_loc = get_ao_pair_loc(smol.uniq_l_ctr[:, 0],
                                           opt.bas_ij_cache)
        # pair classes: bas_ij_cache is walked in order by
        # aggregate_shl_pairs, one arange of block offsets per (i, j) key
        l = smol.uniq_l_ctr[:, 0]
        p = smol.uniq_l_ctr[:, 1]
        off = self.shl_pair_offsets.get()
        self.pair_classes = []
        self.class_sp = {}
        sp0 = 0
        for (i, j), bas_ij in opt.bas_ij_cache.items():
            n = len(bas_ij)
            if n == 0:
                continue
            b0 = int(np.searchsorted(off, sp0))
            b1 = int(np.searchsorted(off, sp0 + n))
            bas_ij = np.asarray(bas_ij.get() if hasattr(bas_ij, "get") else bas_ij)
            ish = bas_ij // smol.nbas
            jsh = bas_ij % smol.nbas
            nprim = int((smol._bas[ish, 2] * smol._bas[jsh, 2]).sum())
            self.pair_classes.append((int(l[i]), int(l[j]), b0, b1, n,
                                      nprim))
            self.class_sp[b0] = sp0
            self.class_sp[b1] = sp0 + n
            sp0 += n
        naux = saux.nao
        nao = smol.nao
        aux_loc = saux.ao_loc
        batch_size = naux
        l_ctr_aux_offsets = np.append(0, np.cumsum(saux.l_ctr_counts))
        l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
            l_ctr_aux_offsets, saux.uniq_l_ctr, batch_size)
        self.ksh_offsets_cpu = l_ctr_aux_offsets
        self.ksh_offsets_gpu = cp.asarray(l_ctr_aux_offsets + smol.nbas,
                                          dtype=np.int32)
        self.aux_offsets = aux_loc[l_ctr_aux_offsets]
        self.aux_batches = []
        for kb in range(len(self.aux_offsets) - 1):
            ks0, ks1 = l_ctr_aux_offsets[kb], l_ctr_aux_offsets[kb+1]
            lk = int(saux._bas[ks0, 1])
            kprim = int(saux._bas[ks0, 2])
            self.aux_batches.append((kb, lk, int(ks1 - ks0), kprim))
        self.nao_pair = int(self.ao_pair_loc[-1])
        self.naux = naux
        self.natm = smol.natm
        self._ksplit = {}
        rng = np.random.default_rng(7)
        # one pseudo-density per auxiliary batch, laid out as _ejk_ip2's
        # `compressed` (nao_pair, naux_in_batch)
        self.dms = []
        for kb in range(len(self.aux_offsets) - 1):
            nk = int(self.aux_offsets[kb+1] - self.aux_offsets[kb])
            self.dms.append(cp.asarray(
                rng.standard_normal((self.nao_pair, nk)) * 1e-2))

    def offsets_lk(self, lk):
        if lk not in self._offsets_lk:
            self._offsets_lk[lk] = self.smol.aggregate_shl_pairs(
                self.opt.bas_ij_cache, self.nsp_tile[lk])[1]
        return self._offsets_lk[lk]

    def ksh_split(self, kb, kchunk):
        ks0, ks1 = self.ksh_offsets_cpu[kb], self.ksh_offsets_cpu[kb+1]
        if kchunk <= 0:
            return self.ksh_offsets_gpu[kb:], 1
        off = list(range(ks0, ks1, kchunk)) + [ks1]
        key = (kb, kchunk)
        if key not in self._ksplit:
            self._ksplit[key] = cp.asarray(np.asarray(off) + self.smol.nbas,
                                           dtype=np.int32)
        return self._ksplit[key], len(off) - 1

    def launch(self, kern, kb, b0, b1, ejk):
        """One launch of `kern` over pair blocks [b0, b1) and aux batch kb."""
        dm = self.dms[kb]
        ksh_offsets, nkb = self.ksh_split(kb, a.kchunk)
        spo = self.shl_pair_offsets
        if a.pchunk < 0:
            spo = self.offsets_lk(self.aux_batches[kb][1])
            if b0 is None:
                b0, b1 = 0, len(spo) - 1
            else:
                # the same pair class, as blocks of this lk's offsets
                off = spo.get()
                b0, b1 = (int(np.searchsorted(off, x)) for x in
                          (self.class_sp[b0], self.class_sp[b1]))
        err = kern(
            ptr(ejk), ptr(dm), lib.c_null_ptr(),
            ctypes.byref(self.opt.int3c2e_envs), ctypes.c_double(self.omega),
            ctypes.c_double(self.lr_factor), ctypes.c_double(self.sr_factor),
            ctypes.c_int(max(self.shm_size_max, a.shm)), ctypes.c_int(b1 - b0),
            ctypes.c_int(nkb),
            ctypes.c_void_p(spo.data.ptr + 4*b0),
            ptr(self.bas_ij_idx), ptr(ksh_offsets),
            ptr(self.gout_stride), ptr(self.ao_pair_loc),
            ctypes.c_int(int(self.aux_offsets[kb])),
            ctypes.c_int(int(self.aux_offsets[kb+1] - self.aux_offsets[kb])))
        if err != 0:
            raise RuntimeError('kernel launch failed')

    def whole(self, kern, ejk):
        nb = len(self.shl_pair_offsets) - 1
        for kb, *_ in self.aux_batches:
            self.launch(kern, kb, *((None, None) if a.pchunk < 0
                                    else (0, nb)), ejk)


def upstream_kernel():
    return libvhf_rys.ejk_int3c2e_ip2


def run(spec):
    name, mol = load(spec)
    s = Setup(mol, a.omega)
    kerns = [('stock', upstream_kernel())]
    if a.ours:
        # a library exporting the same entry point: `--lib`, or the one
        # electrolites/_ip2k.py builds
        if a.lib:
            k = ctypes.CDLL(os.path.abspath(a.lib)).ejk_int3c2e_ip2
        else:
            from electrolites import _ip2launch
            k = _ip2launch.kernel()
        kerns.append(('ours', k))
    print(f'\n## {name}  {mol.natm} atoms  nao {s.smol.nao}  naux {s.naux}  '
          f'pair classes ' + ' '.join(
              f'{li}{lj}:{n}' for li, lj, _, _, n, _ in s.pair_classes) +
          '  aux batches ' + ' '.join(
              f'l{lk}x{nk}p{kp}' for _, lk, nk, kp in s.aux_batches),
          flush=True)
    only = set()
    for c in a.only.split(';'):
        if c:
            only.add(tuple(int(x) for x in c.split(',')))
    if a.nvtx:
        from cupy.cuda import profiler
        profiler.start()
    if a.whole:
        res = {}
        for label, k in kerns:
            ejk = cp.zeros((s.natm, s.natm, 3, 3))
            res[label] = tmin(lambda: s.whole(k, ejk), a.reps)
            ejk[:] = 0
            s.whole(k, ejk)
            res[label + '_ejk'] = ejk.get()
        line = f'# whole: stock {res["stock"]*1e3:.1f} ms'
        if a.ours:
            ref = res['stock_ejk']
            err = abs(res['ours_ejk'] - ref).max() / abs(ref).max()
            line += (f'  ours {res["ours"]*1e3:.1f} ms  '
                     f'{res["stock"]/res["ours"]:.2f}x  rel err {err:.1e}')
        print(line, flush=True)
    if a.no_classes:
        return
    by_lk = {}
    by_cls = {}
    print('  li lj lk  shells(ij x k)   prim-triples  roots   stock ms   '
          'ns/prim-root' + ('   ours ms   x      err' if a.ours else ''))
    for li, lj, b0, b1, npair, nprim_ij in s.pair_classes:
        for kb, lk, nksh, kprim in s.aux_batches:
            if only and (li, lj, lk) not in only:
                continue
            nroots = (li + lj + 2 + lk) // 2 + 1
            ntrip = npair * nksh
            nprim = nprim_ij * nksh * kprim
            row = {}
            for label, k in kerns:
                ejk = cp.zeros((s.natm, s.natm, 3, 3))
                row[label] = tmin(lambda: s.launch(k, kb, b0, b1, ejk),
                                  a.reps)
                ejk[:] = 0
                s.launch(k, kb, b0, b1, ejk)
                row[label + '_ejk'] = ejk.get()
            t = row['stock']
            line = (f'  {li:2d} {lj:2d} {lk:2d}  {npair:6d} x {nksh:4d}  '
                    f'{nprim:12d}  {nroots:5d}  {t*1e3:9.3f}  '
                    f'{t*1e9/(nprim*nroots):12.2f}')
            if a.ours:
                ref = row['stock_ejk']
                err = (abs(row['ours_ejk'] - ref).max() /
                       max(abs(ref).max(), 1e-300))
                line += (f'  {row["ours"]*1e3:8.3f}  '
                         f'{t/row["ours"]:5.2f}  {err:.1e}')
            if not a.summary:
                print(line, flush=True)
            c = by_cls.setdefault((li, lj, lk), [0., 0., 0, 0, nroots])
            c[0] += t
            c[1] += row.get('ours', 0.)
            c[2] += ntrip
            c[3] += nprim
            acc = by_lk.setdefault(lk, [0., 0.])
            acc[0] += t
            if a.ours:
                acc[1] += row['ours']
    tot = sum(v[0] for v in by_lk.values())
    print('# by (li, lj, lk), heaviest first: time, share, shell triples, '
          'ns per primitive triple and root' +
          (', ours, speed-up' if a.ours else ''))
    for (li, lj, lk), (t, to, nt, npr, nr) in sorted(
            by_cls.items(), key=lambda kv: -kv[1][0]):
        line = (f'#   ({li}{lj}|{lk})  {t*1e3:8.2f} ms  {100*t/tot:4.1f} %  '
                f'{nt:9d}  {t*1e9/(npr*nr):8.2f}')
        if a.ours:
            line += f'  {to*1e3:8.2f} ms  {t/to:5.2f}x'
        print(line)
    print('# per-class sum by aux l: ' + '  '.join(
        f'l={lk}: {v[0]*1e3:.1f} ms ({100*v[0]/tot:.0f} %)'
        + (f' -> {v[1]*1e3:.1f}' if a.ours else '')
        for lk, v in sorted(by_lk.items())) + f'  total {tot*1e3:.1f} ms',
        flush=True)
    if a.nvtx:
        from cupy.cuda import profiler
        profiler.stop()


for spec in a.cases.split(','):
    run(spec)
