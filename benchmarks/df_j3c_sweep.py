"""How many shell pairs a thread block of the three-centre Coulomb kernels wants.

`docs/DF_WORK_LIST.md` **W9**.  ``Int3c2eOpt.build`` makes exactly **one**
thread-block range per angular-momentum group and nothing splits them further,
so the auxiliary-vector kernel -- whose grid is
``(groups x auxiliary-shell batches)`` -- launches a few hundred blocks on a
108-SM A100, and the groups differ in size by two orders of magnitude
(11 306 shell pairs against 159 on penicillin V).  Both kernels read their
group's ``(li,lj)`` from the first pair of their range and stride the rest, so
a range may be any sub-interval of a group: splitting one is a launch
configuration change, not a kernel change.

This is the same shape as W3's ``nsp_per_block*4`` finding
(`docs/README_fastdfgrad.md` §4.2) and it is swept here rather than guessed,
because the two kernels want opposite things -- more blocks give the
auxiliary-vector kernel parallelism it does not have, and cost the
density-contraction kernel an extra warp reduction and ``atomicAdd`` per
block.

Each kernel is timed alone, at fixed density, round robin over the chunk
sizes.

    python benchmarks/df_j3c_sweep.py --cases penicillinV,taxol
    python benchmarks/df_j3c_sweep.py --cases PfPMT:-1 --chunks 0,4096,2048
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
ap.add_argument('--cases', default='penicillinV,taxol')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--chunks', default='0,8192,4096,2048,1024,512,256,128',
                help='shell pairs per block; 0 is GPU4PySCF\'s whole group')
ap.add_argument('--shm-fracs', default='',
                help='also sweep the dynamic shared memory the two kernels '
                     'ask for, as a multiple of GPU4PySCF\'s.  Both schemes '
                     'ask for the widest angular-momentum class\'s '
                     'requirement -- 91 KB, one block per A100 SM -- on every '
                     'block of the grid')
ap.add_argument('--per-class', action='store_true',
                help='also time one launch of the auxiliary-vector kernel per '
                     '(shell-pair class, auxiliary class), each asking only '
                     'for the shared memory that class lays out')
ap.add_argument('--ksh-chunks', default='',
                help='also sweep the auxiliary shells per block of the '
                     'auxiliary-vector kernel (16 is GPU4PySCF\'s), at the '
                     "shipped shell-pair block size")
ap.add_argument('--reps', type=int, default=6)
ap.add_argument('--cycles', type=int, default=1)
a = ap.parse_args()

import electrolites                                           # noqa: E402
electrolites.patch(['fastdfj3c'])
from electrolites import fastdfj3c, _j3c                      # noqa: E402

import cupy as cp                                             # noqa: E402
from pyscf import gto                                         # noqa: E402
from gpu4pyscf import dft                                      # noqa: E402
from gpu4pyscf.df import df_jk, j_engine_3c2e                  # noqa: E402

_sync = cp.cuda.Stream.null.synchronize
_SHIPPED_SP = fastdfj3c.SP_CHUNK_AUX
_SHIPPED_KSH = fastdfj3c.KSH_CHUNK
_SHIPPED_FRAC = fastdfj3c.SHM_FRAC
_SHIPPED_PC = fastdfj3c.PER_CLASS


def load(spec):
    name, _, rest = spec.partition(':')
    charge, _, spin = rest.partition(':')
    mol = gto.M(atom=os.path.join(MOL, name + '.xyz'), basis=a.basis,
                charge=int(charge or 0), spin=int(spin or 0),
                verbose=0, max_memory=80000)
    return name, mol


def run(spec, chunks):
    name, mol = load(spec)
    cls = dft.UKS if mol.spin else dft.RKS
    mf = cls(mol, xc='pbe').density_fit(auxbasis=a.auxbasis)
    mf.grids.level = 3
    mf.verbose = 0
    mf.max_cycle = a.cycles
    mf.kernel()
    dm = cp.asarray(mf.make_rdm1())
    # one matrix is what the sweep wants: it times the two kernels, not the
    # driver, and a two-spin density would only run each of them twice
    dm = dm.reshape(-1, dm.shape[-1], dm.shape[-1])[0]
    dfobj = mf.with_df
    if dfobj.intopt is None:
        dfobj.build(build_cderi=False)
    # one whole Coulomb build to make the option object and the caches
    df_jk.get_j(dfobj, dm, 1)
    opt = dfobj.j_engine
    c = _j3c.constants(opt)
    smol = opt.mol
    groups = np.diff(c._sp_offsets_host).tolist()
    print(f'\n## {name}  {mol.natm} atoms  nao {mol.nao}  '
          f'{c.npair} shell pairs in {len(groups)} groups '
          f'{sorted(groups, reverse=True)}  '
          f'{opt.auxmol.nbas} auxiliary shells, {c.ksh_blocks} batches',
          flush=True)

    dm_t = smol.apply_C_mat_CT(dm)
    rho = opt.contract_dm(dm_t, 1)
    rho = opt.auxmol.apply_CT_dot(rho, axis=-1)
    rho = opt.auxmol.apply_C_dot(rho, axis=-1)

    ref = {}
    t = {k: {c_: [] for c_ in chunks} for k in ('contract_dm',
                                                'contract_auxvec')}
    for c_ in chunks:
        fastdfj3c.SP_CHUNK_DM = fastdfj3c.SP_CHUNK_AUX = c_
        ref.setdefault('dm', cp.asnumpy(opt.contract_dm(dm_t, 1)))
        ref.setdefault('aux', cp.asnumpy(opt.contract_auxvec(rho)))
    errs = {}
    for c_ in chunks:
        fastdfj3c.SP_CHUNK_DM = fastdfj3c.SP_CHUNK_AUX = c_
        e1 = float(abs(cp.asnumpy(opt.contract_dm(dm_t, 1)) - ref['dm']).max()
                   / abs(ref['dm']).max())
        e2 = float(abs(cp.asnumpy(opt.contract_auxvec(rho)) - ref['aux']).max()
                   / abs(ref['aux']).max())
        errs[c_] = (e1, e2)

    for _ in range(a.reps):
        for c_ in chunks:                              # round robin, §5.2
            fastdfj3c.SP_CHUNK_DM = fastdfj3c.SP_CHUNK_AUX = c_
            _sync()
            t0 = time.perf_counter()
            opt.contract_dm(dm_t, 1)
            _sync()
            t['contract_dm'][c_].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            opt.contract_auxvec(rho)
            _sync()
            t['contract_auxvec'][c_].append(time.perf_counter() - t0)

    print(f'  {"chunk":>7s} {"blocks":>8s} {"contract_dm":>13s} {"x":>7s} '
          f'{"contract_auxvec":>16s} {"x":>7s} {"rel err":>10s}')
    base = {k: min(t[k][chunks[0]]) for k in t}
    for c_ in chunks:
        nblk = c.pair_offsets(c_)[1]
        d = min(t['contract_dm'][c_]) * 1e3
        v = min(t['contract_auxvec'][c_]) * 1e3
        print(f'  {c_:7d} {nblk:8d} {d:13.3f} {base["contract_dm"]*1e3/d:7.3f} '
              f'{v:16.3f} {base["contract_auxvec"]*1e3/v:7.3f} '
              f'{max(errs[c_]):10.1e}')
    sys.stdout.flush()
    fastdfj3c.SP_CHUNK_DM, fastdfj3c.SP_CHUNK_AUX = 0, _SHIPPED_SP

    if a.shm_fracs:
        fracs = [float(x) for x in a.shm_fracs.split(',')]
        ts = {k: {f: [] for f in fracs} for k in ('dm', 'aux')}
        for _ in range(a.reps):
            for f in fracs:                            # round robin, §5.2
                fastdfj3c.SHM_FRAC = f
                _sync()
                t0 = time.perf_counter()
                opt.contract_dm(dm_t, 1)
                _sync()
                ts['dm'][f].append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                opt.contract_auxvec(rho)
                _sync()
                ts['aux'][f].append(time.perf_counter() - t0)
        fastdfj3c.SHM_FRAC = _SHIPPED_FRAC
        print(f'  shared memory asked for, at {_SHIPPED_SP} pairs per block')
        print(f'  {"frac":>6s} {"shm dm":>8s} {"contract_dm":>13s} {"x":>7s} '
              f'{"shm aux":>8s} {"contract_auxvec":>16s} {"x":>7s}')
        for f in fracs:
            sd = fastdfj3c._scheme('dm', f)[1] / 1024
            sa = fastdfj3c._scheme('auxvec', f)[1] / 1024
            d = min(ts['dm'][f]) * 1e3
            v = min(ts['aux'][f]) * 1e3
            print(f'  {f:6.3f} {sd:7.1f}K {d:13.3f} '
                  f'{min(ts["dm"][fracs[0]])*1e3/d:7.3f} {sa:7.1f}K '
                  f'{v:16.3f} {min(ts["aux"][fracs[0]])*1e3/v:7.3f}')
        sys.stdout.flush()

    if a.per_class:
        tp = {}
        for _ in range(a.reps):
            for pc in (0, 1):                          # round robin, §5.2
                fastdfj3c.PER_CLASS = pc
                _sync()
                t0 = time.perf_counter()
                opt.contract_auxvec(rho)
                _sync()
                tp.setdefault(pc, []).append(time.perf_counter() - t0)
        fastdfj3c.PER_CLASS = _SHIPPED_PC
        ngroups = (len(c.pair_offsets_by_group(_SHIPPED_SP))
                   * len(c.ksh_batches_by_group(_SHIPPED_KSH)))
        print(f'  contract_auxvec, one launch {min(tp[0])*1e3:.3f} ms, '
              f'{ngroups} per-class launches {min(tp[1])*1e3:.3f} ms, '
              f'{min(tp[0])/min(tp[1]):.3f}x')
        sys.stdout.flush()

    if not a.ksh_chunks:
        return
    kchunks = [int(x) for x in a.ksh_chunks.split(',')]
    tk = {k: [] for k in kchunks}
    for _ in range(a.reps):
        for k in kchunks:
            fastdfj3c.KSH_CHUNK = k
            _sync()
            t0 = time.perf_counter()
            opt.contract_auxvec(rho)
            _sync()
            tk[k].append(time.perf_counter() - t0)
    fastdfj3c.KSH_CHUNK = _SHIPPED_KSH
    print(f'  auxiliary shells per block, at {_SHIPPED_SP} pairs per block')
    print(f'  {"ksh":>7s} {"blocks":>8s} {"contract_auxvec":>16s} {"x":>7s}')
    base = min(tk[kchunks[0]])
    for k in kchunks:
        v = min(tk[k]) * 1e3
        print(f'  {k:7d} {c.ksh_batches(k)[1]:8d} {v:16.3f} '
              f'{base*1e3/v:7.3f}')
    sys.stdout.flush()


def main():
    chunks = [int(x) for x in a.chunks.split(',')]
    print(f'# gpu4pyscf {electrolites.gpu4pyscf_version()}  basis {a.basis}')
    for spec in a.cases.split(','):
        run(spec, chunks)


main()
