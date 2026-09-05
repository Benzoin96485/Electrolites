"""How the Coulomb build's cost grows with the number of density matrices.

Catalogue item **B4a**.  GPU4PySCF's ``MD_build_j`` takes at most
``DM_BLOCK = 4`` density matrices per launch (``scf/j_engine.py`` forces
``n_dm = 4`` inside ``_md_j_engine_quartets_scheme``, and the driver steps
``dm_offset += 4``), so a CPHF or TDDFT build with ``n_dm = 126`` runs the
whole kernel -- **including the Hermite ``Rt`` recurrence, which does not
depend on the density at all** -- ``ceil(126/4) = 32`` times over.

This measures the ceiling before anything is written, which is the same thing
``docs/ROUND2_1.8.1.md`` 3b did to the ``pair_vk`` block reduction and the
reason that one was not built.  The cost model is

    T(n) = ceil(n/B) * R  +  n * C  +  L

with ``R`` the density-independent work per pass (Rt, the Boys function, the
geometry, the screening), ``C`` the per-density contraction, ``L`` a constant.
``R`` and ``C`` are solved for from three measured points that straddle a pass
boundary -- ``n = B``, ``n = B+1`` and ``n = 2B`` -- and the ceiling for
perfect amortisation is then ``R + n*C + L``, i.e. the Rt recurrence paid once.

Every density is a copy of the same matrix, so ``dm_cond`` -- and therefore
the surviving quartet list -- is identical at every ``n``.  That is
deliberate: it isolates the block cap (B4a) from the union-screening problem
(B4c), which is a separate entry.

    python benchmarks/ndm_j.py --xyz benchmarks/molecules/PfPMT.xyz \
        --charge -1 --cycles 1
"""
import argparse
import sys
import time

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--xyz', required=True)
ap.add_argument('--charge', type=int, default=0)
ap.add_argument('--spin', type=int, default=0)
ap.add_argument('--basis', default='6-31G*')
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--cycles', type=int, default=1,
                help='SCF cycles to run before timing.  A `minao` guess is '
                     'exactly zero on the polarisation shells and screens far '
                     'harder than a real density -- docs/ROUND2_1.8.1.md 3c')
ap.add_argument('--ndm', default='1,2,4,8,9,12,16,32',
                help='comma separated n_dm values to time')
ap.add_argument('--perclass', action='store_true',
                help='also break the n_dm=1 and n_dm=B..2B rows down by class')
ap.add_argument('--with-k', action='store_true',
                help='time the exchange build at the same n_dm for comparison')
ap.add_argument('--sweep-blocks', default='',
                help='semicolon separated density-block sets to compare, e.g. '
                     '"4;4,8;4,8,16".  All of them run in one process against '
                     'the same density and task list, which is the only fair '
                     'way to A/B them.  This is the ablation that separates '
                     "the kernels' own quality (n_dm = 4 is GPU4PySCF's own "
                     'block size, so a gain there is not amortisation) from '
                     'amortising the Rt recurrence over more densities')
ap.add_argument('--ours', action='store_true',
                help="also time fastj's multi-density kernels at every n_dm, "
                     'and check the AO-space J they build against '
                     "GPU4PySCF's")
a = ap.parse_args()

import cupy as cp                                            # noqa: E402
import numpy as np                                           # noqa: E402
from pyscf import gto                                        # noqa: E402
import gpu4pyscf                                             # noqa: E402
from gpu4pyscf import scf as gscf                            # noqa: E402
from gpu4pyscf.scf import j_engine as JE                     # noqa: E402
import electrolites.fastj as fj                              # noqa: E402

DM_BLOCK = 4          # scf/j_engine.py:301 and md_contract_j.cu's dm_offset+=4

mol = gto.M(atom=a.xyz, basis=a.basis, charge=a.charge, spin=a.spin,
            verbose=0, max_memory=80000)
tag = f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}'
print(f'# gpu4pyscf {gpu4pyscf.__version__}  {cp.cuda.Device().mem_info[1]>>20} MiB  '
      f'{cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}')
print(f'[{tag}] natm={mol.natm} nbas={mol.nbas} nao={mol.nao}', flush=True)

mf = gscf.RHF(mol)
mf.direct_scf_tol = 1e-13
dm = mf.get_init_guess(mol, 'minao')
if a.cycles:
    mf.max_cycle = a.cycles
    mf.conv_tol = 1e-12
    mf.kernel(dm0=dm)
    dm = cp.asarray(mf.make_rdm1())
    print(f'[{tag}] density after {a.cycles} SCF cycle(s)', flush=True)
dm = cp.asarray(dm).reshape(mol.nao, mol.nao)

vhfopt = JE._VHFOpt(mol, mf.direct_scf_tol).build()
sorted_dm = vhfopt.apply_coeff_C_mat_CT(dm.reshape(1, mol.nao, mol.nao))[0]


def time_build(n_dm, reps, force):
    """One whole J build over every task, at this n_dm, through one path.

    ``force`` is ``'ref'`` for GPU4PySCF's own ``MD_build_j`` and ``None`` for
    fastj's dispatch, which is the multi-density kernels plus whatever tail
    they do not cover.  Both drive the *same* task list, screening data and
    Hermite density, so nothing but the kernels differs.
    """
    dms = cp.repeat(sorted_dm[None], n_dm, axis=0)
    s = fj._setup_181(vhfopt, dms)
    if force == 'ref':
        s.use_ours = False
    vj_xyz = cp.zeros_like(s.dm_xyz)

    def once():
        vj_xyz.fill(0.)
        for task in s.tasks:
            fj._launch_task(s, task, vj_xyz, force=force)
    once()
    cp.cuda.Stream.null.synchronize()
    ts = []
    for _ in range(reps):
        cp.cuda.Stream.null.synchronize()
        t = time.perf_counter()
        once()
        cp.cuda.Stream.null.synchronize()
        ts.append(time.perf_counter() - t)
    return min(ts), s, vj_xyz


def time_ref(n_dm, reps):
    t, s, _ = time_build(n_dm, reps, 'ref')
    return t, s


def time_ref_perclass(n_dm, reps):
    dms = cp.repeat(sorted_dm[None], n_dm, axis=0)
    s = fj._setup_181(vhfopt, dms)
    s.use_ours = False
    vj_xyz = cp.zeros_like(s.dm_xyz)
    acc = {}
    for task in s.tasks:
        ts = []
        for _ in range(reps):
            cp.cuda.Stream.null.synchronize()
            t = time.perf_counter()
            tg = fj._launch_task(s, task, vj_xyz, force='ref')
            cp.cuda.Stream.null.synchronize()
            ts.append(time.perf_counter() - t)
        if tg is None:
            continue
        acc[tg] = acc.get(tg, 0.) + min(ts)
    return acc


ndms = [int(x) for x in a.ndm.split(',')]
print(f'\n[{tag}] Coulomb build, GPU4PySCF 1.8.1 kernels, min of {a.reps}')
print(f'{"n_dm":>6} {"T (s)":>10} {"T/n_dm (ms)":>13} {"passes":>7} '
      f'{"T/T(1)":>8}')
T = {}
for n in ndms:
    t, _ = time_ref(n, a.reps)
    T[n] = t
    print(f'{n:6d} {t:10.4f} {t/n*1e3:13.3f} {-(-n//DM_BLOCK):7d} '
          f'{t/T[ndms[0]]:8.2f}', flush=True)

# Straddle a pass boundary *away from the n_dm <= DM_BLOCK end*, where
# MD_build_j still has its own specialisations (n_dm == 1 takes a separate
# kernel entirely).  ceil(n/B) is 2, 3, 3 at these three points, so the pass
# term separates from the density term.
need = (2*DM_BLOCK, 2*DM_BLOCK+1, 3*DM_BLOCK)
if all(n in T for n in need):
    b, b1, b2 = (T[n] for n in need)
    # T(2B)  = 2R + 2B*C     + L
    # T(2B+1)= 3R + (2B+1)C  + L
    # T(3B)  = 3R + 3B*C     + L
    C = (b2 - b1) / (DM_BLOCK - 1)
    R = b1 - b - C
    L = b - 2*R - 2*DM_BLOCK*C
    print(f'\n[{tag}] cost model  T(n) = ceil(n/{DM_BLOCK})*R + n*C + L')
    print(f'  R (density-independent work per pass) = {R*1e3:8.3f} ms')
    print(f'  C (per-density contraction)           = {C*1e3:8.3f} ms')
    print(f'  L (constant)                          = {L*1e3:8.3f} ms')
    if C > 0:
        print(f'  R/C = {R/C:.2f}: one pass costs as much as {R/C:.2f} '
              f'densities of contraction')
    print(f'\n[{tag}] ceiling for perfect amortisation (Rt paid once)')
    print(f'{"n_dm":>6} {"measured":>10} {"model":>10} {"ideal":>10} '
          f'{"ceiling":>8}')
    for n in ndms:
        model = -(-n // DM_BLOCK) * R + n * C + L
        ideal = R + n * C + L
        print(f'{n:6d} {T[n]:10.4f} {model:10.4f} {ideal:10.4f} '
              f'{T[n]/ideal:8.2f}x')

if a.ours:
    print(f'\n[{tag}] fastj multi-density kernels against GPU4PySCF, '
          f'min of {a.reps}')
    print(f'# blocks per class: '
          f'{sorted({tuple(v) for v in fj._MDM_BLOCKS.values()})}')
    print(f'{"n_dm":>6} {"gpu4pyscf":>10} {"fastj":>10} {"speedup":>8} '
          f'{"fastj/n_dm":>11} {"max|dJ|":>10}')
    TO = {}
    for n in ndms:
        tr, sr, vr = time_build(n, a.reps, 'ref')
        to, so, vo = time_build(n, a.reps, None)
        TO[n] = to
        # The Hermite-space vectors are what the kernels write; compare the
        # AO-space J, for the reason perclass_j.py gives -- the Hermite
        # components of a high-l pair are large and cancel in the transform.
        jr = JE._Rt_to_dm(sr.mol, vr.get(), sr.pair_lst, sr.pair_loc,
                          vhfopt.rys_envs)
        jo = JE._Rt_to_dm(so.mol, vo.get(), so.pair_lst, so.pair_loc,
                          vhfopt.rys_envs)
        d = float(cp.abs(cp.asarray(jo) - cp.asarray(jr)).max())
        print(f'{n:6d} {tr:10.4f} {to:10.4f} {tr/to:7.2f}x '
              f'{to/n*1e3:10.3f} {d:10.2e}', flush=True)

if a.sweep_blocks:
    import json as _json
    full = {k: list(v) for k, v in fj._MDM_BLOCKS.items()}
    print(f'\n[{tag}] density-block ablation, min of {a.reps}')
    print(f'{"n_dm":>6} {"blocks":>10} {"T (s)":>10} {"vs gpu4pyscf":>13} '
          f'{"launches/class":>15}')
    for n in ndms:
        tr, _, _ = time_build(n, a.reps, 'ref')
        print(f'{n:6d} {"gpu4pyscf":>10} {tr:10.4f} {1.0:12.2f}x '
              f'{-(-n//DM_BLOCK):15d}')
        for spec in a.sweep_blocks.split(';'):
            keep = {int(x) for x in spec.split(',') if x}
            fj._MDM_BLOCKS.clear()
            for k, v in full.items():
                w = [b for b in v if b in keep]
                if w:
                    fj._MDM_BLOCKS[k] = w
            fj._function.cache_clear()
            to, so, _ = time_build(n, a.reps, None)
            # how many kernel launches the busiest covered class needs
            nl = max((len(fj._dm_split(k, n)[0]) + (fj._dm_split(k, n)[1] > 0)
                      for k in fj._MDM_BLOCKS), default=0)
            print(f'{n:6d} {spec:>10} {to:10.4f} {tr/to:12.2f}x '
                  f'{nl:15d}', flush=True)
    fj._MDM_BLOCKS.clear()
    fj._MDM_BLOCKS.update(full)

if a.with_k:
    print(f'\n[{tag}] exchange build for comparison (scf.jk)')
    from gpu4pyscf.scf import jk as JK
    kopt = JK._VHFOpt(mol, mf.direct_scf_tol).build()
    kdm = kopt.apply_coeff_C_mat_CT(dm.reshape(1, mol.nao, mol.nao))[0]
    print(f'{"n_dm":>6} {"T (s)":>10} {"T/n_dm (ms)":>13} {"T/T(1)":>8}')
    TK = {}
    for n in ndms:
        dms = cp.repeat(kdm[None], n, axis=0)
        f = lambda: kopt.get_k(dms, 1, None)                  # noqa: E731
        f(); cp.cuda.Stream.null.synchronize()
        ts = []
        for _ in range(a.reps):
            cp.cuda.Stream.null.synchronize()
            t = time.perf_counter(); f()
            cp.cuda.Stream.null.synchronize()
            ts.append(time.perf_counter() - t)
        TK[n] = min(ts)
        print(f'{n:6d} {TK[n]:10.4f} {TK[n]/n*1e3:13.3f} '
              f'{TK[n]/TK[ndms[0]]:8.2f}', flush=True)
    print(f'\n[{tag}] J/K time ratio')
    for n in ndms:
        print(f'  n_dm={n:4d}  J/K = {T[n]/TK[n]:6.2f}')

if a.perclass:
    print(f'\n[{tag}] per class, GPU4PySCF kernels')
    rows = {n: time_ref_perclass(n, a.reps) for n in need}
    keys = sorted(set().union(*[set(r) for r in rows.values()]),
                  key=lambda k: -rows[need[0]].get(k, 0.))
    hdr = ' '.join(f'{"n="+str(n):>10}' for n in need)
    print(f'{"class":>7} {hdr} {"R/C":>7}')
    for k in keys:
        v = [rows[n].get(k, 0.) for n in need]
        C = (v[2] - v[1]) / (DM_BLOCK - 1)
        R = v[1] - v[0] - C
        rc = f'{R/C:7.2f}' if C > 1e-9 else '      -'
        print(f'{k:>7} ' + ' '.join(f'{x:10.4f}' for x in v) + f' {rc}')
