"""Per-class timing and correctness for the Coulomb (J) kernels.

Every ``(i,j,k,l)`` task of one real J build is run twice -- once through
GPU4PySCF's ``MD_build_j`` and once through the ``fastj`` kernel for that
``(lij,lkl)`` class -- from the *same* screening data and Hermite density, and
the times are accumulated per class.  ``--check`` additionally compares the **AO-space** J the
two builds produce, which is the correctness oracle for a new class.

Compare the AO-space J, not the Hermite-space ``vj_xyz`` the kernels write.
The Hermite components of a high-l shell pair are large and cancel in the
transform back to AO space, so two runs that keep slightly different sets of
negligible quartets -- which is what changing the tile width does, since the
screening granularity follows it -- differ by up to a per cent in ``vj_xyz``
and by 1e-14 in J.  Judging a kernel by ``vj_xyz`` rejects correct
configurations.

    python benchmarks/perclass_j.py --xyz molecules/PfPMT.xyz --charge -1
    python benchmarks/perclass_j.py --xyz molecules/PfPMT.xyz --charge -1 \
        --basis def2-TZVP --check

Classes for which ``fastj`` has no kernel are timed on GPU4PySCF only and
marked ``--``; that is the fall-through the table is meant to expose.
"""
import argparse
import json
import sys
import time
from collections import OrderedDict

import cupy as cp
import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--xyz', required=True)
ap.add_argument('--charge', type=int, default=0)
ap.add_argument('--spin', type=int, default=0)
ap.add_argument('--basis', default='6-31G*')
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--check', action='store_true',
                help='also compare the J matrix these kernels build')
ap.add_argument('--only', default='',
                help='comma separated class tags to restrict the table to')
ap.add_argument('--cycles', type=int, default=0,
                help='SCF cycles to run before timing.  Use it for the high-l '
                     'classes: a `minao` guess is exactly zero on the '
                     'polarisation shells, so their Hermite density is zero, '
                     'the screen removes their quartets, and what gets timed '
                     'is launch overhead rather than the kernel.')
a = ap.parse_args()

from pyscf import gto                                       # noqa: E402
import gpu4pyscf                                            # noqa: E402
from gpu4pyscf import scf as gscf                           # noqa: E402
from gpu4pyscf.scf import j_engine as JE                    # noqa: E402
import electrolites.fastj as fj                             # noqa: E402

mol = gto.M(atom=a.xyz, basis=a.basis, charge=a.charge, spin=a.spin,
            verbose=0, max_memory=80000)
tag = f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}'
print(f'[{tag}] natm={mol.natm} nbas={mol.nbas} nao={mol.nao} '
      f'density={"scf-" + str(a.cycles) if a.cycles else "minao"} '
      f'gpu4pyscf={gpu4pyscf.__version__}', flush=True)

mf = gscf.RHF(mol)
if a.cycles:
    mf.max_cycle = a.cycles
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    dm = cp.asarray(mf.make_rdm1())
else:
    dm = cp.asarray(mf.get_init_guess(mol, 'minao'))
opt = JE._VHFOpt(mol, 1e-13).build()
nao_orig = dm.shape[-1]
dms = opt.apply_coeff_C_mat_CT(dm.reshape(-1, nao_orig, nao_orig))

s = fj._setup_181(opt, dms)
only = set(a.only.split(',')) if a.only else None
sync = cp.cuda.Stream.null.synchronize

rows = OrderedDict()          # tag -> dict(ref=, ours=, ntasks=, err=)
for task in s.tasks:
    i, j, k, l = task
    t = f'{i+j}_{k+l}'
    if only and t not in only:
        continue
    has_ours = t in fj.LAUNCH
    row = rows.setdefault(t, {'ref': 0.0, 'ours': 0.0, 'ntasks': 0,
                              'err': 0.0, 'has_ours': has_ours})

    def run(force):
        vj = cp.zeros_like(s.dm_xyz)
        ran = fj._launch_task(s, task, vj, force=force)
        return vj, ran

    vj_ref, ran = run('ref')
    if ran is None:
        continue
    row['ntasks'] += 1

    def timeit(force):
        vj = cp.zeros_like(s.dm_xyz)
        fj._launch_task(s, task, vj, force=force)
        sync()
        ts = []
        for _ in range(a.reps):
            sync()
            t0 = time.perf_counter()
            fj._launch_task(s, task, vj, force=force)
            sync()
            ts.append(time.perf_counter() - t0)
        return min(ts)

    row['ref'] += timeit('ref')
    if has_ours:
        row['ours'] += timeit('ours')
        if a.check:
            vj_ours, _ = run('ours')
            d = float(cp.abs(vj_ours - vj_ref).max())
            scale = float(cp.abs(vj_ref).max()) or 1.0
            row['err'] = max(row['err'], d / scale)

# The correctness oracle: the AO-space J of a whole build, both ways.
if a.check:
    def build(force):
        vj = cp.zeros_like(s.dm_xyz)
        for task in s.tasks:
            fj._launch_task(s, task, vj, force=force)
        J = JE._Rt_to_dm(s.mol, vj.get(), s.pair_lst, s.pair_loc,
                         opt.rys_envs)
        return cp.asnumpy(J) * 2.
    J_ref = build('ref')
    J_ours = build(None)
    dJ = float(np.abs(J_ours - J_ref).max())
    print(f'AO-space J: max|dJ| = {dJ:.3e}  relative '
          f'{dJ/float(np.abs(J_ref).max()):.3e}', flush=True)

print(f'{"class":>7} {"tasks":>5} {"gpu4pyscf":>11} {"fastj":>11} '
      f'{"speedup":>8}' + ('   herm.rel' if a.check else ''))
tot_ref = tot_ours = 0.0
for t, r in sorted(rows.items()):
    if r['ntasks'] == 0:
        continue
    tot_ref += r['ref']
    tot_ours += r['ours'] if r['has_ours'] else r['ref']
    sp = f"{r['ref']/r['ours']:8.2f}x" if r['has_ours'] and r['ours'] else '      --'
    line = (f'{t:>7} {r["ntasks"]:5d} {1e3*r["ref"]:9.2f}ms '
            f'{1e3*r["ours"]:9.2f}ms {sp}')
    if a.check:
        line += f'  {r["err"]:10.1e}' if r['has_ours'] else '          --'
    print(line, flush=True)
print(f'{"total":>7} {"":5s} {1e3*tot_ref:9.2f}ms {1e3*tot_ours:9.2f}ms '
      f'{tot_ref/tot_ours:8.2f}x')

if a.check:
    print('the `herm.rel` column is the Hermite-space vj_xyz difference, which\n'
          'is sensitive to the screening granularity a tile width implies and\n'
          'is not the correctness oracle; the AO-space line above is.')
print('RESULT ' + json.dumps(dict(
    tag=tag, gpu4pyscf=gpu4pyscf.__version__, xyz=a.xyz, basis=a.basis,
    natm=mol.natm, nao=int(mol.nao), reps=a.reps,
    rows={t: r for t, r in rows.items() if r['ntasks']},
    tot_ref=tot_ref, tot_ours=tot_ours)), flush=True)
