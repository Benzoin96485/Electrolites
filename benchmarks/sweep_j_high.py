"""Search the launch configuration of the written high-l J kernels.

The tables next to the other kernels here are one-dimensional: one number per
angular-momentum class (``minblocks`` for ``fastj``, ``(nsq, gout_stride)``
for ``fastk``).  JoltQC searches four independent fragmentation factors
instead, and a McMurchie-Davidson J kernel has at least five knobs that are
not equivalent to each other:

  ``threadsx`` x ``threadsy``  the quartet tile a block holds, which sets how
      many quartets share one Rt column and therefore the shared-memory bill;
  ``gout_stride``  how many threads cooperate on one quartet -- it trades the
      cost of the cooperative Rt build against the width of the contraction
      each thread has to carry in registers;
  ``tiley``  how many ket tiles a block walks, which is what the ket vj
      accumulator costs in shared memory, and so what caps blocks per SM;
  ``lreg``  how far up the Rt recurrence runs in registers before it has to
      go through shared memory and take two barriers per level;
  ``dmreg``  whether the bra Hermite density sits in registers.

They interact: raising ``gout_stride`` shrinks the register footprint but
raises the barrier count, and both feed back into occupancy through
``__launch_bounds__``.  A one-dimensional table cannot see that, so this
sweeps the product and reports the best per class.

Every candidate runs the *same* task list, screening data and Hermite density
as the shipped kernel -- `fastj._setup_181` builds it once -- and is checked
against GPU4PySCF's general kernel before it is timed, so a configuration
that is fast because it is wrong cannot win.

The check is on the **AO-space** J, not on the Hermite-space ``vj_xyz`` the
kernels write.  A tile width sets the granularity of the tile-max screening,
so two arithmetically identical kernels with different tiles keep slightly
different sets of negligible quartets; the Hermite components of a high-l
pair are large and cancel on the way back to AO space, so that shows up as up
to a per cent in ``vj_xyz`` and 1e-14 in J.  Checking ``vj_xyz`` rejects every
tile width but the one GPU4PySCF happens to use.

    python benchmarks/sweep_j_high.py --xyz molecules/PfPMT.xyz --charge -1 \
        --basis def2-TZVP --classes 3_3,4_3
    python benchmarks/sweep_j_high.py ... --write   # update the launch table
"""
import argparse
import itertools
import json
import os
import sys
import time

import cupy as cp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'codegen'))
import gen_j_high                                            # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--xyz', required=True)
ap.add_argument('--charge', type=int, default=0)
ap.add_argument('--spin', type=int, default=0)
ap.add_argument('--basis', default='6-31G*')
ap.add_argument('--classes', default=','.join(gen_j_high.CFG))
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--write', action='store_true',
                help='write the winners into fastjhigh_launch.json')
ap.add_argument('--grid', default='default',
                help="'default' or 'wide'")
ap.add_argument('--cycles', type=int, default=0,
                help='SCF cycles to run before timing.  A `minao` guess is '
                     'exactly zero on the polarisation shells these classes '
                     'need, so the correctness check against it passes '
                     'vacuously; a converged density exercises them.')
a = ap.parse_args()

from pyscf import gto                                        # noqa: E402
import gpu4pyscf                                             # noqa: E402
from gpu4pyscf import scf as gscf                            # noqa: E402
from gpu4pyscf.scf import j_engine as JE                     # noqa: E402
import electrolites.fastj as fj                              # noqa: E402

# ---- the search space ----------------------------------------------------
SHAPES = [(8, 8, 4), (8, 4, 8), (16, 4, 4), (4, 8, 8), (16, 2, 8), (8, 8, 2),
          (16, 4, 2), (32, 2, 4)]
GRID = dict(
    default=dict(shape=SHAPES[:5], tiley=[8, 16, 32], lreg=[3],
                 minblocks=[1, 2], dmreg=[True, False]),
    wide=dict(shape=SHAPES, tiley=[4, 8, 16, 32], lreg=[2, 3, 4],
              minblocks=[1, 2, 3], dmreg=[True, False]),
)[a.grid]

SHM_MAX = int(cp.cuda.runtime.getDeviceProperties(0)
              ['sharedMemPerMultiprocessor'])


def candidates(tag):
    lij, lkl = (int(x) for x in tag.split('_'))
    base = gen_j_high.CFG[tag]
    seen = set()
    for (tx, ty, gs), tiley, lreg, mb, dmreg in itertools.product(
            GRID['shape'], GRID['tiley'], GRID['lreg'], GRID['minblocks'],
            GRID['dmreg']):
        if (tx * ty) % 32 or tx * ty * gs > 1024 or tx < 2:
            continue
        if lreg >= lij + lkl:
            continue
        cfg = dict(base, threadsx=tx, threadsy=ty, gout_stride=gs,
                   tiley=tiley, lreg=lreg, minblocks=mb, dmreg=dmreg)
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        yield cfg


mol = gto.M(atom=a.xyz, basis=a.basis, charge=a.charge, spin=a.spin,
            verbose=0, max_memory=80000)
tag_all = f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}'
print(f'[{tag_all}] natm={mol.natm} nao={mol.nao} '
      f'gpu4pyscf={gpu4pyscf.__version__} shm/SM={SHM_MAX/1024:.0f}KB',
      flush=True)

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
dms = opt.apply_coeff_C_mat_CT(dm.reshape(1, mol.nao, mol.nao))
s = fj._setup_181(opt, dms)
sync = cp.cuda.Stream.null.synchronize

best = {}
for tag in a.classes.split(','):
    lij, lkl = (int(x) for x in tag.split('_'))
    tasks = [t for t in s.tasks
             if f'{t[0]+t[1]}_{t[2]+t[3]}' == tag
             and s.pair_mappings[t[0], t[1]][0].size
             and s.pair_mappings[t[2], t[3]][0].size]
    if not tasks:
        print(f'{tag}: no tasks in this basis', flush=True)
        continue

    # the reference, and the shipped configuration, on the same tasks
    def run(force, override=None):
        vj = cp.zeros_like(s.dm_xyz)
        for t in tasks:
            fj._launch_task(s, t, vj, force=force, override=override)
        return vj

    def timeit(force, override=None):
        run(force, override)
        sync()
        ts = []
        for _ in range(a.reps):
            sync()
            t0 = time.perf_counter()
            for t in tasks:
                fj._launch_task(s, t, vj_scratch, force=force,
                                override=override)
            sync()
            ts.append(time.perf_counter() - t0)
        return min(ts)

    vj_scratch = cp.zeros_like(s.dm_xyz)

    def to_ao(vj):
        J = JE._Rt_to_dm(s.mol, vj.get(), s.pair_lst, s.pair_loc,
                         opt.rys_envs)
        return cp.asnumpy(J) * 2.

    vj_ref = run('ref')
    J_ref = to_ao(vj_ref)
    scale = float(np.abs(J_ref).max()) or 1.0
    t_ref = timeit('ref')
    t_now = timeit('ours')
    print(f'\n{tag}: {len(tasks)} tasks   gpu4pyscf {1e3*t_ref:8.3f} ms   '
          f'shipped {1e3*t_now:8.3f} ms  ({t_ref/t_now:.2f}x)', flush=True)

    cfgs = list(candidates(tag))
    src = [gen_j_high.HEAD]
    metas = {}
    for n, cfg in enumerate(cfgs):
        name = f'j_{tag}_v{n}'
        try:
            text, meta = gen_j_high.gen_class(lij, lkl, cfg, name)
        except AssertionError:
            continue
        if meta['shm'] * 8 > SHM_MAX:
            continue
        src.append(text)
        metas[name] = (cfg, meta)
    print(f'  {len(metas)} candidates of {len(cfgs)} fit the shared-memory '
          f'window; compiling', flush=True)
    t0 = time.perf_counter()
    prologue = open(os.path.join(fj.HERE, 'fastj_prologue.cu')).read()
    mod = cp.RawModule(code=prologue + '\n'.join(src),
                       options=('-std=c++17',), backend='nvrtc')
    mod.compile()
    print(f'  compiled in {time.perf_counter()-t0:.1f} s', flush=True)

    rows = []
    for name, (cfg, meta) in metas.items():
        kern = mod.get_function(name)
        if meta['shm'] * 8 > 48 * 1024:
            kern.max_dynamic_shared_size_bytes = meta['shm'] * 8
        over = {tag: (meta, kern)}
        err = float(np.abs(to_ao(run(None, over)) - J_ref).max()) / scale
        if err > 1e-11:
            print(f'  {name} WRONG rel={err:.1e} cfg={cfg}', flush=True)
            continue
        t = timeit(None, over)
        rows.append((t, err, cfg, meta, name))
    rows.sort()
    for t, err, cfg, meta, name in rows[:6]:
        print(f'  {1e3*t:8.3f} ms  {t_ref/t:6.2f}x  '
              f'{meta["threadsx"]}x{meta["threadsy"]}x{meta["gout_stride"]} '
              f'tiley={meta["tiley"]:2d} lreg={meta["lreg"]} '
              f'mb={meta["minblocks"]} dm={"reg" if meta["dmreg"] else "shm"} '
              f'shm={meta["shm"]*8/1024:5.1f}KB rel={err:.0e}', flush=True)
    if rows:
        t, err, cfg, meta, name = rows[0]
        best[tag] = dict(cfg=cfg, t=t, t_ref=t_ref, t_shipped=t_now,
                         speedup=t_ref / t, meta=meta)
        print(f'  BEST {tag}: {t_ref/t:.2f}x against GPU4PySCF, '
              f'{t_now/t:.2f}x against the shipped configuration', flush=True)

print('\nRESULT ' + json.dumps({k: {kk: vv for kk, vv in v.items()
                                    if kk != 'meta'}
                                for k, v in best.items()}), flush=True)
if a.write and best:
    for tag, r in best.items():
        gen_j_high.CFG[tag].update(r['cfg'])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                       'codegen', 'gen_j_high.py')
    print('\nWinning configurations (paste into codegen/gen_j_high.py CFG):')
    for tag, r in best.items():
        c = r['cfg']
        print(f"    '{tag}': dict(threadsx={c['threadsx']}, "
              f"threadsy={c['threadsy']}, gout_stride={c['gout_stride']}, "
              f"tilex={c['tilex']}, tiley={c['tiley']},\n"
              f"                lreg={c['lreg']}, minblocks={c['minblocks']}, "
              f"dmreg={c['dmreg']}),")
