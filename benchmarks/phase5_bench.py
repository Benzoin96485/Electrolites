"""Phase 5 timings (FEATURE_TARGETS §16): one step of each new item, GPU
(after a warm-up on a small system) or the CPU reference (``--cpu``).

    python benchmarks/phase5_bench.py --what ktda --kmesh 2 --xc pbe [--cpu]

what: ktda | pstab | polar | ir | qmmmhess | tdpcmgrad | mep
"""
import argparse
import json
import os
import time

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--what', required=True,
                choices=('ktda', 'pstab', 'polar', 'ir', 'qmmmhess',
                         'tdpcmgrad', 'mep'))
ap.add_argument('--cpu', action='store_true')
ap.add_argument('--xyz', default='benchmarks/molecules/glycine.xyz')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--charge', type=int, default=0)
ap.add_argument('--spin', type=int, default=0)
ap.add_argument('--kmesh', type=int, default=2)
ap.add_argument('--freq', type=float, default=0.)
ap.add_argument('--grid', type=int, default=80)
ap.add_argument('--off', default='', help='env switch to set (A/B runs)')
a = ap.parse_args()
if a.off:
    os.environ[a.off] = '1'
rec = dict(vars(a), threads=os.environ.get('OMP_NUM_THREADS'))


def sync():
    if not a.cpu:
        import cupy
        cupy.cuda.Stream.null.synchronize()


def timed(label, fn, warm=False):
    """with ``warm`` on the GPU the step runs once first (kernel compilation,
    caches), its time kept as label_cold."""
    if warm and not a.cpu:
        timed(label + '_cold', fn)
    sync()
    t = time.perf_counter()
    out = fn()
    sync()
    rec[label] = time.perf_counter() - t
    return out


if not a.cpu:
    import electrolites
    electrolites.patch_all()


def molecule():
    from pyscf import gto
    return gto.M(atom=a.xyz, basis=a.basis, charge=a.charge, spin=a.spin,
                 verbose=0)


def diamond():
    from pyscf.pbc import gto
    return gto.M(a='''0 1.7834 1.7834
1.7834 0 1.7834
1.7834 1.7834 0''', atom='C 0 0 0; C 0.8917 0.8917 0.8917',
                 basis='gth-dzvp', pseudo='gth-pade', verbose=0)


if a.what in ('ktda', 'pstab'):
    cell = diamond()
    kpts = cell.make_kpts([a.kmesh] * 3)
    if a.cpu:
        from pyscf.pbc import dft, scf
    else:
        from gpu4pyscf.pbc import dft, scf
    mf = scf.KRHF(cell, kpts) if a.xc.upper() == 'HF' else \
        dft.KRKS(cell, kpts, xc=a.xc)
    mf.conv_tol = 1e-9
    timed('scf_s', mf.kernel)
    rec['nao'], rec['nk'] = cell.nao, len(kpts)
    if a.what == 'ktda':
        if a.cpu:
            from pyscf.pbc import tdscf
            td = tdscf.KTDA(mf)
            td.nstates, td.kshift_lst, td.conv_tol = 5, [0], 1e-5
            timed('step_s', td.kernel)
            rec['e'] = np.sort(np.asarray(td.e[0]))[:5].tolist()
        else:
            from electrolites.pbc_tdscf import KTDA
            td = KTDA(mf)
            td.conv_tol = 1e-5
            rec['e'] = np.sort(timed('step_s', lambda: td.kernel(5), True))[:5].tolist()
    else:
        if a.cpu:
            from pyscf.pbc.scf import stability
            timed('step_s', lambda: stability.rhf_internal(mf))
        else:
            from electrolites import pbc_stability
            rec['e'] = timed('step_s', lambda: pbc_stability.internal(mf),
                             True)[2]
elif a.what in ('polar', 'ir'):
    mol = molecule()
    rec['nao'], rec['natm'] = mol.nao, mol.natm
    unrestricted = a.spin != 0
    if a.cpu:
        from pyscf import dft
    else:
        from gpu4pyscf import dft
    mf = (dft.UKS if unrestricted else dft.RKS)(mol, xc=a.xc)
    mf.conv_tol = 1e-10
    timed('scf_s', mf.kernel)
    if a.what == 'polar':
        if a.cpu:
            from pyscf.prop.polarizability import rhf as P, uhf as PU
            p = (PU if unrestricted else P).Polarizability(mf)
            al = timed('step_s', lambda: (P if not unrestricted else PU)
                       .polarizability_with_freq(p, freq=a.freq))
        else:
            from electrolites.polar_gpu import polarizability
            al = timed('step_s', lambda: polarizability(mf, a.freq), True)
        rec['iso'] = float(np.trace(al) / 3)
    else:
        if a.cpu:
            from pyscf.prop.infrared import uks as IR
            p = IR.Infrared(mf)
            timed('step_s', p.kernel)
            rec['inten'] = np.asarray(p.ir_inten)[-3:].tolist()
        else:
            from electrolites import ir_uks
            f, i = timed('step_s', lambda: ir_uks.eval_ir_freq_intensity(
                mf, mf.Hessian()))
            rec['inten'] = np.asarray(i)[-3:].tolist()
elif a.what == 'qmmmhess':
    from gpu4pyscf import dft
    from gpu4pyscf.qmmm import itrf
    mol = molecule()
    rng = np.random.default_rng(0)
    centre = mol.atom_coords().mean(0) * 0.529177
    nmm = 200
    xyz = centre + rng.normal(0, 6.0, (nmm, 3))
    q = rng.choice([-0.8, 0.4], nmm)
    mf = itrf.mm_charge(dft.RKS(mol, xc=a.xc), xyz, q, unit='Angstrom')
    mf.conv_tol = 1e-10
    timed('scf_s', mf.kernel)
    timed('step_s', lambda: mf.Hessian().kernel())
    timed('vacuum_hess_s', lambda: mf.undo_qmmm().Hessian().kernel())
    rec['nao'], rec['nmm'] = mol.nao, nmm
elif a.what == 'tdpcmgrad':
    from gpu4pyscf import dft
    mol = molecule()
    mf = dft.RKS(mol, xc=a.xc).PCM()
    mf.with_solvent.method = 'IEF-PCM'
    mf.with_solvent.eps = 78.3553
    mf.conv_tol = 1e-10
    timed('scf_s', mf.kernel)
    td = mf.TDA(equilibrium_solvation=True)
    td.nstates = 3
    td.kernel()
    g = td.nuc_grad_method()
    g.kernel(state=1)                                     # warm-up
    timed('step_s', lambda: g.kernel(state=1))
    rec['nao'] = mol.nao
elif a.what == 'mep':
    mol = molecule()
    if a.cpu:
        from pyscf import dft
        from pyscf.tools import cubegen as C
    else:
        from gpu4pyscf import dft
        from electrolites import cubegen_gpu as C
    mf = dft.RKS(mol, xc=a.xc)
    timed('scf_s', mf.kernel)
    dm = mf.make_rdm1()
    timed('step_s', lambda: C.mep(mol, '/dev/null', dm, nx=a.grid,
                                  ny=a.grid, nz=a.grid), True)
    rec['nao'] = mol.nao
side = 'CPU' if a.cpu else 'GPU'
print(f"[{a.what} {side} {os.path.basename(a.xyz) if a.what not in ('ktda', 'pstab') else 'diamond k%d' % a.kmesh}"
      f"/{a.basis} {a.xc}{' ' + a.off if a.off else ''}] "
      + '  '.join(f'{k} {v:.2f}' for k, v in rec.items()
                  if k.endswith('_s') and isinstance(v, float)))
print('RESULT ' + json.dumps(rec, default=str))
