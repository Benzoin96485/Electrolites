"""Phase 6 timings: one step of each item, GPU (after a warm-up) or the CPU
reference (``--cpu``).

    python benchmarks/phase6_bench.py --what lo --method boys [--cpu]

what: lo (method: boys | pm | ibo | er) | eom (method: ip | ea | ee | et | sf | lip | lea) | lambda | rdm | adc (method: ee | ip | ea) | kmp2 (--kmesh, --cell diamond|diamond8) | x2cgrad | ghftd (method: tda | tdhf)
"""
import argparse
import json
import os
import time

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--what', required=True, choices=('lo', 'eom', 'adc', 'kmp2', 'x2cgrad', 'ghftd', 'lambda',
                         'rdm'))
ap.add_argument('--method', default='boys')
ap.add_argument('--cpu', action='store_true')
ap.add_argument('--xyz', default='benchmarks/molecules/glycine.xyz')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='hf')
ap.add_argument('--charge', type=int, default=0)
ap.add_argument('--spin', type=int, default=0)
ap.add_argument('--nroots', type=int, default=4)
ap.add_argument('--kmesh', type=int, default=2)
ap.add_argument('--cell', default='diamond')
a = ap.parse_args()
rec = dict(vars(a), threads=os.environ.get('OMP_NUM_THREADS'))


def sync():
    if not a.cpu:
        import cupy
        cupy.cuda.Stream.null.synchronize()


def timed(label, fn, warm=False):
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


if a.what == 'lo':
    mol = molecule()
    if a.cpu:
        from pyscf import scf
    else:
        from gpu4pyscf import scf
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    timed('scf_s', mf.kernel)
    from pyscf.data import elements
    ncore = elements.chemcore(mol)
    occ = np.asarray(mf.mo_occ.get() if hasattr(mf.mo_occ, 'get')
                     else mf.mo_occ) > 0
    mo = mf.mo_coeff[:, ncore:int(occ.sum())]
    rec['nao'], rec['nloc'] = mol.nao, mo.shape[1]
    if a.cpu:
        from pyscf import lo
        L = {'boys': lambda: lo.Boys(mol, mo).kernel(),
             'pm': lambda: lo.PM(mol, mo, mf).kernel(),
             'ibo': lambda: lo.ibo.ibo(mol, mo, locmethod='IBO'),
             'er': lambda: lo.ER(mol, mo).kernel()}[a.method]
    else:
        from electrolites import lo_gpu
        L = {'boys': lambda: lo_gpu.Boys(mol, mo).kernel(),
             'pm': lambda: lo_gpu.PM(mol, mo, mf).kernel(),
             'ibo': lambda: lo_gpu.ibo(mol, mo),
             'er': lambda: lo_gpu.ER(mol, mo).kernel()}[a.method]
    c = timed('step_s', L, warm=True)
    c = c.get() if hasattr(c, 'get') else np.asarray(c)
    s = mol.intor_symmetric('int1e_ovlp')
    with mol.with_common_orig(np.einsum('z,zx->x', mol.atom_charges(),
                                        mol.atom_coords())
                              / mol.atom_charges().sum()):
        r = mol.intor_symmetric('int1e_r', comp=3)
    dip = np.einsum('pi,xpq,qi->xi', c, r, c)
    rec['boys_cost'] = float((dip ** 2).sum())
elif a.what == 'eom':
    from pyscf.data import elements
    mol = molecule()
    fc = elements.chemcore(mol)
    if a.cpu:
        from pyscf import scf, cc
        from pyscf.cc import eom_rccsd as E
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        timed('scf_s', mf.kernel)
        mycc = cc.CCSD(mf, frozen=fc)
        timed('ccsd_s', mycc.kernel)
        eom = {'ip': E.EOMIP, 'ea': E.EOMEA, 'ee': E.EOMEESinglet,
               'et': E.EOMEETriplet, 'sf': E.EOMEESpinFlip,
               'lip': E.EOMIP, 'lea': E.EOMEA}[a.method](mycc)
    else:
        from gpu4pyscf import scf
        from gpu4pyscf.cc import ccsd_incore
        from electrolites import eom_gpu as E
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        timed('scf_s', mf.kernel)
        mycc = ccsd_incore.CCSD(mf, frozen=fc)
        timed('ccsd_s', mycc.kernel)
        eom = {'ip': E.EOMIP, 'ea': E.EOMEA, 'ee': E.EOMEESinglet,
               'et': E.EOMEETriplet, 'sf': E.EOMEESpinFlip,
               'lip': E.EOMIP, 'lea': E.EOMEA}[a.method](mycc)
    eom.conv_tol = 1e-7
    kw = {'left': True} if a.method in ('lip', 'lea') else {}
    e = timed('step_s', lambda: eom.kernel(nroots=a.nroots, **kw)[0])
    rec['e_eV'] = (np.asarray(e) * 27.211386).tolist()
    rec['nao'], rec['nfrozen'] = mol.nao, fc
elif a.what == 'lambda':
    from pyscf.data import elements
    mol = molecule()
    fc = elements.chemcore(mol)
    if a.cpu:
        from pyscf import scf, cc
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        timed('scf_s', mf.kernel)
        mycc = cc.CCSD(mf, frozen=fc)
        timed('ccsd_s', mycc.kernel)
        timed('step_s', mycc.solve_lambda)
    else:
        from gpu4pyscf import scf
        from gpu4pyscf.cc import ccsd_incore
        from electrolites import ccsd_lambda_gpu
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        timed('scf_s', mf.kernel)
        mycc = ccsd_incore.CCSD(mf, frozen=fc)
        timed('ccsd_s', mycc.kernel)
        timed('step_s', lambda: ccsd_lambda_gpu.kernel(
            mycc, tol=mycc.conv_tol_normt))
    rec['l1_norm'] = float(np.linalg.norm(mycc.l1))
    rec['nao'], rec['nfrozen'] = mol.nao, fc
elif a.what == 'rdm':
    from pyscf.data import elements
    mol = molecule()
    fc = elements.chemcore(mol)
    if a.cpu:
        from pyscf import scf, cc
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        timed('scf_s', mf.kernel)
        mycc = cc.CCSD(mf, frozen=fc)
        timed('ccsd_s', mycc.kernel)
        timed('lambda_s', mycc.solve_lambda)
        timed('rdm1_s', mycc.make_rdm1)
        timed('rdm2_s', mycc.make_rdm2)
    else:
        from gpu4pyscf import scf
        from gpu4pyscf.cc import ccsd_incore
        from electrolites import ccsd_lambda_gpu, ccsd_rdm_gpu  # noqa
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        timed('scf_s', mf.kernel)
        mycc = ccsd_incore.CCSD(mf, frozen=fc)
        timed('ccsd_s', mycc.kernel)
        timed('lambda_s', mycc.solve_lambda)
        timed('rdm1_s', mycc.make_rdm1, True)
        timed('rdm2_s', mycc.make_rdm2, True)
    rec['nao'], rec['nfrozen'] = mol.nao, fc
elif a.what == 'adc':
    from pyscf.data import elements
    mol = molecule()
    fc = elements.chemcore(mol)
    if a.cpu:
        from pyscf import scf, adc
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        timed('scf_s', mf.kernel)
        my = adc.ADC(mf, frozen=fc)
        my.method_type = a.method
        my.compute_properties = False
        my.approx_trans_moments = True
    else:
        from gpu4pyscf import scf
        from electrolites import adc_gpu
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        timed('scf_s', mf.kernel)
        my = adc_gpu.RADC(mf, frozen=fc, method_type=a.method)
    my.conv_tol = 1e-7
    e = timed('step_s', lambda: my.kernel(nroots=a.nroots)[0])
    rec['e_eV'] = (np.asarray(e) * 27.211386).tolist()
    rec['nao'], rec['nfrozen'] = mol.nao, fc
elif a.what == 'kmp2':
    from pyscf.pbc import gto as pgto
    if a.cell == 'diamond8':
        cell = pgto.M(a=np.eye(3) * 3.5668, atom='''C 0 0 0; C 0 1.7834 1.7834
C 1.7834 0 1.7834; C 1.7834 1.7834 0; C 0.8917 0.8917 0.8917
C 0.8917 2.6751 2.6751; C 2.6751 0.8917 2.6751; C 2.6751 2.6751 0.8917''',
                      basis='gth-dzvp', pseudo='gth-pade', verbose=0)
    else:
        cell = pgto.M(a='''0 1.7834 1.7834
1.7834 0 1.7834
1.7834 1.7834 0''', atom='C 0 0 0; C 0.8917 0.8917 0.8917',
                      basis='gth-dzvp', pseudo='gth-pade', verbose=0)
    kpts = cell.make_kpts([a.kmesh] * 3)
    if a.cpu:
        from pyscf.pbc import scf, mp
        mf = scf.KRHF(cell, kpts).density_fit()
        mf.conv_tol = 1e-9
        timed('scf_s', mf.kernel)
        m = mp.KMP2(mf)
        m.max_memory = 400000
        e = timed('step_s', lambda: m.kernel(with_t2=False)[0])
    else:
        from gpu4pyscf.pbc import scf
        from gpu4pyscf.pbc.df.df import GDF
        from electrolites import kmp2_gpu
        mf = scf.KRHF(cell, kpts)
        mf.with_df = GDF(cell, kpts)
        mf.conv_tol = 1e-9
        timed('scf_s', mf.kernel)
        e = timed('step_s', lambda: kmp2_gpu.kernel(mf), True)
    rec['e_corr'] = float(e)
    rec['nao'], rec['nk'] = cell.nao, len(kpts)
elif a.what == 'x2cgrad':
    from pyscf import gto
    heavy = {'Pt': 'dyall-v2z', 'I': 'dyall-v2z', 'Br': 'cc-pvdz-dk'}
    mol = gto.M(atom=a.xyz, charge=a.charge, spin=a.spin, verbose=0,
                basis={**{e: 'cc-pvdz-dk' for e in ('H', 'C', 'N', 'O', 'Cl',
                                                   'S')}, **heavy})
    if a.cpu:
        from pyscf import dft, grad  # noqa: F401 (registers Gradients)
        mf = (dft.UKS if a.spin else dft.RKS)(mol, xc='pbe').x2c()
        mf.conv_tol = 1e-9
        timed('scf_s', mf.kernel)
        g = mf.Gradients()
        timed('step_s', g.kernel)
    else:
        from gpu4pyscf import dft
        from gpu4pyscf.x2c.sfx2c1e import sfx2c1e
        from electrolites import x2c_grad
        mf = sfx2c1e((dft.UKS if a.spin else dft.RKS)(mol, xc='pbe'))
        mf.conv_tol = 1e-9
        timed('scf_s', mf.kernel)
        g = x2c_grad.Gradients(mf)
        timed('step_s', g.kernel, True)
    rec['nao'] = mol.nao
    rec['e_tot'] = float(mf.e_tot)
elif a.what == 'ghftd':
    mol = molecule()
    if a.cpu:
        from pyscf import scf
        from pyscf.tdscf import ghf as T
        mf = scf.GHF(mol)
    else:
        from gpu4pyscf import scf
        from electrolites import ghf_td_gpu as T
        mf = scf.GHF(mol)
    mf.conv_tol = 1e-9
    timed('scf_s', mf.kernel)
    td = (T.TDA if a.method == 'tda' else T.TDHF)(mf)
    td.nstates = a.nroots
    e = timed('step_s', lambda: td.kernel()[0])
    rec['e_eV'] = (np.asarray(e) * 27.211386).tolist()
    rec['nao'] = mol.nao
side = 'CPU' if a.cpu else 'GPU'
label = (f"{a.cell} k{a.kmesh}" if a.what == "kmp2" else f"{os.path.basename(a.xyz)}/{a.basis}")
print(f"[{a.what} {a.method} {side} {label}] "
      + '  '.join(f'{k} {v:.2f}' for k, v in rec.items()
                  if k.endswith('_s') and isinstance(v, float)))
print('RESULT ' + json.dumps(rec, default=str))
