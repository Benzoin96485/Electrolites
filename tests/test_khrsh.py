"""Both exchange ranges in one launch for the gen_khigh classes (A7).

``fastk.get_k_rsh`` now runs the high-l (gen_khigh) classes once with
coef0*(full range) + coef1*(long range) instead of once per operator.
Checked on glycine and the Ni dithiolene anion in def2-TZVP (f shells, so
the high-l classes carry real work) against the two single-range builds
combined, ``hyb*K + (alpha-hyb)*K_LR``, for hermi = 1 and hermi = 0, with the
fusion on and off.  Bar 1e-10 relative: the fused launch screens on the
full-range q_cond for both operators, so it keeps a few long-range quartets
the separate long-range pass drops below ``direct_scf_tol``.

    python tests/test_khrsh.py
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import cupy as cp
import electrolites
electrolites.patch_all()
from electrolites import fastk
from pyscf import gto
from gpu4pyscf import dft

ok = True
for name, charge, spin in (('glycine', 0, 0), ('tmc_ni_dithiolene', -1, 1)):
    mol = gto.M(atom=os.path.join(HERE, 'benchmarks', 'molecules', name + '.xyz'),
                basis='def2-tzvp', charge=charge, spin=spin, verbose=0)
    mf = dft.RKS(mol, xc='wb97x-d3bj')
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc)
    dm = cp.asarray(mf.get_init_guess(mol, 'minao'))
    if dm.ndim == 3:
        dm = dm[0] + dm[1]
    rng = cp.random.RandomState(0)
    for hermi in (1, 0):
        d = dm if hermi == 1 else dm + 1e-2 * rng.standard_normal(dm.shape)
        ref = (hyb * mf.get_k(mol, d, hermi)
               + (alpha - hyb) * mf.get_k(mol, d, hermi, omega))
        for off in (False, True):
            fastk._FUSED_KH_OFF = off
            opt_l = mf._opt_gpu[omega]
            opt_f = fastk._full_range_opt(opt_l)
            dms = cp.asarray(opt_f.apply_coeff_C_mat_CT(d[None]), order='C')
            vk = fastk.get_k_rsh(opt_f, opt_l, dms, hermi, hyb, alpha - hyb)
            vk = opt_f.apply_coeff_CT_mat_C(vk)[0]
            rel = float(abs(vk - ref).max() / abs(ref).max())
            good = rel < 1e-10
            ok &= good
            print(f'{name}/def2-TZVP hermi={hermi} high-l fused={not off}: '
                  f'rel |dK| {rel:.1e}  {"ok" if good else "FAIL"}', flush=True)
        fastk._FUSED_KH_OFF = False
print('ALL OK' if ok else 'FAILED')
sys.exit(0 if ok else 1)
