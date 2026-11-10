"""Independent DF-RHF energy audits for stringent nuclear differences.

These evaluate the same converged density and fitting basis without
altering the SCF solution or correlation model. They are validation
oracles, not replacements for the reference SCF or its response kernel.
"""
import time

import numpy as np


def reference_energy(mf, backend='scf'):
    """Return an independently evaluated reference energy and provenance.

    ``legacy-gpu`` streams the older GPU4PySCF three-center integral
    engine into occupied blocks. The one- and two-center integrals use
    libcint, independently of the newer GPU Rys implementation. Its
    largest persistent fitted tensor is auxiliary by occupied by occupied;
    a full AO three-center tensor is never allocated.
    """
    if backend not in ('scf', 'cpu', 'legacy-gpu'):
        raise ValueError('Unknown HF energy validation backend')
    start = time.perf_counter()
    original = float(mf.e_tot)
    if backend == 'scf':
        return dict(backend=backend, energy=original, original_energy=original,
                    correction=0., seconds=time.perf_counter()-start)
    cpu = mf.to_cpu() if hasattr(mf, 'to_cpu') else mf
    if getattr(cpu, 'with_df', None) is None:
        raise ValueError('Independent reference energy requires density-fitted RHF')
    occupations = np.asarray(cpu.mo_occ)
    if (cpu.mol.spin != 0 or occupations.ndim != 1
            or not np.all((occupations == 0) | (occupations == 2))):
        raise ValueError('Independent reference energy requires closed-shell RHF occupations')
    dm = cpu.make_rdm1()
    hcore = cpu.get_hcore()
    info = dict(backend=backend, original_energy=original,
                auxiliary_basis=cpu.with_df.auxbasis,
                scf_recomputed=False, density_reoptimized=False)
    if backend == 'cpu':
        j, k = cpu.get_jk(dm=dm, hermi=1)
        energy = float(np.einsum('ij,ji->', dm, hcore+.5*(j-.5*k)))+cpu.mol.energy_nuc()
    else:
        import cupy as cp
        from cupyx.scipy.linalg import solve_triangular
        from pyscf import df
        from gpu4pyscf.df.int3c2e import VHFOpt, _int3c2e_jk_task

        mol = cpu.mol
        aux = df.addons.make_auxmol(mol, cpu.with_df.auxbasis)
        coeff = cp.asarray(cpu.mo_coeff[:, occupations > 0])
        opt = VHFOpt(mol, aux, 'int2e')
        opt.build(1e-14, diag_block_with_triu=True, aosym=True,
                  group_size=128, group_size_aux=128)
        sorted_coeff = opt.sort_orbitals(coeff, axis=[0])
        sorted_density = 2*sorted_coeff@sorted_coeff.T
        charge, occupied = _int3c2e_jk_task(
            opt, range(len(opt.aux_log_qs)), sorted_density, sorted_coeff,
            device_id=cp.cuda.Device().id)
        # The streamed engine returns auxiliary rows in its sorted order.
        occupied = opt.unsort_orbitals(occupied, aux_axis=[0])
        charge = opt.unsort_orbitals(charge, aux_axis=[0])
        metric = cp.asarray(aux.intor('int2c2e'))
        factor = cp.linalg.cholesky(metric)
        fitted = solve_triangular(factor, occupied.reshape(aux.nao, -1), lower=True)
        fitted_charge = solve_triangular(factor, charge, lower=True)
        coulomb = .5*float(cp.dot(fitted_charge, fitted_charge))
        exchange = -float(cp.einsum('Qi,Qi->', fitted, fitted))
        one_electron = float(np.einsum('ij,ji->', dm, hcore))
        energy = one_electron+coulomb+exchange+mol.energy_nuc()
        info.update(occupied_integral_bytes=occupied.nbytes, metric_bytes=metric.nbytes,
                    full_ao_three_center_allocated=False,
                    integral_backend='gpu4pyscf.df.int3c2e',
                    metric_backend='libcint', hcore_backend='libcint')
    info.update(energy=float(energy), correction=float(energy)-original,
                seconds=time.perf_counter()-start)
    return info
