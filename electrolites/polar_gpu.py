"""Static and frequency-dependent dipole polarizability on the GPU, RHF/RKS
and UHF/UKS (`docs/FEATURE_TARGETS_PYSCF_2.13.md` O3).

GPU4PySCF's ``properties.polarizability`` is static and restricted only
(an assertion on UHF/UKS); pyscf-properties' ``polarizability_with_freq``
(and its UHF twin) is on the CPU with SciPy's ``newton_krylov``.  Here the
same linear-response problem,

    [A - w   B    ] [X]     [h]
    [B       A + w] [Y] = - [h],     alpha_xy(w) = -sum (h_x . (X_y + Y_y)) * occ,

h the dipole integrals (origin at the nuclear charge centre, as
pyscf-properties), for all three field directions at once on the device:
transition densities C_v X C_o^T + C_o Y^T C_v^T, the SCF's own response
(``mf.gen_response(hermi=0)``: J, fxc, scaled K), and the refined Krylov
solver of ``electrolites.nmr_uks`` on the preconditioned system with
pyscf-properties' diagonal (e_ai -+ w, raised by 0.1 where below 0.1).

    a = electrolites.polar_gpu.polarizability(gpu_mf, freq=0.05)  # (3, 3)
"""
import numpy as np
import cupy as cp


def _dipole_ints(mol):
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    centre = np.einsum('i,ix->x', charges, coords) / charges.sum()
    with mol.with_common_orig(centre):
        return cp.asarray(mol.intor_symmetric('int1e_r', comp=3))


def polarizability(mf, freq=0., tol=1e-9, max_cycle=50):
    """alpha(freq), (3, 3) a.u., for a GPU4PySCF RHF/RKS or UHF/UKS."""
    from electrolites.nmr_uks import krylov_refined
    mol = mf.mol
    int_r = _dipole_ints(mol)
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asnumpy(mf.mo_occ)
    mo_energy = cp.asarray(mf.mo_energy)
    restricted = mo_coeff.ndim == 2
    if restricted:
        mo_coeff, mo_energy, mo_occ = (mo_coeff[None], mo_energy[None],
                                       mo_occ[None])
        occ_fac = 2.
    else:
        occ_fac = 1.
    nspin = len(mo_coeff)
    co = [mo_coeff[s][:, mo_occ[s] > 0] for s in range(nspin)]
    cv = [mo_coeff[s][:, mo_occ[s] == 0] for s in range(nspin)]
    eo = [mo_energy[s][mo_occ[s] > 0] for s in range(nspin)]
    ev = [mo_energy[s][mo_occ[s] == 0] for s in range(nspin)]
    e_ai = [ev[s][:, None] - eo[s][None, :] for s in range(nspin)]
    h1 = [cv[s].T @ int_r @ co[s] for s in range(nspin)]       # (3, v, o)
    shift = .1
    dx = [e - freq for e in e_ai]
    dy = [e + freq for e in e_ai]
    dx = [cp.where(d < shift, d + shift, d) for d in dx]
    dy = [cp.where(d < shift, d + shift, d) for d in dy]
    sizes = [int(e.size) for e in e_ai]
    vresp = mf.gen_response(hermi=0)

    def split(v):
        out, p = [], 0
        for s in range(nspin):
            n = sizes[s]
            x = v[:, p:p + n].reshape(-1, *e_ai[s].shape)
            y = v[:, p + n:p + 2 * n].reshape(-1, *e_ai[s].shape)
            out.append((x, y))
            p += 2 * n
        return out

    def join(parts):
        return cp.hstack([cp.hstack([x.reshape(len(x), -1),
                                     y.reshape(len(y), -1)])
                          for x, y in parts])

    def aop(v):
        v = v.reshape(-1, sum(2 * n for n in sizes))
        parts = split(v)
        dms = []
        for s, (x, y) in enumerate(parts):
            d = (cv[s] @ x @ co[s].T
                 + co[s] @ y.transpose(0, 2, 1) @ cv[s].T) * occ_fac
            dms.append(d)
        if restricted:
            v1 = cp.asarray(vresp(dms[0]))[None]
        else:
            v1 = cp.asarray(vresp(cp.stack(dms)))
        out = []
        for s, (x, y) in enumerate(parts):
            vvo = cv[s].T @ v1[s] @ co[s]
            vov = (co[s].T @ v1[s] @ cv[s]).transpose(0, 2, 1)
            vvo = vvo + (e_ai[s] - freq - dx[s]) * x
            vov = vov + (e_ai[s] + freq - dy[s]) * y
            out.append((vvo / dx[s], vov / dy[s]))
        return join(out)

    b = join([(-h1[s] / dx[s], -h1[s] / dy[s]) for s in range(nspin)])
    sol = krylov_refined(aop, b, tol, max_cycle)
    alpha = cp.zeros((3, 3))
    for s, (x, y) in enumerate(split(sol)):
        alpha += cp.einsum('xai,yai->xy', h1[s], x + y)
    return cp.asnumpy(-alpha * occ_fac)
