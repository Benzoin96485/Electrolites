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
    b = electrolites.polar_gpu.hyperpolarizability(gpu_mf)        # (3, 3, 3)

``hyperpolarizability`` is the static beta by the 2n+1 rule on the same
first-order orbitals (pyscf-properties' expression, which it has on the
CPU for RHF/RKS and UHF/UKS) plus, for LDA/GGA and global hybrids, the
term int kxc rho_x rho_y rho_z that pyscf-properties leaves out (1.6 a.u.
of 16 on water/PBE).  Against a finite field of the polarizability:
<= 1.4e-3 a.u. closed and open shell (``tests/test_hyperpolar_gpu.py``).
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


def _response_U(mf, tol, max_cycle):
    """Static first-order orbital responses U_x (vo) to the three dipole
    perturbations, with h1 and the orbitals; per spin."""
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
    nspin = len(mo_coeff)
    occ_fac = 2. if restricted else 1.
    co = [mo_coeff[s][:, mo_occ[s] > 0] for s in range(nspin)]
    cv = [mo_coeff[s][:, mo_occ[s] == 0] for s in range(nspin)]
    e_ai = [mo_energy[s][mo_occ[s] == 0][:, None]
            - mo_energy[s][mo_occ[s] > 0][None, :] for s in range(nspin)]
    h1 = [cv[s].T @ int_r @ co[s] for s in range(nspin)]
    sizes = [int(e.size) for e in e_ai]
    vresp = mf.gen_response(hermi=1)

    def dms_of(us):
        out = []
        for s in range(nspin):
            d = cv[s] @ us[s] @ co[s].T * occ_fac
            out.append(d + d.transpose(0, 2, 1))
        return out

    def resp(dms):
        if restricted:
            return cp.asarray(vresp(dms[0]))[None]
        return cp.asarray(vresp(cp.stack(dms)))

    def split(v):
        out, p = [], 0
        for s in range(nspin):
            out.append(v[:, p:p + sizes[s]].reshape(-1, *e_ai[s].shape))
            p += sizes[s]
        return out

    def aop(v):
        us = split(v.reshape(-1, sum(sizes)))
        v1 = resp(dms_of(us))
        return cp.hstack([((cv[s].T @ v1[s] @ co[s]) / e_ai[s]
                           ).reshape(len(us[s]), -1) for s in range(nspin)])

    b = cp.hstack([(-h1[s] / e_ai[s]).reshape(3, -1) for s in range(nspin)])
    us = split(krylov_refined(aop, b, tol, max_cycle))
    return dict(us=us, co=co, cv=cv, int_r=int_r, occ_fac=occ_fac,
                restricted=restricted, nspin=nspin, dms=dms_of(us),
                resp=resp)


def _kxc_term(mf, dms1, restricted):
    """K[x,y,z] = int kxc rho_x rho_y rho_z over the SCF grid (LDA/GGA)."""
    from gpu4pyscf.dft import numint
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    if xctype not in ('LDA', 'GGA'):
        raise NotImplementedError(f'hyperpolarizability: {xctype} functionals')
    mol = mf.mol
    grids = mf.grids
    if grids.coords is None:
        grids.build()
    coords = cp.asarray(grids.coords)
    weights = cp.asarray(grids.weights)
    dm0 = cp.asarray(mf.make_rdm1())
    deriv = 0 if xctype == 'LDA' else 1
    K = cp.zeros((3, 3, 3))
    blk = 20000
    for p0 in range(0, len(weights), blk):
        ao = numint.eval_ao(mol, coords[p0:p0 + blk], deriv=deriv)
        if deriv == 0:
            ao = ao[None]
        w = weights[p0:p0 + blk]

        def rho(d):                     # (ncomp, ng) for symmetric d
            c0 = ao[0] @ d
            r = [(c0 * ao[0]).sum(axis=1)]
            for k in range(1, ao.shape[0]):
                r.append(2 * (c0 * ao[k]).sum(axis=1))
            return cp.stack(r)
        if restricted:
            r0 = rho(dm0)
            k3 = ni.eval_xc_eff(mf.xc, r0 if xctype == 'GGA' else r0[0],
                                deriv=3, xctype=xctype, spin=0)[3]
            k3 = k3.reshape(r0.shape[0], r0.shape[0], r0.shape[0], -1)
            r1 = cp.stack([rho(d) for d in dms1[0]])          # (3, nc, ng)
            K += cp.einsum('abcg,xag,ybg,zcg,g->xyz', k3, r1, r1, r1, w)
        else:
            r0 = cp.stack([rho(dm0[0]), rho(dm0[1])])          # (2, nc, ng)
            k3 = ni.eval_xc_eff(mf.xc, r0 if xctype == 'GGA' else r0[:, 0],
                                deriv=3, xctype=xctype, spin=1)[3]
            nc = r0.shape[1]
            k3 = k3.reshape(2, nc, 2, nc, 2, nc, -1)
            r1 = cp.stack([cp.stack([rho(d) for d in dms1[s]])
                           for s in (0, 1)])                    # (2, 3, nc, ng)
            K += cp.einsum('satbucg,sxag,tybg,uzcg,g->xyz', k3, r1, r1, r1,
                           w)
    return K


def hyperpolarizability(mf, tol=1e-10, max_cycle=50):
    """Static first hyperpolarizability beta (3, 3, 3), a.u., RHF/RKS and
    UHF/UKS: pyscf-properties' 2n+1 expression (first-order orbitals,
    first-order Fock and occupied energies) plus, for KS, the third
    functional derivative of the XC energy contracted with three
    first-order densities, which pyscf-properties leaves out."""
    r = _response_U(mf, tol, max_cycle)
    us, co, cv, int_r = r['us'], r['co'], r['cv'], r['int_r']
    fac = r['occ_fac']
    v1 = r['resp'](r['dms'])
    e3 = cp.zeros((3, 3, 3))
    for s in range(r['nspin']):
        h1ao = int_r + v1[s]                                   # (3, nao, nao)
        fvv = cv[s].T @ h1ao @ cv[s]                           # (3, v, v)
        foo = co[s].T @ h1ao @ co[s]                           # (3, o, o)
        U = us[s]                                              # (3, v, o)
        e3 += cp.einsum('xab,yai,zbi->xyz', fvv, U, U) * fac
        e3 -= cp.einsum('xai,yaj,zij->xyz', U, U, foo) * fac
    e3 = (e3 + e3.transpose(1, 2, 0) + e3.transpose(2, 0, 1)
          + e3.transpose(0, 2, 1) + e3.transpose(1, 0, 2)
          + e3.transpose(2, 1, 0))
    if hasattr(mf, 'xc') and mf._numint._xc_type(mf.xc) != 'HF':
        if mf._numint.rsh_and_hybrid_coeff(mf.xc)[0] != 0:
            raise NotImplementedError('hyperpolarizability: range-separated')
        e3 += _KXC_FACTOR * _kxc_term(mf, r['dms'], r['restricted'])
    return cp.asnumpy(-e3)


_KXC_FACTOR = 1.
