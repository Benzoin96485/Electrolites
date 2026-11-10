"""CPU setup for the existing pair-by-pair DLPNO reference equations.

The GPU production driver is unchanged.  This small independent setup is
useful for response verification on machines without a CUDA device.
"""
from types import FunctionType, SimpleNamespace

import numpy as np
from pyscf import lo
from pyscf.dft import numint
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, minres, eigsh

from . import setup as su
from .ref_rhf import RefDLPNO


def rebind(function, **names):
    """Bind numeric dependencies locally; never monkey-patch module globals."""
    namespace = dict(function.__globals__, **names)
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def orbitals(mol, s, f, c, nocc, ncore, localizer='boys', c_lmo=None,
             mf=None, conv_tol=1e-12, ensure_minimum=False):
    localization = dict(verified_minimum=False, supplied=c_lmo is not None)
    if c_lmo is None:
        if localizer.lower() in ('boys', 'fb'):
            loc = lo.Boys(mol, c[:, ncore:nocc])
        elif localizer.lower() in ('pm', 'pipek', 'pipek_mezey'):
            loc = lo.PM(mol, c[:, ncore:nocc], mf)
            loc.pop_method = 'mulliken'
        else:
            raise ValueError(localizer)
        loc.init_guess = None
        loc.conv_tol = conv_tol
        loc.conv_tol_grad = min(1e-9, np.sqrt(conv_tol * .1))
        loc.max_cycle = 200
        loc.verbose = 0
        c_lmo = loc.kernel()
        if localizer.lower() in ('boys', 'fb'):
            c_lmo = stationary_boys(mol, c_lmo)
            if ensure_minimum:
                c_lmo, localization = minimum_boys(mol, c_lmo)
    result = su.Orbitals(mol, s, f, c, nocc, ncore, localizer, c_lmo=c_lmo,
                         mf=mf, conv_tol=conv_tol)
    result.localization_info = localization
    return result


def stationary_boys(mol, coefficients, tol=1e-10):
    """Polish the final CIAH orbitals using the actual final gradient.

    CIAH's reported gradient can refer to the preceding keyframe.  A small
    residual is harmless for energies but amplified by finite differences
    of a truncated local correlation energy.
    """
    c = np.array(coefficients, copy=True)
    if c.shape[1] < 2:
        return c
    for _ in range(12):
        loc = lo.Boys(mol, c)
        g, hop, _ = loc.gen_g_hop(np.eye(c.shape[1]))
        if np.max(abs(g)) < tol:
            return c
        h = LinearOperator((len(g), len(g)), matvec=hop, rmatvec=hop, dtype=float)
        step, status = minres(h, -g, rtol=1e-11, maxiter=1000)
        if status or np.max(abs(hop(step)+g)) > max(tol*.1, 1e-11):
            raise RuntimeError('Boys polishing Hessian solve did not converge')
        if np.max(abs(step), initial=0.) > .1:
            raise RuntimeError('Boys polishing is outside the local stationary basin')
        c = c @ expm(loc.unpack_uniq_var(step))
    raise RuntimeError('Boys polishing did not reach its gradient tolerance')


def boys_curvature(mol, coefficients):
    """Lowest diagonally scaled Hessian mode; its sign tests a minimum.

    Congruence scaling preserves inertia and avoids the poor conditioning
    of distant orbital rotations in long chains.  The unscaled Rayleigh
    quotient is reported separately; it is not the lowest physical
    Hessian eigenvalue.  No orbital-pair Hessian is assembled.
    """
    n = coefficients.shape[1]
    loc = lo.Boys(mol, coefficients)
    g, hop, diagonal = loc.gen_g_hop(np.eye(n))
    if not len(g):
        return dict(scaled_curvature=0., mode_curvature=0., residual=0.,
                    stationary_residual=0.), np.empty(0)
    scale = 1/np.sqrt(np.maximum(abs(diagonal), 1.))
    action = lambda x: scale*hop(scale*x)
    operator = LinearOperator((len(g), len(g)), matvec=action, dtype=float)
    if len(g) == 1:
        mode = np.ones(1)
        value = action(mode)[0]
    else:
        values, modes = eigsh(operator, k=1, which='SA', tol=1e-9,
                             v0=np.cos(np.arange(len(g))), maxiter=2000)
        value, mode = values[0], modes[:, 0]
    error = float(max(abs(action(mode)-value*mode)))
    if error > 1e-7:
        raise RuntimeError(f'Boys curvature check did not converge: {error:.3e}')
    direction = scale*mode
    direction /= np.linalg.norm(direction)
    return dict(scaled_curvature=float(value), mode_curvature=float(direction@hop(direction)),
                residual=error, stationary_residual=float(max(abs(g)))), direction


def minimum_boys(mol, coefficients, tol=1e-10):
    """Repair stationary Boys saddles with monotonic Jacobi sweeps.

    A zero localization gradient is not a minimum certificate.  Each
    two-orbital Jacobi rotation maximizes the squared dipole centers
    exactly.  A collective negative mode is displaced before the next
    sweep.  Newton polishing is used only after descent has converged,
    followed by a fresh curvature check.
    """
    c = np.array(coefficients, copy=True)
    n = c.shape[1]
    repairs = 0
    for attempt in range(9):
        info, direction = boys_curvature(mol, c)
        if info['scaled_curvature'] >= -1e-8 and info['stationary_residual'] < tol:
            return c, dict(info, verified_minimum=True, saddle_repairs=repairs)
        if attempt == 8:
            raise RuntimeError('Boys localization did not reach a verified minimum')
        repairs += 1
        loc = lo.Boys(mol, c)
        if info['scaled_curvature'] < -1e-8:
            # Escape a collective saddle even if every individual pair
            # rotation is locally stable.  Select the lower-cost sign.
            rotations = [expm(sign*.03*loc.unpack_uniq_var(direction)) for sign in (-1, 1)]
            rotation = min(rotations, key=loc.cost_function)
            c = c@rotation
        dip = lo.boys.dipole_integral(mol, c)
        rotation = np.eye(n)
        for sweep in range(200):
            largest = 0.
            for i in range(n):
                for j in range(i):
                    a, b = (dip[:, j, j]-dip[:, i, i])*.5, dip[:, j, i]
                    theta = .25*np.arctan2(2*a@b, a@a-b@b)
                    largest = max(largest, abs(theta))
                    if abs(theta) < 1e-14:
                        continue
                    co, si = np.cos(theta), np.sin(theta)
                    pair_rotation = np.array([[co, -si], [si, co]])
                    ids = [j, i]
                    dip[:, :, ids] = dip[:, :, ids]@pair_rotation
                    dip[:, ids, :] = pair_rotation.T@dip[:, ids, :]
                    rotation[:, ids] = rotation[:, ids]@pair_rotation
            if largest < 1e-8:
                break
        c = stationary_boys(mol, c@rotation, tol=tol)
    raise AssertionError('unreachable')


def compute_doi(mol, cl, p, grid=(25, 50), blksize=4096, need_uv=True):
    import os
    if os.environ.get('DLPNO_DOI_GRID'):
        g = np.load(os.environ['DLPNO_DOI_GRID'])
        coords, weights = g[:, :3], g[:, 3]
    else:
        coords, weights = su.doi_grid(mol, grid)
    ij, iu = np.zeros((cl.shape[1], cl.shape[1])), np.zeros((cl.shape[1], p.shape[1]))
    uv = np.zeros((p.shape[1], p.shape[1])) if need_uv else None
    for start in range(0, len(weights), blksize):
        end = min(start + blksize, len(weights))
        ao = numint.eval_ao(mol, coords[start:end])
        li, lu, w = (ao @ cl)**2, (ao @ p)**2, weights[start:end, None]
        ij += (li * w).T @ li
        iu += (li * w).T @ lu
        if need_uv:
            uv += (lu * w).T @ lu
    return np.sqrt(np.maximum(ij, 0)), np.sqrt(np.maximum(iu, 0)), (
        np.sqrt(np.maximum(uv, 0)) if need_uv else None)


def dipole_pair_energies(orb, doi_iu, t_cut_do_pre, s_cut, bf_atom, atom_bfs):
    with orb.mol.with_common_orig((0., 0., 0.)):
        r = orb.mol.intor('int1e_r', comp=3)
    cl, p = orb.C_lmo, orb.C_pao
    centers = np.einsum('mi,xmn,ni->ix', cl, r, cl, optimize=True)
    lp = np.einsum('mi,xmn,nu->ixu', cl, r, p, optimize=True)
    ds, es = [], []
    for i in range(orb.naocc):
        dom = su.contract_to_atoms(np.where(abs(doi_iu[i]) > t_cut_do_pre)[0], bf_atom, atom_bfs)
        x, e = su.orthocanonicalize(orb.S_pao[np.ix_(dom, dom)],
                                    orb.F_pao[np.ix_(dom, dom)], s_cut)
        ds.append(lp[i][:, dom] @ x)
        es.append(e)
    energy, bound = np.zeros((orb.naocc, orb.naocc)), np.zeros((orb.naocc, orb.naocc))
    for i in range(orb.naocc):
        for j in range(i + 1, orb.naocc):
            rij = centers[i] - centers[j]
            distance = np.linalg.norm(rij)
            if distance < 1e-12:
                # Coincident orbital centers are never distant pairs.
                bound[i, j] = bound[j, i] = -np.inf
                continue
            direction = rij / distance
            dot = ds[i].T @ ds[j]
            dip = dot - 3 * np.outer(direction @ ds[i], direction @ ds[j])
            den = es[i][:, None] + es[j][None, :] - orb.F_lmo[i, i] - orb.F_lmo[j, j]
            energy[i, j] = energy[j, i] = -4 / distance**6 * np.sum(dip**2 / den)
            bound[i, j] = bound[j, i] = -4 / distance**6 * np.sum((2 * dot)**2 / den)
    return energy, bound


_setup = SimpleNamespace(**{n: getattr(su, n) for n in dir(su) if not n.startswith('__')})
_setup.Orbitals = orbitals
_setup.compute_doi = compute_doi
_setup.dipole_pair_energies = dipole_pair_energies


class CPUDLPNO(RefDLPNO):
    setup = rebind(RefDLPNO.setup, su=_setup)
