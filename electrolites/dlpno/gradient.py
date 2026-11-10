"""Closed-shell DLPNO analytic nuclear-gradient reference driver.

This first implementation prioritizes the derivative of the actual local
energy (including subspace and screening corrections) over scaling.  It
uses analytic AO derivatives and direct coupled responses for each nuclear
coordinate, rather than a production Z-vector/density contraction.  The
energy may come from the GPU driver.  ``backend='cupy'`` evaluates tensor
responses on one GPU; AO derivatives, SCF response, primal amplitude
iterations and Krylov control remain on the CPU.
"""
import numpy as np

from .orbital_response import RHFResponse
from .response import ResponseDLPNO


class Gradients:
    """Analytic dE/dR in Eh/bohr for RHF DLPNO-MP2, CCSD, CCSD(T0/T1).

    ``level='mp2'`` differentiates the preparatory corrected LMP2 energy;
    ``level='ccsd'`` omits triples.  ``level='auto'`` follows the energy
    object's triples setting.  Hard domain/rank decisions are locally
    constant; a finite-difference validation must check their stability.
    """

    def __init__(self, base, level='auto', energy_tolerance=2e-7, backend='numpy'):
        if level not in ('auto', 'mp2', 'ccsd'):
            raise ValueError("level must be 'auto', 'mp2', or 'ccsd'")
        self.base, self.level = base, level
        self.energy_tolerance = energy_tolerance
        if backend not in ('numpy', 'cupy'):
            raise ValueError("backend must be 'numpy' or 'cupy'")
        self.backend = backend
        self.de = None
        self.components = {}
        self.diagnostics = []

    def kernel(self, atmlst=None):
        base = self.base
        if hasattr(base, 'build_solver'):
            if base.solver is None:
                base.kernel()
            base = base.solver
        if not base.e:
            base.kernel()
        if self.level == 'auto':
            level = 'ccsd_t' if base.triples else 'ccsd'
        else:
            level = self.level
        atoms = list(range(base.mol.natm)) if atmlst is None else list(atmlst)
        if len(set(atoms)) != len(atoms) or any(a < 0 or a >= base.mol.natm for a in atoms):
            raise ValueError('atmlst must contain distinct valid atom indices')
        row = {a: k for k, a in enumerate(atoms)}
        reference = RHFResponse(base)
        self.de = reference.hf_gradient[atoms].copy()
        self.components = {'hf': self.de.copy()}
        self.diagnostics = []
        target = 'lmp2' if level == 'mp2' else ('corr' if level == 'ccsd_t' else 'ccsd')
        for direction in reference.directions():
            atom, xyz = direction.atom, direction.xyz
            if atom not in row:
                continue
            response = ResponseDLPNO(base, reference, direction, level, backend=self.backend)
            corr = response.kernel()
            energy_error = float(corr.v - base.e[target])
            if abs(energy_error) > self.energy_tolerance:
                raise RuntimeError(f'Response energy does not match {target}: difference={energy_error:.3e} Eh')
            self.de[row[atom], xyz] += float(corr.d)
            for key, value in response.e.items():
                if key not in self.components:
                    self.components[key] = np.zeros_like(self.de)
                self.components[key][row[atom], xyz] = float(getattr(value, 'd', 0.))
            self.diagnostics.append(dict(atom=atom, xyz=xyz, energy_error=energy_error,
                                         orbital=reference.diagnostics[-1], amplitudes=response.diagnostics))
        if not np.isfinite(self.de).all():
            raise FloatingPointError('Nonfinite DLPNO gradient')
        return self.de

    run = kernel
