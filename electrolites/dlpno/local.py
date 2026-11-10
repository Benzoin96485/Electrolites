"""Shared local-integral energy and analytic-adjoint gradient driver.

Three-center integrals and their nuclear derivatives have only requested
AO supports. Optional spatial screening and a complete hierarchical tail
remove dense domain and distant-pair work. Global reference/orbital
operations remain; their costs are reported separately. No complete
scaling certification is implied by using this class.
"""
import gc
import time

import numpy as np
from pyscf import df

from . import setup as su
from .ref_rhf import RefDLPNO
from .reference_cpu import orbitals, compute_doi, dipole_pair_energies
from .local_integrals import LocalRIBlocks
from .local_blocks import BlockAdjointDLPNO
from .orbital_response import validate_reference
from .orbital_adjoint import orbital_densities, contract_orbital_densities


def _ri_provider(base, backend=None, **options):
    backend = getattr(base, 'ri_backend', 'cpu') if backend is None else backend
    if backend == 'cpu':
        return LocalRIBlocks(base, **options)
    if backend == 'gpu':
        from .gpu_integrals import GPULocalRIBlocks
        return GPULocalRIBlocks(base, **options)
    raise ValueError("ri_backend must be 'cpu' or 'gpu'")


class LocalDLPNO(RefDLPNO):
    """RHF DLPNO energy with screened, on-demand local RI blocks.

    ``lmo_cut`` and ``pao_cut`` select atom-complete coefficient supports;
    zero gives full coefficient supports.  ``pao_screen='atom'`` keeps
    complete AO atom blocks in the PAO-pair screen; ``'component'`` retains
    the legacy component screen.  Zero coefficient cutoffs with the latter
    reproduce ``CPUDLPNO``.  The gradient uses the same screens/supports as
    its parent energy and all continuous orbital response.

    ``dipole_tail_options={...}`` opts into the complete hierarchical
    distant-pair energy and its matching scalar adjoint. The default
    remains the exact pair oracle. ``local_screening_options={...}``
    additionally opts into spatial DOI/populations and local centered
    transition dipoles; these introduce explicit numerical screening
    approximations. Dense reference-orbital operations remain.
    """

    def __init__(self, *args, lmo_cut=None, pao_cut=None, cache_bytes=64 << 20,
                 pao_screen='atom', dipole_tail_options=None, screen_storage='auto',
                 local_screening_options=None, ri_backend='cpu', **kwargs):
        super().__init__(*args, **kwargs)
        if ri_backend not in ('cpu', 'gpu'):
            raise ValueError("ri_backend must be 'cpu' or 'gpu'")
        self.ri_backend = ri_backend
        self.sparse_pairs = True
        if pao_screen not in ('atom', 'component'):
            raise ValueError("pao_screen must be 'atom' or 'component'")
        self.pao_screen = pao_screen
        self.local_ri_options = dict(lmo_cut=lmo_cut, pao_cut=pao_cut, cache_bytes=cache_bytes)
        self.dipole_tail_options = None if dipole_tail_options is None else dict(dipole_tail_options)
        if screen_storage not in ('auto', 'sparse', 'dense'):
            raise ValueError("screen_storage must be 'auto', 'sparse' or 'dense'")
        self.screen_storage = ('sparse' if dipole_tail_options is not None else 'dense') if screen_storage == 'auto' else screen_storage
        if self.screen_storage == 'sparse' and dipole_tail_options is None:
            raise ValueError('Sparse screening storage requires the complete hierarchical dipole tail')
        self.local_screening_options = None if local_screening_options is None else dict(local_screening_options)
        if local_screening_options is not None and self.screen_storage != 'sparse':
            raise ValueError('Local spatial screening requires sparse storage and the complete hierarchical tail')

    def setup(self):
        t = time.perf_counter()
        mf = validate_reference(self)
        mol, opt = self.mol, self.opt
        s = np.asarray(mf.get_ovlp())
        c, occ = np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ)
        # The converged canonical RHF solution already diagonalizes F.
        # Reuse that spectral representation, as the GPU energy driver
        # does, instead of rebuilding a full CPU DF exchange matrix.
        sc = s@c
        f = (sc*np.asarray(mf.mo_energy))@sc.T
        nocc, nfc = int(np.count_nonzero(occ > 0)), su.count_frozen_core(mol)
        ncore = nfc if self.frozen_core else 0
        self.ncore_scale = 0 if self.frozen_core else nfc
        self.orb = orb = orbitals(mol, s, f, c, nocc, ncore, self.localizer,
                                  c_lmo=self.c_lmo, mf=mf, conv_tol=opt['LOCAL_CONVERGENCE'],
                                  ensure_minimum=True)
        self._orbital_reference = orb
        self.n = orb.naocc
        auxbasis = df.make_auxbasis(mol, mp2fit=True) if self.auxbasis is None else self.auxbasis
        self.auxmol = df.addons.make_auxmol(mol, auxbasis)
        (self.bf_atom, self.ribf_atom, self.atom_bfs,
         self.atom_ribfs) = su.basis_maps(mol, self.auxmol)
        t = self._tick('orbitals', t)
        if self.local_screening_options is not None:
            from .local_screening import compute_local_screening
            (self.doi_ij, self.doi_iu, self.doi_uv, self.mkn,
             self.screening_diagnostics) = compute_local_screening(self, **self.local_screening_options)
        else:
            self.__dict__.pop('_local_dipole_maps', None)
            self.__dict__.pop('dipole_preparation_diagnostics', None)
            self.doi_ij, self.doi_iu, self.doi_uv = compute_doi(
                mol, orb.C_lmo, orb.C_pao, opt['DOI_GRID'])
        if self.screen_storage == 'sparse' and self.local_screening_options is None:
            from .screening import SparseValues
            self.doi_ij = SparseValues.from_dense(self.doi_ij, opt['T_CUT_DO_IJ'])
            self.doi_iu = SparseValues.from_dense(self.doi_iu, min(opt[key] for key in (
                'T_CUT_DO_PRE', 'T_CUT_DO', 'T_CUT_DO_TRIPLES_PRE', 'T_CUT_DO_TRIPLES')))
            self.doi_uv = SparseValues.from_dense(self.doi_uv, opt['T_CUT_DO_UV'])
        # AO components on an atom mix under rigid rotations.  Keep or
        # discard complete atom blocks in the PAO/PAO integral screen.
        if self.pao_screen == 'atom':
            if self.screen_storage == 'sparse':
                from .screening import atom_complete_mask
                self.pao_pair_mask = atom_complete_mask(self.doi_uv, opt['T_CUT_DO_UV'], self.bf_atom, mol.natm)
            else:
                starts = np.array([rows[0] for rows in self.atom_bfs], dtype=int)
                atom_doi = np.maximum.reduceat(np.maximum.reduceat(self.doi_uv, starts, axis=0), starts, axis=1)
                atom_mask = atom_doi > opt['T_CUT_DO_UV']
                self.pao_pair_mask = atom_mask[self.bf_atom[:, None], self.bf_atom[None, :]]
        else:
            self.pao_pair_mask = self.doi_uv > opt['T_CUT_DO_UV']
        t = self._tick('doi', t)
        if self.dipole_tail_options is not None:
            from .dipole_tail import DipoleTail, orbital_inputs
            from .dipole_tree import MutualTree
            from .dipole_prescreen import select_dipole_pairs
            self.dipole_tail_driver = DipoleTail(**self.dipole_tail_options)
            with mol.with_common_orig((0., 0., 0.)):
                position = mol.intor('int1e_r', comp=3)
            inputs = orbital_inputs(self, dict(s=s, cl=orb.C_lmo, p=orb.C_pao, sp=orb.S_pao,
                                               fp=orb.F_pao, fl=orb.F_lmo), position)
            selected, self.dipole_prescreen_diagnostics = select_dipole_pairs(
                *inputs, threshold=opt['T_CUT_PRE'])
            # Both dense-oracle and local spatial DOI tables support
            # this union without enumerating distant dipole pairs.
            mask = self.doi_ij > opt['T_CUT_DO_IJ']
            if self.screen_storage == 'sparse':
                mask.add_diagonal()
            else:
                mask |= np.eye(self.n, dtype=bool)
            for i, j in selected:
                mask[i, j] = mask[j, i] = True
            self._initial_pair_mask = mask
            self.dip_e = self.dip_b = self._dipole_reference = None
            self._dipole_tail_exclusions = tuple((i, j) for i, j in su.mask_pairs(mask) if i < j)
            self._dipole_tail_tree = MutualTree(inputs[0], exclusions=self._dipole_tail_exclusions,
                                                **self.dipole_tail_driver.tree_options)
            result = self.dipole_tail_driver.evaluate(*inputs, exclusions=self._dipole_tail_exclusions,
                                                       tree=self._dipole_tail_tree, derivatives=False)
            self._distant_tail_energy, self.dipole_tail_diagnostics = result[0], result[-1]
        else:
            # Reusing a driver after changing the optional backend must
            # not leave a scalar from an earlier energy definition.
            self.__dict__.pop('_distant_tail_energy', None)
            self.__dict__.pop('_initial_pair_mask', None)
            self.__dict__.pop('dipole_tail_diagnostics', None)
            self.__dict__.pop('dipole_prescreen_diagnostics', None)
            self.dip_e, self.dip_b = dipole_pair_energies(
                orb, self.doi_iu, opt['T_CUT_DO_PRE'], opt['S_CUT'], self.bf_atom, self.atom_bfs)
            self._dipole_reference = self.dip_e
        t = self._tick('dipole', t)
        if self.local_screening_options is None:
            self.mkn = su.mulliken_atom_pops(mol, s, orb.C_lmo)
            if self.screen_storage == 'sparse':
                self.mkn = SparseValues.from_dense(self.mkn, min(opt[key] for key in (
                    'T_CUT_MKN', 'T_CUT_MKN_TRIPLES_PRE', 'T_CUT_MKN_TRIPLES')))
            self.screening_diagnostics = dict(storage=self.screen_storage, producer='dense_exact_grid',
                linear_producer=False, entries={name: getattr(getattr(self, name), 'nnz', getattr(self, name).shape[0]*getattr(self, name).shape[1])
                                                for name in ('doi_ij', 'doi_iu', 'doi_uv', 'mkn')})
        self.local_integrals = _ri_provider(self, **self.local_ri_options)
        self.local_integrals.attach(self, tracked=False)
        self._tick('df_provider', t)

    def nuc_grad_method(self, method='adjoint', **kwargs):
        if method != 'adjoint':
            raise ValueError('The screened local energy requires its matching adjoint gradient')
        return LocalGradients(self, **kwargs)

    Gradients = nuc_grad_method


class LocalGradients:
    """One amplitude/orbital adjoint sweep and local nuclear contraction.

    Tensor reverse sweeps may run on NumPy or CuPy. Local three-center
    integrals and global RHF potential/force actions have independent GPU
    backends. Boys and Krylov control remain on the CPU.
    """

    def __init__(self, base, level='auto', backend='numpy', energy_tolerance=2e-8,
                 lmo_cut=None, pao_cut=None, cache_bytes=None, progress=None, boys_pair_radius=None,
                 ri_backend=None, response_backend='cpu', checkpoint_triples=True,
                 checkpoint_integrals=True, triples_backend=None, krylov_memory_mib=8192,
                 ccsd_primal_backend='cpu', checkpoint_ccsd=False, triples_primal_backend='cpu',
                 checkpoint_pnos=False, response_checkpoint=None, ccsd_adjoint_backend=None):
        if level not in ('auto', 'mp2', 'ccsd'):
            raise ValueError("level must be 'auto', 'mp2', or 'ccsd'")
        if backend not in ('numpy', 'cupy'):
            raise ValueError("backend must be 'numpy' or 'cupy'")
        if ri_backend not in (None, 'cpu', 'gpu') or response_backend not in ('cpu', 'gpu'):
            raise ValueError("RI and response backends must be 'cpu' or 'gpu'")
        self.ri_backend, self.response_backend = ri_backend, response_backend
        self.base, self.level, self.backend = base, level, backend
        self.energy_tolerance = energy_tolerance
        self.ri_options = dict(lmo_cut=lmo_cut, pao_cut=pao_cut, cache_bytes=cache_bytes)
        self.de, self.components, self.diagnostics, self.timings = None, {}, {}, {}
        self.progress = progress
        self.boys_pair_radius = boys_pair_radius
        self.checkpoint_triples = checkpoint_triples
        self.checkpoint_integrals = checkpoint_integrals
        self.triples_backend = triples_backend
        self.krylov_memory_mib = krylov_memory_mib
        self.ccsd_primal_backend = ccsd_primal_backend
        if ccsd_adjoint_backend not in (None, 'cupy'):
            raise ValueError("ccsd_adjoint_backend must be None or 'cupy'")
        self.ccsd_adjoint_backend = ccsd_adjoint_backend
        self.checkpoint_ccsd = checkpoint_ccsd
        self.triples_primal_backend = triples_primal_backend
        self.checkpoint_pnos = checkpoint_pnos
        self.response_checkpoint = response_checkpoint
        self._response_restart = None

    def _record(self, name, start):
        self.timings[name] = time.perf_counter()-start
        if self.progress is not None:
            self.progress(dict(event='stage_done', stage=name, seconds=self.timings[name]))

    @classmethod
    def from_response_checkpoint(cls, path, *, response_backend=None, progress=None,
                                 expected_sha256=None):
        """Retry final contractions from a source-verified post-adjoint NPZ.

        The backend can change, for example from GPU to CPU. All physical
        data and the completed correlation/orbital adjoints remain fixed.
        This object has a reference-only base and does not rerun an energy.
        """
        from .response_checkpoint import load_response_checkpoint
        base, arrays, metadata, provenance = load_response_checkpoint(
            path, expected_sha256=expected_sha256)
        backend = metadata['producer_response_backend'] if response_backend is None else response_backend
        result = cls(base, response_backend=backend, progress=progress)
        result.relaxed_densities = {name: arrays[name] for name in ('s', 'f', 'r')}
        result.diagnostics = metadata['diagnostics']
        result.diagnostics['response_restart'] = dict(provenance,
            completed_timings=metadata['timings'], producer_response_backend=metadata['producer_response_backend'])
        previous = result.diagnostics.pop('gpu_reference', None)
        if previous is not None:
            result.diagnostics['response_restart']['producer_gpu_reference'] = previous
        result._response_restart = arrays['local_force']
        return result

    def _finish_response(self, base, atoms, local_force, reference):
        t = time.perf_counter()
        orbital_force = contract_orbital_densities(base, self.relaxed_densities,
            jk_force=None if reference is None else reference.quadratic_force)
        self._record('orbital_contraction', t)
        t = time.perf_counter()
        mf = validate_reference(base)
        hf_force = mf.nuc_grad_method().kernel() if reference is None else reference.hf_gradient()
        self._record('hf_gradient', t)
        self.components = dict(hf=hf_force[atoms], local_ri=local_force[atoms], orbital=orbital_force[atoms])
        self.de = sum(self.components.values())
        if reference is not None:
            self.diagnostics['gpu_reference'] = reference.info
        if not np.isfinite(self.de).all():
            raise FloatingPointError('Nonfinite local DLPNO gradient')
        return self.de

    def kernel(self, atmlst=None):
        base = self.base
        if self._response_restart is not None:
            atoms = list(range(base.mol.natm)) if atmlst is None else list(atmlst)
            if len(set(atoms)) != len(atoms) or any(a < 0 or a >= base.mol.natm for a in atoms):
                raise ValueError('atmlst must contain distinct valid atom indices')
            reference = None
            if self.response_backend == 'gpu':
                from .gpu_response import GPUReferenceBackend
                reference = GPUReferenceBackend(base)
            return self._finish_response(base, atoms, self._response_restart, reference)
        if self.response_checkpoint is not None:
            from pathlib import Path
            destination = Path(self.response_checkpoint)
            if destination.exists() or Path(str(destination)+'.json').exists():
                raise FileExistsError(f'Preserve the existing response checkpoint: {destination}')
        if hasattr(base, 'build_solver'):
            if base.solver is None:
                base.kernel()
            base = base.solver
        if not base.e:
            base.kernel()
        mf = validate_reference(base)
        atoms = list(range(base.mol.natm)) if atmlst is None else list(atmlst)
        if len(set(atoms)) != len(atoms) or any(a < 0 or a >= base.mol.natm for a in atoms):
            raise ValueError('atmlst must contain distinct valid atom indices')
        options = dict(getattr(base, 'local_ri_options', dict(lmo_cut=0., pao_cut=0., cache_bytes=64 << 20)))
        options.update({key: value for key, value in self.ri_options.items() if value is not None})
        t = time.perf_counter()
        provider = _ri_provider(base, self.ri_backend, **options)
        provider.validate_energy_support()
        # This provider inherited the energy's support/envelope plans,
        # but builds its own tracked primitives. Keeping the old energy
        # integral cache would overlap two independent cache budgets.
        parent_provider = getattr(base, 'local_integrals', None)
        released_cache = 0 if parent_provider is None else parent_provider.clear_cache()
        clear_device_cache = getattr(getattr(base, 'dfi', None), 'cache_clear', None)
        if clear_device_cache is not None:
            clear_device_cache()
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
        gc.collect()
        self._record('release_energy_cache', t)
        t = time.perf_counter()
        state = BlockAdjointDLPNO(base, inputs=provider, level=self.level, backend=self.backend,
                                  progress=self.progress, checkpoint_triples=self.checkpoint_triples,
                                  checkpoint_integrals=self.checkpoint_integrals,
                                  triples_backend=self.triples_backend,
                                  krylov_memory_mib=self.krylov_memory_mib,
                                  ccsd_primal_backend=self.ccsd_primal_backend,
                                  ccsd_adjoint_backend=self.ccsd_adjoint_backend,
                                  checkpoint_ccsd=self.checkpoint_ccsd,
                                  triples_primal_backend=self.triples_primal_backend,
                                  checkpoint_pnos=self.checkpoint_pnos)
        energy = state.kernel()
        target = 'lmp2' if self.level == 'mp2' else ('ccsd' if self.level == 'ccsd' else 'corr')
        error = float(energy.v)-base.e[target]
        if abs(error) > self.energy_tolerance:
            raise RuntimeError(f'Local adjoint energy differs from {target} by {error:.3e} Eh; '
                               'use the same coefficient supports for energy and gradient')
        self._record('energy_graph', t)
        t = time.perf_counter()
        primitive_statistics = provider.statistics()['blocks']
        bars = state.densities(release_graph=True, release_inputs=True)
        amplitude_info = state.diagnostics
        self._record('amplitude_adjoint', t)
        t = time.perf_counter()
        # The implicit solve closures form cycles through the energy
        # state. Release them before allocating RI and global response
        # workspaces, rather than retaining two phases' peak storage.
        provider.release_leaves(statistics=primitive_statistics)
        del state, energy
        gc.collect()
        ri_backend = self.ri_backend or getattr(base, 'ri_backend', 'cpu')
        if self.backend == 'cupy' or 'gpu' in (self.response_backend, ri_backend):
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
        self._record('release_amplitude_graph', t)
        t = time.perf_counter()
        sources, local_force = provider.pullback(bars)
        del bars
        self.orbital_sources = sources
        self._record('ri_and_orbital_sources', t)
        t = time.perf_counter()
        reference = None
        if self.response_backend == 'gpu':
            from .gpu_response import GPUReferenceBackend
            reference = GPUReferenceBackend(base)
        relaxed, orbital_info = orbital_densities(base, sources, boys_pair_radius=self.boys_pair_radius,
                                                  fock=None if reference is None else reference.potential)
        self.relaxed_densities = relaxed
        self._record('orbital_adjoint', t)
        self.diagnostics = dict(energy_error=error, amplitudes=amplitude_info,
                                orbitals=orbital_info, integrals=provider.statistics(),
                                released_energy_cache_bytes=released_cache)
        if reference is not None:
            self.diagnostics['gpu_reference'] = reference.info
        if hasattr(base, 'screening_diagnostics'):
            self.diagnostics['screening'] = dict(base.screening_diagnostics)
        if hasattr(base, 'dipole_tail_diagnostics'):
            self.diagnostics['dipole_tail'] = dict(base.dipole_tail_diagnostics)
            self.diagnostics['dipole_prescreen'] = dict(base.dipole_prescreen_diagnostics)
        if hasattr(base, 'dipole_preparation_diagnostics'):
            self.diagnostics['dipole_preparation'] = dict(base.dipole_preparation_diagnostics)
        if self.response_checkpoint is not None:
            from .response_checkpoint import save_response_checkpoint
            t = time.perf_counter()
            self.diagnostics['response_checkpoint'] = save_response_checkpoint(
                self.response_checkpoint, self, base, local_force)
            self._record('response_checkpoint', t)
        return self._finish_response(base, atoms, local_force, reference)

    run = kernel
