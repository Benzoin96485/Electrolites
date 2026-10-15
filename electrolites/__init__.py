"""Electrolites -- drop-in GPU kernel replacements for GPU4PySCF.

Importing one of the modules below monkey-patches the corresponding GPU4PySCF
entry point in place.  Whatever a module does not cover -- an angular-momentum
class, an open-shell density, a functional it declines -- falls through to
GPU4PySCF's own code, so a patched run computes the same quantity as an
unpatched one.

    import electrolites
    electrolites.patch_all()                     # every applicable module

    from pyscf import gto
    from gpu4pyscf import dft
    mol = gto.M(atom='...', basis='def2-tzvp')
    mf = dft.RKS(mol, xc='wb97m-v')
    e = mf.kernel()
    de = mf.nuc_grad_method().kernel()

``patch_all(only=[...])`` or ``patch('fastk', 'fastxc')`` patch a subset, which
is how the ablations in ``docs/`` were measured.  There is no unpatch: the
patches are global and the benchmarks compare separate processes, which is the
only honest way to A/B them.

Which module replaces what, and what it declines, is in README.md.
"""
import importlib
import inspect

__version__ = '0.1.0'

#: GPU4PySCF releases this has been run against end to end.
TESTED_GPU4PYSCF = ('1.7.0', '1.8.1')

#: The SCF modules, then the gradient and Hessian ones, then the response
#: ones, in the order they should load.
MODULES = (
    'fastk', 'fastj', 'fastxc', 'fastrsh', 'fastxcnlc', 'fastnlc', 'fast1e',
    'fastejk', 'fastgrad', 'fastxcgrad', 'fastgradh', 'fastxchess',
    'fastxcfxc', 'fastfnlc', 'fastvv10ft', 'fastvnlch', 'fastjk', 'fastejk2', 'fastdf', 'fastdfj',
    'fastdfj3c', 'fastdfgrad', 'fastdfcderi', 'fastdffp32', 'fastdfhess',
    'fastdfhessip', 'fastdfrsh', 'planguard', 'fastcphf', 'fastir', 'fastnacms', 'fastnmr', 'fasttdejk', 'fastdfmp2', 'fastc6', 'fastqmmm', 'fastpbcgrad', 'fastmcfun', 'fastfftk', 'fastgridresp', 'fastjkip1', 'fastsfxc', 'fastpcmbatch', 'fastscfscreen', 'fastmp2', 'fastccsd', 'fastrshresp', 'fastpbcdf',
    # not kernels: correctness guards and small upstream repairs
    # (docs/FEATURE_TARGETS_PYSCF_2.13.md 3.2, 3.3).  Last, so that they wrap
    # whatever the modules above put in place
    'guards', 'fixes',
)

# What each module replaces: one ``(module path, dotted attribute)`` pair, or a
# tuple of them where a module patches more than one entry point -- checked
# before anything patches, so a GPU4PySCF whose API moved fails loudly here
# instead of silently mis-patching or computing the wrong operator.
#
# A module that covers both spin cases has to name *both*, because
# ``gpu4pyscf.dft.uks`` and ``gpu4pyscf.grad.uks`` bind their own copies of
# these functions at import: patching the restricted one does not reach the
# unrestricted path, and the only symptom is that an open-shell run is as fast
# as an unpatched one.  ``benchmarks/coverage_probe.py`` counts which entry
# points a given calculation actually reaches.
_TARGETS = {
    'fastk': ('gpu4pyscf.scf.jk', '_VHFOpt.get_k'),
    'fastj': ('gpu4pyscf.scf.j_engine', '_VHFOpt.get_j'),
    'fastxc': (('gpu4pyscf.dft.numint', 'NumInt.nr_rks'),
               ('gpu4pyscf.dft.numint', 'NumInt.nr_uks')),
    'fastrsh': ('gpu4pyscf.dft.rks', 'RKS.get_veff'),
    'fastxcnlc': (('gpu4pyscf.dft.rks', 'RKS.get_veff'),
                  ('gpu4pyscf.dft.uks', 'UKS.get_veff'),
                  # `density_fit()` puts this one ahead of both of the above
                  # in the method resolution order, so a DF run reaches
                  # neither; see docs/README_fastdf.md 5
                  ('gpu4pyscf.df.df_jk', '_DFHF.get_veff')),
    'fastnlc': ('gpu4pyscf.dft.numint', '_vv10nlc'),
    'fast1e': ('gpu4pyscf.gto.int3c1e', 'get_int3c1e_density_contracted'),
    'fastejk': ('gpu4pyscf.grad.rhf', '_jk_energy_per_atom'),
    'fastgrad': ('gpu4pyscf.grad.rks', 'Gradients.energy_ee'),
    'fastxcgrad': (('gpu4pyscf.grad.rks', 'get_exc'),
                   ('gpu4pyscf.grad.rks', 'get_nlc_exc'),
                   ('gpu4pyscf.grad.uks', 'get_exc'),
                   ('gpu4pyscf.grad.uks', 'get_nlc_exc')),
    'fastgradh': ('gpu4pyscf.grad.rhf', 'GradientsBase.get_hcore'),
    'fastxchess': (('gpu4pyscf.hessian.rks', '_get_vxc_deriv1'),
                   ('gpu4pyscf.hessian.rks', '_get_vxc_deriv2'),
                   ('gpu4pyscf.hessian.uks', '_get_vxc_deriv1'),
                   ('gpu4pyscf.hessian.uks', '_get_vxc_deriv2')),
    'fastxcfxc': (('gpu4pyscf.dft.numint', 'nr_rks_fxc'),
                  ('gpu4pyscf.dft.numint', 'nr_uks_fxc'),
                  # a restricted Hessian's CPHF grid half, which is not
                  # `nr_rks_fxc` (docs/DF_WORK_LIST.md W16).  The
                  # unrestricted Hessian has no such function and comes
                  # through `nr_uks_fxc`'s `mo1` branch instead
                  ('gpu4pyscf.hessian.rks', 'nr_rks_fxc_mo')),
    'fastfnlc': (('gpu4pyscf.hessian.rks', 'nr_rks_fnlc_mo'),
                 ('gpu4pyscf.hessian.uks', 'nr_rks_fnlc_mo')),
    # CUDA symbols, not Python entry points: every caller of the VV10
    # response pair sum (`hessian/rks.py`, three functions, and `fastfnlc`)
    # and of its grid-weight-derivative twin looks them up on this ctypes
    # library at call time
    'fastvv10ft': (('gpu4pyscf.dft.numint',
                    'libgdft.VXC_vv10nlc_hess_eval_f_t'),
                   ('gpu4pyscf.dft.numint',
                    'libgdft.VXC_vv10nlc_hess_eval_EUW_with_weight1')),
    # the VV10 term of every Hessian's `make_h1`; `hessian/uks.py` and the
    # density-fitted Hessians bind the name at import (B14)
    'fastvnlch': (('gpu4pyscf.hessian.rks', '_get_vnlc_deriv1'),
                  ('gpu4pyscf.hessian.uks', '_get_vnlc_deriv1'),
                  ('gpu4pyscf.df.hessian.rks', '_get_vnlc_deriv1'),
                  ('gpu4pyscf.df.hessian.uks', '_get_vnlc_deriv1')),
    'fastjk': ('gpu4pyscf.scf.jk', '_VHFOpt.get_jk'),
    # this one replaces two CUDA symbols rather than a Python entry point, so
    # the target check only confirms the driver that looks them up still exists
    'fastejk2': ('gpu4pyscf.hessian.rhf', '_partial_ejk_ip2'),
    # the density-fitting path.  `df/df.py:117` is the only caller of this
    # function, and it looks it up on the module, so one target covers every
    # DF entry point: the SCF, the response solve and a Hessian's CPHF
    'fastdf': ('gpu4pyscf.df.df_jk', 'get_jk'),
    # the `only_dfj` (RIJONX) branch of the driver above the one `fastdf`
    # replaces.  It is a repair rather than a kernel: the branch passes `mol`
    # where `DF.get_jk` expects the density and so cannot run at all on 1.8.1
    'fastdfj': (('gpu4pyscf.df.df_jk', '_DFHF.get_jk'),
                ('gpu4pyscf.df.df_jk', '_DFHF.get_k'),
                ('gpu4pyscf.df.df_jk', '_DFHF.get_veff'),
                ('gpu4pyscf.df.df_jk', '_DFHF.Gradients')),
    # the three-centre Coulomb build a *pure* density-fitted functional runs
    # on, and the Coulomb half of every RIJONX run.  The two contractions are
    # the target; the four `SortedGTO` recontraction kernels are named because
    # this module caches their device tables, and they are reached by the
    # cderi build and by `get_hcore` as well as by `get_j`
    'fastdfj3c': (('gpu4pyscf.df.j_engine_3c2e', 'Int3c2eOpt.contract_dm'),
                  ('gpu4pyscf.df.j_engine_3c2e', 'Int3c2eOpt.contract_auxvec'),
                  ('gpu4pyscf.gto.mole', 'SortedGTO.CT_dot_mat'),
                  ('gpu4pyscf.gto.mole', 'SortedGTO.C_dot_mat'),
                  ('gpu4pyscf.gto.mole', 'SortedGTO.mat_dot_C'),
                  ('gpu4pyscf.gto.mole', 'SortedGTO.mat_dot_CT')),
    # the two-electron derivative of a density-fitted gradient.  Four classes,
    # because each binds `jk_energy_per_atom` as a class attribute at
    # class-creation time and so carries its own copy of the function object;
    # `_decompose_j2c` is the identity the cderi reuse rests on
    'fastdfgrad': (('gpu4pyscf.df.grad.rhf', 'Gradients.jk_energy_per_atom'),
                   ('gpu4pyscf.df.grad.rks', 'Gradients.jk_energy_per_atom'),
                   ('gpu4pyscf.df.grad.uhf', 'Gradients.jk_energy_per_atom'),
                   ('gpu4pyscf.df.grad.uks', 'Gradients.jk_energy_per_atom'),
                   ('gpu4pyscf.df.grad.rhf', '_jk_energy_per_atom'),
                   ('gpu4pyscf.df.grad.uhf', '_jk_energy_per_atom'),
                   ('gpu4pyscf.df.df', '_decompose_j2c')),
    # the cderi build itself.  `df/df.py:105` is the only caller and looks it
    # up on the module, so one target covers every density-fitted run.  The
    # other three are what the module reaches into rather than replaces: the
    # sparse apply of the contraction coefficients, the aggregation call the
    # launch configuration is changed through, and the scheme it is checked
    # against
    'fastdfcderi': (('gpu4pyscf.df.df', '_cholesky_eri'),
                    # W13: the two-centre metric integrals, whose per-call
                    # optimiser is 70-80 % of `_decompose_j2c` at the small
                    # end.  `df/df.py:602` looks it up on the module
                    ('gpu4pyscf.df.int3c2e_bdiv', 'int2c2e'),
                    ('gpu4pyscf.gto.mole', 'SortedMole.mat_dot_C'),
                    ('gpu4pyscf.gto.mole', 'SortedMole.aggregate_shl_pairs'),
                    ('gpu4pyscf.df.int3c2e_bdiv', 'int3c2e_scheme')),
    # the two consumers of a **single-precision** cderi that are not
    # `df_jk.get_jk`: the block iterator everything else reaches the tensor
    # through, and the one line of the DF Hessian's CPHF build that moves the
    # tensor to the host with a double-precision transpose.  Both are
    # upstream's own function unchanged while the tensor is double, so this
    # module is inert unless `FASTDF_FP32` is set (`docs/DF_WORK_LIST.md` W7)
    'fastdffp32': (('gpu4pyscf.df.df', 'DF.loop'),
                   ('gpu4pyscf.df.hessian.rhf', '_get_jk')),
    # the DF Hessian's own MO-basis CPHF build -- the fifth two-electron
    # path, which `df/df_jk.py:get_jk` is not on, so `fastdf` is not either.
    # `df/hessian/{rks,uhf,uks}.py` all bind `_get_jk_mo` as a class
    # attribute and that function looks `_get_jk` up on this module, so one
    # target covers the restricted and unrestricted Hessians both.
    # `_balance_batch_size` is upstream's own helper, which this module uses
    # rather than replaces
    'fastdfhess': (('gpu4pyscf.df.hessian.rhf', '_get_jk'),
                   ('gpu4pyscf.df.hessian.rhf', '_get_jk_mo'),
                   ('gpu4pyscf.df.hessian.rhf', '_balance_batch_size')),
    # the density-fitted Hessian's other two two-electron terms, the sixth
    # and seventh code paths (docs/DF_WORK_LIST.md W17 and W18).
    # `df/hessian/{rhf,rks}.py` look both up on this module at call time, so
    # the module globals are the targets; the rest are what the copies
    # import rather than replace
    # `df/hessian/{uhf,uks}.py` look the unrestricted twins up on
    # `df/hessian/uhf.py` the same way (W25)
    'fastdfhessip': (('gpu4pyscf.df.hessian.rhf', '_jk_energy_per_atom'),
                     ('gpu4pyscf.df.hessian.rhf', '_get_veff'),
                     ('gpu4pyscf.df.hessian.uhf', '_jk_energy_per_atom'),
                     ('gpu4pyscf.df.hessian.uhf', '_get_veff'),
                     ('gpu4pyscf.df.hessian.rhf', '_int2c2e_ip2_per_atom'),
                     ('gpu4pyscf.df.hessian.rhf', 'int3c2e_scheme_ip2'),
                     ('gpu4pyscf.df.int3c2e_bdiv', '_int3c2e_ip1_evaluator'),
                     ('gpu4pyscf.df.int3c2e_bdiv', 'libvhf_rys')),
    # a range-separated DF SCF's exchange from one tensor instead of two
    # (docs/DF_WORK_LIST.md W22).  It wraps whatever `_DFHF.get_veff` is --
    # `fastxcnlc`'s under `patch_all()` -- and reaches `DF._rsh_df`, where it
    # keeps the fused tensor so that `DF.reset` drops it
    'fastdfrsh': (('gpu4pyscf.df.df_jk', '_DFHF.get_veff'),
                  ('gpu4pyscf.df.df', 'DF.range_coulomb')),
    # not a kernel: the one function every `contract` in GPU4PySCF and here
    # goes through, looked up at call time, wrapped so that a contraction
    # cuTENSOR refuses is cut into pieces it plans (docs/DF_WORK_LIST.md W27)
    'planguard': ('gpu4pyscf.lib.cutensor', 'contraction'),
    # the IR driver: the Hessian's CPHF solution reused for the dipole
    # derivative (catalogue C10); callers reach it as `ir.eval_ir_...`
    'fastir': ('gpu4pyscf.properties.ir', 'eval_ir_freq_intensity'),
    # force + all NACs through the batched multi-state solver (D12); the
    # TDA/TDDFT classes of both references inherit the method
    'fastnacms': (('gpu4pyscf.tdscf.rhf', 'TDBase.force_and_nacv'),
                  ('gpu4pyscf.tdscf.rhf', 'TDA.NACGradients')),
    # the NMR shielding's XC term with the GIAO AOs on the GPU (C7);
    # `get_vxc` calls the module-level name
    'fastnmr': ('gpu4pyscf.properties.shielding', 'nr_rks'),
    # the TDDFT gradient / NAC two-electron derivative for density pairs
    # (D10); `Gradients.jk_energies_per_atom` looks the name up at call time
    'fasttdejk': ('gpu4pyscf.grad.tdrhf', '_jk_energies_per_atom'),
    # the DF-MP2 / DF-UMP2 pair-energy loops (I1); the drivers look them up
    # on `dfmp2_addons` at call time
    'fastdfmp2': (('gpu4pyscf.mp.dfmp2_addons', 'get_dfmp2_energy_pair_intra'),
                  ('gpu4pyscf.mp.dfmp2_addons', 'get_dfump2_energy_pair_intra')),
    # C6 coefficients: the full-spectrum solve on the device (C5)
    'fastc6': (('gpu4pyscf.properties.c6', '_solve_full_spectrum'),
               ('gpu4pyscf.properties.c6', 'calc_c6'),
               ('gpu4pyscf.tdscf.rhf', 'get_ab')),
    # QM/MM's QM-nucleus--MM-charge energy and forces on the device, cached
    # (F6 / F13)
    # the periodic gradients' local 1e derivative from one density pass (J7)
    'fastpbcgrad': (('gpu4pyscf.pbc.grad.krhf', 'grad_elec'),
                    ('gpu4pyscf.pbc.grad.kuhf', 'grad_elec'),
                    ('gpu4pyscf.pbc.grad.rhf', 'Gradients.grad_elec'),
                    ('gpu4pyscf.pbc.grad.uhf', 'Gradients.grad_elec')),
    # the multi-collinear XC on one sample per antipodal pair (A21);
    # `_eval_xc_lebedev` looks the name up at call time
    'fastmcfun': ('gpu4pyscf.dft.mcfun_gpu', '_make_sph_samples'),
    # FFTDF k-point exchange with time reversal and wide FFT batches (J3);
    # `FFTDF.get_jk` looks the name up on `fft_jk` at call time
    'fastfftk': ('gpu4pyscf.pbc.df.fft_jk', 'get_k_kpts'),
    # the grid-response Hessian's weight second derivative contracted
    # without being stored (B9); the drivers look the names up at call time
    'fastgridresp': (('gpu4pyscf.hessian.rks', '_get_exc_deriv2_grid_response'),
                     ('gpu4pyscf.hessian.rks', '_get_enlc_deriv2'),
                     ('gpu4pyscf.hessian.uks', '_get_exc_deriv2_grid_response')),
    # the direct Hessian's h1 two-electron kernels (B15): bound over
    # `libvhf_rys.RYS_build_jk_ip1`, which `_get_jk_ip1` reads at call time
    'fastjkip1': ('gpu4pyscf.hessian.rhf', 'libvhf_rys.RYS_build_jk_ip1'),
    # the spin-flip multi-collinear XC kernel without the full tensors (D13)
    'fastsfxc': (('gpu4pyscf.tdscf._uhf_resp_sf', 'mcfun_eval_xc_adapter_sf'),
                 ('gpu4pyscf.grad.tduks_sf', 'mcfun_eval_xc_adapter_sf')),
    # int1e_grids with many densities / charge vectors in one pass (F9)
    'fastpcmbatch': (('gpu4pyscf.gto.int3c1e', 'int1e_grids'),
                     ('gpu4pyscf.solvent.pcm', 'int1e_grids')),
    # direct SCF: loose screening while the density is far off, then a
    # tight rebuild (A20)
    'fastscfscreen': (('gpu4pyscf.dft.rks', 'RKS.get_veff'),
                      ('gpu4pyscf.dft.uks', 'UKS.get_veff'),
                      ('gpu4pyscf.scf.hf', 'RHF.get_veff'),
                      ('gpu4pyscf.scf.uhf', 'UHF.get_veff')),
    # conventional MP2's (ia|jb): blocked half transformation (I3)
    'fastmp2': ('gpu4pyscf.scf.int4c2e', 'get_int4c2e_ovov'),
    # GDF k-point cderi on the device when it fits (J1)
    'fastpbcdf': (('gpu4pyscf.pbc.df.rsdf_builder', 'compressed_cderi_kk'),
                  ('gpu4pyscf.pbc.df.df', 'GDF.loop')),
    # range-separated hybrids' response J + K: both exchange ranges in one
    # pass (C12)
    'fastrshresp': (('gpu4pyscf.scf._response_functions', '_gen_rhf_response'),
                    ('gpu4pyscf.scf._response_functions', '_gen_uhf_response')),
    # closed-shell CCSD with amplitudes, MO integrals and DIIS on the
    # device (I4, and the host half of I5)
    'fastccsd': ('gpu4pyscf.cc.ccsd_incore', 'CCSDBase.ccsd'),
    'fastqmmm': (('gpu4pyscf.qmmm.itrf', 'QMMMSCF.energy_nuc'),
                 ('gpu4pyscf.qmmm.itrf', 'QMMMGrad.grad_nuc'),
                 ('gpu4pyscf.qmmm.itrf', 'QMMMGrad.grad_nuc_mm')),
    # the Hessian's CPHF solve (docs/DF_WORK_LIST.md W44): both spin cases'
    # `solve_mo1`, which the Hessian classes' methods look up on their
    # modules at call time, and the Krylov solver and QR it reuses
    'fastcphf': (('gpu4pyscf.hessian.rhf', 'solve_mo1'),
                 ('gpu4pyscf.hessian.uhf', 'solve_mo1'),
                 ('gpu4pyscf.lib.cupy_helper', 'krylov'),
                 ('gpu4pyscf.lib.cupy_helper', '_stable_qr')),
    # the ROHF/ROKS entry points that return a wrong Hessian or crash, and
    # the QM/MM Hessian that silently drops the MM terms (FEATURE_TARGETS
    # 3.2); the density-fitted dispatch is wrapped, not replaced
    'guards': (('gpu4pyscf.scf.rohf', 'ROHF.Hessian'),
               ('gpu4pyscf.dft.roks', 'ROKS'),
               ('gpu4pyscf.df.df_jk', '_DFHF.Gradients'),
               ('gpu4pyscf.df.df_jk', '_DFHF.Hessian'),
               ('gpu4pyscf.qmmm.itrf', 'QMMMSCF')),
    # FEATURE_TARGETS 3.3 D1, D2, D3, D4, D7
    'fixes': (('gpu4pyscf.dft.rks', 'RKS.init_guess_by_vsap'),
              ('gpu4pyscf.dft.uks', 'UKS.get_vasp'),
              ('gpu4pyscf.scf.rohf', 'ROHF.get_init_guess'),
              ('gpu4pyscf.cc.ccsd_incore', 'CCSDBase.to_cpu'),
              ('gpu4pyscf.mp.dfmp2', 'DFMP2._kernel_impl'),
              ('gpu4pyscf.mp.dfump2', 'DFUMP2._kernel_impl')),
}

_patched = []


def gpu4pyscf_version():
    import gpu4pyscf
    return gpu4pyscf.__version__


def check_targets(names=None):
    """Report which patch targets exist in the installed GPU4PySCF.

    Returns ``{module: None}`` when the target is present, or
    ``{module: reason}`` when it is missing.
    """
    out = {}
    for name in (names or MODULES):
        pairs = _TARGETS[name]
        if isinstance(pairs[0], str):                   # a single pair
            pairs = (pairs,)
        out[name] = None
        for mod_path, attr_path in pairs:
            try:
                obj = importlib.import_module(mod_path)
                for part in attr_path.split('.'):
                    obj = getattr(obj, part)
            except (ImportError, AttributeError) as exc:
                out[name] = f'{mod_path}.{attr_path} is missing ({exc})'
                break
    return out


def patch(*names, strict=True):
    """Patch the named modules.  ``strict`` raises if a target is missing."""
    names = [n for group in names for n in
             ((group,) if isinstance(group, str) else group)]
    unknown = [n for n in names if n not in _TARGETS]
    if unknown:
        raise ValueError(f'unknown module(s) {unknown}; known: {list(MODULES)}')
    missing = {k: v for k, v in check_targets(names).items() if v}
    if missing:
        msg = (f'GPU4PySCF {gpu4pyscf_version()} does not expose the entry '
               f'point(s) these modules replace: {missing}. '
               f'Tested against {", ".join(TESTED_GPU4PYSCF)}.')
        if strict:
            raise RuntimeError(msg)
        names = [n for n in names if n not in missing]
    for name in names:
        importlib.import_module(f'.{name}', __name__)
        if name not in _patched:
            _patched.append(name)
    return tuple(_patched)


def patch_all(only=None, strict=True):
    """Patch every module (or ``only`` a subset), in dependency order."""
    names = list(only) if only else list(MODULES)
    return patch([n for n in MODULES if n in names], strict=strict)


def patched():
    """The modules patched in this process, in the order they were applied."""
    return tuple(_patched)


# Features this repository *adds* to GPU4PySCF (stability analysis, CCSD(T),
# CASSCF, ...): the inverse of the patch modules above, see `features.py`
from .features import (enable, enable_all, enabled,         # noqa: E402
                       check_absent, FEATURES)

__all__ = ['patch', 'patch_all', 'patched', 'check_targets',
           'gpu4pyscf_version', 'MODULES', 'TESTED_GPU4PYSCF', '__version__',
           'enable', 'enable_all', 'enabled', 'check_absent', 'FEATURES']
