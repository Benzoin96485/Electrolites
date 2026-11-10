"""Portable post-adjoint state for retrying final RHF/response contractions.

The NPZ contains numerical arrays and JSON, never executable pickle data.
An exclusive, fsynced data file followed by a SHA256 manifest commits a
checkpoint. Incomplete pairs and source mismatches are rejected on load.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

_FORMAT = 'electrolites-dlpno-response-v1'
_STAGE = 'orbital_adjoint_complete'
_ARRAYS = ('mo_coeff', 'mo_occ', 'mo_energy', 's', 'f', 'r', 'local_force')


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_sha256():
    return {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Nonportable checkpoint metadata: {type(value).__name__}')


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, default=_json_default)


def _array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(_json([str(value.dtype), value.shape]).encode())
    digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def _publish(path, write):
    """Publish a complete file without overwriting an earlier calculation."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.',
                                     suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)  # Atomic and fails if path already exists.
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_arrays(mol, arrays):
    n = mol.nao
    if arrays['mo_coeff'].ndim != 2 or arrays['mo_coeff'].shape[0] != n:
        raise ValueError('Response checkpoint has incompatible MO dimensions')
    m = arrays['mo_coeff'].shape[1]
    shapes = dict(mo_coeff=(n, m), mo_occ=(m,), mo_energy=(m,), s=(n, n),
                  f=(n, n), r=(3, n, n), local_force=(mol.natm, 3))
    for name, shape in shapes.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError(f'Invalid response checkpoint array: {name}')
    occ = arrays['mo_occ']
    if not np.all((occ == 0) | (occ == 2)) or int(occ.sum()) != mol.nelectron:
        raise ValueError('Response checkpoint requires closed-shell RHF occupations')


def save_response_checkpoint(path, gradient, base, local_force):
    """Save all data needed after the orbital adjoint, before force allocation."""
    from .orbital_response import validate_reference
    path = Path(path)
    manifest_path = Path(str(path)+'.json')
    if path.exists() or manifest_path.exists():
        raise FileExistsError(f'Preserve the existing response checkpoint: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    mf = validate_reference(base)
    # Runtime SCF callbacks are not reproducible from a molecule and MOs.
    overrides = {'get_hcore', 'get_ovlp', 'get_veff', 'get_jk', 'nuc_grad_method'} & vars(mf).keys()
    if overrides or not type(mf).__module__.startswith('pyscf.'):
        raise NotImplementedError('A response checkpoint requires a standard PySCF RHF reference')
    arrays = {name: np.asarray(getattr(mf, name)) for name in _ARRAYS[:3]}
    arrays.update({name: np.asarray(gradient.relaxed_densities[name]) for name in ('s', 'f', 'r')})
    arrays['local_force'] = np.asarray(local_force)
    _validate_arrays(mf.mol, arrays)
    fitting = getattr(mf, 'with_df', None)
    auxiliary = getattr(fitting, 'auxmol', None)
    reference = dict(molecule=mf.mol.dumps(), converged=bool(mf.converged),
                     energy=mf.e_tot, direct_scf_tol=mf.direct_scf_tol,
                     conv_tol=mf.conv_tol, conv_tol_grad=mf.conv_tol_grad,
                     density_fitted=fitting is not None,
                     auxbasis=None if fitting is None else fitting.auxbasis,
                     auxmol=None if auxiliary is None else auxiliary.dumps(),
                     only_dfj=bool(getattr(mf, 'only_dfj', False)))
    physical = {name: getattr(base, name, None) for name in (
        'opt', 'triples', 'localizer', 'local_ri_options', 'pao_screen',
        'dipole_tail_options', 'local_screening_options', 'screen_storage')}
    physical.update(level=gradient.level, boys_pair_radius=gradient.boys_pair_radius,
                    gradient_ri_options=gradient.ri_options, energy=dict(base.e),
                    local_orbitals_sha256=_array_sha256(base.orb.C_lmo), ncore=int(base.orb.ncore))
    binding = dict(reference=reference, physical_settings=physical,
                   orbital_arrays_sha256={name: _array_sha256(arrays[name]) for name in _ARRAYS[:3]})
    metadata = dict(format=_FORMAT, completed_stage=_STAGE, gradient_units='Eh/bohr',
                    source_sha256=source_sha256(), binding=binding,
                    binding_sha256=hashlib.sha256(_json(binding).encode()).hexdigest(),
                    diagnostics=gradient.diagnostics, timings=gradient.timings,
                    producer_response_backend=gradient.response_backend)
    payload = dict(arrays, metadata=np.asarray(_json(metadata)))
    _publish(path, lambda stream: np.savez(stream, **payload))
    manifest = dict(format=_FORMAT, completed_stage=_STAGE, sha256=file_sha256(path),
                    bytes=path.stat().st_size, binding_sha256=metadata['binding_sha256'],
                    source_sha256=metadata['source_sha256'])
    _publish(manifest_path, lambda stream: stream.write((_json(manifest)+'\n').encode()))
    return dict(path=str(path), sha256=manifest['sha256'], bytes=manifest['bytes'],
                binding_sha256=manifest['binding_sha256'], completed_stage=_STAGE)


def load_response_checkpoint(path, *, expected_sha256=None):
    """Restore only the reference and post-adjoint arrays; no energy is rerun."""
    from pyscf import gto, scf
    from .orbital_response import validate_reference
    path = Path(path)
    manifest = json.loads(Path(str(path)+'.json').read_text())
    if manifest.get('format') != _FORMAT or manifest.get('completed_stage') != _STAGE:
        raise ValueError('Unsupported or incomplete response checkpoint')
    if (path.stat().st_size != manifest['bytes'] or file_sha256(path) != manifest['sha256']
            or (expected_sha256 is not None and manifest['sha256'] != expected_sha256)):
        raise ValueError('Response checkpoint data hash does not match')
    if manifest['source_sha256'] != source_sha256():
        raise ValueError('Response checkpoint source hash does not match')
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(_ARRAYS) | {'metadata'}:
            raise ValueError('Unexpected response checkpoint arrays')
        arrays = {name: data[name].copy() for name in _ARRAYS}
        metadata = json.loads(str(data['metadata']))
    binding = metadata['binding']
    digest = hashlib.sha256(_json(binding).encode()).hexdigest()
    if (metadata.get('format') != _FORMAT or metadata.get('completed_stage') != _STAGE
            or metadata.get('gradient_units') != 'Eh/bohr'
            or metadata['source_sha256'] != manifest['source_sha256']
            or digest != manifest['binding_sha256'] or digest != metadata['binding_sha256']):
        raise ValueError('Response checkpoint provenance does not match')
    reference = binding['reference']
    if not reference['converged']:
        raise ValueError('Response checkpoint reference is not converged')
    mol = gto.loads(reference['molecule'])
    _validate_arrays(mol, arrays)
    for name, expected in binding['orbital_arrays_sha256'].items():
        if _array_sha256(arrays[name]) != expected:
            raise ValueError('Response checkpoint orbital fingerprint does not match')
    mf = scf.RHF(mol)
    if reference['density_fitted']:
        mf = mf.density_fit(auxbasis=reference['auxbasis'], only_dfj=reference['only_dfj'])
        if reference['auxmol'] is not None:
            mf.with_df.auxmol = gto.loads(reference['auxmol'])
    for name in _ARRAYS[:3]:
        setattr(mf, name, arrays[name])
    for name in ('direct_scf_tol', 'conv_tol', 'conv_tol_grad'):
        setattr(mf, name, reference[name])
    mf.e_tot, mf.converged = reference['energy'], True
    physical = binding['physical_settings']
    base = SimpleNamespace(mf=mf, mol=mol, localizer=physical['localizer'], e=physical['energy'])
    validate_reference(base)
    return base, arrays, metadata, dict(path=str(path), sha256=manifest['sha256'],
        binding_sha256=manifest['binding_sha256'], completed_stage=_STAGE)
