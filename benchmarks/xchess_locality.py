"""How local is the XC Hessian's atom dimension, at what grid granularity?

`vxc_hess_probe.py` says 90 % of `_get_vxc_deriv1` is the `for ia in
range(natm)` loop, and that at GPU4PySCF's 4096-point blocks the AO mask keeps
only half the AOs while 75 % of the atoms still own at least one masked AO --
so an atom screen at block granularity is worth 1.33x and the two screens
together 2.7x.  That is a property of the block size, not of the molecule: the
term `d rho / d R_A` is nonzero only where atom A's own basis functions are,
and a 4096-point block covers far more space than one atom's basis.

This script measures, at several grid granularities, the fraction of
(atom, chunk) pairs that carry any work at all, and the fraction of
`natm x nao x ngrids` that the two screens together leave -- i.e. the ceiling
for the atom loop, as a function of how finely the grid is chunked.
`ao_frac` is measured on the same chunks, since refining the AO mask with the
chunk is the other half of it.

    python benchmarks/xchess_locality.py --xyz benchmarks/molecules/penicillinV.xyz \
        --basis def2-svp --xc b3lyp --grids 3
"""
import json

import numpy as np
import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--chunks', default='4096,1024,256,64')
ap.add_argument('--cutoff', type=float, default=1e-10,
                help='|ao| below which an AO counts as absent from a chunk; '
                     'GPU4PySCF screens shell blocks at 1e-10 of the AO value')
a = ap.parse_args()
apply_patch(a.patch)

from gpu4pyscf import dft                                    # noqa: E402
from pyscf.gto.mole import ATOM_OF                           # noqa: E402

mol = build_mol(a)
# the two fractions are properties of the molecule, the basis and the grid --
# the AO mask does not know what the density is, let alone how many spin
# channels it has -- so `--spin` here only picks a mean-field object that will
# accept an open-shell molecule, and the numbers it reports are the ceiling
# for the restricted and the unrestricted pair alike.
mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
mf.kernel()

ni = mf._numint
opt = ni.gdftopt
_sorted_mol = opt._sorted_mol
nao = mol.nao
natm = mol.natm
ao_dims = _sorted_mol.ao_loc[1:] - _sorted_mol.ao_loc[:-1]
i_atom_of_sorted_ao = cp.asarray(
    np.repeat(_sorted_mol._bas[:, ATOM_OF], ao_dims), dtype=cp.int32)

chunks = [int(c) for c in a.chunks.split(',')]
# per chunk size: sum over (atom, chunk) of active*n_pts, and sum of
# active*n_pts*nao_sub_of_chunk
work_atom = {c: 0.0 for c in chunks}
work_both = {c: 0.0 for c in chunks}
ao_pts = {c: 0.0 for c in chunks}
ngrids = 0
for ao_mask, mask, weight, _ in ni.block_loop(_sorted_mol, mf.grids, nao, 1,
                                              None):
    blk = len(weight)
    ngrids += blk
    at = i_atom_of_sorted_ao[mask]                    # (nao_sub,) atom of each
    absao = cp.abs(ao_mask[0])                        # (nao_sub, blk)
    for c in chunks:
        nch = (blk + c - 1) // c
        pad = nch * c - blk
        v = absao if pad == 0 else cp.pad(absao, ((0, 0), (0, pad)))
        # per (AO, chunk) maximum |ao|
        m = v.reshape(len(mask), nch, c).max(axis=2)   # (nao_sub, nch)
        live = m > a.cutoff
        # per (atom, chunk): does any AO of the atom live here?
        # several AOs share an atom, so the scatter has to accumulate
        A = cp.zeros((natm, nch), dtype=cp.int32)
        cp.add.at(A, at, live.astype(cp.int32))
        n_live_ao = live.sum(axis=0)                   # (nch,) AOs per chunk
        npts = cp.full(nch, c, dtype=cp.int64)
        if pad:
            npts[-1] = c - pad
        n_at = (A > 0).sum(axis=0)                     # (nch,) atoms per chunk
        work_atom[c] += float((n_at * npts).sum())
        work_both[c] += float((n_at * n_live_ao * npts).sum())
        ao_pts[c] += float((n_live_ao * npts).sum())

print(f'# {a.xyz.rsplit("/",1)[-1]}/{a.basis}/{a.xc}/L{a.grids}  '
      f'natm={natm} nao={nao} ngrids={ngrids}  cutoff={a.cutoff:g}')
print(f'\n{"chunk":>6s} {"atom_frac":>10s} {"ao_frac":>9s} {"both":>9s} '
      f'{"ceiling":>9s}')
print('-' * 48)
out = {}
for c in chunks:
    af = work_atom[c] / (natm * ngrids)
    of = ao_pts[c] / (nao * ngrids)
    bf = work_both[c] / (natm * nao * ngrids)
    print(f'{c:6d} {af:10.4f} {of:9.4f} {bf:9.4f} {1/bf:8.1f}x')
    out[c] = {'atom_frac': af, 'ao_frac': of, 'both_frac': bf}
print('\n' + json.dumps({'tag': f'{a.xyz.rsplit("/",1)[-1]}/{a.basis}',
                         'natm': natm, 'nao': int(nao), 'ngrids': ngrids,
                         'cutoff': a.cutoff, 'chunks': out}))
