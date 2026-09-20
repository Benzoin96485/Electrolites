"""Three pieces of the density-fitted build that two modules need.

`electrolites/fastdfcderi.py` wrote all of them and `electrolites/dflocal.py`
uses them too -- the sparse form of the auxiliary contraction coefficients,
the one-pass kernel that applies it, and the shell-pair aggregation constant.
They live here rather than in `fastdfcderi` because **importing that module
patches** ``df/df.py:_cholesky_eri``: a module that only wanted the kernel
would silently change the build it is being compared against, which is
exactly the class of mistake `docs/DF_WORK_LIST.md` §5 is a list of.
"""
import os

import numpy as np
import cupy as cp

from gpu4pyscf.df.int3c2e_bdiv import int3c2e_scheme
from gpu4pyscf.gto.mole import SortedMole

#: how many shell pairs one thread block of the three-centre kernel is given,
#: as a multiple of the scheme's `nsp_per_block`.  GPU4PySCF fixes it at 4
#: inside `int3c2e_evaluator`; 1 is a measurement -- see
#: `benchmarks/df_cderi_probe.py` and `docs/README_fastdfcderi.md` section 3.
#: The name keeps `fastdfcderi`'s prefix because that is the module the
#: switch is documented in and the one every ablation column was measured
#: with.
_SPMULT = int(os.environ.get('FASTDFCDERI_SPMULT', 1))


# --------------------------------------------------------------------------
# the aggregation constant, scoped to one evaluator construction
# --------------------------------------------------------------------------
_orig_aggregate = SortedMole.aggregate_shl_pairs


class _spmult:
    """``nsp_per_block*sp_mult`` in ``int3c2e_evaluator``, for one call.

    The constant is written into upstream's body, so the only way to reach it
    without copying the whole evaluator is to intercept what it hands
    ``aggregate_shl_pairs``.  The interception asserts that what it sees is
    the scheme's own array times 4: if upstream ever changes that constant
    the assertion fires rather than this module quietly changing nothing --
    or, worse, changing it by the wrong factor.
    """

    def __init__(self, mult, short_range=False):
        self.mult = mult
        self.ref = int3c2e_scheme(short_range=short_range, gout_width=54,
                                  cache_cart_idx=True)[0][0]

    def __enter__(self):
        mult, ref = self.mult, self.ref

        def aggregate_shl_pairs(this, cache=None, nsp_per_block=512):
            if not isinstance(nsp_per_block, (int, np.integer)):
                assert cp.array_equal(nsp_per_block, ref * 4), \
                    'int3c2e_evaluator no longer aggregates at nsp_per_block*4'
                nsp_per_block = ref * mult
            return _orig_aggregate(this, cache, nsp_per_block)

        SortedMole.aggregate_shl_pairs = aggregate_shl_pairs
        return self

    def __exit__(self, *exc):
        SortedMole.aggregate_shl_pairs = _orig_aggregate


# --------------------------------------------------------------------------
# the contraction coefficients, as the sparse matrix they are
# --------------------------------------------------------------------------
#: ``out[p,L] = sum_k coef[L,k] j3c[p, idx[L,k]]``.  One thread per output
#: element, reading at most ``maxnnz`` entries of one row of ``j3c``; the
#: column index already carries the auxiliary reordering, so the integral
#: kernel keeps the layout it prefers.
_contract_aux = cp.ElementwiseKernel(
    'raw float64 j3c, raw int32 idx, raw float64 coef, raw int32 nnz, '
    'int32 nsorted, int32 naux, int32 maxnnz',
    'float64 out',
    """
    const int L = i % naux;
    const long long p = i / naux;
    const int n = nnz[L];
    double s = 0.;
    for (int k = 0; k < n; ++k) {
        s += coef[L * maxnnz + k] * j3c[p * (long long)nsorted
                                        + idx[L * maxnnz + k]];
    }
    out = s;
    """,
    'fastdfcderi_contract_aux')


def _sparse_ctr_coeff(auxmol, aux_sorting=None):
    """The contraction coefficients as a padded column-sparse triple.

    ``auxmol.ctr_coeff`` is the Cartesian-to-spherical and recontraction
    matrix of the auxiliary basis, built densely as
    ``C_dot_mat(eye(nao))``; it carries about 1.5 non-zeros per column
    (0.02-0.3 % of its entries, measured in ``benchmarks/df_cderi_probe.py``)
    because a spherical function is a short combination of the Cartesian ones
    of its own shell.  ``aux_sorting`` folds the layout the integral kernel
    writes into the row indices, so nothing has to be permuted per block.
    """
    c = auxmol.ctr_coeff
    if aux_sorting is not None:
        c, tmp = cp.empty_like(c), c
        c[aux_sorting] = tmp
    nz = c != 0
    counts = nz.sum(axis=0).astype(np.int32)
    maxnnz = int(counts.max())
    cols, rows = cp.nonzero(nz.T)
    offsets = cp.zeros(len(counts), dtype=np.int64)
    offsets[1:] = cp.cumsum(counts)[:-1]
    pos = cp.arange(cols.size) - offsets[cols]
    idx = cp.zeros((len(counts), maxnnz), dtype=np.int32)
    coef = cp.zeros((len(counts), maxnnz))
    idx[cols, pos] = rows.astype(np.int32)
    coef[cols, pos] = c[rows, cols]
    return idx, coef, counts, maxnnz
