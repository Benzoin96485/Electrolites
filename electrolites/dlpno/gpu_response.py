"""GPU RHF potential actions and relaxed two-electron nuclear contractions.

Response densities can be indefinite and arbitrarily small. Explicit full
factorizations bypass GPU4PySCF's occupied-density eigenvalue cutoffs.
The converged reference and its fitting basis are preserved.
"""
import time

import numpy as np
import cupy as cp
from gpu4pyscf.lib.cupy_helper import tag_array


class _MetricArrayModule:
    """Bound the temporary storage of one owned DF response tensor.

    GPU4PySCF discards ``j3c_oo`` immediately after applying the auxiliary
    metric. Columns of its flattened occupied-pair dimension are
    independent, so each block can replace its input after multiplication.
    All other array operations retain the library implementation.
    """

    def __init__(self, budget, info):
        self.budget, self.info = budget, info

    def __getattr__(self, name):
        return getattr(cp, name)

    def einsum(self, specification, *operands, **kwargs):
        if specification != 'uv,vij->uij' or len(operands) != 2 or kwargs:
            return cp.einsum(specification, *operands, **kwargs)
        metric, tensor = operands
        if (tensor.ndim != 3 or not tensor.flags.c_contiguous
                or metric.shape != (tensor.shape[0], tensor.shape[0])):
            raise ValueError('The streamed DF metric requires an owned contiguous three-index tensor')
        flat = tensor.reshape(tensor.shape[0], -1)
        column_bytes = flat.shape[0]*flat.dtype.itemsize
        width = max(1, self.budget//max(1, 2*column_bytes))
        blocks, workspace = 0, 0
        for start in range(0, flat.shape[1], width):
            stop = min(start+width, flat.shape[1])
            # The RHS copy and product coexist only for this column block.
            block = cp.ascontiguousarray(flat[:, start:stop])
            product = metric@block
            flat[:, start:stop] = product
            blocks += 1
            workspace = max(workspace, block.nbytes+product.nbytes)
            del block, product
        self.info['metric_streaming_calls'] += 1
        self.info['metric_streaming_blocks'] += blocks
        self.info['largest_metric_workspace_bytes'] = max(
            self.info['largest_metric_workspace_bytes'], workspace)
        return tensor



_auxiliary_pair_kernel = None


def _auxiliary_pair_columns(tensor, columns, start, count, transpose, output):
    """Gather only the requested pair columns, without a full transpose copy."""
    global _auxiliary_pair_kernel
    if _auxiliary_pair_kernel is None:
        _auxiliary_pair_kernel = cp.ElementwiseKernel(
            'raw T tensor, int64 columns, int64 start, int64 count, int64 exchange',
            'T output',
            'long long row = i / count; '
            'long long pair = start + i % count; '
            'long long ii = pair / columns, jj = pair % columns; '
            'long long index[3] = {row, exchange ? jj : ii, exchange ? ii : jj}; '
            'output = tensor[index];',
            'dlpno_auxiliary_pair_columns')
    # CArray multi-indexing applies the input strides exactly once, also
    # for negative-stride views; linear raw indexing would apply them again.
    _auxiliary_pair_kernel(tensor, np.int64(columns), np.int64(start), np.int64(count),
                           np.int64(transpose), output)


class _AuxiliaryDensityContraction:
    """Stream the occupied-pair dimension of the DF metric-source density.

    GEMM accumulates into the final auxiliary matrix. Only two pair-column
    buffers are owned here, and together they obey ``budget``. The input,
    output and library-internal BLAS workspace are not counted in that
    application-buffer bound. No density rank or eigenvalue is discarded.
    """

    def __init__(self, original, budget, info):
        self.original, self.budget, self.info = original, int(budget), info

    def __call__(self, specification, a, b, alpha=1., beta=0., out=None):
        if specification not in ('rij,sij->rs', 'rij,sji->rs'):
            return self.original(specification, a, b, alpha=alpha, beta=beta, out=out)
        from cupy.cublas import gemm
        transpose = specification == 'rij,sji->rs'
        if a.ndim != 3 or b.ndim != 3 or a.dtype != b.dtype:
            raise ValueError('The DF auxiliary density needs matching three-index tensors')
        if a.dtype != cp.float64:
            raise ValueError('The DF response contraction requires float64 tensors')
        nr, ni, nj = a.shape
        ns = b.shape[0]
        if b.shape[1:] != ((nj, ni) if transpose else (ni, nj)):
            raise ValueError('Incompatible DF auxiliary-density pair dimensions')
        if out is None:
            out = cp.empty((nr, ns), dtype=a.dtype)
            beta = 0.
        elif (out.shape != (nr, ns) or out.dtype != a.dtype
              or not (out.flags.c_contiguous or out.flags.f_contiguous)):
            raise ValueError('The DF auxiliary-density output must be a contiguous matrix')
        self.info['auxiliary_streaming_calls'] += 1
        if not nr or not ns or not ni or not nj or alpha == 0:
            if beta == 0:
                out.fill(0.)
            else:
                out *= beta
            return out
        column_bytes = (nr+ns)*a.dtype.itemsize
        width = min(ni*nj, self.budget//column_bytes)
        if width < 1:
            raise ValueError('The DF auxiliary workspace must hold at least one pair column')
        left = cp.empty(nr*width, dtype=a.dtype)
        right = cp.empty(ns*width, dtype=a.dtype)
        self.info['largest_auxiliary_workspace_bytes'] = max(
            self.info['largest_auxiliary_workspace_bytes'], left.nbytes+right.nbytes)
        blocks = 0
        for start in range(0, ni*nj, width):
            count = min(width, ni*nj-start)
            al = left[:nr*count]
            br = right[:ns*count]
            _auxiliary_pair_columns(a, nj, start, count, False, al)
            _auxiliary_pair_columns(b, nj, start, count, transpose, br)
            gemm('N', 'T', al.reshape(nr, count), br.reshape(ns, count),
                 out=out, alpha=alpha, beta=beta if blocks == 0 else 1.)
            blocks += 1
        self.info['auxiliary_streaming_blocks'] += blocks
        return out


class GPUReferenceBackend:
    def __init__(self, base, metric_block_bytes=256 << 20):
        if metric_block_bytes is not None and metric_block_bytes <= 0:
            raise ValueError('The reference metric workspace must be positive or None')
        self.metric_block_bytes = metric_block_bytes
        cpu = base.mf.to_cpu() if hasattr(base.mf, 'to_cpu') else base.mf
        self.mf = cpu.to_gpu()
        self.mf.direct_scf_tol = min(cpu.direct_scf_tol, 1e-14)
        self.info = dict(backend='gpu4pyscf', fock_actions=0, quadratic_force_calls=0,
                         fock_seconds=0., quadratic_force_seconds=0., density_factorization='exact_full_rank',
                         metric_workspace_target_bytes=metric_block_bytes,
                         metric_streaming_calls=0, metric_streaming_blocks=0,
                         largest_metric_workspace_bytes=0, auxiliary_workspace_target_bytes=metric_block_bytes,
                         auxiliary_streaming_calls=0, auxiliary_streaming_blocks=0,
                         largest_auxiliary_workspace_bytes=0)

    def prepare_force_contractions(self):
        """Return unused response workspaces before direct CUDA allocations.

        Orbital response leaves reusable blocks in CuPy's default pool.
        GPU4PySCF's final DF force also allocates outside that pool, so its
        allocator cannot reclaim those blocks on demand. This phase boundary
        is reached after all orbital actions: synchronize pending work and
        release only unused blocks, preserving every live reference array.
        """
        cp.cuda.Device().synchronize()
        pool = cp.get_default_memory_pool()

        def snapshot():
            free, total = cp.cuda.runtime.memGetInfo()
            return dict(pool_used_bytes=pool.used_bytes(),
                        pool_reserved_bytes=pool.total_bytes(),
                        device_free_bytes=int(free), device_total_bytes=int(total))

        before = snapshot()
        pool.free_all_blocks()
        after = snapshot()
        report = dict(device=int(cp.cuda.Device().id), before=before, after=after,
                      released_pool_bytes=before['pool_reserved_bytes']-after['pool_reserved_bytes'])
        self.info.setdefault('force_memory_handoffs', []).append(report)
        return report

    @staticmethod
    def _density(matrix):
        value = cp.asarray(matrix, dtype=cp.float64)
        return tag_array(value, factor_l=cp.eye(value.shape[-1]), factor_r=value.T)

    def potential(self, density):
        start = time.perf_counter()
        scale = float(np.max(abs(density), initial=0.))
        if scale == 0.:
            return np.zeros_like(density)
        # get_veff currently strips array tags. Call get_jk directly to
        # preserve the exact signed factorization for both J and K.
        j, k = self.mf.get_jk(self.mf.mol, dm=self._density(density/scale), hermi=1)
        result = cp.asnumpy(j-.5*k)*scale
        self.info['fock_actions'] += 1
        self.info['fock_seconds'] += time.perf_counter()-start
        return result

    def quadratic_force(self, density):
        start = time.perf_counter()
        scale = float(np.max(abs(density), initial=0.))
        if scale == 0.:
            return np.zeros((self.mf.mol.natm, 3))
        # GPU4PySCF's default k_factor=1 includes the RHF exchange 1/2.
        gradient = self.mf.nuc_grad_method()
        force = gradient.jk_energy_per_atom
        if self.metric_block_bytes is not None and getattr(self.mf, 'with_df', None) is not None:
            from .reference_cpu import rebind
            # Bind a private copy of the library namespace. No installed
            # module or global CuPy function is patched, including during
            # concurrent reference calculations in the same interpreter.
            original = force.__func__
            kernel = original.__globals__['_jk_energy_per_atom']
            density_contract = _AuxiliaryDensityContraction(
                kernel.__globals__['contract'], self.metric_block_bytes, self.info)
            bounded = rebind(kernel, cp=_MetricArrayModule(self.metric_block_bytes, self.info),
                             contract=density_contract)
            evaluate = rebind(original, _jk_energy_per_atom=bounded)
            result = evaluate(gradient, self._density(density/scale), hermi=1)*scale**2
        else:
            result = force(self._density(density/scale), hermi=1)*scale**2
        self.info['quadratic_force_calls'] += 1
        self.info['quadratic_force_seconds'] += time.perf_counter()-start
        return cp.asnumpy(result)

    def hf_gradient(self):
        return cp.asnumpy(self.mf.nuc_grad_method().kernel())
