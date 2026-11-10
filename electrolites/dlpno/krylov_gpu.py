"""Device-resident, restarted GMRES for real local-correlation adjoints.

Only the small Hessenberg least-squares problem lives on the host. Arnoldi
vectors, residuals and the solution remain on the GPU; matrix actions are
supplied by the exact amplitude transpose, without a dense Jacobian.
"""
import math

import numpy as np


def gmres(matvec, rhs, diagonal, *, restart=40, maxiter=100,
          rtol=1e-10, atol=1e-11, max_residual=None):
    """Solve A*x=rhs with left diagonal preconditioning on a CUDA device.

    ``maxiter`` counts restart cycles, as in SciPy's non-legacy interface.
    Every accepted result satisfies the *unpreconditioned* residual test.
    The operator must return an owned vector, since its result is reused
    in place. All operands are real float64 arrays on one GPU. The rhs and
    diagonal are borrowed read-only. Exact Arnoldi breakdown is checked
    against the true residual, including singular or inconsistent cases.
    If ``max_residual`` is given, convergence also requires every component
    of the unpreconditioned residual to meet that absolute bound.
    """
    import cupy as cp
    from cupy import cublas

    if rhs.dtype != cp.float64 or diagonal.dtype != cp.float64:
        raise TypeError('GPU adjoint GMRES requires float64 operands')
    if rhs.ndim != 1 or diagonal.shape != rhs.shape:
        raise ValueError('GPU adjoint GMRES requires matching flat operands')
    if not isinstance(restart, (int, np.integer)) or restart < 1:
        raise ValueError('restart must be a positive integer')
    if not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError('maxiter must be a positive integer')
    if not math.isfinite(rtol) or not math.isfinite(atol) or min(rtol, atol) < 0:
        raise ValueError('GMRES tolerances must be finite and nonnegative')
    if max_residual is not None and (not math.isfinite(max_residual) or max_residual < 0):
        raise ValueError('Maximum residual tolerance must be finite and nonnegative')
    size = rhs.size
    restart = min(int(restart), size)
    vector_bytes = rhs.nbytes
    x = cp.zeros_like(rhs)
    info = dict(status=0, iterations=0, cycles=0, matvec_calls=0,
                solver_backend='cupy', krylov_restart=restart,
                basis_bytes=(restart+1)*vector_bytes,
                # Basis, solution and one owned work vector. The caller
                # owns rhs, diagonal and the matrix-action workspace.
                solver_array_bytes_bound=(restart+3)*vector_bytes,
                host_declared_array_bytes_bound=8*((restart+1)*restart+4*(restart+1)),
                host_bound_excludes_small_lapack_workspace=True,
                rhs_copied_to_host=False, residual_norm=0., residual_maximum=0.,
                maximum_residual_tolerance=max_residual)
    if size == 0:
        info.update(basis_bytes=0, solver_array_bytes_bound=0,
                    termination='empty_system', tolerance=float(atol))
        return x, info
    norm_b = float(cublas.nrm2(rhs))
    if not math.isfinite(norm_b):
        raise ValueError('Nonfinite GPU GMRES right-hand side')
    tolerance = max(float(atol), float(rtol)*norm_b)
    if norm_b == 0:
        info.update(termination='true_residual_converged', tolerance=tolerance)
        return x, info
    absmax = cp.ReductionKernel('float64 value', 'float64 result',
        'fabs(value)', 'max(a, b)', 'result = a', '0',
        'dlpno_gmres_absolute_maximum')
    def converged(norm, maximum):
        return norm <= tolerance and (max_residual is None or maximum <= max_residual)

    rhs_maximum = float(absmax(rhs))
    if converged(norm_b, rhs_maximum):
        info.update(residual_norm=norm_b, residual_maximum=rhs_maximum,
                    termination='true_residual_converged', tolerance=tolerance)
        return x, info
    basis = cp.empty((restart+1, size), dtype=cp.float64)
    hessenberg = np.zeros((restart+1, restart), dtype=float)
    residual = rhs.copy()
    residual_norm = norm_b
    residual_maximum = rhs_maximum
    inner_tolerance = None
    eps = np.finfo(float).eps

    for cycle in range(int(maxiter)):
        if converged(residual_norm, residual_maximum):
            break
        # The current residual becomes the first preconditioned basis
        # vector. It is released before any matrix action allocates work.
        residual /= diagonal
        beta = float(cublas.nrm2(residual))
        if not math.isfinite(beta) or beta == 0:
            raise RuntimeError('Invalid preconditioned GPU GMRES residual')
        if inner_tolerance is None:
            inner_tolerance = tolerance*(beta/norm_b)
        cp.divide(residual, beta, out=basis[0])
        del residual
        hessenberg.fill(0.)
        breakdown = False
        for j in range(restart):
            work = matvec(basis[j])
            info['matvec_calls'] += 1
            if not isinstance(work, cp.ndarray) or work.shape != rhs.shape or work.dtype != cp.float64:
                raise TypeError('GPU GMRES operator must return a float64 CUDA vector')
            work /= diagonal
            before = float(cublas.nrm2(work))
            # Two-pass modified Gram-Schmidt avoids loss of orthogonality
            # for nearly dependent vectors. BLAS axpy creates no N-vector
            # multiplication temporary.
            for sweep in range(2):
                for k in range(j+1):
                    coefficient = float(cublas.dot(basis[k], work))
                    hessenberg[k, j] += coefficient
                    cublas.axpy(-coefficient, basis[k], work)
            after = float(cublas.nrm2(work))
            if not math.isfinite(before) or not math.isfinite(after):
                raise RuntimeError('Nonfinite GPU GMRES Arnoldi vector')
            hessenberg[j+1, j] = after
            breakdown = after <= eps*before
            if not breakdown:
                cp.divide(work, after, out=basis[j+1])
            del work
            small_rhs = np.zeros(j+2)
            small_rhs[0] = beta
            reduced = hessenberg[:j+2, :j+1]
            coefficients = np.linalg.lstsq(reduced, small_rhs, rcond=None)[0]
            estimated_residual = float(np.linalg.norm(small_rhs-reduced@coefficients))
            info['iterations'] += 1
            if breakdown or estimated_residual <= inner_tolerance:
                break
        for k, coefficient in enumerate(coefficients):
            cublas.axpy(float(coefficient), basis[k], x)
        residual = matvec(x)
        info['matvec_calls'] += 1
        cp.subtract(rhs, residual, out=residual)
        residual_norm = float(cublas.nrm2(residual))
        residual_maximum = float(absmax(residual))
        info['cycles'] = cycle+1
        if not math.isfinite(residual_norm) or not math.isfinite(residual_maximum):
            raise RuntimeError('Nonfinite true GPU GMRES residual')
        if converged(residual_norm, residual_maximum):
            break
        if breakdown:
            info['status'] = info['iterations']
            info['termination'] = 'arnoldi_breakdown_without_convergence'
            break
        if estimated_residual <= inner_tolerance:
            inner_tolerance = min(inner_tolerance*.1, estimated_residual*.1)
    info.update(residual_norm=residual_norm, residual_maximum=residual_maximum,
                tolerance=tolerance)
    if not converged(residual_norm, residual_maximum) and info['status'] == 0:
        info['status'] = info['iterations']
        info['termination'] = 'maximum_restart_cycles'
    elif converged(residual_norm, residual_maximum):
        info['termination'] = 'true_residual_converged'
    return x, info
