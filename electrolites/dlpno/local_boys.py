"""Sparse Boys Hessian on selected occupied-rotation pairs.

The projected Hessian is exact for the supplied dipole matrix. Restricting
the rotation variables is an approximation to the full response, checked
against the original full equation before it can be used for a gradient.
The sparse solve has bounded-neighbor work, but the current dense dipole/
source adapter and full residual oracle are not linear-scaling producers.
"""
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, minres

from .local_screening import atom_neighbors


class LocalBoysResidualError(RuntimeError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(f"Local Boys response full residual {diagnostics['residual']:.3e}; "
                         'increase the occupied-pair radius')


class SparseBoysHessian:
    """Analytic selected-pair Hessian with only incident-pair couplings.

    ``dipole(i,j)`` returns the three Cartesian occupied transition
    dipoles. A callback permits local input production without requiring
    a dense occupied square. The current integration uses ``from_dense``
    as an exact reference adapter, and reports that limitation explicitly.
    """
    def __init__(self, centers, pairs, dipole):
        start = time.perf_counter()
        self.centers = centers = np.asarray(centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
            raise ValueError('Finite occupied centers with shape (n,3) are required')
        self.n = len(centers)
        pairs = sorted(set((max(int(i), int(j)), min(int(i), int(j)))
                           for i, j in pairs if i != j))
        if any(j < 0 or i >= self.n for i, j in pairs):
            raise ValueError('Occupied rotation index out of range')
        self.pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
        # NumPy tril_indices lists rows in the same lower-triangle order.
        self.full_indices = self.pairs[:, 0]*(self.pairs[:, 0]-1)//2+self.pairs[:, 1]
        neighbors = [dict() for _ in range(self.n)]
        for index, (i, j) in enumerate(pairs):
            neighbors[i][j] = index, 1.
            neighbors[j][i] = index, -1.
        values, rows, columns = [], [], []
        queried = {}
        def coupling(i, j):
            key = min(i, j), max(i, j)
            if key not in queried:
                value = np.asarray(dipole(i, j), dtype=float)
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError('Finite three-component transition dipoles are required')
                queried[key] = value
            return queried[key]
        for row, (i, j) in enumerate(pairs):
            dij = coupling(i, j)
            delta = centers[i]-centers[j]
            rows.append(row); columns.append(row)
            values.append(2*np.dot(delta, delta)-8*np.dot(dij, dij))
            for k, (column, sign) in neighbors[i].items():
                if k == j:
                    continue
                value = sign*(np.dot(coupling(k, j), (centers[j]-centers[i])+(centers[k]-centers[i]))
                              -4*np.dot(dij, coupling(i, k)))
                if value:
                    rows.append(row); columns.append(column); values.append(value)
            for k, (column, sign) in neighbors[j].items():
                if k == i:
                    continue
                value = -sign*(np.dot(coupling(i, k), (centers[i]-centers[j])+(centers[k]-centers[j]))
                               -4*np.dot(dij, coupling(j, k)))
                if value:
                    rows.append(row); columns.append(column); values.append(value)
        self.matrix = sparse.csr_matrix((values, (rows, columns)), shape=(len(pairs), len(pairs)))
        antisymmetric = self.matrix-self.matrix.T
        error = float(np.max(abs(antisymmetric.data), initial=0.))
        if error > 1e-10:
            raise RuntimeError(f'Boys sparse Hessian lost symmetry: {error:.3e}')
        self.diagnostics = dict(variables=len(pairs), full_variables=self.n*(self.n-1)//2,
            hessian_nonzeros=self.matrix.nnz, maximum_pair_neighbors=max(map(len, neighbors), default=0),
            dipole_queries=len(queried), symmetry_error=error,
            hessian_build_seconds=time.perf_counter()-start,
            hessian_storage_bytes=self.matrix.data.nbytes+self.matrix.indices.nbytes+self.matrix.indptr.nbytes,
            linear_algebra_conditional_on_bounded_pair_neighbors=True,
            full_response_pipeline_linear=False)

    @classmethod
    def from_dense(cls, dipoles, radius):
        dipoles = np.asarray(dipoles, dtype=float)
        if dipoles.ndim != 3 or dipoles.shape[0] != 3 or dipoles.shape[1] != dipoles.shape[2]:
            raise ValueError('Occupied dipoles must have shape (3,n,n)')
        centers = np.diagonal(dipoles, axis1=1, axis2=2).T
        neighbors, count = atom_neighbors(centers, radius)
        result = cls(centers, ((i, int(j)) for i, adjacent in enumerate(neighbors) for j in adjacent if i > j),
                     lambda i, j: dipoles[:, i, j])
        result.diagnostics.update(pair_radius=float(radius), neighbor_distance_tests=count,
                                  input_producer='dense_dipole_oracle')
        return result

    def expand(self, local):
        result = np.zeros(self.n*(self.n-1)//2)
        result[self.full_indices] = local
        return result


def constraint_dipole_density(centers, pairs, multipliers, dipole):
    """Transpose sum(z_ij * 2 (d_i-d_j).D_ij) into symmetric dipoles.

    The three sparse matrices B satisfy dL = sum_x Tr(B_x.T dD_x)
    for symmetric occupied dipoles. Only selected edges and their
    diagonal endpoints occur; AO backtransformation is a separate step.
    """
    centers = np.asarray(centers, dtype=float)
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    multipliers = np.asarray(multipliers, dtype=float)
    if multipliers.shape != (len(pairs),):
        raise ValueError('One Boys multiplier is required per selected pair')
    n = len(centers)
    diagonal = np.zeros((3, n))
    rows, columns, values = [], [], []
    for (i, j), z in zip(pairs, multipliers):
        transition = np.asarray(dipole(i, j))
        diagonal[:, i] += 2*z*transition
        diagonal[:, j] -= 2*z*transition
        value = z*(centers[i]-centers[j])
        rows.extend((i, j)); columns.extend((j, i))
        values.extend((value, value))
    rows.extend(range(n)); columns.extend(range(n)); values.extend(diagonal.T)
    values = np.asarray(values).reshape(-1, 3)
    return [sparse.csr_matrix((values[:, axis], (rows, columns)), shape=(n, n))
            for axis in range(3)]


def solve_projected_boys(dipoles, source, radius, reference_action, *, residual_tolerance=1e-8):
    """Solve the local equation and require the original full residual.

    The full reference matvec is a correctness oracle, not part of an
    asserted linear-scaling implementation. It cannot be disabled here.
    """
    hessian = SparseBoysHessian.from_dense(dipoles, radius)
    source = np.asarray(source, dtype=float)
    if source.shape != (hessian.n*(hessian.n-1)//2,):
        raise ValueError('Source must be the full packed occupied-rotation vector')
    if not np.isfinite(source).all() or not np.isfinite(residual_tolerance) or residual_tolerance <= 0:
        raise ValueError('Finite source and positive finite residual tolerance are required')
    rhs = -source[hessian.full_indices]
    iterations = []
    start = time.perf_counter()
    if len(rhs):
        diagonal = np.maximum(abs(hessian.matrix.diagonal()), 1e-12)
        preconditioner = LinearOperator(hessian.matrix.shape, matvec=lambda x: x/diagonal, dtype=float)
        local, status = minres(hessian.matrix, rhs, M=preconditioner, rtol=1e-12, maxiter=1000,
                              callback=lambda x: iterations.append(None))
    else:
        local, status = np.zeros(0), 0
    solution = hessian.expand(local)
    solve_seconds = time.perf_counter()-start
    projected_error = float(np.max(abs(hessian.matrix@local-rhs), initial=0.))
    start = time.perf_counter()
    residual = np.asarray(reference_action(solution))+source
    if residual.shape != source.shape or not np.isfinite(residual).all():
        raise FloatingPointError('Invalid full Boys residual')
    error = float(np.max(abs(residual), initial=0.))
    info = dict(hessian.diagnostics, solves=1, iterations=len(iterations), residual=error,
                projected_residual=projected_error, solver_status=int(status),
                solve_seconds=solve_seconds, full_residual_seconds=time.perf_counter()-start,
                full_residual_oracle_used=True)
    if status or projected_error > residual_tolerance or error > residual_tolerance:
        raise LocalBoysResidualError(info)
    return solution, info
