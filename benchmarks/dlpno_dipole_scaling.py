"""Audit isolated complete-tail hierarchy cost; not full DLPNO scaling.

Synthetic bounded-density chains exercise operation/memory growth. An
optional NPZ of molecular centers/local dipoles/gaps validates the same
kernel against the complete exact virtual-index sum. Outputs belong in
scratch, outside the checkout.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np

from electrolites.dlpno import dipole_laplace, dipole_tree, dipole_tail
from electrolites.dlpno.dipole_laplace import SYMMETRIC_BASIS, exact_pair
from electrolites.dlpno.dipole_tree import MutualTree
from electrolites.dlpno.dipole_tail import DipoleTail


def exact(centers, dipoles, gaps, exclusions):
    excluded = {tuple(sorted(pair)) for pair in exclusions}
    energy, xb = 0., np.zeros_like(centers)
    db, gb = [np.zeros_like(d) for d in dipoles], [np.zeros_like(g) for g in gaps]
    for i in range(len(centers)):
        for j in range(i):
            if (j, i) in excluded:
                continue
            e, di, gi, dj, gj, x = exact_pair(dipoles[i], gaps[i], dipoles[j], gaps[j], centers[i]-centers[j])
            energy += e; db[i] += di; gb[i] += gi; db[j] += dj; gb[j] += gj
            xb[i] += x; xb[j] -= x
    return energy, db, gb, xb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[128, 256, 512, 1024])
    parser.add_argument('--input', type=Path)
    parser.add_argument('--orders', type=int, nargs='+', default=[4, 6, 8])
    parser.add_argument('--theta', type=float, default=.25)
    args = parser.parse_args()
    report = dict(complete=False, scope='isolated complete distant-pair hierarchy; not whole DLPNO',
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  code_sha256={Path(inspect.getfile(m)).name: hashlib.sha256(Path(inspect.getfile(m)).read_bytes()).hexdigest()
                               for m in (dipole_laplace, dipole_tree, dipole_tail)},
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows=[])
    def save():
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2)+'\n'); tmp.replace(args.output)
    save()
    if args.input:
        data = np.load(args.input, allow_pickle=False)
        centers, exclusions = data['centers'], data['exclusions'].tolist()
        ds = [data[f'dipoles_{i}'] for i in range(len(centers))]
        gs = [data[f'gaps_{i}'] for i in range(len(centers))]
        report.update(input=str(args.input), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                      nsource=len(centers))
        start = time.perf_counter(); reference = exact(centers, ds, gs, exclusions)
        report.update(exact_energy=reference[0], exact_with_derivatives_seconds=time.perf_counter()-start)
        save()
        for order in args.orders:
            driver = DipoleTail(order=order, theta=args.theta, channel_block=512)
            start = time.perf_counter()
            forward = driver.evaluate(centers, ds, gs, exclusions=exclusions, derivatives=False)
            forward_seconds = time.perf_counter()-start
            start = time.perf_counter()
            result = driver.evaluate(centers, ds, gs, exclusions=exclusions)
            row = dict(order=order, energy_seconds=forward_seconds,
                       energy_and_adjoint_seconds=time.perf_counter()-start,
                       energy=result[0], energy_error=result[0]-reference[0],
                       energy_forward_consistency=forward[0]-result[0],
                       dipole_source_max_error=max(np.max(abs(a-b)) for a, b in zip(result[1], reference[1])),
                       gap_source_max_error=max(np.max(abs(a-b)) for a, b in zip(result[2], reference[2])),
                       center_derivative_max_error=float(np.max(abs(result[3]-reference[3]))),
                       statistics=result[-1])
            report['rows'].append(row); save(); print(json.dumps(row), flush=True)
    else:
        report['protocol'] = 'Fixed-density chain, 2 separated tensor channels; fixed opening and leaf width. Synthetic kernel audit, not electronic-structure accuracy.'
        for order in args.orders:
            for n in args.sizes:
                rng = np.random.default_rng(761)
                centers = np.zeros((n, 3)); centers[:, 0] = 2.*np.arange(n)+.3
                centers[:, 1:] = rng.uniform(.1, .3, (n, 2))
                matrices = .2*rng.normal(size=(n, 2, 3, 2))
                sources = np.einsum('nqia,nqja,bij->nqb', matrices, matrices, SYMMETRIC_BASIS)
                exclusions = [(i, i+1) for i in range(n-1)]
                start = time.perf_counter()
                tree = MutualTree(centers, order=order, theta=args.theta, exclusions=exclusions)
                construction = time.perf_counter()-start
                start = time.perf_counter(); forward = tree.evaluate(sources, derivatives=False)
                forward_seconds = time.perf_counter()-start
                start = time.perf_counter(); result = tree.evaluate(sources)
                row = dict(order=order, nsource=n, tree_build_seconds=construction,
                           energy_seconds=forward_seconds, energy_and_adjoint_seconds=time.perf_counter()-start,
                           energy=result[0], energy_forward_consistency=forward[0]-result[0],
                           center_derivative_sum=float(np.max(abs(result[2].sum(axis=0)))), statistics=result[-1])
                report['rows'].append(row); save(); print(json.dumps(row), flush=True)
    report.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                  full_correlation_linear_scaling_certified=False)
    save()


if __name__ == '__main__':
    main()
