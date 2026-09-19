"""Is the cderi's sparsity *blocked*, and in which AO order?

`df_fill_probe.py` measures how many AO pairs the screening keeps -- 22.2 % of
the triangle at 284 atoms.  That number is the ceiling on what a sparse
exchange contraction could remove, and it is not reachable: a GEMM cannot
address one pair at a time.  What a GEMM can address is a *tile* of AOs, so
the quantity that decides whether `DF_WORK_LIST.md` W5 is worth writing is the
fraction of `nao x nao` covered by tiles that hold at least one stored pair.

The gap between the two is a question of AO ordering, and this probe is about
that gap.  `electrolites/_dfblock.py` says why any ordering is allowed at all
and where the pair list comes from; the orderings compared here are

* ``native`` -- the order the geometry file lists the atoms in, which is what
  the cderi's addresses already use;
* ``rcm``    -- reverse Cuthill-McKee on the atom adjacency the surviving
  pairs imply, the textbook bandwidth reduction;
* ``rcb``    -- recursive coordinate bisection, so every contiguous chunk is a
  compact spatial cluster;
* ``morton`` -- atoms along a Morton (Z-order) curve, which needs no graph.

``--breakdown`` splits the pair fill into the two independent factors it is a
product of -- how many *atom* pairs survive, and how dense the AO block of a
surviving atom pair is -- which is what says whether a *spatial* reordering
can help at all.

It needs no timing, no SCF and no integrals.

    python benchmarks/df_block_probe.py --basis def2-svp
    python benchmarks/df_block_probe.py --cases PfPMT:0:-1 --breakdown --hist
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--tiles', default='8,16,32,64,128')
ap.add_argument('--orders', default='native,rcm,rcb,morton')
ap.add_argument('--cases', default='penicillinV:0:0,taxol:0:0,PfPMT:0:-1',
                help='comma separated name:spin:charge from benchmarks/molecules')
ap.add_argument('--hist', action='store_true',
                help='print the per-tile-row width distribution, which is what '
                     'decides whether one padded batched call is enough')
ap.add_argument('--breakdown', action='store_true',
                help='split the pair fill into its atom-pair and '
                     'within-atom-block factors')
a = ap.parse_args()

from pyscf import gto                                        # noqa: E402
from electrolites._dfblock import (pair_indices,             # noqa: E402
    atom_order, ao_perm, ao2atom, tile_stats)

print(f'{"molecule":14s} {"order":7s} {"tile":>5s} {"ntile":>6s} '
      f'{"tilefill":>9s} {"pairfill":>9s} {"w_mean":>7s} {"w_max":>6s} '
      f'{"padfill":>8s}')
out = []
for case in a.cases.split(','):
    name, spin, charge = (case.split(':') + ['0', '0'])[:3]
    mol = gto.M(atom=f'benchmarks/molecules/{name}.xyz', basis=a.basis,
                spin=int(spin), charge=int(charge), verbose=0)
    rows0, cols0, nao, naux = pair_indices(mol, a.auxbasis)
    # the pair fill in the tile fill's normalisation: what fraction of the
    # *square* the stored pairs and their mirrors cover
    nnz = 2 * len(rows0) - int((rows0 == cols0).sum())
    pair_fill = nnz / nao**2
    if a.breakdown:
        a2a = ao2atom(mol)
        natm = mol.natm
        M = np.zeros((natm, natm), dtype=bool)
        M[a2a[rows0], a2a[cols0]] = True
        M |= M.T
        cnt = np.bincount(a2a, minlength=natm)
        cov = float((cnt[:, None] * cnt[None, :])[M].sum())
        print(f'# {name}: pair fill {100*pair_fill:.1f}% = atom-pair fill '
              f'{100*M.sum()/natm**2:.1f}% x AO density inside a surviving '
              f'atom block {100*nnz/cov:.1f}%')
    for order in a.orders.split(','):
        perm = ao_perm(mol, atom_order(mol, rows0, cols0, order))
        rows, cols = perm[rows0], perm[cols0]
        for T in [int(x) for x in a.tiles.split(',')]:
            s = tile_stats(rows, cols, nao, T)
            print(f'{name:14s} {order:7s} {T:5d} {s["ntile"]:6d} '
                  f'{100*s["fill"]:8.1f}% {100*pair_fill:8.1f}% '
                  f'{s["width_mean"]:7.0f} {s["width_max"]:6d} '
                  f'{100*s["pad_fill"]:7.1f}%')
            w = s.pop('width')
            if a.hist:
                q = np.percentile(w, [0, 25, 50, 75, 90, 100])
                print('               width percentiles 0/25/50/75/90/100: '
                      + ' '.join(f'{x:.0f}' for x in q))
            s.update(name=name, order=order, nao=nao, naux=naux,
                     natm=int(mol.natm), pair_fill=float(pair_fill),
                     width_p50=float(np.median(w)),
                     width_p90=float(np.percentile(w, 90)))
            out.append(s)
print('RESULT ' + json.dumps(dict(basis=a.basis, rows=out)))
