"""How much of the ip2 call's ``(nao, nao)`` blocks a tile, or a triangle, reaches.

`docs/DF_WORK_LIST.md` **W39**: three stages of ``df/hessian/rhf.py:
_jk_energy_per_atom`` -- D3's ``M C``, stage A's ``C^T J`` and the
pseudo-density's ``C X C^T`` -- are dense ``(nao, nao)`` products on blocks
filled only at the stored AO pairs of the Hessian's own ``Int3c2eOpt``
(Cartesian, ``pair_and_diag_indices(cart=True)``).  This counts, with no
integrals and no timing,

* ``pairfill``: the fraction of the square the stored pairs and their
  mirrors cover (``2 nao_pair / nao^2`` less the diagonal);
* ``tilefill T``: the fraction covered by ``T x T`` AO tiles that hold at
  least one stored pair -- what a tile-sparse product has to compute
  (W39's falsification: most of the block at a tile size GEMMs run well at);
* ``lower T``: what a product that needs only the lower triangle computes
  when it runs one GEMM per row tile of ``T`` AOs over the columns up to the
  last stored pair of that tile (items 29 and 30), as a fraction of the dense
  product;
* the pair list's orientation (``i > j``, ``i < j``, ``i == j``).

    python benchmarks/df_ip2_tile_probe.py --cases taxol,tmc_co_sextet:5:0,PfPMT:0:-1
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--tiles', default='16,32,64,128')
ap.add_argument('--cases', default='penicillinV,taxol,tmc_co_sextet:5:0,'
                                   'PfPMT:0:-1')
a = ap.parse_args()

from pyscf import gto                                        # noqa: E402
from pyscf.df import addons                                  # noqa: E402
from gpu4pyscf.df import int3c2e_bdiv                        # noqa: E402
from gpu4pyscf.gto.mole import SortedMole                    # noqa: E402

out = []
for case in a.cases.split(','):
    name, spin, charge = (case.split(':') + ['0', '0'])[:3]
    mol = gto.M(atom=f'benchmarks/molecules/{name}.xyz', basis=a.basis,
                spin=int(spin), charge=int(charge), verbose=0)
    aux = addons.make_auxmol(mol, a.auxbasis)
    # as `df/df.py:DF.build` makes the Hessian's intopt
    opt = int3c2e_bdiv.Int3c2eOpt(mol, aux)
    opt.mol = SortedMole.from_mol(mol, decontract=True)
    opt.build()
    addr = opt.pair_and_diag_indices(cart=True, original_ao_order=True)[0]
    addr = addr.get()
    nao = int(mol.nao_cart())
    i, j = addr // nao, addr % nao
    hi, lo = np.maximum(i, j), np.minimum(i, j)
    sq = np.unique(np.concatenate([hi * nao + lo, lo * nao + hi]))
    row = dict(name=name, natm=mol.natm, nao=nao, nao_pair=len(addr),
               pairfill=len(sq) / nao**2,
               lower=float((i > j).mean()), upper=float((i < j).mean()),
               diag=float((i == j).mean()))
    line = (f'{name:14s} {mol.natm:4d} atoms  nao {nao:5d}  pairs '
            f'{len(addr):8d}  pairfill {100*row["pairfill"]:5.1f}%  '
            f'(i>j {100*row["lower"]:.1f}%, i<j {100*row["upper"]:.1f}%, '
            f'i=j {100*row["diag"]:.2f}%)')
    print(line)
    for T in [int(x) for x in a.tiles.split(',')]:
        nt = (nao + T - 1) // T
        key = np.unique((hi // T) * nt + lo // T)
        I, J = key // nt, key % nt
        mask = np.zeros((nt, nt), dtype=bool)
        mask[I, J] = mask[J, I] = True
        size = np.full(nt, T)
        size[-1] = nao - T * (nt - 1)
        tf = float((size[:, None] * size[None, :])[mask].sum()) / nao**2
        # one GEMM per row tile P over the columns [0, qmax(P))
        qmax = np.zeros(nt, dtype=np.int64)
        np.maximum.at(qmax, hi // T, lo + 1)
        lf = float((size * qmax).sum()) / nao**2
        row[f'tilefill_{T}'] = tf
        row[f'lower_{T}'] = lf
        print(f'    T={T:4d}  tilefill {100*tf:5.1f}%   lower-triangle row '
              f'tiles {100*lf:5.1f}%')
    out.append(row)
print('RESULT ' + json.dumps(out))
