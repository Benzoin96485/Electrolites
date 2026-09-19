"""How much of the AO-pair space does a DF calculation actually store?

The exchange build unpacks the packed cderi into a dense `(nao, nao)` block
per auxiliary function and contracts it densely.  The packed form is smaller
than that dense block, because `int3c2e_bdiv`'s screening drops shell pairs
whose overlap cannot contribute.  Everything the dense contraction does on the
dropped pairs is arithmetic on exact zeros, so this ratio is the ceiling on
what a sparse -- block-sparse over shell pairs -- exchange build could remove,
and it is the one quantity that says whether that is worth writing.

It needs no timing and no SCF: the screening is a property of the geometry and
the two basis sets.

    python benchmarks/df_fill_probe.py --basis def2-svp
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--cases', default='glycine:0:0,alanine_dipeptide:0:0,'
                                   'penicillinV:0:0,taxol:0:0,'
                                   'tmc_ni_dithiolene:1:-1,PfPMT:0:-1',
                help='comma separated name:spin:charge from benchmarks/molecules')
a = ap.parse_args()

from pyscf import gto                                        # noqa: E402
from gpu4pyscf.df import int3c2e_bdiv                        # noqa: E402
from pyscf.df import addons                                  # noqa: E402

rows = []
print(f'{"molecule":22s} {"natm":>5s} {"nao":>6s} {"naux":>6s} '
      f'{"npair":>10s} {"nao(nao+1)/2":>13s} {"fill":>7s} {"cderi/GB":>9s}')
for case in a.cases.split(','):
    name, spin, charge = (case.split(':') + ['0', '0'])[:3]
    mol = gto.M(atom=f'benchmarks/molecules/{name}.xyz', basis=a.basis,
                spin=int(spin), charge=int(charge), verbose=0)
    aux = addons.make_auxmol(mol, a.auxbasis)
    opt = int3c2e_bdiv.Int3c2eOpt(mol, aux)
    from gpu4pyscf.gto.mole import SortedMole
    opt.mol = SortedMole.from_mol(mol, decontract=True)
    opt.build()
    npair = len(opt.pair_and_diag_indices()[0])
    tri = mol.nao * (mol.nao + 1) // 2
    fill = npair / tri
    gb = npair * aux.nao * 8 / 1e9
    print(f'{name:22s} {mol.natm:5d} {mol.nao:6d} {aux.nao:6d} '
          f'{npair:10d} {tri:13d} {100*fill:6.1f}% {gb:9.2f}')
    rows.append(dict(name=name, natm=mol.natm, nao=int(mol.nao),
                     naux=int(aux.nao), npair=int(npair), tri=int(tri),
                     fill=float(fill), cderi_gb=float(gb)))
print('RESULT ' + json.dumps(dict(basis=a.basis, auxbasis=a.auxbasis,
                                  rows=rows)))
