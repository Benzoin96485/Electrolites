"""Regenerate the Hessian benchmark geometries from SMILES.

`penicillinV` (42 atoms) and `taxol` (113 atoms) are the two molecules
`instructions/06` §2.2–2.3 profiles, so a Hessian measured on them can be put
next to the catalogue's own numbers; `glycine` and `alanine_dipeptide` are the
small systems `tests/test_xchess.py` uses.  RDKit ETKDGv3 with a fixed seed
then MMFF94, so these are not stationary points -- a Hessian at a
non-stationary geometry is the same computation, and the geometries are
reproducible byte for byte, which matters more here.

The two cluster geometries next to these (`PfPMT`, `HcgC`) are real
structures and are not generated.

    python benchmarks/molecules/generate.py benchmarks/molecules
"""
import sys
from collections import Counter

from rdkit import Chem
from rdkit.Chem import AllChem

SEED = 20260908
SMI = {
    # phenoxymethylpenicillin, C16H18N2O5S
    'penicillinV': 'CC1(C)S[C@@H]2[C@H](NC(=O)COc3ccccc3)C(=O)N2[C@H]1C(=O)O',
    # taxol / paclitaxel, C47H51NO14
    'taxol': 'CC1=C2[C@@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@]'
             '(C2(C)C)(C[C@@H]1OC(=O)[C@@H]([C@H](C1=CC=CC=C1)NC(=O)C1=CC=CC'
             '=C1)O)O)OC(=O)C1=CC=CC=C1)(CO4)OC(=O)C)O)C)OC(=O)C',
    'glycine': 'NCC(=O)O',
    'alanine_dipeptide': 'CC(=O)N[C@@H](C)C(=O)NC',
}

out_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
for name, smi in SMI.items():
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    ps = AllChem.ETKDGv3()
    ps.randomSeed = SEED
    assert AllChem.EmbedMolecule(m, ps) == 0, name
    assert AllChem.MMFFOptimizeMolecule(m, maxIters=5000) in (0, 1), name
    conf = m.GetConformer()
    lines = [f'{m.GetNumAtoms()}',
             f'{name}  {smi}  (RDKit ETKDGv3 seed {SEED} + MMFF94)']
    for at in m.GetAtoms():
        p = conf.GetAtomPosition(at.GetIdx())
        lines.append(f'{at.GetSymbol():2s} {p.x:16.8f} {p.y:16.8f} '
                     f'{p.z:16.8f}')
    with open(f'{out_dir}/{name}.xyz', 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'{name:20s} {m.GetNumAtoms():4d} atoms  '
          f'{dict(Counter(a.GetSymbol() for a in m.GetAtoms()))}')
