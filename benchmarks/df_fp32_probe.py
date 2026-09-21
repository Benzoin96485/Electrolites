"""What does a single-precision cderi cost in accuracy?  W7's whole question.

``docs/DF_WORK_LIST.md`` **W7** asks for exactly this table and says what has
to be in it: "report the error it introduces in the total energy and in the
exchange matrix on the §3.2 ladder".  This is that report, and it adds two
things the item does not ask for and that decide whether the mode is usable.

* **The error has to be read against the approximation it sits inside.**
  Density fitting is *already* an approximation to the four-centre integrals,
  worth 1e-4 Eh or so on a small molecule, and a perturbation a thousand
  times smaller than that is a different proposition from one of the same
  size.  ``--exact`` runs the conventional-ERI energy too, below a size cap,
  so the table has both numbers in the same column units.
* **Whether it grows with the system.**  An error per atom that is constant
  is a rounding mode; one that grows is a size limit.  The ladder here is
  ``docs/DF_WORK_LIST.md`` §3.2's, and ``--alkanes`` drives a homologous
  series through it where the only thing changing is the count.

Two chemistries, not one: the closed-shell organics of the ladder and the
transition-metal cases of ``docs/OPENSHELL_TMC.md``, because an exchange
matrix on an open-shell 3d metal is where a narrowed integral would be
expected to show first.

The reference and the narrow run are **one precision at a time** in one
process -- the double tensor is freed before the narrow one is built -- so
that the two never contend for the card (``docs/DF_WORK_LIST.md`` §5.10).
Past about a hundred atoms even that is not enough, and ``--only`` with
``--cache`` splits them over two processes.

    python benchmarks/df_fp32_probe.py --cases glycine,alanine_dipeptide
    python benchmarks/df_fp32_probe.py --cases H2O,CH4,glycine --exact
    python benchmarks/df_fp32_probe.py --cases taxol:0:0 --basis def2-svp \\
        --only fp64 --cache $SCRATCH/fp32
    python benchmarks/df_fp32_probe.py --alkanes 1,2,4,8,16
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--cases', default='glycine,alanine_dipeptide,penicillinV',
                help='comma separated `name[:charge[:spin]]`, resolved as '
                     'benchmarks/molecules/<name>.xyz or as a name in the '
                     'small-molecule table below')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--xc', default='b3lyp', help='empty for HF')
ap.add_argument('--conv-tol', type=float, default=1e-10)
ap.add_argument('--grids', type=int, default=3)
ap.add_argument('--alkanes', default='',
                help='instead of --cases, the idealised all-anti alkane '
                     'series `_common.alkane` builds: "1,2,4,8,16"')
ap.add_argument('--exact', action='store_true',
                help='also run the conventional four-centre energy, so the '
                     'narrowing error can be read against the fitting error '
                     'it sits inside')
ap.add_argument('--max-exact-nao', type=int, default=400)
ap.add_argument('--no-grad', action='store_true')
ap.add_argument('--only', default='both', choices=('both', 'fp64', 'fp32'))
ap.add_argument('--cache', default='',
                help='prefix for the double-precision reference (density, '
                     'exchange matrix, energy), so `--only` can split the '
                     'two precisions over two processes')
a = ap.parse_args()

import cupy as cp                                            # noqa: E402
from pyscf import gto                                        # noqa: E402
from _common import alkane, needs_ecp                        # noqa: E402

import electrolites                                          # noqa: E402
electrolites.patch_all()
from electrolites import fastdf                              # noqa: E402
from electrolites import _dffp32 as FP32                     # noqa: E402
from gpu4pyscf import dft, scf as gscf                       # noqa: E402

SMALL = {
    'H2O': 'O 0.0 0.0 0.117; H 0.0 0.757 -0.467; H 0.0 -0.757 -0.467',
    'CH4': ('C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; '
            'H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629'),
}
HERE = os.path.dirname(os.path.abspath(__file__))


def build_mol(name, charge, spin):
    atom = SMALL.get(name)
    if atom is None and a.alkanes:
        atom = GEOM[name]
    if atom is None:
        atom = os.path.join(HERE, 'molecules', f'{name}.xyz')
    kw = dict(atom=atom, basis=a.basis, charge=charge, spin=spin, verbose=0,
              max_memory=200000)
    if needs_ecp(atom, a.basis):
        kw['ecp'] = a.basis
    return gto.M(**kw)


def make_mf(mol, density_fit=True):
    mf = dft.RKS(mol, xc=a.xc) if a.xc else gscf.RHF(mol)
    if mol.spin:
        mf = dft.UKS(mol, xc=a.xc) if a.xc else gscf.UHF(mol)
    mf.verbose = 0
    mf.conv_tol = a.conv_tol
    if a.xc:
        mf.grids.level = a.grids
    return mf.density_fit(auxbasis=a.auxbasis) if density_fit else mf


def run(mol, fp32):
    """One whole SCF in one precision.  Returns the scalars and the density."""
    FP32._FP32 = fp32
    try:
        mf = make_mf(mol)
        e = float(mf.kernel())
        dm = cp.asarray(mf.make_rdm1())
        cderi = mf.with_df._cderi[0]
        info = dict(e=e, cycles=int(mf.cycles), converged=bool(mf.converged),
                    on_gpu=isinstance(cderi, cp.ndarray),
                    gb=float(cderi.nbytes) * 1e-9,
                    dtype=str(np.dtype(cderi.dtype)))
        if not a.no_grad:
            info['grad'] = np.asarray(mf.nuc_grad_method().kernel())
        vj, vk = mf.with_df.get_jk(dm, hermi=1, with_j=True, with_k=True)
        info['vk'] = cp.asnumpy(cp.asarray(vk)).astype(np.float64)
        info['vj'] = cp.asnumpy(cp.asarray(vj)).astype(np.float64)
        info['dm'] = cp.asnumpy(dm)
        info['ej'], info['ek'] = jk_energies(info['dm'], info['vj'],
                                             info['vk'])
        del mf, vj, vk
        cp.get_default_memory_pool().free_all_blocks()
        return info
    finally:
        FP32._FP32 = False


def jk_at(mol, dm, fp32):
    """J and K at a *given* density, in one precision."""
    FP32._FP32 = fp32
    try:
        dfobj = make_mf(mol).with_df
        dfobj.build()
        cderi = dfobj._cderi[0]
        vj, vk = dfobj.get_jk(cp.asarray(dm), hermi=1, with_j=True,
                              with_k=True)
        out = dict(vj=cp.asnumpy(cp.asarray(vj)).astype(np.float64),
                   vk=cp.asnumpy(cp.asarray(vk)).astype(np.float64),
                   on_gpu=isinstance(cderi, cp.ndarray),
                   gb=float(cderi.nbytes) * 1e-9)
        del dfobj, vj, vk
        cp.get_default_memory_pool().free_all_blocks()
        return out
    finally:
        FP32._FP32 = False


def jk_energies(dm, vj, vk):
    """The Coulomb and exchange energies a J/K pair carries at a density.

    These are what the *total* energy sees, and they are the right place to
    read the narrowing error: a matrix error can be large where the density
    is small and cost nothing.  Evaluated for both precisions at the **same**
    density, their difference is the first-order perturbation the total
    energy should show -- the SCF is stationary, so the density relaxation
    contributes only at second order, and whether that is actually what
    happens is a check rather than an assumption.
    """
    dm, vj, vk = np.asarray(dm), np.asarray(vj), np.asarray(vk)
    if dm.ndim == 3:                       # unrestricted: one row per spin
        ej = .5 * float(np.sum((dm[0] + dm[1]) * (vj[0] + vj[1])))
        ek = -.5 * float(np.sum(dm * vk))
    else:
        ej = .5 * float(np.sum(dm * vj))
        ek = -.25 * float(np.sum(dm * vk))
    return ej, ek


def rel(x, y):
    scale = max(float(np.abs(y).max()), 1e-300)
    return float(np.abs(x - y).max()) / scale


if a.alkanes:
    GEOM = {f'c{n}': alkane(int(n)) for n in a.alkanes.split(',')}
    CASES = [(f'c{n}', 0, 0) for n in a.alkanes.split(',')]
else:
    CASES = []
    for c in a.cases.split(','):
        parts = (c.split(':') + ['0', '0'])[:3]
        CASES.append((parts[0], int(parts[1]), int(parts[2])))

print(f'# gpu4pyscf {electrolites.gpu4pyscf_version()}  basis {a.basis}  '
      f'xc {a.xc or "HF"}  only={a.only}', flush=True)
print(f'{"molecule":18s} {"natm":>4s} {"nao":>5s} {"cderi64":>8s} '
      f'{"cderi32":>8s} {"where":>9s} {"dE":>10s} {"dE/atom":>9s} '
      f'{"dEj":>10s} {"dEk":>10s} {"dEdirect":>10s} {"relK":>9s} '
      f'{"relJ":>9s} {"dgrad":>9s} {"cyc":>8s}',
      flush=True)

rows = []
for name, charge, spin in CASES:
    mol = build_mol(name, charge, spin)
    row = dict(name=name, natm=int(mol.natm), nao=int(mol.nao),
               basis=a.basis, charge=charge, spin=spin)
    cache = f'{a.cache}.{name}.{a.basis}' if a.cache else ''

    ref = None
    if a.only in ('both', 'fp64'):
        ref = run(mol, False)
        if cache:
            np.savez(cache + '.npz', dm=ref['dm'], vk=ref['vk'],
                     vj=ref['vj'],
                     grad=ref.get('grad', np.zeros(0)),
                     scal=np.array([ref['e'], ref['ek'], ref['cycles'],
                                    ref['gb'], ref['on_gpu'], ref['ej']]))
    elif cache and os.path.exists(cache + '.npz'):
        z = np.load(cache + '.npz')
        ref = dict(dm=z['dm'], vk=z['vk'], vj=z['vj'], e=float(z['scal'][0]),
                   ek=float(z['scal'][1]), cycles=int(z['scal'][2]),
                   gb=float(z['scal'][3]), on_gpu=bool(z['scal'][4]),
                   ej=float(z['scal'][5]), converged=True)
        if z['grad'].size:
            ref['grad'] = z['grad']
    if ref is None:
        raise SystemExit(f'{name}: no double-precision reference; run '
                         f'--only fp64 --cache first')

    row.update(e64=ref['e'], ek64=ref['ek'], ej64=ref.get('ej'),
               cycles64=ref['cycles'],
               gb64=ref['gb'], on_gpu64=ref['on_gpu'])

    if a.only == 'fp64':
        print(f'{name:18s} {mol.natm:4d} {mol.nao:5d} {ref["gb"]:7.2f}G '
              f'{"":>8s} {"gpu" if ref["on_gpu"] else "host":>6s} '
              f'{"(reference written)":>40s}', flush=True)
        rows.append(row)
        continue

    got = run(mol, True)
    at_ref = jk_at(mol, ref['dm'], True)
    # both precisions at the *same* density, which is the first-order
    # perturbation the total energy should equal
    ej_d, ek_d = jk_energies(ref['dm'], at_ref['vj'], at_ref['vk'])
    row.update(e32=got['e'], ek32=got['ek'], ej32=got['ej'],
               cycles32=got['cycles'],
               gb32=got['gb'], on_gpu32=got['on_gpu'],
               converged32=got['converged'],
               dE=abs(got['e'] - ref['e']),
               dEk=abs(ek_d - ref['ek']),
               dEj=abs(ej_d - ref['ej']),
               dE_direct=abs((ej_d + ek_d) - (ref['ej'] + ref['ek'])),
               relK=rel(at_ref['vk'], ref['vk']),
               relJ=rel(at_ref['vj'], ref['vj']))
    if 'grad' in got and 'grad' in ref and ref['grad'].size:
        row['dgrad'] = float(np.abs(got['grad'] - ref['grad']).max())

    if a.exact and mol.nao <= a.max_exact_nao:
        mf = make_mf(mol, density_fit=False)
        row['e_exact'] = float(mf.kernel())
        row['dE_df'] = abs(row['e64'] - row['e_exact'])
        del mf
        cp.get_default_memory_pool().free_all_blocks()

    print(f'{name:18s} {mol.natm:4d} {mol.nao:5d} {row["gb64"]:7.2f}G '
          f'{row["gb32"]:7.2f}G '
          f'{("gpu" if row["on_gpu64"] else "host") + "/" + ("gpu" if row["on_gpu32"] else "host"):>9s} '
          f'{row["dE"]:10.2e} {row["dE"]/mol.natm:9.2e} '
          f'{row["dEj"]:10.2e} {row["dEk"]:10.2e} '
          f'{row["dE_direct"]:10.2e} {row["relK"]:9.2e} '
          f'{row["relJ"]:9.2e} '
          f'{row.get("dgrad", float("nan")):9.2e} '
          f'{row["cycles64"]:3d}->{row["cycles32"]:3d}', flush=True)
    if 'dE_df' in row:
        print(f'{"":18s} the fitting error itself: '
              f'|E_DF - E_exact| = {row["dE_df"]:.3e} Eh, '
              f'{row["dE_df"]/max(row["dE"], 1e-300):.0f}x the narrowing '
              f'error', flush=True)
    rows.append(row)

print('RESULT ' + json.dumps(
    dict(basis=a.basis, xc=a.xc, only=a.only, rows=rows), default=float),
    flush=True)
print(f'# fastdf {fastdf.stats()}', flush=True)
print(f'# fp32 {FP32.stats()}', flush=True)
