"""Where the time goes inside the three-centre density-fitted Coulomb build.

`docs/DF_WORK_LIST.md` **W9**.  A *pure* density-fitted functional never
builds a cderi -- `df/df.py:111` returns before the tensor is touched -- so its
whole two-electron cost is `df/df_jk.py:get_j`, which is

    dm  -> mol.apply_C_mat_CT          (decontract the density)
        -> Int3c2eOpt.contract_dm      (rho_P = sum_ij (ij|P) D_ij)
        -> auxmol.apply_CT_dot
        -> two triangular solves with the Coulomb metric
        -> auxmol.apply_C_dot
        -> Int3c2eOpt.contract_auxvec  (vj_ij = sum_P (ij|P) rho_P)
        -> mol.apply_CT_mat_C

and the two `contract_*` calls are each a host-side setup, a Hermite
transformation kernel and one McMurchie-Davidson integral kernel out of
`libgvhf_md`.  `docs/README_fastdfj.md` §3.1 measured the whole of that at
0.736 s on 284 atoms and called the engine good; this script says which of its
nine stages the 0.736 s is in, which is the only thing that can rank the work
inside W9.

Nothing is patched by this script beyond a timer: every stage is GPU4PySCF's
own function, wrapped so that its device work is synchronised and charged to
it.  The sum of the stages is therefore slightly *larger* than an unwrapped
`get_j` -- the syncs serialise what would otherwise overlap -- and the
`get_j (unwrapped)` row is printed alongside so the reader can see by how
much.

    python benchmarks/df_j3c_probe.py --cases glycine,penicillinV,taxol
    python benchmarks/df_j3c_probe.py --cases PfPMT:-1 --basis def2-svp
    python benchmarks/df_j3c_probe.py --cases penicillinV --ndm 1,2,4
    python benchmarks/df_j3c_probe.py --cases penicillinV --shapes  # no GPU time
"""
import argparse
import ctypes
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MOL = os.path.join(HERE, 'molecules')

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--cases', default='glycine,alanine_dipeptide,penicillinV',
                help='comma separated NAME[:charge[:spin]]; NAME is a file in '
                     'benchmarks/molecules without the .xyz')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--ndm', default='1', help='comma separated density counts')
ap.add_argument('--reps', type=int, default=5)
ap.add_argument('--cycles', type=int, default=1,
                help='run this many SCF cycles first and probe on the '
                     'resulting density rather than on the `minao` guess')
ap.add_argument('--patch', default='',
                help="'all' or a module list, to probe a patched build")
ap.add_argument('--shapes', action='store_true',
                help='print the pair/auxiliary shapes and the formal work '
                     'count only; no GPU time, so this runs on a login node')
a = ap.parse_args()

#: `fastdfj3c`, or None when this run is meant to be GPU4PySCF's own.
#: **Importing it patches**, so it is imported inside the `--patch` branch and
#: nowhere else: a module-level `from electrolites import fastdfj3c` here
#: silently turned the stock column into a patched one, which is
#: `docs/DF_WORK_LIST.md` §5.12 in a third costume.
_FAST = None

if a.patch:
    sys.path.insert(0, os.path.dirname(HERE))
    import electrolites
    if a.patch == 'all':
        electrolites.patch_all()
    else:
        electrolites.patch(a.patch.split(','))
    print(f'# patched: {electrolites.patched() or "(none)"}', flush=True)
    if 'fastdfj3c' in electrolites.patched():
        from electrolites import fastdfj3c as _FAST

import cupy as cp                                            # noqa: E402
from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                     # noqa: E402
from gpu4pyscf.df import df_jk, j_engine_3c2e, int3c2e_bdiv   # noqa: E402
from pyscf.df.addons import make_auxmol                       # noqa: E402
from gpu4pyscf.gto.mole import SortedGTO, SortedMole          # noqa: E402

_sync = cp.cuda.Stream.null.synchronize

#: stage -> seconds, filled by the wrappers below
T = {}
#: stage -> how many times it was entered
N = {}
#: the order the table prints in
ORDER = ['apply_C_mat_CT', 'contract_dm.setup', 'contract_dm.aggregate',
         'contract_dm.dm_to_Rt',
         'contract_dm.kernel', 'apply_CT_dot', 'metric', 'apply_C_dot',
         'contract_auxvec.setup', 'contract_auxvec.aggregate',
         'contract_auxvec.aux_to_Rt',
         'contract_auxvec.kernel', 'contract_auxvec.Rt_to_dm',
         'apply_CT_mat_C']


def _reset():
    T.clear()
    N.clear()


def _charge(tag):
    """``seconds -> add to T[tag]``, synchronising the device first."""
    def add(dt):
        T[tag] = T.get(tag, 0.) + dt
        N[tag] = N.get(tag, 0) + 1
    return add


def _wrap(obj, name, tag, is_ctypes=False):
    """Time one function in place and return the original."""
    orig = getattr(obj, name)
    add = _charge(tag)

    def timed(*args, **kw):
        _sync()
        t0 = time.perf_counter()
        r = orig(*args, **kw)
        _sync()
        add(time.perf_counter() - t0)
        return r
    setattr(obj, name, timed)
    return orig


class Instrumented:
    """Wrap every stage of ``df_jk.get_j`` for the duration of a ``with``.

    The two ``contract_*`` methods are timed as a whole and their two kernels
    separately; the difference is the host-side setup, which is the thing this
    script exists to size.
    """

    def __enter__(self):
        self.saved = []
        lib = j_engine_3c2e.libvhf_md

        def save(obj, name, tag):
            self.saved.append((obj, name, _wrap(obj, name, tag)))

        save(SortedGTO, 'apply_C_mat_CT', 'apply_C_mat_CT')
        save(SortedGTO, 'apply_CT_mat_C', 'apply_CT_mat_C')
        save(SortedGTO, 'apply_CT_dot', 'apply_CT_dot')
        save(SortedGTO, 'apply_C_dot', 'apply_C_dot')
        save(j_engine_3c2e, '_dm_to_Rt', 'contract_dm.dm_to_Rt')
        save(j_engine_3c2e, '_Rt_to_dm', 'contract_auxvec.Rt_to_dm')
        # `fastdfj3c` calls the same two CUDA kernels through copies of its
        # own (they take the cached `ao_loc` rather than uploading it), so a
        # patched run would charge them to `setup` and the two columns would
        # not be comparable
        if _FAST is not None:
            save(_FAST, '_dm_to_Rt', 'contract_dm.dm_to_Rt')
            save(_FAST, '_Rt_to_dm', 'contract_auxvec.Rt_to_dm')
        save(lib, 'contract_int3c2e_dm', 'contract_dm.kernel')
        save(lib, 'contract_int3c2e_auxvec', 'contract_auxvec.kernel')
        save(lib, 'aux_to_Rt', 'contract_auxvec.aux_to_Rt')
        # the *Rys* engine, which `df_jk.get_j` uses instead whenever more
        # than one density is asked for -- a response solve on a pure
        # functional is entirely on it.  Its two kernels and its per-call
        # shell-pair aggregation are charged to the same rows.
        save(int3c2e_bdiv.libvhf_rys, 'contract_int3c2e_dm',
             'contract_dm.kernel')
        save(int3c2e_bdiv.libvhf_rys, 'contract_int3c2e_auxvec',
             'contract_auxvec.kernel')
        save(SortedMole, 'aggregate_shl_pairs', 'aggregate')
        save(j_engine_3c2e.Int3c2eOpt, 'contract_dm', 'contract_dm')
        save(j_engine_3c2e.Int3c2eOpt, 'contract_auxvec', 'contract_auxvec')
        save(int3c2e_bdiv.Int3c2eOpt, 'contract_dm', 'contract_dm')
        save(int3c2e_bdiv.Int3c2eOpt, 'contract_auxvec', 'contract_auxvec')
        # the two triangular solves against the Coulomb metric, which
        # `df_jk.get_j` does inline between them
        save(df_jk, 'solve_triangular', 'metric')
        return self

    def __exit__(self, *exc):
        for obj, name, orig in reversed(self.saved):
            setattr(obj, name, orig)
        return False


def _derive():
    """Fill in the two setup rows from the totals.

    ``aggregate`` is charged once and split between the two contractions in
    the ratio of the calls each made, because the Rys engine runs it inside
    both.
    """
    agg = T.pop('aggregate', 0.)
    nagg = N.pop('aggregate', 0)
    halves = []
    for whole, parts in (
            ('contract_dm', ('contract_dm.dm_to_Rt', 'contract_dm.kernel')),
            ('contract_auxvec', ('contract_auxvec.aux_to_Rt',
                                 'contract_auxvec.kernel',
                                 'contract_auxvec.Rt_to_dm'))):
        if whole in T:
            halves.append((whole, parts))
    for whole, parts in halves:
        share = agg / len(halves) if halves else 0.
        T[f'{whole}.aggregate'] = share
        N[f'{whole}.aggregate'] = nagg / max(len(halves), 1)
        T[f'{whole}.setup'] = (T[whole] - share
                               - sum(T.get(p, 0.) for p in parts))
        N[f'{whole}.setup'] = N[whole]


def load_case(spec):
    name, _, rest = spec.partition(':')
    charge, _, spin = rest.partition(':')
    xyz = os.path.join(MOL, name + '.xyz')
    mol = gto.M(atom=xyz, basis=a.basis, charge=int(charge or 0),
                spin=int(spin or 0), verbose=0, max_memory=80000)
    return name, mol


def shapes(mol):
    """Pair/auxiliary counts and the formal (ij|P) work, without a GPU."""
    auxmol = make_auxmol(mol, a.auxbasis)
    smol = SortedGTO.from_mol(mol, decontract=True, diffuse_cutoff=1e200)
    saux = SortedGTO.from_mol(auxmol, decontract=True, diffuse_cutoff=1e200)
    return dict(nao=mol.nao, naux=auxmol.nao,
                nao_p=int(smol.ao_loc[-1]), naux_p=int(saux.ao_loc[-1]),
                nbas=smol.nbas, nauxbas=saux.nbas)


def run_case(spec, ndms):
    name, mol = load_case(spec)
    sh = shapes(mol)
    print(f'\n## {name}  {mol.natm} atoms  nao {sh["nao"]} '
          f'(decontracted {sh["nao_p"]})  naux {sh["naux"]} '
          f'(decontracted {sh["naux_p"]})  '
          f'aux shells {sh["nauxbas"]}', flush=True)
    if a.shapes:
        return

    cls = dft.UKS if mol.spin else dft.RKS
    mf = cls(mol, xc='pbe').density_fit(auxbasis=a.auxbasis)
    mf.grids.level = 3
    mf.verbose = 0
    mf.conv_tol = 1e-9
    if a.cycles:
        mf.max_cycle = a.cycles
        mf.kernel()
        dm0 = cp.asarray(mf.make_rdm1())
        dm0 = dm0.reshape(-1, dm0.shape[-1], dm0.shape[-1])[0]
    else:
        dm0 = cp.asarray(mf.get_init_guess())
        dm0 = dm0.reshape(-1, dm0.shape[-1], dm0.shape[-1])[0]
    dfobj = mf.with_df
    if dfobj.intopt is None:
        dfobj.build(build_cderi=False)

    for ndm in ndms:
        dm = dm0 if ndm == 1 else cp.repeat(dm0[None], ndm, axis=0)
        # warm every cache df_jk.get_j keeps on the DF object
        df_jk.get_j(dfobj, dm, 1)
        _sync()

        t_raw = []
        for _ in range(a.reps):
            _sync()
            t0 = time.perf_counter()
            df_jk.get_j(dfobj, dm, 1)
            _sync()
            t_raw.append(time.perf_counter() - t0)

        _reset()
        with Instrumented():
            for _ in range(a.reps):
                df_jk.get_j(dfobj, dm, 1)
        _derive()

        engine = ('j_engine_3c2e (MD)' if ndm == 1 else
                  'int3c2e_bdiv (Rys)')
        tot = sum(T.get(k, 0.) for k in ORDER) / a.reps
        print(f'\n  ndm = {ndm}, engine {engine}, '
              f'min of {a.reps} unwrapped: {min(t_raw)*1e3:.2f} ms, '
              f'instrumented mean: {tot*1e3:.2f} ms')
        print(f'  {"stage":30s} {"ms/call":>9s} {"%":>6s} {"calls":>6s}')
        for k in ORDER:
            if k not in T:
                continue
            ms = T[k] / a.reps * 1e3
            print(f'  {k:30s} {ms:9.3f} {100*T[k]/sum(T[x] for x in ORDER if x in T):6.1f} '
                  f'{N[k]/a.reps:6.1f}')
        sys.stdout.flush()


def main():
    ndms = [int(x) for x in a.ndm.split(',')]
    print(f'# basis {a.basis}  auxbasis {a.auxbasis or "(default jkfit)"}  '
          f'cycles {a.cycles}  reps {a.reps}')
    for spec in a.cases.split(','):
        run_case(spec, ndms)


main()
