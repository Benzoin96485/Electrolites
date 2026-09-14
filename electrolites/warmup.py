"""Fill the caches a first run pays for, before the run rather than inside it.

Nothing here is compiled at install time: every kernel is built by NVRTC at the
moment something first wants it, and CuPy caches the result on disk against the
source hash.  For one calculation that JIT is part of the wall clock -- about
36 s for a 6-31G\\* job and 98 s for a basis that reaches f, which every def2
set on a transition metal does.  For a dataset run it should be the cost of the
first molecule and of nothing after it, and for a *parallel* dataset run it
should be the cost of one process and not of every rank starting at once on a
cold cache.

This module makes that a step you can run rather than a property you hope for.
It walks the same ``_module`` functions the drivers call, with the same
arguments they will ask for, so every compile it does is an entry the run finds
already there -- and it does them in worker processes, so the wall clock is the
longest single compile and not their sum.

    # what a def2-TZVP transition-metal job will want, on eight cores
    python -m electrolites.warmup --basis def2-tzvp --xyz complex.xyz --jobs 8

    # the same without a geometry: one atom of each element is enough, because
    # every argument below is a property of the basis and the elements
    python -m electrolites.warmup --basis def2-tzvp --elements Cu,Cl,C,H,N,O

    # fill a tree that a whole cluster can share, and point jobs at it
    python -m electrolites.warmup --basis def2-svp --elements Fe,Cl \\
        --generated-dir /shared/electrolites/cuda --cupy-cache /shared/cupy
    export ELECTROLITES_GENERATED_DIR=/shared/electrolites/cuda
    export CUPY_CACHE_DIR=/shared/cupy

Three caches are involved and they are filled in this order, because the later
ones read the earlier:

* the **generated CUDA** (`_gen`), 15 MB written by the generators in about
  0.8 s and cached under ``~/.cache/electrolites/generated/``;
* the two **nvcc** libraries (`_nvcc`), about 25 s each, which only `fastk`'s
  general-kernel fall-through and `fastejk2` use;
* the **NVRTC modules**, which is nearly all of the time.

What is compiled depends on the basis and on what the run will do, not on the
molecule: the angular momenta present (`_ksplit.names_within`), whether any bra
primitive pair is longer than the default bound (`_ksplit.mpp_plan`), and
whether the ground state is open-shell (`fastejk`'s ``EJK_NDM``).  Passing a
basis and the elements is therefore enough, and passing a geometry gains
nothing but convenience.
"""
import argparse
import os
import shutil
import sys
import time

from . import _gen
from . import _ksplit

#: Which module compiles what, given a plan.  Each entry returns a list of
#: ``(label, callable)``; the callable compiles one NVRTC module.  A module
#: that is not installed, or whose GPU4PySCF entry point is missing, is
#: skipped rather than failing the warmup.
_JOBS = {}


def _register(name):
    def deco(fn):
        _JOBS[name] = fn
        return fn
    return deco


@_register('fastk')
def _fastk_jobs(plan):
    from . import fastk
    out = [(f'fastk lmax={plan["lmax"]}',
            lambda: fastk._module(plan['lmax']).compile())]
    mpp, mpp_lmax = plan['mpp'], plan['mpp_lmax']
    if mpp != fastk.MAX_PRIM_PAIR:
        out.append((f'fastk mpp={mpp} lmax={mpp_lmax}',
                    lambda: fastk._module(mpp_lmax, mpp).compile()))
    return out


@_register('fastjk')
def _fastjk_jobs(plan):
    from . import fastjk
    out = [(f'fastjk lmax={plan["lmax"]}',
            lambda: fastjk._module(plan['lmax']).compile())]
    mpp, mpp_lmax = plan['mpp'], plan['mpp_lmax']
    if mpp != fastjk.MAX_PRIM_PAIR:
        out.append((f'fastjk mpp={mpp} lmax={mpp_lmax}',
                    lambda: fastjk._module(mpp_lmax, mpp).compile()))
    return out


@_register('fastj')
def _fastj_jobs(plan):
    from . import fastj
    out = [(f'fastj lmax={plan["lmax"]}',
            lambda: fastj._module(plan['lmax']).compile())]
    if plan['response']:
        # the multi-density family, which only a CPHF/TDDFT/polarizability
        # run reaches -- an ordinary SCF never compiles it
        out.append((f'fastj mdm lmax={plan["lmax"]}',
                    lambda: fastj._module(plan['lmax'], 'mdm').compile()))
    return out


@_register('fastejk')
def _fastejk_jobs(plan):
    from . import fastejk
    out = []
    mpp, mpp_lmax = plan['ejk_mpp'], plan['ejk_mpp_lmax']
    for n_dm in ((1, 2) if plan['open_shell'] else (1,)):
        out.append((f'fastejk lmax={plan["lmax"]} n_dm={n_dm}',
                    lambda n=n_dm: fastejk._module(plan['lmax'], n).compile()))
        if mpp != fastejk.MAX_PRIM_PAIR:
            out.append((f'fastejk mpp={mpp} lmax={mpp_lmax} n_dm={n_dm}',
                        lambda n=n_dm, m=mpp, lm=mpp_lmax:
                        fastejk._module(lm, n, m).compile()))
    return out


@_register('nvcc')
def _nvcc_jobs(plan):
    from . import _nvcc
    return [('nvcc libmykg', _nvcc.general_k_library),
            ('nvcc libmyejk2', _nvcc.ejk_ip2_library)]


@_register('fastxcgrad')
def _fastxcgrad_jobs(plan):
    from . import fastxcgrad
    return [('fastxcgrad', lambda: fastxcgrad._module(fastxcgrad._NT)
             .compile())]


@_register('fastxcfxc')
def _fastxcfxc_jobs(plan):
    from . import fastxcfxc
    # `fastfnlc` compiles the same source with the same two definitions, so
    # this entry warms both
    return [('fastxcfxc / fastfnlc',
             lambda: fastxcfxc._module(fastxcfxc._WX, fastxcfxc._TSY)
             .compile())]


@_register('fastnlc')
def _fastnlc_jobs(plan):
    from . import fastnlc
    return [('fastnlc',
             lambda: fastnlc._module(fastnlc.RCP, fastnlc.NRF).compile())]


@_register('fast1e')
def _fast1e_jobs(plan):
    from . import fast1e
    # keyed on the basis' lmax, as the drivers key it
    return [(f'fast1e lmax={plan["lmax"]}',
             lambda: fast1e._module(plan['lmax']))]


#: The grid and one-electron modules: one small kernel each, a second or two
#: together against minutes for the four two-electron families.  They are here
#: so that ``--modules all`` means all of it, not because they are worth a
#: worker.
GRID_MODULES = ('fastxcgrad', 'fastxcfxc', 'fastnlc', 'fast1e')

DEFAULT_MODULES = ('fastk', 'fastj', 'fastjk', 'fastejk', 'nvcc')


def plan_for(basis, elements=None, xyz=None, charge=0, spin=0,
             open_shell=None, response=False, ecp=None):
    """What a run in this basis over these elements will ask NVRTC for.

    Everything in the returned plan is a property of the basis and of the
    elements -- the maximum angular momentum, the longest bra primitive pair,
    whether the second bra-pair module is needed at all -- so a one-atom-per-
    element molecule answers it exactly as the real one would.  The group
    table comes from GPU4PySCF's own ``_VHFOpt``, so it is the table the
    drivers will sort against and not a re-derivation of it.
    """
    from pyscf import gto
    from gpu4pyscf.scf import jk as JK
    from .compat import sorted_meta as _sorted_meta
    from . import fastk, fastejk

    if xyz:
        atoms = xyz
    else:
        # one atom of each element, far enough apart that nothing about the
        # geometry matters -- only the (l, nprim) groups it produces do
        # (8 Angstrom, the default unit, which is far on any scale here)
        syms = [s.strip() for s in elements.split(',') if s.strip()]
        atoms = ';'.join(f'{s} {8.0 * i} 0 0' for i, s in enumerate(syms))
    # ``spin=None`` lets PySCF pick a consistent multiplicity: one atom of
    # each element need not add up to an even electron count, and nothing
    # below depends on the multiplicity -- only on which (l, nprim) groups the
    # basis produces.  ``open_shell`` carries the part that does matter.
    kw = dict(atom=atoms, basis=basis, charge=charge, spin=None, verbose=0)
    if ecp is None:
        ecp = (str(basis).lower().startswith('def2')
               and any(gto.charge(s) > 36 for s, _ in gto.format_atom(atoms)))
    if ecp:
        kw['ecp'] = basis
    mol = gto.M(**kw)
    uniq_l_ctr = _sorted_meta(JK._VHFOpt(mol, 1e-13).build())[0]
    mpp, mpp_lmax = _ksplit.mpp_plan(uniq_l_ctr, fastk.MAX_PRIM_PAIR)
    ejk_mpp, ejk_mpp_lmax = _ksplit.mpp_plan(uniq_l_ctr,
                                             fastejk.MAX_PRIM_PAIR)
    return dict(lmax=int(uniq_l_ctr[:, 0].max()),
                mpp=mpp, mpp_lmax=mpp_lmax,
                ejk_mpp=ejk_mpp, ejk_mpp_lmax=ejk_mpp_lmax,
                open_shell=bool(spin) if open_shell is None else open_shell,
                response=response, nao=int(mol.nao), basis=str(basis))


def _run_one(args):
    """Compile one module in this process; returns ``(label, seconds, error)``."""
    module, index, plan = args
    try:
        jobs = _JOBS[module](plan)
    except Exception as exc:                         # not installed, or no
        return (f'{module}[{index}]', 0.0, repr(exc))   # GPU4PySCF entry point
    if index >= len(jobs):
        return (f'{module}[{index}]', 0.0, 'no such job')
    label, fn = jobs[index]
    t = time.perf_counter()
    try:
        fn()
    except Exception as exc:
        return (label, time.perf_counter() - t, repr(exc))
    return (label, time.perf_counter() - t, None)


def _job_index(module, plan):
    try:
        return len(_JOBS[module](plan))
    except Exception:
        return 0


def warm(plan, modules=DEFAULT_MODULES, jobs=1, generated_dir=None,
         verbose=True):
    """Compile everything ``plan`` says a run will want.  Returns the rows.

    ``jobs`` worker *processes*, not threads: NVRTC is called through CuPy and
    the point of the exercise is wall clock, so the safe way to overlap the
    compiles is to give each one its own interpreter.  They write into the same
    CuPy cache directory, which is keyed on the source hash and written with a
    rename, so two workers that happen to compile the same thing race harmlessly
    rather than corrupting an entry.
    """
    import multiprocessing as mp

    rows = []
    t0 = time.perf_counter()
    if verbose:
        print(f'# basis {plan["basis"]}  lmax {plan["lmax"]}  '
              f'bra-pair bound {plan["mpp"]}'
              + (f' (second module up to l={plan["mpp_lmax"]})'
                 if plan['mpp_lmax'] >= 0 else ' (default only)')
              + f'  open-shell {plan["open_shell"]}', flush=True)
        print(f'# NVRTC cache {os.environ.get("CUPY_CACHE_DIR", "~/.cupy")}, '
              f'generated CUDA {_gen.cache_dir()}', flush=True)

    t = time.perf_counter()
    paths = _gen.ensure_all()
    rows.append(('generated CUDA', time.perf_counter() - t, None))
    if generated_dir:
        os.makedirs(generated_dir, exist_ok=True)
        for name, path in paths.items():
            dst = os.path.join(generated_dir, name)
            if not os.path.exists(dst):
                shutil.copy2(path, dst)
        rows.append((f'copied to {generated_dir}', 0.0, None))

    work = [(m, i, plan) for m in modules for i in range(_job_index(m, plan))]
    if jobs > 1 and len(work) > 1:
        ctx = mp.get_context('spawn')
        with ctx.Pool(min(jobs, len(work))) as pool:
            rows += pool.map(_run_one, work)
    else:
        rows += [_run_one(w) for w in work]

    if verbose:
        for label, dt, err in rows:
            note = '' if err is None else f'   SKIPPED: {err}'
            print(f'  {label:44s} {dt:7.2f}s{note}', flush=True)
        print(f'# {len(rows)} steps, {sum(r[1] for r in rows):.1f}s of work '
              f'in {time.perf_counter() - t0:.1f}s of wall clock '
              f'({jobs} worker{"s" if jobs > 1 else ""})', flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--basis', required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--elements', help='comma separated, e.g. Cu,Cl,C,H,N,O')
    g.add_argument('--xyz', help='a geometry, if one is at hand')
    ap.add_argument('--charge', type=int, default=0)
    ap.add_argument('--spin', type=int, default=0,
                    help='non-zero also compiles fastejk at EJK_NDM=2, which '
                         'is what an open-shell gradient launches')
    ap.add_argument('--response', action='store_true',
                    help='also the multi-density Coulomb family, which only a '
                         'CPHF, TDDFT or polarizability run reaches')
    ap.add_argument('--modules', default=','.join(DEFAULT_MODULES),
                    help=f'default {",".join(DEFAULT_MODULES)}; "all" adds '
                         f'the grid modules, which are seconds rather than '
                         f'minutes')
    ap.add_argument('--jobs', type=int, default=1,
                    help='worker processes.  The compiles are independent, so '
                         'this is close to a straight division of the wall '
                         'clock until it runs out of cores')
    ap.add_argument('--generated-dir',
                    help='copy the generated CUDA here as well, for '
                         'ELECTROLITES_GENERATED_DIR')
    ap.add_argument('--cupy-cache',
                    help='set CUPY_CACHE_DIR for this run, so the NVRTC cache '
                         'lands somewhere a whole cluster can read')
    a = ap.parse_args(argv)

    if a.cupy_cache:
        os.environ['CUPY_CACHE_DIR'] = a.cupy_cache
    modules = (list(DEFAULT_MODULES) + list(GRID_MODULES)
               if a.modules == 'all' else a.modules.split(','))
    unknown = [m for m in modules if m not in _JOBS]
    if unknown:
        sys.stderr.write(f'# nothing to warm for {unknown}; '
                         f'known: {sorted(_JOBS)}\n')
        modules = [m for m in modules if m in _JOBS]
    plan = plan_for(a.basis, elements=a.elements, xyz=a.xyz, charge=a.charge,
                    spin=a.spin, response=a.response)
    warm(plan, modules, jobs=a.jobs, generated_dir=a.generated_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
