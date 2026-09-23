"""What `fastdfgrad` is worth, round robin, in one process.

Five variants of the same quantity, on the same converged density, timed in
turn rather than one after the other -- ``docs/DF_WORK_LIST.md`` §5 item 2 is
why: a competing job's load drifts over the minutes a row takes, and a
sequential schedule puts all of that drift on one variant.

    stock        GPU4PySCF's own method and its ``_jk_energy_per_atom``,
                 which evaluates the three-centre integrals for itself
    cderi only     the reuse, and nothing else
    + spmult       and a quarter of the shell pairs per ip1 thread block
    fastdfgrad     and the SCF's own ``Int3c2eOpt`` instead of a rebuilt one
    + jpair (off)  and the Coulomb term of the pseudo-density on the pair
                   list -- the item that is *off* in the shipped default,
                   because it does strictly less work and is slower

Each timed call is preceded by an **untimed** ``with_df.build()`` when the
cderi is gone, because ``jk_energy_per_atom`` deletes it on its first line and
the point of the module is that it does not have to.  That rebuild is the
SCF's cost, not the gradient's -- an SCF hands its gradient a live cderi -- so
counting it here would price the wrong thing.  ``--with-rebuild`` reports it
anyway, for the reader who wants to see what the tensor costs.

    python benchmarks/df_grad_bench.py \
        --xyz benchmarks/molecules/taxol.xyz --basis def2-svp
"""
import json
import time

import numpy as np
import cupy as cp

from _common import build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--with-rebuild', action='store_true',
                help='also report the untimed cderi rebuild')
ap.add_argument('--whole', action='store_true',
                help='also time the whole gradient per variant')
ap.add_argument('--grad-verbose', type=int, default=0,
                help="the gradient object's verbosity.  5 prints the stage "
                     'timers inside `_ejk`, which is the only stage table '
                     'available at a size where the traced clone in '
                     'df_grad_probe.py cannot allocate')
ap.add_argument('--oostore', default='auto',
                help="where the (naux, nocc, nocc) array goes -- 'auto', "
                     "'device' or 'host'.  docs/DF_WORK_LIST.md W10")
ap.add_argument('--ooslice', type=int, default=0,
                help='occupied columns per slice of the metric pass; 0 takes '
                     'as many as fit')
ap.add_argument('--variants', default='',
                help='comma separated subset of the variants below, for a '
                     'system where the untimed cderi rebuild between rows is '
                     'the expensive part of the run')
a = ap.parse_args()

# this benchmark is about one module and switches it from inside, so it
# patches that module alone rather than going through `--patch`
import electrolites                                           # noqa: E402
electrolites.patch('fastdfgrad')

import gpu4pyscf                                              # noqa: E402
from gpu4pyscf import dft                                     # noqa: E402
from electrolites import fastdfgrad                           # noqa: E402

fastdfgrad._OOSTORE = a.oostore
fastdfgrad._OOSLICE = a.ooslice

_sync = cp.cuda.Stream.null.synchronize

mol = build_mol(a)
mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc).density_fit(
    auxbasis=a.auxbasis)
mf.conv_tol = a.conv_tol
mf.grids.level = a.grids
mf.verbose = 0
if a.cycles:
    mf.max_cycle = a.cycles
mf.kernel()
dm = mf.make_rdm1()
omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
g = mf.nuc_grad_method()
g.verbose = a.grad_verbose

cderi = mf.with_df._cderi[cp.cuda.Device().id]
tag = (f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.auxbasis or "auto"}/{a.xc}')
print(f'[{tag}] natm={mol.natm} nao={mol.nao} naux={mf.with_df.naux} '
      f'hyb={hyb} cderi={cderi.nbytes/1e9:.2f} GB '
      f'on_gpu={isinstance(cderi, cp.ndarray)} '
      f'gpu4pyscf={gpu4pyscf.__version__}', flush=True)

_BASE = dict(_JPAIR=False, _OOSIZING='new', _FILLBUF=True)
VARIANTS = (
    ('stock', dict(_BASE, _CDERI=False, _INTOPT=False, _SPMULT=4)),
    ('cderi only', dict(_BASE, _CDERI=True, _INTOPT=False, _SPMULT=4)),
    ('+ spmult', dict(_BASE, _CDERI=True, _INTOPT=False, _SPMULT=1)),
    ('fastdfgrad', dict(_BASE, _CDERI=True, _INTOPT=True, _SPMULT=1)),
    # the two ablations, both off in the shipped default
    ('+ jpair (off)', dict(_BASE, _CDERI=True, _INTOPT=True, _SPMULT=1,
                           _JPAIR=True)),
    ('old sizing', dict(_BASE, _CDERI=True, _INTOPT=True, _SPMULT=1,
                        _OOSIZING='upstream')),
    ('old fill buf', dict(_BASE, _CDERI=True, _INTOPT=True, _SPMULT=1,
                          _FILLBUF=False)),
)
if a.variants:
    want = [v.strip() for v in a.variants.split(',')]
    unknown = [v for v in want if v not in dict(VARIANTS)]
    if unknown:
        raise SystemExit(f'unknown variant(s) {unknown}; '
                         f'known: {[n for n, _ in VARIANTS]}')
    VARIANTS = tuple((n, f) for n, f in VARIANTS if n in want)
    if VARIANTS[0][0] != 'stock':
        # a system where `stock` cannot be run at all -- or where the untimed
        # cderi rebuild between rows is most of the wall clock -- still has a
        # baseline, it is just the first row rather than upstream's
        print(f'[note] `stock` is not in --variants; ratios and the '
              f'agreement column are against `{VARIANTS[0][0]}`', flush=True)
R = {name: {} for name, _ in VARIANTS}
REBUILD = []
ANS = {}


def rebuild():
    """Put the cderi back, untimed.  It is the SCF's, not the gradient's.

    A **range-separated** functional leaves an SCF holding two of them, the
    full-range tensor and a long-range one under ``with_df._rsh_df`` -- and
    its gradient makes one call against each.  Both have to be restored, or
    the second row of such a run would be timing a fall-through and calling
    it a variant.
    """
    if mf.with_df._cderi is None:
        _sync()
        t = time.perf_counter()
        mf.with_df.build()
        _sync()
        REBUILD.append(time.perf_counter() - t)
    if omega != 0 and alpha != hyb:
        key = '%.6f' % omega
        rsh = mf.with_df._rsh_df.get(key)
        if rsh is None or rsh._cderi is None:
            _sync()
            t = time.perf_counter()
            mf.get_jk(mol, dm, omega=omega)
            _sync()
            REBUILD.append(time.perf_counter() - t)


def timed(fn):
    rebuild()
    _sync()
    t = time.perf_counter()
    out = fn()
    _sync()
    return time.perf_counter() - t, out


def jk_call():
    # `verbose` explicitly, not `g.verbose`: the method takes its logger from
    # this argument and falls back to `mol.verbose`, which is 0 here, so
    # setting it on the gradient object alone prints nothing
    return g.jk_energy_per_atom(dm, j_factor=1, k_factor=hyb or 1., hermi=1,
                                verbose=a.grad_verbose or None)


# one untimed pass of every variant first: the first call in a process pays a
# JIT compile for the ip1 kernel and a cuTENSOR plan for each contraction, and
# `docs/DF_WORK_LIST.md` §5 item 3 is about exactly that being read as a
# speedup
for name, flags in VARIANTS:
    for k, v in flags.items():
        setattr(fastdfgrad, k, v)
    timed(jk_call)

for rep in range(a.reps):
    for name, flags in VARIANTS:
        for k, v in flags.items():
            setattr(fastdfgrad, k, v)
        dt, out = timed(jk_call)
        R[name].setdefault('jk', []).append(dt)
        ANS[name] = out
        if a.whole:
            dt, de = timed(g.kernel)
            R[name].setdefault('grad', []).append(dt)
            ANS[name + '/grad'] = de

BASE = VARIANTS[0][0]
base = min(R[BASE]['jk'])
print(f'\n{"variant":14s} {"jk_energy_per_atom":>20s} {"":>8s}'
      + (f' {"whole gradient":>16s} {"":>8s}' if a.whole else ''))
for name, _ in VARIANTS:
    t = min(R[name]['jk'])
    line = f'{name:14s} {t*1e3:17.1f} ms {base/t:7.3f}x'
    if a.whole:
        tg = min(R[name]['grad'])
        line += f' {tg*1e3:13.1f} ms {min(R[BASE]["grad"])/tg:7.3f}x'
    print(line, flush=True)

print(f'\nagreement against {BASE}, per atom and component:')
for name, _ in VARIANTS[1:]:
    print(f'  {name:14s} {float(np.abs(ANS[name] - ANS[BASE]).max()):.3e} '
          f'Eh/Bohr   (|ejk|max {np.abs(ANS[BASE]).max():.3e})', flush=True)
    if a.whole:
        print(f'  {name + "/grad":14s} '
              f'{float(np.abs(ANS[name + "/grad"] - ANS[BASE + "/grad"]).max()):.3e} '
              f'Eh/Bohr', flush=True)

if a.with_rebuild and REBUILD:
    print(f'\ncderi rebuild (untimed above, the SCF pays it): '
          f'{min(REBUILD):.3f} s, {len(REBUILD)} of them', flush=True)

st = fastdfgrad.stats()
if st['j3c_batched']:
    # at a size where GPU4PySCF's own function cannot allocate its two dense
    # (naux, nocc, nocc) arrays, the `stock` row is *this* module's batched
    # clone of that function rather than the function -- same loop, same
    # integrals, one array instead of two.  Saying so, because a column
    # labelled `stock` that is not upstream's code is the kind of thing
    # docs/DF_WORK_LIST.md §5 is a list of.
    print(f'\n[note] {st["j3c_batched"]} call(s) ran the batched clone of '
          f'GPU4PySCF\'s first loop: at this size upstream\'s own function '
          f'cannot allocate. The `stock` row is that clone.', flush=True)
if st['oo_host']:
    print(f'[note] the (naux, nocc, nocc) array was host-resident on '
          f'{st["oo_host"]} call(s), {st["oo_gb"]:.2f} GB, '
          f'{st["oo_slices"]} metric slice(s).', flush=True)

print(f'\nstats: {fastdfgrad.stats()}', flush=True)
print('RESULT ' + json.dumps(dict(
    tag=tag, gpu4pyscf=gpu4pyscf.__version__, xyz=a.xyz, basis=a.basis,
    auxbasis=a.auxbasis, xc=a.xc, natm=mol.natm, nao=int(mol.nao),
    naux=int(mf.with_df.naux), hyb=float(hyb),
    cderi_gb=cderi.nbytes / 1e9, on_gpu=bool(isinstance(cderi, cp.ndarray)),
    t={k: {kk: min(vv) for kk, vv in v.items()} for k, v in R.items()},
    rebuild=min(REBUILD) if REBUILD else None,
    stats=fastdfgrad.stats())), flush=True)
