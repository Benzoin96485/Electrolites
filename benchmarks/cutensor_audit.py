"""Every cuTENSOR contraction a density-fitted run makes, and how big it gets.

``docs/DF_WORK_LIST.md`` **W27**.  W21's rule (``docs/README_fastdfplan.md``)
says a contraction can be refused only when its *output* passes 2**29
elements.  This script runs a density-fitted SCF, gradient and Hessian with
every call to ``gpu4pyscf.lib.cutensor.contraction`` recorded by call site --
the first frame outside ``gpu4pyscf/lib`` -- with the largest output each site
produced, the shape it had, and cuTENSOR's answer for that shape
(``electrolites/_cuplan.py``, asked of a descriptor, so nothing is allocated).

Each extent of the largest output is then *labelled* with the molecule's
dimension it equals (``nao``, ``nao_c`` for the Cartesian count, ``naux``,
``naux_c``, ``nocc``, ``nvir``, ``nmo``, ``natm``, or a small constant), and
the output is re-evaluated with the labels at another molecule's dimensions
(``--at``, by default HcgC/def2-SVP, 545 atoms).  An extent that matches no
label is a block chosen by the caller -- usually from free memory -- and is
printed as ``blk=<n>``: its bound is the caller's sizing rule, not the
molecule, and ``docs/README_fastdfaudit.md`` says which rule each one has.

    python benchmarks/cutensor_audit.py --xyz benchmarks/molecules/penicillinV.xyz \\
        --basis def2-svp --xc b3lyp --patch all --what scf,grad,hess

``--json`` writes the table for merging runs (``--merge a.json,b.json``).
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--xyz', default='')
ap.add_argument('--charge', type=int, default=0)
ap.add_argument('--spin', type=int, default=0)
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--patch', default='all')
ap.add_argument('--what', default='scf,grad,hess',
                help='any of scf, grad, hess, tdscf (comma separated)')
ap.add_argument('--grids', type=int, default=3)
ap.add_argument('--host-cderi', action='store_true',
                help='keep the cderi on the host (the large-system branch)')
ap.add_argument('--at', default='HcgC',
                help="the molecule to bound the outputs at: a file in "
                     "benchmarks/molecules, or 'nao=..,naux=..,...'")
ap.add_argument('--at-charge', type=int, default=0)
ap.add_argument('--also', default='taxol,PfPMT:-1',
                help="more molecules to print the bound at, smallest first, "
                     "as 'name[:charge],...'")
ap.add_argument('--json', default='')
ap.add_argument('--merge', default='', help='merge these --json files and '
                'print the table; no run')
ap.add_argument('--all', action='store_true',
                help='print every site, not only those whose bound can pass')
a = ap.parse_args()

LIMIT = 1 << 29
HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def dims_of(xyz, charge, spin, basis, auxbasis):
    from pyscf import gto, df
    from _common import needs_ecp
    kw = dict(atom=xyz, basis=basis, charge=charge, spin=spin, verbose=0)
    if needs_ecp(xyz, basis):
        kw['ecp'] = basis
    mol = gto.M(**kw)
    aux = df.make_auxmol(mol, auxbasis)
    nocc = (mol.nelectron + mol.spin) // 2
    return mol, {'nao': mol.nao, 'nao_c': mol.nao_cart(), 'naux': aux.nao,
                 'naux_c': aux.nao_cart(), 'nocc': nocc,
                 'noccb': (mol.nelectron - mol.spin) // 2,
                 'nvir': mol.nao - nocc, 'nmo': mol.nao, 'natm': mol.natm}


def label(n, dims):
    """The names of the dimensions ``n`` equals, most specific first."""
    if n <= 3:
        return str(n)
    order = ['naux_c', 'naux', 'nao_c', 'nao', 'nocc', 'noccb', 'nvir',
             'natm']
    for k in order:
        if dims[k] == n:
            return k
    for k in ('nao_c', 'nao', 'naux_c', 'naux', 'nocc', 'natm'):
        for m in (2, 3, 6, 9):
            if dims[k] * m == n:
                return f'{m}*{k}'
    if n <= 16:
        return str(n)
    return f'blk={n}'


def bound(labels, at):
    tot, free = 1, []
    for lab in labels:
        if lab.startswith('blk='):
            free.append(int(lab[4:]))
            continue
        if '*' in lab:
            m, k = lab.split('*')
            tot *= int(m) * at[k]
        elif lab.isdigit():
            tot *= int(lab)
        else:
            tot *= at[lab]
    return tot, free


def rows_of(sites, dims, at, tag=''):
    """One row per call site: its largest output, labelled and re-bounded."""
    out = {}
    for key, s in sites.items():
        labels = [label(n, dims) for n in s['shape']]
        b, free = bound(labels, at)
        # a block extent is carried at the largest value it was seen at, so
        # the bound of a blocked output is per block of that size
        blk = 1
        for n in free:
            blk *= n
        also = [bound(labels, d)[0] * blk for d in ALSO]
        out[key] = dict(site=key, s=s, labels=labels, per_block=b, free=free,
                        bound=b * blk, also=also, tags={tag} if tag else set())
    return out


def merge_rows(acc, rows):
    for k, r in rows.items():
        t = acc.get(k)
        if t is None:
            acc[k] = r
            continue
        t['tags'] |= r['tags']
        t['s'] = dict(t['s'], calls=t['s']['calls'] + r['s']['calls'],
                      refused=t['s'].get('refused', 0) +
                      r['s'].get('refused', 0))
        if r['bound'] > t['bound']:
            r = dict(r, tags=t['tags'], s=dict(r['s'], calls=t['s']['calls'],
                                               refused=t['s']['refused']))
            acc[k] = r
    return acc


def _first(r):
    if r['bound'] <= LIMIT:
        return '-'
    for n, b in zip(ALSO_NAMES, r['also']):
        if b > LIMIT:
            return n
    return a.at


def table(rows, at):
    rows = sorted(rows.values(), key=lambda r: -r['bound'])
    past = [r for r in rows if r['bound'] > LIMIT]
    print(f'\n# {len(rows)} call sites, {len(past)} with a bound past 2**29; '
          'at: ' + ' '.join(f'{k}={v}' for k, v in at.items()))
    print('# first past 2**29 at: ' + ', '.join(
        f'{n} ({d["natm"]} atoms)' for n, d in zip(ALSO_NAMES, ALSO)) +
          f', or only at the --at molecule ({at["natm"]} atoms)')
    print(f'# {"site":46s} {"pattern":22s} {"calls":>6s} {"max seen":>9s} '
          f'plan  {"labelled shape":30s} {"bound":>9s} first past')
    for r in rows:
        s = r['s']
        site, pattern = r['site'].split('|', 1)
        if not a.all and r['bound'] <= LIMIT:
            continue
        print(f'  {site:46s} {pattern:22s} {s["calls"]:6d} {s["max"]:9.2e} '
              f'{"ok  " if s["plans"] else "NO  "}  '
              f'{"(" + ",".join(r["labels"]) + ")":30s} {r["bound"]:9.2e}'
              f' {_first(r):10s}'
              + (f'  [{",".join(sorted(r["tags"]))}]' if r['tags'] else ''))
    return rows


def _dims_named(spec):
    name, _, ch = spec.partition(':')
    xyz = name if os.path.exists(name) else os.path.join(
        HERE, 'molecules', name + '.xyz')
    return dims_of(xyz, int(ch or 0), 0, a.basis, a.auxbasis)[1]


ALSO_NAMES = [x.partition(':')[0] for x in a.also.split(',') if x]
ALSO = [_dims_named(x) for x in a.also.split(',') if x]

if a.merge:
    acc, at = {}, None
    for f in a.merge.split(','):
        d = json.load(open(f))
        at = at or d['at']
        tag = os.path.basename(f).rsplit('.', 1)[0]
        merge_rows(acc, rows_of(d['sites'], d['dims'], at, tag))
    table(acc, at)
    sys.exit(0)

# ---------------------------------------------------------------------------
# the recorder: installed before anything binds `contract`
# ---------------------------------------------------------------------------
import numpy as np                                            # noqa: E402
import cupy as cp                                             # noqa: E402
import gpu4pyscf.lib.cutensor as G                            # noqa: E402

_orig = G.contraction
SITES = defaultdict(lambda: {'calls': 0, 'max': 0, 'shape': (), 'plans': True,
                             'refused': 0})
_LIB = os.path.join('gpu4pyscf', 'lib') + os.sep
#: planguard's frames sit between a call site and the recorder
_GUARD = ('_cuplan.py', 'planguard.py')


def _site():
    f = sys._getframe(2)
    while f is not None and (_LIB in f.f_code.co_filename or
                             f.f_code.co_filename.endswith(_GUARD)):
        f = f.f_back
    if f is None:
        return '?'
    fn = os.path.realpath(f.f_code.co_filename)
    if 'site-packages' in fn:
        fn = fn.split('site-packages' + os.sep, 1)[1]
    elif fn.startswith(ROOT):
        fn = os.path.relpath(fn, ROOT)
    return f'{fn}:{f.f_lineno}'


def contraction(pattern, x, y, alpha, beta, out=None, **kw):
    p = pattern.replace(' ', '')
    sa, rest = p.split(',')
    sb, sc = rest.split('->')
    ext = dict(zip(sa + sb, tuple(x.shape) + tuple(y.shape)))
    shape = tuple(out.shape) if out is not None else tuple(ext[k] for k in sc)
    n = int(np.prod(shape)) if shape else 1
    s = SITES[f'{_site()}|{p}']
    s['calls'] += 1
    if n > s['max']:
        s['max'], s['shape'] = n, shape
        if n > LIMIT // 4:
            from electrolites import _cuplan
            c = out if out is not None else shape
            s['plans'] = bool(_cuplan.planable(p, x, y, c))
    try:
        return _orig(pattern, x, y, alpha, beta, out=out, **kw)
    except Exception:
        s['refused'] += 1
        raise


G.contraction = contraction

sys.argv = [sys.argv[0]]
from _common import apply_patch                               # noqa: E402

apply_patch(a.patch)
from gpu4pyscf import dft                                     # noqa: E402

mol, dims = dims_of(a.xyz, a.charge, a.spin, a.basis, a.auxbasis)
mol.max_memory = 80000
if a.at.startswith(('nao', 'naux', 'nocc', 'natm')) and '=' in a.at:
    at = dict(dims)
    at.update({k: int(v) for k, v in (x.split('=') for x in a.at.split(','))})
else:
    xyz = a.at if os.path.exists(a.at) else os.path.join(
        HERE, 'molecules', a.at + '.xyz')
    at = dims_of(xyz, a.at_charge, 0, a.basis, a.auxbasis)[1]
print(f'# {mol.natm} atoms  ' + ' '.join(f'{k}={v}' for k, v in dims.items()),
      flush=True)

uks = a.spin != 0
mf = (dft.UKS if uks else dft.RKS)(mol, xc=a.xc).density_fit(
    auxbasis=a.auxbasis)
mf.grids.level = a.grids
mf.conv_tol = 1e-9
mf.verbose = 0
if a.host_cderi:
    mf.with_df.use_gpu_memory = False
what = a.what.split(',')
t0 = time.perf_counter()
e = mf.kernel()
print(f'# SCF {time.perf_counter() - t0:.1f} s  E = {e:.10f}', flush=True)
if 'grad' in what:
    t0 = time.perf_counter()
    mf.nuc_grad_method().kernel()
    print(f'# gradient {time.perf_counter() - t0:.1f} s', flush=True)
if 'hess' in what:
    t0 = time.perf_counter()
    h = mf.Hessian()
    h.verbose = 0
    h.kernel()
    print(f'# Hessian {time.perf_counter() - t0:.1f} s', flush=True)
if 'tdscf' in what:
    t0 = time.perf_counter()
    td = mf.TDA()
    td.nstates = 5
    td.verbose = 0
    td.kernel()
    print(f'# TDA {time.perf_counter() - t0:.1f} s', flush=True)

sites = {k: dict(v) for k, v in SITES.items()}
table(rows_of(sites, dims, at), at)
refused = {k: v['refused'] for k, v in sites.items() if v['refused']}
print(f'# refused on this run: {refused or "none"}')
from electrolites import _cuplan                               # noqa: E402
print(f'# _cuplan {_cuplan.stats()}  (cut: contractions planguard cut)')
if a.json:
    json.dump({'dims': dims, 'at': at, 'sites': sites,
               'args': vars(a)}, open(a.json, 'w'), default=list)
