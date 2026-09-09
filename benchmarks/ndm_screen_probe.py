"""Catalogue B4c: is a multi-density K build's screening really a union?

Entry **B4c** of `instructions/06` says the screening in a CPHF or TDDFT
exchange build is a union over the perturbed densities that costs far more than
it should, because "each `dm1[A]` has support only near atom A, yet every
density pays the full quartet list".  `scf/jk.py`'s `dm_cond = condense('absmax',
dms, ao_loc)` is one `nbas x nbas` table -- the max over all `n_dm` -- and the
task filter in `create_tasks.cu` tests against that, so the claim about the code
is right.  Whether the claim about the *densities* is right is a measurement,
and this is it.

For each shell quartet that survives the union screen, count how many of the
`n_dm` densities would survive a screen built from **their own** `dm_cond`.
The ratio

    sum_quartets (densities passing their own screen)
    -----------------------------------------------------
    n_dm * (quartets passing the union screen)

is the fraction of the contraction work a per-density screen would leave, i.e.
the reciprocal of B4c's ceiling on the contraction.  Every count is weighted by
`nfi*nfj*nfk*nfl`, the contraction work of the class, as well as reported raw.

The densities are not invented: they are the ones a real solve hands to
`get_k`.  `--stage hessian` captures the perturbed densities of an analytic
Hessian's CPHF solve (`n_dm = 3*natm`, the entry's own case), `--stage cphf`
those of a polarizability, `--stage tda` those of a Davidson solve.  Each
`get_k` call is a separate row, because the first iteration's densities are the
atom-local right-hand sides and the later ones are not.

    python benchmarks/ndm_screen_probe.py --xyz molecules/penicillinV.xyz \
        --basis def2-svp --stage hessian
"""
import json

import cupy as cp
import numpy as np

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--stage', default='hessian',
                choices=['hessian', 'cphf', 'tda'])
ap.add_argument('--conv-tol-cpscf', type=float, default=1e-6)
ap.add_argument('--nstates', type=int, default=5)
ap.add_argument('--max-calls', type=int, default=3,
                help='analyse at most this many get_k calls')
ap.add_argument('--chunk', type=int, default=32,
                help='bra pairs per chunk of the (ij, kl, i_dm) count')
a = ap.parse_args()
apply_patch(a.patch)

from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.scf import jk as JK                           # noqa: E402
from gpu4pyscf.lib.cupy_helper import condense               # noqa: E402

from electrolites.compat import (                            # noqa: E402
    sorted_meta as _sorted_meta, dense_q_cond as _dense_q_cond,
    tril_pair_mappings as _tril_pair_mappings)

_sync = cp.cuda.Stream.null.synchronize
ROWS = []


def _nf(l):
    return (l + 1) * (l + 2) // 2


def analyse(opt, dms, hermi, tag):
    """One get_k call: union survivors against per-density survivors."""
    mol = opt.sorted_mol
    nbas = mol.nbas
    if hermi == 0:
        dms = cp.vstack([dms, dms.transpose(0, 2, 1)])
    n_dm = dms.shape[0]
    # exactly jk.py's tables, but kept per density as well as maxed over them
    per = cp.stack([condense('absmax', d[None], mol.ao_loc) for d in dms])
    per = cp.log(cp.abs(per) + 1e-300).astype(np.float32)
    assert per.ndim == 3 and per.shape[1:] == (nbas, nbas), per.shape
    union = per.max(axis=0)
    cutoff = np.float32(np.log(opt.direct_scf_tol))
    q_cond = _dense_q_cond(opt)
    uniq_l_ctr, l_ctr_offsets = _sorted_meta(opt)
    pair_mappings = _tril_pair_mappings(
        l_ctr_offsets, q_cond, float(cutoff) - float(union.max()), tile=6)
    uniq_l = uniq_l_ctr[:, 0]
    n_groups = int(np.count_nonzero(uniq_l <= JK.LMAX))

    tot_u = tot_p = 0            # quartets: union survivors, (quartet, dm)
    cost_u = cost_p = 0.0        # the same, weighted by the class's nf product
    for i in range(n_groups):
        for j in range(i + 1):
            for k in range(i + 1):
                for l in range(k + 1):
                    pm_ij = pair_mappings[i, j]
                    pm_kl = pair_mappings[k, l]
                    if pm_ij.size == 0 or pm_kl.size == 0:
                        continue
                    w = float(np.prod([_nf(int(uniq_l[x]))
                                       for x in (i, j, k, l)]))
                    u, p = _count(pm_ij, pm_kl, nbas, q_cond, union, per,
                                  cutoff, a.chunk)
                    tot_u += u
                    tot_p += p
                    cost_u += w * u
                    cost_p += w * p
    row = dict(tag=tag, n_dm=n_dm, hermi=hermi,
               quartets_union=int(tot_u), quartet_dm_pairs=int(tot_p),
               frac=(tot_p / (n_dm * tot_u)) if tot_u else float('nan'),
               frac_cost=(cost_p / (n_dm * cost_u)) if cost_u else
               float('nan'))
    ROWS.append(row)
    print(f'  {tag:22s} n_dm={n_dm:4d} hermi={hermi}  '
          f'union quartets {tot_u:12d}  (quartet,dm) kept {tot_p:14d}  '
          f'fraction {row["frac"]:8.4f}  cost-weighted {row["frac_cost"]:8.4f}',
          flush=True)


def _count(pm_ij, pm_kl, nbas, q_cond, union, per, cutoff, chunk):
    """(union survivors, (quartet, density) survivors) for one class."""
    ij = cp.asarray(pm_ij, dtype=cp.int64)
    kl = cp.asarray(pm_kl, dtype=cp.int64)
    q_kl = q_cond.ravel()[kl]
    ksh, lsh = kl // nbas, kl % nbas
    n_u = 0
    n_p = 0
    for lo in range(0, ij.size, chunk):
        b = ij[lo:lo + chunk]
        ish, jsh = b // nbas, b % nbas
        # the task filter: q_ij + q_kl + max(dm_cond over the four K blocks)
        d_cut = cutoff - q_cond.ravel()[b][:, None] - q_kl[None, :]
        keep = b[:, None] >= kl[None, :]             # the filter's own tril
        # union
        m = cp.maximum(
            cp.maximum(union[ish[:, None], ksh[None, :]],
                       union[jsh[:, None], ksh[None, :]]),
            cp.maximum(union[ish[:, None], lsh[None, :]],
                       union[jsh[:, None], lsh[None, :]]))
        surv = keep & (m > d_cut)
        n_u += int(surv.sum())
        if not bool(surv.any()):
            continue
        # per density, only where the union kept the quartet
        bi, bk = cp.nonzero(surv)
        i_, j_ = ish[bi], jsh[bi]
        k_, l_ = ksh[bk], lsh[bk]
        dc = d_cut[bi, bk]
        mp = cp.maximum(
            cp.maximum(per[:, i_, k_], per[:, j_, k_]),
            cp.maximum(per[:, i_, l_], per[:, j_, l_]))
        n_p += int((mp > dc[None, :]).sum())
    return n_u, n_p


def capture(opt_cls=JK._VHFOpt):
    """Wrap _VHFOpt.get_k so the first few calls are analysed."""
    orig = opt_cls.get_k
    state = {'n': 0}

    def wrapped(self, dms, hermi, verbose=None, *rsh):
        state['n'] += 1
        if state['n'] <= a.max_calls:
            d = dms() if callable(dms) else dms
            _sync()
            analyse(self, cp.asarray(d), hermi, f'call {state["n"]}')
        return orig(self, dms, hermi, verbose, *rsh)

    opt_cls.get_k = wrapped
    return orig


mol = build_mol(a)
tag = f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.xc}/{a.stage}'
print(f'# {tag}  natm={mol.natm} nao={mol.nao}', flush=True)

mf = dft.RKS(mol, xc=a.xc)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
mf.conv_tol_cpscf = a.conv_tol_cpscf
e = mf.kernel()
print(f'# SCF {e:.9f} converged={mf.converged}', flush=True)

capture()
if a.stage == 'hessian':
    h = mf.Hessian()
    h.auxbasis_response = 0
    h.kernel()
elif a.stage == 'cphf':
    from gpu4pyscf.properties import polarizability
    polarizability.eval_polarizability(mf)
else:
    td = mf.TDA()
    td.nstates = a.nstates
    td.kernel()

print('\n' + json.dumps({'tag': tag, 'natm': mol.natm, 'nao': int(mol.nao),
                         'rows': ROWS}))
