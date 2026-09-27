"""How much of W19's two kernels a screen *inside* them could skip.

``docs/DF_WORK_LIST.md`` **W26**, found by W20.  W20's screen
(`benchmarks/df_j3c_screen_probe.py`) keeps or drops a shell pair for every
auxiliary shell at once, so its density-side bound carries ``max_P q_P`` and
its auxiliary-side bound ``sum_P q_P |dc_P|`` over *every* auxiliary shell --
which is why the auxiliary-vector kernel kept 87-92 % of its work.  W19's
kernels walk a warp of 32 consecutive pairs against a batch of auxiliary
shells, so a test at that granularity can use the tight factor on both sides:

* density contraction: skip (warp, batch) when
  ``max_warp(q_ij max|dD_ij|) * max_batch q_P < tau``;
* auxiliary-vector contraction: skip (warp, batch) when
  ``max_warp q_ij * sum_batch q_P max|dc_P| < tau``.

A warp is 32 *consecutive* pairs in the plan's order.  GPU4PySCF sorts each
angular-momentum group by Schwarz factor, so consecutive pairs are anywhere in
the molecule and a warp's ``max|dD|`` is close to the global one; ``--orders``
prices the same bounds with the pairs of each group re-sorted along a Morton
curve through their product centres, which is a permutation the kernels do
not care about (they index pairs by position through gathered tables).

Every fraction is of the kernel's modelled work: a (warp, batch) weighs its
pair count times its shell count times ``nf3(L) + nf3(lij) * nk`` -- the
recurrence plus the contraction.  One converged DF SCF, the density every
cycle hands ``get_j``, nothing timed.

    python benchmarks/df_j3c_blockscreen_probe.py \\
        --xyz benchmarks/molecules/taxol.xyz --basis def2-svp --xc pbe
"""
import json
import time

import numpy as np
import cupy as cp

from _common import build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--taus', default='1e-10,1e-11,1e-12,1e-13')
ap.add_argument('--orders', default='stock,morton')
ap.add_argument('--save', default='', help='write the per-cycle table as JSON')
ap.add_argument('--time', type=int, default=0,
                help='also time both contractions of every difference, '
                     'unscreened and screened inside the kernels at each tau, '
                     'min of this many round-robin repetitions (whole node '
                     'only: docs/DF_WORK_LIST.md §5)')
a = ap.parse_args()

from gpu4pyscf import dft                                      # noqa: E402
from gpu4pyscf.df import df_jk as DFJK                         # noqa: E402

taus = [float(x) for x in a.taus.split(',')]
orders = a.orders.split(',')
mol = build_mol(a)
mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc).density_fit(
    auxbasis=a.auxbasis)
mf.conv_tol = a.conv_tol
mf.grids.level = a.grids
mf.verbose = 0

dms = []
_orig_get_j = DFJK._DFHF.get_j


def get_j(self, mol=None, dm=None, hermi=1, *args, **kw):
    d = cp.asarray(dm)
    dms.append((d[0] + d[1]) if d.ndim == 3 else d.copy())
    return _orig_get_j(self, mol, dm, hermi, *args, **kw)


DFJK._DFHF.get_j = get_j
t0 = time.perf_counter()
mf.kernel()
DFJK._DFHF.get_j = _orig_get_j
t_scf = time.perf_counter() - t0
dfobj = mf.with_df
opt = dfobj.j_engine
mols, auxs = opt.mol, opt.auxmol
print(f'# {a.xyz.split("/")[-1]} {a.basis} {a.xc}  natm {mol.natm}  nao '
      f'{mol.nao}  naux {dfobj.auxmol.nao}  sorted: {mols.nbas} shells, '
      f'{auxs.nbas} auxiliary shells')
print(f'# SCF {t_scf:.1f} s, {mf.cycles} cycles, E = {mf.e_tot:.10f}, '
      f'{len(dms)} Coulomb builds')

from electrolites import _j3c, _j3cinc, _j3ck                  # noqa: E402
from electrolites.codegen import gen_j3c                       # noqa: E402
from electrolites.codegen.gen_j_high import nf3                # noqa: E402
c = _j3c.constants(opt)
scr = _j3cinc.screen(opt, c)
kp = _j3ck.plan(opt, c)
q_pair, qP = scr.q_pair, scr.q_aux
qP_host = cp.asnumpy(qP)
print(f'# stored pairs {c.npair} ({kp.pairs_ours} on W19\'s kernels), '
      f'max q_ij {scr.q_pair_max:.3g}, max q_P {scr.q_aux_max:.3g}')


def morton(xyz):
    """A 3 x 10-bit Morton key of each point."""
    lo, hi = xyz.min(0), xyz.max(0)
    g = ((xyz - lo) / max(float((hi - lo).max()), 1e-12) * 1023).astype(
        np.int64)
    key = np.zeros(len(xyz), dtype=np.int64)
    for b in range(10):
        for d in range(3):
            key |= ((g[:, d] >> b) & 1) << (3 * b + d)
    return key


centres = cp.asnumpy(kp.pair_data[:, :3])
perms = {}
for o in orders:
    perm = np.arange(c.npair)
    if o == 'morton':
        for _, lo, hi in kp.ours:
            perm[lo:hi] = lo + np.argsort(morton(centres[lo:hi]),
                                          kind='stable')
    perms[o] = cp.asarray(perm)


def warp_max(v, lo, hi):
    """``max`` of ``v[lo:hi]`` over 32-aligned chunks from ``lo``, and the
    pair count of each chunk."""
    n = hi - lo
    pad = (-n) % 32
    seg = cp.concatenate([v[lo:hi], cp.zeros(pad)]) if pad else v[lo:hi]
    cnt = cp.full(len(seg) // 32, 32.)
    cnt[-1] -= pad
    return seg.reshape(-1, 32).max(1), cnt


# the auxiliary batches of each class, and the model cost of a quartet
batches = {}
for kind in ('dm', 'aux'):
    for lij, lk in kp.classes:
        kb = kp.kb_of(kind, lij, lk)
        for alk, k0, k1 in kp.aux_groups:
            if alk == lk:
                ks = np.arange(k0, k1, kb)
                batches[kind, lij, lk] = (ks, np.minimum(ks + kb, k1))
cost = {(lij, lk): nf3(lij + lk) + nf3(lij) * len(gen_j3c.parity(lk))
        for lij, lk in kp.classes}


def fractions(w_dm, e_aux, perm):
    """Kept fraction of each kernel's modelled work, per tau."""
    wd = w_dm[perm]
    qa = q_pair[perm]
    ce = cp.concatenate([cp.zeros(1), cp.cumsum(e_aux)])
    tot = {'dm': 0., 'aux': 0., 'aux1': 0.}
    kept = {(k, t): 0. for k in tot for t in taus}
    for lij, lo, hi in kp.ours:
        wm_d, cnt = warp_max(wd, lo, hi)
        wm_a, _ = warp_max(qa, lo, hi)
        for (lij2, lk) in kp.classes:
            if lij2 != lij:
                continue
            for kind, wm in (('dm', wm_d), ('aux', wm_a)):
                k0, k1 = batches[kind, lij, lk]
                nk = cp.asarray((k1 - k0).astype(np.float64))
                if kind == 'dm':
                    bb = cp.asarray(np.maximum.reduceat(
                        qP_host[k0[0]:k1[-1]], k0 - k0[0]))
                else:
                    bb = ce[cp.asarray(k1)] - ce[cp.asarray(k0)]
                work = cnt[:, None] * nk[None, :] * cost[lij, lk]
                tot[kind] += float(work.sum())
                b = wm[:, None] * bb[None, :]
                for t in taus:
                    kept[kind, t] += float((work * (b >= t)).sum())
                if kind == 'aux':
                    # the same kernel tested per (warp, auxiliary shell)
                    e = e_aux[int(k0[0]):int(k1[-1])]
                    w1 = cnt[:, None] * cost[lij, lk]
                    tot['aux1'] += float(w1.sum()) * len(e)
                    b1 = wm[:, None] * e[None, :]
                    for t in taus:
                        kept['aux1', t] += float((w1 * (b1 >= t)).sum())
    return {t: {k: kept[k, t] / tot[k] for k in tot} for t in taus}


j2c, tag = dfobj._cd_j2c


def coeffs(d):
    ds = mols.apply_C_mat_CT(d)
    rho = auxs.apply_CT_dot(opt.contract_dm(ds, 1), axis=-1)
    if tag == 'cd':
        x = DFJK.solve_triangular(j2c, rho[:, None], lower=True)
        x = DFJK.solve_triangular(j2c.T, x, lower=False)[:, 0]
    else:
        x = cp.linalg.solve(j2c, rho)
    return auxs.apply_C_dot(x, axis=-1), ds


# W20's per-pair screen, as a reference column, on the same work model
lsh = cp.asarray(np.asarray(mols._bas[:, 1]))
lij_p = lsh[scr.ish] + lsh[scr.jsh]
w_pair = ((lij_p + 1) * (lij_p + 2) * (lij_p + 3) // 6).astype(np.float64)

if a.time:
    # imported only now: importing it binds its replacements, and the SCF
    # above is GPU4PySCF's own
    from electrolites import fastdfj3c                        # noqa: E402
    _sync = cp.cuda.Stream.null.synchronize

    def _t(fn):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        return time.perf_counter() - t0

    def timings(dds, dcs):
        """ms of each contraction of the difference: unscreened, and with
        W26's screen at each tau (its two weight launches included)."""
        fns = {'none': (lambda: fastdfj3c._contract_dm_screened(
                            c, kp, dds, None),
                        lambda: fastdfj3c._contract_auxvec_screened(
                            c, kp, dcs, None))}
        for t in taus:
            fns[t] = ((lambda t=t: fastdfj3c._contract_dm_screened(
                           c, kp, dds, (scr.pair_weights(dds), qP, t))),
                      (lambda t=t: fastdfj3c._contract_auxvec_screened(
                           c, kp, dcs, (q_pair, scr.aux_weights(dcs), t))))
        best = {k: [1e9, 1e9] for k in fns}
        for f in fns.values():
            f[0](), f[1]()
        for _ in range(a.time):
            for k, f in fns.items():
                best[k][0] = min(best[k][0], _t(f[0]))
                best[k][1] = min(best[k][1], _t(f[1]))
        return {k: (v[0] * 1e3, v[1] * 1e3) for k, v in best.items()}

rows = []
prev = None
hdr = '  '.join(f'{"tau=%.0e" % t:>38s}' for t in taus)
print(f'\n{"cycle":>5s} {"order":>7s}  {hdr}')
print(f'{"":>5s} {"":>7s}  ' + '  '.join(
    f'{"W20 dm/aux   warp dm/aux/aux1":>38s}' for _ in taus))
for n, d in enumerate(dms):
    cs, ds = coeffs(d)
    if prev is not None:
        dds, dcs = ds - prev[1], cs - prev[0]
        w_dm = q_pair * scr.pair_absmax(dds)
        e_aux = qP * scr.aux_absmax(dcs)
        s_aux = float(e_aux.sum())
        w20 = {t: (float((w_pair * (w_dm * scr.q_aux_max >= t)).sum())
                   / float(w_pair.sum()),
                   float((w_pair * (q_pair * s_aux >= t)).sum())
                   / float(w_pair.sum())) for t in taus}
        row = {'cycle': n, 'max_dD': float(cp.abs(dds).max()),
               'max_dc': float(cp.abs(dcs).max()), 'w20': w20, 'warp': {}}
        for o in orders:
            f = fractions(w_dm, e_aux, perms[o])
            row['warp'][o] = f
            cells = [f'{w20[t][0]:6.3f} {w20[t][1]:6.3f}   '
                     f'{f[t]["dm"]:6.3f} {f[t]["aux"]:6.3f} {f[t]["aux1"]:6.3f}'
                     for t in taus]
            print(f'{n:5d} {o:>7s}  ' + '  '.join(f'{x:>38s}' for x in cells),
                  flush=True)
        if a.time:
            tm = timings(dds, dcs)
            row['ms'] = {str(k): v for k, v in tm.items()}
            print(f'{n:5d} {"ms":>7s}  unscreened {tm["none"][0]:7.3f} '
                  f'{tm["none"][1]:7.3f}   ' + '  '.join(
                      f'{t:.0e}: {tm[t][0]:7.3f} {tm[t][1]:7.3f}'
                      for t in taus), flush=True)
        rows.append(row)
    prev = (cs, ds)

if a.time and rows:
    tot = {k: [sum(r['ms'][k][i] for r in rows) for i in (0, 1)]
           for k in rows[0]['ms']}
    base = sum(tot['none'])
    print(f'\n# both contractions summed over {len(rows)} differences: '
          f'unscreened {base:.1f} ms; ' + ', '.join(
              f'tau {k}: {sum(v):.1f} ms ({sum(v) / base:.3f})'
              for k, v in tot.items() if k != 'none'))

out = {'xyz': a.xyz, 'basis': a.basis, 'xc': a.xc, 'natm': mol.natm,
       'cycles': mf.cycles, 'rows': rows}
if a.save:
    with open(a.save, 'w') as fh:
        json.dump(out, fh, default=str)
