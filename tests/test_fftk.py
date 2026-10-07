"""FFTDF k-point exchange with time reversal against GPU4PySCF's (J3).

``electrolites/fastfftk.py`` builds ``K(k)`` for one ``k`` of each pair
``{k, -k}`` and takes ``K(-k) = conj(K(k))`` when the k-point set and the
densities allow it.  ``fft_jk.get_k_kpts`` against ``_get_k_kpts_orig`` on
the same density, element-wise: an odd mesh (where it applies), an even
Gamma-centred mesh (every point its own partner, so nothing is saved), a shifted mesh with no
inversion partners, a density without orbital tags, two spin densities, a
density that breaks time reversal (must fall back), and ``kpts_band``.

    python tests/test_fftk.py
"""
import sys

import numpy as np

import electrolites
electrolites.patch('fastfftk')
import cupy as cp                                            # noqa: E402
from pyscf.pbc import gto                                    # noqa: E402
from gpu4pyscf.pbc import dft                                # noqa: E402
from gpu4pyscf.pbc.df import fft_jk as F                     # noqa: E402
from electrolites import fastfftk                            # noqa: E402

fails = []
cell = gto.M(atom='C 0 0 0; C 0.8917 0.8917 0.95',
             a='0 1.7834 1.7834; 1.7834 0 1.7834; 1.7834 1.7834 0',
             basis='gth-szv', pseudo='gth-pade', verbose=0)


def check(tag, mydf, dm, kpts, kpts_band=None, expect_tr=None):
    s0 = fastfftk.stats()['time_reversal']
    ours = F.get_k_kpts(mydf, dm, 1, kpts, kpts_band, 'ewald')
    used = fastfftk.stats()['time_reversal'] > s0
    ref = F._get_k_kpts_orig(mydf, dm, 1, kpts, kpts_band, 'ewald')
    err = float(abs(cp.asarray(ours) - cp.asarray(ref)).max())
    ok = err < 1e-12 and (expect_tr is None or used == expect_tr)
    if not ok:
        fails.append(f'{tag}: {err:.3e} time reversal {used}')
    print(f'  {"ok  " if ok else "FAIL"} {tag:44s} {err:.3e}  '
          f'time reversal {"used" if used else "not used"}', flush=True)


for mesh, center, tr in (([3, 3, 3], None, True), ([2, 2, 2], None, None),
                         ([3, 3, 1], None, True),
                         ([2, 2, 2], [.1, .2, .3], False)):
    kw = {} if center is None else {'scaled_center': center}
    kpts = cell.make_kpts(mesh, **kw)
    mf = dft.KRKS(cell, kpts=kpts, xc='pbe')
    mf.max_cycle = 4
    mf.kernel()
    dm = mf.make_rdm1()
    tag = f'{mesh}' + (f' shifted {center}' if center else '')
    check(f'{tag} KRKS density', mf.with_df, dm, kpts, expect_tr=tr)
    check(f'{tag} untagged density', mf.with_df, cp.asarray(dm), kpts,
          expect_tr=tr)
    if tr:
        bad = cp.asarray(dm).copy()
        bad[1] += 1e-3 * cp.eye(cell.nao)        # dm(-k) != conj(dm(k))
        check(f'{tag} broken time reversal', mf.with_df, bad, kpts,
              expect_tr=False)
        check(f'{tag} kpts_band', mf.with_df, dm, kpts,
              kpts_band=kpts[:2], expect_tr=False)

kpts = cell.make_kpts([3, 3, 3])
mfu = dft.KUKS(cell.copy().set(spin=0), kpts=kpts, xc='pbe')
mfu.max_cycle = 4
mfu.kernel()
check('[3,3,3] KUKS (two densities)', mfu.with_df, mfu.make_rdm1(), kpts,
      expect_tr=True)
print('# fastfftk', fastfftk.stats())
if fails:
    print('\nFAILED:\n  ' + '\n  '.join(fails))
    sys.exit(1)
print('\nall ok')
