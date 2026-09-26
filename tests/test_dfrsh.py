"""Correctness of a range-separated DF run's tensors.

``docs/DF_WORK_LIST.md`` **W22**, ``electrolites/fastdfrsh.py`` and the host
budget in ``electrolites/fastdfcderi.py``/``electrolites/_dfpin.py``.

* **The long-range-only path is upstream's energy.**  For ``hyb = 0`` the
  module builds only the long-range tensor and takes J from the three-centre
  engine; nothing in that is a new approximation, so the SCF energy is held to
  the bar every other exact module here is, against the same process with the
  module switched off -- restricted and unrestricted, three functionals.
* **The fused tensor is the fit of the fused operator.**  ``FASTDFRSH=fuse``
  is a different approximation, so it cannot be held to upstream's number.
  What it can be held to is its own definition: the exchange matrix it gives
  is ``B_g^T J_g^{-1} B_g D`` for ``g = a/r + b erf(w r)/r``, which PySCF's
  CPU integrals can build independently.  Checked element-wise on both
  residencies, and the SCF energy is reported against upstream's.
* **The host budget.**  The note is recorded for every host-resident tensor,
  the live count comes back to zero when the tensors go, a tensor that
  cannot fit raises ``MemoryError`` with its size instead of being built, and
  a pooled buffer is evicted to make room for a build that needs it.

    python tests/test_dfrsh.py
    python tests/test_dfrsh.py --basis def2-tzvp
"""
import argparse
import gc

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--etol', type=float, default=1e-9, help='SCF energy, Eh')
ap.add_argument('--ktol', type=float, default=1e-9,
                help='fused K against its CPU definition, relative to its '
                     'largest element')
a = ap.parse_args()

import cupy as cp                                             # noqa: E402
from pyscf import gto, df as pdf                              # noqa: E402
from pyscf.df import addons                                   # noqa: E402

import electrolites                                           # noqa: E402
electrolites.patch_all()
from electrolites import fastdfrsh as R                       # noqa: E402
from electrolites import fastdfcderi as FC                    # noqa: E402
from electrolites import _dfpin as PIN                        # noqa: E402
from gpu4pyscf import dft                                     # noqa: E402

# the file's own path: `gto.M` reads an .xyz file's count and comment lines
GLYCINE = __file__.rsplit('/', 2)[0] + '/benchmarks/molecules/glycine.xyz'
CH2OH = '''C 0.000 0.000 0.000; O 1.350 0.000 0.000; H 1.690 0.890 0.000
H -0.540 0.930 0.000; H -0.540 -0.930 0.000'''

fails = []


def check(label, err, tol):
    ok = err <= tol
    print(f'  {"ok  " if ok else "FAIL"} {label:58s} {err:.2e}  (tol {tol:.0e})',
          flush=True)
    if not ok:
        fails.append(label)


def scf(atom, xc, spin=0, mode='auto', use_gpu_memory=None):
    mol = gto.M(atom=atom, basis=a.basis, spin=spin, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc).density_fit()
    mf.conv_tol = 1e-11
    mf.with_df.use_gpu_memory = use_gpu_memory
    old, R._MODE = R._MODE, mode
    try:
        e = mf.kernel()
    finally:
        R._MODE = old
    assert mf.converged
    return e, mf


print('# 1. long-range-only functionals: the energy is upstream\'s')
for atom, name, spin in ((GLYCINE, 'glycine', 0), (CH2OH, 'CH2OH', 1)):
    for xc in ('hyb_gga_xc_wb97', 'hyb_gga_xc_lrc_wpbe', 'lc-vv10'):
        if spin and xc == 'lc-vv10':
            continue
        n0 = R.STATS['lr_only']
        e_ref, _ = scf(atom, xc, spin, mode='off')
        e, mf = scf(atom, xc, spin, mode='auto')
        assert R.STATS['lr_only'] > n0, 'the long-range-only path was not taken'
        # and it held one tensor, not two
        assert mf.with_df._cderi is None
        check(f'{name} {"UKS" if spin else "RKS"} {xc}', abs(e - e_ref), a.etol)

print('# 2. the fused tensor is the fit of the fused operator')
mol = gto.M(atom=GLYCINE, basis=a.basis, verbose=0)
auxmol = addons.make_auxmol(mol, 'def2-universal-jkfit')
rng = np.random.default_rng(7)
c = rng.standard_normal((mol.nao, 6)) * .1
dm = c @ c.T
hyb, alpha, omega = 0.195728, 1.0, 0.25
fa, fb = hyb, alpha - hyb


def cpu_fit(op_a, op_b, om):
    j2 = op_a * auxmol.intor('int2c2e')
    j3 = op_a * pdf.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s1')
    with mol.with_range_coulomb(om), auxmol.with_range_coulomb(om):
        j2 = j2 + op_b * auxmol.intor('int2c2e')
        j3 = j3 + op_b * pdf.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s1')
    low = np.linalg.cholesky(j2)
    B = np.linalg.solve(low, j3.reshape(-1, j3.shape[-1]).T).reshape(
        -1, mol.nao, mol.nao)
    return np.einsum('Pik,kl,Plj->ij', B, dm, B)


k_ref = cpu_fit(fa, fb, omega)
for where in (None, False):
    mf = dft.RKS(mol, xc='hyb_gga_xc_wb97x_d3').density_fit()
    mf.with_df.auxbasis = 'def2-universal-jkfit'
    mf.with_df.use_gpu_memory = where
    mf.with_df.build(build_cderi=False)
    fdf = R._fused_df(mf.with_df, omega, alpha, hyb)
    vk = cp.asnumpy(fdf.get_jk(cp.asarray(dm), 1, with_j=False)[1])
    label = 'device' if where is None else 'host'
    check(f'fused K, {label}-resident cderi', abs(vk - k_ref).max()
          / abs(k_ref).max(), a.ktol)
# and the long-range-only end of the same build is the stock long-range one
e_ref, _ = scf(GLYCINE, 'hyb_gga_xc_wb97x_d3', mode='off')
e_fuse, _ = scf(GLYCINE, 'hyb_gga_xc_wb97x_d3', mode='fuse')
print(f'  info fused wB97X-D3 energy against upstream\'s two tensors: '
      f'{e_fuse - e_ref:+.3e} Eh (a different fit; not a bar)')
if abs(e_fuse - e_ref) > 1e-5:
    fails.append('fused energy is not within 1e-5 of the two-tensor fit')

print('# 3. the host budget')
del mf, fdf
gc.collect()
live0 = PIN.live_bytes()
n0 = len(FC.HOST_BUILDS)
e, mf = scf(GLYCINE, 'hyb_gga_xc_wb97x_d3', mode='off', use_gpu_memory=False)
recs = FC.HOST_BUILDS[n0:]
assert [r['omega'] for r in recs] == [0.0, 0.25], recs
tensors = [mf.with_df._cderi[0]] + [d._cderi[0]
                                    for d in mf.with_df._rsh_df.values()]
check('live host bytes = the two tensors', abs(
    PIN.live_bytes() - live0 - sum(t.nbytes for t in tensors)), 4096 * 2)
print(f'  info notes: {[round(r["gb"], 4) for r in recs]} GB, the second '
      f'with {recs[1]["live_gb"]:.4f} GB already alive')
del tensors
mf.with_df.reset()
del mf
gc.collect()
check('live host bytes back to where they were', abs(
    PIN.live_bytes() - live0), 0)

# a tensor that cannot fit is refused with its size.  The pool is emptied
# first: it holds the two buffers just released, and a build of the same size
# would be handed them -- which is right, and is why the test for the refusal
# counts pooled bytes as available
PIN.free_pool()
_host_avail = PIN.host_avail
PIN.host_avail = lambda: 1 << 20
try:
    try:
        scf(GLYCINE, 'hyb_gga_xc_wb97x_d3', mode='off', use_gpu_memory=False)
        fails.append('no MemoryError with 1 MB of host memory')
        print('  FAIL no MemoryError')
    except MemoryError as exc:
        print(f'  ok   MemoryError: {str(exc)[:110]}...')
finally:
    PIN.host_avail = _host_avail

# a pooled buffer is evicted for a build that needs the room
buf = PIN.empty_mapped_pooled((1 << 17,))
del buf
gc.collect()
assert PIN._pool, 'the released buffer was not retained'
ev0 = PIN.STATS['evicted_for_room']
PIN.host_avail = lambda: PIN._limits()[1] // 2
try:
    PIN._make_room(1 << 20)
finally:
    PIN.host_avail = _host_avail
check('pooled buffer evicted for room', 0 if PIN.STATS['evicted_for_room']
      > ev0 else 1, 0)

print('RESULT', 'FAIL ' + ', '.join(fails) if fails else 'PASS')
raise SystemExit(1 if fails else 0)
