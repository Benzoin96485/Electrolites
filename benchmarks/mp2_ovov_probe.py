"""I3 probe: where conventional RMP2's ovov build spends its time.

Replicates gpu4pyscf.scf.int4c2e.get_int4c2e_ovov with CUDA-event timing of
the GINTfill_int2e calls against the 8 einsums, and reports the largest
class-pair block (the memory wall)."""
import sys, time, argparse
import numpy as np, cupy
ap = argparse.ArgumentParser()
ap.add_argument('--xyz', default='molecules/alanine_dipeptide.xyz')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--sizes-only', action='store_true')
ap.add_argument('--check', action='store_true', help='compare with the stock build')
ap.add_argument('--hog', type=float, default=0,
                help='GB held by a raw cudaMalloc, to emulate a smaller card')
a = ap.parse_args()
import electrolites; electrolites.patch_all()
from pyscf import gto
from gpu4pyscf import scf
from gpu4pyscf.scf import int4c2e
from gpu4pyscf.scf.int4c2e import _VHFOpt, _einsum

_hog = cupy.cuda.memory._malloc(int(a.hog * 1e9)) if a.hog else None
mol = gto.M(atom=a.xyz, basis=a.basis, verbose=0)
vhfopt = _VHFOpt(mol, 'int2e').build(1e-13)
nao = mol.nao
cart_ao_loc = mol.ao_loc_nr(cart=True)
cart_ao_loc = [cart_ao_loc[cp] for cp in vhfopt.l_ctr_offsets]
n = np.diff(cart_ao_loc)
n = [int(x) for x in n]
big = max(n[i]*n[j]*n[k]*n[l] for i in range(len(n)) for j in range(i+1)
          for k in range(len(n)) for l in range(k+1))
print('nao', nao, 'nclasses', len(n), 'class sizes', list(n),
      'largest block GB', big*8/1e9)
if a.sizes_only:
    sys.exit()
mf = scf.RHF(mol); mf.verbose = 0; mf.kernel()
nocc = mol.nelectron // 2
co = cupy.asarray(mf.mo_coeff[:, :nocc]); cv = cupy.asarray(mf.mo_coeff[:, nocc:])
from electrolites import fastmp2
print('nocc', nocc, 'nvir', cv.shape[1], 'ovov GB', nocc**2*cv.shape[1]**2*8/1e9)

orig = int4c2e.loop_int4c2e_general
tfill = []
def timed_loop(*args, **kw):
    it = orig(*args, **kw)
    while True:
        e0 = cupy.cuda.Event(); e0.record()
        try:
            item = next(it)
        except StopIteration:
            return
        e1 = cupy.cuda.Event(); e1.record()
        tfill.append((e0, e1))
        yield item
int4c2e.loop_int4c2e_general = timed_loop
for rep in range(2):
    tfill.clear()
    cupy.cuda.Device().synchronize(); t0 = time.perf_counter()
    ovov = int4c2e.get_int4c2e_ovov(mol, co, cv)
    cupy.cuda.Device().synchronize(); t1 = time.perf_counter()
    fill = sum(cupy.cuda.get_elapsed_time(x, y) for x, y in tfill) / 1e3
    print(fastmp2.STATS); print(f'rep {rep} {getattr(int4c2e.get_int4c2e_ovov, "__module__", "")}: ovov build {t1-t0:.2f} s, GINTfill {fill:.2f} s, '
          f'rest (einsums) {t1-t0-fill:.2f} s, blocks {len(tfill)}')
if a.check:
    from electrolites import fastmp2
    ref = int4c2e._get_int4c2e_ovov_orig(mol, co, cv)
    new = cupy.asarray(ovov).reshape(ref.shape)
    print('max |new - stock|', float(abs(new - ref).max()), 'max |stock|', float(abs(ref).max()))
    cupy.cuda.Device().synchronize(); t0 = time.perf_counter()
    int4c2e._get_int4c2e_ovov_orig(mol, co, cv)
    cupy.cuda.Device().synchronize(); print('stock warm', time.perf_counter() - t0)
if 0: np.save('/global/homes/w/weiliang/vibeqc/_el_jobs/i3/ref_%s.npy' % a.xyz.rsplit('/',1)[-1][:-4],
        cupy.asnumpy(ovov[:4]))
