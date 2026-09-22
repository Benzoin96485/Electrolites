"""What the Krylov subspace of a DF Hessian actually costs, and when.

``docs/DF_WORK_LIST.md`` **W8** flags one line of
``df/hessian/rhf.py:_get_jk`` (``:1258-1273``): a device-resident cderi is
moved to the host whenever it is more than **0.4** of the free memory, with
the comment *"Leave space for storing Krylov subspace vectors"*.  Nothing in
the source says what that space is, and the consequence is not small -- on
the host the tensor crosses PCIe once per Krylov cycle, and
``electrolites/fastdf.py``'s partial-residency item is switched *off* for a
DF Hessian on the strength of that comment alone.

This measures the quantity the 0.4 is standing in for.  It runs a whole
density-fitted Hessian with a sampler on the allocator and on
``cudaMemGetInfo``, reports

  * the peak device memory the run reaches, and where in the run it is;
  * how many Krylov cycles the CPHF actually takes, which is what the
    subspace's size is linear in;
  * the subspace's own high-water mark, taken across the ``_get_jk`` calls
    that bracket it;
  * what upstream's rule and ``fastdfhess``'s reserve-aware one would each
    decide at this size, and at the sizes the rest of the round measures.

    python benchmarks/df_hess_krylov_probe.py \\
        --xyz benchmarks/molecules/penicillinV.xyz --basis def2-svp
"""
import json
import time

import numpy as np
import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--conv-tol-cpscf', type=float, default=1e-6)
a = ap.parse_args()
apply_patch(a.patch)

from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.lib import cupy_helper                        # noqa: E402

import electrolites                                          # noqa: E402
from electrolites import fastdfhess                          # noqa: E402

GB = 1024.0 ** 3
_sync = cp.cuda.Stream.null.synchronize
pool = cp.get_default_memory_pool()

marks = []


def mark(tag):
    _sync()
    marks.append({'tag': tag, 't': time.perf_counter(),
                  'used_gb': pool.used_bytes() / GB,
                  'pool_gb': pool.total_bytes() / GB,
                  'free_gb': cp.cuda.runtime.memGetInfo()[0] / GB})
    return marks[-1]


# count the Krylov cycles and sample memory at every one of them
cycles = {'n': 0}
_orig_krylov = cupy_helper.krylov


def krylov(aop, b, *args, **kw):
    def counting(x):
        cycles['n'] += 1
        mark(f'krylov cycle {cycles["n"]}')
        return aop(x)
    return _orig_krylov(counting, b, *args, **kw)


cupy_helper.krylov = krylov
import gpu4pyscf.hessian.rhf as HRHF                         # noqa: E402
HRHF.krylov = krylov

mol = build_mol(a)
mf = dft.RKS(mol, xc=a.xc).density_fit(auxbasis=a.auxbasis)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
mf.conv_tol_cpscf = a.conv_tol_cpscf
mf.verbose = 0
mark('start')
t0 = time.perf_counter()
e = mf.kernel()
mark('after SCF')
print(f'# {mol.natm} atoms  nao {mol.nao}  {a.basis}  {a.xc}')
print(f'# DF-SCF {time.perf_counter() - t0:.1f} s   E = {e:.10f}', flush=True)

nao = mol.nao
nocc = int((cp.asarray(mf.mo_occ) > 0).sum())
nmo = nao
naux = mf.with_df._cderi[cp.cuda.Device().id].shape[0]
cderi_gb = mf.with_df._cderi[cp.cuda.Device().id].nbytes / GB
print(f'# naux {naux}  nocc {nocc}  cderi {cderi_gb:.2f} GB')

hess = mf.Hessian()
hess.auxbasis_response = 2
t0 = time.perf_counter()
h = hess.kernel()
mark('after Hessian')
t_h = time.perf_counter() - t0
print(f'# whole Hessian {t_h:.1f} s   trace '
      f'{float(np.trace(cp.asnumpy(cp.asarray(h)).reshape(3 * mol.natm, -1))):.10f}')

n_dm = 3 * mol.natm
base = min(m['free_gb'] for m in marks[:2])
peak = max(marks, key=lambda m: m['pool_gb'])
low = min(marks, key=lambda m: m['free_gb'])
print(f'\n## memory')
print(f'  Krylov cycles                {cycles["n"]}')
print(f'  peak pool                    {peak["pool_gb"]:7.2f} GB  '
      f'at "{peak["tag"]}"')
print(f'  lowest free (memGetInfo)     {low["free_gb"]:7.2f} GB  '
      f'at "{low["tag"]}"')
print(f'  cderi                        {cderi_gb:7.2f} GB')
print(f'  one Krylov vector pair       '
      f'{2 * n_dm * nmo * nocc * 8 / GB:7.3f} GB '
      f'(n_dm {n_dm}, nmo {nmo}, nocc {nocc})')
print(f'  x {cycles["n"]} cycles           '
      f'{cycles["n"] * 2 * n_dm * nmo * nocc * 8 / GB:7.2f} GB')
print(f'  fastdfhess reserve           '
      f'{fastdfhess._krylov_reserve(n_dm, nmo, nocc) / GB:7.2f} GB '
      f'(KRYLOV_CYCLES {fastdfhess.KRYLOV_CYCLES}, factor 4)')
print(f'  stats                        {fastdfhess.stats()}')

print('\n@@JSON ' + json.dumps({
    'atoms': mol.natm, 'nao': int(nao), 'nocc': nocc, 'naux': int(naux),
    'n_dm': n_dm, 'cderi_gb': cderi_gb, 'cycles': cycles['n'],
    'peak_pool_gb': peak['pool_gb'], 'peak_tag': peak['tag'],
    'min_free_gb': low['free_gb'], 't_hessian': t_h,
    'reserve_gb': fastdfhess._krylov_reserve(n_dm, nmo, nocc) / GB,
    'marks': marks}))
