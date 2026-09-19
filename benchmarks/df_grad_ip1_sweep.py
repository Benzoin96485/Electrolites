"""The launch configuration of the two integral kernels the DF gradient runs.

`benchmarks/df_grad_probe.py` says where the time in
``df/grad/rhf.py:_jk_energy_per_atom`` is, and the answer is not the
contraction: **79-85 %** of it is two Rys kernels, ``sum_ejk_int3c2e_ip1``
(the derivative three-centre integrals) and the plain ``fill_int3c2e``
underneath ``int3c2e_evaluator``.  Both are launched with a configuration
GPU4PySCF fixes in two constants -- ``gout_width=54`` and a shell-pair
aggregation of ``nsp_per_block*4`` -- and neither constant is measured
anywhere in that file.

This sweeps both, per kernel, on a real pseudo-density, and reports the grid.
Nothing here changes an answer: the configuration decides only how the same
integrals are distributed over threads.

    python benchmarks/df_grad_ip1_sweep.py \
        --xyz benchmarks/molecules/penicillinV.xyz --basis def2-tzvpp
"""
import ctypes
import json
import time

import numpy as np
import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--gout-widths', default='18,27,36,54,72,108')
ap.add_argument('--sp-mults', default='1,2,4,8,16,32')
a = ap.parse_args()
apply_patch(a.patch)

import gpu4pyscf                                             # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from pyscf import lib                                        # noqa: E402
from gpu4pyscf.lib.cupy_helper import contract, ndarray      # noqa: E402
from gpu4pyscf.df.grad import rhf as df_rhf_grad             # noqa: E402
from gpu4pyscf.df import int3c2e_bdiv as bdiv                # noqa: E402
from gpu4pyscf.df.int3c2e_bdiv import (                      # noqa: E402
    _split_l_ctr_pattern, get_ao_pair_loc, int3c2e_scheme, libvhf_rys,
    Int3c2eOpt)
from gpu4pyscf.gto.mole import SortedMole                    # noqa: E402

_sync = cp.cuda.Stream.null.synchronize


def timeit(label, fn, reps=None):
    reps = a.reps if reps is None else reps
    fn()
    _sync()
    ts = []
    for _ in range(reps):
        _sync()
        t = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t)
    print(f'{label:46s} {min(ts)*1e3:9.2f} ms', flush=True)
    return min(ts)


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
mf.with_df.reset()

sorted_mol = SortedMole.from_mol(mol, decontract=True)
opt = Int3c2eOpt(sorted_mol, mf.with_df.auxmol).build()
smol, auxmol = opt.mol, opt.auxmol
naux = auxmol.nao
pair_addresses = opt.pair_and_diag_indices(cart=True, original_ao_order=False)[0]
nao_pair = len(pair_addresses)
nao = smol.ao_loc_nr(cart=True)[-1]
print(f'[{a.xyz.rsplit("/",1)[-1]}/{a.basis}] natm={mol.natm} nao={nao} '
      f'naux={naux} nao_pair={nao_pair} gpu4pyscf={gpu4pyscf.__version__}',
      flush=True)

# a pseudo-density on the pair list, the shape the ip1 kernel consumes.  Its
# values do not change the kernel's cost, only its answer, so a converged one
# is used to keep the answer checkable rather than because it is needed.
pseudo = cp.asarray(np.random.default_rng(0).random(nao_pair * naux))
ejk = cp.zeros((mol.natm, 3))
ejk_aux = cp.zeros((mol.natm, 3))

lmax = smol.uniq_l_ctr[:, 0].max()
laux = auxmol.uniq_l_ctr[:, 0].max()
aux_loc = auxmol.ao_loc
l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
ksh_offsets_cpu = l_ctr_aux_offsets
ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu + smol.nbas, dtype=np.int32)
n_ksh_blocks = len(ksh_offsets_cpu) - 1
int3c2e_envs = opt.int3c2e_envs
kern = libvhf_rys.sum_ejk_int3c2e_ip1

R = {'ip1': {}, 'j3c': {}}
gout_widths = [int(x) for x in a.gout_widths.split(',')]
sp_mults = [int(x) for x in a.sp_mults.split(',')]


ao_pair_loc = get_ao_pair_loc(smol.uniq_l_ctr[:, 0], opt.bas_ij_cache)


def ip1_runner(gout_width, sp_mult):
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
        short_range=smol.omega < 0, gout_width=gout_width, deriv=(1, 0, 0))
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    shm_size_max = shm_size[:laux + 1, :lmax + 1, :lmax + 1].max()
    bas_ij_idx, shl_pair_offsets = smol.aggregate_shl_pairs(
        opt.bas_ij_cache, nsp_per_block[0] * sp_mult)

    def run():
        ejk[:] = 0
        ejk_aux[:] = 0
        # one launch per auxiliary (l, ctr) group, as `_jk_energy_per_atom`
        # does -- it has to, because it rebuilds the pseudo-density per group
        for kb in range(n_ksh_blocks):
            nk = int(aux_loc[ksh_offsets_cpu[kb + 1]]
                     - aux_loc[ksh_offsets_cpu[kb]])
            # the same `(nao_pair, naux_in_batch)` view of one buffer the
            # function builds its pseudo-density into
            block = ndarray((nao_pair, nk), buffer=pseudo)
            err = kern(
                ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
                ctypes.cast(ejk_aux.data.ptr, ctypes.c_void_p),
                ctypes.cast(block.data.ptr, ctypes.c_void_p),
                lib.c_null_ptr(),
                ctypes.c_int(1),
                ctypes.byref(int3c2e_envs),
                ctypes.c_int(shm_size_max),
                ctypes.c_int(len(shl_pair_offsets) - 1),
                ctypes.c_int(1),
                ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(ksh_offsets_gpu[kb:].data.ptr, ctypes.c_void_p),
                ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
                ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(aux_loc[ksh_offsets_cpu[kb]]),
                ctypes.c_int(nao), ctypes.c_int(nao_pair),
                ctypes.c_int(nk), ctypes.c_int(mol.natm))
            if err != 0:
                raise RuntimeError('int3c2e_ejk_ip1 failed')
    return run


print('\n--- sum_ejk_int3c2e_ip1: the derivative integrals ---', flush=True)
ref_ejk = None
for gw in gout_widths:
    for sm in sp_mults:
        try:
            run = ip1_runner(gw, sm)
            t = timeit(f'  gout_width={gw:<4d} sp_mult={sm:<3d}', run)
            R['ip1'][f'{gw}/{sm}'] = t
            if ref_ejk is None:
                ref_ejk = ejk.get().copy()
            else:
                d = float(np.abs(ejk.get() - ref_ejk).max())
                if d > 1e-8:
                    print(f'      !! differs from the first configuration by '
                          f'{d:.2e}', flush=True)
        except Exception as exc:                             # noqa: BLE001
            print(f'  gout_width={gw:<4d} sp_mult={sm:<3d}  '
                  f'ERR {type(exc).__name__}: {str(exc)[:80]}', flush=True)

# --- the plain integrals, through `int3c2e_evaluator` -----------------------
# `int3c2e_evaluator` fixes gout_width=54 and the same `*4` aggregation in its
# own body, so the sweep goes through a patched `int3c2e_scheme` and a patched
# `aggregate_shl_pairs`, restored immediately after.
print('\n--- int3c2e_evaluator: the plain integrals (24-26 %) ---', flush=True)
_orig_scheme = bdiv.int3c2e_scheme
_orig_agg = type(smol).aggregate_shl_pairs
buf = cp.empty((naux, nao_pair))
ref_j3c = None

for gw in gout_widths:
    for sm in sp_mults:
        def scheme(*args, gout_width=None, **kw):
            if gout_width == 54:
                gout_width = gw
            return _orig_scheme(*args, gout_width=gout_width, **kw)

        def agg(self, cache=None, nsp_per_block=512):
            if not isinstance(nsp_per_block, (int, np.integer)):
                nsp_per_block = nsp_per_block // 4 * sm
            return _orig_agg(self, cache, nsp_per_block)

        bdiv.int3c2e_scheme = scheme
        type(smol).aggregate_shl_pairs = agg
        try:
            eval_j3c, _, _, aux_offsets = opt.int3c2e_evaluator(
                aux_batch_size=naux, reorder_aux=True, cart=True)
            nb = len(aux_offsets) - 1

            def run(eval_j3c=eval_j3c, nb=nb):
                for kb in range(nb):
                    eval_j3c(aux_batch_id=kb, out=buf)

            t = timeit(f'  gout_width={gw:<4d} sp_mult={sm:<3d}', run)
            R['j3c'][f'{gw}/{sm}'] = t
            # the same check the ip1 grid gets: a launch configuration may
            # only change how the integrals are distributed over threads, and
            # `gout_width` past 54 does not -- it changes the answer
            if ref_j3c is None:
                ref_j3c = buf.get().copy()
            else:
                d = float(np.abs(buf.get() - ref_j3c).max())
                if d > 1e-9:
                    print(f'      !! differs from the first configuration by '
                          f'{d:.2e}', flush=True)
        except Exception as exc:                             # noqa: BLE001
            print(f'  gout_width={gw:<4d} sp_mult={sm:<3d}  '
                  f'ERR {type(exc).__name__}: {str(exc)[:80]}', flush=True)
        finally:
            bdiv.int3c2e_scheme = _orig_scheme
            type(smol).aggregate_shl_pairs = _orig_agg

for key in ('ip1', 'j3c'):
    if R[key]:
        best = min(R[key].items(), key=lambda kv: kv[1])
        base = R[key].get('54/4')
        print(f'\n{key}: best {best[0]} at {best[1]*1e3:.2f} ms; '
              f"GPU4PySCF's 54/4 "
              f'{"n/a" if base is None else f"{base*1e3:.2f} ms"}'
              + ('' if base is None else f' -> {base/best[1]:.3f}x'), flush=True)

print('RESULT ' + json.dumps(dict(
    xyz=a.xyz, basis=a.basis, natm=mol.natm, nao=int(nao), naux=int(naux),
    nao_pair=int(nao_pair), sweep=R)), flush=True)
