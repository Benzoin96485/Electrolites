"""Anatomy of ``df/grad/rhf.py:_jk_energy_per_atom``, stage by stage.

`docs/DF_WORK_LIST.md` **W3** names this function as the one uncovered stage
of the density-fitting path that *grows* with basis size -- 29 % of a
def2-SVP DF gradient and 83 % of a def2-TZVPP one -- and its first instruction
is "profile it first, stage by stage, the way `benchmarks/df_probe.py` does
the SCF build", with §2's lesson attached: **do not assume the contraction is
the cost**.

This is that profile.  It runs a traced clone of the function -- the same
arithmetic in the same order, with a synchronising timer between stages -- and
reports the six stages it has:

  * **setup**, the ``SortedMole`` and the ``Int3c2eOpt.build()`` that
    ``jk_energy_per_atom`` throws away and rebuilds on every call;
  * **eval_j3c**, the plain three-centre integrals, evaluated once per
    auxiliary batch into the compressed ``(nao_pair, naux)`` pair list;
  * **the unpack**, two fancy-index scatters expanding that pair list into a
    dense ``(nao, nao, blk)`` block -- the stage W3 singles out, because
    unlike the SCF build's unpack this one is not a kernel;
  * **half-transform 1 and 2**, ``(pq|r) -> (iq|r) -> (ij|r)``;
  * **the metric solve** and the ``dm_oo`` contraction against it;
  * **the pseudo-density**, which runs the two half-transforms backwards into
    a *second* dense ``(nao, nao, blk)`` expansion and then compresses it back
    onto the pair list with ``cp.take``;
  * **the ip1 kernel**, ``sum_ejk_int3c2e_ip1``, the derivative integrals;
  * **int2c2e_ip1**, the auxiliary-basis response term.

Each stage is reported against its roofline where one exists: the two dense
expansions against memory bandwidth, the four contractions against fp64 peak.

    python benchmarks/df_grad_probe.py \
        --xyz benchmarks/molecules/penicillinV.xyz --basis def2-tzvpp

`--whole` adds the wall clock of a whole gradient around it, so the stage
table can be read as a share of something.
"""
import ctypes
import json
import time

import numpy as np
import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--whole', action='store_true',
                help='also time the whole gradient and its energy_ee')
ap.add_argument('--blk', type=int, default=0,
                help='override the auxiliary block size the traced clone uses')
a = ap.parse_args()
apply_patch(a.patch)

import gpu4pyscf                                             # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from pyscf import lib                                        # noqa: E402
from gpu4pyscf.lib.cupy_helper import contract, ndarray, get_avail_mem  # noqa: E402
from gpu4pyscf.df.grad import rhf as df_rhf_grad             # noqa: E402
from gpu4pyscf.df.int3c2e_bdiv import (                      # noqa: E402
    _split_l_ctr_pattern, get_ao_pair_loc, int3c2e_scheme, libvhf_rys,
    Int3c2eOpt, int2c2e)
from gpu4pyscf.gto.mole import SortedMole                    # noqa: E402

_sync = cp.cuda.Stream.null.synchronize
#: fp64 peak and HBM bandwidth of the A100 every number in `docs/` is measured
#: on, so a stage can be read against the machine and not only against itself.
PEAK_TFLOPS = 19.5
PEAK_GBS = 1555.0

T = {}
N = {}


def _acc(label, dt, flops=0., bytes_=0.):
    e = T.setdefault(label, {'t': 0., 'n': 0, 'flops': 0., 'bytes': 0.})
    e['t'] += dt
    e['n'] += 1
    e['flops'] += flops
    e['bytes'] += bytes_


class stage:
    """A synchronising timer that accumulates into `T` under one label."""

    def __init__(self, label, flops=0., bytes_=0.):
        self.label, self.flops, self.bytes = label, flops, bytes_

    def __enter__(self):
        _sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _sync()
        _acc(self.label, time.perf_counter() - self.t0, self.flops, self.bytes)
        return False


def traced_jk_energy_per_atom(int3c2e_opt, dm, j_factor=1., k_factor=1.,
                              hermi=1, auxbasis_response=True, blk_override=0):
    """`df/grad/rhf.py:_jk_energy_per_atom`, 1.8.1, with timers between stages.

    Copied rather than wrapped: the stages are inside one function there, and
    the point of this script is the split.  Its answer is compared against the
    stock function's below, which is what keeps the copy honest.
    """
    from gpu4pyscf.pbc.df.int2c2e import int2c2e_ip1_per_atom
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol

    with stage('factorize dm'):
        dm_factor_l, dm_factor_r = df_rhf_grad._factorize_dm(mol, dm, hermi)
        nao, nocc = dm_factor_l.shape

    with stage('pair addresses'):
        pair_addresses = int3c2e_opt.pair_and_diag_indices(
            cart=True, original_ao_order=False)[0]
        i_addr, j_addr = divmod(pair_addresses, nao)
        nao_pair = len(pair_addresses)
    naux = auxmol.nao

    mem_free = get_avail_mem(exclude_memory_pool=True)
    mem_avail = mem_free - naux * nocc**2 * 8 - nao**2 * 8
    batch_size = max(1, min(naux, int(mem_avail * .5 / (nao_pair * 8))))
    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, reorder_aux=True, cart=True)
    aux_batches = len(aux_offsets) - 1
    blksize = max(1, min(naux, int(mem_avail * .4 / (nao * nao * 2 * 8)) // 8 * 8))
    if blk_override:
        blksize = min(naux, blk_override)

    N.update(nao=int(nao), nocc=int(nocc), naux=int(naux),
             nao_pair=int(nao_pair), batch_size=int(batch_size),
             blksize=int(blksize), aux_batches=int(aux_batches),
             natm=int(mol.natm),
             fill=float(nao_pair / (nao * (nao + 1) / 2)))

    aux0 = aux1 = 0
    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((batch_size, nao_pair))
    buf1 = cp.empty((blksize, nocc, nao))
    j3c_oo = cp.empty((naux, nocc, nocc))
    for kbatch in range(aux_batches):
        with stage('eval_j3c (3c2e integrals)'):
            compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
        naux_in_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:, :, :dk]
            with stage('unpack -> dense (nao,nao,blk)',
                       bytes_=3. * float(nao_pair) * dk * 8):
                j3c[j_addr, i_addr] = j3c[i_addr, j_addr] = compressed[:, k0:k1]
            tmp = ndarray((nocc, nao, dk), buffer=buf1)
            with stage('half-transform 1  pqr,pi->iqr',
                       flops=2. * float(nao) * nao * nocc * dk,
                       bytes_=float(nao) * nao * dk * 8):
                contract('pqr,pi->iqr', j3c, dm_factor_r, out=tmp)
            with stage('half-transform 2  iqr,qj->rij',
                       flops=2. * float(nocc) * nao * nocc * dk):
                contract('iqr,qj->rij', tmp, dm_factor_l,
                         out=j3c_oo[aux0:aux1])
    j3c_full = buf = buf1 = eval_j3c = j3c = tmp = compressed = None

    with stage('metric solve (naux x naux)'):
        aux_coeff = cp.asarray(auxmol.ctr_coeff)
        aux_coeff, tmp = cp.empty_like(aux_coeff), aux_coeff
        aux_coeff[aux_sorting] = tmp
        tmp = None
        j2c = int2c2e(auxmol)
        if mol.omega <= 0 and not auxmol.mol.cart:
            metric = aux_coeff.dot(df_rhf_grad._gen_metric_solver(j2c, 'CD')(aux_coeff.T))
        else:
            metric = aux_coeff.dot(df_rhf_grad._gen_metric_solver(j2c, 'ED')(aux_coeff.T))
        j2c = aux_coeff = None
    with stage('dm_oo = metric . j3c_oo',
               flops=2. * float(naux) * naux * nocc * nocc):
        dm_oo = cp.einsum('uv,vij->uij', metric, j3c_oo)
        metric = j3c_oo = None
        if j_factor != 0:
            auxvec = dm_oo.trace(axis1=1, axis2=2)

    with stage('ip1 setup'):
        nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
            short_range=mol.omega < 0, gout_width=54, deriv=(1, 0, 0))
        gout_stride = cp.asarray(gout_stride, dtype=np.int32)
        lmax = mol.uniq_l_ctr[:, 0].max()
        laux = auxmol.uniq_l_ctr[:, 0].max()
        shm_size_max = shm_size[:laux + 1, :lmax + 1, :lmax + 1].max()
        bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
            int3c2e_opt.bas_ij_cache, nsp_per_block[0] * 4)
        ao_pair_loc = get_ao_pair_loc(mol.uniq_l_ctr[:, 0], int3c2e_opt.bas_ij_cache)
        aux_loc = auxmol.ao_loc
        l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
        l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
            l_ctr_aux_offsets, auxmol.uniq_l_ctr, batch_size)
        ksh_offsets_cpu = l_ctr_aux_offsets
        ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu + mol.nbas, dtype=np.int32)
        aux_offsets = aux_loc[ksh_offsets_cpu]
        aux_batches = len(aux_offsets) - 1
        if j_factor != 0:
            dm = dm_factor_l.dot(dm_factor_r.T)

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libvhf_rys.sum_ejk_int3c2e_ip1
    aux0 = aux1 = 0
    buf = cp.empty((nao_pair * batch_size))
    buf1 = cp.empty((blksize, nao, nao))
    buf2 = cp.empty((blksize, nao, nao))
    ejk = cp.zeros((mol.natm, 3))
    ejk_aux = cp.zeros((mol.natm, 3))
    for kbatch in range(aux_batches):
        naux_in_batch = aux_offsets[kbatch + 1] - aux_offsets[kbatch]
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
        compressed = ndarray((nao_pair, naux_in_batch), buffer=buf)
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            dm_tensor = ndarray((nao, nao, dk), buffer=buf1)
            tmp = ndarray((nocc, nao, dk), buffer=buf2)
            beta = 0
            if j_factor != 0:
                with stage('pseudo-dm: J term  dm[p,q]*auxvec[r]',
                           bytes_=float(nao) * nao * dk * 8):
                    cp.multiply(dm[:, :, None], auxvec[aux0:aux1], out=dm_tensor)
                beta = j_factor
            with stage('pseudo-dm 1  rji,qj->iqr',
                       flops=2. * float(nocc) * nocc * nao * dk):
                contract('rji,qj->iqr', dm_oo[aux0:aux1], dm_factor_l, out=tmp)
            with stage('pseudo-dm 2  iqr,pi->pqr  (dense)',
                       flops=2. * float(nao) * nao * nocc * dk,
                       bytes_=float(nao) * nao * dk * 8):
                contract('iqr,pi->pqr', tmp, dm_factor_r, -.5 * k_factor, beta,
                         out=dm_tensor)
            with stage('cp.take -> pair list',
                       bytes_=(float(nao) * nao + nao_pair) * dk * 8):
                if hermi == 1:
                    cp.take(dm_tensor.reshape(-1, dk), pair_addresses, axis=0,
                            out=compressed[:, k0:k1])
                else:
                    dm_tensor1 = ndarray((nao, nao, dk), buffer=buf2)
                    dm_tensor1[:] = dm_tensor.transpose(1, 0, 2)
                    dm_tensor1[:] += dm_tensor
                    cp.take(dm_tensor1.reshape(-1, dk), pair_addresses, axis=0,
                            out=compressed[:, k0:k1])
        with stage('ip1 kernel (3c2e derivatives)'):
            err = kern(
                ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
                ctypes.cast(ejk_aux.data.ptr, ctypes.c_void_p),
                ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
                lib.c_null_ptr(),
                ctypes.c_int(1),
                ctypes.byref(int3c2e_envs),
                ctypes.c_int(shm_size_max),
                ctypes.c_int(len(shl_pair_offsets) - 1),
                ctypes.c_int(1),
                ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(ksh_offsets_gpu[kbatch:].data.ptr, ctypes.c_void_p),
                ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
                ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(aux_ao_offset),
                ctypes.c_int(nao), ctypes.c_int(nao_pair),
                ctypes.c_int(naux_in_batch), ctypes.c_int(mol.natm))
            if err != 0:
                raise RuntimeError('int3c2e_ejk_ip1 failed')
    buf = buf1 = buf2 = compressed = dm_tensor = tmp = None
    if hermi == 1:
        ejk *= 2
        ejk_aux *= 2

    if auxbasis_response:
        with stage('int2c2e_ip1 (aux response)'):
            if j_factor == 0:
                dm_aux = None
            else:
                dm_aux = auxvec[:, None] * auxvec
            if dm_factor_l is dm_factor_r:
                dm_aux = contract('rij,sij->rs', dm_oo, dm_oo,
                                  alpha=-.5 * k_factor, beta=j_factor, out=dm_aux)
            else:
                dm_aux = contract('rij,sji->rs', dm_oo, dm_oo,
                                  alpha=-.5 * k_factor, beta=j_factor, out=dm_aux)
            dm_aux = dm_aux[aux_sorting[:, None], aux_sorting]
            ejk_aux -= cp.asarray(int2c2e_ip1_per_atom(auxmol, dm_aux))
            ejk += ejk_aux
    return ejk.get()


# --------------------------------------------------------------------------
mol = build_mol(a)
mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc).density_fit(
    auxbasis=a.auxbasis)
mf.conv_tol = a.conv_tol
mf.grids.level = a.grids
mf.verbose = 0
if a.cycles:
    mf.max_cycle = a.cycles
mf.kernel()

g = mf.nuc_grad_method()
g.auxbasis_response = True
dm = mf.make_rdm1()
from gpu4pyscf.dft import numint                              # noqa: E402
omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

tag = (f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.auxbasis or "auto"}/{a.xc}/'
       f'{a.patch or "stock"}')
print(f'[{tag}] natm={mol.natm} nao={mol.nao} hyb={hyb} '
      f'gpu4pyscf={gpu4pyscf.__version__}', flush=True)

whole = {}
if a.whole:
    for label, fn in (('whole gradient', lambda: g.kernel()),
                      ('energy_ee', lambda: g.energy_ee(mol, dm))):
        ts = []
        for _ in range(a.reps):
            _sync()
            t = time.perf_counter()
            fn()
            _sync()
            ts.append(time.perf_counter() - t)
        whole[label] = min(ts)
        print(f'{label:38s} {min(ts)*1e3:10.2f} ms  (min of {a.reps})',
              flush=True)

# the stock function, timed as one, and then the traced clone
ts = []
for _ in range(a.reps):
    _sync()
    t = time.perf_counter()
    ref = g.jk_energy_per_atom(dm, j_factor=1, k_factor=hyb or 1., hermi=1)
    _sync()
    ts.append(time.perf_counter() - t)
whole['jk_energy_per_atom (stock, incl. setup)'] = min(ts)
print(f'{"jk_energy_per_atom (stock)":38s} {min(ts)*1e3:10.2f} ms  '
      f'(min of {a.reps})', flush=True)

# the setup `jk_energy_per_atom` does on every call, timed on its own
with stage('setup: SortedMole + Int3c2eOpt.build'):
    mf.with_df.reset()
    sorted_mol = SortedMole.from_mol(mf.with_df.mol, decontract=True)
    int3c2e_opt = Int3c2eOpt(sorted_mol, mf.with_df.auxmol).build()

# The clone is run `--reps` times and the cheapest run is the one reported:
# the first pays a JIT compile for the ip1 kernel and a cuTENSOR plan for each
# of the four contractions, and `docs/DF_WORK_LIST.md` §5 item 3 is about
# exactly that kind of cold start being read as a speedup.
setup = T.get('setup: SortedMole + Int3c2eOpt.build')
best = None
for _ in range(max(1, a.reps)):
    T.clear()
    got = traced_jk_energy_per_atom(int3c2e_opt, dm, 1., hyb or 1., hermi=1,
                                    blk_override=a.blk)
    run_total = sum(e['t'] for e in T.values())
    if best is None or run_total < best[0]:
        best = (run_total, {k: dict(v) for k, v in T.items()})
T.clear()
T.update(best[1])
if setup is not None:
    T['setup: SortedMole + Int3c2eOpt.build'] = setup
err = float(np.abs(got - ref).max())
print(f'{"traced clone agrees with stock to":38s} {err:10.2e} Eh/Bohr',
      flush=True)

total = sum(e['t'] for e in T.values())
print(f'\n{"stage":42s} {"ms":>9s} {"n":>5s} {"%":>6s} {"TF/s":>7s} {"GB/s":>7s}')
print('-' * 82)
for label, e in sorted(T.items(), key=lambda kv: -kv[1]['t']):
    tf = e['flops'] / e['t'] / 1e12 if e['flops'] else 0.
    gbs = e['bytes'] / e['t'] / 1e9 if e['bytes'] else 0.
    print(f'{label:42s} {e["t"]*1e3:9.2f} {e["n"]:5d} {100*e["t"]/total:6.1f} '
          f'{tf:7.2f} {gbs:7.1f}')
print('-' * 82)
print(f'{"traced total":42s} {total*1e3:9.2f}')
print(f'\nnao={N["nao"]} nocc={N["nocc"]} naux={N["naux"]} '
      f'nao_pair={N["nao_pair"]} fill={100*N["fill"]:.1f}% '
      f'blksize={N["blksize"]} batch_size={N["batch_size"]} '
      f'aux_batches={N["aux_batches"]}')
print(f'fp64 peak {PEAK_TFLOPS} TF/s, HBM {PEAK_GBS} GB/s', flush=True)

print('RESULT ' + json.dumps(dict(
    tag=tag, gpu4pyscf=gpu4pyscf.__version__, xyz=a.xyz, basis=a.basis,
    auxbasis=a.auxbasis, xc=a.xc, patch=a.patch, hyb=float(hyb),
    sizes=N, whole=whole, traced_total=total, clone_err=err,
    stages={k: v for k, v in T.items()})), flush=True)
