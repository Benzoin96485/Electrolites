# GPU4PySCF 1.7.0 → 1.8.1: environment upgrade and compatibility audit

Environment: conda env `barw-dev`
(`/global/common/software/m4787/miniconda/envs/barw-dev`), 2026-09-04.

## 1. What changed in the environment

`gpu4pyscf-cuda12x` **1.7.0 → 1.8.1**, and nothing else. 1.8.1's requirements
(`pyscf>=2.8.0`, `cupy-cuda12x>=13,!=13.4.0`, `gpu4pyscf-libxc-cuda12x==0.8.1`,
`pyscf-dispersion`, `geometric`, `packaging`) were all already satisfied by the
env, so the install ran `--no-deps` and touched exactly one package.
`pip check` reports no new conflicts.

| | |
|---|---|
| installed | `gpu4pyscf-cuda12x 1.8.1` (default in `barw-dev`) |
| 1.7.0 kept side by side | `PYTHONPATH=/pscratch/sd/w/weiliang/g4pver/1.7.0` |
| pre-upgrade `pip freeze` | `/pscratch/sd/w/weiliang/g4pver/pip-freeze-before-181.txt` |
| full rollback | `pip install gpu4pyscf-cuda12x==1.7.0` |

The 1.7.0 tree is a clean `pip install --no-deps --target` of the 1.7.0 wheel,
so it carries its own `gpu4pyscf_cuda12x.libs`; prepending it to `PYTHONPATH`
switches versions inside the same env with no reinstall. It lives on
`$PSCRATCH`, which is purge-eligible — the one command above recreates it.

Note that `barw-dev` is group-readable to `m4787`, so anyone else in the
project who activates it now gets 1.8.1.

## 2. Compatibility verdict, as measured

Every module imports under 1.8.1, which proves nothing: four bind to parts of
the driver 1.8.0 replaced.  The table is the *measured* outcome of running
`bench/test_full.py` per module on 1.8.1 (`H2O/6-31G*/B3LYP-D3(BJ)` and
`H2O/def2-TZVPD/wB97M-V`), not a source reading.

| module | patch target | on 1.8.1, before this round | cause |
|---|---|---|---|
| `fastk` | `jk._VHFOpt.get_k` | `TypeError: takes 4 positional arguments but 7 were given` | the target gained `(omega, lr_factor, sr_factor)`; it also reads `self.q_cond`, `self.s_estimator` (now properties that `raise RuntimeError('deprecated')`), `self.l_ctr_offsets`, `self.uniq_l_ctr` (no longer on the option object) |
| `fastj` | `j_engine._VHFOpt.get_j` | `AttributeError: '_VHFOpt' object has no attribute 'prim_mol'` | signature unchanged, but `prim_mol` and `prim_to_ctr_mapping` are gone and `q_cond` raises. `lib/gvhf-rys/unrolled_rys_j.cu` was deleted upstream and the Hermite transform moved to the GPU |
| `fastnlc` | `numint._vv10nlc` | `TypeError: _vv10nlc() missing 2 required positional arguments` | upstream is now `(rho_drho, coords, weights, nlc_pars)`; `VXC_vv10nlc` was replaced by `VXC_vv10nlc_fock_eval_UWE` / `..._omega_derivative` |
| `fastejk` | `grad.rhf._jk_energy_per_atom` | `AttributeError: '_VHFOpt' object has no attribute 'uniq_l_ctr'` | signature unchanged; reads `q_cond`, `s_estimator`, `uniq_l_ctr`, `l_ctr_offsets` |
| `fastrsh` | `dft.rks.RKS.get_veff` | `RuntimeError: deprecated` | own patch point unchanged; imports `fastk` and builds two `_VHFOpt`s |
| `fastgrad` | `grad.rks.Gradients.energy_ee` | `AttributeError: ... 'uniq_l_ctr'` | own patch point unchanged; imports `fastejk` |
| `fastxc` | `numint.NumInt.nr_rks` | `IndexError: too many indices` | 1.8.0's API change made `eval_xc_eff` return `exc` with shape `(N,)` instead of `(N,1)`, so `exc[:, 0]` fails.  One line |
| `fastxcgrad` | `grad.rks.get_exc`, `get_nlc_exc` | `get_exc` **passed**; `get_nlc_exc` gave `TypeError: _vv10nlc() takes 4 positional arguments but 6 were given` | both signatures and both return contracts (`(None, exc1)`) are unchanged, and it only takes `[1]` from `eval_xc_eff`; but its VV10 call site used the old `_vv10nlc` |
| `fastgradh` | `grad.rhf.GradientsBase.get_hcore` | **passed** | signature unchanged; `int1e_grids_ip1` and `get_ecp_ip` unchanged |
| `fastxcnlc` | (internal to `fastrsh`) | `IndexError: too many indices` | the same `exc[:, 0]`, spelled `cupy.asarray(exc, order='C')[:, 0]` |

Structures that did **not** change, and therefore needed no attention:
`RysIntEnvVars` (identical ctypes layout, so the kernels' ABI is safe),
`vhf.cuh`'s structs (only `QUEUE_DEPTH` moved 262144 -> 65536 on the C side),
`eval_ao`, `eval_rho`, `_eval_rho2`, `_scale_ao`, `_tau_dot`,
`_contract_rho{,1}`, `_grouped_block_loop`, and every `cupy_helper` symbol the
modules import.

## 3. Why: the 1.8.0 J/K refactor

The changelog line is "Refactor the molecular J/K matrix construction,
improving K-matrix performance for range-separated hybrid functionals." In
practice `_VHFOpt` was rebuilt:

* `build()` now calls `SortedGTO.from_mol(mol, decontract=True,
  diffuse_cutoff=0.3)` instead of `group_basis(...)`. The sorted molecule is
  **decontracted and split on a diffuse-exponent cutoff**, so it is not the
  same sorted basis the generated kernels were written against, and the
  `nprim`-based decisions in `fastk._usable` no longer describe it.
* `q_cond` / `s_estimator` are gone, replaced by
  `bas_pair_cache[(i,j)] -> (pair_mapping, q_cond_ij, s_cond_ij)` — per
  angular-momentum-group screening data.
* `RYS_build_k` gained `omega, lr_factor, sr_factor`, `s_cond_ij/s_cond_kl`,
  `diffuse_exps` and `dm_penalty`. The range-separated operator and a
  density-aware plus diffuse-exponent screen are now inside the kernel.
* `l_ctr_offsets`, `ao_idx`, `l_ctr_pad_counts`, `q_cond_cpu` are no longer
  attributes of the opt; `uniq_l_ctr` / `l_ctr_counts` live on `sorted_mol`.

So the four broken modules are not broken by renames. `fastxc` needs one line;
`fastk`, `fastj`, `fastejk` need their drivers ported to the new screening
contract.

## 4. What upstream now does that overlaps this work

These three changelog items land on top of modules in this repo, and mean the
measured speedups in `README.md` are **1.7.0 numbers that no longer describe
the gap**:

1. "improving K-matrix performance for range-separated hybrid functionals" —
   the same ground as `fastk`'s `omega` path and all of `fastrsh`. Also closes
   or shrinks catalogue target **A1** (the short-range `erfc` operator).
2. "Refactor the VV10 Hessian implementation to improve performance" plus the
   `_vv10nlc` rewrite — `fastnlc`'s 3.22× was against the old double sum.
3. "Optimize one-electron integral derivatives for Gradients and Hessians" —
   `fastgradh`'s territory.

Catalogue targets confirmed **still open** in 1.8.1:

* **A2** (unrolled-free high-`lij+lkl` `md_j` classes): `unrolled_md_j.cu`
  still has the same 12 kernels, so `(3,3)`, `(4,2)`, `(4,3)`, `(4,4)` still
  fall through to the general kernel.
* the K and gradient class counts are unchanged: `unrolled_rys_k.cu` still
  unrolls 25 classes and `unrolled_ejk_ip1.cu` still 18, so the "40 classes
  GPU4PySCF does not unroll" premise of `fastk` and the 47 of `fastejk` hold.
* **G1** (`_grouped_block_loop` reaching every grid path): still has no
  callers at all.
* **G4** (`lib/multi_gpu.py:99`, `for i in num_devices`): still present,
  verbatim, in 1.8.1 — still a clean upstream PR.

## 5. A 1.7.0 defect that touches this repo's baseline

Upstream commit `d41ec55` ("Minor reorg of gout_pattern bug", #791, in 1.8.0)
fixes a `>>`-for-`<<` bug in the 4-bit dispatch mask of the **non-unrolled**
Rys K/JK kernels. It is present in 1.7.0 at
`lib/gvhf-rys/rys_contract_k.cu:666` (and in `rys_contract_jk.cu`,
`pbc/rys_contract_k.cu`, `gvhf-md/pbc_md_contract_j.cu`), and it is exactly the
fallback path that `fastk` and `fastejk` were written to replace.

The commit message says the bug "silently produc[ed] wrong K/JK matrices for
fallback quartets with an s-shell in i/j/k". Reading the code, the mis-packed
mask collapses to `(ll == 0)`, so `switch (gout_pattern)` selected
`inner_dot<3,3,3,3>` or `<3,3,3,1>` — the *general* instantiations — where a
cheaper `<1,...>` specialization was available for each `l == 0` index. Those
general instantiations look numerically correct, and this repo's own evidence
supports that reading: the 15-case cross-process suite in `bench/test_full.py`
agreed with unpatched 1.7.0 to 9.1e-13 Eh, which a genuinely wrong reference
could not have done.

**This is now settled empirically, and it is a performance bug, not a
correctness bug.**  `src/cuda/myk2/rys_k_general.cu` is this repository's own
copy of that kernel, and it carried the same `>>`.  Changing it to `<<` and
rebuilding `libmykg.so` leaves every energy **bit-identical** while making the
build faster -- H2O/cc-pVQZ B3LYP 3.32 s -> 2.88 s of SCF, SiCl4/def2-TZVP
6.63 s -> 6.46 s -- which is only possible if the general
`inner_dot<3,3,3,3>` path the mis-packed mask forced was already numerically
right and merely slower than the `<1,...>` specialisations it should have
selected.  The fix is in, with `FASTKG_LIB` to switch builds.

The consequence for the published numbers stands, though: **GPU4PySCF 1.7.0's
fall-back K kernel was running deoptimized**, so part of the measured
advantage over "GPU4PySCF's general kernel" on the non-unrolled classes was
against an accidentally slow baseline, and every speedup on those classes has
to be re-measured on 1.8.1.  That is what section 9 does.

## 6. The port

All nine modules (and `fastxcnlc`, a tenth that `fastrsh` uses internally) now
run on 1.8.1.  Everything version-dependent went into one new file,
`src/g4pcompat.py`, so each module keeps a single code path wherever the
difference is only a name:

* `NEW_JK_ABI`, `NEW_VV10` -- capability probes, read off `inspect.signature`
  of the upstream functions rather than a version string.
* `sorted_meta(opt)` -- `(uniq_l_ctr, l_ctr_offsets)` from wherever the release
  keeps them.
* `dense_q_cond(opt)` -- rebuilds the `nbas x nbas` `log(sqrt(absmax((ij|ij))))`
  matrix the kernels index by scattering 1.8.1's per-group `bas_pair_cache`
  values back, keyed by `ish*nbas + jsh` (the same encoding), with `-inf`
  where 1.8.1's build-time overlap mask dropped a pair.
* `tril_pair_mappings(...)` -- 1.7.0's `jk._make_tril_pair_mappings`, vendored
  because 1.8.1 removed it, so the kernels keep getting the tile=6 lists they
  were written for.
* `resolve_rsh(opt, ...)` -- wraps `jk._check_rsh_factors`.
* `diffuse_exps(opt)`, `vv10nlc(...)` -- the two other new upstream arguments
  and the changed VV10 signature.

Then, per module:

| module | what the port does |
|---|---|
| `fastxc` | `exc[:, 0]` -> `exc.ravel()`, which is correct on both releases |
| `fastxcgrad`, `fastxcnlc` | the same `exc` fix, and their VV10 call goes through `g4pcompat.vv10nlc` |
| `fastnlc` | its own `_vv10nlc` split into `_vv10nlc_two_sets` (the 1.7.x signature, which its kernels support and its gradient needed) and `_vv10nlc_one_set` (1.8.x's), with the release deciding which one goes onto `numint` |
| `fastejk` | screening data via `sorted_meta`/`dense_q_cond`/`tril_pair_mappings`; a second `_run_ref_new` that calls 1.8.x's `RYS_per_atom_jk_ip1` with per-group `q_cond`/`s_cond`, `diffuse_exps` and a `dm_penalty` of 0, using GPU4PySCF's own pair lists for the classes handed back |
| `fastk` | the new `(omega, lr_factor, sr_factor)` parameters; the single-range operator built unscaled and multiplied by the factor on the way out, so the handed-back classes can be asked for the unscaled operator too; `_run_ref_new` for 1.8.x's `RYS_build_k`; and the fused range-separated path below |
| `fastj` | a second driver, `get_j_181`: 1.8.0 folded the primitive molecule into `sorted_mol` (`decontract=True`, one primitive per shell), moved the Hermite transform to the GPU (`_dm_to_Rt`/`_Rt_to_dm` with `rys_envs`, in place of the host-side `Et_dot_dm`/`jengine_dot_Et` -- catalogue item **A3**, closed upstream), and made `MD_build_j` take per-group `q_cond_ij`/`q_cond_kl`.  The kernels are untouched: they get a dense q_cond from `g4pcompat`, and `_make_tile_max_hierarchy`'s layout is the one `_qd_offset_for_threads` already assumed |
| `fastrsh` | **retired on 1.8.x.**  1.8.0's `dft.rks.get_veff` makes a single `ks.get_k(mol, dm, hermi, omega, alpha, hyb)` call where 1.7.x made two and scaled them, so replacing `get_veff` buys nothing -- and a 1.7.x-shaped reimplementation of it would get in the way of upstream's own incremental `ecoul` handling.  `apply_patch()` stands down and `fastk` serves the fused request at its own patch point |

### The fused range-separated path

This is the one place where the port is a design change rather than a rename.
1.8.x asks for the whole range-separated operator in one call:
`lr_factor*erf(w r)/r + sr_factor*erfc(w r)/r`, with `omega < 0` telling the
kernel to allocate twice the Rys roots.  `fastk` has no `erfc` kernel and
never will, but `erf + erfc == 1` turns the request into

    sr_factor * (full range) + (lr_factor - sr_factor) * (long range)

which is exactly the linear combination `fastk.get_k_rsh` already builds in
one pass over the quartets.  The full-range half has to be screened on a
full-range `q_cond`, and the option object 1.8.x hands us was built inside
`mol.with_range_coulomb(|omega|)`, so `_full_range_opt` builds one sibling
`_VHFOpt` per SCF for it and asserts that the two sorted the shells the same
way.  `FASTK_NO_FUSED=1` hands the request straight back, for the ablation.

Every range-separated functional now takes this path -- HSE06 included, which
1.7.x fell through entirely, because 1.8.x routes the short-range operator
through the same call.

## 7. Correctness on 1.8.1

Both suites, re-run on 1.8.1 with the ported stack:

| suite | what it compares | worst difference | the same suite on 1.7.0 (README) |
|---|---|---|---|
| `bench/test_full.py`, 15 cases | energy **and** gradient, patched process vs unpatched process, all eight modules on | **8.2e-12 Eh**, **2.8e-12** per gradient component | 9.1e-13 Eh, 3.3e-11 |
| `bench/test_grad.py`, 23 cases | gradient at the same converged density, in one process | **2.8e-12** | 3.3e-11 |

The cases span six basis sets to cc-pVQZ, twelve functionals including three
meta-GGAs, `omega < 0`, third-row elements, a cation and an open-shell radical.

## 8. Performance on 1.8.1

One A100-SXM4-40GB, `conv_tol=1e-9`, `direct_scf_tol=1e-13`, level-3 grids,
one process at a time on an otherwise idle node.  The gradient is timed twice
on the converged density and the second timing is reported.  "ported" is
`fastk, fastxc, fastnlc, fastejk, fastgrad, fastxcgrad, fastgradh` -- seven
modules; `fastrsh` is retired on 1.8.x and `fastj` is measured separately
below.

**`PfPMT`, 284 atoms, 6-31G\*, 2268 AO.**

| | GPU4PySCF 1.7.0 | GPU4PySCF 1.8.1 | ported, on 1.8.1 | speedup on 1.8.1 |
|---|---|---|---|---|
| **B3LYP-D3(BJ)** SCF | 113.12 s | 121.57 s | **63.70 s** | **1.91x** |
| **B3LYP-D3(BJ)** gradient | 21.79 s | 19.51 s | **12.40 s** | **1.57x** |
| **B3LYP-D3(BJ)** energy + gradient | 134.91 s | 141.08 s | **76.10 s** | **1.85x** |
| **wB97X** SCF | -- | 122.80 s | **72.48 s** | **1.69x** |
| **wB97X** gradient | -- | 28.82 s | **17.21 s** | **1.67x** |
| **wB97X** energy + gradient | -- | 151.62 s | **89.69 s** | **1.69x** |

Energies agree with unpatched 1.8.1 to 1e-10 Eh on a -7652 Eh total
(B3LYP: -7652.5600864500 both; wB97X: -7650.3619215439 against
-7650.3619215441), and the gradient norms to the printed ten digits.

Two things worth reading off that table.  **Stock 1.8.1 is 7 % slower than
stock 1.7.0** on this B3LYP SCF (121.57 s against 113.12 s) while its gradient
is 12 % faster (19.51 s against 21.79 s) -- the 1.8.0 J/K refactor was aimed
at range-separated hybrids, and a plain hybrid at 6-31G\* paid a little for
it.  And the **wB97X** row is the one that answers whether upstream's new
single-pass `lr*erf + sr*erfc` kernel closed the range-separated opportunity:
it did not.  Decomposing the request as `sr*(full range) +
(lr-sr)*(long range)` and building both ranges in one pass over the quartets
is still 1.69x faster than allocating twice the Rys roots, on a real system.

For reference, the numbers `README.md` publishes for the same molecule against
1.7.0 with all nine modules are 112.6 -> 56.2 s of SCF (2.00x) and
22.49 -> 13.04 s of gradient (1.72x).  Seven modules on 1.8.1 reach 1.91x and
1.57x, so the port did not lose the result -- but it has not recovered all of
it either, and the gap is `fastj` and `fastrsh`'s grid fusion.

### Build by build, on both sizes

The wall-clock table above cannot say *where* the time goes, and on `HcgC` it
cannot be taken at all (next subsection).  So both clusters were also measured
build by build on one fixed `minao` density, `min` of three (two for the
gradient), one process at a time on an otherwise idle node:

| fixed density | `PfPMT` stock | `PfPMT` ported | | `HcgC` stock | `HcgC` ported | |
|---|---|---|---|---|---|---|
| `get_k` | 2.022 s | **0.372 s** | **5.44x** | 4.665 s | **1.193 s** | **3.91x** |
| `nr_rks` (the XC build) | 0.920 s | **0.305 s** | 3.02x | 3.311 s | **0.828 s** | **4.00x** |
| `get_j` | 0.451 s | **0.328 s** | 1.38x | 1.574 s | **1.187 s** | 1.33x |
| **`get_veff`** (one cycle's work) | **3.375 s** | **1.012 s** | **3.33x** | **9.618 s** | **3.226 s** | **2.98x** |
| **gradient** | **16.711 s** | **10.244 s** | **1.63x** | **48.810 s** | **30.585 s** | **1.60x** |

Three things follow, and the first is the most useful result of this round.

**`get_veff` is 3.33x on `PfPMT` while the SCF wall clock is 2.13x, so about a
third of that SCF is not in J, K or XC at all** -- it is DIIS, the fp64
diagonalisation, grid construction and Python overhead, none of which any of
these modules touches.  The catalogue files those under **H1/H2** as *negative
controls* (`cusolverDnDsygvd`, `lib/diis.py`); on this molecule they are not a
control but a real Amdahl ceiling: making `get_veff` twice as fast again would
move the `PfPMT` SCF only from 2.13x to about 2.6x.  Nothing in the wall-clock
table alone shows this.

**`get_j` is 1.3x on both sizes and is now the weakest link**, which is exactly
where **A2** points -- the four high-`lij+lkl` `md_j` classes that 1.8.1 still
sends to its general kernel.

**The two ratios move in opposite directions with size**: K gets relatively
better on the smaller system (5.44x against 3.89x) and XC on the larger one
(3.02x against 3.98x).  So no single-system "overall speedup" extrapolates,
and every number here carries its molecule.

### The second size, and why it is measured differently

`HcgC` is 545 atoms and charge -4, and section 5's warning applies to it in
full: stock 1.8.1 does not converge this SCF in 50 cycles (659.59 s,
`conv=False`) while the ported stack does converge it (272.95 s, `conv=True`).
The two runs therefore did different amounts of work, and their wall-clock
ratio measures the DIIS path, not the kernels.  Which one converges is luck.

So `HcgC` is measured the way this repository's own oracle table prescribes for
a large ill-conditioned system -- **fixed density, build by build**.  One
`minao` guess is built once and every build is timed on it, `min` of three
(two for the gradient), one process at a time on an otherwise idle node:

| `HcgC`, 545 atoms, 4170 AO | GPU4PySCF 1.8.1 | ported | speedup |
|---|---|---|---|
| `get_k` | 4.665 s | **1.193 s** | **3.91x** |
| `nr_rks` (the XC build) | 3.311 s | **0.828 s** | **4.00x** |
| `get_j` | 1.574 s | **1.187 s** | **1.33x** |
| **`get_veff`** (one SCF cycle's work) | **9.618 s** | **3.226 s** | **2.98x** |
| **gradient** | **48.810 s** | **30.585 s** | **1.60x** |

The three builds sum to 9.55 s of the 9.618 s, so nothing material hides
outside them.  Two readings: the gradient speedup is the same on both sizes
(1.60x here, 1.58x on `PfPMT`), while the SCF side is *better* on the larger
system (2.98x against 2.13x of `PfPMT` wall clock) because K is 48.5 % of a
stock `HcgC` cycle.  And **`get_j` at 1.32x is now the weakest link** -- which
is exactly where catalogue item **A2** points, the four high-`lij+lkl` `md_j`
classes that 1.8.1 still leaves on its general kernel.

## 9. One JoltQC idea, tried and rejected

JoltQC (`ByteDance-Seed/JoltQC`, Apache-2.0, read at commit `fd55a75`) does
its J/K task screening in a **separate kernel** that compacts the surviving
quartets with a warp-level prefix sum, so a warp reserves its slots with one
atomic instead of one per surviving task
(`jqc/backend/jk/screen_jk_tasks.cu`).  The three task-queue builders here --
`fastk_prologue.cu`'s `fill_vk_tasks` (18 kernels) and `fill_vk_tasks2` (51),
and `fastejk_prologue.cu`'s `fill_ejk_tasks` (65) -- all used
`atomicAdd(ntasks, 1)` per surviving pair, on a shared-memory counter that
every lane of a warp hits at the same address.  One helper replacing those
three lines covers 134 kernels.

It was implemented (with `__activemask()` rather than a full ballot, because
eight of the high-l launch configurations use block sizes that are not
multiples of 32 -- 144, 200, 240 threads -- and the call site sits inside a
divergent `if`), verified to give bit-identical energies on four cases, and
measured on two sizes of the same problem:

| `PfPMT`, fixed density | warp-aggregated | per-task `atomicAdd` | ratio |
|---|---|---|---|
| `get_k`, 6-31G\* (2268 AO) | 0.3750 s | 0.3729 s | 0.995 |
| `get_k`, def2-TZVP (5241 AO) | 3.072 s | 3.064 s | 0.997 |
| gradient, 6-31G\* (the `fastejk` call site) | 11.930 s | 11.920 s | 0.999 |

In the same runs the two builds the change does *not* touch, `get_j` and
`nr_rks`, moved by 0.999 and 1.006 -- that is the noise floor, and every
number above sits inside it.  **Reverted.**  The mechanism is that the
screening loop is bound by the global-memory latency of `q_cond` and
`dm_cond`, not by the shared-memory atomic, so collapsing 32 atomics into one
does not touch the critical path.  JoltQC needs the prefix sum because its
screening is a standalone kernel filling a *global* queue; inlined screening
against a per-bra-pair persistent grid has no equivalent to gain.

Two other JoltQC items were checked and dropped before implementation:
`linalg_helper.py`'s `l2_block_pooling` is exported but never called there
(both real call sites use `max_block_pooling`, i.e. `condense('absmax', ...)`),
and its VV10 padded-struct shared tile is already bettered here --
`fastvv10.cu`'s mixed path packs a far pair into a `float4` plus a `float2`,
"two shared loads instead of six".  Its stated rationale is also wrong: all
threads reading the same shared address is a broadcast, not a bank conflict.

Its NVRTC options were tried too, and are also a dead end here.  JoltQC
compiles with `("-std=c++17", "--use_fast_math", "--minimal")`; catalogue
**G3** flags this repository's first build as "tens of seconds", so
`--minimal` looked like a free win.  Compiling `fastk`'s whole generated
module (58 352 lines) with CuPy's disk cache disabled, so every timing is a
real compile:

| options | compile |
|---|---|
| `-std=c++17` (what we use) | 28.50 s |
| `+ --minimal` | 27.47 s |
| `+ --use_fast_math` | 27.42 s |
| both | 27.35 s |

Four percent, i.e. one second.  The options matter for JoltQC because it JITs
one small kernel per angular-momentum class, where fixed overhead is a large
share; here the time is set by the sheer volume of generated code.  So **G3
cannot be fixed with compile flags** -- it needs the module split so that a
6-31G\* job does not compile the spdfg kernels it will never launch, which is
worth doing anyway when this ships as an installable plugin and a user's first
run pays the bill.  `--use_fast_math` is separately not worth taking: under a
second of compile time against changing the precision semantics of `exp`,
`rsqrt` and division, when agreement with upstream at 1e-12 is one of the
things this work claims.

What is still worth taking, in order: the **four-way `(fi,fj,fk,fl)`
fragmentation search** of `jqc/backend/data/generate_fragment.py` (our tables
are one-dimensional, `(nsq, gout_stride)`), the **`pair_vk` block reduction**
in place of the four `atomicAdd`s that end every K kernel, and the per-block
**log-magnitude AO screen** of `dft/estimate_log_aovalue.cu` for `fastxc`.

## 10. What is left


1. **`fastrsh`'s other half.**  Retiring it on 1.8.x gives up the exchange
   fusion to upstream, which is right -- but it also gives up
   `fastxcnlc.nr_rks_nlc`, the pass that built the XC potential and the VV10
   non-local correlation from one sweep of the grid.  Nothing on 1.8.x picks
   that up, because `dft.rks.get_veff` still calls `nr_rks` and `nr_nlc_vxc`
   separately and the fusion needs a patch point above both.  This is the
   clearest remaining item for range-separated meta-GGAs.
2. **`fastk`'s short-range operator.**  The fused path decomposes 1.8.x's
   request into full-range plus long-range and never evaluates `erfc`.  That
   is exact, but it means the *screening* of the full-range half uses a
   sibling option object's q_cond, i.e. one extra `_VHFOpt.build()` per SCF.
   Evaluating `erfc` directly (catalogue **A1**) would remove the sibling.
3. **`dense_q_cond` memory.**  Rebuilding the `nbas x nbas` matrix costs
   `nbas^2` float32 -- 5 MB on `PfPMT`'s jk option object, but the J engine's
   `sorted_mol` is fully decontracted, so it is far larger there.  1.8.x
   itself never materialises it.  Teaching the kernels to index the per-group
   compact `q_cond` would remove the allocation.
4. **Multi-GPU.**  1.8.x runs `get_k`, `get_j` and `_jk_energy_per_atom`
   through `multi_gpu.run`; every module here still asserts a single device.
5. **The catalogue items 1.8.1 leaves open**, re-checked against it in
   section 4: **A2** (the four unrolled-free high-`lij+lkl` `md_j` classes),
   **G1** (`_grouped_block_loop` still has no callers at all), **G4**
   (`lib/multi_gpu.py:99`, `for i in num_devices`, verbatim in 1.8.1).  **A3**
   (the J engine's Hermite transform on the GPU) is closed -- upstream did it.
6. **`README.md` still quotes 1.7.0 numbers.**  Section 8 is the 1.8.1 set for
   `PfPMT`; the second size and the def2-TZVPD regime have to be re-measured
   before anything is republished, and every number should carry the GPU4PySCF
   version it was measured against.

## 11. Reproducing

This document is a lab record: the paths and the environment name in section 1
are the machine it was measured on (one A100-SXM4-40GB, NERSC Perlmutter).  In
this repository the same suites and benchmarks run as:

```bash
pip install -e .                    # or: pip install .
python tests/test_full.py --patch all      # 15 cases, energy + gradient
python tests/test_full.py --patch ''       # the unpatched half, to diff against
python tests/test_grad.py                  # 23 cases, gradient at one density

python benchmarks/cluster_bench.py  --xyz benchmarks/molecules/PfPMT.xyz --charge -1 --patch all
python benchmarks/fixed_density.py  --xyz benchmarks/molecules/HcgC.xyz  --charge -4 --patch all
python benchmarks/konly.py --xyz benchmarks/molecules/PfPMT.xyz --charge -1 --basis def2-TZVP --patch fastk
```

To A/B against a different GPU4PySCF release without touching the environment,
install that release into its own prefix and put it first on `PYTHONPATH`:

```bash
pip install --no-deps --target /path/to/g4p-1.7.0 gpu4pyscf-cuda12x==1.7.0
PYTHONPATH=/path/to/g4p-1.7.0 python tests/test_full.py --patch all
```

Ablations: `FASTK_NO_FUSED=1` hands 1.8.x's fused range-separated request
straight back to GPU4PySCF; `FASTK_NO_GENERAL`, `FASTK_NO_K2`, `FASTK_NO_HIGH`
and `FASTK_NO_OMEGA` hand back the corresponding class groups;
`ELECTROLITES_LIBMYKG` selects a prebuilt general-K library instead of the
on-demand build.  README.md lists the rest.
