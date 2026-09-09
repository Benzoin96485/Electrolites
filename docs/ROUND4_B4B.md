# Rank 8 (B4b + B4c) and rank 9 (B3): what the two entries are worth, measured

`instructions/06` §5 ranks **B4b + B4c** eighth and **B3** ninth.  Both are
Hessian entries; B4b is also, though the entry does not say so, the whole of
the two-electron half of every response property.  This is the round that took
them.

Three things came out of it that the entries do not contain, and they are the
reason this file exists next to `docs/README_fastjk.md`:

1. **B4b is two entry points, and one of them nothing here was on.**  The
   Hessian's CPHF calls `jk._VHFOpt.get_k` with `n_dm = 3·natm`; every
   response property calls `jk._VHFOpt.get_jk`, a *fused* kernel.  Both are
   served now.  §2 and §3.
2. **The entry's mechanism has a measurable ceiling, and the two regimes sit on
   opposite sides of it.**  Fitting the build as `T = R + n_dm·C` gives
   `R` 3.55x better here and `C` **1.01x** — the per-density contraction is
   the same code.  So the same kernels are worth 2.8x at `n_dm = 1` and 1.16x
   at `n_dm = 126`, and that is arithmetic, not implementation quality.  §4.
3. **B4c's premise is false on 1.8.1, for a structural reason.**  The entry
   says a CPHF's perturbed densities are atom-local so the union screening
   wastes most of the quartet list; measured, a perfect per-density screen
   would leave 87 % of the work.  1.8.1's CPHF runs in the MO basis, and
   canonical MOs are delocalised.  §5.
4. **B3 is not "as B1".**  The entry says the ip2 kernels are the same shape as
   the gradient's ip1 kernels, so the P1–P5 treatment transfers.  GPU4PySCF
   unrolls **seven** angular-momentum classes for ip2 against twenty-five for
   the exchange build — nothing above `(pp|pp)` — so at def2-SVP **63 % of
   `ejk_ip2` runs on a general kernel**, which is a different problem and a
   bigger one.  §6.

Everything below is one A100-PCIE-40GB, GPU4PySCF 1.8.1, B3LYP/def2-SVP,
`grids.level = 3`, `conv_tol = 1e-9`, `OMP_NUM_THREADS=32`, on the catalogue's
own two molecules — penicillin V (42 atoms, 430 AOs) and taxol (113 atoms,
1123 AOs).  Every stock/patched pair is run back to back, for the shared-node
reason `docs/RESPONSE_1.8.1.md` §4b gives.

---

## 1. What changed

| | before | after |
|---|---|---|
| `jk._VHFOpt.get_k`, `n_dm > 1` | handed back (`fastk.py`'s `if n_dm != 1`) | served: the density loop is inside the shell quartet in all three kernel families |
| `jk._VHFOpt.get_k`, `hermi = 0` | handed back, because the caller doubles `n_dm` to reach both triangles | served, as a side effect of the above |
| `jk._VHFOpt.get_jk` (fused) | **not patched by any module** | `fastjk`: 65 classes, `hermi` 0 and 1, any `n_dm` |

The exchange change is small: the `for (int i_dm = 0; i_dm < n_dm; ++i_dm)`
loop GPU4PySCF wraps its contraction in, restored around ours, with the offset
widened to `size_t` because `i_dm * nao * nao` overflows a signed int at 339
densities on a 2268-AO basis.  `codegen/gen_k2_kernels.py` had been replacing
`kmat.n_dm` with the literal `1`; that one word is now `n_dm`.

`fastjk` is a new module and a new kernel family; `docs/README_fastjk.md` is
its documentation.  Its two lifted families come from GPU4PySCF's
`unrolled_rys_jk.cu`, which is possible because that file's Rys arithmetic is
*textually identical* to `unrolled_rys_k.cu`'s — checked by
`gen_kernels.assert_same_integrals` rather than assumed — so one template
serves both and only the contraction differs.

## 2. The Hessian's `get_k`

Whole analytic Hessian, penicillin V, `conv_tol_cpscf = 1e-6`, everything else
patched, `FASTK_NO_NDM=1` as the ablation (i.e. this repository before the
round):

| stage | before | after | |
|---|---|---|---|
| **whole Hessian** | **56.49 s** | **53.36 s** | **1.06x** |
| `partial_hess_elec` | 9.44 | 9.26 | 1.02x |
| `..ejk_ip2` (**B3**) | 7.45 | 7.43 | 1.00x |
| `make_h1` | 5.74 | 5.57 | 1.03x |
| `solve_mo1` | 41.09 | 38.33 | 1.07x |
| `....nr_rks_fxc_mo` (B4d) | 7.04 | 7.05 | 1.00x |
| `......get_j`, `n_dm` ≤ 126 (B4a) | 13.14 | 13.12 | 1.00x |
| `......get_k`, `n_dm` ≤ 126 (**B4b**) | **19.65** | **16.96** | **1.16x** |

The two Hessians agree to 1.2e-9 absolute on the trace
(4.181647004638e+01 against 4.181647004519e+01, i.e. 2.8e-11 relative) and to
3.7e-11 on the Frobenius norm — the same `atomicAdd`-and-CPHF floor
`docs/HESSIAN_1.8.1.md` reports for its own runs.

1.16x on the largest single row of a Hessian is a modest return, and §4 is why:
at `n_dm = 126` the part of the build these kernels made cheaper is a sixth of
it.

The second size for this half is §4's fit rather than a second whole run: a
113-atom Hessian A/B is 25 minutes per half and the pair was lost twice to node
conditions, so the only end-to-end Hessian figure in this round is the 42-atom
one.  `benchmarks/ndm_k.py` on taxol gives the same shape one size up (`base`
2.36x, `C` 1.16x, 1.75x at `n_dm = 1` falling to 1.20x at 32), and on
`docs/HESSIAN_1.8.1.md` §4's shares 1.16x on a `get_k` that is over half the
wall clock would be about 1.08x on the run -- an inference, written as one.

## 3. The response path

`docs/README_fastjk.md` §2 and §6 have the tables.  In summary:

| | GPU4PySCF 1.8.1 | + Electrolites (before) | + Electrolites (this round) |
|---|---|---|---|
| **penicillin V** polarizability (CPKS, 3 RHS) | 16.77 s | 14.26 s (1.15x) | **8.06 s (2.08x)** |
| ↳ its fused J+K build | 11.76 s | unchanged, ±0.6 % | **5.37 s (2.19x)** |
| **penicillin V** TDA, 5 roots | 22.21 s | 17.74 s (1.24x) | **14.26 s (1.56x)** |
| ↳ its fused J+K build | 13.53 s | unchanged | **10.17 s (1.33x)** |
| **taxol** polarizability | 101.08 s | — | **75.06 s (1.35x)** |
| ↳ its fused J+K build | 82.72 s | — | **62.28 s (1.33x)** |
| **taxol** TDA, 5 roots | 189.66 s | — | **148.67 s (1.28x)** |
| ↳ its fused J+K build | 165.57 s | — | **136.72 s (1.21x)** |

The two sizes disagree by a factor of 1.5 on the whole-run ratio and §4 is why:
the taxol fit puts `base` at 2.36x where penicillin V's is 3.55x, and at
`nset = 3` that difference is most of the gap.  Both gain, the mechanism
accounts for the difference, and neither ratio depends on a threshold — which
is what catalogue §3.4's size-independence rule asks of a result.

The observables are unchanged to every digit printed at both sizes: the
polarizability trace and Frobenius norm to thirteen digits
(1.601787534941e+03, 9.365475710724e+02 on taxol) and the five excitation
energies to nine decimals.

The fused kernel itself is 2.79x at `nset = 1`, 1.96x at 3 and 1.65x at 5 for
`hermi = 1`, and 2.30x / 1.53x / 1.36x for `hermi = 0`.

**The fused arrangement is now unambiguously right.**
`docs/RESPONSE_1.8.1.md` §4 had measured a `fastj` + `fastk` split at 2.30x of
the fused kernel at one density and 0.73x at three, and refused to pick a
threshold between them on the grounds that it would be tuned to one molecule.
A fused rewrite removes the choice: it beats both arrangements at every row.

## 4. The measurement that bounds the entry: `T = R + n_dm·C`

`benchmarks/ndm_k.py` times the whole exchange build at a range of `n_dm` on
one fixed density — every density a copy of the same matrix, so `dm_cond` and
the surviving quartet list are identical at every `n_dm` and only the
per-density cost varies — and fits

    T(n_dm) = base + n_dm · C

against both implementations in one process.  `base` is the per-quartet
integral work (Rys roots, 2D recurrences, geometry, screening) plus the launch
constant; `C` is the density contraction and its atomicAdds.

**Penicillin V, `hermi = 1`:**

| `n_dm` | GPU4PySCF | ours | |
|---|---|---|---|
| 1 | 0.4877 | 0.1485 | **3.28x** |
| 2 | 0.4971 | 0.1715 | 2.90x |
| 4 | 0.5208 | 0.2219 | 2.35x |
| 8 | 0.5817 | 0.3297 | 1.76x |
| 16 | 0.7887 | 0.5516 | 1.43x |
| 32 | 1.2434 | 0.9906 | 1.26x |
| 64 | 2.2178 | 1.8827 | 1.18x |

    gpu4pyscf   base 0.4019 s   C 0.02775 s/density
    ours        base 0.1133 s   C 0.02759 s/density      base 3.55x, C 1.01x

**Taxol, `hermi = 1`:** 1.75x at `n_dm = 1` falling to 1.20x at 32, with

    gpu4pyscf   base 1.0181 s   C 0.48109 s/density
    ours        base 0.4306 s   C 0.41634 s/density      base 2.36x, C 1.16x

Two conclusions, and they are the round's main result.

* **`C` is the same code in both.**  1.01x at 42 atoms and 1.16x at 113 (where
  more classes fall to the written `gen_khigh` family, whose grouping saves
  some atomicAdds).  The P1–P5 treatment applies to integrals; it does not
  touch the contraction, so the asymptote of the whole entry as `n_dm → ∞` is
  1.01x–1.16x.  Every ratio in §2 and §3 is `(R_stock + nC)/(R_ours + nC)`
  interpolating between 3.3x and that.
* **This settles catalogue Q4 in the direction that matters for planning.**
  Q4 guesses that "at large `n_dm` the bottleneck moves to uncoalesced density
  reads and `atomicAdd` contention" and marks it `[?]`, inferred from code
  shape.  It is now measured that the bottleneck is *the contraction* — 94 %
  of the build at `n_dm = 126` on penicillin V — which is the part of Q4's
  claim that decides what to do next.  Which half of the contraction it is,
  the strided reads or the atomics, is §7.

`hermi = 0` costs almost exactly twice as much per density
(`C` = 0.0655 against 0.0277 s), which is the `2·nset` the caller passes to
reach both triangles, and confirms that nothing else about that path is
different.

## 5. B4c, measured: the perturbed densities are not atom-local

Rank 8's other half is the one the catalogue calls "algorithmic — the hardest
and the most valuable":

> each `dm1[A]` has support only near atom A, yet every density pays the full
> quartet list.  Per-density screening (or grouping densities by spatial
> locality) is an **algorithmic** gain that grows with system size

The statement about the code is right: `scf/jk.py`'s
`dm_cond = condense('absmax', dms, ao_loc)` is one `nbas × nbas` table, the max
over all `n_dm`, and `create_tasks.cu` screens against that.  The statement
about the densities is what `benchmarks/ndm_screen_probe.py` tests, on the
densities a **real** solve hands to `get_k`: for every quartet the union screen
keeps, how many densities would survive a screen built from their own
`dm_cond`?

| system | call | `n_dm` | union quartets | (quartet, density) kept | fraction | cost-weighted |
|---|---|---|---|---|---|---|
| glycine/6-31G\*, Hessian | 1 | 30 | 3.19e5 | 9.38e6 | 0.979 | 0.990 |
| " | 2 | 27 | 3.19e5 | 8.39e6 | 0.975 | 0.989 |
| penicillin V/def2-SVP, Hessian | 1 | 126 | 6.02e7 | 6.61e9 | **0.870** | 0.877 |
| " | 2 | 123 | 5.81e7 | 6.08e9 | **0.850** | 0.858 |

A *perfect* per-density screen — no implementation cost charged — would remove
12–15 % of the contraction at 42 atoms and 1–2 % at 10.  That is **1.14x–1.18x
on `C`**, not an algorithmic gain.

The reason is structural rather than accidental, which is why this is a
correction to the entry and not just a disappointing number.  1.8.1's CPHF
solve runs **in the MO basis**: `hessian/rhf.py:_get_jk_mo` forms its AO
densities as `C·mo1·C_occ^T`, and the canonical orbitals are delocalised, so
`dm1[A]` has support wherever the occupied and virtual orbitals do.  The
entry describes the *right-hand sides* `h1`, which really are atom-local, and
not the perturbed densities the iterations contract.  The trend with size
(0.98 → 0.87) says the ceiling grows slowly, so it is worth re-measuring at
500+ atoms before the entry is closed — but not worth building here.

Two side results from the same probe.  Its `cphf` and `tda` stages return **no
rows at all**, because those paths never call `get_k`; they call `get_jk`.  And
a per-density screen is not free even at the ceiling: it needs an
`n_dm × nbas × nbas` float table (20 MB at penicillin V's size, 340 MB at
taxol's) plus one extra load per (quartet, density) inside the contraction.

## 6. Rank 9, B3: `ejk_ip2` is a general-kernel problem, not a P1–P5 one

The entry reads:

> `unrolled_ejk_ip2_type12.cu` (14568 lines), `unrolled_ejk_ip2_type3.cu`
> (15696 lines), same shape as B1. … as B1.  Review B §B.5.1: "register
> pressure worse than high-L ERIs at the same nominal basis L", so P3 is worth
> more here than on K

Both files are as long as the entry says.  But they contain **seven kernels
each**, not twenty-five: `0000 1000 1010 1011 1100 1110 1111`, i.e. nothing
with a d function anywhere.  Everything else goes to
`rys_contract_jk_ip2.cu`'s general kernel, which carries its g-array addresses
in shared `idx_i/j/k/l` arrays and its loop bounds (`lij`, `lkl`, `nroots`,
`nfij*nfkl`) at run time.

`benchmarks/ejk_ip2_probe.py` wraps each `RYS_per_atom_jk_ip2_type{12,3}` call
with a synchronization and attributes it to its angular-momentum class, which
the driver derives from `shls_slice` exactly as the kernel does.  Penicillin
V/def2-SVP, converged density, `_partial_ejk_ip2` = 8.63 s of which 98.9 % is
in the two kernels:

| group | time/s | share of the stage |
|---|---|---|
| type12, the 7 unrolled classes | 1.892 | 21.9 % |
| type12, the general kernel | **2.494** | **28.9 %** |
| type3, the 7 unrolled classes | 1.283 | 14.9 % |
| type3, the general kernel | **2.862** | **33.2 %** |

**63 % of B3 is the general kernel** at a basis with only d functions, and it
would be more at def2-TZVP.  The twelve most expensive classes are
`type3/2111`, `type12/1111`, `type3/2110`, `type12/2110`, `type12/2111`, …:
low angular momentum with many surviving pairs, not the wide classes.

So the P1–P5 treatment the entry points at can reach at most 37 % of the
stage — and this round applied it there and measured what it is worth, rather
than stopping at the estimate.  `codegen/gen_ejk_ip2.py` lifts all three
GPU4PySCF sources and takes **every** fp64 division out of the primitive and
Rys-root loops (eighteen and a `sqrt` per primitive quartet at five roots
become two and an `rsqrt`), optionally adding the `__launch_bounds__` the entry
asks for.  Timed against upstream's kernels in one process, penicillin
V/def2-SVP, `min` of three after a warm-up:

| variant | time/s | over stock |
|---|---|---|
| stock | 7.430 | 1.00x |
| **the launch scheme alone** (two lines of Python) | **6.585** | **1.13x** |
| P1 (all divisions folded) | 7.267 | 1.02x |
| P1 + P3 (a 128-register cap) | 12.726 | **0.58x** |

**Both of the entry's expectations come out the other way.**

* **P1 is worth 1.02x here against 1.7x–2x on the exchange build**, so these
  kernels are not division-bound; §1 says what they are bound by, and no
  arithmetic cleanup reaches it.
* **P3 is a 1.75x regression where the entry expects it to pay more than on
  K.**  Its premise — "register pressure worse than high-L ERIs at the same
  nominal basis L" — is right about the pressure and backwards about what to do
  with it: these kernels want all 255 registers, and capping them at 128 to fit
  two blocks per SM buys occupancy with spills that cost more.  P3 paid on the
  exchange side because `gen_khigh`'s kernels were *designed* to a register
  budget, not capped after the fact.
* What does pay is the thing the entry does not mention at all: the block shape,
  which `_ip2_quartets_scheme` derives from the shared-memory budget alone.
  **1.13x**, no compiler, no vendored code.  That is what `fastejk2` ships; the
  kernels stay behind `FASTEJK2_KERNELS=1`, correct to 1e-13 and off by
  default.
* The other 63 % is the problem `electrolites/codegen/gen_khigh.py` was
  written for on the exchange side, one shell of angular momentum earlier: an
  exactly-tiled kernel per class with compile-time addresses.  That is a
  generator project of the same size as `gen_ejk.py`, for an entry whose share
  **falls** from 13.9 % of a 42-atom Hessian to 3.0 % of a 113-atom one
  (`docs/HESSIAN_1.8.1.md` §4).
* The cheapest thing to try first is the launch configuration, which
  GPU4PySCF derives from the shared-memory budget alone
  (`_ip2_quartets_scheme`: 256 threads, as many quartets as shared memory
  allows, the rest to `gout_stride`) and never swept.
  `benchmarks/ejk_ip2_sweep.py` is that sweep; §7 records what it found.

So rank 9's own ordering is confirmed — it is a small entry that shrinks with
system size — and the shape of the work it needs is *not* the shape the entry
describes.

## 6b. The regression the round could have caused, and did not

The `n_dm` loop went into all three `fastk` kernel families, i.e. into the
kernels an ordinary SCF and gradient launch at `n_dm = 1`.  So the three suites
that cover those paths were re-run, and they reproduce their previous numbers
exactly:

| suite | cases | worst difference | tolerance |
|---|---|---|---|
| `tests/test_grad.py` | 23, six bases to cc-pVQZ, twelve functionals | 2.87e-12 absolute on a gradient component | as before (`README.md` quotes 2.9e-12) |
| `tests/test_j_ndm.py` | 162 multi-density Coulomb builds | 1.536e-13 relative on J | 1e-12 |
| `tests/test_jk.py` | 6 molecule/basis pairs x `hermi` 0,1 x `n_dm` 1-8 x 3 density shapes | 1.04e-14 (K), 2.05e-14 (fused J) | 1e-12 |
| `tests/test_ejk_ip2.py` | 5, plus the whole analytic Hessian on each | 7.0e-13 relative on `ejk`, 6.2e-11 absolute on the Hessian | 1e-11 / 1e-8 |
| `tests/test_full.py`, both halves | 15 SCF + gradient pairs, separate processes | 7.28e-12 Eh on the energy, 3.06e-12 on a gradient component; all 30 runs converged | as before |

## 7. What is left, in the order the measurements put it

1. **`C`, the multi-density contraction.**  It is 94 % of a `n_dm = 126`
   exchange build and identical in both codes, so it is now the largest single
   thing between this repository and a fast Hessian: `get_k` is 32 % of a
   42-atom Hessian after this round and, on
   `docs/HESSIAN_1.8.1.md`'s numbers, over half of a 113-atom one.  Q4 names
   two candidates for where it goes, "uncoalesced density reads and `atomicAdd`
   contention", and this round measured the first of them and **rejected it**:

   * **The density layout is not the problem, and transposing it is 3x worse.**
     Both codes address the densities as `dm[i_dm*nao*nao + a*nao + b]`, so a
     quartet does read the `n_dm` copies of one matrix element from `n_dm`
     different cache lines.  In the transposed layout
     `dm[(a*nao+b)*n_dm + i_dm]` one 128-byte line would serve sixteen
     densities — and `benchmarks/ndm_layout_probe.py`, which measures the
     access pattern alone with no integrals in the way and no charge for the
     transpose itself, puts that at **0.30x–0.38x** at `n_dm` ≥ 32 for both a p
     and a d shell.  The hypothesis leaves out the reuse that matters: in the
     current layout each density's `nao²` block is compact (1.5 MB at
     penicillin V's size, an L2 resident) and is reused by every one of the
     million quartets, and each thread's `nf` consecutive row elements stay
     contiguous.  Transposing trades a stride *between densities* for a stride
     *between quartets*, and there are four orders of magnitude more quartets.
   * So what is left of Q4 is the **`atomicAdd` traffic** — 4·nf² atomics per
     quartet per density, 144 of them for a d shell — and it is now the only
     unexamined candidate.
   * **Per-density screening** is the other lever on `C`, and §5 bounds it at
     1.14x–1.18x at 42 atoms, growing slowly with system size.
2. **The general fused kernel.**  `fastjk` hands classes with `l > 3` on any
   centre to GPU4PySCF's general fused kernel, where `fastk` has a faster copy
   of the exchange-only one (`cuda/myk2/rys_k_general.cu`).  A def2-TZVP
   response property still spends most of its two-electron time there.
3. **B3's general ip2 kernel** (§6), which is the same kind of work as 2 and
   is worth less.
4. Unchanged from `docs/CATALOGUE_06_STATUS.md` §7: **A5** (rank 3, the grid
   AO-evaluation kernel), **A4**'s SCF mGGA path (rank 5, pending Q13), and
   `VXC_vv10nlc_hess_eval_f_t`, which is 65 % of the VV10 response term and is
   in no catalogue entry.
