# The unrestricted twins of W17 and W18: a transplant, a refusal, and one item that only pays when the atoms are big

`docs/DF_WORK_LIST.md` **W25**.  An open-shell density-fitted Hessian reaches
`df/hessian/uhf.py:_jk_energy_per_atom` (the `ip2` term of
`partial_hess_elec`) and `df/hessian/uhf.py:_get_veff` (the `h1` term of
`make_h1`) — separate functions from the restricted ones W17 and W18
replaced, looked up on `df/hessian/uhf.py` by `df/hessian/{uhf,uks}.py` at
call time, and nothing in this repository was on them.  The entry said
`fastdfhessip`'s items 1–5 should transplant.  They do, and the round is
`electrolites/_dfhessip_u.py`, loaded and patched by `fastdfhessip` under the
same switches.  Every integral, screening decision, metric and symmetry
factor is GPU4PySCF's; the answers agree element-wise to round-off (§5).

---

## 0. Summary

* **The functions are the restricted ones with a spin index**, as the entry
  guessed: occupied factors `(2, nao, nocc)` with `nocc = max(nocca, noccb)`
  and the beta columns past `noccb` zero, every `(i, j)` a `(n, i, j)`, and
  `-k` on each spin density where the restricted code has `-k/2` on the
  total.  Upstream's unrestricted functions have no occupied batching.  So
  items 1 (the launch constant), 2 (the per-atom occupied block as Gram
  matrices, now once per spin), 3 (the first `ipaux` pass folded into the
  second), 4 (h1's per-atom intermediate hoisted), 5 (h1's Coulomb per-atom
  loop as one GEMM), 7 (the pure-functional `ip2` term, which upstream hands
  to the *restricted* `_j_energy_per_atom` on the total density) and 9–11
  (W24's `dsyrk`s and the carved `h_aux` temporaries) all carry over; item 8
  (W24's Cholesky factor) does not, because it needs the whole auxiliary
  index in one chunk of the second pass and the 113-atom open-shell call
  has two or three (§2.3).
* **`ip2` is 1.14x–1.85x and `h1` 1.16x–1.91x** on nine open-shell cases
  from 5 to 60 atoms — organic radical cations, a triplet Pt(II) complex with
  an ECP, Ni(II), Cu(II) and Fe(III) complexes, def2-SVP and def2-TZVPP (§3).
* **At 113 atoms GPU4PySCF's `ip2` does not run.**  On the taxol radical
  cation and on a Co(II) sextet it raises `CUTENSOR_STATUS_NOT_SUPPORTED` in
  the pseudo-density contraction: an output past cuTENSOR's 2²⁹ elements
  whose extents do not factor (`docs/DF_WORK_LIST.md` W21).  This module
  asks cuTENSOR before each contraction that can pass the limit and sends a
  refused one to cuBLAS as the GEMM it is (**item 13**, §2.2); both calls run,
  in 63.4 s and 48.2 s.
* **Item 12 is new, and it only pays where the atoms are big.**  h1's
  per-atom exchange term can be formed with the AO pair projected on the
  occupied orbitals *before* the atom's contraction instead of after it:
  `8 nao² nocc` flops per auxiliary function and direction instead of
  `4 nao³`, for one more launch per atom.  It is 1.37x of the function in
  def2-TZVPP and 1.09x–1.11x at 113 atoms, and it *loses* at every def2-SVP
  case up to 60 atoms (down to 0.71x); a shape score picks it, and picks the
  better form on all eleven systems it was measured on (§2.1).
* **A whole UKS Hessian gains 1.05x–1.12x** (§4).  The two terms were 19–31 %
  of an unpatched one and are 12–27 % after; what is left is mostly the CPHF
  — W8's `_get_jk` and `nr_uks_fxc` — at 35–75 %.
* Two things measured on the way: **the order of stage A's spin sum is the
  one round-off this function does not forgive** (reassociated, the
  long-range operator's term moves by 1.1e-10 relative on a triplet; §5),
  and **a 42-atom UKS DF-PBE Hessian is not reproducible in GPU4PySCF
  itself** — the SCF is not (§4.1).

Everything is one A100-SXM4-80GB of a whole node (`--qos=debug`, four chains
on four GPUs), GPU4PySCF 1.8.1, def2-SVP with its jkfit set unless a row says
otherwise, level-3 grids, `conv_tol_cpscf = 1e-6`, `patch_all()` on both
sides; the "before" of a whole Hessian is `patch_all()` with
`FASTDFHESSIP_OFF=1`.  Function tables are
`benchmarks/df_hess_terms_probe.py`, which now runs a UKS SCF for any
`--spin`, variants round robin in one process against upstream's function
object (`DF_WORK_LIST.md` §5.2).

---

## 1. Where the time was

`benchmarks/df_hess_anatomy.py` (which now wraps the unrestricted entry
points for an open shell), the penicillin V radical cation, UKS
DF-B3LYP/def2-SVP, round 2 of 2 in the second of two before/after pairs:

| stage | before | | after | | |
|---|---|---|---|---|---|
| **whole Hessian** | **35.30 s** | | **31.93 s** | | **1.11x** |
| `partial_hess_elec` | 7.8 s | | 5.7 s | | |
| ↳ `ip2` (W25) | 5.57 s | 16 % | 3.44 s | 11 % | 1.62x |
| `make_h1` | 6.0 s | | 5.2 s | | |
| ↳ `h1 _get_veff` (W25) | 2.78 s | 8 % | 1.95 s | 6 % | 1.43x |
| `solve_mo1` (CPHF) | 21.42 s | 61 % | 21.18 s | 66 % | |
| ↳ `_get_jk` (W8), 7 calls | 12.45 s | 35 % | 12.28 s | 38 % | |
| ↳ `nr_uks_fxc`, 7 calls | 7.40 s | 21 % | 7.52 s | 24 % | |

An open-shell Hessian is a CPHF with two spin blocks, and that is where it
spends its time before this round and more so after.  The CPHF grid half
(`nr_uks_fxc`'s `mo1` branch) is already `fastxcfxc`'s, and `_get_jk` is W8's.

---

## 2. What changed

`electrolites/_dfhessip_u.py` is upstream's single-device path for both
functions with items 1–5, 7 and 9–11 of `fastdfhessip` on the spin index —
the module docstring says which line each one moved — plus the two below.

### 2.1 Item 12: h1's per-atom exchange term on the occupied side

With enough memory (`mem_sufficient`), upstream accumulates for every atom
`A` and direction `x` an AO-pair matrix

    X_A[n,p,q] = sum_{u in A, r} j3c[u,p,r] t[n,u,r,q],   t = C_o . dm_3c,

symmetrises it at the end and projects it on the occupied orbitals.  Item 4
already forms `t` once per block instead of per atom.  But the projection is
linear, so it can come first — `(X + X^T) C` is

    sum_{u in A, r} j3c[u,q,r] s[n,u,r,j]   +   sum_{u in A, r} j3c_pri[n,u,r,j] t[n,u,r,q]

with `s = C_o . dm_oo` (formed once per block) and `j3c_pri` the
`j3c . C_o` that the same block's `vhf1` term has just formed.  That is
upstream's own memory-short branch with both of its intermediates hoisted,
and it costs `8 nao² nocc` flops per auxiliary function and direction where
the AO-pair form costs `4 nao³` — a factor of `nao / (2 nocc)`, 2.3 at
def2-SVP and 5.5 at def2-TZVPP.  It is also two contractions per atom
instead of one, both with the occupied index as their short side.

Measured on the function (both forms against upstream, round robin):

| system | nao / natm | `score` | AO-pair form | occupied form | chosen |
|---|---|---|---|---|---|
| glycine+, 10 atoms | 10.0 | 6.0e4 | **1.61x** | 1.38x | AO |
| FeCl₄⁻ sextet, 5 atoms | 22.4 | 3.0e4 | **1.28x** | 1.16x | AO |
| alanine dipeptide+, 22 | 9.5 | 2.7e5 | **1.68x** | 1.20x | AO |
| Ni(II) dithiolene, 21 | 16.8 | 8.4e5 | **1.20x** | 1.17x | AO |
| Pt(II) triplet, 23 | 11.5 | 4.4e5 | **1.50x** | 1.38x | AO |
| penicillin V+, 42 | 10.8 | 1.3e6 | **1.46x** | 1.44x | AO |
| penicillin V+ DF-PBE | 10.8 | 1.3e6 | **1.57x** | 1.54x | AO |
| Cu(II) macrocycle, 60 | 9.0 | 1.7e6 | **1.69x** | 1.65x | AO |
| penicillin V+, def2-TZVPP | 27.3 | 3.0e7 | 1.39x | **1.91x** | occupied |
| taxol+, 113 | 10.5 | 9.1e6 | 1.53x ‡ | **1.69x** ‡ | occupied |
| Co(II) sextet, 113 | 9.9 | 7.7e6 | 1.55x ‡ | **1.68x** ‡ | occupied |

(`nao` is the Cartesian count the function works in; ‡ in separate
processes, reps 1; the rest from one process, round robin.)  The loss at
small sizes is the extra launches: the saving per atom and auxiliary block
is `4 (nao/natm) dk nao (nao - 2 nocc)` flops, and where that is small the
second contraction's launch and the narrow shape of both cost more than it
saves (inferred from where the crossover falls, not profiled kernel by
kernel).  So
the module takes the occupied form where
`score = (nao/natm) nao (nao - 2 nocc)` passes 4e6 — every def2-SVP case up
to 60 atoms is below 1.7e6 and every case that wins is above 7.7e6, organic
and transition-metal alike — and the default (`FASTDFHESSIP_H1ATOM=auto`)
takes the better form on all eleven rows (it is within 1 % of it on ten, and
10 % off on FeCl₄⁻, a 200 ms call whose AO-form time was 194 ms here and
220 ms in an earlier job).  `=occ` and
`=batched` force either one; `FASTDFHESSIP_H1OCC_SCORE` moves the threshold.
At 113 atoms the stage goes 9.04 → 5.13 s; in def2-TZVPP 2.66 → 0.84 s.

The restricted `_get_veff` has the same term (W18's item 4) and the same
choice, and it is not changed here: `DF_WORK_LIST.md` **W30**.

### 2.2 Item 13: the contractions cuTENSOR refuses at 113 atoms

GPU4PySCF's own `ip2` on the taxol radical cation (113 atoms,
`nao_cart = 1185`, `nocc = 226`, `naux = 6598`):

```
File ".../gpu4pyscf/df/hessian/uhf.py", line 167, in proc
    contract('iqr,pi->pqr', tmp, dm_factor_r[0], -k_factor, beta, out=dm_tensor)
cupy_backends.cuda.libs.cutensor.CuTensorError: CUTENSOR_STATUS_NOT_SUPPORTED
```

— the pseudo-density `(nao, nao, blk)`, past W21's 2²⁹ elements as soon as
`blk` passes 382 at this `nao`, with a cut mode that does not factor.  The Co(II) sextet
fails the same way.  Four contractions in this function can pass 2²⁹
elements on a large open shell: the metric applied to `j3c_oo` (twice,
`(naux, 2, nocc, nocc)`), the `h_aux` block's `j2c_10 . dm_oo`
(`(3, naux, 2 nocc, nocc)`, 1.7e9 at 113 atoms), and the second pass's
metric contraction into `j3c_100` (`(dv, 2, nao, nocc)` per direction).
Each is one GEMM on its flattened operands, so the module asks
`electrolites/_cuplan.py` first and sends a refused one to cuBLAS
(`_mm`); the pseudo-density's block is walked with `_cuplan.prange`, which
shrinks a refused block to one that plans.  The query is cached and costs
nothing measurable; a contraction cuTENSOR accepts runs exactly as before.
At 113 atoms one of the four is refused per call on both molecules
(`plan_gemm` counts two over the probe's two calls), and `tests/test_dfuhessip.py` runs the cuBLAS route with the
refusal simulated.  The restricted `_h_aux_block` carries the same guard,
since this module calls it.

### 2.3 Why not item 8

W24's item 8 applies the metric as its Cholesky factor on the spherical
auxiliary index, and it needs the whole `(10|0)` block resident: the second
pass in one chunk.  The unrestricted block is twice the restricted one, and
at 113 atoms the second pass takes **three** chunks (taxol+, 1872 of 5598
metric functions each, with 48 GB free) and two (the Co sextet) — so item 8
would fall back to the chunked arithmetic exactly where it would be worth
something, and below 42 atoms W24 measured it at 1.05x–1.06x.

---

## 3. The functions

`benchmarks/df_hess_terms_probe.py --terms ip2,h1 --variants stock,ours
--reps 2`, one process per system, min of two:

| system | charge / 2S | basis | `ip2` before → after | | `h1` before → after | |
|---|---|---|---|---|---|---|
| glycine, 10 atoms | +1 / 1 | SVP | 236.0 → 205.2 ms | 1.15x | 300.3 → 187.8 ms | 1.60x |
| FeCl₄⁻, 5 | −1 / 5 | SVP | 538.5 → 471.7 | 1.14x | 248.0 → 213.7 | 1.16x |
| alanine dipeptide, 22 | +1 / 1 | SVP | 810.6 → 569.9 | 1.42x | 536.3 → 320.3 | 1.67x |
| Ni(II) dithiolene, 21 | −1 / 1 | SVP | 3635 → 2919 | 1.25x | 1540 → 1287 | 1.20x |
| Pt(II) amine, 23, ECP | 0 / 2 | SVP | 1700 → 1276 | 1.33x | 947 → 615 | 1.54x |
| penicillin V, 42 | +1 / 1 | SVP | 5546 → 3440 | **1.61x** | 2769 → 1888 | 1.47x |
| penicillin V, DF-PBE | +1 / 1 | SVP | 1384 → 1134 | 1.22x | 2360 → 1497 | 1.58x |
| Cu(II) macrocycle, 60 | +2 / 3 | SVP | 10957 → 5929 | **1.85x** | 4894 → 2904 | 1.69x |
| penicillin V, 42 | +1 / 1 | TZVPP | 14188 → 8144 | **1.74x** | 9773 → 5125 | **1.91x** |
| taxol, 113 | +1 / 1 | SVP | *refused* → 63.4 s | — | 57.9 → 34.3 s | 1.69x |
| Co(II) sextet, 113 | 0 / 5 | SVP | *refused* → 48.2 s | — | 47.1 → 28.0 s | 1.68x |

(DF-PBE's `ip2` is the Coulomb-only term, `_j_energy_per_atom` on the total
density, i.e. W17's item 7 reaching an open shell.)  The margin grows with
the system the way W17's did, and the smallest rows are where a call is a
few hundred milliseconds of launches.  Where the time now is, from
`FASTDFHESSIP_TIME=1`:

| | penicillin V+ | Cu macrocycle | taxol+ (113) |
|---|---|---|---|
| `ip2` kernel (W29) | 1350 ms, 39 % | 2575 ms, 43 % | 4.7 s, 7 % |
| D5, per-atom cross terms | 472 ms, 14 % | 854 ms, 14 % | 23.7 s, 37 % |
| D3, `(10|0)` into the metric's basis | 270 ms | 503 ms | 13.1 s, 21 % |
| D3, the derivative integrals | 585 ms | 785 ms | 6.5 s (three chunks) |
| `h1`: `j2c_10`, per auxiliary atom | 305 ms | 519 ms | 10.5 s |
| `h1`: the metric | 192 ms | 301 ms | 4.8 s |

At 113 atoms the `ip2` call is the D3/D5 contractions W24 attacked in the
restricted case — `2·3·naux²·2 nao nocc` flops each, near the GEMM rate — and
at 21–60 atoms it is the `ip2` kernel, W29, as in the restricted case.

---

## 4. Whole Hessians

`benchmarks/df_hess_anatomy.py --rounds 2`, before and after in alternating
processes on one GPU, the pair run twice; second pair, second round:

| system | basis | before | after | | traces, before / after |
|---|---|---|---|---|---|
| alanine dipeptide+, 22 atoms | SVP | 5.99 s | **5.59 s** | 1.07x | 18.7196865118 / …119 |
| Ni(II) dithiolene⁻, 21 | SVP | 16.86 s | **16.00 s** | 1.05x | 22.0335904269 / …216 |
| Pt(II) amine triplet, 23 | SVP | 11.62 s | **10.59 s** | 1.10x | 20.2392777850 / …853 |
| penicillin V+, 42 | SVP | 35.30 s | **31.93 s** | 1.11x | 41.1493831215 / …210 |
| penicillin V+, DF-PBE | SVP | 17.89 s | **16.63 s** | 1.08x | see §4.1 |
| Cu(II) macrocycle²⁺ quartet, 60 | SVP | 82.53 s | **75.54 s** | 1.09x | 49.0709840162 / …056 |
| penicillin V+, 42 | TZVPP | 104.17 s | **93.06 s** | 1.12x | 39.2371971539 / …534 |

(The DF-PBE row is from the round before item 12, which that system does not
take.)  The traces agree to the unpatched code's own run-to-run spread (the
`ip2` kernel's atomics, 1e-9) everywhere but the Cu complex, where they
differ by 1.1e-8 Eh/bohr²: that is item 2's Gram form, §5.  A 113-atom
open-shell DF Hessian cannot be run by GPU4PySCF (§2.2), and one was not
run here: the restricted one is 700 s, and the CPHF of an open shell is
twice that.

### 4.1 A UKS DF-PBE Hessian of the penicillin V cation is not reproducible, and it is the SCF

The DF-PBE row's traces are 12.4236374420, 12.4236759087 before and
12.4236526450, 12.4236795539 after — a spread of 4e-5 on *both* sides.
GPU4PySCF with nothing patched does the same: 12.4236105849 and
12.4236498778 in two processes, with SCF energies of −1501.678471345587 and
−1501.678471364504 at `conv_tol = 1e-9`.  The SCF does not land on the same
density twice, and the Hessian inherits it; the alanine dipeptide cation in
DF-PBE and every B3LYP case here reproduce to 3e-10.  It is not this round's
and not a kernel's, and it is W23's falsification criterion — a convergence
setting, not a bug — firing this time.

---

## 5. Correctness

`tests/test_dfuhessip.py`, element-wise against GPU4PySCF's own function
objects in one process on one converged UKS SCF, relative to the largest
element:

| | worst |
|---|---|
| ip2 and h1, ethanol⁺ doublet, B3LYP / HF / `k = 0` factors, every switch alone | 1.9e-13 (ip2), 1.1e-15 (h1) |
| the ethanol²⁺ triplet (`nocca − noccb = 2`), and the long-range operator (`omega = 0.3`) on it and on water⁺ | 1.3e-13 / 1.9e-15 |
| the memory-short paths: second pass in 2–11 chunks, fused and not; h1 without the per-atom AO matrices | 1.9e-13 / 2.1e-15 |
| the cuBLAS route of item 13, refusal simulated | 2.9e-14 |
| a whole ethanol⁺ UKS DF-B3LYP Hessian, patched against unpatched | **8.9e-12 Eh/bohr²** |

and `tests/test_dfhessip.py` and `tests/test_patch_aliases.py` still pass —
the latter now also checks that nothing holds a stale reference to either
module's four saved originals.

**Stage A is computed one spin at a time, in upstream's order, on purpose.**
The first version contracted both spins of `j3c_oo` in one cuTENSOR call.
On the triplet the long-range operator's `ip2` term came out 1.2e-10 from
upstream's; with stage A as upstream's and every other stage reassociated
the same way it is 4.6e-14, and upstream against itself is 3e-14.  The
metric `dm_oo` goes through next is the long-range one, whose conditioning
turns a last-bit difference in `j3c_oo` into that — the same lesson as W24's
(the factor has to be the only factor), on a different stage.

**Item 2's Gram form is 1.1e-10 from upstream on the Cu(II) macrocycle**
(4e-9 Eh/bohr² absolute), where upstream against itself is 3e-12 and this
module with the Gram form off (`FASTDFHESSIP_OOSUM=transpose`) 1.0e-11.  Every
other switch leaves it where it is, so it is the rearrangement itself: the
Gram form sums `P∘R + Q` over every AO pair of two atoms where upstream sums
the squares of symmetrised occupied blocks, and on this molecule the terms
cancel further.  At 4e-9 Eh/bohr² it is four orders under what
`conv_tol_cpscf = 1e-6` leaves in the Hessian, and the whole Hessian's trace
moves by 1.1e-8.  On every other system it is 1e-14–2e-11.

`benchmarks/coverage_probe.py --hessian --density-fit` now has rows for the
two unrestricted entry points.

---

## 6. Tools

```bash
python tests/test_dfuhessip.py [--basis def2-tzvpp] [--no-hessian]
python benchmarks/df_hess_terms_probe.py --xyz benchmarks/molecules/penicillinV.xyz \
    --charge 1 --spin 1 --basis def2-svp --xc b3lyp --patch all --terms ip2,h1 \
    --variants stock,ours [--env 'ao:FASTDFHESSIP_H1ATOM=batched,occ:FASTDFHESSIP_H1ATOM=occ']
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_terms_probe.py ...   # stage tables ([... U])
python benchmarks/df_hess_anatomy.py --xyz ... --charge 1 --spin 1 --patch all --rounds 2
FASTDFHESSIP_OFF=1 python benchmarks/df_hess_anatomy.py ...        # the baseline
```

The switches are `fastdfhessip`'s (listed at the top of
`electrolites/fastdfhessip.py`) and mean the same thing here, plus
`FASTDFHESSIP_H1ATOM=auto|occ|batched|upstream` and
`FASTDFHESSIP_H1OCC_SCORE` for item 12.  `OCC_BATCH`, `METRIC`, `TRI`,
`T2SPH`, `H1METRIC` and `IP2_GOUT` have no unrestricted counterpart.
