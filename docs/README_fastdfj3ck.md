# The three-centre Coulomb kernels themselves

`docs/DF_WORK_LIST.md` **W19**, found by W9.  After W9 the Coulomb build of a
pure density-fitted functional is **85 % two integral kernels at 42 atoms and
97 % at 284** (`docs/README_fastdfj3c.md` §3), both out of GPU4PySCF's
`gvhf-md/contract_int3c2e.cu`:

    contract_int3c2e_dm      rho_P = sum_ij (ij|P) D_ij
    contract_int3c2e_auxvec  J_ij  = sum_P  (ij|P) c_P

The entry named three things inside them — six divisions and a square root per
(pair, auxiliary shell), a Hermite index table read from global memory per
term because only `lk` is a template parameter, and a run-time `omega == 0`
test on the innermost path — and a falsification criterion: *profile one class
first; if the low classes are bandwidth-bound on the Hermite density, the
generator is not worth writing.*

They are not, and it was.  `electrolites/codegen/gen_j3c.py` writes both
kernels again from the angular-momentum class alone, and the Coulomb build is
**1.3x–5.0x faster than W9's** on every system measured — 1.3x at ten atoms,
where the call is mostly host work, and 3.4x–5.0x from 42 atoms up — and
**2.8x–27x faster than GPU4PySCF's**, with no approximation.

---

## 0. Summary

*(A100-SXM4-80GB of a whole Perlmutter GPU node, GPU4PySCF 1.8.1, def2-SVP
and its jkfit set unless another basis is named; variants round robin in one
process at a fixed density — `docs/DF_WORK_LIST.md` §5.)*

`benchmarks/df_j3c_bench.py`, one `df_jk.get_j`, milliseconds:

| | GPU4PySCF | W9 | **W19** | W19 against W9 |
|---|---|---|---|---|
| glycine, 10 atoms | 3.265 | 1.533 | **1.165** | 1.32x |
| alanine dipeptide, 22 | 7.149 | 3.318 | **1.611** | 2.06x |
| penicillin V, 42 | 26.205 | 14.706 | **4.271** | 3.44x |
| taxol, 113 | 157.288 | 139.677 | **29.114** | 4.80x |
| PfPMT, 284 | 732.056 | 712.174 | **143.670** | 4.96x |
| penicillin V, def2-TZVPP | 69.336 | 53.559 | **13.720** | 3.90x |
| Ni dithiolene, 21, doublet | 317.743 | 24.028 | **11.688** | 2.06x |
| Cu macrocycle, 60, quartet | 418.816 | 62.027 | **19.311** | 3.21x |
| penicillin V, two densities | 130.795 | 29.389 | **8.540** | 3.44x |
| penicillin V, four densities | 155.390 | 56.771 | **15.254** | 3.72x |

The largest relative difference against GPU4PySCF anywhere in the table is
7.5e-13, on the open-shell complexes, and it is the same with W9's bindings
(7.4e-13): that is the MD engine against the Rys one that GPU4PySCF's
multi-density branch uses, not this round.  Against GPU4PySCF's *own* MD
kernels the worst is 3.5e-13 (taxol); `tests/test_dfj3c.py` holds the whole
Coulomb matrix to 1e-12 and every SCF to 1e-9 Eh.

* **The margin grows with the molecule**, the opposite of W9's: W9 removed
  host work and the kernels took over as the molecule grew; this removes
  kernel work, and the kernels *are* what grows.  Measured on the two kernels
  alone (`benchmarks/df_j3c_kernel_sweep.py`), the density contraction is
  **4.9x, 6.0x and 6.5x** GPU4PySCF's at 42, 113 and 284 atoms and the
  auxiliary-vector one **4.4x, 4.9x and 4.8x** W9's split version of
  GPU4PySCF's.
* **It generalises across chemistry**: a 3d complex whose auxiliary set
  reaches `i` on the metal and a 60-atom Cu(II) quartet are 2.1x and 3.2x on
  top of W9's already-large multi-engine win, and def2-TZVPP (orbital `f`,
  `lij` up to 6) is 3.9x.
* **A whole DF-PBE SCF**: penicillin V **1.11x** over W9 (2.108 → 1.894 s),
  **1.14x** against a `patch_all()`-style baseline, taxol **1.34x**
  (7.582 → 5.644 s), PfPMT **1.52x** (85.1 → 56.0 s for the same 50
  cycles), the Cu macrocycle 1.09x — same energies to 1.8e-12 Eh,
  same cycle counts (§4).
* **Nothing about the tuning is size- or system-dependent, measured.**  One
  compiled configuration per class chosen per system by sweeping — thread
  count, auxiliary shells per block, occupancy target — composes to *no*
  better than the defaults (1.03x at 42 atoms, 0.92x at 113, 0.97x at 284 on
  the density kernel), so the defaults ship and there is no table (§3).

---

## 1. The profile first

`benchmarks/df_j3c_class_probe.py` gives each class its own time by
launching GPU4PySCF's kernels restricted to it (the density kernel by pair
group and a prefix of the auxiliary shells, the auxiliary-vector kernel by
both block lists).  On penicillin V the time is spread over every class — no
class is more than 8 % — and the time per quartet climbs from 0.07–0.10 ns
at `(ss|s)` to 1.5–6 ns at order 8.  At `(ss|s)` the useful work is one Boys
function and one multiply-add: 70 ps per quartet is an order of magnitude past
an A100's double-precision rate for that, and the Hermite density a quartet
reads is served from L2 (8 MB at 113 atoms, re-read once per auxiliary shell).
It is instruction-bound, which is what the criterion needed to know.

---

## 2. What the kernels change

One thread carries one shell pair through a batch of auxiliary shells with
everything in registers.  For each `(lij, lk)`:

1. **The pair data is read, not recomputed.**  Upstream evaluates
   `(ai ri + aj rj)/aij` (three divisions) for every (pair, auxiliary shell),
   and then `aij ak/(aij+ak)` and `ck/(aij ak sqrt(aij+ak))`.  Here the
   product centre, `aij` and `1/aij` are a table built once per
   `Int3c2eOpt` (`electrolites/_j3ck.py`) — with the same expression upstream
   evaluates, so the centre is bit-identical — `1/ak` is one per auxiliary
   shell, and one `rsqrt(aij+ak)` supplies both remaining reciprocals.  No
   division is left.
2. **Every Hermite index is a literal.**  The recurrence (the order
   `gen_j_high.rt_recurrence` emits, which is GPU4PySCF's), the contraction
   and the Hermite-to-Cartesian step are straight-line code; upstream's
   `Rt[p1_ij[k*nf3ij+i]*nsp]` — a `uint16` load, a multiply-add, then the
   double — is gone, and with it the `c_Rt_idx` tables of `iter_Rt_n`.
3. **`omega == 0` is compiled in**, through `fastj_prologue.cu`'s register
   Boys function.  A range-separated Coulomb call (`mol.omega != 0`) keeps
   GPU4PySCF's kernels.
4. **Hermite components that are zero are never formed.**  A Cartesian
   Gaussian of angular momentum `L` has Hermite components only at
   `e+f+g = L, L−2, …`; upstream's density contraction accumulates all
   `nf3(L)` and its `_dot_Et` then reads the right-parity ones, so for `L = 1`
   a quarter and for `L = 2` three tenths of that contraction multiplies by
   zero.  The phase `(-1)^(e+f+g)` becomes one sign per class.
5. **No shared memory and no barrier in the quartet loop.**  Upstream's
   threads cooperate on a pair's `Rt` through shared memory with a
   `__syncthreads` per recurrence level; here one thread owns the tensor.
   That is right for the low classes and turned out right for the high ones
   too: a thread's `Rt` spills past order ~6 and the spilled kernel is still
   ahead — at order 7–8 the auxiliary-vector classes on penicillin V are
   7.7x–10.2x upstream's alone (login-node measurement, `--ours`).
6. **Classes share a launch.**  Each class is a device function, classes of
   similar order are grouped into six *tiers*, and each tier is one kernel
   that reads its blocks' `(class, pair range, auxiliary range)` from a
   table and switches on the class, uniform over the block.  A build is a
   handful of launches on a pool of four streams; one launch per class — the
   first version — was 2x slower than GPU4PySCF at ten atoms, because a
   class's grid can be a handful of blocks (W9's §8 measured the same thing
   at 0.42x).

The shell-pair groups that meet a class past `lij + lk = 10`
(`FASTDFJ3C_KMAX_ORDER`) stay on GPU4PySCF's kernels, launched with the
offsets of those groups alone; in the table above that is the `lij = 5, 6`
groups of the two 3d complexes (14 pairs out of 15 539 on the Ni one).  The
ceiling is there for the compile time: the kernels are generated and
compiled on first use and cached, **8–10 s** for a def2-SVP organic basis
(25 classes) and **~22 s** for def2-TZVPP or a 3d metal (35), once per
machine.

---

## 3. The launch configuration, swept and not kept

`benchmarks/df_j3c_kernel_sweep.py` compiles one module per `(auxiliary
shells per block, occupancy target)` pair, times every class alone inside its
tier, chooses per tier the occupancy whose classes' best times sum lowest and
per class the batch that is best there, and then times the whole build both
ways:

| | GPU4PySCF | defaults | per-class choice |
|---|---|---|---|
| penicillin V, `contract_dm` | 7.133 ms | **1.465** (4.87x) | 1.418 (5.03x) |
| penicillin V, `contract_auxvec` | 5.909 (W9) | **1.359** (4.35x) | 1.351 (4.38x) |
| taxol, `contract_dm` | 75.356 | **12.585** (5.99x) | 13.617 (5.53x) |
| taxol, `contract_auxvec` | 59.982 | **12.209** (4.91x) | 13.307 (4.51x) |
| PfPMT, `contract_dm` | 395.219 | **61.015** (6.48x) | 62.866 (6.29x) |
| PfPMT, `contract_auxvec` | 296.640 | **61.911** (4.79x) | 58.495 (5.07x) |

Per-class optima measured alone do not compose — the classes share the card
when the build runs — and the defaults (128 threads, one auxiliary shell per
thread and batch in the density kernel where the register budget says so,
sixteen in the auxiliary-vector one) are within 6 % of the choice everywhere
and ahead of it at 113 atoms.  For the auxiliary-vector kernel 64 and 256
threads are both slower than 128 (2.70 and 2.99 ms against 2.41 summed over
classes at 42 atoms; 13.67 and 13.50 against 13.20 at 113).  The density kernel's pairs per block is flat from 1024 up
(12.55, 12.32, 12.25 ms at 1024/2048/4096 on taxol) and 2048 ships.  Every
configuration of the sweep agrees with GPU4PySCF to ≤ 2.3e-15.

---

## 4. Whole runs

`benchmarks/df_j3c_bench.py --scf --scf-variants w9,fast,inc`, each variant
twice in one process and the second quoted:

| DF-PBE | W9 | **W19** | | cycles | ΔE |
|---|---|---|---|---|---|
| penicillin V, 42 atoms | 2.108 s | **1.894 s** | 1.113x | 15 / 15 | 1.4e-12 |
| taxol, 113 | 7.582 | **5.644** | 1.343x | 17 / 17 | 1.8e-12 |
| Cu macrocycle, 60, quartet | 5.913 | **5.436** | 1.088x | 23 / 23 | 9.1e-13 |
| penicillin V, over `fastxc`/`fastnlc`/`fastxcnlc`/`fastdf` | 1.514 | **1.332** | 1.137x | 15 / 15 | 1.8e-12 |

| PfPMT, 284 atoms (40 GB card) | 85.118 | **55.974** | **1.521x** | 50 / 50 | 1.8e-10 |

The 284-atom run is DF-PBE on an anion with a small gap and does not converge
in 50 cycles with either binding; both do the same 50, so the row is 51
Coulomb builds each and its ratio is per build — 0.58 s of Coulomb work a
cycle removed from a 1.7 s cycle.

The Coulomb build is now a small part of a pure-functional SCF — at 113 atoms
17 builds of 29 ms are 0.5 s of a 5.6 s run — and that is what bounds the
whole-run number: what is left is the grid.

---

## 5. Correctness

`tests/test_dfj3c.py` compares against GPU4PySCF's own bindings in one process:
every W9 row (now 2e-15–4e-15 against the same engine, where W9's bit-identical
launches read 1.6e-16), plus three W19 rows — the Ni complex with every class
this module's (1.0e-14), the same with the ceiling at 8 so that the groups past
`lij = 2` go back to GPU4PySCF and both sets of kernels write one output
(8.3e-15), and ethanol with the ceiling at 6 (2.3e-15) — and the seven SCF
cases and the gradient to the same bars as before.  Per class, against
GPU4PySCF's kernel restricted to that class, the auxiliary-vector outputs agree
to 5e-16 at low order and 4e-12 at `(4,4)`, relative to the class's own largest
element — the high-order Hermite recurrences cancel, and two orderings of the
same sums round differently there.

---

## 6. Tools

```bash
python tests/test_dfj3c.py [--basis def2-tzvpp]
python benchmarks/df_j3c_class_probe.py --cases penicillinV [--ours]
python benchmarks/df_j3c_kernel_sweep.py --cases penicillinV,taxol
python benchmarks/df_j3c_bench.py --cases ... --variants stock,w9,fast
python benchmarks/df_j3c_bench.py --cases ... --scf --scf-variants w9,fast
python -m electrolites.codegen.gen_j3c > j3c.cu       # the generated source
```

Switches: `FASTDFJ3C_KERNELS=upstream` (GPU4PySCF's kernels for every
class), `FASTDFJ3C_KMAX_ORDER=<n>` (the ceiling), `FASTDFJ3C_KSTREAMS=<n>`,
`FASTDFJ3C_KDM_PPB=<n>`.
