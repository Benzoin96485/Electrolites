# The unrestricted h1 term on a 40 GB card: the occupied index in batches, and dm_3c on the host

`docs/DF_WORK_LIST.md` **W38**, found by W35.  At 113 atoms the unrestricted
twin of the density-fitted Hessian's `h1` term, `_dfhessip_u._get_veff`,
asserted "Insufficient GPU memory" on a 40 GB card before it sized a block:
its budget holds `dm_3c` and `dm_oo` for both spins and every occupied
orbital, `2 naux nocc (nao + nocc)` words, about 32 GB on taxol⁺.  The
restricted function batches the occupied index when that does not fit
(`num_occ_batches`); upstream's twin has no such loop.  The entry asked for
that loop, measured on taxol⁺ and the Co(II) sextet at a 40 GB card's
memory; success was the call running there within about 1.3x of its 80 GB
time; the falsification, the occupied batches repeating the integral passes
enough to cost more than the memory-short branch at 54.6 GB.

**It runs, at 1.11x of the full card on taxol⁺ and 1.14x on the Co(II)
sextet, and 1.25x on the call that pays the host buffer's page-locking.**
The restricted function's loop, transplanted, is not what does it: it runs
too (1.43x–1.50x), but stages D and E contract the batch's occupied index
away, so batching them repeats their integrals and D2's products once per
batch.  What does it is item 23's default: stages A–C batched, each batch's
`dm_3c` copied into page-locked host memory behind stage C, and stages D and
E run **once**, reading `dm_3c` back one auxiliary batch at a time behind
the previous batch's work.

---

## 0. Summary

* **Item 23** (§1): `_dfhessip_u._get_veff` batches the occupied index when
  both spins' density factors would take more than 70 % of what the metric
  leaves (the restricted function's rule, with the spin index).  Stages A–C
  (the `(00|0)` integrals contracted with the batch's orbitals, the metric,
  the `j2c_10` term) run per batch; `dm_oo[r,n,i,j]` is kept whole on the
  device (`2 naux nocc²`, 5.4 GB at taxol⁺), `dm_3c[r,n,i,q]` (27 GB) goes
  to the host.  Stages D and E run once over every orbital, each auxiliary
  batch's rows of `dm_3c` fetched on a stream of their own into one of two
  device buffers while the previous batch computes.
* **taxol⁺ doublet, 113 atoms** (§2): 40 GB card (ballast 40 before the
  SCF, 27.6 GB left for the call), 4 occupied batches: **33.5 s** against
  **30.1 s** on the full card (**1.11x**); the call that page-locks the
  27 GB host buffer, which a Hessian pays once, **37.6 s (1.25x)**.
  Upstream's single batch (`FASTDFHESSIP_H1UBATCH=0`, i.e. W35):
  `Insufficient GPU memory`.
* **Co(II) sextet, 113 atoms** (§2): its UKS SCF does not fit a 40 GB card
  at all (W35), so the card's memory is withheld after the SCF: at the
  27.6 GB taxol⁺'s call gets, **26.4 s** against **23.1–23.8 s**
  (**1.11x–1.14x**; first call 28.8 s, 1.21x–1.24x), where W35's code
  asserts; at 44.6 GB left it fits one batch and is 24.7 s (1.04x–1.07x).
* **The restricted function's layout** (`=occ`, §3): D and E per occupied
  batch as well, 43.0 s on taxol⁺ (1.43x) and 34.7 s on the Co sextet
  (1.50x) — D2 is 11.3 s instead of 3.9 because it is the batch's `s` and
  `t` against every orbital's `j3c C_o`, formed once per batch.  It passes
  the success bar only by the entry's word "about", and it is the one the
  falsification describes; kept as a switch.
* **The falsification does not fire** (§3): the repeated integral passes are
  stage A's, 0.9–1.0 s per extra batch; nothing else is repeated, and at
  half the memory the call is faster than the memory-short branch at
  54.6 GB (75.6 s before W35, 34.6 s after).
* **The copies' overlap** (§2.2): synchronously the copies cost 2.4 s
  (`C dm_3c out` 1.11 s, `D dm_3c in` 1.28 s on taxol⁺); on their own stream
  0.35 s.  Two things had to be right for that: the copy out is issued after
  `j2c_10` is formed (its sort frees a copy of it, and a `cudaFree` waits for
  every stream), and the fetches wait for the null stream once before the
  first one — the pool's blocks are stream-ordered on the null stream only,
  and without that wait a def2-TZVPP test read a buffer's previous owner
  (1.5e-2 error, §4).
* **Full card, nothing moves** (§5): one batch, the same arithmetic as W35;
  five open-shell systems from 5 to 60 atoms and a def2-TZVPP case, this
  branch against `main` in separate processes, 1.00x–1.04x, and against
  GPU4PySCF 1.6e-15 – 1.8e-14.
* **Correctness** (§4): element-wise against the unbatched call on the same
  SCF 1.1e-13 (Co sextet) and 1.3e-13 (taxol⁺); every branch of the call,
  forced into 2–3 batches on ethanol⁺ and the triplet ethanol²⁺, in all
  three modes, at most 4.1e-15 against GPU4PySCF; the whole ethanol⁺ UKS
  Hessian 3.4e-12 Eh/bohr² against unpatched.

Everything is one A100-SXM4-80GB of a whole Perlmutter node (`--qos=debug`,
one GPU visible), GPU4PySCF 1.8.1, def2-SVP and its jkfit set, B3LYP,
level-3 grids, `patch_all()`.  Function times are
`benchmarks/df_hess_terms_probe.py`, variants round robin in one process,
the minimum of two after a warm call, with `FASTDFHESSIP_TIME=1`
(synchronised stage timers).

---

## 1. What changed

`electrolites/_dfhessip_u.py:_get_veff` (item 23 of `fastdfhessip`), and
nothing in the restricted function or in `_jk_energy_per_atom`.

**The decision.**  With `W = ` the words free after the metric,

    num_occ_batches = int(2 naux nocc (nao + nocc) / (0.7 W) + 1)

— the restricted rule with a factor 2 for the spins.  On a full card at 113
atoms this is 1 (taxol⁺: 4.0e9 words of factors against about 0.7 × 9e9
free), so every call
that ran before takes the same path, sized the same way, and stays
bit-for-bit the arithmetic of W35.  On a 40 GB card it is 2, and then the
occupied batch itself is sized from what the D loop leaves (below), which
gave 4 batches on taxol⁺ and 3 on the Co sextet.

**What each stage needs of the occupied index.**  In the unrestricted h1
term every intermediate carries the spin `n` and an occupied pair `(i, j)`,
and the exchange terms are of two kinds: those whose output index is `i`
(the Coulomb halves: `vhf_atm[..., i, q] += dm_3c[r,n,i,q] ...`), which a
batch writes into its own rows, and those summed over `i` with output `j`
(every exchange product: `sum_i dm_3c[r,n,i,q] dm_oo[r,n,i,j]`, the per-atom
AO matrices, D2's `s = C_o dm_oo` and `t = C_o dm_3c`), which accumulate over
batches.  Two quantities need every `i` at once and are formed from the
total density instead, as the restricted function does: the Coulomb vector
`auxvec = sum_i dm_oo[r,n,i,i]` (= `V W^-1 V^T (j3c . D)`) and its
derivative `sum_i j3c_aux[x,r,n,i,i]` (= `-(dj3c . D)`).

**Two layouts.**

* `FASTDFHESSIP_H1UBATCH=occ` is the restricted function's: every stage per
  occupied batch.  D2's second product, `sum_{u in A, r} pri[n,j,u,r]
  t[n,u,r,q]`, needs `pri = j3c C_o` over every `j` while `t` holds the
  batch's `i`, so each batch forms `pri` twice — its own for D1 and the
  whole one for D2 (the restricted function's `pri_full`) — and D's and E's
  integrals are evaluated once per batch.
* `host` (the default): stages A–C per batch, and each batch's `dm_3c`
  (`naux × 2 × nob × nao`) copied into its columns of one page-locked host
  array `(naux, 2, nocc, nao)` with one `cudaMemcpy2DAsync` (each `(r, n)`
  row is a contiguous run of `nob nao` doubles going into a row of
  `nocc nao`).  Its `dm_oo` goes into a device array for every orbital.
  Stages D and E then run once, exactly as the unbatched call, except that
  `dm_3c[aux0:aux1]` — the only way either stage reads it — is a device
  buffer holding that auxiliary batch's rows, fetched from the host.

The host array comes from `_dfpin.empty_mapped_pooled` (W12's pool: ordinary
memory, `cudaHostRegister`ed, kept for the next request of the same size),
27 GB on taxol⁺ and 21 GB on the Co sextet; its page-locking is the one cost
a single call cannot hide (§2.1).

**The copies' stream** (`FASTDFHESSIP_H1UHOST_ASYNC=0` turns it off).  The
copy out is issued on a second stream after the batch's `j2c_10` is formed
and runs behind stage C, which only reads `dm_3c`; the batch waits for it at
the end of C.  In D, auxiliary batch `k + 1`'s rows are fetched into the
second of two device buffers while batch `k` computes, each fetch waiting
on an event recorded when the batch that last read that buffer finished E.

**Sizing in host mode.**  D's budget holds `dm_oo` whole and, per auxiliary
function of the integral batch, two rows of `dm_3c` (`2 × 2 nocc nao`) next
to W35's `6 max(nao_pair, 2 nocc nao)`; the block and the item-21 minimum
are then sized as before.  The occupied batch of A–C is what that budget
leaves at 90 %, less stage A's two buffers, per `2 naux (nao + nocc)`
words.  `FASTDFHESSIP_OCC_BATCH=<n>` forces `n` orbitals per batch (and
forces batching), as in the restricted tests.

---

## 2. The 40 GB card

### 2.1 taxol⁺ and the Co(II) sextet

`df_hess_terms_probe.py --terms h1 --variants ours --reps 2 --env
'sync:FASTDFHESSIP_H1UHOST_ASYNC=0,occ:FASTDFHESSIP_H1UBATCH=occ'`; the
40 GB card for taxol⁺ is `--ballast 40` (a 40 GB array held from before the
SCF: 44.6 GB free, 27.6 GB of it left when the call starts); the Co(II)
sextet's SCF does not fit that, so `--free-after 27.6` withholds memory
after the SCF down to the same 27.6 GB, and `--free-after 44.6` down to the
card's total.  Full-card rows are the same probe without either, in a
separate process.

| system | memory for the call | layout | occupied batches | call | against 80 GB |
|---|---|---|---|---|---|
| taxol⁺ doublet, 113 atoms, nocc 226 | 80 GB card | one batch (W35, unchanged) | 1 | **30.09 s** | |
| | 27.6 GB (40 GB card) | W35 / `H1UBATCH=0` | — | `Insufficient GPU memory` | |
| | | **host, async (default)** | 4 | **33.53 s** | **1.11x** |
| | | … its first call (page-locks 27 GB) | 4 | 37.64 s | 1.25x |
| | | host, synchronous copies | 4 | 35.70 | 1.19x |
| | | `occ` (restricted layout) | 2 | 42.98 | 1.43x |
| Co(II) sextet, 113 atoms, nocc 207 | 80 GB card | one batch | 1 | **23.12–23.76 s** | |
| | 44.6 GB | one batch (fits) | 1 | 24.73 | 1.04x–1.07x |
| | 27.6 GB | W35 / `H1UBATCH=0` | — | `Insufficient GPU memory` | |
| | | **host, async** | 3 | **26.36** | **1.11x–1.14x** |
| | | … its first call | 3 | 28.76 | 1.21x–1.24x |
| | | host, synchronous | 3 | 27.66 | 1.16x–1.20x |
| | | `occ` | 2 | 34.70 | 1.46x–1.50x |

(The Co range is the two full-card runs, one per job; the 80 GB call is
also the reference the 27.6 GB rows were checked against, §4.)

In a Hessian `h1` is called once (twice for a range-separated functional),
so the first-call row is the one a 40 GB Hessian pays: the page-locking is
3.4 s for 27 GB (8 GB/s, W12's rate) and then pooled.  Both first-call rows
are inside the entry's 1.3x.

### 2.2 Where the 3.4 s goes

The stage tables, taxol⁺, second call, full card against 40 GB host-async:

| stage | 80 GB | 40 GB | |
|---|---|---|---|
| A `j3c_00` | 1.48 s | 4.07 | the `(00|0)` integrals evaluated per occupied batch: 4 × |
| B metric | 4.35 | 4.38 | |
| C (`gemm` + `K` + `J`) | 9.03 | 9.26 | linear in the batch; the grouped GEMMs are the same |
| C `dm_3c out` | — | 0.007 | synchronously 1.11 |
| D `dm_3c in` | — | 0.34 | synchronously 1.28 |
| D integrals | 1.54 | 1.09–1.26 | |
| D1 + D2 + D3 | 8.54 | 8.90 | block 48 against 88 |
| E fill + per-aux-atom | 3.99 | 4.15 | |
| setup, F | 1.10 | 1.07 | |
| **total** | **30.09** | **33.53** | |

Stage A's integral passes are the difference (2.6 s of 3.4), then the
smaller block (0.4 s) and the fetch that is not hidden (0.34 s).  There is
no residue to attribute: the stage timers are synchronised and add up to the
call to within 0.1 s, so no profile was needed to find the gap.  (A kernel-
level profile would show the same A rows as four evaluations of the same
integral batches.)

The asynchronous copies were first measured in a version that kept one
`j2c_10` for all batches so that stage C would free nothing large during the
copy: that cost 1 GB of the A–C budget, five occupied batches instead of
four, and one more stage A pass — 35.07 s against 35.23 s synchronous.
Issuing the copy after `j2c_10` is formed instead gives the same overlap at
four batches (§2.1).  Registering the host buffer on a host thread behind
stage A was also tried in isolation and rejected: `cudaHostRegister` of
10 GB beside 40 cuBLAS GEMMs made the GEMMs 2.7x slower (2.53 → 6.91 s) —
the driver serialises them — which is worse than paying it up front.

---

## 3. The falsification

"The occupied batches repeat stage A/B's integral passes enough to cost more
than the memory-short branch at 54.6 GB."  W35's §3 measured that branch on
taxol⁺ at 54.6 GB free: 75.6 s before W35, **34.6 s** after it.  The host
layout at 27.6 GB — half that memory — is 33.5 s, and what it repeats is
stage A only: 0.86 s per extra occupied batch on taxol⁺, 1.0 s on the Co
sextet.  It does not fire.

The restricted function's layout is where the entry's worry is real.  With
D and E per occupied batch, D2 is 11.3 s on taxol⁺ instead of 3.9: its first
product `sum_u j3c[u,q,r] s[n,u,r,j]` has `s` summed over the batch's `i`
and is cheap to split, but the second, `pri[n,j,u,r] t[n,u,r,q]`, contracts
every `j` of `pri = j3c C_o` against the batch's `t`, so the whole `pri` and
the full-size GEMMs are formed once per batch — and so are D's and E's
integrals (D integrals 2.5 s against 1.1).  That is why `=occ` is 43.0 s,
and why the default keeps D and E whole and moves `dm_3c` instead: at
27 GB, 26 GB/s of PCIe each way is 2 s, of which 0.35 s is not hidden.

---

## 4. Correctness

`tests/test_dfuhessip.py` (530 checks with `--no-hessian`, all pass):

| | worst (relative to the largest element) |
|---|---|
| h1, occupied batches forced (`FASTDFHESSIP_OCC_BATCH=8, 5`: 2 and 3 batches) × host async / host synchronous / `occ`, on ethanol⁺ (doublet, `nocca − noccb = 1`) for B3LYP, HF, `k = 0`, and the long-range operator; on the triplet ethanol²⁺, whose padded beta columns fall in the last batch; every h1 branch (`H1ATOM=batched|upstream|occ`, `H1D2=atom`, `H1KAO=0` with and without `H1D2=atom`/`H1C=atom`, `H1C=atom`, `H1J=upstream`, `K0=compute`, `H1BLK=24`) | **4.1e-15** |
| `H1UBATCH=0` with `OCC_BATCH=5` (one batch, asserted) | 8.7e-16 |
| the rest of the file, unchanged | as before |
| whole ethanol⁺ UKS B3LYP Hessian, patched against unpatched | 3.4e-12 Eh/bohr² |

At 113 atoms, on the same SCF (`--prefree-env FASTDFHESSIP_H1UBATCH=0`: the
unbatched call made before memory is withheld): the Co sextet's three
layouts **1.1e-13 – 1.2e-13**; taxol⁺'s synchronous host layout and `occ`
layout 1.3e-13 (an earlier job, `el_w38a-59019236.out`), and the three
layouts agree with each other to 1.0e-13 in the §2.1 run.
(Against a saved output from another process the difference is 7e-10 —
that is the SCF's own convergence, not the call.)

**A race the tests caught.**  The first asynchronous version passed in
def2-SVP and failed on the triplet in def2-TZVPP, one call, 1.5e-2: the
device buffers for the fetches come from CuPy's pool, which orders a
block's reuse on the null stream only, and the first two fetches — on the
copy stream, with nothing to wait for — wrote into a block whose previous
owner's null-stream kernels had not finished.  One event before them fixes
it; the def2-TZVPP run then passes every item-23 row.  (Its other failure,
"the caps reached the batched paths" missing `h1 mem_insufficient` and `h1
rebalanced`, is the memory ladder not reaching those branches at def2-TZVPP
sizes, and is the same on `main`: `$SCRATCH/w38/test_u_main_tz.log`.)

---

## 5. On the full card

Nothing batches there (`occ_batches_u = 1` on every system), and the code
path is W35's.  Measured anyway, h1, B3LYP, this branch against GPU4PySCF in
one process and `main` in a separate one (`$SCRATCH/w38/f.sh`):

| system | GPU4PySCF | main | this branch | max rel. vs GPU4PySCF |
|---|---|---|---|---|
| FeCl₄⁻ sextet, 5 atoms | 244.5 ms | 192.6 | 190.5 | 1.8e-14 |
| Ni(II) dithiolene⁻ doublet, 21 | 1551.6 | 989.0 | 977.4 | 1.7e-14 |
| Pt(II) amine triplet, 23 | 951.1 | 494.4 | 485.6 | 1.6e-15 |
| Cu(II) macrocycle²⁺ quartet, 60 | 4835.6 | 2172.4 | 2165.7 | 1.6e-14 |
| penicillin V⁺ doublet, 42, def2-TZVPP (one rep) | 9948.6 | 4876.0 | 4686.0 | 2.0e-15 |

All within the spread of separate processes (1–4 %).

---

## 6. What is left, and what was not done

* **The unrestricted ip2 term at 40 GB is 3.3x its full-card time.**  With
  h1 fixed, a 40 GB UKS Hessian at 113 atoms reaches `_dfhessip_u.
  _jk_energy_per_atom`: taxol⁺ runs, but in **195.3 s** against **58.6 s**
  on the full card (`$SCRATCH/w38/el_w38h-*.out`, `el_w38j-*.out`; one
  call each, no stage table) — 5x the h1 term at the same memory.
  Proposed as W40.
* **Stage A's integral passes.**  2.6 s of item 23's 3.4 s at 113 atoms is
  the `(00|0)` integrals evaluated once per occupied batch.  Keeping one
  auxiliary batch's `j3c` block and applying every occupied batch to it
  would need the batches' `j3c_00` resident together, which is what did not
  fit; with `dm_3c` already on the host, stage A could write its output
  there too, and one pass would do.  Worth 2–3 s of 34 at 40 GB, nothing on
  the full card.  Not done.
* **Whole Hessians at 40 GB were not run**: the Co sextet's SCF does not
  fit there, and taxol⁺'s would take its ip2 term's 195 s and a CPHF nobody
  has sized at 40 GB, past a debug job's 29 minutes.  The ethanol⁺ whole
  Hessian checks the default path; the forced-batch rows check item 23.

---

## 7. Tools

```bash
python tests/test_dfuhessip.py [--no-hessian] [--basis def2-tzvpp]
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_terms_probe.py --patch all \
    --terms h1 --variants ours --reps 2 --ballast 40 \
    --xyz benchmarks/molecules/taxol.xyz --charge 1 --spin 1 \
    --basis def2-svp --xc b3lyp \
    --env 'sync:FASTDFHESSIP_H1UHOST_ASYNC=0,occ:FASTDFHESSIP_H1UBATCH=occ'
FASTDFHESSIP_TIME=1 python benchmarks/df_hess_terms_probe.py --patch all \
    --terms h1 --variants ours --reps 2 --free-after 27.6 \
    --prefree-env 'FASTDFHESSIP_H1UBATCH=0' \
    --xyz benchmarks/molecules/tmc_co_sextet.xyz --spin 5 \
    --basis def2-svp --xc b3lyp --env '...'
```

Switches: `FASTDFHESSIP_H1UBATCH=host|occ|0` (item 23's layout: `dm_3c` on
the host with D and E once, the default; D and E per occupied batch; or
upstream's single batch), `FASTDFHESSIP_H1UHOST_ASYNC=0` (the host copies
synchronous), `FASTDFHESSIP_OCC_BATCH=<n>` (test-only: `n` orbitals per
batch, batching forced).  New in the probe: `--free-after <GB>` withholds
memory after the SCF, down to that much free (for a molecule whose SCF does
not fit the card being modelled); `--prefree-env '<VAR=val;...>'` makes each
term's reference the module's call with those switches, on the same SCF,
before the memory is withheld; `--save-out`/`--ref-out` save and compare a
term's result across processes.
