# Pair-atomic exchange: the contraction order, and what it is worth

`docs/DF_WORK_LIST.md` **W6** ends on one sentence, and this round is about
it:

> Robust pair-atomic **exchange does not factorise** the way `vk = rhok
> rhok^T` does: the coefficients depend on the pair, so the density cannot be
> contracted first, and that identity is what the whole of `fastdf` rests on.
> PARI-K is a different contraction order and estimating its cost is the next
> piece of work.

Estimating it needed an implementation, because the cost turns on a quantity
the pair list alone does not give.  So this round is `electrolites/dflocal.py`
— a robust pair-atomic J and K build that never forms a four-index tensor —
with `benchmarks/df_local_k_probe.py`, `benchmarks/df_local_k_bench.py` and
`tests/test_dflocal.py`.  `docs/README_dflocal.md` is the round before it,
which decided *robust* over non-robust and measured what a pair-atomic fit
would store.

**Five results, and the first is a correction to the sentence above.**

* **It does factorise, and the density can be contracted first.**  The robust
  expansion collapses to **one** contraction of the same shape as `fastdf`'s
  first one — `vk = M X + (M X)^T` with `M = B - T/2` folded once at build
  time — and in the second of the two orderings below, the first contraction
  is *identical* to global DF's.  What PARI replaces is `fastdf`'s **second**
  contraction, not its first.
* **What it is worth is set by one number.**  `nlambda`, the number of AOs
  that overlap an auxiliary function's own atom.  Global DF's exchange costs
  `4 nao^2 nocc naux` flops and PARI's costs `2 nao^2 naux nlambda`, so the
  ratio is `2 nocc / nlambda` — and measured on six systems that is
  **0.42x at 10 atoms, 1.09x at 284 and 1.36x at 545**, growing like
  **N^0.31**.  `nlambda` is the quantity W6 hoped would be flat and it is
  not: it grows like N^0.68 where `nlocal`, the size of a pair's *domain*,
  really is flat.  Unscreened, PARI-K passes global DF-K only at about 250
  atoms.
* **Half of `nlambda` is free.**  A `lambda` at the edge of the screening
  carries a fitting coefficient of 1e-8 and dropping it is an error in the
  exchange matrix, not in the fit.  Measured on the exchange matrix itself:
  at taxol a threshold of 1e-6 leaves **48 %** of `nlambda` and changes `K`
  by **1.2e-7** relative — against the **5.4e-4** that pair-atomic fitting
  is already away from global DF.  That is 4400 times inside the
  approximation's own error, and it is what makes `nlambda` flat after all:
  screened, it is **236, 407, 374, 465** from 42 atoms to 545, and the ratio
  is **2.93x at 284 atoms** and **4.35x at 545**.  Those are flop ratios;
  measured end to end at the last size where the tensor is device-resident,
  PARI's `get_jk` is **1.39x** stock global DF's at 113 atoms.  At 284 atoms,
  where both tensors are on the host, it is **2.0x slower** and the call is
  PCIe-bound on both sides — §4.1, and it is the strongest argument for the
  bullet below.
* **The storage saving is about the fit, and reaches the build only through
  the residual.**  A robust
  expansion needs `(mn|Q)` for **every** stored pair against **every**
  auxiliary function — not only the domain ones — so the tensor a PARI
  exchange build touches is the same 86.4 GB at 284 atoms that global DF
  stores.  The escape is to store the fit *residual* `R = B - cJ` and rebuild
  `T = cJ` per pass, and measured on the exchange matrix it works: at 113
  atoms a threshold costing **2.9e-5** relative on `K` — twenty times inside
  the approximation's own 5.4e-4 — leaves **0.45 GB against 14.31**, which is
  within 1.4x of the bound the coefficient count alone would give.  The
  container is not written; §2 says what it would take.
* **Robust pair-atomic fitting is size-extensive, and the size was never the
  question.**  `docs/README_dflocal.md` could take its homologous series to
  C4H10 because its reference builds an `nao^4` tensor; this one takes it to
  **C20H42, 62 atoms**, and the error per atom is **flat and falling** —
  1.90e-5, 2.51e-5, 2.49e-5, 2.22e-5 Eh from C1 to C20.  What does move is
  **composition**: the same quantity is 3.3e-5 on glycine, 6.2e-5 on
  penicillin V and **7.4e-5 on taxol at 113 atoms**, a factor of three across
  the chemistry at a fixed claim of size-extensivity.  W6's success criterion
  asks for the §3.2 ladder; this is it, and the number to quote is a range
  and not a constant.

---

## 0. The algebra, and why there is only one contraction

Write `rho_mn` for an AO pair density and `rho~_mn` for its fit inside the
pair's own domain.  The robust (Dunlap) integral is

    (mn|ls) ~ (mn|rho~_ls) + (rho~_mn|ls) - (rho~_mn|rho~_ls)

whose error is second order in the fit residual where the non-robust
`(rho~_mn|rho~_ls)` alone is first order — which is the whole of
`docs/README_dflocal.md` §2.1, where the non-robust form collapses
variationally by 16.5 Eh on butane.  With `B[mn,Q] = (mn|Q)` the exact
three-centre integrals, `c[mn,P]` the local fitting coefficients and `J` the
auxiliary Coulomb metric, that expansion is

    eri = B c^T + c B^T - T c^T,        T = c J

The third term is **symmetric**, so it may be split evenly between the two
halves of the first two.  Define

    M = B - T/2

and every working form follows in one line.  With `D` the density,
`d = c^T D`, `a = M^T D` and `X[n,Q,l] = sum_s c[ls,Q] D[n,s]`:

    vj = M d + c a
    vk = M X + (M X)^T

**`T` is never formed at run time** — the fold happens once, in the build, in
the same pass that evaluates the integrals — and the exchange is **one**
contraction where global DF needs two.  `tests/test_dflocal.py` checks both
against a dense four-index reference assembled independently in numpy, at
**4e-13** relative on `vk` and **1e-11** on `vj`.

### 0.1 Two orderings, and which one wins where

`X` can be built from the density or from its occupied factor, and the two
give different costs.

**AO order** (`DFLOCAL_KMODE=ao`).  Contract the coefficients with the density
first, which is what the expression above says: `X` is `(nao, naux, nlambda)`
and the contraction `M X` is

    2 nao^2 naux nlambda        flops, one pass

**MO order** (`DFLOCAL_KMODE=mo`).  With `D = C C^T`,

    vk[m,l] = sum_{i,Q} rhok[m,i,Q] G[i,Q,l],
        rhok[m,i,Q] = sum_n M[mn,Q] C[n,i]
        G[i,Q,l]    = sum_s c[ls,Q] C[s,i]

and `rhok` is **exactly the intermediate `fastdf` builds**, on a different
tensor.  So this order leaves global DF's first contraction alone and replaces
only its second:

    2 nao^2 naux nocc  +  2 nao nocc naux nlambda      flops

`vk = rhok rhok^T` — `fastdf`'s item 1, the `dsyrk` — becomes `rhok G`, which
is `nao / nlambda` times cheaper.  **So W6's sentence is wrong in the useful
direction**: the density *can* be contracted first, and what does not
factorise is the second contraction, not the first.  The two orders cross at

    nlambda = nocc / (1 - nocc/nao)

which is 686 of 1004 at 284 atoms: the MO order is the faster one at every
size measured, the AO order is the asymptotic one (its ratio to global DF
grows like N^0.31 where the MO order's saturates at 2x), and `auto` compares
the two flop counts per call.

---

## 1. What it costs

`benchmarks/df_local_k_probe.py`, no integrals and no GPU: `nlambda` is a
property of the screened pair list and the auxiliary basis, weighted by how
many auxiliary functions each atom carries.  def2-SVP with its own jkfit set.

| | atoms | nao | naux | nocc | `nlambda` | `nl/nao` | global-K | PARI-K | |
|---|---|---|---|---|---|---|---|---|---|
| glycine | 10 | 95 | 471 | 20 | 95 | 1.00 | 0.0 TF | 0.0 TF | 0.42x |
| alanine dipeptide | 22 | 200 | 974 | 39 | 200 | 1.00 | 0.0 | 0.0 | 0.39x |
| penicillin V | 42 | 430 | 2 191 | 92 | 361 | 0.84 | 0.1 | 0.3 | 0.51x |
| taxol | 113 | 1 123 | 5 598 | 226 | 853 | 0.76 | 6.4 | 12.0 | 0.53x |
| PfPMT | 284 | 2 697 | 13 365 | 547 | 1 004 | 0.37 | 212.7 | 195.3 | **1.09x** |
| HcgC | 545 | 5 037 | 24 722 | 1 011 | 1 487 | 0.30 | 2 536.5 | 1 864.9 | **1.36x** |

```
 nlambda ~ N^0.68        nocc ~ N^0.99        ratio ~ N^0.31
```

The last column is the AO order against **stock** global DF.  Three things
follow.

**`nlambda` is not flat, and that is the whole story.**  The item's hope was
the one `nlocal` satisfies: 118 to 130 auxiliary functions per pair domain on
every system from 10 to 545 atoms (`docs/README_dflocal.md` §1).  `nlambda` is
a different quantity — how many AOs overlap an *atom*, not how many auxiliary
functions fit a *pair* — and it grows like **N^0.68** under def2-SVP's 1e-14
screening, against `nocc`'s N^0.99.  The ratio therefore grows like N^0.31,
so PARI-K passes global DF-K at about **250 atoms** and would need roughly
**2 400** atoms to reach 2x on this half.

**Below that it is a loss, not a wash.**  At 10 and 22 atoms every AO overlaps
every other, `nlambda = nao`, and the AO order does 2.4x global DF's
arithmetic.

**Against `fastdf` rather than stock the crossing moves out.**  `fastdf`'s
`dsyrk` halves global DF's second contraction above 800 AOs, so global-K is
`3 nao^2 nocc naux` there and the AO order's ratio is `1.5 nocc / nlambda` —
0.82x at 284 atoms.  PARI's own second contraction is *not* symmetric (it is
`rhok G`, not `rhok rhok^T`) so it cannot take the same factor back.  The MO
order does better against both: `3 nao / (2 nao + 2 nlambda)` = **1.09x**
against `fastdf` at 284 atoms and `2 nao / (nao + nlambda)` = **1.46x**
against stock.

### 1.1 Half of `nlambda` is free

`nlambda` is the number of AOs that overlap an atom under the *integral*
screening, which is a 1e-14 bound on a pair's own Schwarz norm.  The fitting
coefficient of a pair at that edge is 1e-8, and dropping it does not change
the fit at all — it changes the **exchange matrix**, by an amount that can be
measured rather than bounded.  `DFLOCAL_LAMBDA_TOL` is that threshold, off by
default, and `--lam` sweeps it and builds the exchange matrix at each value:

| | `tol` | `nlambda` | of full | `relK` vs unscreened PARI | `relK` vs global DF |
|---|---|---|---|---|---|
| penicillin V | 0 | 361 | 1.00 | — | 2.50e-4 |
| | 1e-10 | 305 | 0.84 | 1.4e-12 | 2.50e-4 |
| | 1e-8 | 274 | 0.76 | 1.9e-10 | 2.50e-4 |
| | 1e-6 | 236 | **0.65** | 7.8e-8 | 2.50e-4 |
| | 1e-4 | 181 | 0.50 | 8.2e-6 | 2.50e-4 |
| taxol | 0 | 853 | 1.00 | — | 5.42e-4 |
| | 1e-10 | 635 | 0.74 | 3.9e-12 | 5.42e-4 |
| | 1e-8 | 528 | 0.62 | 3.8e-10 | 5.42e-4 |
| | 1e-6 | 407 | **0.48** | 1.2e-7 | 5.42e-4 |
| | 1e-4 | 268 | 0.31 | 2.3e-5 | 5.42e-4 |

The last column does not move, which is the point: the pair-atomic
approximation's own distance from global DF is 2.5e-4 to 5.4e-4 on the
exchange matrix, and a threshold of 1e-6 sits **three to four orders of
magnitude inside it**.  The fraction kept also *falls* with size — 0.65 at 42
atoms, 0.48 at 113 — so the screened `nlambda` grows more slowly than the raw
one.

`--lam-fit` runs the same sweep on the fit alone.  It builds no tensor and
therefore no exchange matrix and no error column, but it runs where the
tensor is 86 GB and 414 GB and a contraction cannot:

| | atoms | `tol` | `nlambda` | of full | best order vs global-K |
|---|---|---|---|---|---|
| penicillin V | 42 | 0 | 361 | 1.00 | 1.09x |
| | | 1e-8 | 274 | 0.76 | 1.22x |
| | | **1e-6** | **236** | **0.65** | **1.29x** |
| | | 1e-4 | 181 | 0.50 | 1.41x |
| taxol | 113 | 0 | 853 | 1.00 | 1.14x |
| | | 1e-8 | 528 | 0.62 | 1.36x |
| | | **1e-6** | **407** | **0.48** | **1.47x** |
| | | 1e-4 | 268 | 0.31 | 1.69x |
| PfPMT | 284 | 0 | 1 004 | 1.00 | 1.46x |
| | | 1e-8 | 515 | 0.51 | 2.12x |
| | | **1e-6** | **374** | **0.37** | **2.93x** |
| | | 1e-4 | 234 | 0.23 | 4.68x |
| HcgC | 545 | 0 | 1 487 | 1.00 | 1.54x |
| | | 1e-8 | 676 | 0.45 | 2.99x |
| | | **1e-6** | **465** | **0.31** | **4.35x** |
| | | 1e-4 | 270 | 0.18 | 7.49x |

**And that is the number W6 was looking for.**  The screened `nlambda` is
**236, 407, 374, 465** from 42 atoms to 545 — *flat*, where the unscreened one
runs 361, 853, 1004, 1487 and grows like N^0.68.  The quantity that decides a
pair-atomic exchange build is size-independent after all; it is just not the
quantity the screening the *integrals* use leaves behind.  At 284 atoms
PARI-K is then **2.93x** fewer flops than stock global DF's exchange and
2.2x fewer than `fastdf`'s, and at 545 atoms **4.35x** and 3.3x.

Two caveats on that table, both of which the row above them is the answer to.
The error a threshold costs is measured at 42 and 113 atoms (§1.1) and
**extrapolated** at 284 and 545 — those two rows come from `--lam-fit`, which
builds the coefficients and no tensor, because the tensor is 86 GB and
414 GB.  And it is 1e-6 that is quoted rather than 1e-4 because 1e-6 is the
last threshold whose error (1.2e-7 relative on `K`) is more than three orders
inside the approximation's own (5.4e-4); 1e-4 is only 24 times inside it, and
this repository does not spend an approximation's error budget twice without
saying so.

---

## 2. What it would have to store

W6 inherits a large number from the round before it: a pair domain holds 118
to 130 auxiliary functions on every system from 10 to 545 atoms, so the
fitting coefficients are **0.79 GB at 284 atoms against global DF's 86.4**,
and 1.98 against 414 at 545.  That number is real and this section does not
dispute it.  **It is a number about the fit, and the robust exchange build is
not a contraction of the fit.**

The first robust term is `(mn | rho~_ls)`: the *exact* pair density `mn`
against the *fitted* partner density `ls`.  As `ls` ranges over every stored
pair its domain ranges over every atom, so the term needs `(mn|Q)` for every
stored pair against every auxiliary function — the same `npair x naux` object
global DF's cderi is, to the byte.

There is one escape and it was measured.  Split `B = R + T` with `T = c J`,
which is rebuildable from the coefficients in a GEMM (`npair nlocal naux`
flops, 2.6e12 at 284 atoms, a fraction of a second), and `R = B - T` the
*residual's* potential — which is **exactly zero on the domain**
(`tests/test_dflocal.py` checks that identity at 1e-11) and might be expected
to decay faster than either half, because a Coulomb-metric fit leaves a
residual whose low multipoles are small.  `--residual` builds `R` and `B` and
measures what survives a threshold **on `B`'s scale**, which is the scale the
exchange build's accuracy is set on:

| | atoms | `tol` | `R` kept | `B` kept | `R + c` | global | | `relK` of the threshold |
|---|---|---|---|---|---|---|---|---|
| glycine | 10 | 1e-4 | 6.1 % | 42.5 % | 0.007 GB | 0.02 GB | 2.6x | 1.8e-5 |
| | | 1e-6 | 42.5 % | 75.1 % | 0.013 | 0.02 | 1.3x | 2.3e-7 |
| alanine dipeptide | 22 | 1e-4 | 3.0 % | 23.5 % | 0.027 | 0.14 | 5.3x | 2.2e-5 |
| | | 1e-6 | 29.2 % | 51.6 % | 0.064 | 0.14 | 2.2x | 2.9e-7 |
| penicillin V | 42 | 1e-4 | 1.1 % | 13.1 % | 0.112 | 1.07 | **9.5x** | 1.4e-5 |
| | | 1e-6 | 16.3 % | 34.5 % | 0.274 | 1.07 | 3.9x | 1.3e-7 |
| taxol | 113 | 1e-4 | **0.42 %** | 6.5 % | 0.454 | 14.31 | **31.5x** | 2.9e-5 |
| | | 1e-6 | 8.4 % | 20.9 % | 1.592 | 14.31 | 9.0x | 4.0e-7 |
| | | 1e-8 | 25.6 % | 38.2 % | 4.058 | 14.31 | 3.5x | 2.5e-9 |

`max|R| / max|B|` is **3.2e-2** on all four, and the last column is the
exchange matrix the thresholded tensor gives against the unthresholded one —
so a threshold of 1e-4 costs 1.4e-5 to 2.9e-5 relative where the pair-atomic
approximation it sits inside is already 2.5e-4 to 5.4e-4 away from global DF,
and a threshold of 1e-6 costs a thousandth of that.

**So it works, and the first reading of it did not.**  Normalised to its own
largest element `R` keeps the same fraction as `B` at every threshold, which
says the two have the same *shape* and was written down here as a negative
result before the scale was fixed.  On `B`'s scale — the one the exchange
build's accuracy is set on — `R` keeps **6 to 15 times fewer** entries than
`B` does, and the gap widens with the molecule: 0.42 % against 6.5 % at 113
atoms.

**And the ceiling is the coefficients.**  `R + c` at taxol and 1e-4 is
0.454 GB of which **0.319 is `c`**, so the scheme is already within 1.4x of
the bound `docs/README_dflocal.md` §1 quotes for the fit alone (45x at 113
atoms, 110x at 284, 210x at 545).  That is W6's storage number, reached — but
through the residual split and not through the coefficient count by itself,
and at a threshold whose cost is measured rather than assumed.

**What is not here is the container.**  The probe *counts* what survives and
*zeroes* what does not; it does not store `R` sparsely and it does not rebuild
`T = cJ` per pass at run time.  Both are straightforward — the rebuild is the
`_fold` GEMM of §4 run per exchange pass instead of once per build, about
0.1 s a call at 113 atoms — and neither is written.  That is the single
largest thing this round leaves, because it is the one shape in which a
545-atom robust density-fitted calculation becomes possible at all: global DF
needs **414 GB** there against a node's 251, and this needs about **3**.

---

## 3. Whether it gives the right energy, above 14 atoms

W6's second precondition was *"anything built here must be benchmarked on
**energies** against global DF, not only on wall clocks"*, and
`docs/README_dflocal.md` §2.2 answered it on an alkane series that stops at
C4H10 — because its reference builds an explicit four-index tensor and is
`nao^4`.  `electrolites/dflocal.py` is not, so the same series runs to
**C20H42**.  HF/def2-SVP, both sides converged to 1e-10, the `pair` domain:

| | atoms | nao | `E(PARI-r) - E(global DF)` | per atom | per heavy atom |
|---|---|---|---|---|---|
| CH4 | 5 | 34 | +9.53e-5 | 1.90e-5 | 9.52e-5 |
| C2H6 | 8 | 58 | +1.48e-4 | 1.85e-5 | 7.40e-5 |
| C4H10 | 14 | 106 | +3.51e-4 | 2.51e-5 | 8.78e-5 |
| C6H14 | 20 | 154 | +5.07e-4 | 2.53e-5 | 8.45e-5 |
| C8H18 | 26 | 202 | +6.48e-4 | 2.49e-5 | 8.10e-5 |
| C12H26 | 38 | 298 | +9.10e-4 | 2.40e-5 | 7.59e-5 |
| C16H34 | 50 | 394 | +1.15e-3 | 2.30e-5 | 7.19e-5 |
| C20H42 | 62 | 490 | +1.37e-3 | **2.22e-5** | 6.87e-5 |

**Flat, and falling slightly.**  The first four rows reproduce
`docs/README_dflocal.md` §2.2 exactly, which is the check that this
implementation and that reference are the same approximation; the last four
are new and they close the item's "still open: energies above 14 atoms".
Robust pair-atomic fitting is size-extensive.

**The variation that is left is chemistry, not size.**  The same quantity on
molecules that are not alkanes:

| | atoms | per atom | per heavy atom |
|---|---|---|---|
| glycine | 10 | 3.32e-5 | 6.64e-5 |
| alanine dipeptide | 22 | 3.96e-5 | 8.71e-5 |
| penicillin V | 42 | 6.23e-5 | 1.09e-4 |
| taxol | 113 | **7.38e-5** | 1.35e-4 |

That looks like growth with size and is not: the alkane series says the size
dependence is flat, and these four differ in composition — penicillin V
carries a sulfur and taxol is oxygen-rich, and both are far from the C-H
single bonds the series is made of.  **So the number to carry forward is a
range, 2e-5 to 8e-5 Eh per atom, and not the single 2.5e-5 the round before
this one quoted from an alkane series.**  Anything that wants a tighter
number wants a wider domain:

| | atoms | `pair` per atom | `pair+3.0` per atom |
|---|---|---|---|
| CH4 | 5 | 1.90e-5 | 5.7e-15 |
| C2H6 | 8 | 1.85e-5 | 7.5e-8 |
| C4H10 | 14 | 2.51e-5 | 1.0e-7 |
| C8H18 | 26 | 2.49e-5 | 9.3e-8 |

which is `docs/README_dflocal.md` §2.2's 1e-7 Eh per atom, confirmed on a
different implementation and carried from 14 atoms to 26.  It costs about
eight times the coefficients and — the part that matters here — about eight
times `nlambda`, so it is an accuracy reference and not an operating point
for the exchange build.

---

## 4. What it is measured at

`benchmarks/df_local_k_bench.py`, one A100-SXM4-80GB of a whole node,
def2-SVP, a **converged** density (a `minao` guess screens far harder than a
converged one -- `docs/ROUND2_1.8.1.md` §2), minimum of three, and **one side
per process** with the density passed between them through `--dm-cache`, for
the reason §5.1 gives.  `pair` domain, `DFLOCAL_KMODE=auto`, which chose the
MO order on every row here.

| | atoms | build, global | build, PARI | `get_jk`, global | `get_jk`, PARI | | `relK` |
|---|---|---|---|---|---|---|---|
| glycine | 10 | 0.068 s | 0.078 s | 0.0040 s | 0.0090 s | 0.44x | 5.1e-4 |
| alanine dipeptide | 22 | 0.088 | 0.137 | 0.0060 | 0.0170 | 0.35x | 5.2e-4 |
| penicillin V | 42 | 0.236 | 0.361 | 0.0340 | 0.0330 | 1.03x | 2.5e-4 |
| taxol | 113 | **1.984** | **1.712** | **0.642** | **0.533** | **1.20x** | 5.4e-4 |
| taxol, `lambda` at 1e-6 | 113 | — | 1.758 | 0.642 | **0.463** | **1.39x** | 5.4e-4 |

Against `fastdf`+`fastdfcderi` rather than stock GPU4PySCF the taxol row reads
**1.015 s** and **0.633 s**, so the exchange ratios are 1.19x and 1.37x and
the build ratio inverts: this repository's own cderi build is **1.69x** faster
than the pair-atomic one, where upstream's is **1.16x slower** than it.

**The exchange build tracks the model.**  §1 predicts `2 nao / (nao +
nlambda)` for the MO order — 1.14x at taxol unscreened and 1.47x screened —
and the measured 1.20x and 1.39x are that, against a `get_jk` that also
carries the Coulomb pass and the unpack.  Below a hundred atoms the model
predicts a loss and the measurement is a loss; penicillin V's 1.03x is that
loss inside the noise of a 33 ms call and not a win.  `fastdf`'s `dsyrk` is
worth little here because the second contraction it halves is the one PARI
replaces.

**The build is the part that is implementation and not algebra.**  It is
1.16x *faster* than upstream's and **1.69x slower** than this repository's
own, and the second of those is a gap against what the arithmetic says: the
fold that stands in for global DF's metric contraction is
`npair nlocal naux` against upstream's `2 npair naux_sorted naux`, which is
**49x** fewer flops.  Profiled with a synchronising timer between stages
(`--anatomy`), the 1.71 s at taxol is

| stage | | what it is |
|---|---|---|
| `int3c2e` | 370 ms (25 %) | the three-centre integrals and the sparse contraction-coefficient gather, both upstream's |
| `fit+fold` | 249 ms (16 %) | one batched solve per atom-pair domain, then `B -= cJ/2` |
| `blocks` | 272 ms (18 %) | laying the coefficients out per auxiliary atom; a `numpy` loop over atoms, and the obvious thing to take next |
| `store` | 63 ms (4 %) | |
| the `get_jk` pass, of which | 554 ms | |
| ↳ `unpack` | 108 ms | `fill_symmetric`, the same kernel `fastdf` calls |
| ↳ `step1` | 237 ms | `rhok = M . C`, which *is* `fastdf`'s first contraction |
| ↳ `step2` | 152 ms | `rhok G` in place of `rhok rhok^T` |
| ↳ `local` | 30 ms | `G`, the pair-atomic intermediate — 2 % of a pass |
| ↳ `J` | 25 ms | the Coulomb pass, which rides the same tensor |

Two things in that table are worth carrying.  **`local` is 2 %** — the
intermediate that makes this a different method costs almost nothing, and the
whole argument is about `step2`, which is 152 ms against the 237 ms `step1`
that PARI and global DF share.  And **the fold is 249 ms**, which took two
rewrites: as one cuSPARSE product per batch (`DFLOCAL_FOLD=csr`, kept
runnable) the same stage is **1 513 ms**, and with a 269 MB temporary
allocated per group the dense form was 835 ms.  The structure a sparse matrix throws
away is that a pair's domain is *two contiguous auxiliary blocks*, so the
fold is one dense GEMM per auxiliary atom; `_fold` is that, and §5.2 is the
allocation.

### 4.1 And at 284 atoms, where neither tensor is resident

The size `docs/DF_WORK_LIST.md` is written around is 284 atoms, where the
cderi is 86.4 GB and lives on the host.  The pair-atomic tensor is the same
86.4 GB (§2), so both sides stream, and the row is worth having for what it
says about which of the two costs is binding.  Same node, one side per
process, `--reps 2`, `lambda` screened at 1e-6:

| PfPMT, 284 atoms | build | `get_jk` |
|---|---|---|
| global DF, stock | 27.16 s | **12.31 s** |
| global DF, `fastdf`+`fastdfcderi` | 27.58 s | 15.94 s |
| PARI, `lambda` at 1e-6 | 43.55 s | **24.60 s** |

**PARI is 2.0x slower on a call whose arithmetic is 2.93x smaller**, and the
stage table says the arithmetic is not what is being measured:

| stage of the 24.6 s call | |
|---|---|
| `stream` — 86.4 GB off the host | **12.64 s**, i.e. **6.8 GB/s** |
| `step2` — the contraction itself | 10.58 s, 7.3e13 flops, 6.9 TF/s |
| `unpack`, `local`, `J` | 0.66 s together |

GPU4PySCF moves the same 86.4 GB at **26.4 GB/s** and hides all of it behind
the exchange contractions (`DF_WORK_LIST.md` §3.3): a double-buffered
prefetch out of *mapped, page-locked* memory, where this module does a
synchronous `cupy.asarray` of a pageable slice.  At upstream's rate the
transfer is 3.3 s and disappears behind the 10.6 s of arithmetic, and the
call is level with global DF's — not 2.93x, because at this size the
exchange build is **bound by the tensor crossing PCIe and not by the
arithmetic**, and PARI's tensor is the same size.  The build says the same
thing more bluntly: **24.9 s of its 43.6 is the host store**, against 7.0 s
of integrals and 7.8 s of fit and fold.

So the 284-atom row is not a measurement of the contraction order.  It is a
measurement of the two host-memory paths this module does not have and
`electrolites/fastdfcderi.py` already does — `cudaHostRegister` in place of a
pageable allocation, and `transpose_write` in place of a strided assignment —
and, behind those, of §2: the only thing that makes the flop ratio visible at
284 atoms is not moving 86.4 GB at all.

### 4.2 And a whole SCF

Both sides converged, taxol, HF/def2-SVP, `conv_tol = 1e-10`, one process
each: global DF **-2909.773416466** in 12.0 s, PARI-r **-2909.765077098** in
38.0 s unscreened and **-2909.765076707** in 13.2 s with the `lambda` list
screened at 1e-6.  Three things in that line.

* The energy difference is **+8.34e-3 Eh**, which is the 7.38e-5 per atom of
  §3 and not a convergence artefact -- the two SCFs take the same number of
  cycles.
* The screening changes the total energy by **3.9e-7 Eh** on a 113-atom
  molecule, which is 3.5e-9 per atom: it is invisible next to the
  approximation it sits inside, exactly as §1.1's exchange-matrix
  measurement says it should be.
* And PARI is **slower end to end** even though its `get_jk` is faster,
  because the build is 1.67x and an SCF pays it once against fifteen cheap
  calls.  The build is §4's second half and it is implementation.

---

## 5. Four measurement rules this round learned

The first three are corollaries of rules `docs/DF_WORK_LIST.md` §5 already
has, and each cost a wrong number before it was found.  The fourth is an
observation about something else that this round tripped over.

**5.1 Two tensors of the same size do not fit on one card, and the loser goes
to the host silently.**  The first taxol row measured here read PARI's
`get_jk` at **2.86 s** against global DF's **0.445** — a 6.4x loss where the
model says 1.14x.  The cause was not the contraction: `benchmarks/
df_local_k_bench.py` held a converged `_DFHF` object (cderi 14.3 GB, on the
device) alive while it built the pair-atomic tensor (also 14.3 GB), and the
second one's `_on_gpu` test therefore put it on the **host**.  Nothing failed
and nothing warned; the column that said so was `store`, which is why it is
printed now.  `--only global|pari` runs one side per process and `--dm-cache`
makes the two processes drive exactly the same density.  This is
`DF_WORK_LIST.md` §5.3's rule — the benchmarks compare separate processes —
arriving from a different direction: *within* one process, two variants of a
tensor that is a fifth of the card each will not both be resident.

**5.2 `fill_symmetric` allocates its output if you let it, and that is 57 %
of a call.**  Profiled at 113 atoms with a synchronising timer between stages,
the unpack was **1.04 s of a 1.84 s** `get_jk` — against a written volume of
56.5 GB, which is 40 ms of an A100's bandwidth.  The kernel was not slow; the
`cp.zeros((nao, nao, nQ))` it makes when `out=` is not given was, once per
pass, together with the `cudaFree` that ends each one.  That is
`DF_WORK_LIST.md` §5.8 exactly — "anything allocated inside a block loop
belongs outside it" — and `docs/README_fastdfblock.md` §4.1 is the round that
wrote the rule after losing 2.33 s of an 11.73 s call to it.  Every buffer a
pass needs is now allocated once, before the loop.

**5.3 A module that imports another module's kernels imports its patches.**
`dflocal` takes three things from `electrolites/fastdfcderi.py` -- the sparse
form of the auxiliary contraction coefficients, the kernel that applies it,
and the shell-pair aggregation constant.  Importing that module **patches
`df/df.py:_cholesky_eri`**, so for one round of measurements the "stock"
global-DF build column was not stock: `benchmarks/df_local_k_bench.py`
imports `dflocal` at the top, and that was enough.  The symptom was a
patched column and an unpatched one agreeing to three digits.  The three
pieces now live in `electrolites/_dfaux.py`, which patches nothing, and
`fastdfcderi` imports them from there.

**5.4 And one that is not about this module.**  At 284 atoms, where both
tensors are host-resident, `get_jk` on the **patched** global-DF path is
**15.94 s** against stock GPU4PySCF's **12.31 s** on the same node in the same
job (`--reps 2`, one A100-SXM4-80GB).  `docs/README_fastdfblock.md` measures
that call at **1.21x** the other way.  The two are not the same
configuration — that round fixed the auxiliary block size and drove the tensor
through `--cpu-cderi`, and this one takes whatever `DF.build()` chooses on an
80 GB card, which §5.6 of the work list says is a different block — but it is
a 1.6x discrepancy on a published number and it is recorded here rather than
left to be found.  Nothing in this round depends on it: every PARI comparison
above is against **stock** GPU4PySCF, and the patched column is reported
beside it.

---

## 6. What this does not say

* **Nothing about gradients**, and the module makes that explicit rather than
  leaving it to be discovered: `nuc_grad_method`, `Gradients` and `Hessian` on
  a PARI object raise `NotImplementedError`.  A non-variational fit carries
  terms global DF does not, and the density-fitted gradient drivers would
  build their own *global* cderi from the same auxiliary basis and hand back
  the derivative of a different energy than the one the SCF converged.
* **Nothing about range separation.**  `DF.range_coulomb` hands a
  short-range build a copy of the object with an `omega`; the fit here is a
  Coulomb-metric fit and `get_jk` refuses an `omega`.  So a range-separated
  hybrid — which is most of what this repository is benchmarked on — cannot
  use this module at all yet.
* **Nothing about the contraction at 284 atoms.**  §4.1 measures that size
  and finds the call PCIe-bound on both sides: the object a robust
  pair-atomic exchange build streams is the same size as the cderi, so the
  binding constraint — 86.4 GB across a gen4 x16 link every cycle,
  `DF_WORK_LIST.md` §3.3 — is unchanged, and the 2.93x cannot show.  Two of
  the three things that would let it are already written, in
  `electrolites/fastdfcderi.py`; the third is §2's container.
* **Nothing about an integral-direct variant**, which is the other shape in
  which W6's storage number becomes the operating number: evaluate `B` per
  pass instead of storing it, keep only the 0.79 GB of coefficients, and pay
  the three-centre integrals every cycle instead of once.  §4.1 prices it
  without meaning to — the integral stage of a 284-atom build is **7.03 s**
  against the **3.3 s** a saturated gen4 x16 link would take for the same
  tensor — so it is about twice a *hidden* stream and removes 86.4 GB of host
  memory, and it is the variant to try if §2's container turns out to be
  awkward.

---

## 7. Tools

```bash
python tests/test_dflocal.py [--basis def2-tzvpp]        # the bar

python benchmarks/df_local_k_probe.py                    # §1, no GPU
python benchmarks/df_local_k_probe.py --lam    --cases taxol:0:0
python benchmarks/df_local_k_probe.py --lam-fit --cases PfPMT:0:-1,HcgC:0:0
python benchmarks/df_local_k_probe.py --residual --residual-k \
    --cases penicillinV:0:0,taxol:0:0                    # §2

python benchmarks/df_local_k_bench.py --no-jk --scf --alkanes 1,2,4,8,16,20
python benchmarks/df_local_k_bench.py --no-jk --scf --domain pair+3.0 \
    --alkanes 1,2,4,8                                    # §3
python benchmarks/df_local_k_bench.py --only global --dm-cache $SCRATCH/dm \
    --cases taxol:0:0
python benchmarks/df_local_k_bench.py --only pari --anatomy \
    --dm-cache $SCRATCH/dm --cases taxol:0:0             # §4, and §5.1
DFLOCAL_FOLD=csr python benchmarks/df_local_k_bench.py --only pari \
    --anatomy --dm-cache $SCRATCH/dm --cases taxol:0:0   # the fold ablation
```

`dflocal`'s switches -- `DFLOCAL_DOMAIN`, `DFLOCAL_STORE`, `DFLOCAL_BLK`,
`DFLOCAL_KMODE`, `DFLOCAL_LAMBDA_TOL`, `DFLOCAL_FOLD` -- are listed with their
meanings at the top of `electrolites/dflocal.py`, and every ablation column
above is measured with them.  An 80 GB card is what §2's `--residual` and
`--lam-fit` at 545 atoms need; on a 40 GB one the 545-atom fit runs out of
memory in the metric gather.
