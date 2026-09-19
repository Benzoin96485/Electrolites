# Pair-atomic fitting: the two measurements that decide it

`docs/DF_WORK_LIST.md` **W6** proposes fitting each AO pair in the auxiliary
functions of its own two atoms rather than in all of them, and it attaches two
conditions before anything is written: *"read the PARI / local-RI literature
before writing anything; decide robust vs non-robust fitting first, because it
changes the whole data structure"*, and *"anything built here must be
benchmarked on **energies** against global DF, not only on wall clocks"*.

This round is those two conditions and nothing else.  No GPU kernel, no
`electrolites/` module: two benchmarks, a reference implementation that could
never be fast, and a decision.

**The decision is robust (Dunlap) fitting, and it is not close.**
Non-robust pair-atomic fitting does not merely lose accuracy — it collapses
variationally, and the collapse *grows with the molecule*: **−0.81 Eh** on
ethane, **−6.50** on propane, **−16.5** on butane, **−10.6** on an
ammonia–borane adduct.  Robust fitting on the same coefficients is
**2.5e-5 Eh per atom** against global DF and that number is flat from 5 atoms
to 14.

**And the storage claim is real and larger than the item estimated.**  A pair
domain holds **118 to 130** auxiliary functions on every system from 10 to 545
atoms — not the 300 the item guessed — so the tensor goes from **86.4 GB to
0.79 GB** at 284 atoms and from **414 GB to 1.98 GB** at 545.  The exponent
falls from N^2.52 to **N^1.53**.

That 414 GB is itself a correction to the list: `DF_WORK_LIST.md` §3.3
demoted W6 because "at 284 atoms on a 251 GB node capacity is not what binds".
At 545 atoms it binds — a global cderi for `HcgC`/def2-SVP does not fit in the
node, which is why `docs/README_fastdfblock.md` could only microbenchmark that
size.

Everything below is def2-SVP with its own jkfit auxiliary set.  Neither
benchmark needs a GPU.

---

## 1. What it would store

`benchmarks/df_local_probe.py`.  No SCF and no integrals beyond the screening,
which is a property of the geometry and the two basis sets.

| | atoms | stored pairs | naux | global | `nlocal` | pair-atomic | |
|---|---|---|---|---|---|---|---|
| glycine | 10 | 4 623 | 471 | 0.02 GB | 112 | 0.004 GB | 4x |
| alanine dipeptide | 22 | 18 148 | 974 | 0.14 | 113 | 0.016 | 9x |
| penicillin V | 42 | 60 845 | 2 191 | 1.07 | 130 | 0.063 | 17x |
| taxol | 113 | 319 589 | 5 598 | 14.31 | 125 | 0.319 | **45x** |
| PfPMT | 284 | 808 156 | 13 365 | 86.41 | 122 | 0.789 | **110x** |
| HcgC | 545 | 2 094 539 | 24 722 | **414.25** | 118 | 1.975 | **210x** |

```
     pair: storage ~ N^1.53   (global ~ N^2.52)
 pair+3.0: storage ~ N^1.71
```

`nlocal` — the mean number of auxiliary functions a pair is fitted in — is
**flat**, which is the whole of W6's claim and is now measured rather than
asserted.  The global exponent of 2.52 is the one `DF_WORK_LIST.md` §3.2
estimates as "about O(N^2.5)" from two points; this is six.

Widening the domain to every atom within 3 Å of either of the pair's two
multiplies `nlocal` by about eight (1017 at 284 atoms, 1065 at 545) and still
leaves a 13x to 23x reduction.  §2 says what that buys in accuracy, and it is
three orders of magnitude.

### 1.1 The local metrics are better conditioned than the global one

A pair-atomic fit factorises one `nlocal x nlocal` Coulomb metric per *atom*
pair instead of one `naux x naux` metric for the molecule.  `--cond`:

| | global metric | local pair domains (median) |
|---|---|---|
| glycine, 10 atoms | 2.2e8 | 2.3e7 |
| alanine dipeptide, 22 | 5.1e8 | 1.8e7 |
| penicillin V, 42 | 2.3e9 | 2.3e7 |

The global condition number grows with the molecule and the local ones do
not, because a local metric is a property of two atoms and their basis.  This
is worth stating because it is the opposite of what "smaller domains are
worse conditioned" would suggest, and because `_decompose_j2c`'s cuSOLVER
factorisation of the global metric is `DF_WORK_LIST.md` **W13**'s subject:
pair-atomic fitting replaces one badly conditioned `naux^3` factorisation
with many small well conditioned ones.

---

## 2. Whether it gives the right energy

`benchmarks/df_local_accuracy.py`.  A **reference** implementation: dense
`(nao^2, naux)` fitting coefficients, an explicit four-index tensor, and an
SCF driven through it.  It is `nao^4` and caps at about 110 basis functions;
nothing in it could be fast and nothing in it is meant to be.  Four expansions
at the same geometry and basis, so that the only difference is the expansion:

* `exact` — the four-centre integrals;
* `global` — ordinary density fitting, which is what `fastdf` contracts;
* `pari` — pair-atomic, **non-robust**: `(mn|ls) ~ c J c`;
* `pari-r` — pair-atomic, **robust** (Dunlap): `(mn|Q)c + c(P|ls) - c J c`.

With a *global* domain the last two coincide with `global` to 1e-13, which is
the implementation's own check.

### 2.1 Non-robust fitting collapses, and the collapse grows

An idealised alkane series, HF/def2-SVP, energies in Eh:

| | atoms | `global` - exact | `pari` - exact | `pari-r` - exact |
|---|---|---|---|---|
| CH4 | 5 | +1.8e-5 | +1.0e-2 | +1.1e-4 |
| C2H6 | 8 | +5.2e-5 | **-8.1e-1** | +2.0e-4 |
| C3H8 | 11 | +6.1e-5 | **-6.5e+0** | +3.3e-4 |
| C4H10 | 14 | +7.0e-5 | **-1.6e+1** | +4.2e-4 |

and two molecules chosen for their bonding rather than their size:

| | atoms | `global` - exact | `pari` - exact | `pari-r` - exact |
|---|---|---|---|---|
| H2O | 3 | +5.6e-5 | -1.4e-3 | +8.9e-5 |
| NH3BH3 | 8 | +1.5e-5 | **-1.1e+1** | +1.8e-4 |
| ethanol | 9 | +8.0e-5 | **-3.3e+0** | +2.7e-4 |

The sign is the point, and so is the growth: the non-robust energy goes
*below* the exact one and keeps going, reaching **1.2 Eh per atom** at butane.

It is not the coefficients.  Those were checked, because inflated fitting
coefficients are the obvious suspect: the largest `|c|` is **2.41** for both
the pair-atomic and the global fit on every member of the series, and the
pair-atomic fit's Frobenius norm is the *smaller* of the two (14.6 against
34.1 at butane).  The fit is well behaved; the expansion is not.

Write the fitted product density as `p~ = p - d`.  The non-robust integral is
`(p~|s~) = (p|s) - (d|s) - (p|d') + (d|d')`, whose error is **first order** in
the residual `d`; the robust one is `(p|s~) + (p~|s) - (p~|s~) = (p|s) -
(d|d')`, **second order**.  And on the diagonal the non-robust value is the
Coulomb-metric projection of `p` onto the domain, so `(p~|p~) = (p|p) -
(d|d) <= (p|p)`: it can only *under*estimate, which is the sign of the
collapse.  Those two facts — first order against second, and one-signed — are
the four orders of magnitude between the last two columns of the tables above.

Widening the domain to 3 Å repairs the non-robust variant too (C4H10 reads
+4.4e-4 there), which is another way of saying the failure is the domain and
not the arithmetic.

**So the data structure is decided: robust.**  That costs what W6 warned it
would — the robust expression needs the three-centre integrals `(mn|P)` on the
domain *as well as* the coefficients `c^mn_P`, so the stored tensor is two of
§1's arrays rather than one.  At 284 atoms that is 1.6 GB against 86.4, and
the conclusion is unchanged.

### 2.2 Robust fitting is size-extensive

The error that matters is against `global`, because global DF is what the
pair-atomic fit would replace, and it has to be **per atom** and flat:

| | atoms | `pari-r` - `global`, per atom | with a 3 Å domain |
|---|---|---|---|
| CH4 | 5 | 1.90e-5 | 7.1e-15 |
| C2H6 | 8 | 1.85e-5 | 7.5e-8 |
| C3H8 | 11 | 2.45e-5 | 6.7e-8 |
| C4H10 | 14 | 2.51e-5 | 1.0e-7 |

**1.9e-5 to 2.5e-5 Eh per atom, flat**, against global DF's own 5e-6 Eh per
atom against exact.  So robust pair-atomic fitting is four to five times
global DF's error and does not accumulate — the property that decides whether
an approximation survives at 284 atoms, and the one non-robust fitting fails.

The exchange matrix, which is where pair-atomic fitting is reported to be
weakest, behaves the same way: at a converged density the largest element of
`K - K_exact` relative to `K_exact` is **2.2e-4** for global DF, **7.8e-4**
for robust PARI and **1.3e-3** for non-robust, all flat across the series.

And the 3 Å domain is not a compromise but a different operating point
entirely: **1e-7 Eh per atom**, i.e. global-DF quality, for 8x the storage of
the pair domain and still 13x less than global.

---

## 3. What this does not say

* **Nothing about speed.**  The storage exponent is not the arithmetic
  exponent.  Robust pair-atomic exchange does *not* factorise the way
  `vk = rhok rhok^T` does — the fitting coefficients depend on the pair, so
  the density cannot be contracted first, which is the identity the whole of
  `fastdf` rests on.  PARI-K is a different algorithm with its own
  contraction order, and estimating it is the next piece of work rather than
  a corollary of this one.
* **Nothing about gradients.**  A non-variational fit has Pulay-like terms
  that global DF does not, and `fastdfgrad` assumes the global ones.
* **Nothing above 14 atoms, on energies.**  The reference is `nao^4`.  The
  series is flat over a factor of three in size and that is the evidence
  there is; the item's own success criterion asks for the §3.2 ladder, and
  reaching it needs the implementation this round deliberately did not write.

---

## 4. Tools

```bash
python benchmarks/df_local_probe.py --domains pair,pair+3.0     # §1, no GPU
python benchmarks/df_local_probe.py --cases penicillinV:0:0 --cond
python benchmarks/df_local_accuracy.py                          # §2
python benchmarks/df_local_accuracy.py --cases c1,c2,c3,c4      # §2.2
python benchmarks/df_local_accuracy.py --xc b3lyp --cases H2O,CH4
```
