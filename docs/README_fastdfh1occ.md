# The restricted h1's per-atom exchange term on the occupied side, and the exchange a pure functional never needed

`docs/DF_WORK_LIST.md` **W30**, found by W25.  W25's item 12 formed the
unrestricted `h1` term's per-atom exchange matrix `(X + X^T) C` with the AO
pair projected on the occupied orbitals *before* each atom's contraction —
`8 nao² nocc` flops per auxiliary function and direction where W18's item 4
spends `4 nao³` — and took it where a shape score said the saving per atom
was large.  The restricted `fastdfhessip._get_veff` has the same term (stage
D2, 4.7 s of 21.3 at 113 atoms) and did not have the choice.  The entry
asked for the same two contractions and the same score in the restricted
function, and warned that the restricted crossover could sit higher,
because each restricted per-atom contraction is one spin's work where the
twin's is two spins' in one launch.

It does not sit higher.  And the same function turned out to form every one
of its exchange contractions for a pure functional, whose exchange factor
is zero.

---

## 0. Summary

* **Item 12 in the restricted function** (§1): `sum_{u∈A,r} j3c[u,q,r]
  s[u,r,j] + j3c_pri[u,r,j] t[u,r,q]` per atom, with `s = C_o·dm_oo` formed
  once per block, `t` item 4's array and `j3c_pri` the `j3c·C_o` that D1
  has just formed — upstream's memory-short branch with both intermediates
  hoisted.  With the occupied index batched, D1's `j3c·C_o` covers the batch
  only, so the form builds its own over every occupied orbital.
* **D2 moves by what it moved in the twin** (§2): **2.97x** in def2-TZVPP at
  42 atoms (1350 → 455 ms) and **1.91x** at 113 atoms (4.70 → 2.47 s),
  against the entry's 3.2x and 1.8x.  The function is **1.87x** GPU4PySCF's
  in def2-TZVPP where the AO form was 1.48x, and **1.76x at 113 atoms**
  where it was 1.53x.
* **The crossover is where the twin's is** (§2).  On fourteen restricted
  systems — organic 10–113 atoms, Ni, Pt, Cu and Co complexes, def2-SVP
  and def2-TZVPP — the AO form wins or ties at every
  score up to 1.67e6 and the occupied form wins at every score from 6.1e6,
  so the twin's threshold of 4e6 carries over unchanged and
  `FASTDFHESSIP_H1ATOM=auto` takes the better form, or one within 2 % of
  it, on all fourteen.  The entry's falsification does not fire.
* **Item 14: a pure functional's h1 term computes no exchange** (§3).
  `df/hessian/rks.py` calls `_get_veff(..., 1., hyb)` with `hyb = 0` for
  PBE, and upstream — and W18's copy of it — forms every exchange
  contraction of stages C, D1, D2 and E and scales it by zero.  Skipping
  them (and the `natm·3·nao²` array only they write) is **1.06x–1.16x** on
  top of W18 for DF-PBE from 22 to 113 atoms, and the whole h1 call is now
  **1.5x–2.1x** GPU4PySCF's for a pure functional.  The unrestricted twin
  gets it too (1.07x–1.08x at 42–60 atoms).
* **A whole Hessian gains what the term was** (§4): h1 was 3–15 % of one,
  so the def2-TZVPP B3LYP Hessian is 1.02x–1.05x (46.6/47.2 → 45.5/45.0 s)
  and the rest is inside the run-to-run spread.
* Correctness (§5): every switch alone, both forms, forced occupied batching
  with the occupied form, and `k = 0`, element-wise against GPU4PySCF's own
  function: **1.0e-15** worst.

Everything is one A100-SXM4-80GB of a whole Perlmutter node, GPU4PySCF 1.8.1,
def2-SVP with its jkfit set unless a row says otherwise, level-3 grids,
`patch_all()` on both sides.  Function tables are
`benchmarks/df_hess_terms_probe.py`, variants round robin in one process
against upstream's function object, minimum of 3–6 (1 at 113 atoms).

---

## 1. What changed

`electrolites/fastdfhessip.py:_get_veff`, the restricted h1 term.  With
enough memory, W18's item 4 accumulates for every atom `A` and direction `x`

    X_A[p,q] = sum_{u in A, r} j3c[u,p,r] t[u,r,q],     t = C_o . dm_3c

in the AO basis, symmetrises all of them at the end and projects each on
the occupied orbitals.  The projection is linear, so it can come first:

    ((X + X^T) C)[q,j] = sum_{u in A, r} j3c[u,q,r] s[u,r,j]  +  sum_{u in A, r} j3c_pri[u,r,j] t[u,r,q]

with `s = C_o · dm_oo` formed once per block and `j3c_pri = j3c · C_o`.
That is two contractions per atom, each with the occupied index as its
short side, and `2·2·nao_A·dk·nao·nocc` flops per atom and block where the
AO form spends `2·nao_A·dk·nao²`; the saving per atom and block is
`4 nao_A dk nao (nao − 2 nocc)` flops, and the score is that per atom,
`(nao/natm) · nao · (nao − 2 nocc)`, as in `_dfhessip_u`.

Two details the twin did not have:

* **The occupied index is batched** in the restricted function when memory
  is short (`FASTDFHESSIP_OCC_BATCH` forces it).  Then `t` and `s` sum over
  the batch's occupied orbitals only, which is right — they accumulate over
  batches — but D1's `j3c·C_o` covers the batch too, and the second
  contraction needs it over every occupied orbital.  With more than one
  batch the form forms its own (`pri_full`), from the same buffer.
* **The per-atom AO matrices are still allocated** when item 12 is taken,
  because stages C and E write their exchange terms into them; only D2's
  moves to the occupied side.

`FASTDFHESSIP_H1ATOM=auto` (the default) takes the occupied form where the
score passes `FASTDFHESSIP_H1OCC_SCORE` (4e6); `=occ` and `=batched` force
either form, `=upstream` is upstream's per-atom loop.

---

## 2. Where the crossover is

h1 against GPU4PySCF's function, both forms (round robin, one process per
system):

| system | basis | `nao_c` / `nocc` | score | GPU4PySCF | AO form | occupied form | `auto` |
|---|---|---|---|---|---|---|---|
| glycine, 10 atoms | SVP | 100 / 20 | 6.0e4 | 287.5 ms | **180.6 (1.59x)** | 208.5 (1.38x) | AO |
| alanine dipeptide, 22 | SVP | 210 / 39 | 2.7e5 | 490.3 | **292.0 (1.68x)** | 336.2 (1.46x) | AO |
| Pt(II) amine, 23, ECP | SVP | 265 / 59 | 4.5e5 | 869.7 | **547.3 (1.59x)** | 582.5 (1.49x) | AO |
| Ni(II) dithiolene²⁻, 21 | SVP | 352 / 105 | 8.4e5 | 1208.8 | **1049.8 (1.15x)** | 1053.3 (1.15x) | AO |
| penicillin V, 42 | SVP | 454 / 92 | 1.3e6 | 2036.0 | **1495.3 (1.36x)** | 1504.4 (1.35x) | AO |
| penicillin V, ωB97X (two calls) | SVP | 454 / 92 | 1.3e6 | 3736.6 | 2862.7 (1.31x) | 2858.4 (1.31x) | AO |
| glycine | TZVPP | 255 / 20 | 1.4e6 | 293.1 | **194.3 (1.51x)** | 218.1 (1.34x) | AO |
| Cu macrocycle, +1, closed shell, 60 | SVP | 541 / 99 | 1.7e6 | 3725.6 | 2248.0 (1.66x) | **2205.6 (1.69x)** | AO |
| alanine dipeptide | TZVPP | 540 / 39 | 6.1e6 | 946.1 | 743.7 (1.27x) | **690.0 (1.37x)** | occupied |
| Co complex, cation, closed shell, 113 | SVP | 1116 / 204 | 7.8e6 | 28106 | 18080 (1.55x) | **15425 (1.82x)** | occupied |
| taxol, 113 | SVP | 1185 / 226 | 9.1e6 | 33799 | 22103 (1.53x) | **19177 (1.76x)** | occupied |
| Pt(II) amine | TZVPP | 656 / 59 | 1.0e7 | 1830.6 | 1378.6 (1.33x) | **1248.1 (1.47x)** | occupied |
| Ni(II) dithiolene²⁻ | TZVPP | 848 / 105 | 2.2e7 | 3932.2 | 2828.1 (1.39x) | **2495.7 (1.58x)** | occupied |
| penicillin V | TZVPP | 1146 / 92 | 3.0e7 | 7064.7 | 4762.0 (1.48x) | **3787.3 (1.87x)** | occupied |

(Two jobs: glycine, alanine dipeptide and the Pt complex in def2-SVP, taxol,
and penicillin V and the Ni complex in def2-TZVPP are the first, minimum of
2–5; the rest the second, minimum of 3–6.  Penicillin V/def2-SVP, measured
in both, moved by 10–15 % between them for both forms alike — the spread of
a 1.5 s call on a node whose other three GPUs are busy — which is why the
Ni, penicillin V and Cu rows, 1–2 % apart, are read as ties.)  Below 1.7e6
the occupied form loses by up to 15 % or ties; from 6.1e6 up it wins by 8–26 %.  The twin's measurements put
its crossover in the same interval (lost at 1.7e6, won at 7.7e6), so the
extra launches of a one-spin contraction do not move it measurably — the
work per launch at a given score is the same `nao_A · dk · nao · nocc`, and
the twin's second spin doubles both sides of the comparison.

Stage D2, the per-atom exchange term alone (`FASTDFHESSIP_TIME=1`,
synchronised stage timers, second call of two):

| | penicillin V, TZVPP | taxol | penicillin V, SVP |
|---|---|---|---|
| D2, AO form (item 4) | 1349.5 ms | 4703.1 ms | 206.7 ms |
| D2, occupied form (item 12) | **454.4 ms** | **2465.7 ms** | 320.7 ms |
| | **2.97x** | **1.91x** | 0.64x |
| whole call (timers on) | 4834 → 3917 ms | 21839 → 19611 ms | 1641 → 1761 ms |

Everything else in the call is unchanged to within the timers' spread.  At
113 atoms what is left is C (5.6–6.8 s, the per-auxiliary-atom `j2c_10`
contractions), D1 and B, as W18 found.

---

## 3. Item 14: the exchange a pure functional never needed

`df/hessian/rks.py:make_h1` calls `_get_veff(intopt, mo_coeff, mo_occ, 1.,
hyb)`, and for PBE `hyb = 0`.  Upstream's `_get_veff` — and W18's copy of it
— has no branch for that: stage C's `j2c_10` exchange contraction, D1's
`j3c·C_o` and its product with `dm_3c`, every D2 per-atom contraction, and
stage E's, are formed and added with a factor of `-0.5 · 0`.  So is the
`(natm, 3, nao, nao)` array only they write, which at 113 atoms is 3.8 GB of
the card that could have been auxiliary block.  With `k_factor = 0` the
function now does the Coulomb terms alone: `j3c_1` in stage C only where
the Coulomb half needs it, no `dm_oo`, no item 4 array, no AO matrices.  The
unrestricted twin, whose pure-functional h1 call has the same shape, gets
the same change.  `FASTDFHESSIP_K0=compute` puts the zero-scaled exchange
back.

| DF-PBE h1 | GPU4PySCF | W18 (`K0=compute`) | this round | vs W18 |
|---|---|---|---|---|
| alanine dipeptide, 22 atoms | 483.9 ms | 273.4 | **234.8** | 1.16x |
| Pt(II) amine, 23 | 912.6 | 505.6 | **447.2** | 1.13x |
| penicillin V, 42 | 1840.3 | 1280.8 | **1207.3** | 1.06x |
| Cu macrocycle⁺, 60 | 3376.4 | 1861.5 | **1751.2** | 1.06x |
| penicillin V, TZVPP | 5381.0 | 3189.8 | **2877.9** | 1.11x |
| taxol, 113 (reps 1) | 27430 | 15431 | **14056** | 1.10x |

and in the unrestricted twin:

| UKS DF-PBE h1 | GPU4PySCF | W25 (`K0=compute`) | this round | vs W25 |
|---|---|---|---|---|
| FeCl₄⁻ sextet, 5 atoms | 336.8 ms | 261.9 | 268.9 | 0.97x |
| penicillin V⁺, 42 | 2367.1 | 1492.8 | **1395.8** | 1.07x |
| Cu(II) macrocycle²⁺ quartet, 60 | 4262.4 | 2329.3 | **2153.4** | 1.08x |

(FeCl₄⁻ is a 0.3 s call of five atoms, and its two variants are within its
own spread.)

The margin is smaller than the share of the flops removed would suggest:
with `k = 0`, D1 and D2 were already the cheaper stages at def2-SVP (§2's
stage table), and the three-centre derivative integrals (stage D) and the
Coulomb per-auxiliary-atom terms are what a pure functional's call is.

---

## 4. Whole Hessians

`benchmarks/df_hess_anatomy.py --rounds 2 --patch all`, HEAD (W26) and this
round in alternating processes on one GPU, the pair run twice, round 1 of
each (so both columns carry `planguard` on the right and not on the left,
`docs/README_fastdfaudit.md` §3.3):

| system | basis | xc | HEAD | this round | h1, HEAD → here | traces |
|---|---|---|---|---|---|---|
| penicillin V, 42 atoms | TZVPP | B3LYP | 46.55 / 47.17 s | **45.51 / 44.96 s** | 5.26 / 4.78 → **3.83 / 3.79 s** | 39.777317038(8/4) / …(3/6) |
| Pt(II) amine, 23 | TZVPP | B3LYP | 14.31 / 15.40 | 14.59 / 15.91 | 1.37 / 1.55 → 1.28 / 1.46 | 19.44362174(26/57) / …(52/48) |
| penicillin V | SVP | PBE | 8.58 / 9.75 | 8.29 / 8.43 | 1.32 / 1.28 → 1.23 / 1.23 | 40.764712845(0/4) / …(4/4) |
| penicillin V⁺, UKS | SVP | PBE | 16.59 / 16.53 | 16.32 / 17.33 | 1.56 / 1.50 → 1.44 / 1.97 | see below |
| taxol, 113 (one pair, round 0) | SVP | B3LYP | 700.15 | 697.84 | 21.47 → **18.52** | 110.843737378(9) / …(68) |

h1 is 3–15 % of a whole Hessian after W18 — 3 % at 113 atoms, where W8's
`_get_jk` is 512 s of the 700 — and the CPHF is most of the rest, so the
whole-run gain is the def2-TZVPP B3LYP row's 1.02x–1.05x and taxol's 3 s;
the other rows move by less than their own two pairs differ.  The UKS DF-PBE
traces spread by 6e-5 on both sides — 12.42368013 / 12.42367661 before,
12.42365196 / 12.42362399 after — which is W25's §4.1: that SCF does not
land on the same density twice, in GPU4PySCF itself.  The other traces
agree to 1e-9, the ip2 kernel's atomics.

---

## 5. Correctness

`tests/test_dfhessip.py` and `tests/test_dfuhessip.py`, element-wise against
GPU4PySCF's own function objects on one converged SCF, relative to the
largest element:

| | worst |
|---|---|
| restricted h1, ethanol: B3LYP / HF / `k = 0` factors, `H1ATOM=batched`, `=occ`, `=upstream`, `K0=compute` and every other switch alone | 6.5e-16 |
| restricted h1, water, the long-range operator (`j = 0`, `omega = 0.3`), both forms | 1.0e-15 |
| restricted h1 with the occupied index forced into 2 and 3 batches, both forms (the occupied form's own `j3c·C_o`) | 8.7e-16 |
| unrestricted h1, ethanol⁺, `K0=compute` and `k = 0` | 6.1e-16 |

The ip2 rows and the memory-short ladder are unchanged.  `h1_occ` and
`h1_score` in `fastdfhessip.stats()` say which form a call took.

---

## 6. Tools

```bash
python tests/test_dfhessip.py --no-hessian
python tests/test_dfuhessip.py --no-hessian
python benchmarks/df_hess_terms_probe.py --xyz ... --basis ... --xc b3lyp --patch all \
    --terms h1 --variants stock --env 'ao:FASTDFHESSIP_H1ATOM=batched,occ:FASTDFHESSIP_H1ATOM=occ'
python benchmarks/df_hess_terms_probe.py ... --xc pbe --terms h1 --variants stock,ours \
    --env 'k0c:FASTDFHESSIP_K0=compute'
FASTDFHESSIP_TIME=1 FASTDFHESSIP_H1ATOM=occ python benchmarks/df_hess_terms_probe.py ...
```
