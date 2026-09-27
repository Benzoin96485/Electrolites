# Every other contraction past 2²⁹ elements: an audit, one guard for all of them, and a gradient that did not run

`docs/DF_WORK_LIST.md` **W27**, found by W21.  W21's rule
(`docs/README_fastdfplan.md`) says which cuTENSOR contractions can be
refused — any whose *output* passes 2²⁹ = 536 870 912 elements, unless one
particular mode of it factors into pieces of at most 2²⁹ by factors no
larger than 13 — and four places in this repository asked cuTENSOR before
they ran one.  The entry asked for everything else: every `contract(` on the
density-fitting path bounded at the 545-atom HcgC case, the query in front
of the ones that can pass, and `benchmarks/cutensor_plan_probe.py --run` on
real runs.

---

## 0. Summary

* **The list is measured, not grepped** (§1).  `benchmarks/cutensor_audit.py`
  records every call to `gpu4pyscf/lib/cutensor.py:contraction` that a
  density-fitted SCF, gradient, Hessian and TDA calculation makes, by call
  site, with the largest output each site produced; labels every extent of
  that output with the molecule's dimension it equals (`nao`, `naux`,
  `nocc`, `natm`, …, or `blk` for a block the caller sized); and evaluates
  the labelled shape at taxol (113 atoms), PfPMT (284) and HcgC (545).  Ten
  runs — penicillin V in DF-B3LYP, DF-PBE, DF-ωB97X, def2-TZVPP, open-shell,
  host-resident and TDA; the Cu(II) macrocycle open-shell; taxol closed- and
  open-shell; PfPMT's SCF and gradient — reach **235 call sites**.
* **101 of them can pass 2²⁹ by 545 atoms, and 17 already do at 113**
  (§2).  The 101 are 43 in `fastdfhessip`, 26 in `_dfhessip_u`, 10 in
  `fastdfhess`, 6 in `fastdfgrad`, 4 in `fastxchess`, 1 in `fastdf` — and
  **11 in GPU4PySCF's own code** that no module here replaces: the CPHF's
  `gen_vind` products and `_ao2mo` in `hessian/{rhf,uhf}.py`, `hess_elec`'s
  projection of the CPHF solution, and `df/df_jk.py`'s exchange pass where
  `fastdf` hands a call back.  Of the 17 that pass at 113 atoms, nine are
  behind one of W21's or W25's queries; the other eight ran **because of
  the arithmetic of their extents** — every def2-SVP CHNO molecule has a
  Cartesian AO count divisible by 5, and W18's own item 4 planned at taxol
  for that reason and is refused at `nao_c = 1187` — which is exactly the
  property W21 said not to rely on.  No run here was refused (every site
  answered "plans" at its largest shape), so this is a list of latent
  failures, not of observed ones.
* **One guard instead of a hundred** (§3).  A per-site query would be a
  list to keep in step with every future edit, in two codebases.
  `electrolites/planguard.py` wraps the one function every `contract` in
  GPU4PySCF and here goes through, looked up at call time.  Below 2²⁹ it is
  the plain call (one product of extents; **≈2 µs on a 55 µs call**,
  §3.2); above, a cached plan query; a refused contraction is **cut along a
  free output mode into pieces that plan** (`_cuplan.contract_with`) and
  computes every element with the same arithmetic.  `tests/test_planguard.py`
  runs W21's refused gradient contraction and the CPHF's `gen_vind` product
  at 450 right-hand sides and `nao = 1123` — both refused unguarded — through
  it, to **0.0** against `cp.matmul`.  `planguard` is in `patch_all()`.
* **What the audit's first run found instead** (§4): **every
  range-separated density-fitted gradient crashed under `patch_all()`** —
  `AssertionError: ((2590, 992), 2191)` in `fastdfgrad._metric_pass`, on
  ωB97X, CAM-B3LYP, every basis and molecule tried.  W13's cached two-centre
  optimiser (`_dfj2c`) set the long-range ω on the `cell` it had been built
  from, which is the full-range `Int3c2eOpt`'s own `auxmol`, and left it
  there; the gradient's reuse of that optimiser (W3) then built the
  long-range metric for its full-range call.  One `try/finally`; the
  gradients now agree with GPU4PySCF to 4e-11–7e-9 Eh/bohr, and a
  range-separated DF Hessian, which builds its own optimiser, was never
  affected (6e-10 before and after).

Everything is one A100-SXM4-80GB of a whole Perlmutter node, GPU4PySCF 1.8.1,
cuTENSOR 2.6.0, def2-SVP with its jkfit set unless a row says otherwise,
`patch_all()`.

---

## 1. The audit

```
python benchmarks/cutensor_audit.py --xyz benchmarks/molecules/penicillinV.xyz \
    --xc b3lyp --what scf,grad,hess,tdscf --json pen.json
python benchmarks/cutensor_audit.py --merge pen.json,taxol.json,...
```

The recorder is installed over `gpu4pyscf.lib.cutensor.contraction` before
anything is imported, so every module — ours and upstream's, whichever name
it bound `contract` under — reaches it.  A call site is the first frame
outside `gpu4pyscf/lib` (and outside `planguard`'s two frames).  For the
largest output a site produced it keeps the shape and asks `_cuplan` whether
that shape plans.

**Labelling.**  Each extent is matched to the running molecule's
dimensions, most specific first — `naux_c`, `naux`, `nao_c`, `nao`, `nocc`,
`noccb`, `nvir`, `natm`, then small multiples (`3*natm`, `2*nocc`) — and an
extent that matches none is a block the caller chose (`blk=920`), which is
carried at the size it was seen at.  So a blocked output's bound is *per
block of that size*, and the true bound at 545 atoms is whatever the
caller's sizing gives there — larger, since the blocks are sized from free
memory and not from the molecule.  Two cases that labelling gets wrong in
the safe direction: an extent equal by coincidence to a dimension is
bounded as that dimension (never smaller than it is); an extent the caller
clips to a dimension (`min(blk, naux)`) is labelled as the dimension where
the whole of it fitted.  Merging keeps, per site, the run whose labelled
shape gives the largest bound.

**Coverage.**  Ten runs: penicillin V (42 atoms) in DF-B3LYP with SCF,
gradient, Hessian and a five-state TDA; DF-PBE and DF-ωB97X with SCF,
gradient and Hessian; def2-TZVPP B3LYP; the cation as UKS; the host-resident
cderi branch; the Cu(II) macrocycle quartet (60 atoms, UKS) with a Hessian;
taxol (113) closed-shell with a Hessian (701 s) and its cation's SCF and
gradient; and PfPMT (284 atoms, 86 GB host cderi) SCF and gradient.  What it
cannot see is a branch no run took: `fastdfhessip`'s memory-short paths,
`dflocal` (opt-in, not patched), multi-GPU paths (asserted away here).
`planguard` covers those too, which is the point of §3.

---

## 2. What can pass

Full table: §6.  By where each site first passes 2²⁹:

| first past at | sites | of which upstream's |
|---|---|---|
| taxol, 113 atoms | 17 | 1 (`hessian/uhf.py` `gen_vind`) |
| PfPMT, 284 atoms | 69 | 9 |
| HcgC, 545 atoms only | 15 | 1 (`df/df_jk.py` exchange pass) |
| never (at 545 atoms) | 134 | |

The 17 at 113 atoms, and what kept each of them running there:

| site | output | largest seen | why it ran |
|---|---|---|---|
| `fastdfhess` CPHF build, `Lpq,snqi->snpiL` and `snpiL,spj->snjiL` | `(s, 3natm, nao, nocc, blk)` | 2.6e9 at 113 atoms | W21's batch query |
| `_dfhessip_u._mm` (W25 item 13), three patterns | `(naux, 2, nao_c, nocc)`, … | 2.7e8 | W25's query |
| `fastdfhessip._h_aux_block`, `xuv,vij->xuij` | `(3, naux_c, 2nocc, nocc)` | 1.8e8 | W25's query |
| `fastdfgrad` ip1 loop, `iqr,pi->pqr` and the UKS `niqr,npi->pqr` | `(nao_c, nao_c, blk)` | 1.35e9 at 113 atoms | W21's `prange` |
| `_dfhessip_u` ip2 pseudo-density, `niqr,npi->pqr` | `(nao_c, nao_c, blk)` | 2.9e8 | W25's `prange` |
| `fastdfhessip` ip2 pseudo-density, `iqr,pi->pqr` | `(nao_c, nao_c, blk)` | 1.6e8 | **nothing**: at 113 atoms the block's own factors (774 = 2·3²·43) |
| `fastdfhessip` h1 item 4, `ui,riq->urq` | `(nao_c, blk, nao_c)` | **1.29e9 at 113 atoms** | **nothing**: 1185 = 3·5·79 |
| `_dfhessip_u` h1 item 4, `nui,rniq->nurq` | `(2, nao_c, blk, nao_c)` | 5.8e8 | nothing: a factor |
| `_dfhessip_u` C stage, `xrs,sniq->xrniq` | `(3, blk, 2, nocc, nao_c)` | 1.2e8 | not reached past 2²⁹ yet |
| `_dfhessip_u` B stage, `rniq,nqj->rnij` | `(naux_c, 2, nocc, nocc)` | 6.0e7 | not reached past 2²⁹ yet |
| `fastxchess`, `ypg,ayg->apg` (restricted and unrestricted) | `(3natm, blk, blk_g)` | 4.1e8 at 113 atoms | under the limit, just |
| `hessian/uhf.py:gen_vind`, `snpi,sqi->snpq` | `(2, 3natm, nao, nao)` | 9.6e7 | nothing: plans at taxol⁺'s shape by the query |

("largest seen" is over the ten runs; the "first past" of a row whose
largest output was seen on a smaller molecule is its labelled shape at
taxol's dimensions.)  **Nine are behind one of W21's or W25's queries; the
other eight ran because of the arithmetic of their extents.**  The item-4
row is this repository's own W18 code: a `(1185, 920, 1185)` output plans
on taxol because 1185 has the factors 3 and 5, and the query says the same
output at `nao_c = 1187` is refused.  A def2-SVP CHNO molecule always has a
Cartesian count divisible by 5 (15 functions per heavy atom, 5 per
hydrogen); a transition-metal complex, an ECP or another basis family need
not.  At PfPMT's dimensions the CPHF's own `npi,qi->npq` at 852
right-hand sides is refused by the query outright, and a 284-atom Hessian
would have stopped there.

---

## 3. One guard for all of them

### 3.1 What it does

`electrolites/planguard.py` replaces `gpu4pyscf/lib/cutensor.py:contraction`
— which `cupy_helper.contract`, and so every `contract` bound anywhere, calls
by global lookup at call time — with `_cuplan.contract_with` over the
original:

1. **Output at most 2²⁹ elements** (`out.size`, or the product of the
   output extents): the plain call.  W21's rule says such a call is never
   refused.
2. **Larger**: `_cuplan.planable` on the real operands' shapes and strides
   (cached; ~0.3 ms the first time).  If it plans, the plain call.
3. **Refused**: the contraction is cut along an output mode that belongs to
   one operand only — the largest such — into the largest pieces that plan
   (`_cuplan.prange`, which also halves a refused remainder), and each piece
   is the same call on views of that operand and of the output.  Every
   element is computed over the same summed indices, so the arithmetic is
   unchanged.
4. **If no free mode can be cut into pieces that plan**, `cupy.einsum`
   (upstream's own fallback path, `_contract_einsum`), which has no limit.
   No pattern measured needs it.

The four W21/W25 guards stay: they pick a block that plans whole, which is
one launch where a cut is several.  `PLANGUARD_OFF=1` restores the refusal.

### 3.2 Correctness and cost

`tests/test_planguard.py`, on real 4.5–10 GB outputs:

| | unguarded | guarded |
|---|---|---|
| W21's gradient contraction, `iqr,pi->pqr` at `(226, 1185, 914)`, through a module's `contract` | `CUTENSOR_STATUS_NOT_SUPPORTED` | **0.0** vs `cp.matmul` |
| the same with `out`, `alpha = 2`, `beta = 0.5` | — | 0.0 |
| operands swapped (free mode on the second operand) | — | 0.0 |
| into a strided view, with the other half of the buffer untouched | — | 0.0 |
| `hessian/rhf.py:gen_vind`'s `npi,qi->npq`, 450 RHS, `nao = 1123` | refused | **0.0** |
| a 30 × 50 contraction | — | the plain call, no query |

`tests/test_cuplan.py` has the same checks on `_cuplan.contract`.  The cost
below the limit is one Python call layer and a product:

| 20 000 calls of a `(30,40)·(40,50)` contraction | plain | guarded |
|---|---|---|
| with `out` | 53.1–54.2 µs | 54.9–56.2 µs |
| without | 58.3–59.6 µs | 61.8–63.2 µs |

— 2–3.5 µs on a call that is itself 55 µs of Python and plan creation.  On
whole Hessians (§3.3) it is inside the run-to-run spread.

### 3.3 Whole runs

`benchmarks/df_hess_anatomy.py --rounds 2 --patch all`, `PLANGUARD_OFF=1` and
not in alternating processes on one GPU, the pair twice, round 1 of each:

| whole Hessian, penicillin V/def2-SVP | guard off | guard on | traces |
|---|---|---|---|
| DF-B3LYP, restricted | 15.33 / 15.39 s | 15.54 / 15.44 s | 41.78582245(65/66) / …(69/64) |
| DF-B3LYP, the cation, UKS | 31.93 / 33.98 s | 33.45 / 32.15 s | 41.14938312(18/08) / …(11/09) |

On the restricted Hessian that is +0.1–0.2 s, about 1 % — tens of thousands
of `contract` calls at 2–3 µs each — and on the open-shell one it is inside
a 6 % spread.  Nothing is cut in either: every site the audit saw
plans at its largest shape (§2), so what is measured is the guard's cost on
the path that does not need it.

---

## 4. The range-separated DF gradient

The first ωB97X row of the audit did not get past its gradient:

```
File ".../electrolites/fastdfgrad.py", line 787, in _metric_pass
    assert coef.shape[1] == naux_in, (coef.shape, naux_in)
AssertionError: ((2590, 992), 2191)
```

and neither does any range-separated DF gradient under `patch_all()` at
HEAD: water, glycine and ethanol, ωB97X and CAM-B3LYP, def2-SVP, 6-31G and
cc-pVDZ all raise it.  Without `fastdfcderi` it runs; without `fastdfgrad`
it runs; `fastdfrsh` (W22) makes no difference.  `tests/test_dfgrad.py`
covers ωB97X-V but patches `fastdfgrad` alone, which is why it never saw it.

**The cause.**  After an ωB97X SCF, `mf.with_df.intopt.auxmol.omega` is
**0.3**, on the full-range optimiser.  W13's `_dfj2c.int2c2e` keeps one
built `Int2c2eOpt` per auxiliary basis, and `SortedGTO.from_cell` returns a
`SortedGTO` it is given unchanged, so the kept optimiser's `cell` is the
auxmol of whichever build first made it — the full-range `Int3c2eOpt`'s.
Each call sets `opt.cell.omega = mol.omega` so that the kernel sees the
caller's operator; the long-range cderi build's call set 0.3 on that object
and nothing set it back.  The SCF is unaffected (its full-range tensor was
built first and is cached; energies agree to 2e-13).  The gradient's first
call reuses the SCF's `Int3c2eOpt` (W3's `reuse_opt`) and asks
`_decompose_j2c` for the *full-range* metric on its auxmol — and gets the
long-range one, rank 992 of 2590 by `LINEAR_DEP_THR`, against a full-range
cderi of 2191 rows.  The long-range metric is always numerically
rank-deficient, so every case measured fails loudly at the assertion rather
than silently with the long-range integrals.

**The fix** puts the caller's ω on the kept `cell` for the call and the old
one back after it (`_dfj2c.int2c2e`, `try/finally`).
`tests/test_dfcderi.py` now builds the optimiser from one auxmol, makes a
long-range call through a second one, and checks that the first is still at
ω = 0 and still gets the full-range metric — it fails at HEAD (ω = 0.3,
2.5e-1 off) and passes with the fix.

| gradient, `patch_all()` against unpatched | HEAD | fixed |
|---|---|---|
| water, CAM-B3LYP/def2-SVP | AssertionError | 3.6e-10 |
| water, ωB97X/6-31G | AssertionError | 7.7e-11 |
| water, CAM-B3LYP/cc-pVDZ | AssertionError | 1.3e-10 |
| ethanol, CAM-B3LYP/def2-SVP | AssertionError | 7.2e-9 |
| glycine, ωB97X/def2-SVP | AssertionError | runs (0.0360 max) |

(Eh/bohr, largest element; the ethanol row is the SCF's own convergence at
`conv_tol = 1e-10`, two processes.)  A DF-ωB97X Hessian of glycine agrees
with GPU4PySCF's to 5.6e-10 at HEAD and 6.1e-10 fixed — it builds its own
optimiser, so it never met the stale ω.

---

## 5. Tools

```bash
python benchmarks/cutensor_audit.py --xyz ... --xc ... --what scf,grad,hess[,tdscf] \
    [--charge q --spin s] [--host-cderi] --json out.json
python benchmarks/cutensor_audit.py --merge a.json,b.json,... [--all] \
    [--at HcgC] [--also taxol,PfPMT:-1]
python tests/test_planguard.py
python tests/test_cuplan.py
python tests/test_dfcderi.py
PLANGUARD_OFF=1 ...          # cuTENSOR's refusal back
```

---

## 6. The list

`--merge` of the ten runs, every site whose bound passes 2²⁹ by 545 atoms,
largest first.  `bound` is the labelled shape at HcgC (`nao 5037`,
`nao_c 5293`, `naux 24722`, `naux_c 29124`, `nocc 1011`, `natm 545`), per
block of the size seen where there is a `blk`; `first past` is the smallest
of taxol, PfPMT and HcgC at which it passes; `plan` is cuTENSOR's answer at
the largest shape seen.  Line numbers are this commit's for `electrolites/` and GPU4PySCF 1.8.1's for the rest.  The runs are from before
W29 (`docs/README_fastdfhessip3.md`); the penicillin V Hessian re-audited
after it reaches the same call sites and patterns.  The four
`fastxchess` rows have no line: W28 (`docs/README_fastxchessmgga.md`)
writes those operands with a kernel of its own instead of `contract`
(W28 did not re-run this audit).  Every row is
guarded by `planguard`; the rows marked in §2 are also behind their own
query.

```
# site                                           pattern                 calls  max seen plan  labelled shape                     bound first past
  fastdfhess._jk_once:590                      Lpq,snqi->snpiL        4852  2.55e+09 ok    (2,3*natm,nao,nocc,blk=256)     4.26e+12 taxol
  fastdfhess._jk_once:608                      snpiL,spj->snjiL       4852  5.46e+08 ok    (2,3*natm,nocc,nocc,blk=256)    8.56e+11 taxol
  _dfhessip_u._mm                              rs,npir->snpi          87  2.74e+08 ok    (naux,2,nao_c,nocc)             2.65e+11 taxol
  fastdfhessip._h_aux_block:1283               xuv,vij->xuij          7  1.82e+08 ok    (3,naux_c,2*nocc,nocc)          1.79e+11 taxol
  hessian.uhf.fx:358                           snpi,sqi->snpq         13  9.55e+07 ok    (2,3*natm,nao,nao)              8.30e+10 taxol
  _dfhessip_u._mm                              uv,vnij->unij          2  6.05e+07 ok    (naux_c,2,nocc,nocc)            5.95e+10 taxol
  _dfhessip_u._get_veff:768                    rniq,nqj->rnij         2  6.05e+07 ok    (naux_c,2,nocc,nocc)            5.95e+10 taxol
  _dfhessip_u._get_veff:865                    nui,rniq->nurq         29  5.80e+08 ok    (2,nao_c,blk=990,nao_c)         5.55e+10 taxol
  _dfhessip_u._mm                              uv,unij->vnij          2  5.07e+07 ok    (naux,2,nocc,nocc)              5.05e+10 taxol
  hessian.rhf.fx:751                           npi,qi->npq            30  2.33e+07 ok    (3*natm,nao,nao)                4.15e+10 PfPMT
  fastdfhessip._jk_energy_per_atom:506         uv,vij->uij            5  2.19e+07 ok    (naux_c,nocc,nocc)              2.98e+10 PfPMT
  fastdfhessip._get_veff:2077                  riq,qj->rij            5  2.19e+07 ok    (naux_c,nocc,nocc)              2.98e+10 PfPMT
  fastdfgrad.ip1_loop:1132                     niqr,npi->pqr          44  2.90e+08 ok    (nao_c,nao_c,blk=990)           2.77e+10 taxol
  _dfhessip_u._ejk_ip2:590                     niqr,npi->pqr          29  2.90e+08 ok    (nao_c,nao_c,blk=990)           2.77e+10 taxol
  fastdfgrad.ip1_loop:1136                     iqr,pi->pqr            310  1.35e+09 ok    (nao_c,nao_c,blk=960)           2.69e+10 taxol
  fastdfhessip._get_veff:2165                  ui,riq->urq            67  1.29e+09 ok    (nao_c,blk=920,nao_c)           2.58e+10 taxol
  fastdfhessip._jk_energy_per_atom:504         uv,uij->vij            5  1.85e+07 ok    (naux,nocc,nocc)                2.53e+10 PfPMT
  fastdfhessip._ejk_ip2:1816                   iqr,pi->pqr            73  1.60e+08 ok    (nao_c,nao_c,blk=774)           2.17e+10 taxol
  _dfhessip_u._get_veff:1029                   nxspq,sqj->nxsjp       2  1.95e+07 ok    (natm,3,2,nocc,nao_c)           1.75e+10 PfPMT
  _dfhessip_u._get_veff:1039                   nxsjp,spi->snxij       2  1.85e+07 ok    (2,natm,3,nao,nocc)             1.67e+10 PfPMT
  hessian.uhf.fx:357                           snai,spa->snpi         13  1.85e+07 ok    (2,3*natm,nao,nocc)             1.67e+10 PfPMT
  fastdfhess._jk_once:576                      spiL,nL->nspi          4828  9.97e+06 ok    (3*natm,2,nao,nocc)             1.67e+10 PfPMT
  fastdfhess._jk_once:606                      snpiL,sjiL->nspj       4852  9.97e+06 ok    (3*natm,2,nao,nocc)             1.67e+10 PfPMT
  fastdfhess._jk_once:609                      spiL,snjiL->nspj       4852  9.97e+06 ok    (3*natm,2,nao,nocc)             1.67e+10 PfPMT
  fastdfhess._to_mo:649                        nspi,spq->nsqi         43  1.85e+07 ok    (3*natm,2,nao,nocc)             1.67e+10 PfPMT
  fastdfhess._to_mo:653                        nspi,spq->nsqi         37  1.85e+07 ok    (3*natm,2,nao,nocc)             1.67e+10 PfPMT
  _dfhessip_u._get_veff:788                    xrs,sniq->xrniq        102  1.22e+08 ok    (3,blk=377,2,nocc,nao_c)        1.21e+10 taxol
  fastdfgrad.ip1_loop:1131                     nrji,nqj->niqr         44  1.07e+08 ok    (2,nocc,nao_c,blk=990)          1.06e+10 PfPMT
  _dfhessip_u._ejk_ip2:588                     rnji,nqj->niqr         29  1.07e+08 ok    (2,nocc,nao_c,blk=990)          1.06e+10 PfPMT
  _dfhessip_u._ipaux_batch:443                 pqr,npi->niqr          87  1.07e+08 ok    (2,nocc,nao_c,blk=990)          1.06e+10 PfPMT
  _dfhessip_u._jk_energy_per_atom:313          pqr,nqi->npir          87  1.07e+08 ok    (2,nao_c,nocc,blk=990)          1.06e+10 PfPMT
  _dfhessip_u._get_veff:746                    pqr,npi->rniq          29  1.07e+08 ok    (blk=990,2,nocc,nao_c)          1.06e+10 PfPMT
  _dfhessip_u._get_veff:876                    pqr,nqi->npri          87  1.07e+08 ok    (2,nao_c,blk=990,nocc)          1.06e+10 PfPMT
  _dfhessip_u._get_veff:971                    pqr,npi->rniq          87  1.07e+08 ok    (blk=990,2,nocc,nao_c)          1.06e+10 PfPMT
  fastdfhessip._get_veff:2343                  nxpq,qj->nxjp          5  5.26e+06 ok    (natm,3,nocc,nao_c)             8.75e+09 PfPMT
  hessian.rhf._ao2mo:826                       nij,jo->nio            56  9.27e+06 ok    (3*natm,nao,nocc)               8.33e+09 PfPMT
  hessian.rhf._ao2mo:827                       nio,ip->npo            56  9.27e+06 ok    (3*natm,nao,nocc)               8.33e+09 PfPMT
  hessian.uhf.hess_elec:81                     kxai,pa->kxpi          2  9.27e+06 ok    (natm,3,nao,nocc)               8.33e+09 PfPMT
  hessian.uhf.hess_elec:82                     kxai,pa->kxpi          2  8.99e+06 ok    (natm,3,nao,noccb)              8.33e+09 PfPMT
  fastdfhessip._get_veff:2351                  nxjp,pi->nxij          6  4.98e+06 ok    (natm,3,nao,nocc)               8.33e+09 PfPMT
  hessian.rhf.fx:750                           npo,ip->nio            30  4.98e+06 ok    (3*natm,nao,nocc)               8.33e+09 PfPMT
  hessian.rhf.hess_elec:91                     kxai,pa->kxpi          5  4.98e+06 ok    (natm,3,nao,nocc)               8.33e+09 PfPMT
  fastdfhess._to_mo:660                        pqn,sqi->nspi          6  4.98e+06 ok    (3*natm,1,nao,nocc)             8.33e+09 PfPMT
  fastdfhess._to_mo:661                        nspi,spq->nsqi         6  4.98e+06 ok    (3*natm,1,nao,nocc)             8.33e+09 PfPMT
  fastdfhessip._h_aux_block:1301               xr,ys->xrys            6  8.24e+07 ok    (3,naux_c,3,naux_c)             7.63e+09 PfPMT
  fastdfhessip._h_aux_block:1334               xrt,ytr->xryt          7  8.24e+07 ok    (3,naux_c,3,naux_c)             7.63e+09 PfPMT
  fastdfhessip._j_energy_per_atom:981          xrs,yts->rtxy          1  6.04e+07 ok    (naux_c,naux_c,3,3)             7.63e+09 PfPMT
  fastdfhessip._j_energy_per_atom:991          xrt,ytr->rtxy          1  6.04e+07 ok    (naux_c,naux_c,3,3)             7.63e+09 PfPMT
  fastxchess._deriv1_block_u                   yqg,ayg->aqg           592  7.37e+07 ok    (3*natm,nocc,blk=4096)          6.77e+09 PfPMT
  fastxchess._deriv1_block                     yqg,ayg->aqg           844  4.75e+07 ok    (3*natm,nocc,blk=4096)          6.77e+09 PfPMT
  df.df_jk.proc:558                            Lij,njk->nikL          144  5.06e+07 ok    (5,nao,nocc,blk=256)            6.52e+09 PfPMT
  _dfhessip_u._jk_energy_per_atom:341          xrnuj,sr->xsnuj        102  6.54e+07 ok    (3,naux_c,2,blk=36,nocc)        6.36e+09 PfPMT
  _dfhessip_u._jk_energy_per_atom:343          ysnij,nui->ysnuj       102  6.54e+07 ok    (3,naux_c,2,blk=36,nocc)        6.36e+09 PfPMT
  _dfhessip_u._get_veff:946                    rniq,ar->aniq          87  6.49e+06 ok    (natm,2,nocc,nao_c)             5.83e+09 PfPMT
  fastxchess._deriv1_block                     ypg,ayg->apg           844  4.11e+08 ok    (3*natm,blk=797,blk=4096)       5.34e+09 taxol
  fastdfhessip._jk_energy_per_atom:711         rs,pir->spi            39  4.14e+07 ok    (blk=992,nao_c,nocc)            5.31e+09 PfPMT
  _dfhessip_u._jk_energy_per_atom:156          pqr,pi->iqr            58  5.36e+07 ok    (nocc,nao_c,blk=990)            5.30e+09 PfPMT
  fastdfgrad.ip1_loop:1135                     rji,qj->iqr            310  2.57e+08 ok    (nocc,nao_c,blk=960)            5.14e+09 PfPMT
  fastdfhessip._get_veff:2046                  pqr,pi->riq            80  2.46e+08 ok    (blk=920,nocc,nao_c)            4.92e+09 PfPMT
  fastdfhessip._get_veff:2176                  pqr,qi->pri            201  2.46e+08 ok    (nao_c,blk=920,nocc)            4.92e+09 PfPMT
  fastdfhessip._get_veff:2280                  pqr,pi->riq            240  2.46e+08 ok    (blk=920,nocc,nao_c)            4.92e+09 PfPMT
  fastdfhessip._get_veff:2169                  ui,rij->urj            28  2.46e+08 ok    (nao_c,blk=920,nocc)            4.92e+09 PfPMT
  fastdfhessip._jk_energy_per_atom:473         pqr,pi->iqr            73  3.23e+07 ok    (nocc,nao_c,blk=774)            4.14e+09 PfPMT
  fastdfhessip._ejk_ip2:1815                   rji,qj->iqr            73  3.23e+07 ok    (nocc,nao_c,blk=774)            4.14e+09 PfPMT
  fastdfhessip._ipaux_batch:1189               pqr,pi->iqr            219  3.23e+07 ok    (nocc,nao_c,blk=774)            4.14e+09 PfPMT
  fastdfhessip._jk_energy_per_atom:706         pqr,qi->pir            219  3.23e+07 ok    (nao_c,nocc,blk=774)            4.14e+09 PfPMT
  fastdfhessip._chol_d4_d5:1439                ysji,ui->ysuj          239  2.90e+07 ok    (3,naux,blk=48,nocc)            3.60e+09 PfPMT
  fastdfhessip._jk_energy_per_atom:718         rs,pir->spi            180  2.69e+07 ok    (blk=645,nao_c,nocc)            3.45e+09 PfPMT
  fastxchess._deriv1_block_u                   ypg,ayg->apg           592  3.38e+08 ok    (3*natm,blk=458,blk=4096)       3.07e+09 taxol
  fastdfhessip._get_veff:2252                  riq,ar->aiq            201  1.75e+06 ok    (natm,nocc,nao_c)               2.92e+09 HcgC
  fastdfhess._jk_once:563                      Lpq,sqj->spjL          441  2.64e+07 ok    (2,nao,nocc,blk=256)            2.61e+09 PfPMT
  fastdfhessip._h_aux_block:1326               xrij,sij->xrs          7  2.75e+07 ok    (3,naux_c,naux_c)               2.54e+09 PfPMT
  fastdfhessip._h_aux_block:1329               xr,t->xrt              6  2.75e+07 ok    (3,naux_c,naux_c)               2.54e+09 PfPMT
  fastdfhessip._h_aux_block:1332               xrt,st->xrs            7  2.75e+07 ok    (3,naux_c,naux_c)               2.54e+09 PfPMT
  fastdfhessip._j_energy_per_atom:990          xrt,st->xrs            1  2.01e+07 ok    (3,naux_c,naux_c)               2.54e+09 PfPMT
  fastdfhessip._get_veff:2098                  xrs,siq->xriq          281  1.94e+07 ok    (3,blk=155,nocc,nao_c)          2.49e+09 PfPMT
  fastdfhessip._get_veff:2092                  xrs,siq->xriq          42  1.94e+07 ok    (3,blk=155,nocc,nao_c)          2.49e+09 PfPMT
  _dfhessip_u._get_veff:764                    rs,sniq->rniq          25  2.62e+07 ok    (naux_c,2,8,nao_c)              2.47e+09 PfPMT
  fastdfhessip._get_veff:2073                  uv,vij->uij            45  1.88e+07 ok    (naux_c,16,nao_c)               2.47e+09 PfPMT
  fastdfhessip._h_aux_block:1279               xrs,st->xrt            7  2.30e+07 ok    (3,naux_c,naux)                 2.16e+09 PfPMT
  _dfhessip_u._oo_gram:492                     ys,sr->ysr             2  2.30e+07 ok    (3,naux_c,naux)                 2.16e+09 PfPMT
  _dfhessip_u._jk_energy_per_atom:351          xrnuj,snuj->xsr        102  2.30e+07 ok    (3,naux_c,naux)                 2.16e+09 PfPMT
  fastdfhessip._j_energy_per_atom:979          xrs,st->xrt            1  1.70e+07 ok    (3,naux_c,naux)                 2.16e+09 PfPMT
  _dfhessip_u._jk_energy_per_atom:350          snij,nui->snuj         102  2.18e+07 ok    (naux_c,2,blk=36,nocc)          2.12e+09 PfPMT
  _dfhessip_u._get_veff:762                    rs,rniq->sniq          25  2.19e+07 ok    (naux,2,8,nao_c)                2.09e+09 PfPMT
  fastdfhessip._get_veff:2071                  uv,uij->vij            45  1.59e+07 ok    (naux,16,nao_c)                 2.09e+09 PfPMT
  _dfhessip_u._ipaux_batch:445                 niqr,nqj->rnij         87  1.98e+07 ok    (blk=990,2,nocc,nocc)           2.02e+09 PfPMT
  fastdfhessip._chol_d4_d5:1447                suj,xtuj->xst          239  1.44e+07 ok    (3,naux,naux)                   1.83e+09 HcgC
  fastdfhessip._jk_energy_per_atom:772         xruj,sr->xsuj          42  1.36e+07 ok    (3,naux_c,blk=19,nocc)          1.68e+09 HcgC
  fastdfhessip._jk_energy_per_atom:774         ysji,ui->ysuj          42  1.36e+07 ok    (3,naux_c,blk=19,nocc)          1.68e+09 HcgC
  fastdfhessip._chol_d4_d5:1443                sji,ui->suj            239  1.14e+07 ok    (naux_c,blk=48,nocc)            1.41e+09 HcgC
  fastdfgrad._fill_from_cderi:703              pqr,pi->iqr            284  1.32e+07 ok    (nocc,nao,blk=256)              1.30e+09 HcgC
  df.df_jk.proc:555                            Lij,jk->ikL            144  1.01e+07 ok    (nao,nocc,blk=256)              1.30e+09 HcgC
  _dfhessip_u._jk_energy_per_atom:157          iqr,qj->rij            58  9.90e+06 ok    (blk=990,nocc,nocc)             1.01e+09 HcgC
  fastdf._get_jk:994                           Lij,njk->nikL          2266  2.08e+08 ok    (2,nao,blk=361,blk=256)         9.31e+08 HcgC
  fastdfgrad._metric_pass:815                  nrk,nsk->rs            18  9.16e+06 ok    (naux_c,naux_c)                 8.48e+08 HcgC
  fastdfhessip._h_aux_block:1311               r,s->rs                6  9.16e+06 ok    (naux_c,naux_c)                 8.48e+08 HcgC
  fastdfhessip._h_aux_block:1321               rs,ts->rt              63  9.16e+06 ok    (naux_c,naux_c)                 8.48e+08 HcgC
  fastdfhessip._jk_energy_per_atom:474         iqr,qj->rij            73  6.55e+06 ok    (blk=774,nocc,nocc)             7.91e+08 HcgC
  fastdfhessip._ipaux_batch:1191               iqr,qj->rij            219  6.55e+06 ok    (blk=774,nocc,nocc)             7.91e+08 HcgC
  fastdfhessip._jk_energy_per_atom:780         sji,ui->suj            42  4.53e+06 ok    (naux_c,blk=19,nocc)            5.59e+08 HcgC
```
