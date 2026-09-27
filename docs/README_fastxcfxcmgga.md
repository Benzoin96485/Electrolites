# The CPHF's meta-GGA response kernel: at the GEMM rate already, so the arithmetic moves to the occupied side

`docs/DF_WORK_LIST.md` **W33**, found by W28.  After W28 a DF-TPSS Hessian
at 42 atoms was 57 % CPHF in def2-SVP and 72 % in def2-TZVPP, almost all of
it `nr_rks_fxc_mo` — which already ran `fastxcfxc`'s blocked code (W16) —
at 3.9x B3LYP's time on the same molecule in def2-TZVPP.  The entry asked
for a per-stage profile, GEMM rate against operand-formation rate (W28's
finding), and said it was falsified if the call is already at the GEMM rate
for four density products.  **It is**: the falsification fires as written.
What that leaves is the arithmetic itself, and a CPHF right-hand side
carries a cheaper form of it than the AO-matrix route uses.
`electrolites/fastxcfxc.py` is the module, extended; the unrestricted
Hessian gets the same route through `hessian/uks.py:get_veff_resp_mo`.

---

## 0. Summary

* **The falsification fires** (§1).  On W16's route the two GEMM stages of a
  meta-GGA call run at **15.6–17.7 TF/s** (penicillin V in def2-SVP and
  def2-TZVPP, taxol), the rate W8 measured for this card's large DGEMMs;
  forming their operands — W28's problem — is 84 ms of a 6.0 s call.  The
  3.9x over B3LYP was the four products a meta-GGA needs where a GGA needs
  one, and arithmetic.
* **The CPHF carries a cheaper arithmetic** (§2).  Both callers want `V C_o`,
  not `V` (`_ao2mo` multiplies by `C_o` next), and hand a density with
  factors, `D = M (2C_o)^T + h.c.`.  On a grid block with `nsub` AOs, the
  density from `X[n] = C_o^T ao[n]` (shared by every right-hand side) and
  `Y[n] = M^T ao[n]`, and `V C_o` as four GEMMs `ao[n] · W[n]^T` with `W`
  formed from `X` and the `fxc`-weighted density, cost `8 nocc` per set and
  grid point where the matrix route costs `8 nsub`.  Two kernels
  (`fxc_rho_occ_*`, `fxc_w_occ_*`), all GEMMs of inner dimension `nsub` or
  `ngrids`, no nocc-deep ones.  A block goes that way where `nsub > nocc`.
* **The function** (§3), restricted meta-GGA, against W16's route:
  **5.3x–7.5x in def2-TZVPP** (penicillin V, alanine dipeptide, Pt(II)
  amine; 3.6x on Ni(II) dithiolene²⁻), **1.6x–3.2x in def2-SVP** from 10 to
  60 atoms, **1.9x at 113 atoms**; TPSS, r2SCAN, M06-L and M06-2X alike.
  Against GPU4PySCF's own function 3.7x–12.6x.  **TPSS over B3LYP** on
  penicillin V/def2-TZVPP goes from **4.23x to 1.15x** (the arithmetic ratio
  is now 1.58x), in def2-SVP from 3.48x to 1.49x.
* **The unrestricted Hessian** (§4): 1.7x–3.9x for meta-GGAs and 1.6x–2.1x
  for GGAs on Ni, Pt, Cu and Fe complexes — its matrix route evaluated the
  density with upstream's `eval_rho4` at 4–9 TF/s, so here the density half
  moves for GGAs too.  A restricted GGA takes the potential half only:
  1.2x–1.4x in def2-TZVPP, a tie in def2-SVP.
* **Whole DF Hessians** (§5): TPSS penicillin V **40.6 → 17.6 s (2.31x)** in
  def2-TZVPP and 12.5 → 8.6 s (1.44x) in def2-SVP; taxol 256.8 → 190.0 s (1.35x);
  UKS Ni and Pt complexes 1.14x–1.18x; B3LYP 1.03x in def2-TZVPP.  The six
  CPHF calls of the def2-TZVPP TPSS Hessian are **28.5 → 5.6 s**, against
  5.0 s for B3LYP's.  Traces agree to 1e-9.
* **The gain is `nsub/nocc`, and that falls with the molecule** (§6): in a
  fixed basis `nsub` is local and saturates while `nocc` grows with the
  molecule, so the route's margin is 5.2x at `nsub/nocc` 5.4 (penicillin V,
  def2-TZVPP), 2.4x at 2.5 (def2-SVP) and 1.7x–1.9x at 1.8 (taxol).  Blocks
  with `nsub ≤ nocc` stay on the matrix route, so it never loses; the next
  step is a local `nocc` (§7).
* **Memory** (§6.1): the first taxol Hessian ran out of memory, because the
  occupied route's buffers were stacked on the matrix route's eagerly
  allocated ones.  Those are now allocated at the first block that takes
  that route, sized for the blocks that will, and share the occupied route's
  buffer where they fit: at 113 atoms 12 GB of buffers against the 41 GB the
  matrix route held.
* Correctness (§8): `tests/test_fxc_occ.py` (new), 13 systems, restricted
  against GPU4PySCF's AO-basis function taken to the MO basis and
  unrestricted against upstream's `get_veff_resp_mo`, every block / the
  default share / none: **≤ 3.7e-15** relative.

Everything is one A100-SXM4-80GB of a Perlmutter node (a 4-GPU interactive
allocation, one GPU per process, `CUDA_VISIBLE_DEVICES` set), GPU4PySCF 1.8.1,
level-3 grids, `patch_all()` on both sides.  Function tables are
`benchmarks/df_hess_terms_probe.py` (restricted, variants round robin, one
process per system, minimum of 3; 2 at 113 atoms) or
`benchmarks/fxc_occ_probe.py` (both routes in one process, minimum of 3), on
the right-hand sides one CPHF cycle hands the function (`3 natm` tagged
densities).

---

## 1. The profile: already at the GEMM rate

`benchmarks/fxc_occ_probe.py` times the call per stage with
`FASTXCFXC_TIME=1` (synchronised stage timers) and works out each stage's
flops block by block from the same grid and the same decisions the module
makes.  W16's route (`FASTXCFXC_OCC=0`), TPSS:

| stage | penicillin V, def2-TZVPP | penicillin V, def2-SVP | taxol, def2-SVP |
|---|---|---|---|
| `gemm1`: `D·ao[n]`, n = 0..3 | 2695 ms, **17.7 TF/s** | 627 ms, **15.6 TF/s** | 13255 ms, **17.4 TF/s** |
| `rho1` kernel | 263 | 108 | 1467 |
| `scale`: the potential's operands | 84 | 39 | 484 |
| `gemm2`: `aow·ao[0]^T` + three tau GEMMs | 2865 ms, **16.6 TF/s** | 665 ms, **14.7 TF/s** | 14017 ms, **16.5 TF/s** |
| `accum`, `add_sparse` | 96 | 24 | 637 |
| **call** | **6051 ms** | **1485 ms** | **30042 ms** |

    python benchmarks/fxc_occ_probe.py --xyz benchmarks/molecules/penicillinV.xyz \
        --basis def2-tzvpp --xc tpss

Both GEMM stages are at the card's DGEMM rate — W8 measured the DF Hessian's
large GEMMs at 15 TF/s — and together they are 92–94 % of the call.  The
operand formation that was half of W28's function is 1.4 % here: this path
has had its own `fxc_scale` kernel since `docs/README_fastxcfxc.md`, and the
tau term's three operands are that kernel with one component.  So the entry's
falsification fires, and its reading is the right one: a meta-GGA's density
needs `D · ao[n]` for all four `n` and its potential four GEMMs where a GGA's
needs one each, and on the matrix route that is 4x the arithmetic at the
same rate.  B3LYP on the same molecule spends 474 + 670 ms in the two stages
(W16's factor route takes its density at nocc cost), which is the 4.2x.

---

## 2. What changed: the occupied side

A CPHF right-hand side is not a general density matrix.
`hessian/rhf.py:gen_vind` builds

    D_i = M_i (2 C_o)^T + (2 C_o) M_i^T,     M_i = C mo1_i

and tags it with both factors, and what `get_veff_resp_mo` does with the
potential is `_ao2mo(V, C_o, C) = C^T (V C_o)`.  With, on one grid block,

    X[n][k][g] = sum_u C_o[u][k] ao[n][u][g]                 (shared by all sets)
    Y[n][i][k][g] = sum_u M_i[u][k] ao[n][u][g]

and `sum_uv D_uv a_u b_v = 2 sum_k (Y^a X^b + X^a Y^b)`, the five density
components are sums over the occupied index (`fxc_rho_occ_mgga`):

    rho[0] = 4 sum_k Y0 X0,   rho[x] = 4 sum_k (Yx X0 + Y0 Xx),   tau = 2 sum_k sum_x Yx Xx

and with `T = sum_g [sum_c wv_c ao_c ao_0^T + sum_x wv_4 ao_x ao_x^T]` the
block matrix the matrix route builds and `V = T + T^T`,

    (V C_o)[u][k] = sum_n sum_g ao[n][u][g] W[n][i][k][g]
    W[0] = 2 wv_0 X0 + sum_x wv_x Xx,     W[x] = wv_x X0 + 2 wv_4 Xx

(`fxc_w_occ_mgga`, written into `Y`'s memory).  Per set and grid point that
is `4 · 2 nocc nsub` for `Y` and `4 · 2 nocc nsub` for the potential, against
`4 · 2 nsub²` twice; `X` is `1/nset` of `Y`.  Every GEMM has the long side
where cuBLAS wants it — `Y` is `(nset·nocc) × nsub × ng`, the potential
`nsub × ng × (nset·nocc)` — unlike W16's GGA factor route, two of whose three
GEMMs are nocc deep and run at 13 TF/s.  `V C_o` is accumulated as
`(nao, nset·nocc)` rows (the natural output of `ao[n] · W^T`); blocks left on
the matrix route accumulate `V` as before, and `V C_o` for them is one
product at the end.

**A GGA** has no tau row, so its `W[x]` is `wv_x X0` and the `x` terms fold
into `aow · X0^T`, `aow` being what `fxc_scale_gga` already writes:
`V C_o = ao_0 · (sum_c wv_c X_c)^T + aow · X0^T`, two nocc-wide GEMMs against
the matrix route's one nsub-wide one.  So a restricted GGA takes the
potential half where `nsub > 3 nocc` (measured, §3.1) and keeps W16's density
route; `FASTXCFXC_OCC_RHO_GGA=1` puts its density on the occupied side too
and is off (§3.1).

**The unrestricted Hessian.**  There is no `nr_uks_fxc_mo`:
`hessian/uks.py:get_veff_resp_mo` calls `nr_uks_fxc` on the tagged pair and
`_ao2mo`s each spin.  It is replaced — the fxc call and the projection, with
the NLC, Coulomb and exchange lines upstream's — and asks `_fxc_blocked` for
`V_s C_s` directly.  `D_s = M_s C_s^T + h.c.` there, half the restricted
form, which halving `M` puts back; the beta orbitals are the ones
`_stack_uhf_occ_oribtals` pads to `nocca`, and the padded columns are sliced
off at the end.  The two spins' densities and potentials run one after the
other through the same kernels, which take a row stride so that each spin
reads and writes its rows of the interleaved `(nset, 2, ncomp, ng)` arrays
the unrestricted `fxc` contraction wants.  Here the density half moves for a
GGA as well: the matrix route evaluated it with upstream's `eval_rho4`, one
set at a time, at 3.6–8.7 TF/s (§4).

Switches, read at call time: `FASTXCFXC_OCC=0` (every block on the matrix
route — the ablation, W16's code), `FASTXCFXC_OCC_RATIO` (1: a block goes to
the occupied side where `ratio · nocc < nsub`; meta-GGA and the unrestricted
density), `FASTXCFXC_OCC_RATIO_GGA` (3: a GGA's potential half),
`FASTXCFXC_OCC_RHO_GGA` (0).

---

## 3. The function, restricted

`hessian/rks.py:nr_rks_fxc_mo`, three columns round robin in one process:
GPU4PySCF's function (W23-repaired, which `patch_all()` does; it agrees with
the other two to ≤ 1.5e-14), W16's (`FASTXCFXC_OCC=0`) and W33's.

| system | basis | `nao` / `nocc` | xc | GPU4PySCF | W16 | **W33** | W33 / W16 |
|---|---|---|---|---|---|---|---|
| glycine, 10 | SVP | 95 / 20 | TPSS | 233.7 ms | 35.7 | **20.0** | 1.79x |
| alanine dipeptide, 22 | SVP | 200 / 39 | TPSS | 934.5 | 228.5 | **74.4** | 3.07x |
| Pt(II) amine, 23, ECP | SVP | 249 / 59 | TPSS | 1091.2 | 303.5 | **123.8** | 2.45x |
| Ni(II) dithiolene²⁻, 21 | SVP | 327 / 105 | TPSS | 874.2 | 287.5 | **184.0** | 1.56x |
| penicillin V, 42 | SVP | 430 / 92 | TPSS | 3278.3 | 1464.4 | **593.2** | 2.47x |
| penicillin V | SVP | | r2SCAN | 3335.9 | 1466.5 | **593.0** | 2.47x |
| penicillin V | SVP | | M06-L | 3481.3 | 1465.0 | **607.4** | 2.41x |
| penicillin V | SVP | | M06-2X | 3338.4 | 1466.7 | **599.2** | 2.45x |
| Cu macrocycle⁺, 60, closed shell | SVP | 515 / 99 | TPSS | 7095.3 | 5302.9 | **1661.7** | 3.19x |
| taxol, 113 | SVP | 1123 / 226 | TPSS | 37378 | 31243 | **16263** | 1.92x |
| alanine dipeptide | TZVPP | 478 / 39 | TPSS | 907.7 | 833.1 | **110.6** | 7.53x |
| Pt(II) amine | TZVPP | 573 / 59 | TPSS | 1137.6 | 1095.4 | **186.3** | 5.88x |
| Ni(II) dithiolene²⁻ | TZVPP | 728 / 105 | TPSS | 1057.1 | 1008.4 | **283.8** | 3.55x |
| penicillin V | TZVPP | 1007 / 92 | TPSS | 5442.3 | 6011.6 | **1138.8** | 5.28x |
| penicillin V | TZVPP | | r2SCAN | 5401.3 | 6014.4 | **1138.2** | 5.28x |
| penicillin V | TZVPP | | M06-2X | 5421.4 | 6010.8 | **1138.9** | 5.28x |
| penicillin V | SVP | | B3LYP | 1131.1 | 421.1 | **397.2** | 1.06x |
| taxol | SVP | | B3LYP | 18515 | 8820 | **8291** | 1.06x |
| Ni(II) dithiolene²⁻ | TZVPP | | B3LYP | 467.0 | 274.0 | **224.4** | 1.22x |
| penicillin V | TZVPP | | B3LYP | 2117.0 | 1422.1 | **987.2** | 1.44x |

    bash final_r.sh  # = for each row:
    python benchmarks/df_hess_terms_probe.py --xyz benchmarks/molecules/<m>.xyz \
        --basis <b> [--charge <q>] --xc <xc> --terms fxc --variants stock,ours \
        --env 'w16:FASTXCFXC_OCC=0' --patch all

(Metal complexes closed-shell as in `docs/README_fastdfh1occ.md`: the Ni
complex as the dianion, the Cu macrocycle as the cation, the Pt complex as a
singlet.)  The four meta-GGAs are one row each to three digits: the route
does not see the functional beyond its five components.  In def2-TZVPP
W16's route was 10 % slower than GPU4PySCF's on penicillin V (0.90x; that
is W16's result and was not taken apart here, since the route replaces it
on every block of that call); W33 is 4.8x GPU4PySCF's there.

**The entry's success criterion** — the def2-TZVPP call moves toward the
arithmetic ratio over B3LYP: on penicillin V TPSS/B3LYP was **4.23x**
(6011.6 / 1422.1 ms) and is **1.15x** (1138.8 / 987.2); the flop counts of
the two calls are now 16.7 and 10.6 TF, a ratio of 1.58, so the meta-GGA
call is under its arithmetic ratio because its GEMMs are all long and
B3LYP's density GEMMs are not.  In def2-SVP 3.48x → 1.49x, at 113 atoms
3.54x → 1.96x.

The stages after, same three cases as §1:

| stage | penicillin V, def2-TZVPP | penicillin V, def2-SVP | taxol, def2-SVP |
|---|---|---|---|
| `gemm1`: `X`, `Y` | 484 ms, 17.3 TF/s | 256 ms, 14.8 TF/s | 7144 ms, 16.4 TF/s |
| `rho1`: `fxc_rho_occ_mgga` | 47 | 47 | 829 |
| `gemm2`: `fxc_w_occ_mgga` + four GEMMs + scatter | 588 ms, 14.1 TF/s | 291 ms, 12.9 TF/s | 9920 ms, 11.8 TF/s |
| **call** | **1165 ms (5.19x)** | **610 ms (2.43x)** | **18046 ms (1.66x)** |
| flops, matrix → occupied | 95.4 → 16.7 TF | 19.5 → 7.6 TF | 462 → 235 TF |

The GEMMs are still at the rate; `gemm2`'s lower effective rate is the `W`
kernel, which writes `4·nset·nocc·ng` doubles per block (10 GB at 113 atoms)
for the GEMMs to read back — §7 has what that is worth.

### 3.1 The two thresholds, and what was rejected

The meta-GGA ratio, TPSS, function time in ms (`FASTXCFXC_OCC_RATIO`):

| system | basis | ratio 0 (every block) | **1** | 1.5 | 2.5 | W16 |
|---|---|---|---|---|---|---|
| glycine | SVP | 10.6 | 10.6 | 10.9 | 11.6 | 24.9 |
| Pt amine | SVP | 111.8 | 112.2 | 114.3 | 138.0 | 308.1 |
| Ni dithiolene²⁻ | SVP | 164.3 | 166.0 | 177.7 | 284.5 | 285.4 |
| Cu macrocycle⁺ | SVP | 1668.9 | 1669.2 | 1666.8 | 2031.2 | 5315.7 |
| penicillin V | SVP | — | 595.7 | 618.2 | — | 1464.1 |
| Pt amine | TZVPP | 191.2 | 196.7 | 211.4 | 205.9 | 1094.5 |
| Ni dithiolene²⁻ | TZVPP | 293.2 | 284.4 | 294.5 | 297.3 | 1008.5 |

The break-even is `nsub ≈ nocc`, as the flop count says (`8 nocc` against `8
nsub`, plus `X`); 1 is kept rather than 0 so that a block with fewer AOs
than occupied orbitals — which a large molecule in a small basis has, §6 —
stays on the matrix route.

The GGA potential ratio and the GGA density on the occupied side, B3LYP, ms:

| system | basis | W16 | ratio 0 | 2 | **3** | 5 | density too, ratio 0 | density too, 2 |
|---|---|---|---|---|---|---|---|---|
| alanine dipeptide | SVP | 68.8 | 72.4 | 73.8 | 63.5 | 69.5 | 60.1 | 60.1 |
| Pt amine | SVP | 97.2 | 92.2 | 91.0 | 85.2 | 90.2 | 105.0 | 106.5 |
| Ni dithiolene²⁻ | SVP | 82.5 | 92.6 | 85.3 | 84.2 | 84.7 | 148.9 | 97.6 |
| penicillin V | SVP | 425.8 | 413.7 | 410.3 | 406.9 | 423.8 | 511.1 | 481.5 |
| alanine dipeptide | TZVPP | 206.6 | 144.7 | 150.3 | 147.1 | 144.8 | **115.9** | 127.0 |
| Ni dithiolene²⁻ | TZVPP | 281.4 | 245.7 | 233.2 | 242.4 | 293.1 | 263.0 | 244.1 |
| penicillin V | TZVPP | 1421.6 | 974.3 | 984.6 | 972.3 | 1015.4 | 955.0 | 965.4 |

A restricted GGA's density on the occupied side (`FASTXCFXC_OCC_RHO_GGA=1`)
is four nocc-wide `Y` GEMMs against W16's factor route's three: it wins
where `nocc` is small against `nsub` (alanine dipeptide in def2-TZVPP,
`nsub/nocc` ≈ 10) and loses by up to 1.8x in def2-SVP.  It stays in the
module, off.  The potential half is flat between ratio 2 and 3 and 3 is
kept: it is the value at which no row loses to W16 by more than the spread.

---

## 4. The unrestricted Hessian

The grid half of `hessian/uks.py:get_veff_resp_mo` — `nr_uks_fxc` on
`gen_vind`'s tagged pair and both spins taken to the MO basis — on W25's
route (`FASTXCFXC_OCC=0`: the blocked `nr_uks_fxc`, whose `mo1` branch
evaluates the density with upstream's `eval_rho4`, then `_ao2mo`) and on
W33's:

| system | basis | `nocc` α/β | xc | W25 route | **W33** | |
|---|---|---|---|---|---|---|
| Ni(II) dithiolene⁻, 21 | SVP | 105/104 | B3LYP | 426.3 ms | **272.7** | 1.56x |
| Ni(II) dithiolene⁻ | SVP | | TPSS | 583.8 | **343.4** | 1.70x |
| Pt(II) amine triplet, 23 | SVP | 60/58 | TPSS | 641.6 | **232.4** | 2.76x |
| Pt(II) amine triplet | TZVPP | 60/58 | B3LYP | 714.5 | **342.5** | 2.09x |
| Pt(II) amine triplet | TZVPP | | TPSS | 1473.1 | **380.5** | 3.87x |
| Cu(II) macrocycle²⁺ quartet, 60 | SVP | 100/97 | B3LYP | 4404.8 | **2827.4** | 1.56x |
| Cu(II) macrocycle²⁺ quartet | SVP | | TPSS | 8086.9 | **3375.8** | 2.40x |
| FeCl₄⁻ sextet, 5 | TZVPP | 50/45 | B3LYP | 37.2 | **20.5** | 1.81x |
| FeCl₄⁻ sextet | TZVPP | | TPSS | 49.6 | **21.9** | 2.26x |

    python benchmarks/fxc_occ_probe.py --xyz benchmarks/molecules/tmc_cu_macrocycle.xyz \
        --charge 2 --spin 3 --basis def2-svp --xc tpss

On the Cu complex the W25 route's `eval_rho4` is **2.75 s at 8.7 TF/s** for
the density and its potential GEMMs 4.9 s at 15.9 TF/s; W33's `Y` GEMMs are
1.55 s at 15.4 TF/s.  So for a GGA the density half is most of the gain
(B3LYP: 2.84 → 1.56 s) and the potential half, at `nsub/nocc` 3.2, about a
tie (1.11 → 1.10 s).  The same sweep as §3.1 (`FASTXCFXC_OCC_RATIO` 0, 1,
2; `_GGA` 1.5, 3, 5, `benchmarks/fxc_occ_probe.py --ratio --ratio-gga`) puts
1 and 3 at or within 5 % of the best on all four complexes, so the
restricted thresholds carry over.

---

## 5. Whole Hessians

`benchmarks/df_hess_anatomy.py --patch all --rounds 2`, before
(`FASTXCFXC_OCC=0`) and after alternating in separate processes, two pairs,
the second round of each (taxol: `--rounds 1`, each process warmed by a
glycine Hessian first):

| system | xc | before (pair 1, 2) | after (pair 1, 2) | | CPHF grid half, 6 calls, before → after | traces |
|---|---|---|---|---|---|---|
| penicillin V, 42 | TPSS | 12.53, 12.45 s | **8.62, 8.64 s** | **1.44x** | 6.65 → 2.78 s | 40.9142598200 / …198 |
| penicillin V, def2-TZVPP | TPSS | 40.33, 40.56 | **17.43, 17.57** | **2.31x** | 28.45 → 5.60 s | 38.8754196468 / …460 |
| taxol, 113 (round 0) | TPSS | 257.29, 256.34 | **190.48, 189.56** | **1.35x** | 5 calls: 135.5 → 71.6 s | 108.3279512880 / …877 |
| Ni(II) dithiolene⁻, UKS | TPSS | 9.29, 9.17 | **7.86, 8.32** | 1.14x | 8 calls: 3.29 → 1.99 s | 23.0879804021 / …3960 |
| Pt(II) amine triplet, UKS | TPSS | 10.01, 9.87 | **8.43, 8.48** | 1.18x | 7 calls: 2.66 → 1.22 s | 19.9589432158 / …163 |
| penicillin V, def2-TZVPP | B3LYP | 40.81, 39.95 | **38.90, 39.38** | 1.03x | 8.05 → 5.40 s | 39.7773170386 / …385 |
| Ni(II) dithiolene⁻, UKS | B3LYP | 14.75, 14.31 | 14.35, 14.39 | 1.00x | CPHF 9.13 → 8.59 s (grid half 2.3 s before; `_get_jk` 5.7) | 22.0335904214 / …257 |

(The UKS rows are the second of two jobs, which agreed to 5 %; the Ni
trace spreads by 6e-9 before and after alike, the `ip2` atomics W28
recorded.  The B3LYP row is a second job: in the first, one "after" process ran 20 s
slower than every other B3LYP run while its CPHF row did not, on a node
whose other GPUs were busy.  The anatomy's unrestricted `nr_uks_fxc` row
now also counts W33's route, which does not call `nr_uks_fxc`; its first
version did so by wrapping `_nr_uks_fxc_blocked`, which the replaced method
then failed to recognise by identity and handed back to upstream — the
method now tests the mode by name.)

The def2-TZVPP TPSS Hessian was 70 % grid response and is now 32 %; the six
calls are 5.6 s against B3LYP's 5.0 s.  For a B3LYP or UKS Hessian the CPHF
is `_get_jk` (W8's), which this does not touch.

---

## 6. Where the gain depends on the system

The saving per block is `8 nset (nsub − nocc)` per grid point against `8
nset nsub`: the route's margin is `nsub/nocc`, grid-weighted over the
blocks.  From the probe:

| case | `nsub` mean / max | `nocc` | `nsub/nocc` | W33 / W16 |
|---|---|---|---|---|
| penicillin V, def2-TZVPP | 499 / 800 | 92 | 5.43 | 5.19x |
| penicillin V, def2-SVP | 226 / 367 | 92 | 2.46 | 2.43x |
| taxol, def2-SVP | 400 / 701 | 226 | 1.77 | 1.66x–1.92x |
| Ni dithiolene⁻, def2-SVP, UKS | 188 / 250 | 105 | 1.79 | 1.70x |

Two dependences follow, and both are real rather than tuning artefacts.
**Basis**: a larger basis raises `nsub` at fixed `nocc`, so def2-TZVPP gains
3.5x–7.5x where def2-SVP gains 1.6x–3.2x.  **Molecule size**: in a fixed
basis `nsub` is set by the grid block's neighbourhood and saturates (400 at
113 atoms against 226 at 42 in def2-SVP), while `nocc` grows with the
molecule, so the margin falls — 2.4x at 42 atoms, 1.7x–1.9x at 113 — and a
molecule with `nocc` above its typical `nsub` (the 284-atom PfPMT in
def2-SVP has `nocc` = 547) would put most blocks back on the matrix route.
That is the ratio test doing its job — the route never loses — but it means
the win is largest on the Hessians people run in triple-zeta and smallest on
the largest systems.  Chemical class enters only through these two numbers:
the metal complexes sit on the same curve as the organics (Ni²⁻ in def2-SVP,
`nocc` 105 at 327 AOs, is the lowest restricted row at 1.56x).

### 6.1 Memory

The first taxol Hessian with the route ran out of memory in `_occ_setup`.
W16's code allocated its buffers up front for the largest block — at 113
atoms and `nset` = 339, `c0` alone is `4·nset·nsub_max·blk` doubles, 31 GB,
and `aow` 7.8 GB — and the route's own buffers went on top.  They are now
allocated at the first block that takes the matrix route, sized for the
largest block that will (`nsub ≤ nocc` for a meta-GGA), and `c0` and a
meta-GGA's `aow` live in the route's `Y` buffer, idle on those blocks.  At
113 atoms that is 10 GB of `Y`, 1.4 GB of `M^T` and `V C_o`, and small
change, against 41 GB before.  The rule is §5.8's in another costume: a
second route must not stack on the first one's buffers.

---

## 7. What is left, and what it becomes

* **A local `nocc`.**  §6's size dependence is `nocc` being global: every
  block contracts over every occupied orbital although, with localised
  orbitals, only those with support on the block's AOs contribute to `X`,
  and `rho = sum_k Y_k X_k` then needs `Y` only for those.  The rotation is
  free — `D` and `V C_o C_o^T` are invariant under `C_o → C_o U`, `M → M U`
  — and W15's localisation (`docs/README_fastdfunpack.md`) measured the
  supports at 284 atoms.  That would make the route's margin
  `nsub/nocc_local` and stop it falling with the molecule.
* **The `Y`/`W` round trip.**  At 113 atoms the route writes and reads back
  `4·nset·nocc·ng` doubles twice per block (Y for the density kernel, W for
  the GEMMs), about 40 GB per block; it is why `gemm2` runs at 11.8 TF/s
  there against the GEMMs' 16.5.  Fusing the density contraction into the
  `Y` GEMM's epilogue, or tiling the block's grid so `Y` stays in L2, is
  worth up to the 20 % of the call that gap is.
* The remaining CPHF time in B3LYP and UKS Hessians is `_get_jk` (W8).

---

## 8. Correctness

`tests/test_fxc_occ.py` (new), relative to the largest element:

| | cases | worst |
|---|---|---|
| restricted `nr_rks_fxc_mo` against GPU4PySCF's AO `nr_rks_fxc` + `_ao2mo` | ethanol TPSS/def2-TZVPP, M06-2X, PBE; glycine M06-L; Pt amine r2SCAN (ECP); Ni dithiolene²⁻ B3LYP — each on every block, the default share, a GGA's density too, and none | 3.7e-15 |
| unrestricted `get_veff_resp_mo` against upstream's with GPU4PySCF's `nr_uks_fxc` | CH₃ TPSS (6-31G*) and PBE0 (def2-TZVPP), O₂ M06-2X, ethanol⁺ r2SCAN, FeCl₄⁻ TPSS, Ni dithiolene⁻ B3LYP, Pt amine triplet M06-L — every block, default, none | 3.2e-15 |

and the route was taken (`occ_blocks`, `occ_u_calls` asserted non-zero).
`tests/test_dfhessip.py`'s CPHF section gained r2SCAN and the every-block /
none variants for LDA, GGA and meta-GGA at 1, 3 and 6 right-hand sides
(≤ 1.3e-15), and still passes its whole-Hessian check (7.9e-12);
`tests/test_ufxc.py`'s whole unrestricted Hessians, which compared the
patched `nr_uks_fxc` against GPU4PySCF's, now swap the replaced CPHF method
back too for the reference, and pass 4/4 inside the Hessian's own
run-to-run spread; `tests/test_fxc.py` 18/18.  Whole-Hessian traces are in
§5.
