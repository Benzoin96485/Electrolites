# Open-shell transition-metal complexes

Everything in this repository before this round was measured on closed-shell
organic systems -- peptides, drug molecules, a 545-atom cluster -- in Pople and
def2 basis sets that stop at d or f.  This document is what happened when the
same modules were pointed at **open-shell transition-metal complexes**, which
differ in three ways that turn out to matter:

* **The ground state is unrestricted.**  Of the 2 824 complexes in the
  `by_ccd` test split of BOS-TMC-28242, **2 344 (83.0 %)** have a multiplicity
  above 1: 485 triplets, 484 doublets, 476 quartets, 460 sextets, 439
  quintets.  A UKS SCF is the *common* case there, not a corner.
* **The basis reaches f at double-zeta quality.**  def2-SVP puts one f shell on
  a 3d metal, so every TMC case here has `lmax = 3` where an organic def2-SVP
  or 6-31G\* case has `lmax = 2`.  def2-TZVPP and def2-QZVP reach g -- and §4
  item 7 is what two of the modules do when they get there.
* **The metal's s shells are long contractions.**  def2-TZVP contracts the 3d
  metals' innermost s shell over **8** primitives, and S and Cl -- the two
  commonest donor atoms after N and O -- over 7, against a maximum of 6
  anywhere in a Pople set.

The test cases are in `benchmarks/molecules/tmc_*.xyz`, with their charge,
multiplicity and source record in `benchmarks/molecules/tmc_cases.txt`.  They
were chosen to span the three axes above at a size that can be A/B'd in
minutes: a 3-atom doublet and a 5-atom sextet for the arithmetic, a 21-atom Ni
dithiolene and a 23-atom Pt complex (**with an ECP**, which no earlier case
here has) for a realistic ligand field, and a 60-atom Cu macrocycle and a
113-atom Co complex so that any ratio can be read at more than one size.

**This document is a round's own record and is not kept up to date as a
status page.**  Five of the eight gaps its §4 ranks have since been taken;
§4's status block says which, where the work is, and what each of those
rounds found this list had wrong about it.  Everything else here is left as
it was measured.

## 1. What was actually on the code path, and what was not

The modules fall through to GPU4PySCF for whatever they do not cover, which is
why a patched run is correct.  It also means a module can be **inert** on a
code path with nothing saying so: the run is simply as fast as an unpatched
one.  `benchmarks/coverage_probe.py` was written to make that visible.  It
counts, per entry point, how many calls our code answered and how many it
handed back, and for the four two-electron modules how many
angular-momentum class blocks went each way.

Two things it found, on a UKS B3LYP/def2-SVP SCF and gradient:

**`grad.uks` and `dft.uks` bind their own copies of the functions these
modules patch.**  `gpu4pyscf.dft.uks.get_veff` calls `NumInt.nr_uks`, not
`nr_rks`; `gpu4pyscf.grad.uks.energy_ee` calls `grad/uks.py`'s own module-level
`get_exc`, not `grad.rks`'s; `grad.uks.Gradients.energy_ee` is `grad/uks.py`'s
own function, not `grad.rks.Gradients`'s.  So on an open-shell ground state
`fastxc`, `fastxcgrad`, `fastgrad`, `fastxcnlc` and `fastrsh` were all patching
functions that were never called -- the entire XC half of the SCF *and* of the
gradient ran GPU4PySCF's own code.

**`fastk` was never the problem.**  The README said `fastk` "is spin-restricted
closed-shell only: an unrestricted calculation reaches them with two density
matrices and is handed back".  It is not: `_get_k_dispatch` has no spin test at
all, and on the UKS SCF above it answered 16 of 16 calls and 25 631 of 25 631
class blocks.  The exchange build has always covered the unrestricted case --
the density loop is inside the shell quartet and does not care what the two
densities mean.  Nothing was wrong with the code; the documented limit was
wrong, and it is the kind of claim that stops anyone from measuring.

Here is the whole table, for a UKS B3LYP/def2-SVP SCF *and* gradient on the
3-atom CuCl2 doublet, before and after this round.  `(-)` means the entry point
was never called at all; the indented rows count angular-momentum class blocks
inside the call, ours against GPU4PySCF's.

| entry point | before: calls / ours | after: calls / ours |
|---|---|---|
| `fastk` `get_k` | 16 / 16 | 16 / 16 |
| `fastj` `get_j` | 16 / 16 | 16 / 16 |
| `fastxc` `nr_rks` | (-) | (-) |
| **`fastxc` `nr_uks`** | **(-)** | **16 / 16** |
| `fastejk` `_jk_energy_per_atom` | 1 / 1 | 1 / 1 |
| **`fastxcgrad` `grad.uks.get_exc`** | **(-)** | **1 / 1** |
| `fastgradh` `get_hcore` | (-) | (-) |
| `fastgrad` `energy_ee` | (-) | (-) |
| `fastxcnlc` `RKS.get_veff` | (-) | (-) |
| `fastxcfxc`, `fastxchess`, `fastfnlc` | (-) | (-) |
| ↳ `fastk` class blocks | 25631 / 25631 | 25631 / 25631 |
| ↳ **`fastejk` class blocks** | **1649 / 0** | **1649 / 1649** |

The three rows that stay `(-)` are not bugs and are not fixed here: `nr_rks`
because this is a UKS run, `get_hcore` because no 1.8.1 gradient calls it, and
the response and Hessian entries because nothing in this run reaches them.
`fastgrad` and `fastxcnlc` stay `(-)` because `grad/uks.py` and `dft/uks.py`
have their own `energy_ee` and `get_veff`; §4 says what that costs.

## 2. Corrections to the README

| the README said | measured |
|---|---|
| `fastk` … "spin-restricted closed-shell only: an unrestricted calculation reaches them with two density matrices and is handed back" | `fastk` covers UKS in full: 16/16 calls, 25631/25631 class blocks, on a UKS SCF |
| "the four gradient modules are spin-restricted closed-shell only" | was true of `fastxcgrad` and `fastgrad` for a namespace reason and of `fastejk` for an explicit `n_dm != 1` guard -- both fixed here for `fastxcgrad` and `fastejk`, `fastgrad` still restricted-only.  `fastgradh` was never reached by *any* 1.8.1 gradient, restricted or not |
| `fastgradh` … "the nuclear-attraction part of the one-electron derivative, on the GPU" | on 1.8.1 `grad_elec` takes `_hcore_energy`, which calls `_grad_nuc_without_ecp` and `get_ecp_ip` directly; `GradientsBase.get_hcore` is reached only from `get_grad_hcore`, i.e. the Hessian's CPHF right-hand side.  The module is inert on a gradient on this release |
| "a 6-31G\* job's first run pays about 36 s of JIT rather than 98 s … A basis with f functions reaches every class and pays for all of them" | still true, but **every** TMC case pays the full cost: def2-SVP already reaches f on the metal, so the split buys nothing on this class of system |
| `fastxc` … "hands **meta-GGA** back in the SCF" | now covered on the unrestricted side (§3); still handed back by `nr_rks`, where `fastxcnlc` reaches it instead |

## 3. What was added

### `fastxc`: `NumInt.nr_uks`

`nr_uks` had no replacement at all.  GPU4PySCF's own copy has exactly the
structure `fastxc` was written against -- one pass over the grid for rho and a
second for the potential, one GEMM per block, one block issued at a time --
and it does all of it **twice per block**, once per spin.  So the three changes
in `fastxc`'s docstring carry over, and the third one (batching the GEMMs
across grid blocks so a `nao_sub x nao_sub` product does not leave two thirds
of the device idle) carries further: both spins of every block in a group go
into one grouped launch, which is `2*nblocks` tiles where GPU4PySCF issues one
GEMM at a time.

Meta-GGA is covered here even though `nr_rks` hands it back, and the reason is
asymmetry in the rest of the repository rather than in the arithmetic: on the
restricted side `fastxcnlc` reaches meta-GGA through `RKS.get_veff`, and there
is no unrestricted counterpart of that module.  Without it, a complex run with
M06-2X, TPSSh or wB97M-V -- the usual choices for a transition metal -- would
get no XC acceleration at all.  The tau term is three more
`nabla_m phi . (nabla_m phi w)` products per spin per block, and they go into
the same grouped launch as the GGA one.

One difference that is not a speed question: `nr_uks` allocates `rho` for the
**whole grid** before calling libxc, `2*nvar*ngrids` doubles at once.  On the
113-atom complex at level-3 grids that is 1.2 GB where the pass here holds one
group, about 60 MB.

### `fastxcgrad`: `grad.uks.get_exc`

The unrestricted XC gradient is the restricted one run once per spin, with
libxc called *per grid block* -- and the docstring's change 2 says that call is
the largest single item in the term, 42 % of the GGA time on the systems it was
measured on.  An unrestricted gradient makes two such calls per block.  The
module now patches `grad.uks.get_exc` as well as `grad.rks`'s, with the loop
body unchanged and a spin index on the density products, the potential product
and the weights; the one libxc call still covers a whole group, and now both
spins of it.

The VV10 term of an unrestricted gradient still falls through:
`grad.rks.get_nlc_exc` is reached with two densities and hands them back.


### `fastejk`: two density matrices

`_usable` carried `if n_dm != 1: return False`, so on an open-shell gradient
**1 649 of 1 649** angular-momentum class blocks went to GPU4PySCF -- the
module was loaded, correct and inert.  The kernels contracted one density
matrix; they now run the exchange contraction over the `n_dm` matrices and the
Coulomb one over their sum, which is the shape both references need:

    E_J  = 1/2 Tr[D_t J(D_t)]                      D_t = D_a + D_b
    E_x  = -1/2 sum_s Tr[D_s K(D_s)]               (-1/4 Tr[D_t K(D_t)] closed shell)

The factor 2 between the two exchange expressions is the `-2*k_factor` the
driver already passed at `n_dm = 2` and never used.

The loop bound is a **preprocessor definition** (`-DEJK_NDM=`), not a kernel
argument.  Written as an argument it is a runtime bound around an unrolled
straight-line contraction, and the closed-shell path -- the common one, and the
one every earlier measurement in this repository rests on -- would have paid
for the open-shell one in register pressure it cannot be shown not to pay.  As
a macro, `EJK_NDM = 1` is the kernel that was there before.  The price is a
second NVRTC compile of the largest generated family in the package (6.8 MB of
CUDA), which only an open-shell run pays and which CuPy caches on disk on the
source hash, so it is a first-run cost per machine rather than per process.

`benchmarks/ejk_ndm.py` times both `n_dm` against GPU4PySCF in one process, so
the closed-shell column is a regression check rather than an assumption and the
open-shell column is what taking the case back is worth.

One thing to leave alone: the **screening estimate stays GPU4PySCF's**, the max
over the spin densities rather than over their sum.  Widening it to cover the
total density is defensible -- `|D_a + D_b|` is the largest factor the kernel
forms -- and it moves the agreement with GPU4PySCF from 4e-14 to 7e-12
*relative*, because the extra quartets it keeps are exactly the ones GPU4PySCF
drops.  The estimate is low by at most `log 2` against a `1e-13` cutoff, which
is inside the margin the cutoff already carries, and matching the reference is
worth more than being marginally tighter than it.

### Correctness

`tests/test_uxc.py` (19 cases plus the fall-through check),
`tests/test_uxcgrad.py` (10) and `tests/test_uejk.py` (8) call both
implementations on the same converged density in one process, and the first two
do it in both the grouped mode and the per-block ablation.  Over LDA, GGA,
meta-GGA and range-separated meta-GGA, at def2-SVP,
def2-TZVP, def2-TZVPP (which reaches g) and 6-31G\*, with and without an ECP:

* the potential matrix agrees to **6.7e-15 relative** at worst, and the
  per-block route is bit-identical to GPU4PySCF's on most cases;
* the XC gradient term agrees to **2.7e-13 relative** at worst -- 8.0e-14 at
  def2-TZVPP, 4.4e-14 on the Pt complex, 5.9e-15 for M06-2X and 3.6e-15 for
  TPSS, i.e. the meta-GGA branch this module already had is right for a pair of
  spin densities too;
* the two-electron per-atom derivative agrees to **2.2e-14 relative** at
  `n_dm = 2` and **5.1e-14** at `n_dm = 1`, checked with `(j, k)` = (1, 0.2),
  (1, 0) and (0, 0.25) so that a factor error in either term would show in only
  one column;
* a whole patched UKS SCF energy matches an unpatched one, in separate
  processes, to **every digit printed** on FeCl4(-)/def2-SVP/B3LYP
  (-3104.15150513555 both ways, gradient norm to 4e-13), to **2.3e-12 Eh** on
  the 21-atom Ni dithiolene, **1.9e-12** on the Pt complex (i.e. with an ECP)
  and **8.2e-13** on FeCl4(-)/def2-TZVPP (i.e. with g shells present), with
  gradient norms agreeing to 1e-12 or better throughout.

One negative result worth keeping: CuCl2/def2-SVP/**M06-2X** does not converge
in 60 UKS cycles either patched or unpatched, so the two runs stop at different
points of an oscillating path and their energies differ at 4e-7.  That is a
statement about the SCF, not about the kernels -- the same functional agrees to
6.7e-15 on the potential matrix and 5.9e-15 on the gradient term when both
implementations are handed the *same* density in one process, which is why
those are the columns the tests check.

## 4. What is still not covered, ranked

Everything below falls through and is correct; the cost is that the run is as
fast as an unpatched one on that term.

> ### Status: five of these eight have since been taken
>
> The list, its wording and its numbers are left exactly as this round wrote
> them, so that what it measured stays what it says it measured.  What is
> added is a marker on each entry that has been done, saying where the work
> is, what it turned out to be worth, and what that round found **this list**
> had wrong about it.  The corrections live in those documents and are not
> edited into the entries below.
>
> | # | entry | taken in | what it came out at |
> |---|---|---|---|
> | **1** | the unrestricted response path | `docs/OPENSHELL_FXC.md` | 2.2x-11.9x on the term; 5.5x in situ in an open-shell TDA; 1.9x in an unrestricted Hessian's CPHF |
> | **2** | the unrestricted Hessian's XC terms | `docs/OPENSHELL_HESSIAN.md` | 6.7x-8.8x on the two terms together; 8.8 % of a stock UKS Hessian |
> | **3** | `uks.get_veff` has no `fastxcnlc` | `docs/VV10_OPENSHELL.md` | two of four grid passes removed, 1.6e-7-2.5e-7 s per grid point -- reported as a *comparison*, because it competes with `fastxc`'s grouping rather than adding to it |
> | **4** | the unrestricted VV10 gradient | `docs/VV10_OPENSHELL.md` | 1.17x-1.70x on the term, **2.7x-3.6x** on the grid passes it actually replaces |
> | **5** | `hessian.uks.nr_uks_fnlc_mo` | `docs/VV10_OPENSHELL.md` | 1.47x-1.63x on a path that was 1.00x by construction |
> | 6, 7 | `MAX_PRIM_PAIR`, and g shells in `fastj`/`fastejk` | — | open; both are basis-set limits rather than spin ones |
> | 8 | first-run JIT | — | open, and unchanged: the answer is a warm cache, not a code change |
>
> **Three of the five needed a patch point this list either did not name or
> named wrongly**: item 3's module-level `dft.uks.get_veff` alongside the
> class attribute, item 4's `grad.uks.get_nlc_exc`, and item 5's binding,
> which is `hessian.uks.nr_rks_fnlc_mo` and not the name given here.  One
> pattern is behind all three -- `dft/uks.py`, `grad/uks.py` and
> `hessian/uks.py` each bind their own copies of what looks like a single
> function -- and a patch that misses the copy is **silent**: the answer stays
> right and only the wall clock says anything.  `tests/test_patch_aliases.py`
> is the check that keeps that visible, and §1 of this document is the round
> that learned it.

1. **The unrestricted response path.**  `fastxcfxc` replaces `nr_rks_fxc`;
   `nr_uks_fxc` has no replacement, so the grid half of *every* open-shell
   polarizability, TDA/TDDFT, NMR and C6 run is stock.  `fastjk` is already on
   the two-electron half of it (it takes any number of densities and does not
   care about spin), so this is the one remaining half.  The arithmetic is
   `nr_rks_fxc`'s per spin with `spin=1` in the libxc call, i.e. the same
   relation `nr_uks` bears to `nr_rks`.

   > **Taken** -- `docs/OPENSHELL_FXC.md`.  **2.2x-11.9x** on the term over a
   > sweep of `nset`, at two sizes and in two chemical classes; **5.5x** in
   > situ inside a five-root open-shell TDA, which is 1.09x-1.16x on the whole
   > run because the XC half is 10.5 % of an open-shell TDA where it is 25 %
   > of a closed-shell one.  Its §1 records two things this entry has wrong.
   > **The arithmetic is not "`nr_rks_fxc`'s per spin":** `fxc` is a 2x2 block
   > in the spin index, so running the restricted function once per channel
   > would drop the off-diagonal blocks and give a wrong answer, not a slower
   > one.  And **the list of consumers is missing the analytic Hessian**:
   > there is no `nr_uks_fxc_mo`, so an unrestricted Hessian's CPHF calls this
   > same function, which makes this entry and the first item of
   > `docs/OPENSHELL_HESSIAN.md` §5 the same target named twice without
   > either document noticing.
2. **The unrestricted analytic Hessian's XC terms.**
   `hessian.uks._get_vxc_deriv1` and `_get_vxc_deriv2` are `hessian.rks`'s
   twins with a spin index, and `fastxchess` patches only the `hessian.rks`
   pair.  `fastxchess` is 7.3x on
   those terms at 42 atoms, so this is the largest single number in the list
   -- but a Hessian of an open-shell complex is a rarer thing to want than a
   gradient, which is why it is not first.

   > **Taken** -- `docs/OPENSHELL_HESSIAN.md`.  **6.7x-8.8x** on the two terms
   > together across four systems and two chemical classes, *above* the
   > restricted 7.25x rather than equal to it, because GPU4PySCF pays 2.38x
   > for the second spin channel where the port pays 1.96x.  Its §1 records
   > two things this entry has wrong.  **"Twins with a spin index"
   > understates it** in the one place that matters: `fxc` couples the
   > channels, so the two spins cannot be run as independent passes and have
   > to share a grid block -- which is what makes the port beat the restricted
   > ratio.  And **the 7.3x quoted here is the restricted number**, which is
   > not evidence about a different implementation of a different function; it
   > came out at 8.82x on the same molecule and 6.69x at 60 atoms.
3. **`uks.get_veff` has no `fastxcnlc`.**  On the restricted side that module
   fuses the XC and VV10 grid sweeps into one; on the unrestricted side the two
   sweeps stay separate, so a UKS wB97M-V walks the grid twice.  `fastnlc`
   already covers the double sum itself for UKS -- `nr_nlc_vxc` sums the
   spin densities before it -- so what is left is the second sweep of `eval_ao`
   and the second density evaluation.

   > **Taken** -- `docs/VV10_OPENSHELL.md` §2, §4b.  The patch point was
   > `dft.uks.UKS.get_veff` **and** the module-level `dft.uks.get_veff`, which
   > this entry does not name.  Reported as a comparison rather than a
   > speedup: the fusion and `fastxc`'s grouped `nr_uks` remove cost from the
   > same two grid passes from opposite directions and do not compose, so
   > `FASTXCNLC_OFF=1` exists for a user to A/B on their own system.
4. **The unrestricted VV10 gradient.**  `grad.rks.get_nlc_exc` is reached with
   two densities and hands them back; the fix is `get_exc_u`'s, a spin loop
   around the same body.

   > **Taken** -- `docs/VV10_OPENSHELL.md` §2, §4a.  **1.17x-1.70x** on the
   > term and **2.7x-3.6x** on the grid passes it actually replaces, stable
   > over a 20x range of system size, two chemical classes, three basis sets
   > and both functional types.  Its §1 records that **there is no spin
   > loop**: VV10 is a functional of the total density, so the fix is
   > `D_a + D_b` and nothing else -- and that the term this entry calls a
   > fall-through is GPU4PySCF's own two-density branch rather than nothing at
   > all, so the ratio is against a real baseline.
5. **`hessian.uks.nr_uks_fnlc_mo`.**  Two problems at once: `fastfnlc`
   declines UKS, *and* `hessian/uks.py` imported `nr_rks_fnlc_mo` by name at
   import time, so even a UKS-capable `fastfnlc` would not be reached without
   patching that binding too.  `tests/test_patch_aliases.py` is the check that
   keeps this kind of thing visible.

   > **Taken** -- `docs/VV10_OPENSHELL.md` §2, §4c.  **1.47x-1.63x** on a path
   > that was 1.00x by construction.  Its §1 records that the binding is
   > `hessian.uks.nr_rks_fnlc_mo`, not `nr_uks_fnlc_mo` -- the forwarding
   > function is fine and the stale name is the one it forwards *to* -- and
   > that this path has a caller the restricted one does not: `tdscf/uhf.py`
   > sets no `exclude_nlc`, so an open-shell TDDFT pays the VV10 response term
   > on every Davidson iteration and a closed-shell one pays none of it.
6. **`MAX_PRIM_PAIR = 36`, "6-prim x 6-prim is the worst case in Pople sets".**
   It is not the worst case in a def2 set for a metal: def2-TZVP contracts a 3d
   metal's innermost s shell over 8 primitives and an S or Cl ligand's over 7,
   so `(s,s)` bra pairs of 49, 56 and 64 exist and every quartet built on them
   falls through.  Measured on the CuCl2 doublet, which is the smallest case
   here and so the one where a handful of core `(s,s)` pairs weighs most:

   | basis | `nprim` max | `fastk` class blocks ours / total | `fastejk` |
   |---|---|---|---|
   | def2-SVP | 6 | 25631 / 25631 | 1649 / 1649 |
   | def2-TZVP | 8 | 97327 / 97852 (**525 fall through**) | 4943 / 4968 |

   So the bound does fire, on exactly the basis a transition-metal calculation
   is usually run in, and on nothing in this repository's Pople-set test suite.
   0.5 % of the class blocks is not 0.5 % of the *work* -- these are compact
   core s pairs and the blocks differ enormously in cost -- so this is on the
   list because the comment is wrong about the basis sets a user will actually
   run, not because it has been shown to cost anything.  Raising the bound
   costs shared memory (six `double[MAX_PRIM_PAIR]` arrays per block) and
   should not be done without measuring what it buys.
7. **g shells: `fastj` and `fastejk` hand them back.**  This is the other half
   of "the basis reaches further than the test suite does", and it is not about
   spin at all.  `fastk` covers g: on CuCl2/def2-TZVPP (`lmax = 4`) it answered
   91 780 of 92 305 class blocks, the 525 being the `MAX_PRIM_PAIR` pairs of
   item 6 and nothing to do with the angular momentum.  The other two do not:

   | basis | `lmax` | `fastk` ours / total | `fastj` fell through | `fastejk` ours / total |
   |---|---|---|---|---|
   | def2-SVP | 3 | 25631 / 25631 | 0 | 1649 / 1649 |
   | def2-TZVP | 3 | 97327 / 97852 | 0 | 4943 / 4968 |
   | **def2-TZVPP** | **4** | 91780 / 92305 | **672 blocks** | **3613 / 4860 (26 % stock)** |

   Both numbers are zero until a g shell appears and jump the moment one does,
   so this is an angular-momentum limit rather than a screening one: `fastejk`'s
   `IMPLEMENTED` table has no entry for a g-containing quartet, and `fastj`'s
   written classes stop before them too (the README says as much for the
   multi-density family -- "the classes past `(4,4)`" -- but the single-density
   one has the same edge).  def2-TZVPP is not exotic for a transition metal, and
   def2-QZVP puts 53 g shells on the 113-atom complex where def2-TZVPP puts one.
   Whether writing them is worth it is a question for
   `benchmarks/ejk_ip2_probe.py`'s per-class treatment applied to `ip1`, not one
   to answer from the block counts: 26 % of the blocks at one g shell per metal
   is not 26 % of the time.
8. **First-run JIT.**  The README's split -- compile only the angular-momentum
   classes the basis can reach -- is what keeps a 6-31G\* job's first run at
   about 36 s instead of 98 s.  A def2-SVP TMC reaches f on the metal and so
   pays the full 98 s, and every def2 basis a transition metal is run in does.
   The split is not wrong, it just does not fire on this class of system; a
   shared `ELECTROLITES_GENERATED_DIR` and a warm CuPy cache is the answer for
   a dataset run, not a change to the split.

## 5. Measurement

Every number here is from a batch allocation with **one** GPU visible to the
process.  Getting that right took two attempts and both failures are worth
recording, because each produced a plausible-looking number that was wrong:

* On a **shared login node** at 100 % utilisation the same stock Coulomb build
  measured 10.8 s in one process and 77.3 s in the next, and the XC build
  measured faster unpatched than patched.  Nothing measured there is usable.
* On a batch node, `--qos=shared -G 1` **does not hide the node's other GPUs**.
  gpu4pyscf reads `getDeviceCount()`, so GPU4PySCF's own build distributed
  itself over four devices while ours -- which asserts a single device -- ran
  on one.  That comparison came out at 1.06x and looked like a regression.
  `export CUDA_VISIBLE_DEVICES=0` in the job script is the fix, and `${VAR:-0}`
  is not enough because Slurm sets the variable to the list of allocated GPUs.

### The two-electron per-atom derivative, both implementations in one process

`benchmarks/ejk_ndm.py`, same density and same `_VHFOpt`, minimum of three,
Ni dithiolene/def2-SVP (327 AOs):

| `n_dm` | GPU4PySCF | `fastejk` | ratio |
|---|---|---|---|
| 1 (closed-shell path) | 5.77 s | 1.91 s | **3.02x** |
| 2 (unrestricted) | 5.78 s | 1.89 s | **3.06x** |

The `n_dm = 2` row was 1.00x before this round by construction -- every class
block fell through.  The `n_dm = 1` row is the regression check on the
compile-time density loop, and it sits above the 1.64x-2.15x
`docs/README_fastejk.md` records for this build on other systems rather than
below it.

### A UKS SCF and gradient, stage by stage

`benchmarks/scf_anatomy.py --spin ...` run twice, `--patch ''` then
`--patch all`, B3LYP/def2-SVP, level-3 grids, both converging in the same
number of cycles:

| stage | Ni dithiolene, 21 atoms / 327 AO | Cu macrocycle, 60 atoms / 515 AO |
|---|---|---|
| **whole SCF** | 101.9 -> 22.0 s **4.64x** | 72.6 -> 25.6 s **2.84x** |
| `..get_k` | 90.1 -> 15.4 s 5.86x | 60.7 -> 18.1 s 3.36x |
| `..get_j` | 7.52 -> 3.06 s 2.46x | 6.49 -> 2.88 s 2.26x |
| `..nr_uks` | 2.93 -> 2.12 s **1.38x** | 3.68 -> 3.13 s **1.17x** |
| **whole gradient** | 7.84 -> 3.05 s **2.57x** | 9.60 -> 4.94 s **1.94x** |
| `grad ejk` | 5.86 -> 2.71 s 2.16x | 8.53 -> 4.56 s 1.87x |
| `grad xc` | 1.81 -> 0.173 s **10.5x** | 0.888 -> 0.197 s **4.5x** |

Three things to take from this table rather than from the headline.

**`nr_uks` is the weakest row, at 1.17x-1.38x.** That is well below what the
restricted side of `fastxc` gets, and it is the module this round was written
for.  The reason is visible in the same table: the exchange build is **83-88 %
of a UKS SCF** on these systems and `nr_uks` is **3-5 %**, so the XC build is
not latency-bound here the way it is on the organic clusters `fastxc` was
tuned against -- there is simply less idle device to fill.  The work was still
worth doing (it is the difference between 3 % of the SCF being free and not),
but the claim that the unrestricted build should beat the restricted one
because it issues twice the GEMMs is **not** what the measurement says, and
`docs/README_fastxc.md` should not be read as predicting otherwise for a
transition-metal complex.

**`grad xc` is the strongest row, at 4.5x-10.5x**, and it is the one that was
running GPU4PySCF's code entirely because of the namespace miss.  Its share of
the stock gradient is 9-23 %, so it moves the whole-gradient number by less
than its own ratio suggests.

**The ratio falls with size on every row.** 4.64x to 2.84x on the SCF, 2.57x to
1.94x on the gradient, over a 1.6x change in AO count.  Two points is not a
scaling law; it is enough to say the numbers above are not a ceiling that holds
at 500 atoms.

### One thing that is *not* a regression, and how it was checked

On a cold node the patched **closed-shell** glycine SCF measured 21.5 s against
6.4 s stock, `get_j` alone going 0.45 -> 8.56 s.  That is first-run NVRTC: this
basis is `lmax = 2` and the TMC cases ahead of it in the same job are
`lmax = 3`, and `_module` is keyed on `lmax`, so it is a fresh compile inside
the timed region.  Running the same patched job three times on one node
confirms it -- 5.14 s, 3.14 s, 5.89 s against 6.51 s stock, nowhere near 21.5 s
-- rather than leaving it to the explanation being plausible, which is how the
two failures at the top of this section got past a first reading.

That check leaves one real observation: `fastj`'s `get_j` is consistently
*slower* than GPU4PySCF's on this case -- 0.93, 1.30, 1.11 s against 0.48 s
across the three warm runs.  It is a 95-AO closed-shell system where Python
launch overhead dominates, no part of it is what this round touched, and the
tables `fastj` was measured on are 284 and 545 atoms; it is recorded here
because it turned up, not as a finding about `fastj`.

### Shares, for what to attack next

Stock UKS gradient, same two systems: `grad ejk` is 74.7 % and 88.8 % of it,
`grad xc` 23.1 % and 9.3 %.  Stock UKS SCF: `get_k` is 88.4 % and 83.6 %,
`nr_uks` 2.9 % and 5.1 %.  On a transition-metal complex the exchange build is
the whole game and the XC build is a rounding error -- the opposite of the
balance on the organic clusters in the README, where the XC build is a third of
the SCF.  Anyone taking the next item off §4's list should check the share on
their own system first.

## 6. Reproducing this

The cases are in the repository; the dataset is not needed to re-run any of it.

One thing to get right first on a cluster: **every module here asserts a single
device**, and a batch allocation that asks for one GPU does not necessarily
hide the node's others -- gpu4pyscf reads `getDeviceCount()`, not the
allocation.  A whole job of this can fall over on `fastk currently supports a
single device` with nothing else wrong; `export CUDA_VISIBLE_DEVICES=0` in the
job script is the fix.

```bash
# what is on the code path of a given calculation, and what is not
python benchmarks/coverage_probe.py --xyz benchmarks/molecules/tmc_fecl4.xyz \
    --charge -1 --spin 5 --basis def2-svp --xc b3lyp
python benchmarks/coverage_probe.py --xyz benchmarks/molecules/glycine.xyz \
    --basis def2-svp --xc b3lyp          # the closed-shell rows, to compare

# no GPU: does every reference to a patched function see the patch
python tests/test_patch_aliases.py

# correctness, both implementations in one process
python tests/test_uxc.py        # the unrestricted XC build
python tests/test_uxcgrad.py    # its gradient term, and the whole gradient
python tests/test_uejk.py       # the two-electron per-atom derivative at
                                # n_dm = 1 and n_dm = 2

# where the time goes, and what the two-electron derivative is worth
python benchmarks/scf_anatomy.py --xyz benchmarks/molecules/tmc_cu_macrocycle.xyz \
    --charge 2 --spin 3 --basis def2-svp --xc b3lyp [--patch all]
python benchmarks/ejk_ndm.py --xyz benchmarks/molecules/tmc_cucl2.xyz \
    --spin 1 --basis def2-svp --no-grad
```

To pick different cases out of BOS-TMC-28242 -- the split used here is
`by_ccd/test.aselmdb`, and every row carries `charge`, `spin` (the
multiplicity) and the CSD refcode in `row.data`:

```python
from ase.db import connect
db = connect('.../BOS-TMC-28242/splits/by_ccd/test.aselmdb')
for row in db.select():
    d = dict(row.data)
    if int(d['spin']) > 1 and row.natoms <= 30:
        print(row.natoms, row.formula, d['charge'], d['spin'], d['name'])
```

The six here were chosen for spread rather than for being representative:
3d and 5d metals, multiplicities 2 to 6, charges -1 to +2, an ECP, and sizes
from 3 to 113 atoms so that every ratio can be read at two sizes.  Nothing in
the modules depends on the size of the molecule, but that is a claim to check
rather than assume -- it is why `tmc_cu_macrocycle` (60 atoms) and
`tmc_co_sextet` (113) are here alongside `tmc_cucl2` (3).
