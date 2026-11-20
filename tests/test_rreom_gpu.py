"""Rank-reduced CCSD / Λ / EOM-EE-CCSD (``rreom_gpu``):

* with U = identity (no compression) RR-CCSD, RR-Λ and the compressed
  right/left Davidson reproduce canonical CCSD, Λ, EOM-EE and oscillator
  strengths;
* with a truncated projector: the RR-CCSD projected residual vanishes,
  left and right RR-EOM roots agree and equal the eigenvalues of the
  explicitly built compressed matrix;
* the projectors reproduce their generators (MP2 doubles; the CC2-like
  excited-state doubles) built densely from the DF integrals.

    python tests/test_rreom_gpu.py
"""
import sys

import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf import scf
from gpu4pyscf.cc import ccsd_incore

import electrolites
from electrolites import eom_gpu as G, rreom_gpu as RR

electrolites.patch_all()
fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:58s} {detail}', flush=True)


mol = gto.M(atom='O 0 0 0.11; H 0 0.78 -0.45; H 0.05 -0.74 -0.5',
            basis='6-31g', verbose=0)
mf = scf.RHF(mol).run(conv_tol=1e-12)
mycc = ccsd_incore.CCSD(mf, frozen=1)
mycc.conv_tol = 1e-12
mycc.conv_tol_normt = 1e-9
mycc.kernel()
e_ref = mycc.e_corr
t1_ref, t2_ref = cp.asarray(mycc.t1), cp.asarray(mycc.t2)
nroots = 3
eom, props = G.eomee_singlet_properties(mycc, nroots, conv_tol=1e-10)
l1_ref, l2_ref = cp.asarray(mycc.l1), cp.asarray(mycc.l2)
no, nv = t1_ref.shape
ov = no * nv
F = RR.Factors(mycc, int_thresh=1e-14)

# ------------------------------------------------ U = identity: canonical
sp = RR.Space(cp.eye(ov), F.dvo, no, nv)
mycc.e_hf = mycc.get_e_hf()
conv, ecc, t1, T = RR.rr_ccsd(mycc, sp, tol=1e-12, tolnormt=1e-10)
report('full rank: RR-CCSD energy == CCSD', abs(ecc - e_ref) < 1e-10,
       f'{ecc - e_ref:.1e}')
report('full rank: RR-CCSD t2 == CCSD t2',
       float(abs(cp.asarray(mycc.t2) - t2_ref).max()) < 1e-8)
RR.rr_lambda(mycc, sp, tol=1e-10)
report('full rank: RR-lambda == lambda',
       float(abs(mycc.l2 - l2_ref).max()) < 1e-7 and
       float(abs(mycc.l1 - l1_ref).max()) < 1e-7,
       f'{float(abs(mycc.l2 - l2_ref).max()):.1e}')
ops = RR.Ops(mycc)
x0 = cp.zeros((nroots + 2, sp.dim))
idx = cp.argsort(ops.diag(sp))[:nroots + 2]
x0[cp.arange(nroots + 2), idx] = 1
conv, w, X = RR.davidson(lambda V: ops.right(sp, V), x0, ops.diag(sp),
                         nroots, tol_e=1e-11, tol_r=1e-8, max_space=40)
report('full rank: right roots == EOM-EE', abs(w - props['e']).max() < 1e-8,
       f'{abs(w - props["e"]).max():.1e}')
convl, wl, XL = RR.davidson(lambda V: ops.left(sp, V), X, ops.diag(sp),
                            nroots, tol_e=1e-11, tol_r=1e-8, max_space=40)
report('full rank: left roots == right roots', abs(wl - w).max() < 1e-9,
       f'{abs(wl - w).max():.1e}')
r1, R = sp.unpack(X)
l1, Lc = sp.unpack(XL)
rv = [cp.asnumpy(G._pack_singlet(r1[k:k + 1],
                                 RR.expand(sp.U, R[k:k + 1], no, nv))[0])
      for k in range(nroots)]
lv = [cp.asnumpy(G._pack_singlet(
    l1[k:k + 1], RR.g_inv(RR.expand(sp.U, Lc[k:k + 1], no, nv)))[0])
    for k in range(nroots)]
p = G.transition_properties(mycc, w, rv, lv, mycc.l1, mycc.l2)
report('full rank: oscillator strengths == EOM-EE',
       abs(p['f'] - props['f']).max() < 1e-8,
       f'{abs(p["f"] - props["f"]).max():.1e}')

# ------------------------------------------------ truncated projector
U0, g0 = RR.ground_projector(F, thresh=1e-14, den_thresh=1e-14)
Bov = F.Bov
x = F.dvo
Gd = -(Bov @ Bov.T) / (x[:, None] + x[None, :])
rec = (U0 * g0) @ U0.T
report('ground projector reproduces the MP2 generator',
       float(abs(rec - Gd).max()) < 1e-8, f'{float(abs(rec - Gd).max()):.1e}')
r1k = cp.asarray(rv[0][:ov]).reshape(no, nv)
wk = float(w[0])
Us, gs = RR.state_projector(F, r1k, wk, thresh=1e-14, den_thresh=1e-14)
from electrolites.casscf_gpu import GPUDF                       # noqa: E402
from pyscf import df                                            # noqa: E402
mask = mycc.get_frozen_mask()
mo = cp.asarray(mycc.mo_coeff)[:, cp.asarray(mask)]
d = GPUDF(mol, df.make_auxbasis(mol, mp2fit=True))
Bvv = d.lmo(mo[:, no:], mo[:, no:])
Boo = d.lmo(mo[:, :no], mo[:, :no])
B = Bov.reshape(no, nv, -1)
Rt = (cp.einsum('ib,baP->iaP', r1k, Bvv) - cp.einsum('ja,ijP->iaP', r1k, Boo)
      ).reshape(ov, -1)
num = Bov @ Rt.T
num = num + num.T
Gs = -num / (x[:, None] + x[None, :] - wk)
rec = (Us * gs) @ Us.T
report('state projector reproduces the CC2-like generator',
       float(abs(rec - Gs).max()) < 1e-8, f'{float(abs(rec - Gs).max()):.1e}')

U, g = RR.ground_projector(F, thresh=1e-3, den_thresh=1e-2)
sp = RR.Space(U, F.dvo, no, nv)
conv, ecc, t1, T = RR.rr_ccsd(mycc, sp, tol=1e-12, tolnormt=1e-10)
from electrolites import fastccsd as FC                         # noqa: E402
E = FC._DevERIs(FC._make_eris(mycc), no)
t1n, t2n = FC.update_amps(mycc, cp.asarray(mycc.t1), cp.asarray(mycc.t2), E)
eia = E.mo_energy[:no, None] - E.mo_energy[None, no:]
om = (t2n - mycc.t2) * (eia[:, None, :, None] + eia[None, :, None, :])
pr = float(abs(RR.project(sp.U, om[None])).max())
full = float(abs(om).max())
report(f'rank {sp.N}/{ov}: projected RR-CCSD residual vanishes', pr < 1e-8,
       f'projected {pr:.1e}, full {full:.1e}, dE {ecc - e_ref:.2e}')
RR.rr_lambda(mycc, sp, tol=1e-10)
ops = RR.Ops(mycc)
Us, _ = RR.state_projector(F, r1k, wk, thresh=1e-3)
sp = RR.Space(Us, F.dvo, no, nv)
M = ops.right(sp, cp.eye(sp.dim))      # rows: sigma of unit vectors
Mm = cp.asnumpy(M).T
ev = np.sort(np.linalg.eigvals(Mm).real)
x0 = cp.zeros((nroots + 2, sp.dim))
idx = cp.argsort(ops.diag(sp))[:nroots + 2]
x0[cp.arange(nroots + 2), idx] = 1
conv, w, X = RR.davidson(lambda V: ops.right(sp, V), x0, ops.diag(sp),
                         nroots, tol_e=1e-11, tol_r=1e-8, max_space=40)
report(f'rank {sp.N}: Davidson roots == dense compressed matrix',
       abs(w - ev[:nroots]).max() < 1e-8, f'{abs(w - ev[:nroots]).max():.1e}')
convl, wl, XL = RR.davidson(lambda V: ops.left(sp, V), X, ops.diag(sp),
                            nroots, tol_e=1e-11, tol_r=1e-8, max_space=40)
report(f'rank {sp.N}: left roots == right roots', abs(wl - w).max() < 1e-8,
       f'{abs(wl - w).max():.1e}')
# the left operator is the adjoint of the right one in the pairing
# <L|R> = 2 l1.r1 + Frobenius(Lc, Rc): W^-1 M^T W in packed coordinates
ML = cp.asnumpy(ops.left(sp, cp.eye(sp.dim))).T
tri = cp.asnumpy(sp.tri[0] == sp.tri[1])
W = np.concatenate([np.full(sp.ov, 2.), np.where(tri, 1., 2.)])
adj = Mm.T * W[None, :] / W[:, None]
report(f'rank {sp.N}: left operator == adjoint of right',
       abs(ML - adj).max() < 1e-9 * abs(Mm).max(),
       f'{abs(ML - adj).max():.1e}')

# ------------------------------------------------ factorised DF ladder
Bvv = RR.df_vv(F)
vvvv = cp.einsum('aeA,bfA->aebf', Bvv, Bvv)
lad = RR.DFLadder(Bvv, sp, cp.asarray(mycc.t1))
t1c = cp.asarray(mycc.t1)
rng = np.random.default_rng(1)
Rr = cp.asarray(rng.standard_normal((2, sp.N, sp.N)))
Rr = Rr + Rr.transpose(0, 2, 1)
r1r = cp.asarray(rng.standard_normal((2, no, nv)))
rt = cp.einsum('nia,jb->nijab', r1r, t1c)
tau2 = RR.expand(sp.U, Rr, no, nv) + rt + rt.transpose(0, 2, 1, 4, 3)
ref = RR.project(sp.U, cp.einsum('aebf,nijef->nijab', vvvv, tau2))
got = lad.project(Rr, r1r)
report('DF ladder: factorised projection == direct',
       float(abs(got - ref).max()) < 1e-10 * float(abs(ref).max()),
       f'{float(abs(got - ref).max()):.1e}')
L2r = RR.expand(sp.U, Rr, no, nv)
full = cp.einsum('aebf,nijef->nijab', vvvv, L2r)
ref1 = 2 * cp.einsum('nijab,jb->nia', full, t1c) - \
    cp.einsum('njiab,jb->nia', full, t1c)
got1 = lad.l1_part(L2r, t1c)
report('DF ladder: singles contraction == direct',
       float(abs(got1 - ref1).max()) < 1e-10 * float(abs(ref1).max()),
       f'{float(abs(got1 - ref1).max()):.1e}')
opsd = RR.Ops(mycc, Bvv=Bvv)
V = cp.asarray(rng.standard_normal((2, sp.dim)))
a, b = opsd.right(sp, V), ops.right(sp, V)
report('right sigma with the DF ladder ~ AO ladder (DF error)',
       float(abs(a - b).max()) < 3e-3 * float(abs(b).max()),
       f'{float(abs(a - b).max() / abs(b).max()):.1e} rel')
a, b = opsd.left(sp, V), ops.left(sp, V)
report('left sigma with the DF ladder ~ AO ladder (DF error)',
       float(abs(a - b).max()) < 3e-3 * float(abs(b).max()),
       f'{float(abs(a - b).max() / abs(b).max()):.1e} rel')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
