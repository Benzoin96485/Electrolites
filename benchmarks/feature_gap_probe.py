"""What happens today when a PySCF feature is asked of a GPU4PySCF object?

`docs/FEATURE_TARGETS_PYSCF_2.13.md` §3 is the table this script prints.  Each
row is one PySCF 2.13 feature tried on a GPU4PySCF 1.8.1 mean field (or the
GPU object handed to the PySCF driver), on molecules small enough for a login
node: water, H2O+ (doublet), N2, and a two-atom H2 cell.  A row says

* ``ok``     -- it ran; the class/module of what came back is printed, so a
               result that silently came from the CPU is visible;
* ``raises`` -- the exception and the innermost PySCF/GPU4PySCF frame;
* ``WRONG``  -- it ran and returned a number that disagrees with an
               independent reference (finite differences) by more than the
               stated bound -- the rows that matter most.

Nothing here is timed.  ``--patch all`` applies Electrolites first.  Since
Phase 0 that changes six rows: ``guards`` turns the two ``WRONG`` rows (S3's
ROHF Hessian, V2's QM/MM Hessian, and V2's control with it) and the ROHF TDA
into explicit ``NotImplementedError`` raises and sends the ROHF gradient to PySCF on the CPU
(an ``ok`` that returns NumPy), and ``fixes`` makes the vSAP guess run.  No
other row changes: none of these features exists yet.

    python benchmarks/feature_gap_probe.py [--patch all] [--only S3,V2]
"""
import argparse
import os
import sys
import tempfile
import traceback
import warnings

import numpy as np

warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--patch', default='')
ap.add_argument('--only', default='',
                help='comma separated row-id prefixes to run (default: all)')
args = ap.parse_args()

if args.patch:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _common import apply_patch
    apply_patch(args.patch)

import cupy
from pyscf import gto, lib
from pyscf import scf as cscf, dft as cdft
from gpu4pyscf import scf, dft

TMP = tempfile.mkdtemp(prefix='fgp_')
ROWS = []


def _where(tb):
    """Innermost frame inside pyscf / gpu4pyscf / electrolites."""
    keep = None
    for fr in traceback.extract_tb(tb):
        for pkg in ('gpu4pyscf', 'pyscf', 'electrolites'):
            tag = os.sep + pkg + os.sep
            if tag in fr.filename:
                rel = fr.filename.split(tag, 1)[1]
                prefix = {'gpu4pyscf': 'G:', 'pyscf': 'P:',
                          'electrolites': 'electrolites/'}[pkg]
                keep = f'{prefix}{rel}:{fr.lineno}'
                break
    return keep or '-'


def row(rid, label):
    def deco(fn):
        if args.only and not any(rid.startswith(p)
                                 for p in args.only.split(',')):
            return fn
        try:
            out = fn()
            if isinstance(out, tuple) and out and out[0] == 'WRONG':
                ROWS.append((rid, label, 'WRONG', out[1], '-'))
            else:
                what = out if isinstance(out, str) else (
                    type(out).__module__ + '.' + type(out).__name__)
                ROWS.append((rid, label, 'ok', what, '-'))
        except Exception as e:
            msg = str(e).splitlines()[0][:90] if str(e) else ''
            ROWS.append((rid, label, 'raises',
                         f'{type(e).__name__}: {msg}', _where(e.__traceback__)))
        return fn
    return deco


water = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467',
              basis='6-31g', verbose=0)
cation = gto.M(atom='O 0 0 0.1; H 0 0.76 -0.45; H 0 -0.76 -0.45',
               basis='sto-3g', charge=1, spin=1, verbose=0)
n2 = gto.M(atom='N 0 0 0; N 0 0 1.098', basis='6-31g', verbose=0)

rhf = scf.RHF(water).run(conv_tol=1e-11)
pbe = dft.RKS(water, xc='pbe').run(conv_tol=1e-11)
nocc = water.nelectron // 2

# ---------------------------------------------------------------- S: SCF/DFT


@row('S1', 'gpu RHF.COSX()')
def _():
    return rhf.COSX()


@row('S1', 'pyscf.sgx.sgx_fit(gpu RHF).kernel()')
def _():
    from pyscf import sgx
    return sgx.sgx_fit(scf.RHF(water)).kernel()


@row('S2', 'gpu RHF.stability()')
def _():
    return rhf.stability()


@row('S2', 'pyscf rhf_stability(gpu RHF)')
def _():
    from pyscf.scf import stability
    return stability.rhf_stability(rhf)


@row('S3', 'gpu ROHF.Gradients().kernel()')
def _():
    return scf.ROHF(cation).run().Gradients().kernel()


def _hess_vs_fd(mol, gpu_cls, cpu_cls):
    """max |H[O, :, z, :] - d(gradient)/dz_O| for one mean-field class."""
    h = gpu_cls(mol).run(conv_tol=1e-12).Hessian().kernel()
    step = 1e-4
    c0 = mol.atom_coords()
    gs = []
    for s in (1, -1):
        c = c0.copy()
        c[0, 2] += s * step
        m = mol.set_geom_(c, unit='Bohr', inplace=False)
        gs.append(cpu_cls(m).run(conv_tol=1e-12).Gradients().kernel())
    fd = (gs[0] - gs[1]) / (2 * step)          # d g[j,k] / d z_O
    return abs(h[0, :, 2, :] - fd).max()


@row('S3', 'control: gpu UHF.Hessian() vs FD, same doublet')
def _():
    return f'agrees, {_hess_vs_fd(cation, scf.UHF, cscf.UHF):.1e}'


@row('S3', 'gpu ROHF.Hessian() vs FD of CPU ROHF gradient')
def _():
    err = _hess_vs_fd(cation, scf.ROHF, cscf.ROHF)
    if err > 1e-4:
        return ('WRONG', f'max |H - FD| = {err:.3g} Eh/bohr^2 (no error raised)')
    return f'agrees, {err:.1e}'


@row('S3', 'gpu ROHF.TDA().kernel()')
def _():
    return scf.ROHF(cation).run().TDA().kernel()


@row('S4', 'cpu DHF(mol).to_gpu()')
def _():
    return cscf.DHF(water).to_gpu()


@row('S5', 'gpu RKS.sfx2c1e().Gradients()')
def _():
    return dft.RKS(water, xc='pbe').sfx2c1e().run().Gradients().kernel()


@row('S6', 'gpu GHF.TDA()')
def _():
    return scf.GHF(cation).run().TDA()


@row('S7', 'lo.Boys(mol, cupy mo_occ).kernel()')
def _():
    from pyscf import lo
    return lo.Boys(water, rhf.mo_coeff[:, :nocc]).kernel()


@row('S7', 'lo.ibo.ibo(mol, cupy mo_occ)')
def _():
    from pyscf import lo
    return lo.ibo.ibo(water, rhf.mo_coeff[:, :nocc])


@row('S8', "gpu RKS init_guess='sap'")
def _():
    return dft.RKS(water, xc='pbe').set(init_guess='sap').run().e_tot


@row('S8', "gpu RKS init_guess='vsap'")
def _():
    return dft.RKS(water, xc='pbe').set(init_guess='vsap').run().e_tot


@row('S9', 'tools.cubegen.density(mol, cupy dm)')
def _():
    from pyscf.tools import cubegen
    return cubegen.density(water, os.path.join(TMP, 'd.cube'),
                           rhf.make_rdm1(), nx=8, ny=8, nz=8)

# ------------------------------------------------------- M: multireference


@row('M1', 'mol.to_gpu().CASSCF')
def _():
    return n2.to_gpu().CASSCF


@row('M1', 'pyscf.mcscf.CASSCF(gpu RHF, 6, 6).run()')
def _():
    from pyscf import mcscf
    return mcscf.CASSCF(scf.RHF(n2).run(), 6, 6).run()


@row('M2', 'CASCI(6,6) [400 dets] with the GPU FCI solver')
def _():
    from pyscf import mcscf
    from gpu4pyscf.fci import direct_spin1
    mc = mcscf.CASCI(cscf.RHF(n2).run(), 6, 6)
    mc.fcisolver = direct_spin1.FCI(n2)
    e = mc.kernel()[0]
    ref = mcscf.CASCI(cscf.RHF(n2).run(), 6, 6).kernel()[0]
    return f'dE = {abs(e - ref):.1e} (pspace path; GPU kernel not called)'


@row('M2', 'CASCI(8,8) [4900 dets] with the GPU FCI solver')
def _():
    from pyscf import mcscf
    from gpu4pyscf.fci import direct_spin1
    mc = mcscf.CASCI(cscf.RHF(n2).run(), 8, 8)
    mc.fcisolver = direct_spin1.FCI(n2)
    return mc.kernel()[0]


@row('M2', 'CASSCF(6,6) with the GPU FCI solver')
def _():
    from pyscf import mcscf
    from gpu4pyscf.fci import direct_spin1
    mc = mcscf.CASSCF(cscf.RHF(n2).run(), 6, 6)
    mc.fcisolver = direct_spin1.FCI(n2)
    return mc.kernel()[0]

# ----------------------------------------- P: single-reference correlated


@row('P1', 'gpu4pyscf CCSD(gpu RHF).run().ccsd_t()')
def _():
    from gpu4pyscf.cc import ccsd_incore
    return ccsd_incore.CCSD(rhf).run().ccsd_t()


@row('P1', 'pyscf.cc.CCSD(gpu RHF).run()')
def _():
    from pyscf import cc
    return cc.CCSD(rhf).run()


@row('P2', 'gpu4pyscf MP2(gpu RHF).run().Gradients()')
def _():
    from gpu4pyscf import mp
    return mp.MP2(rhf).run().Gradients()


@row('P2', 'gpu4pyscf MP2(gpu UHF)')
def _():
    from gpu4pyscf import mp
    return mp.MP2(scf.UHF(cation).run())


@row('P3', "dft.RKS(mol, xc='B2PLYP') (CPU or GPU)")
def _():
    return cdft.RKS(water, xc='B2PLYP').run().e_tot


@row('P4', 'pyscf.gw.GW(gpu PBE)')
def _():
    from pyscf import gw
    return gw.GW(pbe).kernel()


@row('P4', 'pyscf.gw.rpa.RPA(gpu PBE)')
def _():
    from pyscf.gw import rpa
    return rpa.RPA(pbe).kernel()


@row('P5', 'pyscf.adc.ADC(gpu RHF)')
def _():
    from pyscf import adc
    return adc.ADC(rhf).kernel()

# --------------------------------------------- O, V: properties, solvent


@row('O1', 'gpu UKS NMR shielding')
def _():
    from gpu4pyscf.properties import shielding
    uks = dft.UKS(cation, xc='b3lyp').run()
    return shielding.eval_shielding(uks)


@row('O2', 'gpu polarizability: frequency-dependent entry point')
def _():
    from gpu4pyscf.properties import polarizability
    return polarizability.polarizability_with_freq


@row('V1', 'gpu RKS.ddCOSMO()')
def _():
    return dft.RKS(water, xc='pbe').ddCOSMO()


def _qmmm_hess_vs_fd(scale):
    """max |H[O, :, x, :] - d(QM/MM gradient)/dx_O| with charges * scale."""
    from gpu4pyscf.qmmm import itrf
    mmc = [[3., 0.5, 0.2], [0.3, 3., -0.4]]
    mmq = [0.8 * scale, -0.8 * scale]
    c0 = water.atom_coords()

    def build(c):
        m = water.set_geom_(c, unit='Bohr', inplace=False)
        mf = itrf.add_mm_charges(dft.RKS(m, xc='pbe'), mmc, mmq, unit='Bohr')
        return mf.run(conv_tol=1e-12)
    h = build(c0).Hessian().kernel()
    step = 1e-4
    gs = []
    for s in (1, -1):
        c = c0.copy()
        c[0, 0] += s * step
        gs.append(build(c).Gradients().kernel())
    fd = (gs[0] - gs[1]) / (2 * step)
    return abs(h[0, :, 0, :] - fd).max()


@row('V2', 'control: same, charges scaled by 1e-3')
def _():
    return f'agrees, {_qmmm_hess_vs_fd(1e-3):.1e}'


@row('V2', 'gpu QM/MM Hessian vs FD of QM/MM gradient (+-0.8 e at 3 bohr)')
def _():
    err = _qmmm_hess_vs_fd(1.)
    if err > 1e-3:
        return ('WRONG', f'max |H - FD| = {err:.3g} Eh/bohr^2 (no error raised)')
    return f'agrees, {err:.1e}'

# ------------------------------------------------------------- X: periodic


def _cell():
    from pyscf.pbc import gto as pgto
    return pgto.M(a=np.eye(3) * 4.0, atom='H 0 0 0; H 0 0 0.74',
                  basis='sto-3g', pseudo=None, verbose=0)


@row('X1', 'gpu PBC RKS (Gamma).TDA()')
def _():
    from gpu4pyscf.pbc import dft as pdft
    return pdft.RKS(_cell(), xc='pbe').run().TDA()


@row('X1', 'pyscf.pbc.tdscf.TDA(gpu PBC RKS)')
def _():
    from pyscf.pbc import tdscf
    from gpu4pyscf.pbc import dft as pdft
    return tdscf.TDA(pdft.RKS(_cell(), xc='pbe').run()).kernel()


@row('X2', 'pyscf.pbc.mp.KMP2(gpu KRHF)')
def _():
    from pyscf.pbc import mp as pmp
    from gpu4pyscf.pbc import scf as pscf
    cell = _cell()
    kmf = pscf.KRHF(cell, cell.make_kpts([2, 1, 1])).density_fit().run()
    return pmp.KMP2(kmf).kernel()


@row('X3', 'gpu PBC RKS (Gamma) with VV10')
def _():
    from gpu4pyscf.pbc import dft as pdft
    return pdft.RKS(_cell(), xc='wb97m-v').run().e_tot


# ----------------------------------------------------------------- report
w = max(len(r[1]) for r in ROWS)
print(f'# PySCF {__import__("pyscf").__version__}, GPU4PySCF '
      f'{__import__("gpu4pyscf").__version__}, '
      f'patch: {args.patch or "(none)"}')
for rid, label, kind, what, where in ROWS:
    print(f'{rid:3s} {label:{w}s}  {kind:6s}  {what}'
          + (f'   [{where}]' if where != '-' else ''))
