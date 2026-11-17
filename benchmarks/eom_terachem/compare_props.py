"""Electrolites EOM-EE-CCSD energies + oscillator strengths (results_props/) vs
TeraChem RR-EOM-CCSD (last 'Transition Properties' table)."""
import glob
import json
import os
import re

import numpy as np

ROOT = ('/global/homes/w/weiliang/m4787_cfs/data/EOM-RR-CCSD/'
        'xyz_15-30atoms_spin1_opt')


def parse_tc(case):
    s = open(os.path.join(ROOT, case, 'EOM-CCSD.out')).read()
    blk = s.split('RR-EOM-CCSD Transition Properties')[-1]
    rows = re.findall(r'^\s+0 ->\s+(\d+)\s+([\d.]+)\s+([\d.eE+-]+)\s*$', blk, re.M)
    ev = np.array([float(r[1]) for r in rows])
    f = np.array([float(r[2]) for r in rows])
    m = re.search(r'Total processing time:\s+([\d.]+)', s)
    return ev, f, float(m.group(1))


allf = []
print(f"{'case':22s} {'nao':>4s}  root  {'E ours':>7s} {'E TC':>7s}  {'f ours':>8s} {'f TC':>8s}")
for fn in sorted(glob.glob('results_props/*.json')):
    r = json.load(open(fn))
    ev_tc, f_tc, t_tc = parse_tc(r['case'])
    ev = np.array(r['ee_ev'])
    f = np.array(r['f'])
    for k in range(len(ev)):
        print(f"{r['case']:22s} {r['nao']:4d}  {k + 1:4d}  {ev[k]:7.4f} {ev_tc[k]:7.4f}  "
              f"{f[k]:8.5f} {f_tc[k]:8.5f}")
        allf.append((f[k], f_tc[k]))
    t = r['times']
    print(f"   times: SCF {t['scf']:.0f} CCSD {t['ccsd']:.0f} EOM-R {t['eom_right']:.0f} "
          f"Λ {t['lambda']:.0f} EOM-L {t['eom_left']:.0f} props {t['props']:.0f} "
          f"total {t['total']:.0f} s  |  TC total {t_tc:.0f} s  ({t_tc / t['total']:.1f}x)"
          f"   max|eL-eR| {max(abs(a - b) for a, b in zip(r['e_left_au'], r['ee_au'])):.1e} Eh")
if allf:
    a = np.array(allf)
    big = a[:, 1] > 0.01
    print(f"\n{len(a)} roots: |f - f_TC| mean {abs(a[:, 0] - a[:, 1]).mean():.5f}, "
          f"max {abs(a[:, 0] - a[:, 1]).max():.5f}; for f_TC > 0.01 ({big.sum()} roots) "
          f"median rel. diff {np.median(abs(a[big, 0] / a[big, 1] - 1)) * 100:.1f} %")
