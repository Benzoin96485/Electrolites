"""Compare Electrolites EOM-EE-CCSD results (results_eom/*.json) with TeraChem RR-EOM-CCSD."""
import glob
import json
import os
import re

ROOT = ('/global/homes/w/weiliang/m4787_cfs/data/EOM-RR-CCSD/'
        'xyz_15-30atoms_spin1_opt')


def parse_tc(case):
    s = open(os.path.join(ROOT, case, 'EOM-CCSD.out')).read()
    blocks = re.findall(r'RR-EOM-CCSD  Energies <=+\n(.*?)\n  -{20,}\n\n', s, re.S)
    last = blocks[-1]
    ev = [float(m) for m in re.findall(
        r'^\s+\d+\s+-\d+\.\d+\s+\d\.\d+\s+(\d+\.\d+)\s*$', last, re.M)]
    g = lambda pat: float(re.search(pat, s).group(1))
    return dict(
        e_scf=g(r'Reference Energy \(recalculated\)\s+=\s+(-?\d+\.\d+)'),
        e_corr=g(r'Correlation Energy\s+=\s+(-?\d+\.\d+)'),
        ev=ev, t_chol=g(r'Cholesky Time:\s+([\d.]+)'),
        t_rrcc=g(r'RRCCSD Time:\s+([\d.]+)'),
        t_total=g(r'Total processing time:\s+([\d.]+)'),
        nchol=int(g(r'Number of Cholesky vectors:\s+(\d+)')),
        rank=int(g(r'Rank of doubles amplitudes:\s+(\d+)')))


rows = []
for f in sorted(glob.glob('results_eom/*.json')):
    r = json.load(open(f))
    tc = parse_tc(r['case'])
    ev = sorted(r['ee_ev'])
    d = [a - b for a, b in zip(ev, tc['ev'])]
    rows.append((r, tc, d))
    print(f"\n== {r['case']}  nao {r['nao']}  o_act {r['nocc_act']} v {r['nvir']}"
          f"  (TC: {tc['nchol']} Cholesky vecs, RR rank {tc['rank']})")
    print(f"  E_SCF   ours {r['e_scf']:.8f}  TC {tc['e_scf']:.8f}  "
          f"diff {1e3 * (r['e_scf'] - tc['e_scf']):+.4f} mEh")
    print(f"  E_corr  ours {r['e_corr']:.6f} (CCSD)  TC {tc['e_corr']:.6f} (RR-CCSD)"
          f"  diff {1e3 * (r['e_corr'] - tc['e_corr']):+.3f} mEh")
    print('  roots eV  ours  ' + ' '.join(f'{x:7.4f}' for x in ev))
    print('            TC    ' + ' '.join(f'{x:7.4f}' for x in tc['ev']))
    print('            diff  ' + ' '.join(f'{x:+7.4f}' for x in d))
    t = r['times']
    print(f"  time  ours: SCF {t['scf']:.0f}s CCSD {t['ccsd']:.0f}s EOM {t['eom']:.0f}s "
          f"total {t['total']:.0f}s [{r['gpu']}]   TC: Cholesky {tc['t_chol']:.0f}s "
          f"RR-CC(+Λ+EOM) {tc['t_rrcc']:.0f}s total {tc['t_total']:.0f}s [H200]  "
          f"speedup {tc['t_total'] / t['total']:.1f}x")

alld = [x for _, _, d in rows for x in d]
if alld:
    import numpy as np
    a = np.array(alld)
    print(f'\nALL {len(rows)} cases, {len(a)} roots: mean signed {a.mean():+.4f} eV, '
          f'MAE {abs(a).mean():.4f} eV, max |d| {abs(a).max():.4f} eV')
json.dump([dict(case=r['case'], nao=r['nao'], ours=r, tc=tc, diff_ev=d)
           for r, tc, d in rows], open('comparison.json', 'w'), indent=1)
