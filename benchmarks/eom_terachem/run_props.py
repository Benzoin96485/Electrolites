"""EOM-EE-CCSD (singlet, 5 roots) + Λ + left vectors + oscillator strengths.
python run_props.py Ir/ACRHCO02_comp_0"""
import sys
import time
import json
import os

sys.argv = sys.argv[:2] + ['5']
src = open('run_eom.py').read().split(
    "t0 = time.perf_counter()\nfrom electrolites import eom_gpu")[0]
exec(src)

import cupy as cp
import numpy as np
from electrolites import eom_gpu, ccsd_lambda_gpu


def timed(key, fn, *a, **k):
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    out = fn(*a, **k)
    cp.cuda.Device().synchronize()
    t[key] = time.perf_counter() - t0
    return out


eom = eom_gpu.EOMEESinglet(mycc)
eom.conv_tol = 1e-7
e, rv = timed('eom_right', eom.kernel, nroots=nroots)
timed('lambda', ccsd_lambda_gpu.kernel, mycc, tol=1e-6)
convl, el, lv = timed('eom_left', eom_gpu._left_kernel, eom, nroots, guess=rv)
order = [int(np.argmin(abs(el - x))) for x in e]
el = el[order]
lv = [lv[i] for i in order]
p = timed('props', eom_gpu.transition_properties, mycc, e, rv, lv)
t['total'] = time.perf_counter() - T0
HARTREE2EV = 27.211386245988
out = dict(case=case, nao=mol.nao, natm=nat, frozen=nfrz,
           nocc_act=int(mycc.nocc), nvir=int(mycc.nmo - mycc.nocc),
           e_scf=float(mf.e_tot), e_corr=float(mycc.e_corr),
           ee_au=list(map(float, e)), ee_ev=list(map(float, np.asarray(e) * HARTREE2EV)),
           e_left_au=list(map(float, el)),
           f=p['f'].tolist(), mu_0k=p['mu_0k'].tolist(), mu_k0=p['mu_k0'].tolist(),
           r0=p['r0'].tolist(), right_conv=list(map(bool, eom.converged)),
           left_conv=list(map(bool, np.asarray(convl)[order])),
           times=t, gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
print('RESULT', json.dumps(out), flush=True)
os.makedirs('results_props', exist_ok=True)
with open(os.path.join('results_props', case.replace('/', '__') + '.json'), 'w') as fh:
    json.dump(out, fh, indent=1)
