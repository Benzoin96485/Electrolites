"""The interface for added features (`electrolites/features.py`).

A throw-away feature is registered and enabled to check every path a real
one (stability, CCSD(T), CASSCF, ...) will take:

* the "absent upstream" check passes on a missing or ``NotImplemented`` slot
  and refuses a filled one, in ``check_absent`` and in ``enable(strict=True)``;
* ``attach`` binds a method and refuses to shadow an existing one;
* ``register_gpu_module`` makes PySCF's ``to_gpu()`` find the GPU class;
* ``cpu_fallback`` warns and returns the PySCF object.

    python tests/test_features.py
"""
import os
import sys
import tempfile
import types
import warnings

import electrolites
from electrolites import features

from pyscf import gto, lib                                   # noqa: E402
from gpu4pyscf import scf                                    # noqa: E402

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


# a CPU class in a fake PySCF module, so that `to_gpu` has something to map
cpu_mod = types.ModuleType('pyscf.elselftest')


class Thing(lib.StreamObject):
    to_gpu = lib.to_gpu

    def __init__(self, x=1):
        self.x = x


Thing.__module__ = 'pyscf.elselftest'
cpu_mod.Thing = Thing
sys.modules['pyscf.elselftest'] = cpu_mod

# the feature's own module: what `enable` imports.  Written to a temporary
# directory on the package path, so nothing runs until `enable` imports it
feat_src = '''
from gpu4pyscf.scf import hf
from electrolites.features import attach, register_gpu_module
import types

def selftest(self):
    return "gpu-feature"

attach(hf.RHF, "electrolites_selftest", selftest)

gpu = types.ModuleType("gpu4pyscf.elselftest")
class Thing:
    @classmethod
    def from_cpu(cls, obj):
        out = cls(); out.x = obj.x; return out
Thing.__module__ = "gpu4pyscf.elselftest"
gpu.Thing = Thing
register_gpu_module("gpu4pyscf.elselftest", gpu)
'''
tmp = tempfile.mkdtemp(prefix='elfeat_')
with open(os.path.join(tmp, '_selftest_feature.py'), 'w') as f:
    f.write(feat_src)
electrolites.__path__.append(tmp)
features.register('selftest', '_selftest_feature',
                  absent=[('gpu4pyscf.scf.hf', 'RHF.electrolites_selftest'),
                          ('gpu4pyscf.scf.hf', 'RHF.stability')],
                  gpu_modules=['gpu4pyscf.elselftest'])
features.register('clash', '_selftest_feature',
                  absent=[('gpu4pyscf.scf.hf', 'RHF.Gradients')])

res = features.check_absent(['selftest', 'clash'])
report('missing + NotImplemented slots count as absent',
       res['selftest'] is None, str(res['selftest']))
report('a filled upstream slot is reported', res['clash'] is not None)
try:
    electrolites.enable('clash')
    report('enable(strict) refuses a filled slot', False, 'did not raise')
except RuntimeError:
    report('enable(strict) refuses a filled slot', True)

electrolites.enable('selftest')
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf = scf.RHF(mol)
report('enable() attaches the method',
       mf.electrolites_selftest() == 'gpu-feature')
report('enabled() lists it', 'selftest' in electrolites.enabled())
g = Thing(x=7).to_gpu()
report('to_gpu() finds the registered GPU module',
       type(g).__module__ == 'gpu4pyscf.elselftest' and g.x == 7)
res = features.check_absent(['selftest'])
report('our own registered module is not a clash', res['selftest'] is None,
       str(res['selftest']))
try:
    features.attach(scf.hf.RHF, 'Gradients', lambda self: None)
    report('attach() refuses to shadow', False, 'did not raise')
except RuntimeError:
    report('attach() refuses to shadow', True)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    cpu = features.cpu_fallback(mf.run(), 'selftest', 'no GPU path')
report('cpu_fallback warns and returns PySCF',
       type(cpu).__module__.startswith('pyscf.')
       and any('cpu_fallback' in str(x.message) for x in w))

print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
