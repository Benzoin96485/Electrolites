"""Run one Electrolites test script with patch() made per-module tolerant.

Measurement shim only.  On GPU4PySCF 1.9.0 `patch_all()` refuses to patch
anything (fastpbcgrad lost its target) and a module that fails at import
(fastpbcdf's source substitution) aborts the rest.  Here every module is
patched on its own; a missing target or an import failure is reported on
stderr as `SHIM-SKIP <module>: <reason>` and the rest are still applied.
Usage: python shim_run.py tests/test_x.py [args...]
"""
import os, runpy, sys
import electrolites
_orig = electrolites.patch
def patch(*names, strict=True):
    names = [n for g in names for n in ((g,) if isinstance(g, str) else g)]
    for n in names:
        try:
            _orig([n], strict=True)
        except Exception as e:                      # noqa: BLE001
            msg = str(e).splitlines()[0][:300] if str(e) else ''
            print(f'SHIM-SKIP {n}: {type(e).__name__}: {msg}', file=sys.stderr, flush=True)
    return electrolites.patched()
electrolites.patch = patch
script = os.path.abspath(sys.argv[1])
sys.argv = [script] + sys.argv[2:]
sys.path[0] = os.path.dirname(script)
runpy.run_path(script, run_name='__main__')
