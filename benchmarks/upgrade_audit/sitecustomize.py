# Measurement shim for the upgrade audit (docs/UPGRADE_1.9_PLAN.md 1.3),
# the subprocess-reaching twin of shim_run.py: put this directory first on
# PYTHONPATH and every Python process -- including the subprocesses that
# tests/test_ccsd.py, test_qmmm.py, test_scfscreen.py ... spawn -- gets an
# electrolites.patch() that patches module by module and reports a module it
# cannot patch as `SHIM-SKIP <module>: <reason>` instead of aborting.
# Shadows NERSC's own sitecustomize (pymon) for those processes; harmless for
# a test run.  Not for production use.
import sys
try:
    import electrolites
    _orig = electrolites.patch

    def patch(*names, strict=True):
        names = [n for g in names for n in ((g,) if isinstance(g, str) else g)]
        for n in names:
            try:
                _orig([n], strict=True)
            except Exception as e:                      # noqa: BLE001
                msg = str(e).splitlines()[0][:300] if str(e) else ''
                print(f'SHIM-SKIP {n}: {type(e).__name__}: {msg}',
                      file=sys.stderr, flush=True)
        return electrolites.patched()
    electrolites.patch = patch
except Exception as e:                                  # noqa: BLE001
    print(f'SITESHIM failed: {e}', file=sys.stderr)
