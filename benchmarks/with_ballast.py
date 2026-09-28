"""Run a benchmark script with part of the card taken first.

    python benchmarks/with_ballast.py 40 benchmarks/df_hess_terms_probe.py ...

allocates a 40 GB CuPy array (written once, so it is resident) and then runs
the script in this process with the rest of the arguments -- the memory of
a 40 GB A100 on an 80 GB one.  ``FASTDFHESSIP_MEM`` and its kind cap what a
module's sizing *reads*; a ballast is what every allocation of the run,
including GPU4PySCF's own, actually has left (`docs/README_fastdfh1c.md`
§3.1, `docs/README_fastdfhessip5.md` §4).
"""
import os
import runpy
import sys

import cupy as cp

gb = float(sys.argv[1])
_BALLAST = cp.ones(int(gb * 1e9) // 8) if gb > 0 else None
cp.cuda.Stream.null.synchronize()
print(f'# ballast {gb:g} GB, free {cp.cuda.runtime.memGetInfo()[0] / 1e9:.1f} GB',
      flush=True)
script = sys.argv[2]
sys.argv = sys.argv[2:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name='__main__')
