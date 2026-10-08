"""Write electrolites/kernels/cuda/myjkip1/ from GPU4PySCF 1.8.1's
lib/gvhf-rys/{rys_contract_jk_ip1.cu, unrolled_rys_jk_ip1.cu} (catalogue B15).

The kernels are copied unchanged.  The driver changes in two ways:

* ``RYS_build_jk_ip1`` calls ``cudaGetDeviceProperties(&prop, 0)`` -- a
  millisecond-scale host call, and on device 0 whatever the current device
  -- on every angular-momentum class launch, thousands per Hessian.  Here the
  multiprocessor count of the *current* device is queried once and cached.
* only ``RYS_build_jk_ip1`` is kept, renamed ``MYJKIP1_build_jk_ip1``; the
  three per-atom energy entry points (and with them the unrolled ``ejk_ip1``
  kernels they need) are dropped.

and every ``rys_vjk_ip1`` kernel (unrolled and general) takes **several
densities for K** (catalogue B15 (b)): with ``n_dm > 1``, J is built from
density 0 and K from each of densities ``1..n_dm-1`` into
``vk[(ia*3+x)*(n_dm-1) + d]``, so an open-shell Hessian builds J of the total
density and both spins' K from one integral pass instead of two.  ``n_dm ==
1`` is GPU4PySCF's contract unchanged.

And the unrolled kernels get B3's **P1** (``codegen/gen_ejk_ip2.py:p1``, the
same anchors -- the ip1 and ip2 files come from one generator upstream): no
fp64 division in the primitive or root loops.  ``--no-p1`` copies them
verbatim (the control).

    python codegen/gen_jk_ip1.py /path/to/gpu4pyscf/lib/gvhf-rys [--no-p1]
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_ejk_ip2 import p1, check_no_fp64_div        # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith('--')]
NO_P1 = '--no-p1' in sys.argv
src_dir = args[0]
out_dir = args[1] if len(args) > 1 else 'electrolites/kernels/cuda/myjkip1'
os.makedirs(out_dir, exist_ok=True)

def multi_k(text, where):
    """Wrap each vjk kernel's ``if (do_k) {...}`` in a loop over the K
    densities (see the module docstring)."""
    out, pos, n = [], 0, 0
    key = 'if (do_k) {'
    anchor = 'double *vk_x = jk.vk +'
    while True:
        # the matrix kernels: the first `if (do_k)` after their vk set-up
        a = text.find(anchor, pos)
        if a < 0:
            break
        i = text.find(key, a)
        assert i > 0, where
        depth, j = 1, i + len(key)
        while depth:
            c = text[j]
            depth += (c == '{') - (c == '}')
            j += 1
        line0 = text.rfind('\n', 0, i) + 1
        ind = text[line0:i]
        loop = (f'for (int kdm_ = (jk.n_dm > 1); kdm_ < jk.n_dm; ++kdm_) {{\n'
                f'{ind}size_t nkdm_ = jk.n_dm - (jk.n_dm > 1);\n'
                f'{ind}size_t ldm_ = kdm_ - (jk.n_dm > 1);\n'
                f'{ind}double *dm = jk.dm + (size_t)kdm_ * nao * nao;\n')
        for x, c in (('x', 0), ('y', 1), ('z', 2)):
            loop += (f'{ind}double *vk_{x} = jk.vk + ((ia*3+{c})*nkdm_ + ldm_)'
                     f'*(size_t)nao*nao;\n')
        out.append(text[pos:i] + loop + ind + text[i:j] + '\n' + ind + '}')
        pos = j
        n += 1
    out.append(text[pos:])
    sys.stderr.write(f'  {where}: {n:3d} x K over n_dm densities\n')
    assert n, where
    return ''.join(out), n


s = open(os.path.join(src_dir, 'rys_contract_jk_ip1.cu')).read()
old = ('    cudaDeviceProp prop;\n'
       '    cudaGetDeviceProperties(&prop, 0);\n'
       '    int workers = prop.multiProcessorCount;\n')
assert s.count(old) == 4, s.count(old)
s = s.replace(old, '    int workers = myjkip1_sm_count();\n')
helper = '''
// electrolites (codegen/gen_jk_ip1.py): the multiprocessor count of the
// current device, queried once per device instead of once per launch
static int myjkip1_sm_count()
{
    static int cache[64] = {0};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) {
        int n = 0;
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n;
    }
    if (cache[dev] == 0)
        cudaDeviceGetAttribute(&cache[dev], cudaDevAttrMultiProcessorCount, dev);
    return cache[dev];
}
'''
inc = '#include "build_rys_gxyz.cuh"\n'
assert inc in s
s = s.replace(inc, inc + helper, 1)
# keep RYS_build_jk_ip1 only
i = s.index('int RYS_build_jk_ip1(')
j = s.index('\nint RYS_per_atom_jk_ip1(')
k = s.rindex('}')                       # the extern "C" block's brace
s = s[:j] + '\n' + s[k:]
s = s.replace('int RYS_build_jk_ip1(', 'int MYJKIP1_build_jk_ip1(')
s = s.replace('in RYS_build_jk_ip1,', 'in MYJKIP1_build_jk_ip1,')
s, _ = multi_k(s, 'rys_contract_jk_ip1.cu')
if not NO_P1:
    # the general kernels the driver falls back to (every class past the
    # unrolled set) get P1 too; build_rys_gxyz.cuh is then the P1 copy
    # under gvhf-rys181/ (fastejk2's), which expects iaij, iakl, inv_s2
    # two of the ejk kernels (unused here, kept compiled) recompute akl
    # between the bra lines; put it first so p1's bra anchor matches
    s = s.replace('double aij = ai + aj;\n'
                  '                double akl = ak + al;\n'
                  '                double aj_aij = aj / aij;',
                  'double aij = ai + aj;\n'
                  '                double aj_aij = aj / aij;')
    s, n = p1(s, 'rys_contract_jk_ip1.cu')
head = ('// GENERATED by codegen/gen_jk_ip1.py from GPU4PySCF 1.8.1\'s\n'
        '// lib/gvhf-rys/rys_contract_jk_ip1.cu -- do not edit.\n')
open(os.path.join(out_dir, 'rys_contract_jk_ip1.cu'), 'w').write(head + s)
u = open(os.path.join(src_dir, 'unrolled_rys_jk_ip1.cu')).read()
u, _ = multi_k(u, 'unrolled_rys_jk_ip1.cu')
if not NO_P1:
    u, n = p1(u, 'unrolled_rys_jk_ip1.cu')
    bad = check_no_fp64_div(u, 'unrolled_rys_jk_ip1.cu')
    assert not bad, bad[:5]
    u = ('// GENERATED by codegen/gen_jk_ip1.py (P1) from GPU4PySCF 1.8.1\'s\n'
         '// lib/gvhf-rys/unrolled_rys_jk_ip1.cu -- do not edit.\n' + u)
open(os.path.join(out_dir, 'unrolled_rys_jk_ip1.cu'), 'w').write(u)
# with --no-p1, GPU4PySCF's own build_rys_gxyz.cuh, since the copy under
# gvhf-rys181/ is fastejk2's P1 one (an #include of a same-directory file
# resolves here first)
hdr = os.path.join(out_dir, 'build_rys_gxyz.cuh')
if NO_P1:
    shutil.copy(os.path.join(src_dir, 'build_rys_gxyz.cuh'), hdr)
elif os.path.exists(hdr):
    os.remove(hdr)
# c_div_nf lives in GPU4PySCF's rys_jk_driver.cu, which nothing else of
# that file is wanted from
d = open(os.path.join(src_dir, 'rys_jk_driver.cu')).read()
i = d.index('__constant__ float c_div_nf[]')
j = d.index('};', i) + 2
open(os.path.join(out_dir, 'constants.cu'), 'w').write(
    '// GENERATED by codegen/gen_jk_ip1.py from GPU4PySCF 1.8.1\'s\n'
    '// lib/gvhf-rys/rys_jk_driver.cu -- do not edit.\n'
    '#include "vhf.cuh"\n\n' + d[i:j] + '\n')
print('wrote', out_dir)
