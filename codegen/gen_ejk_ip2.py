"""
Lift GPU4PySCF's second-derivative ERI kernels (catalogue B3) and take the
divisions out of them.

`hessian/rhf.py:_partial_ejk_ip2` drives two C entry points,
``RYS_per_atom_jk_ip2_type12`` and ``RYS_per_atom_jk_ip2_type3``.  Between them
they are 8.6 s of a 42-atom B3LYP/def2-SVP Hessian and 98.9 % of that stage
(``benchmarks/ejk_ip2_probe.py``).  Two things about them are worth changing,
and neither needs the integral arithmetic rewritten:

* **P1: no double-precision division in the primitive or root loops.**  Per
  primitive quartet the kernels evaluate ``aj/aij``, ``al/akl``,
  ``aij*akl/(aij+akl)`` and ``cicj*ckcl/(aij*akl*sqrt(aij+akl))``, and then
  ``rt/(aij+akl)``, ``.5/aij`` and ``.5/akl`` **once per Rys root** -- eighteen
  fp64 divisions and a ``sqrt`` per primitive quartet for a five-root class,
  each about twenty instructions on sm_80.  ``1/aij`` and ``1/akl`` are formed
  once where the exponents are read, one ``rsqrt(aij+akl)`` supplies both
  ``1/(aij+akl)`` and ``1/sqrt(aij+akl)``, and every division in the hot loops
  becomes a multiply: two divisions and one ``rsqrt`` per primitive quartet,
  none per root.
* **P3: a register cap -- off by default, because it is measured at 0.58x.**
  Neither ip2 file carries a ``__launch_bounds__`` or ``__maxnreg__`` anywhere,
  so every one of these kernels is compiled at the 255-register ceiling, which
  is one 256-thread block per SM.  ``--p3`` caps them at 128 registers and two
  blocks -- and that is a **1.75x regression**: these kernels want all 255
  registers, and the spills the cap forces cost more than the second block
  buys (docs/README_fastejk2.md 2b).  The catalogue's B3 entry singles P3 out
  as worth *more* here than on the exchange build; the flag exists so that can
  be reproduced rather than argued about.

  When the cap *is* asked for, it has to be expressed at the block size the
  driver actually launches, and finding that out was the round's one real
  surprise.
  ``__launch_bounds__(512, 1)`` is the *same* 128-register cap -- 65536/(512*1)
  and 65536/(256*2) are both 128 -- but with a 512-thread maximum, and it
  produces numbers **9x to 37x wrong** where ``(256, 2)`` agrees with upstream
  to 1e-13.  Measured with ``vhf.cuh``'s ``adjust_threads`` live *and* with it disabled --
  that is, with the launch geometry identical at 256 threads in both cases --
  so the thread doubling that macro can do is **not** the cause.  The cause is
  not identified: two builds with the same register cap and the same launch
  geometry, differing only in the ``maxThreadsPerBlock`` written in the bound,
  give answers 1e-13 and 3.7e+01 apart.  It is recorded because it is a trap
  for exactly the person B3's P3 prescription sends here, and because it means
  the bound has to be written at the block size the driver uses rather than at
  a safe upper limit.

  ``adjust_threads`` is disabled anyway, in the copy of ``vhf.cuh`` this has to
  vendor.  It doubles the quartets per block when a kernel uses 128 registers
  or fewer -- dead code on stock GPU4PySCF, because without a cap no ip2 kernel
  is ever under 128 -- and with the cap in place it is the one thing that could
  change the block shape out from under the bound.  ``--keep-adjust-threads``
  puts it back; at ``(256, 2)`` that is still correct.

Everything else -- the Rys recurrences, the per-atom contraction, the
screening, the symmetry factors, the launch schemes, the Python driver -- is
GPU4PySCF's, untouched.  The two entry points are renamed ``MYEJK2_*`` and
`electrolites/fastejk2.py` binds them over GPU4PySCF's on the ctypes handle, so
the driver above them is upstream's too.

``rsqrt(s)*rsqrt(s)`` differs from ``1/s`` by about two units in the last
place, which is the same substitution `fastk` and `fastejk` make and the reason
the agreement below is 1e-13 relative rather than bit for bit.

Run:  python codegen/gen_ejk_ip2.py /path/to/gpu4pyscf/lib/gvhf-rys \\
              electrolites/kernels/cuda           # P1 only, what ships
      ... --p3            # add the register cap: 0.58x, the B3 prescription
      ... --no-p1         # neither: a control that must reproduce upstream
"""
import argparse
import os
import re
import shutil
import sys

#: GPU4PySCF files copied verbatim, because the ip2 kernels need this release's
#: struct layouts and a wheel install carries no CUDA source.  They go in their
#: own directory rather than next to the 1.7.0 copies `fastk` uses: BoundsInfo
#: gained per-group q_cond/s_cond between the two, so the two sets are not
#: interchangeable.
SUPPORT = ('vhf.cuh', 'rys_roots.cu', 'rys_roots.cuh', 'rys_roots_for_k.cu',
           'gamma_inc.cu', 'rys_constant.cu', 'rys_roots_dat.cu',
           'create_tasks.cu', 'rys_contract_k.cuh', 'build_rys_gxyz.cuh',
           'unrolled_kernels.cuh')

#: the files that are transformed
KERNELS = ('rys_contract_jk_ip2.cu', 'unrolled_ejk_ip2_type12.cu',
           'unrolled_ejk_ip2_type3.cu')

#: `build_rys_gxyz.cuh` holds the shared 2D-recurrence macro, and three of the
#: per-root divisions live in it, so it is transformed too rather than copied
MACRO = 'build_rys_gxyz.cuh'


def _sub(text, old, new, where, expect=None):
    n = text.count(old)
    if expect is None:
        assert n >= 1, f'{where}: pattern not found: {old[:60]!r}'
    else:
        assert n == expect, f'{where}: {old[:60]!r} appears {n} times, ' \
                            f'expected {expect}'
    sys.stderr.write(f'  {where}: {n:3d} x {old.strip()[:64]}\n')
    return text.replace(old, new), n


def p1(text, where):
    """Fold every fp64 division out of the primitive and root loops."""
    total = 0
    # 1/akl, next to where the ket exponents are read: `al / akl` sits in the
    # klp loop, so one reciprocal there serves every primitive quartet under it
    pat = re.compile(r'( *)double al_akl = al / akl;')
    hits = pat.findall(text)
    assert hits, f'{where}: the ket primitive pair anchor is missing'
    text = pat.sub(lambda m: (f'{m.group(1)}double iakl = 1. / akl;\n'
                              f'{m.group(1)}double al_akl = al * iakl;'), text)
    sys.stderr.write(f'  {where}: {len(hits):3d} x ket primitive pair '
                     f'(iakl)\n')
    total += len(hits)
    # 1/aij and rsqrt(aij+akl), next to where the bra exponents are read.  The
    # anchor is the *pair* of lines, because `double aij = ai + aj;` also
    # appears in the cicj cache loop, where akl is not in scope.
    pat = re.compile(r'( *)double aij = ai \+ aj;\n( *)double aj_aij = aj / aij;')
    hits = pat.findall(text)
    assert hits, f'{where}: the bra primitive pair anchor is missing'
    text = pat.sub(lambda m: (
        f'{m.group(1)}double aij = ai + aj;\n'
        f'{m.group(1)}double iaij = 1. / aij;\n'
        # one rsqrt for 1/(aij+akl) and 1/sqrt(aij+akl) both
        f'{m.group(1)}double inv_s = rsqrt(aij + akl);\n'
        f'{m.group(1)}double inv_s2 = inv_s * inv_s;\n'
        f'{m.group(2)}double aj_aij = aj * iaij;'), text)
    sys.stderr.write(f'  {where}: {len(hits):3d} x bra primitive pair '
                     f'(iaij, inv_s, inv_s2)\n')
    total += len(hits)
    # the geometric prefactor
    for old, new in (
            ('cicj / (aij*akl*sqrt(aij+akl))', 'cicj * iaij * iakl * inv_s'),
            ('cicj * ckcl / (aij*akl*sqrt(aij+akl))',
             'cicj * ckcl * iaij * iakl * inv_s')):
        if old in text:
            text, n = _sub(text, old, new, where)
            total += n
    # theta, and the three per-root divisions
    for old, new in (('double theta = aij * akl / (aij + akl);',
                      'double theta = aij * akl * inv_s2;'),
                     ('double rt_aa = rt / (aij + akl);',
                      'double rt_aa = rt * inv_s2;'),
                     ('.5/aij', '.5*iaij'),
                     ('.5/akl', '.5*iakl')):
        if old in text:
            text, n = _sub(text, old, new, where)
            total += n
    return text, total


#: What may still contain a `/` after P1: integer index arithmetic, the two
#: reciprocals P1 itself introduces (one each per primitive pair, replacing
#: three to eighteen divisions per primitive *quartet*), and `theta_ij`, which
#: is evaluated once per bra primitive pair per block in the cicj cache loop
#: and is not in a hot loop at all.
_OK_DIV = re.compile(
    r'/ *(nbas|jprim|lprim|nfij|nfi|nfk|npairs_ij|bounds\.|3\b|9\b|2\b)'
    r'|double i(aij|akl) = 1\. / a(ij|kl);'
    r'|double theta_ij = ai \* aj / aij;'
    r'|#include|^\s*\*|/\*|\*/')


def check_no_fp64_div(text, where):
    """Every remaining `/` must be integer indexing or a comment."""
    bad = []
    for i, line in enumerate(text.split('\n'), 1):
        l = re.sub(r'//.*', '', line)
        if '/' not in l:
            continue
        if _OK_DIV.search(l):
            continue
        bad.append((i, l.strip()))
    return bad


#: `vhf.cuh`'s own register-driven thread doubling, which the P3 cap makes
#: reachable.  See the module docstring: a 512-thread ip2 block is measured to
#: compute wrong numbers, so nothing here may be able to produce one.
ADJUST = 'if (attr.numRegs <= 128) threads *= 2; }'
ADJUST_OFF = ('(void)attr; /* electrolites: disabled, see '
              'codegen/gen_ejk_ip2.py -- a 512-thread ip2 block is wrong */ }')


def p3(text, where, maxthreads, minblocks):
    """A per-kernel register cap.

    ``maxthreads`` is the largest block the Python driver ever asks for, 256,
    and stays 256 because `adjust_threads` is disabled alongside this.
    """
    pat = re.compile(r'__global__ static\nvoid (rys_ejk_ip2_\w+)\(')
    names = pat.findall(text)
    text = pat.sub(
        lambda m: f'__global__ static '
                  f'__launch_bounds__({maxthreads}, {minblocks})\n'
                  f'void {m.group(1)}(', text)
    sys.stderr.write(f'  {where}: {len(names):3d} x __launch_bounds__'
                     f'({maxthreads}, {minblocks}) '
                     f'(<= {65536 // (maxthreads * minblocks)} registers)\n')
    return text, len(names)


def rename_entries(text):
    """Our own symbols, so the library can sit beside GPU4PySCF's."""
    n = 0
    for name in ('RYS_per_atom_jk_ip2_type12', 'RYS_per_atom_jk_ip2_type3'):
        assert text.count(f'int {name}(') == 1
        text = text.replace(f'int {name}(', f'int MYEJK2{name[3:]}(')
        n += 1
    return text, n


#: the per-class instances of the general kernels: every class with a d shell
#: and nothing above it (the classes up to p are GPU4PySCF's unrolled ones)
TMPL_LMAX = 2
_CONSTS = (('int li = bounds.li;', 'constexpr int li = LI;'),
           ('int lj = bounds.lj;', 'constexpr int lj = LJ;'),
           ('int lk = bounds.lk;', 'constexpr int lk = LK;'),
           ('int ll = bounds.ll;', 'constexpr int ll = LL;'),
           ('int stride_j = bounds.stride_j;', 'constexpr int stride_j = LI + 2;'),
           ('int stride_k = bounds.stride_k;',
            'constexpr int stride_k = (LI + 2) * (LJ + 2);'),
           ('int stride_l = bounds.stride_l;',
            'constexpr int stride_l = (LI + 2) * (LJ + 2) * (LK + 2);'),
           ('int g_size = bounds.g_size;',
            'constexpr int g_size = (LI + 2) * (LJ + 2) * (LK + 2) * (LL + 2);'),
           ('int nroots = bounds.nroots;',
            'constexpr int nroots = (LI + LJ + LK + LL + 2) / 2 + 1;'),
           ('int nfi = bounds.nfi;', 'constexpr int nfi = (LI+1)*(LI+2)/2;'),
           ('int nfj = bounds.nfj;', 'constexpr int nfj = (LJ+1)*(LJ+2)/2;'),
           ('int nfk = bounds.nfk;', 'constexpr int nfk = (LK+1)*(LK+2)/2;'),
           ('int nfl = bounds.nfl;', 'constexpr int nfl = (LL+1)*(LL+2)/2;'))

_SM_HELPER = (
    '\n// electrolites (codegen/gen_ejk_ip2.py): the multiprocessor count of the\n'
    '// current device, queried once per device instead of once per launch\n'
    'static int myejk2_sm_count()\n'
    '{\n'
    '    static int cache[64] = {0};\n'
    '    int dev = 0;\n'
    '    cudaGetDevice(&dev);\n'
    '    if (dev < 0 || dev >= 64) {\n'
    '        int m = 0;\n'
    '        cudaDeviceGetAttribute(&m, cudaDevAttrMultiProcessorCount, dev);\n'
    '        return m;\n'
    '    }\n'
    '    if (cache[dev] == 0)\n'
    '        cudaDeviceGetAttribute(&cache[dev], cudaDevAttrMultiProcessorCount, dev);\n'
    '    return cache[dev];\n'
    '}\n')


def templated(text):
    """Per-class instances of the two general kernels (catalogue B3, the P6
    problem): the angular momenta, the strides, the g-array size, the root
    count and the cartesian counts become compile-time constants, so every
    index table, division and loop bound in the kernel is resolved by nvcc.
    The arithmetic is the general kernel's, unchanged.  The drivers launch an
    instance for every long-range class whose shells are all at most d, and
    the run-time kernel otherwise.  Also: the multiprocessor count is
    queried once per device, not per launch."""
    total = 0
    for kind in ('type12', 'type3'):
        name = f'rys_ejk_ip2_{kind}_kernel'
        head = f'__global__ static\nvoid {name}('
        i = text.index(head)
        # the kernel's own closing brace, by matching (its while-loop also
        # closes in column 0)
        j, depth = text.index('{', i) + 1, 1
        while depth:
            depth += (text[j] == '{') - (text[j] == '}')
            j += 1
        j += 1
        body = text[i:j]
        t = body.replace(head, 'template <int LI, int LJ, int LK, int LL>\n'
                               f'__global__ static\nvoid {name}_t(')
        consts = _CONSTS
        if kind == 'type3':
            # the type3 driver's g-array: j and l are not raised
            # (stride_k = stride_j*(lj+1), g_size = stride_l*(ll+1))
            consts = tuple(
                (o, n.replace('(LI + 2) * (LJ + 2)', '(LI + 2) * (LJ + 1)')
                     .replace('* (LL + 2)', '* (LL + 1)'))
                for o, n in _CONSTS)
        for old, new in consts:
            t = t.replace(old, new)
        assert 'bounds.li' not in t and 'bounds.nroots' not in t, kind
        text = text[:j] + '\n' + t + text[j:]
        launch = f'        {name}<<<workers, threads, buflen>>>('
        k = text.index(launch)
        end = text.index(');', k) + 2
        args = text[k + len(launch):end - 2]
        cases = []
        for li in range(TMPL_LMAX + 1):
            for lj in range(TMPL_LMAX + 1):
                for lk in range(TMPL_LMAX + 1):
                    for ll in range(TMPL_LMAX + 1):
                        if max(li, lj, lk, ll) < TMPL_LMAX:
                            continue
                        key = ((li * 3 + lj) * 3 + lk) * 3 + ll
                        f = f'{name}_t<{li},{lj},{lk},{ll}>'
                        cases.append(
                            f'        case {key}:\n'
                            f'            if (buflen > 48000) cudaFuncSetAttribute('
                            f'(const void *)&{f}, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen);\n'
                            f'            {f}<<<workers, threads, buflen>>>({args});\n'
                            f'            break;\n')
                        total += 1
        disp = ('        if (omega >= 0 && li <= %d && lj <= %d && lk <= %d && '
                'll <= %d) {\n'
                '        switch (((li * 3 + lj) * 3 + lk) * 3 + ll) {\n'
                % ((TMPL_LMAX,) * 4) + ''.join(cases) +
                '        default:\n'
                f'            {name}<<<workers, threads, buflen>>>({args});\n'
                '        }\n        } else\n')
        text = text[:k] + disp + text[k:]
    old = ('    cudaDeviceProp prop;\n'
           '    cudaGetDeviceProperties(&prop, 0);\n'
           '    int workers = prop.multiProcessorCount;\n')
    n = text.count(old)
    text = text.replace(old, '    int workers = myejk2_sm_count();\n')
    inc = '#include "build_rys_gxyz.cuh"\n'
    text = text.replace(inc, inc + _SM_HELPER, 1)
    sys.stderr.write(f'  templated: {total} class instances, {n} device queries '
                     f'cached\n')
    return text, total


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('src', help='GPU4PySCF lib/gvhf-rys of a 1.8.x checkout')
    ap.add_argument('out', help='electrolites/kernels/cuda')
    ap.add_argument('--maxthreads', type=int, default=256,
                    help='the largest block the Python driver asks for')
    ap.add_argument('--minblocks', type=int, default=2,
                    help='blocks per SM to fit; caps registers at '
                         '65536/(maxthreads*minblocks)')
    ap.add_argument('--keep-adjust-threads', action='store_true',
                    help="leave vhf.cuh's register-driven thread doubling in "
                         'place, which the P3 cap switches on and which then '
                         'computes wrong numbers -- the measurement behind '
                         'the module docstring')
    ap.add_argument('--no-templated', action='store_true',
                    help='no per-class instances of the general kernels and '
                         'the per-launch device query kept (the ablation)')
    ap.add_argument('--no-p1', action='store_true',
                    help='keep the divisions (the ablation)')
    ap.add_argument('--p3', action='store_true',
                    help='add the register cap.  **Off by default**, because '
                         'it is measured at 0.58x -- these kernels want all '
                         '255 registers and capping them to fit two blocks '
                         'per SM buys occupancy with spills that cost more '
                         '(docs/README_fastejk2.md 2b).  The flag exists so '
                         'that measurement can be reproduced, and because the '
                         'catalogue B3 entry asks for exactly this')
    a = ap.parse_args()

    sup = os.path.join(a.out, 'gvhf-rys181')
    ker = os.path.join(a.out, 'myejk2')
    os.makedirs(os.path.join(sup, 'gvhf-rys'), exist_ok=True)
    os.makedirs(ker, exist_ok=True)

    for f in SUPPORT:
        if f == MACRO:
            continue
        text = open(os.path.join(a.src, f)).read()
        if f == 'vhf.cuh' and a.p3 and not a.keep_adjust_threads:
            text, _ = _sub(text, ADJUST, ADJUST_OFF, f, expect=1)
        open(os.path.join(sup, 'gvhf-rys', f), 'w').write(text)
    sys.stderr.write(f'copied {len(SUPPORT)-1} support files'
                     f'{", vhf.cuh with adjust_threads disabled" if a.p3 and not a.keep_adjust_threads else ""}\n')

    # the shared 2D-recurrence macro carries three of the per-root divisions
    text = open(os.path.join(a.src, MACRO)).read()
    sys.stderr.write(f'{MACRO}:\n')
    if not a.no_p1:
        for old, new in (('double rt_aa = rt / (aij + akl); \\',
                          'double rt_aa = rt * inv_s2; \\'),
                         ('.5/aij', '.5*iaij'), ('.5/akl', '.5*iakl')):
            text, _ = _sub(text, old, new, MACRO)
    open(os.path.join(sup, 'gvhf-rys', MACRO), 'w').write(text)

    for f in KERNELS:
        text = open(os.path.join(a.src, f)).read()
        sys.stderr.write(f'{f}:\n')
        if not a.no_p1:
            text, _ = p1(text, f)
            bad = check_no_fp64_div(text, f)
            assert not bad, (f, bad[:6])
        if a.p3:
            text, _ = p3(text, f, a.maxthreads, a.minblocks)
        if f == 'rys_contract_jk_ip2.cu':
            text, _ = rename_entries(text)
            if not a.no_templated:
                text, _ = templated(text)
        open(os.path.join(ker, f), 'w').write(text)
        sys.stderr.write(f'  wrote {os.path.join(ker, f)}\n')


if __name__ == '__main__':
    main()
