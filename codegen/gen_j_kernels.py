"""
Generate fastj's per-class CUDA kernels for the Coulomb (J) matrix.

The integral arithmetic -- the Hermite (Rt) recurrences and the contraction of
the Hermite density with them -- is taken verbatim out of GPU4PySCF's generated
file gvhf-md/unrolled_md_j.cu, so both codes evaluate literally the same
expressions in the same order.  What this generator changes is the scaffolding
around that arithmetic:

  * the Boys/incomplete-gamma values move from dynamic shared memory (stride
    256) into registers;
  * the two double-precision divisions and the sqrt that GPU4PySCF evaluates
    once per shell-pair quartet become one rsqrt and a few multiplies, using
    1/aij and 1/akl cached alongside the Gaussian product centres they are
    already computed next to;
  * omega == 0 is compiled in, which removes the range-separation branches;
  * the shared-memory request drops by the freed gamma_inc block, and each
    class gets its own __launch_bounds__ instead of the family-wide
    __maxnreg__(128).

Run:  python gen_j_kernels.py <path-to-unrolled_md_j.cu> > fastj_generated.cu
"""
import re, sys, os, json

# Threads per block is 16x16 = 256 for every class GPU4PySCF unrolls.  MINBLOCKS
# caps registers at 65536/(256*MINBLOCKS); the right cap is set by how many
# vj_ij accumulators and Rt intermediates a class carries, which is a property
# of the angular-momentum class alone, not of the molecule.  Measured on an
# A100-SXM4-40GB (sm_80); re-tune with bench/sweep_j.sh on a different GPU.
NTHREADS = 256
HOIST = True
# Measured with bench/perclass_j.py --variants on an A100-SXM4-40GB (sm_80):
# each class was timed at MINBLOCKS 1..4 and the fastest kept.  The wide
# classes (large nf3ij, hence many vj_ij accumulators) need 128 or 256
# registers and spill badly at 3 or 4 blocks per SM.
LAUNCH = {
    '0_0': 4, '1_0': 4, '1_1': 4, '2_0': 4, '2_1': 3, '2_2': 2,
    '3_0': 2, '3_1': 2, '3_2': 1, '4_0': 2, '4_1': 1, '5_0': 1,
}

DROP_LINES = [
    r'int \*pair_ij_mapping = bounds\.pair_ij_mapping;',
    r'int \*pair_kl_mapping = bounds\.pair_kl_mapping;',
    r'float \*q_cond = bounds\.q_cond;',
    r'int \*bas = envs\.bas;',
    r'int \*pair_ij_loc = bounds\.pair_ij_loc;',
    r'int \*pair_kl_loc = bounds\.pair_kl_loc;',
    r'int nbas = envs\.nbas;',
    r'double \*env = envs\.env;',
    r'double \*vj = jk\.vj;',
    r'int npairs_ij = bounds\.npairs_ij;',
    r'int npairs_kl = bounds\.npairs_kl;',
    r'float \*qd_ij_max = bounds\.qd_ij_max;',
    r'float \*qd_kl_max = bounds\.qd_kl_max;',
    r'double omega = jk\.omega;',
]

HEAD = r'''
#define JARGS double *vj, double *dm_all, int *bas, double *env,       \
              int nbas, int npairs_ij, int npairs_kl,                  \
              int *pair_ij_mapping, int *pair_kl_mapping,              \
              int *pair_ij_loc, int *pair_kl_loc,                      \
              float *qd_ij_max, float *qd_kl_max,                      \
              float *q_cond, float cutoff
'''


def _body_of(src, name):
    m = re.search(r'\nvoid ' + name + r'\(RysIntEnvVars envs, JKMatrix jk, '
                  r'MDBoundsInfo bounds\)\n\{\n(.*?)\n\}\n', src, re.S)
    return None if m is None else m.group(1)


def transform(body, name, order):
    """Apply the rewrites described in the module docstring to one kernel body."""
    # --- shared-memory layout, needed to size the launch request -------------
    a = int(re.search(r'double \*Rq_cache = vj_kl_cache \+ (\d+);', body).group(1))
    b = int(re.search(r'double \*Rp_cache = Rq_cache \+ (\d+);', body).group(1))
    c = int(re.search(r'double \*dm_ij_cache = Rp_cache \+ (\d+) \+ tx;', body).group(1))
    bsizey = b // 4
    assert b == bsizey * 4, (name, b)
    nf3ij = int(re.search(r'for \(int n = ty; n < (\d+); n \+= 16\) \{', body).group(1))
    zero_to = int(re.search(r'for \(int n = thread_id; n < (\d+); n \+= 256\) \{',
                            body).group(1))
    # gamma_inc was the last block in the dynamic buffer; dropping it frees it.
    # vj_cache aliases Rp_cache and needs 256 doubles of its own.
    shm = a + b + max(256, c + 16 * nf3ij)
    # the prologue zeroes a prefix of the buffer; it must stay inside it
    assert shm >= zero_to, (name, shm, zero_to)

    # --- drop the struct unpacking, which is now in the argument list --------
    for pat in DROP_LINES:
        body, n = re.subn(r'[ \t]*' + pat + r'\n', '', body)
        assert n >= 1, (name, pat)
    body = body.replace('bounds.cutoff', 'cutoff')
    body = body.replace('double *dm = jk.dm', 'double *dm = dm_all')

    # no reference to the structs may survive
    assert not re.search(r'\b(jk|bounds|envs)\.', body), \
        (name, re.findall(r'\b(?:jk|bounds|envs)\.\w+', body))

    # --- Boys values in registers -------------------------------------------
    body, n = re.subn(r'[ \t]*double \*gamma_inc = Rp_cache \+ \d+ \+ sq_id;\n',
                      f'    double gamma_inc[{order+1}];\n', body)
    assert n == 1, name
    body, n = re.subn(r'gamma_inc\[(\d+)\*256\]', r'gamma_inc[\1]', body)
    assert n > 0, name
    assert not re.search(r'gamma_inc\[[^]]*256', body), name

    # --- 1/aij and 1/akl caches ---------------------------------------------
    # ket side: filled in the same loop that computes the product centre
    old = ('            double akl = ak + al;\n'
           '            double xkl = (ak * rk[0] + al * rl[0]) / akl;\n'
           '            double ykl = (ak * rk[1] + al * rl[1]) / akl;\n'
           '            double zkl = (ak * rk[2] + al * rl[2]) / akl;\n')
    new = ('            double akl = ak + al;\n'
           '            double iakl = 1. / akl;\n'
           '            double xkl = (ak * rk[0] + al * rl[0]) * iakl;\n'
           '            double ykl = (ak * rk[1] + al * rl[1]) * iakl;\n'
           '            double zkl = (ak * rk[2] + al * rl[2]) * iakl;\n'
           '            s_iakl[n] = iakl;\n')
    assert body.count(old) == 1, name
    body = body.replace(old, new)
    body = re.sub(r'(\n            Rq_cache\[n\+' + str(3 * bsizey) + r'\] = 1\.;\n)',
                  r'\1            s_iakl[n] = 1.;\n', body, count=1)
    assert body.count('s_iakl[n] = 1.;') == 1, name

    # bra side
    old = ('                double aij = ai + aj;\n'
           '                double xij = (ai * ri[0] + aj * rj[0]) / aij;\n'
           '                double yij = (ai * ri[1] + aj * rj[1]) / aij;\n'
           '                double zij = (ai * ri[2] + aj * rj[2]) / aij;\n')
    new = ('                double aij = ai + aj;\n'
           '                double iaij = 1. / aij;\n'
           '                double xij = (ai * ri[0] + aj * rj[0]) * iaij;\n'
           '                double yij = (ai * ri[1] + aj * rj[1]) * iaij;\n'
           '                double zij = (ai * ri[2] + aj * rj[2]) * iaij;\n'
           '                s_iaij[thread_id] = iaij;\n')
    assert body.count(old) == 1, name
    body = body.replace(old, new)
    old = '                Rp_cache[thread_id+48] = 1.; // aij\n'
    assert body.count(old) == 1, name
    body = body.replace(old, old + '                s_iaij[thread_id] = 1.;\n')

    # --- no division or sqrt in the innermost loop ---------------------------
    old = '            fac = fac / (aij*akl*sqrt(aij+akl));\n'
    new = ('            double inv_s = rsqrt(aij + akl);\n'
           '            fac *= s_iaij[tx] * s_iakl[sq_kl] * inv_s;\n')
    assert body.count(old) == 1, name
    body = body.replace(old, new)
    old = f'''            double theta = aij * akl / (aij + akl);
            boys_fn(gamma_inc, theta, rr, omega, fac, {order}, 0, 256);
'''
    new = f'''            double theta = aij * akl * (inv_s * inv_s);
            boys0_fn_reg<{order}>(gamma_inc, theta, rr, fac);
'''
    assert body.count(old) == 1, (name, order)
    body = body.replace(old, new)

    # The only floating-point divisions left may be the two reciprocals in the
    # tile prologues, which run once per bra/ket tile rather than per quartet.
    leftover = [l for l in body.split('\n')
                if '/' in re.sub(r'//.*', '', l)
                and not re.search(r'(/ nbas|% nbas|/ 2|/= 2|/ 16|/ \d+;|/\d)', l)
                and '1. / akl' not in l and '1. / aij' not in l]
    assert not leftover, (name, leftover)

    if HOIST:
        # The bra tile's product centre, aij and 1/aij do not change inside the
        # ket loop, but GPU4PySCF re-reads all four from shared memory on every
        # quartet.  Read them once per bra tile into registers instead.  The
        # barrier that guarded those reads moves up to just after the last
        # shared write of the bra tile (the Hermite density cache), so the ket
        # loop then carries no barrier at all.
        body = body.replace('s_iaij[tx]', 'iaij_t')
        old = re.search(r'( *)for \(int n = ty; n < \d+; n \+= 16\) \{\n'
                        r' *dm_ij_cache\[n\*16\] = dm\[n\];\n *\}\n', body)
        assert old, name
        ind = old.group(1)
        body = body[:old.end()] + (
            f'{ind}__syncthreads();\n'
            f'{ind}double xij = Rp_cache[tx+0];\n'
            f'{ind}double yij = Rp_cache[tx+16];\n'
            f'{ind}double zij = Rp_cache[tx+32];\n'
            f'{ind}double aij = Rp_cache[tx+48];\n'
            f'{ind}double iaij_t = s_iaij[tx];\n') + body[old.end():]
        old = ('            __syncthreads();\n'
               '            double xij = Rp_cache[tx+0];\n'
               '            double yij = Rp_cache[tx+16];\n'
               '            double zij = Rp_cache[tx+32];\n'
               '            double aij = Rp_cache[tx+48];\n')
        assert body.count(old) == 1, name
        body = body.replace(old, '')

    decl = (f'    __shared__ double s_iaij[16];\n'
            f'    __shared__ double s_iakl[{bsizey}];\n')
    return decl + body, shm, bsizey, nf3ij


def generate(path, variants=False):
    src = open(path).read()
    # (lij, lkl) -> order, from md_j_unrolled's dispatch table
    classes = sorted(
        {(int(m.group(1)), int(m.group(2)))
         for m in re.finditer(r'void md_j_(\d+)_(\d+)\(RysIntEnvVars', src)})
    out = [HEAD]
    table = {}
    for lij, lkl in classes:
        name = f'md_j_{lij}_{lkl}'
        body = _body_of(src, name)
        if body is None:
            sys.stderr.write(f'skip {name}: not found\n')
            continue
        order = lij + lkl
        tag = f'{lij}_{lkl}'
        body, shm, bsizey, nf3ij = transform(body, name, order)
        mb = LAUNCH[tag]
        bsizex = int(re.search(r'int task_ij0 = blockIdx\.x \* (\d+);', body).group(1))
        mbs = [1, 2, 3, 4] if variants else [mb]
        for m in mbs:
            suffix = f'_mb{m}' if variants else ''
            out.append(f'extern "C" __global__ void __launch_bounds__({NTHREADS}, {m})\n'
                       f'j_{tag}{suffix}(JARGS)\n{{\n{body}\n}}\n')
        table[tag] = dict(order=order, shm=shm, bsizex=bsizex, bsizey=bsizey,
                          nf3ij=nf3ij, minblocks=mb)
        sys.stderr.write(f'generated j_{tag}: order={order} shm={shm*8/1024:.1f}KB '
                         f'tile={bsizex}x{bsizey} launch_bounds({NTHREADS},{mb})\n')
    if not variants:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'fastj_launch.json'), 'w') as f:
            json.dump(table, f, indent=1, sort_keys=True)
    return '\n'.join(out)


if __name__ == '__main__':
    variants = '--variants' in sys.argv[2:]
    if '--nohoist' in sys.argv[2:]:
        HOIST = False
    for spec in sys.argv[2:]:                # e.g. 2_1:4
        if spec.startswith('--'):
            continue
        tag, mb = spec.split(':')
        LAUNCH[tag] = int(mb)
    print(generate(sys.argv[1], variants))
