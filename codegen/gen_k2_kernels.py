"""
Generate fastk's kernels for the exchange-matrix classes that GPU4PySCF splits
across threadIdx.y.

``gen_kernels.py`` handles the classes GPU4PySCF unrolls with one thread per
shell quartet; those keep the 2D integrals in registers, so the Rys roots can
go there too.  The classes here (``(dp|pp)``, ``(ds|dp)``, ``(dp|ds)``,
``(dd|ps)`` for a 6-31G* organic system) spread their ``gout`` accumulators
over ``threadIdx.y`` and stage the 2D integrals through shared memory, which
has to stay.  What this generator changes is the arithmetic around it, taking
the integral expressions verbatim from ``gvhf-rys/unrolled_rys_k.cu``:

  * **No double-precision division or sqrt in the innermost loop.**  GPU4PySCF
    evaluates ``aj/aij``, ``cicj/(aij*akl*sqrt(aij+akl))`` and
    ``aij*akl/(aij+akl)`` once per primitive quartet and ``rt/(aij+akl)``,
    ``.5/aij`` and ``.5/akl`` once per Rys root -- eleven DP divisions and a DP
    sqrt per primitive quartet for these three-root classes, each ~20
    instructions on sm_80.  ``1/aij`` is cached with the bra pair, ``1/akl``
    with the ket, one ``rsqrt(aij+akl)`` supplies both ``1/(aij+akl)`` and
    ``1/sqrt(aij+akl)``, and every division becomes a multiply.
  * **Block-uniform bra data computed once.**  For a given block the bra shell
    pair is fixed, so ``aij``, ``1/aij``, ``aj/aij`` and the Gaussian product
    centre depend only on the primitive-pair index.  GPU4PySCF recomputes them
    per thread per primitive quartet and stages the product centre through a
    two-element shared buffer; we compute them once per bra pair, next to the
    ``cicj`` cache it already builds.
  * **omega == 0 compiled in**, which drops the range-separation branches of
    the Rys root evaluation.
  * **Per-class __launch_bounds__.**  GPU4PySCF compiles these with no register
    limit at all; the right cap is a property of the class.

``--jk`` lifts the same six classes from ``gvhf-rys/unrolled_rys_jk.cu``
instead -- the *fused* Coulomb-and-exchange build the response path asks for
(catalogue B4b).  The two files' integral arithmetic is textually identical
(``gen_kernels.assert_same_integrals`` checks it); only the contraction differs,
so the same transformations apply unchanged.

Run:  python gen_k2_kernels.py <path-to-unrolled_rys_k.cu> > fastk2_generated.cu
      python gen_k2_kernels.py --jk <path-to-unrolled_rys_jk.cu> \
                                                       > fastjk2_generated.cu
"""
import re, sys, os, json

CLASSES = ['2021', '2111', '2120', '2210', '3011', '3110']

# MINBLOCKS caps registers at 65536/(256*MINBLOCKS).  Measured per class with
# bench/perclass_k2.py on an A100-SXM4-40GB (sm_80).
LAUNCH = {'2021': 2, '2111': 2, '2120': 2, '2210': 2, '3011': 2, '3110': 2}

MAX_PRIM_PAIR = 36      # matches fastk_prologue.cu


def _body_of(src, name, struct='kmat'):
    """The body of `name`, matched by braces: the generated kernels put the
    persistent while-loop's closing brace in column 0, so a regex that stops at
    the first '\n}\n' cuts the function short."""
    i = src.index(f'\nvoid {name}(RysIntEnvVars envs, JKMatrix {struct}, '
                  f'BoundsInfo bounds, int *pool, int *head)\n{{\n')
    j = src.index('{', i + 1)
    depth, k = 1, j + 1
    while depth:
        if src[k] == '{':
            depth += 1
        elif src[k] == '}':
            depth -= 1
        k += 1
    body = src[j+1:k-1]
    assert body.count('{') == body.count('}'), name
    return body


def _drop(body, line, name, count=None):
    pat = r'[ \t]*' + re.escape(line) + r'\n'
    body, n = re.subn(pat, '', body)
    assert n >= 1, (name, line)
    return body


def _sub1(body, old, new, name):
    assert body.count(old) == 1, (name, old, body.count(old))
    return body.replace(old, new)


def transform(body, tag, nroots, jk=False):
    st = 'jk' if jk else 'kmat'
    name = ('rys_jk_' if jk else 'rys_k_') + tag
    nsq = int(re.search(r'constexpr int nsq_per_block = (\d+);', body).group(1))
    g_size = int(re.search(r'constexpr int g_size = (\d+);', body).group(1))

    # ---- task queue and screening -----------------------------------------
    body = _sub1(body, 'pool + blockIdx.x * QUEUE_DEPTH',
                 'pool + blockIdx.x * queue_depth', name)
    body = _sub1(body, 'bounds.npairs_ij', 'npairs_ij', name)
    body = _sub1(body, 'bounds.pair_ij_mapping[pair_ij]',
                 'pair_ij_mapping[pair_ij]', name)
    # the operator branch, whose condition GPU4PySCF writes two ways
    v = 'jk' if jk else 'k'
    new_fill = ("""    fill_v%s_tasks2(&ntasks, bas_kl_idx, bas_ij, nbas, pair_kl_mapping,
                   npairs_kl, q_cond, dm_cond, cutoff);
""" % v)
    for cond in ('%s.lr_factor != 0' % st, '%s.omega >= 0' % st):
        old_fill = ("""    if (%s) {
        _fill_v%s_tasks(&ntasks, bas_kl_idx, bas_ij, envs, bounds);
    } else {
        _fill_sr_v%s_tasks(&ntasks, bas_kl_idx, bas_ij, envs, bounds);
    }
""" % (cond, v, v))
        if old_fill in body:
            body = _sub1(body, old_fill, new_fill, name)
            break
    else:
        raise AssertionError((name, 'no task-filter branch matched'))

    # ---- the struct fields are kernel arguments now ------------------------
    for line in ('int nbas = envs.nbas;', 'int *bas = envs.bas;',
                 'double *env = envs.env;', 'int *ao_loc = envs.ao_loc;',
                 'int nao = ao_loc[nbas];', 'int nroots = bounds.nroots;',
                 '__shared__ double aij_cache[2];'):
        body = _drop(body, line, name)
    for p in 'ijkl':
        body = body.replace(f'bounds.{p}prim', f'{p}prim_a')
    body = body.replace('bounds.nroots', 'NROOTS')
    body = re.sub(r'\bnroots\b', 'NROOTS', body)
    for v in ('dm', 'vk', 'vj'):
        body = body.replace(f'{st}.{v}', f'{v}_all')
    # The i_dm loop GPU4PySCF wraps its contraction in stays, with n_dm a
    # kernel argument: the quartet's 2D integrals are already in shared memory,
    # so n_dm densities cost n_dm contractions and one integral pass, which is
    # what a CPHF or TDDFT call needs (catalogue B4b).  int nao*nao overflows
    # at 339 densities on a 2268-AO basis, so the offset is computed in size_t.
    body = body.replace(f'{st}.n_dm', 'n_dm')
    body = body.replace('dm_all + i_dm * nao * nao',
                        'dm_all + i_dm * (size_t)nao * nao')
    body = body.replace('vk_all + i_dm * nao * nao',
                        'vk_all + i_dm * (size_t)nao * nao')

    # ---- one more slot in the ket cache, for 1/akl -------------------------
    body = _sub1(body, 'double *fac_ijkl = shared_memory + nsq_per_block * 8 + sq_id;',
                 'double *fac_ijkl = shared_memory + nsq_per_block * 9 + sq_id;', name)
    body = _sub1(body, 'double *gx = shared_memory + nsq_per_block * 9 + sq_id;',
                 'double *gx = shared_memory + nsq_per_block * 10 + sq_id;', name)
    body = _sub1(body, 'double *rw = shared_memory + nsq_per_block * (g_size*3+9) + sq_id;',
                 'double *rw = shared_memory + nsq_per_block * (g_size*3+10) + sq_id;', name)
    body = _sub1(body,
                 'double *cicj_cache = shared_memory + nsq_per_block * (g_size*3+NROOTS*2+9);',
                 'double *cicj_cache = shared_memory + nsq_per_block * (g_size*3+NROOTS*2+10);',
                 name)

    # ---- block-uniform bra data, computed once per bra shell pair ----------
    old = """    for (int ij = thread_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double theta_ij = ai * aj / aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
    }
"""
    new = """    double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
    for (int ij = thread_id; ij < iprim*jprim; ij += threads) {
        double ai = expi[ij/jprim];
        double aj = expj[ij%jprim];
        double aij = ai + aj;
        double iaij = 1. / aij;
        double aj_aij = aj * iaij;
        s_aij  [ij] = aij;
        s_iaij [ij] = iaij;
        s_ajaij[ij] = aj_aij;
        s_xij  [ij] = ri[0] + xjxi * aj_aij;
        s_yij  [ij] = ri[1] + yjyi * aj_aij;
        s_zij  [ij] = ri[2] + zjzi * aj_aij;
        cicj_cache[ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai * aj_aij * rr_ij);
    }
"""
    body = _sub1(body, old, new, name)

    # ---- 1/akl next to the ket product centre ------------------------------
    body = _sub1(body, """                double akl = ak + al;
                double al_akl = al / akl;""",
                 """                double akl = ak + al;
                double iakl = 1. / akl;
                double al_akl = al * iakl;""", name)
    body = _sub1(body, """                akl_cache[0] = akl;
                akl_cache[nsq_per_block] = al_akl;""",
                 """                akl_cache[0] = akl;
                akl_cache[nsq_per_block] = al_akl;
                akl_cache[2*nsq_per_block] = iakl;""", name)

    # ---- the primitive-quartet prologue reads the caches -------------------
    old = """                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double akl = akl_cache[0];
                double al_akl = akl_cache[nsq_per_block];
                double xij = ri[0] + (rjri[0]) * aj_aij;
                double yij = ri[1] + (rjri[1]) * aj_aij;
                double zij = ri[2] + (rjri[2]) * aj_aij;"""
    new = """                double aij = s_aij[ijp];
                double iaij = s_iaij[ijp];
                double akl = akl_cache[0];
                double al_akl = akl_cache[nsq_per_block];
                double iakl = akl_cache[2*nsq_per_block];
                double inv_s = rsqrt(fma(aij * akl, inv_om2, aij + akl));
                double inv_s2 = inv_s * inv_s;
                double xij = s_xij[ijp];
                double yij = s_yij[ijp];
                double zij = s_zij[ijp];"""
    body = _sub1(body, old, new, name)

    # ---- every division in the hot loop becomes a multiply -----------------
    body = _sub1(body, 'gx[nsq_per_block*g_size] = cicj / (aij*akl*sqrt(aij+akl));',
                 'gx[nsq_per_block*g_size] = cicj * iaij * iakl * inv_s * coef0;', name)
    body = _sub1(body, """                    if (sq_id == 0) {
                        aij_cache[0] = aij;
                        aij_cache[1] = aj_aij;
                    }
""", '', name)
    body = _sub1(body, 'double theta = aij * akl / (aij + akl);',
                 'double theta = aij * akl * inv_s2;', name)
    # The fused file calls a different root routine, with the block geometry
    # in its signature; both reduce to the same recurrences for omega >= 0,
    # and the omega scaling rides on inv_s here either way (see fastk.py).
    rs = re.search(r'rys_roots_rs\(NROOTS, theta, rr, %s\.omega, rw, '
                   r'nsq_per_block, gout_id, \d+\);' % st, body)
    if rs is not None:
        body = _sub1(body, rs.group(0),
                     f'rys_roots_tab<{nroots}>(theta * rr, rw, nsq_per_block,'
                     ' gout_id, blockDim.y, tab);', name)
    else:
        body = _sub1(body,
                 f'rys_roots_for_k(NROOTS, theta, rr, rw, {st}.omega,\n'
                     f'                                {st}.lr_factor, {st}.sr_factor);'
                     if f'rys_roots_for_k(NROOTS, theta, rr, rw, {st}.omega,\n' in body else
                     f'rys_roots_for_k(NROOTS, theta, rr, rw, {st}.omega, {st}.lr_factor, {st}.sr_factor);',
                     f'rys_roots_tab<{nroots}>(theta * rr, rw, nsq_per_block, gout_id,'
                     ' blockDim.y, tab);', name)
    body = _sub1(body, """                    double aij = aij_cache[0];
                    double rt_aa = rt / (aij + akl);
                    double akl = akl_cache[0];
""", '                    double rt_aa = rt * inv_s2;\n', name)
    body = body.replace('aij_cache[1]', 's_ajaij[ijp]')
    n = body.count('.5/aij')
    assert n >= 1, name
    body = body.replace('.5/aij', '.5*iaij').replace('.5/akl', '.5*iakl')

    assert 'aij_cache' not in body, name
    assert not re.search(r'\b(%s|bounds|envs)\.' % st, body), \
        (name, re.findall(r'\b(?:%s|bounds|envs)\.\w+' % st, body))
    leftover = [l for l in body.split('\n')
                if '/' in re.sub(r'//.*', '', l)
                and not re.search(r'(/ *nbas|% *nbas|/ *jprim|% *jprim|/ *lprim'
                                  r'|% *lprim|/ *2\b|/= *2)', l)
                and '1. / aij' not in l and '1. / akl' not in l]
    assert not leftover, (name, leftover)

    decl = (f'    __shared__ double s_aij[{MAX_PRIM_PAIR}];\n'
            f'    __shared__ double s_iaij[{MAX_PRIM_PAIR}];\n'
            f'    __shared__ double s_ajaij[{MAX_PRIM_PAIR}];\n'
            f'    __shared__ double s_xij[{MAX_PRIM_PAIR}];\n'
            f'    __shared__ double s_yij[{MAX_PRIM_PAIR}];\n'
            f'    __shared__ double s_zij[{MAX_PRIM_PAIR}];\n'
            f'    constexpr int NROOTS = {nroots};\n')
    # nsq*(3*g_size+10) for the 2D integrals and caches, NROOTS*2*nsq for the
    # roots, iprim*jprim for cicj; the last term is added by the launcher.
    shm_fixed = nsq * (3 * g_size + 10) + nroots * 2 * nsq
    return decl + body, nsq, g_size, shm_fixed


def generate(path, variants=False, jk=False):
    src = open(path).read()
    pfx = 'jk2' if jk else 'k2'
    args = 'JKARGS2' if jk else 'KARGS2'
    out = []
    table = {}
    for tag in CLASSES:
        raw = _body_of(src, ('rys_jk_' if jk else 'rys_k_') + tag,
                       'jk' if jk else 'kmat')
        nroots = sum(int(c) for c in tag) // 2 + 1
        body, nsq, g_size, shm = transform(raw, tag, nroots, jk)
        mb = LAUNCH[tag]
        gout_stride = 256 // nsq
        names = [('', mb)]
        if variants:      # keep the plain name so fastk.py still resolves
            names += [(f'_mb{m}', m) for m in (1, 2, 3, 4)]
        for suffix, mm in names:
            out.append(f'extern "C" __global__ void __launch_bounds__(256, {mm})\n'
                       f'{pfx}_{tag}{suffix}({args})\n{{\n{body}\n}}\n')
        table[tag] = dict(nsq=nsq, gout_stride=gout_stride, nroots=nroots,
                          g_size=g_size, shm_fixed=shm, minblocks=mb)
        sys.stderr.write(f'generated {pfx}_{tag}: nroots={nroots} g_size={g_size} '
                         f'threads={nsq}x{gout_stride} shm>={shm*8/1024:.1f}KB '
                         f'launch_bounds(256,{mb})\n')
    if not variants:
        _tab = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f'fast{pfx}_launch.json')
        with open(_tab + '.tmp', 'w') as f:
            json.dump(table, f, indent=1, sort_keys=True)
        os.replace(_tab + '.tmp', _tab)
    return '\n'.join(out)


if __name__ == '__main__':
    argv = sys.argv[1:]
    jk = '--jk' in argv
    argv = [x for x in argv if x != '--jk']
    variants = '--variants' in argv[1:]
    for spec in argv[1:]:
        if spec.startswith('--'):
            continue
        tag, mb = spec.split(':')
        LAUNCH[tag] = int(mb)
    print(generate(argv[0], variants, jk))
