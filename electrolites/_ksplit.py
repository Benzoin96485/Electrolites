"""Compile only the kernels a basis set can actually launch.

Every generator here emits one CUDA file per family and the driver hands the
whole file to NVRTC.  That is 58 000 lines for the exchange build and 140 000
for its gradient, and a first run pays for all of it -- about 30 s for `fastk`,
more for `fastejk` -- even for a 6-31G* job, which never launches a single
kernel with an f or a g shell in it.  Compile options do not help: measured
over the whole `fastk` module, `--minimal` and `--use_fast_math` together are
worth 4 %, because the cost is the volume of generated code and not NVRTC's
fixed overhead.

So the fix is to hand NVRTC less code.  A generated file is a preamble
followed by independent ``extern "C" __global__`` kernels, each named for its
angular-momentum class, so the driver can name the classes its basis reaches
and compile exactly those.  The subset is derived from the *set of angular
momenta in the basis*, never from how many pairs survive screening, so two
molecules in the same basis produce the same subset, the same source text and
therefore the same CuPy disk-cache entry.

``ELECTROLITES_FULL_MODULE=1`` compiles every kernel, which is the ablation
and the escape hatch if a driver ever asks for a kernel it did not declare.
"""
import functools
import os
import re

FULL = bool(int(os.environ.get('ELECTROLITES_FULL_MODULE', 0)))

_KERNEL = re.compile(r'extern\s+"C"\s+__global__[^\n]*\n\s*([A-Za-z_]\w*)\s*\(')
#: `fastk` and `fastejk` put each class's real work in a
#: ``template <int NRANGE> ... kbody_<class>`` that the ``extern "C"`` wrappers
#: instantiate.  A subset that calls one it did not keep would not compile, so
#: `source` checks for that and falls back to the whole file.
_HELPER_USE = re.compile(r'\b(kbody_\w+)\s*<')
_HELPER_DEF = re.compile(r'\n(kbody_\w+)\s*\(')


def _body_end(src, pos):
    """Index just past the closing brace of the definition starting at ``pos``.

    Brace matching rather than "the next line that is a bare ``}``", because
    `fastk` and `fastejk` emit their wrappers as one-liners
    (``k_1000(KARGS) { kbody_1000<1>(KFWD); }``) and a line rule would run
    past them into the next class's body.  Comments and character/string
    literals are skipped so that a brace inside one cannot unbalance the
    count.
    """
    i = src.find('{', pos)
    if i < 0:
        return len(src)
    depth = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c == '/' and i + 1 < n:
            if src[i+1] == '/':
                i = src.find('\n', i)
                if i < 0:
                    return n
                continue
            if src[i+1] == '*':
                j = src.find('*/', i + 2)
                i = n if j < 0 else j + 2
                continue
        elif c in '"\'':
            q, i = c, i + 1
            while i < n and src[i] != q:
                i += 2 if src[i] == '\\' else 1
            i += 1
            continue
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


@functools.lru_cache(maxsize=32)
def chunks(path):
    """``(preamble, {kernel name: source text})`` for one generated file.

    A chunk runs from the **end of the previous kernel's body** to the end of
    this one's, not from one ``extern "C"`` to the next.  The difference
    matters: `fastk` and `fastejk` put the real work in a
    ``template <int NRANGE> __device__ ... kbody_<class>`` that sits *between*
    the previous class's wrappers and its own, so cutting at ``extern "C"``
    would file each class's body under the previous class's name and dropping
    a class would take the next class's body with it.  Cutting at the closing
    brace keeps a kernel with everything declared for it.

    The first chunk carries the file's shared prologue, so it is always kept
    (its class is ``0000``/``0_0``, which every basis reaches anyway).
    """
    src = open(path).read()
    starts = [(m.start(), m.group(1)) for m in _KERNEL.finditer(src)]
    if not starts:
        return src, {}
    ends = [_body_end(src, pos) for pos, _ in starts]
    out = {}
    lo = 0
    for n, (pos, name) in enumerate(starts):
        out[name] = src[lo:ends[n]]
        lo = ends[n]
    if lo < len(src):                       # trailing text belongs to no one
        out[starts[-1][1]] += src[lo:]
    return '', out


def source(files, names=None):
    """Concatenate ``files``, keeping only the kernels in ``names``.

    ``files`` is a list of paths: those with no ``extern "C" __global__`` in
    them (a prologue) are taken whole.  ``names`` of ``None`` -- or
    ``ELECTROLITES_FULL_MODULE=1`` -- keeps everything.
    """
    parts = []
    for path in files:
        preamble, ks = chunks(path)
        parts.append(preamble)
        if not ks:
            continue
        first = next(iter(ks))
        keep = list(ks) if (names is None or FULL) else [
            k for k in ks if k in names or k == first]
        text = ''.join(ks[k] for k in keep)
        missing = (set(_HELPER_USE.findall(text))
                   - set(_HELPER_DEF.findall(text)))
        if missing:
            # A helper a kept kernel needs was filed under a dropped one.
            # Never risk it: take the whole file and lose only compile time.
            text = ''.join(ks.values())
        parts.append(text)
    return ''.join(parts)


def counts(files, names=None):
    """``(kept, total)`` kernels, for logging and for the benchmark."""
    kept = total = 0
    for path in files:
        _, ks = chunks(path)
        total += len(ks)
        first = next(iter(ks), None)
        kept += len(ks) if (names is None or FULL) else sum(
            1 for k in ks if k in names or k == first)
    return kept, total


_QUARTET = re.compile(r'_(\d)(\d)(\d)(\d)$')
_PAIRSUM = re.compile(r'_(\d+)_(\d+)$')


def names_within(files, lmax):
    """Kernel names in ``files`` whose class a basis of this ``lmax`` reaches.

    Two naming conventions appear: the J/K/gradient quartet kernels end in the
    four angular momenta ``(li,lj,lk,ll)``, and the McMurchie-Davidson J
    kernels end in the two pair sums ``(lij,lkl)`` -- a basis with maximum
    angular momentum ``lmax`` reaches ``lij <= 2*lmax``.  A name matching
    neither is always kept, since nothing can be concluded about it.
    """
    keep = set()
    for path in files:
        _, ks = chunks(path)
        for name in ks:
            m = _QUARTET.search(name)
            if m:
                if max(int(c) for c in m.groups()) <= lmax:
                    keep.add(name)
                continue
            m = _PAIRSUM.search(name)
            if m:
                if max(int(c) for c in m.groups()) <= 2 * lmax:
                    keep.add(name)
                continue
            keep.add(name)
    return frozenset(keep)
