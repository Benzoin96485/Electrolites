"""Write the "no vk atomics" ablation of the exchange kernels.

The density contraction that ends every K kernel finishes with a global
`atomicAdd` per accumulator -- twelve per shell quartet for `(ps|ps)`, 3356
across the generated files.  JoltQC replaces them with a block reduction over
its `pair_vk` accumulator, and the previous round's verdict on the *other*
JoltQC atomic (the warp-aggregated task counter, measured inside the noise
floor and reverted) says to find out what the atomics actually cost before
building anything.

This writes the kernels with every `atomicAdd(vk+...)` guarded by a condition
that is never true.  The gout contraction, the `dm` loads and the address
arithmetic all still run; only the store does not, and the compiler cannot
remove the contraction because its result is what the guard tests.  The
result is wrong by construction.  What it measures is the ceiling: no scheme
for collapsing those atomics -- a block reduction, a warp aggregation over
the lanes that share a `k0` -- can beat the version that does not do them at
all.

    python codegen/gen_noatomic.py
    FASTK_SRC=fastk_noatomic_generated.cu \\
    FASTK2_SRC=fastk2_noatomic_generated.cu \\
    FASTKH_SRC=fastkhigh_noatomic_generated.cu \\
        python benchmarks/konly.py --xyz ... --patch fastk
"""
import os
import re
import shutil
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                    'electrolites', 'kernels')
PAIRS = [('fastk_generated.cu', 'fastk_noatomic_generated.cu'),
         ('fastk2_generated.cu', 'fastk2_noatomic_generated.cu'),
         ('fastkhigh_generated.cu', 'fastkhigh_noatomic_generated.cu')]
# fastk resolves a high-l launch table from the source file name
TABLES = [('fastkhigh_launch.json', 'fastkhigh_noatomic_launch.json')]

_VK = re.compile(r'atomicAdd\(vk\+([^;]*), (\w+)\);')


def main(out_dir=HERE):
    for src, dst in PAIRS:
        text = open(os.path.join(HERE, src)).read()
        n = len(_VK.findall(text))
        text = _VK.sub(r'if (\2 == 1e300) atomicAdd(vk+\1, \2);', text)
        with open(os.path.join(out_dir, dst), 'w') as f:
            f.write(text)
        print(f'{dst}: {n} vk atomicAdds disabled', file=sys.stderr)
    for src, dst in TABLES:
        shutil.copyfile(os.path.join(HERE, src), os.path.join(out_dir, dst))


if __name__ == '__main__':
    main(*sys.argv[1:])
