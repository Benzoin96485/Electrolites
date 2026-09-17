# 二、硬件相关的调优与成组矩阵乘法

[返回章节目录](README.md) · [上一章](01-gpu-and-integrals.md) · [下一章：XC 数据流](03-xc-pipeline.md)

## 6. `__launch_bounds__` 的结论是否依赖 A100？

**依赖。**可推广的是“控制每个任务的资源占用，从而在并行度与单线程效率之间取舍”；不能直接推广的是某个角动量类的最佳线程数、寄存器用量或 `minBlocks` 值。

### 6.1 这个声明实际影响什么？

典型声明为：

```cpp
__global__ __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
void kernel(...);
```

第一个参数告诉编译器允许的最大线程块尺寸，实际启动不能超过它。第二个参数参与编译器对寄存器资源的规划：程序希望每个 SM 能容纳至少这么多个对应尺寸的线程块，编译器据此调整寄存器使用策略。

这**不是调度器保证同时运行这么多个 block 的命令**。共享内存、实际启动尺寸、硬件限制和工作量仍可能使实际驻留数更低。寄存器约束过紧时，编译器可能增加指令、重算中间量，或将值溢出到 local memory；约束宽松时，单线程可能更快，但同时驻留的线程减少。更高的 occupancy（活跃 warp 占硬件上限的比例）也不保证总时间更短。[NVIDIA 对 launch bounds 的说明](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#launch-bounds)

一个简化估算：SM 若有 65536 个 32 位寄存器，每 block 256 个线程，每线程使用 128 个寄存器，仅按寄存器算可容纳 2 个 block；若降到 64 个，则可容纳 4 个。真实分配存在粒度和其他限制，这个除法只是解释取舍，不能代替实际 occupancy 查询。

### 6.2 更新的 GPU 会不会改变最佳值？

会，而且不能只按上市时间判断。Hopper 的共享内存容量增加，并引入新的数据搬运能力，但其每 SM 寄存器文件和每线程寄存器上限并未同比扩大。旧代码也不会因运行在新设备上就自动变成利用新指令的数据流。[Hopper 调优指南](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)

甚至同为 Blackwell，不同计算能力对应的资源也不同：NVIDIA 指南列出的 cc 10.0 每 block 共享内存上限为 227 KB，而 cc 12.0 为 99 KB；两者每 SM 最大 warp 数也不同。举例来说，如果一个 J 配置在 A100 上合法且需要 130 KB 共享内存，它就不能原样用于这个 99 KB 上限的目标。这里比较的是**每 block 上限**，不能拿每 SM 总量代替。[Blackwell 调优指南](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)

因此要把三个问题分开：

1. **能否编译和启动？** 编译目标、CUDA/CUTLASS 版本、指令支持、每 block 资源必须满足要求。
2. **结果是否正确？** 换架构或编译器后，重新检查矩阵、能量及目标性质。
3. **是否仍然最快？** 在该设备重新比较候选配置，原来的赢家可能改变。

不能由“A100 上减少共享内存更快”推出“H100 上也应减少相同比例”，也不能由“新卡整体更快”推出“该补丁相对上游的加速比更高”。上下游可能受益于不同硬件改进。

### 6.3 仓库已经做了什么，还缺什么证据？

仓库的生成器和按角动量分类的配置表，使研究者可以改变任务布局、寄存器与共享内存安排。[`sweep_j_high.py`](../../benchmarks/sweep_j_high.py) 展示了扫描线程布局、`gout_stride`、`minblocks` 等参数，并先验证 AO 空间 J、再计时的开发方法。验证最终 J 很重要：改变筛选分块可能改变极小贡献的保留方式，中间 Hermite 分量的相对差异不能直接作为物理结果错误的判据。

但不能把这描述为“每次在任意 GPU 上启动都会自动调参”。当前文档主要提供 A100 的测量，没有足够的 H100/B200 等交叉设备数据来声明同一配置仍最优。此外，上述扫描脚本在本基线仍引用根目录 `codegen`，实际生成器位于 `electrolites/codegen`；复用这类开发脚本前需要核对导入路径，不能把历史辅助脚本都视为开箱即用的接口。本次只改文档，没有修复或运行该 GPU 扫描。

移植时应记录 GPU 型号与计算能力、CUDA/驱动、编译选项和库版本；先排除资源不合法配置，再对正确候选计时，并把冷启动编译成本与热运行分开。还应覆盖不同角动量、收缩深度和密度批量，避免只为一个分子挑选赢家。

## 7. grouped GEMM 是否也依赖规模和架构？

**依赖，而且“分子大小”只是间接指标。**直接决定工作形状的是每块保留 AO 数 $m$、网格点数 $g$、占据轨道数 $o$、一个组里的块数，以及矩阵乘法使用的 tile 和数据类型。

### 7.1 成组为什么可能更快，也可能不划算？

假设一个势矩阵乘法输出为 $m\times m$，内维为 $g$。如果它只能切成很少几个输出 tile，单独启动时可能没有足够工作分给整张 GPU。把许多独立块成组提交，能增加同时待处理的 tile 数，并摊薄启动成本。

代价包括组织各矩阵的尺寸、指针和步长，准备工作区，以及同时保留多块 AO、中间密度和加权 AO 的显存。若只有一两个块，或一个乘法本身已经足以利用 GPU，这些成本可能超过收益。很不均衡的矩阵尺寸还会造成负载不均衡。

更大的分子往往提供更多块，因此成组机会更多；但弥散基组、网格疏密、AO 筛选、泛函导数需求，都能在原子数不变时显著改变矩阵形状。更大的显存有助于容纳更多块，但不能保证吞吐率同比提高。

### 7.2 当前到底用了什么分组和硬件判断？

Electrolites 调用上游的 `_grouped_block_loop`、`grouped_gemm` 和 `grouped_dot`，并没有实现通用的在线 GEMM 自动调优器。

在 GPU4PySCF v1.8.1 中，`_grouped_block_loop` 读取当时可用显存，并累计已经求出的 AO 数组；累计 AO 字节数**超过约 20% 可用显存**时交出一组。判断在添加块之后进行，且这个计数不是整个工作流程的峰值内存上限：后续密度、加权 AO、矩阵输出和底层工作区仍需空间。组越大越好、一定不会内存不足，这两种解释都不成立。[上游分组实现](https://github.com/pyscf/gpu4pyscf/blob/5b284c258a4260baef80e3d150b4e7a81a9dbd57/gpu4pyscf/dft/numint.py)

Python helper 会组织各组矩阵的尺寸和指针，底层再调用 CUTLASS grouped 内核。此版本 `grouped_gemm.cu` 按计算能力 `<80` 与 `>=80` 选择 SIMT 与 Sm80 双精度 TensorOp 模板；这是一项粗粒度架构分支，不是针对 Hopper/各类 Blackwell 分别寻找最佳内核。还需区分双精度矩阵指令与宣传材料里的低精度 AI 吞吐率，后者不能直接用来估计这里的速度。[Python helper](https://github.com/pyscf/gpu4pyscf/blob/5b284c258a4260baef80e3d150b4e7a81a9dbd57/gpu4pyscf/lib/cupy_helper.py)、[grouped GEMM 内核](https://github.com/pyscf/gpu4pyscf/blob/5b284c258a4260baef80e3d150b4e7a81a9dbd57/gpu4pyscf/lib/cupy_helper/grouped_gemm.cu)

原模块说明使用 `324×324` 输出、`64×64` tile 的历史例子解释“约 36 个 tile”。它说明的是任务不足的机制，不是当前所有 grouped 算子的统一尺寸。v1.8.1 的 Sm80 `grouped_dot` 模板使用 `128×128` 输出 tile，同样大小的输出按这两维划分为 9 块；实际调度还取决于对应内核，不能把这些数直接等同于测得的 SM 利用率。[grouped dot 内核](https://github.com/pyscf/gpu4pyscf/blob/5b284c258a4260baef80e3d150b4e7a81a9dbd57/gpu4pyscf/lib/cupy_helper/grouped_dot.cu)

### 7.3 开发与测试目前如何处理？

现有证据有三个层次，不应混写：

| 层次 | 仓库提供的内容 | 能说明什么 |
|---|---|---|
| 开发消融 | `FASTXC_MODE=grouped/perblock`、限制性路径的 `FASTXC_RHO=auto/dm/mo`；[早期 XC 报告](../README_fastxc.md)比较不同规模和块尺寸 | 成组、路线选择、单遍 AO 复用分别贡献什么 |
| 正确性回归 | [`test_uxc.py`](../../tests/test_uxc.py)比较 grouped/perblock 与上游；[`test_full.py`](../../tests/test_full.py)安排独立进程的完整 SCF/梯度比较 | 相同输入下势矩阵、能量和最终计算是否一致；也覆盖部分回退条件 |
| 硬件性能证据 | 文档中的 A100 单卡结果 | 这些配置在所列环境有效，不能证明所有更新架构同样加速 |

早期报告中的 `bench/check_xc.py` 等路径不全是当前目录结构；复现实验应先查实际脚本和其参数。当前 UKS 覆盖也比早期报告完整，以源码和现有测试为准。

对新 GPU 的合理补充实验是：固定同一份密度与网格，对 grouped/perblock、auto/dm/mo 分别测势矩阵误差、热运行时间和显存峰值；再跑完整 SCF 检查总时间与迭代数。小/中/大体系、稠密/局域 AO、不同泛函都要覆盖。这里是建议的验证矩阵，**不是声称仓库已经完成这些跨架构实验**。
