# 从 HF 到 GPU 优化：细节补充

这组章节接在[主导读](../INTRO_FROM_HF.md)之后。主导读按 SCF → DFT → 梯度与响应建立整体图景；这里展开数学公式、GPU 执行方式和当前实现的边界。每章可独立阅读。

**源码基线：**Electrolites [`2f298e4`](https://github.com/Benzoin96485/Electrolites/tree/2f298e45452bb677ecbbce842faa95b305ffdc9e)；需要追到上游时，使用 GPU4PySCF **v1.8.1**，提交 [`5b284c2`](https://github.com/pyscf/gpu4pyscf/tree/5b284c258a4260baef80e3d150b4e7a81a9dbd57)。以下“当前实现”均指这个基线。仓库已有性能证据主要来自 A100；本次文档补充没有进行新的 GPU 性能或数值实验。

| 章节 | 要回答的问题 |
|---|---|
| [一、GPU 算术、共享内存与积分任务划分](01-gpu-and-integrals.md) | 1. fp64 除法为什么贵；2. 共享内存的统一术语；3. bra primitive pair 的不变量；4. 一线程一个 quartet 的适用条件；5. `(ss|ss)` 为什么只需求积权重 |
| [二、硬件相关的调优与成组矩阵乘法](02-architecture-and-grouping.md) | 6. `__launch_bounds__` 与 A100 以外的架构；7. grouped GEMM 的规模依赖、开发和测试 |
| [三、XC 的数据流与密度路线选择](03-xc-pipeline.md) | 8. 密度、泛函求值、势收缩如何组合；9. DM/MO 路线的真实判断逻辑 |
| [四、VV10 的误差界与线性响应](04-error-and-response.md) | 10. 裁剪误差界及其适用范围；11. 融合 J/K；12. CPHF/CPKS；13. VV10 中可缓存与不可缓存的部分 |
| [五、核导数内核与 AO 掩码](05-derivatives-and-masks.md) | 14. `fastejk` 与能量计算的区别；15. AO 掩码在哪里、如何产生，以及“贡献为零”的含义 |

建议先读主导读前五章，理解密度矩阵、J/K、壳层与 quartet，再按顺序读这里。已经熟悉这些概念的读者，可以直接从表中跳转。

本文统一将 CUDA **shared memory** 称为**共享内存**。程序把数据放入共享内存以复用时，称为“共享内存中的数据缓冲”；硬件 L1/L2 cache 称为“缓存”。缓存基态结果、缓存已编译内核等软件行为仍可称为“缓存”，不能因为中文同名就把它们当成同一层存储。
