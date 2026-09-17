# 三、XC 的数据流与密度路线选择

[返回章节目录](README.md) · [上一章](02-architecture-and-grouping.md) · [下一章：误差与响应](04-error-and-response.md)

## 8. “密度、泛函求值和势收缩组合”具体做了什么？

下面使用实数 AO、对称密度矩阵和限制性 LDA/GGA 说明 [`fastxc.py`](../../electrolites/fastxc.py) 的数据流。其他自旋或泛函类型需要相应分量，不能不加修改地套用这些式子。

对一个有 $g$ 个网格点、保留 $m$ 个 AO 的块，定义

$$
A_{\mu t}=\chi_\mu(\mathbf r_t),\qquad
A^x_{\mu t}=\partial_x\chi_\mu(\mathbf r_t),
$$

其中 $\mu=1,\ldots,m$、$t=1,\ldots,g$。$A,A^x,A^y,A^z$ 的形状都是 $m\times g$。$D$ 是从全局密度矩阵抽取相同 AO 行列得到的 $m\times m$ 子矩阵。

### 8.1 第一步：由 AO 和电子态得到网格密度

先计算 $B=DA$，再逐网格点收缩：

$$
\rho_t=\sum_\mu A_{\mu t}B_{\mu t},\qquad
\partial_x\rho_t=2\sum_\mu A^x_{\mu t}B_{\mu t}.
$$

第二式中的 2 使用了实对称 $D$：对两个 AO 因子求导得到的两项相同。$y,z$ 同理。所以 GGA 虽然需要四种密度分量，却可以复用同一个较昂贵的 $DA$ 矩阵乘法；其余是逐点的短收缩。

### 8.2 第二步：泛函把密度信息转成局域导数

LDA 能量可以写成 $E_{xc}\approx\sum_t w_t f(\rho_t)$；GGA 写成

$$
E_{xc}\approx\sum_t w_t f(\rho_t,\nabla\rho_t).
$$

这里 $f$ 是每体积能量密度。如果库返回的是每电子能量 $\epsilon_{xc}$，积能量时还要乘 $\rho$，不能把二者混淆。

泛函库求出能量及对密度变量的导数。GGA 的底层表示可能使用 $\sigma=|\nabla\rho|^2$，上游 `eval_xc_eff` 会为后续收缩提供对应有效分量；数学上可以用

$$
u_t=\frac{\partial f}{\partial\rho_t},\qquad
\mathbf v_t=\frac{\partial f}{\partial(\nabla\rho_t)}
$$

理解这些系数。这里的 $\mathbf v$ 是三维梯度变量的导数，不是完整 AO 势矩阵。

### 8.3 第三步：把局域导数收缩回 AO 势矩阵

对 $D_{\mu\nu}$ 求导，可得到示意式

$$
V_{\mu\nu}^{xc}=\sum_t w_t\left[
 u_t A_{\mu t}A_{\nu t}
 +\sum_{k=x,y,z}v_{k,t}
 (A^k_{\mu t}A_{\nu t}+A_{\mu t}A^k_{\nu t})\right].
$$

实现上，不需要为每个 $(\mu,\nu,t)$ 写一个 Python 循环。可以先形成加权 AO 数组，再做矩阵乘法，得到该块的 $m\times m$ 势矩阵，最后按 AO 索引加回全局矩阵。

例如定义 $\widetilde A_{\mu t}=w_t(u_tA_{\mu t}/2+\sum_kv_{k,t}A^k_{\mu t})$，则 $A\widetilde A^T$ 加其转置，恰好给出上式。这个构造解释了 GGA 代码中“某个标量分量先乘一半，最后再对称化”的来源，不是随意改变泛函系数。LDA 可以用更直接的 $A\operatorname{diag}(wu)A^T$。

### 8.4 组合的是数据流程，不是一个巨大的 CUDA 内核

原本可以先遍历全部网格求密度，把密度交给泛函库，再遍历网格求 AO 并收缩势。这样同一块 AO 可能被求值两次。`fastxc` 的 grouped 路线则是：

```text
收集一组网格块，计算并保留各块 AO/所需导数
    → 成组矩阵乘法，生成各块密度
    → 拼接本组密度；本组调用一次泛函求值
    → 按块切回局域权重，构造加权 AO
    → 成组矩阵乘法，生成各块势矩阵
    → 按各块 AO 索引加回全局矩阵
释放本组中间量，继续下一组
```

源码对应 `_nr_rks_grouped`、`_xc_and_weights`、`grouped_gemm`、`grouped_dot` 和 `add_sparse`。AO 在本组中复用，泛函调用也被摊到一组块上；电子数和能量在设备端累计，避免每个小块都取回 CPU 标量。

这仍然包含多个 GPU 内核与库调用，是上层流程的组合。只改成“每块独立求密度、调用泛函、算势”，会制造大量小泛函调用；仓库的 perblock 消融正说明，少走一遍 AO 并不必然更快。成组和复用要一起看。[已有消融结果](../README_fastxc.md)

## 9. DM 与 MO 路线究竟如何选择？

### 9.1 两种公式为什么等价？

若占据轨道系数为 $C$、占据数为 $f_i$，先把 $\sqrt{f_i}$ 吸收到列中，记为 $\widetilde C$，就有

$$
D=\widetilde C\widetilde C^T,\qquad
\Psi=\widetilde C^TA,\qquad
\rho_t=\sum_i\Psi_{it}^2.
$$

DM 路线使用上面的 $DA$；MO 路线先形成每个占据轨道在网格上的值 $\Psi$，再平方求和。GGA 的 MO 路线还要形成 $\Psi^x,\Psi^y,\Psi^z$，利用 $\partial_x\rho_t=2\sum_i\Psi_{it}\Psi^x_{it}$。

### 9.2 代码使用的是算术量启发式

令 $o$ 为非零占据轨道数，$n_{var}$ 为所需 AO 分量数（LDA 为 1，GGA 为 4），主要矩阵乘法的运算量近似为

$$
W_{DM}\approx2m^2g,\qquad
W_{MO}\approx2n_{var}omg.
$$

忽略较小的逐点收缩后，DM 更便宜的条件是

$$
m<n_{var}o.
$$

`_rho_route` 正是用这个比较。举例，GGA 的 $m=320,o=546$，MO 与 DM 的这个成本估计比约为 $4\times546/320=6.83$。这只是该步骤的算术量比，不是整个 XC 构建或 SCF 的加速比。反之，若 $m=200,o=30$，阈值只有 120，估计会选 MO。

它没有测量当前 GPU 上两条路线的实际运行时间，也没有显式建模访存、tile 利用率、临时数组容量、构造 $D$ 的成本和元数据开销。因此它是便宜且有物理尺寸依据的选择规则，不是保证全局最优的自动调参。

### 9.3 默认 grouped 模式不是每块独立选择

当前实现的准确行为是：

| 情形 | 行为 |
|---|---|
| 没有 MO 信息，仅有普通密度矩阵 | 用 DM；不能凭空改走 MO |
| 限制性 `perblock` 模式，DM/MO 均可用 | 用该块的 $m$ 比较阈值 |
| 限制性默认 `grouped` 模式，DM/MO 均可用 | 用**该组最大的 $m$** 比较，整组统一选择 |
| `FASTXC_RHO=dm/mo` | 在相应数据可用时覆盖自动比较，用于消融 |
| 当前 `nr_uks` 实现 | 使用自旋密度矩阵路线，不调用上述 RKS 的自动选择规则 |

统一组内路线方便把矩阵乘法成组组织，但不保证每个块都选到单独最便宜的路径。例如同组两个块的 $m$ 分别为 80 和 180，阈值为 120，grouped 会统一选择 MO，尽管 80 的块按局部估算更适合 DM。

`_prep` 会从带 MO 标签的输入提取占据轨道，吸收占据数并构造相应密度矩阵。未带标签的普通 $D$ 则没有这种信息。当前 RKS 的这套快速路径主要覆盖 LDA/GGA；不能把 UKS 的 meta-GGA 支持或响应密度的处理直接套进上述 $n_{var}$ 规则。完整条件应查看 `_prep`、`_rho_route`、`nr_rks` 和 `nr_uks`。[当前源码](../../electrolites/fastxc.py)
