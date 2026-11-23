# GPU4PySCF 1.8.1 → 1.9.x：兼容性审计与移植计划

2026-11-23。本文是审计结果和工作计划，还没有改任何模块代码。审计对象是 `main`
（55e1ce5）上的 Electrolites，以及上游 GPU4PySCF 的 `v1.9.0`（2026-10-07）、
`v1.9.1`（11-03）、`v1.9.2`（11-10）三个 tag。上游的变化见它的 `CHANGELOG`。

---

## 0. 结论

1. **在 1.9.0 上，Electrolites 目前整体失效。**
   * `patch_all()` 一个模块都不会打。原因是 `fastpbcgrad` 的目标
     `pbc.grad.krhf.grad_elec` 被删掉了，而 strict 检查只要有一个目标缺失就拒绝全部。
   * 改成逐模块放行后再测：在 1.8.1 上通过的 90 个测试里，有 52 个在 1.9.0 上失败；
     另有 2 个原本受环境限制的测试也因 1.9 的新问题失败（§2）。
2. **最危险的是不报错、只给错数字的情况。** 实测有 6 个测试属于这一类，全部由
   Electrolites 自己造成：
   * DF 梯度 `ejk` 偏 13 倍；
   * DF-MP2 梯度和双杂化梯度的误差达到 1e7–1e45；
   * 过渡金属配合物（f/g 壳层）的 JK 误差 0.45–0.74，ejk 误差 1e-2；
   * DFT 的 C6 偏 60 %；
   * NEVPT2 偏 0.36。

   根源有两类：
   * 上游给一批 C kernel 的参数表**中间插入了三个 double** `omega, lr_factor, sr_factor`（#873、#859）。
     Electrolites 仍按 1.8.1 的参数表用 ctypes 调用。在 x86-64 上整数和指针参数照样对得上，
     double 却从残留的 XMM 寄存器里读，所以不会崩，只会算错。
   * `int4c2e` 重写（#936）后直接返回完整对称的 ERI，`fastc6` 又自己对称化了一次。

   `check_targets()` 原理上看不到这一类问题。
3. **加速本身大多仍然成立。**
   * 直接 SCF 的 J/K：上游的展开类数量完全没变（K 25 类、JK 25 类、ejk_ip1 18 类、
     rys_jk_ip1 33 类、ip2 7+7 类，`unrolled_md_j.cu` 未改）。
   * VV10 的 C kernel 字节级未变。
   * `fastdf`、`_dfblock` 的 K 分块，以及 DF 梯度复用 cderi，上游都没有对应的改动。
   * 上游已经覆盖或部分覆盖的工作：
     * `fastdfrsh`（#859 的 `mix_inside_kernel`）；
     * `fastdfhessip` 的 W31 / W29（#934、#970）；
     * `fastdfcderi` 的 `sp_mult`（#970）；
     * `sap`（#997）；
     * `stability` 的一部分（#1005）；
     * `fixes` 的 D1、D2、D14；
     * `fastqmmm` 的能量缓存（#914）；
     * `fastmp2` 的半变换（`_ao2mo_general`）。
4. **建议直接以 1.9.2 为目标，而不是 1.9.0。** 1.9.0 自身有这些缺陷：
   * MP2 和 CCSD 的屏蔽积分未初始化（#995）；
   * 多密度 RSH 的 J 有误（#990）；
   * 网格分组在 128 个原子以上时有竞态（#995）；
   * 直接 SCF 的梯度和 Hessian kernel 缺同步（#1024）；
   * DF 预取有竞态（#1036）；
   * GHF/GKS 的若干问题，包括广义自旋 J/K 的组装（#1004）。本次实测：GHF TDHF 在
     1.9.0 上偏 0.5 eV，到 1.9.2 上恢复正常；stock GKS mcol 的能量在 1.9.0 上与
     1.8.1 差 1.5e-2 Eh。

   如果以 1.9.0 为基线测，基准本身就会受这些缺陷污染。另外，1.9.1 又改了一轮
   `RYS_fill_int4c2e`、`RYS_build_j` 的签名和 `_response_functions` 的结构，
   目标定在 1.9.0 会白做一部分工作。兼容层继续按"能力探测"保留对 1.8.1 的支持。
5. **在 Phase 1–2 完成之前，不要把共享环境 `barw-dev` 升到 1.9.x。** 这个环境对
   m4787 组内的人可读。Electrolites 现在在 1.9.x 上"不出错"，完全是因为
   `fastpbcgrad` 的目标碰巧缺失。一旦有人用 `strict=False`，或者用 `patch()` 只打部分
   模块，拿到的就是静默的错误结果。Phase 0 的第一项就是把这个偶然变成显式的版本门。
6. **工作量较大**：单人估计 23–36 个工作日（§6），所以按要求先写成计划，没有动代码。

---

## 1. 环境与方法

### 1.1 并排安装（不改共享环境）

| 版本 | 位置 | 说明 |
|---|---|---|
| 1.8.1 | `barw-dev` 默认 | LibXC 7.0.0（`gpu4pyscf-libxc-cuda12x 0.8.1`） |
| 1.9.0 | `PYTHONPATH=/pscratch/sd/w/weiliang/g4pver/1.9.0:/pscratch/sd/w/weiliang/g4pver/_libxc090` | `pip install --no-deps --target`；LibXC 7.1.2（`gpu4pyscf-libxc-cuda12x 0.9.0`，即新装时会拉到的版本） |
| 1.9.2 | `PYTHONPATH=/pscratch/sd/w/weiliang/g4pver/1.9.2:/pscratch/sd/w/weiliang/g4pver/_libxc090` | 同上 |
| 上游源码 | `/pscratch/sd/w/weiliang/g4p-upstream-src/gpu4pyscf` | 带 tag 的 git clone，用于 `git diff v1.8.1 v1.9.0` |

1.9.0 的依赖变化：`gpu4pyscf-libxc` 从 `==0.8.1` 放宽为 `>=0.8.1`；`pyscf-dispersion`
移到了 extra。`$PSCRATCH` 会被清理，丢了之后照上表的命令重装即可。

### 1.2 静态审计

* `electrolites.check_targets()` / `check_absent()`：
  * 1.9.0 上只有 `fastpbcgrad` 缺目标；
  * 1.9.2 上又多了 `fixes`（`UKS.get_vasp` 被删）；
  * 1.9.2 上 `stability`、`sap` 的槽位已被上游占用，所以
    `enable_all(strict=True)` 会抛异常。
* 符号级 diff（`benchmarks/upgrade_audit/`）：用 AST 列出各模块引用的 589 个上游符号，
  分别在 1.8.1、1.9.0、1.9.2 中解析，比较签名和源码。1.8.1→1.9.0 有 8 个符号消失、
  38 个签名变化、81 个源码变化。结果在 `/pscratch/sd/w/weiliang/g4p190_audit/symdiff/`。
  **这种 diff 看不到 C ABI 的变化**，第 0 节第 2 条那一类问题只能靠读 `.cu` 和实测发现。
* 按领域人工读码，分四块：直接 SCF 的 J/K、密度拟合（含 DF Hessian）、XC/网格/VV10/
  单电子/性质、post-HF/PBC/新增功能。每条结论都对照了上游 v1.9.0、v1.9.2 的行号和
  Electrolites 的行号。§3 中标"读码"的结论没有运行验证。

### 1.3 实测

* 跑了 `tests/` 全套（99 个脚本；`test_full` 分 patched 和 unpatched 两次，共 100 项）。
  每个版本各跑一遍，在一个 interactive 节点上（4×A100-SXM4-40GB，job 61871494、
  61890654），每项超时 25 分钟。
  * 1.8.1 全套约 5 GPU·h，4 卡并行约 75–90 分钟，放得进一次 interactive 分配。
* 1.9.0 上用了测量用垫片（`benchmarks/upgrade_audit/shim_run.py`、`sitecustomize.py`），
  只用于测量。垫片让 `patch()` 逐模块打补丁，打不上的模块记为 `SHIM-SKIP`、其余照打，
  这样才能看到每个模块各自的问题。`sitecustomize.py` 版本同时覆盖测试自己起的子进程。
* 日志：`/pscratch/sd/w/weiliang/g4p190_audit/{181,190s,190site,probe}/`，
  汇总表在 `symdiff/suite_table.txt`。

---

## 2. 实测结果

| 类别 | 数目 | 测试 |
|---|---|---|
| 两边都通过 | 38 | 例如 `test_full --patch ''`、`test_scfscreen`、`test_jk`、`test_j_ndm`、`test_fast1e`、`test_dfj`、`test_dfj3c`、`test_stability`、`test_sap`；`test_ejk_ip2` 也在这里，但只是"空过"：`fastejk2` 在 1.9 上不生效，实际是上游和上游比 |
| 1.9.0 通过，1.8.1 那次没跑完 | 2 | `test_xchess`、`test_xchess_fd`（1.8.1 那次分配到时间先结束了） |
| **1.8.1 通过、1.9.0 失败** | **52** | 见下 |
| 1.8.1 本就失败或超时，1.9.0 出现新的失败原因 | 2 | `test_ccsd`（1.8.1：40 GB 卡 OOM）、`test_rshresp`（1.8.1：超时） |
| 两边都失败（与升级无关） | 6 | `test_cuplan`、`test_planguard`（40 GB 卡 OOM）、`test_nacms`、`test_uxc`、`test_uxchess`（超时）、`run_upstream_int1e_suites.py`（缺 `--src` 参数） |

1.9.0 上的 54 个失败，按根因分组：

| # | 根因 | 表现 | 测试 |
|---|---|---|---|
| R1 | strict 检查：`fastpbcgrad` 的目标消失 | `patch_all()` 整体拒绝 | 所有调用 `patch_all()` 的测试（垫片绕过后才看得到下面各项） |
| R2 | `fastpbcdf` 的源码替换锚点 `store_col_segment(` 不存在了 | import 时 RuntimeError | `test_pbcdf` |
| R3 | `(omega, lr_factor, sr_factor)` 贯穿各 API：上游调用方传入了新参数（`test_dfhess` 是测试脚本自己按旧签名调 `range_coulomb`） | TypeError | `test_df`、`test_dfrsh`、`test_cphf`、`test_dfcderi`、`test_dffp32`、`test_dfhess`、`test_dfhessip`、`test_dfuhessip`、`test_qmmm_hess`、`test_x2c_grad`、`test_tdejk`（11 个） |
| R4 | `eval_ao` 不再接受 `shls_slice`，改为 `screening_info` | NotImplementedError | `test_full --patch all`、`test_grad`、`test_rsh_deriv`、`test_unlc`、`test_uxcgrad`（5 个），另外 `test_qmmm` 的梯度子进程 |
| R5 | `_block_loop` 删掉了 `strict_grid_order` 参数 | TypeError | `test_fnlc`、`test_ufnlc`、`test_vv10ft`、`test_vnlch`、`test_fxc`、`test_fxc_occ`、`test_ufxc`、`test_gridresp`、`test_ir`、`test_pcmbatch`、`test_rshresp`、`test_sfxc`（12 个） |
| R6 | `scf/int4c2e` 重写（#936） | `fastmp2`：`_VHFOpt(mol,'int2e')` 把字符串当成了 tol，TypeError；`fastccsd`：`LMAX_ON_GPU`、`_fill_eri_block` 不存在了；**`fastc6`：ERI 被对称化两次，结果静默错误** | `test_mp2`、`test_adc_gpu`、`test_ccsd_t`、`test_ccsd`、`test_eom_gpu`、`test_eom_more_gpu`、`test_eomee_lambda_gpu`、`test_eomee_props_gpu`、`test_rreom_gpu`、**`test_c6`**（10 个） |
| R7 | **C ABI 变了而 ctypes 调用没改，结果静默错误** | `sum_ejk_int3c2e_ip1`、`RYS_build_jk`、`RYS_per_atom_jk_ip1(_multidm)` 少传三个 double；`fci` 的 `_gather` 参数顺序对调 | **`test_dfgrad`**（ejk 相对误差 13–14）、**`test_dfmp2_grad`**（3e45）、**`test_dhdft`**（1e7–1e8）、**`test_prim_pair`**（TM/def2-TZVP 的 ejk 为 1e-2；def2-TZVPP/QZVP 的 JK 为 0.45/0.74，出现 NaN）、**`test_nevpt2_gpu`**（0.36）、`test_casscf_gpu`（非法访存）（6 个） |
| R8 | `_dfaux._spmult` 断言旧的 `nsp*4` | AssertionError | `test_dflocal` |
| R9 | 版本门让 kernel 不生效 | `fastjkip1`、`fastejk2` 在非 1.8.x 上不生效 | `test_jkip1`（`fastejk2` 的测试仍然"通过"，但实际跑的是上游代码） |
| R10 | PBC 重组 | `fastpbcgrad` 的目标消失；multigrid v3 的 `get_j` 只接受单个密度（`pbc_tdscf` 的批量 J 会触发断言）；1.9.0 要求 k 网格包含 Γ 点（`test_fftk` 的平移网格算例） | `test_pbcgrad`、`test_pbc_tdscf`、`test_pbc_ktda`、`test_fftk`（4 个，加上 R2 的 `test_pbcdf`） |
| R11 | **上游 1.9.0 自身的问题**（Electrolites 没打补丁的路径） | GHF TDHF 激发能偏 0.5 eV（未打补丁的 1.9.2 上通过；原因未定位，`ghf_td_gpu` 不走 `mf.get_jk`，探针也显示 hermi=0 的 J/K 正确，推测与 #965/#1004 的 GHF 修复有关）；stock GKS mcol 能量与 1.8.1 差 1.5e-2 Eh，而 `fastmcfun` 复现出 1.8.1 的值（尚未在 1.9.2 上复核）；#914 的 `_e_nuc_ext` 缓存在 MM 电荷移动后不失效，使测试里的参考值过期（1.9.0–1.9.2 都有，可以报给上游） | `test_ghf_td_gpu`、`test_mcfun`、`test_qmmm` |

另外用一个探针确认（`probe/k*.txt`）：上游 `SCF.get_j/get_k` 在 hermi=0、多密度时，
1.8.1、1.9.0、1.9.2 三个版本都与 PySCF 一致，误差在 1e-13 以内。所以 R7 的问题
在 Electrolites 的调用一侧，不在上游。

---

## 3. 逐模块结论

图例：

* ✅ 可以直接用；
* 🔧 需要移植（会报错）；
* ⚠️ 需要移植并修正（会静默出错）；
* ♻️ 上游已覆盖，退役或改成策略开关；
* 💤 被版本门关掉，不生效；
* 尺寸：S 小于 1 天，M 1–3 天，L 3 天以上。

依据前标【实测】的，由 §2 的测试确认；其余为读码结论。

### 3.1 直接 SCF 的 J/K 及其导数

| 模块 | 状态 | 依据 | 动作 | 尺寸 |
|---|---|---|---|---|
| `compat` | ✅ | `_VHFOpt.build` 和 `bas_pair_cache` 的结构未变；`dense_q_cond`、`sorted_meta` 仍然有效 | 新增 C ABI 能力探测，例如 `'lr_factor' in signature(grad.rhf._jk_energy_per_atom)`；新增 helper，返回 `(omega, lr, sr)` 三个 `c_double` | S |
| `fastk` | ✅（RSH 需重测） | 交还上游的 `RYS_build_k` 在 1.8.1 就带三个 double；【实测】test_prim_pair 中"K vs ref"仍为 1e-14 | 只需重测 RSH | — |
| `fastj` | ✅ | `j_engine._VHFOpt` 在 1.8.1→1.9.2 间逐字节相同；`MD_build_j` 未变 | — | — |
| `fastjk` | ⚠️ | 交还上游的 `_run_ref`（fastjk.py:252）调 `RYS_build_jk` 时少传三个 double；凡是带 g 壳层的类都会被交还（70 个类中没有 l=4 的）。【实测】TM/TZVPP 的 JK 误差 0.45，QZVP 为 0.74 | 在能力门后传入 `c_double(0),(1),(1)` | S |
| `fastejk` | 🔧⚠️ | (a) 上游按位置传 8 个参数，替换函数只接 5 个，TypeError【实测】；(b) `_run_ref_new` 少传三个 double，f/g 类交还时静默出错【实测：TM/TZVP 的 ejk 为 1e-2】 | 补上签名和断言，修正 ABI 调用 | S |
| `fastgrad` | 🔧（只受 `fastejk` 牵连） | 抄写的 `energy_ee` 与上游相同，源码变换的锚点都还能匹配；1.9.x 的 `omega=None` 导致每次梯度多建一个 `_VHFOpt` | 显式传 `omega=0` | S |
| `fasttdejk` | 🔧⚠️ | 签名不符，TypeError【实测】；回退路径按位置把 `sum_results` 传到了 omega 的位置；`_stock_group` 少传三个 double；工作区仍按 `DD_CACHE_MAX` 估算，而上游 #879 已改为按 scheme 估算 | 修签名，回退改用关键字传参，修 ABI，工作区按 scheme 估算 | S–M |
| `fastejk2` | 💤 + 需要重新生成 | launch-scheme 那一半仍在工作；kernel 那一半在 1.9.x 上关闭。上游 ip2 kernel 加了三个 double、#909 屏蔽、#1024 的 `nsq_per_block = blockDim.x` | 用 `codegen/gen_ejk_ip2.py` 从 1.9.2 重新生成，解除版本门，重测 | M |
| `fastjkip1` | 💤 + 需要重新生成 | `RYS_build_jk_ip1` 加了三个 double；只解除版本门的话，`cutoff` 会读成 0，静默出错 | 用 `codegen/gen_jk_ip1.py` 从 1.9.2 重新生成 | M |
| `fastrshresp` | 1.9.0 ✅，1.9.1+ 🔧 | 1.9.2 中 RHF 的 `hermi==2` 分支多了长程项，正则的命中数从 2/1 变成 2/3，于是 `_transform` 抛异常，并让 `patch_all` 整体中止 | 放宽对命中次数的断言（新增的位置模式相同，改写是正确的） | S |
| `fastrsh` | ✅（1.8.x 起退役） | — | — | — |
| `fastscfscreen` | ✅【实测】 | `get_veff` 签名未变 | 与 #978 GDM 一起用时需验证 | — |
| `giao_jk`、`warmup` | ✅ | `warmup` 仍会编译在 1.9.x 上不生效的 `libmyejk2`，白花约 25 s | 门控关闭时跳过 | S |

### 3.2 密度拟合

1.9.0 的默认 `range_separated_mode='mix_outside_kernel'` 下，SCF、梯度和 Hessian 都**只传
`None`**；只有 `mix_inside_kernel`（可选打开，用 cuEST 时强制）才会传非 `None` 的因子。
但"接收参数后直接丢掉"并不安全：第一次调用会建成纯 ω 的张量，第二次就会撞上
`_rsh_parameters` 的断言。所以每个替换函数要么把因子转发下去，要么在因子非 `None`
时交还上游。

| 模块 | 状态 | 依据 | 动作 | 尺寸 |
|---|---|---|---|---|
| `fastdf` | 🔧 | `get_jk` 没有 lr/sr 参数，而 1.9.0 的 `DF.get_jk`（df.py:183）总是传入这两个参数，所以每次 cderi 的 J/K 都 TypeError【实测 test_df/test_dfrsh/test_cphf，调用栈 `_DFHF.get_jk`→`DF.get_jk`→fastdf】；`_blocks` 的主机预取缺 #1036 的 event 等待 | 接收 lr/sr，非 `None` 时交还；补 `record`/`wait_event` | S |
| `fastdfj` | 🔧 | 默认模式下自身签名没有问题（实测调用栈里出现的 TypeError 来自它下游的 `fastdf`）；`mix_inside_kernel` 下 TypeError（读码）；它修的 RIJONX `only_dfj` 那一行，上游已在 **1.9.1（#1004）修复** | 转发 lr/sr；那项修复只对 1.9.1 之前的版本生效 | S |
| `fastdfj3c` | ⚠️ | 丢掉了 #864：`CT_dot_mat` 的 grid.z 上限 65535；**1.9.1+**：Cholesky 失败时存为 `(j2c, None)`，回退到上游 `get_j` 后会被当成特征分解因子，J 静默出错 | grid.z 分块；采用上游的 `'eig'` 回退和标签 | S |
| `fastdfcderi` | 🔧 | `_cholesky_eri` 的签名，每次 DF build 都 TypeError【实测 test_dfcderi】；`_create_pair_recontraction` 改名为 `_create_pair_recontractor`，cc-pVXZ 这类一般收缩基会 AttributeError；`_spmult` 断言失败 | 补签名、加改名垫片；`sp_mult` 退役（§5） | M |
| `_dfaux` | 🔧 | `_spmult` 断言 `ref*4`，而 1.9.0 是 `min(4n, max(n//4,16))`【实测 test_dflocal】 | 删除，或按新粒度重定基线 | S |
| `_dfj2c` | 🔧 | `int2c2e` 没有 lr/sr 参数，TypeError | 转发 | S |
| `_dfstore` | ⚠️ | `transpose_write` 第一个参数改成了 stream（decompress.cu:116），旧调用的参数整体错位 | 把 stream 作为第一个参数传入 | S |
| `fastdfgrad` | ⚠️ | `sum_ejk_int3c2e_ip1` 在 `n_dm` 后插入了三个 double，调用没改，**ejk 静默偏 13 倍**【实测】；`_pick_df` 的缓存键变了，长程张量复用失效（只影响速度）；无 cderi 的路径仍用 #1019 已删掉的显式形式 | 在能力门后传 `(0,1,1)` 或 `(ω,1,0)`；改用新的键；无 cderi 路径改用 `_gen_metric_solver` | S |
| `dfmp2_grad` | ⚠️ | 同一个 kernel，**3e45**【实测】；`dhdft` 和 `casscf_grad` 也受牵连 | 传 `(0,1,1)` | S |
| `fastdffp32` | 🔧 | `_get_jk` 包装是无条件安装的，8 个参数，而上游 `_get_jk_mo` 会传 10 个；fp32 的 `loop` 缺 #1036 | 补签名，补 event 等待 | S |
| `fastdfhess` | 🔧⚠️ | `_get_jk` 只接 8 个参数，而上游 `_get_jk_mo` 传 10 个，CPHF 一步就 TypeError（读码；test_dfhess 实测失败的直接原因是测试脚本自己用了旧的 `range_coulomb(omega)`，还没走到模块）；`transpose_write` 旧调用不检查返回码，cderi 落到主机后得到未初始化的缓冲区 | 转发 lr/sr；`transpose_write` 照抄上游（stream、同步、返回码） | S |
| `fastdfhessip`、`_dfhessip_u` | 🔧 | `_spmult` 断言；`mix_inside_kernel` 下 TypeError；拒绝路径按位置传 `omega, verbose`，TypeError【实测】 | 修签名，改用关键字传参，`_spmult` 改为新公式；最省事的做法是因子非 `None` 时直接拒绝 | S（拒绝）/ M（完整支持） |
| `fastcphf` | ✅（缺 #951） | 上游现在传 `lindep=max(tol**2, eps)` | 补上 | S |
| `fastdfrsh` | ♻️ | #859 的 `mix_inside_kernel` 就是同一个算法，而且上游梯度和 Hessian 会跟着这个模式走，fastdfrsh 原来的主要缺陷因此消失；但上游默认仍是两张张量 | 改写为策略层：按 `auto`/`fuse` 设置 `range_separated_mode`，前提是 fastdf、fastdfcderi、_dfj2c、fastdfgrad 都能处理 lr/sr | M |
| `dflocal` | 🔧（需显式启用） | `get_jk` 签名；`_spmult` | 接收 lr/sr、拒绝非 `None`；去掉 `_spmult` | S |
| `fastdfmp2`、`planguard`、`_cuplan`、`_dfpin`、`_dffp32`、`_dfblock`、`_j3c*`、`_ksplit`、`_h1d2`、`_ip2launch`/`myip2` | ✅ | 用到的上游符号未变；`myip2` 的回退调用 1.9.0 把 double 挪到了 `envs*` 之前，在 x86-64 上无害，但应该改过来 | （可选）调整参数顺序 | S |

### 3.3 XC、网格、VV10、单电子、性质

| 模块 | 状态 | 依据 | 动作 | 尺寸 |
|---|---|---|---|---|
| `fastxcgrad` | 🔧 | `eval_ao(shls_slice=...)` 在 1.9.0 上 NotImplementedError【实测】；**1.8.1 上就有的潜在 bug**：跳过空 AO 块后，VV10 梯度的 rho 拼接与 `_vv10nlc` 的 `(4, ngrids)` 断言对不上 | 改为 `screening_info=(nao_sub, non0shl_idx, ao_loc_slice, ctr_offsets)`（注意顺序与 `non0ao_idx` 不同）；空块照样产出 | S |
| `fastxchess` | 🔧 | `block_loop` 现在会产出空块，`Layout.block_atoms` 却跳过它们，于是迭代器错位、断言失败；空块的 launch 网格为 0 | 保留空项，在循环里跳过 | S |
| `fastxcfxc` | 🔧 | `strict_grid_order` 引发 TypeError【实测】；`_get_veff_resp_mo_u` 抄的是 1.8.1 版本，没有 `range_separated_mode` 分支，`omega > 1e-10` 也应改为 `!= 0`（**负 ω、纯短程的交换响应会算错**） | 去掉这个参数；同步 1.9 的杂化块 | S |
| `fastfnlc`、`fastvnlch` | 🔧 | `strict_grid_order`【实测】 | 去掉这个参数（1.9.x 的行为就是原来的 strict 模式；1.9.2 之后的 master 又以 no-op 形式加回来了，所以不传在所有版本上都成立） | S |
| `fastxc`、`fastxcnlc`、`fastnlc`、`fastvv10ft`、`fastgridresp`、`fastsfxc`、`fastmcfun`、`fast1e`、`fastgradh`、`fastpcmbatch`、`fastnacms`、`fastnmr`、`nmr_uks`、`ir_uks`、`raman_uks`、`polar_gpu`、`cubegen_gpu`、`ssc_gpu`、`qmmm_hess` | ✅ | `vv10.cu` 逐字节相同；`int3c1e`、`get_hcore`、`tdscf/rhf.py` 均未变；`fastgridresp` 的源码改写在 1.9.0、1.9.2 上都能匹配，并且顺带继承了 #929；`fastxcnlc` 在 1.9.x 上覆盖面更大 | `fastxc` 的 perblock 路径要测空块 | — |
| `fastqmmm` | ✅（能量那一半部分 ♻️） | 【实测】test_qmmm 的能量差，是**测试里的参考值过期**：上游 #914 的缓存在 MM 移动后不失效。fastqmmm 的缓存键包含坐标和电荷，结果是对的 | 测试中先 `mf.reset()` 再取参考值；重测能量部分的加速，力的缓存仍是 Electrolites 独有 | S |
| `fastir` | 🔧 | 抄的 1.8.1 驱动；1.9.0 对 `with_solvent` 抛 NotImplementedError，fastir 却**静默返回 PCM 下的强度** | 照搬上游的守卫，或者验证复用 PCM Hessian 的 `mo1` 是否正确 | S |
| `fastc6` | ⚠️ | `get_int4c2e` 现在返回完整 ERI，fastc6 又对称化一次，**C6 偏 60 %**【实测】；HF 恰好"通过"，推断是因为 4 倍的交换项让 A−B 不定，触发回退到 CPU（与计数器 `stock_eigh: 1` 吻合） | ≥1.9 上去掉两行 `+=`；要求 ≥1.9.1（#995 的未初始化积分）；内存门放宽到约 3.5·nao⁴；RSH 的排除可以重新评估 | S |

### 3.4 post-HF（四中心积分驱动）

| 模块 | 状态 | 依据 | 动作 | 尺寸 |
|---|---|---|---|---|
| `fastmp2` | 🔧 | `_VHFOpt` 改成了 `jk._VHFOpt(mol, direct_scf_tol, tile)`【实测】；`loop_int4c2e_general` 已弃用；上游 `_ao2mo_general` 已经是"保留最小一对作中间量"的四指标变换，fastmp2 当初要躲开的那次逐块 O(o²v²) 写入已经不存在了 | 改写为在 `_ao2mo_general` 外加占据轨道分批和主机溢出；先与上游比，赢了才保留 | S–M |
| `fastccsd` | 🔧 | `LMAX_ON_GPU`、`_fill_eri_block`、`vhfopt.l_ctr_offsets/log_qs/uniq_l_ctr` 都不存在了【实测】；上游 CCSD 改为按角动量类对调 `RYS_fill_int4c2e`，不分组也不缓存；但 `update_amps` 仍在主机上循环，所以 fastccsd 的设备端循环仍然有价值 | 写一个共用的 AO 块引擎：`SortedMole.from_mol(group_size=g)` + `bas_pair_cache` + `RYS_fill_int4c2e`，每个块先清零（被屏蔽的四元组不写入） | M |
| `eom_gpu`、`rreom_gpu`、`ccsd_lambda_gpu`、`ccsd_rdm_gpu`、`ccsd_t`、`adc_gpu` | 🔧（间接） | 依赖 `fastmp2.get_int4c2e_ovpq` 和 `eom_gpu._ladder*` | 改为调用共用引擎 | 包含在上面的 M 里 |
| `fci_gpu` | ⚠️ | `_gather` 从 `(out, t1, …)` 改成了 `(t1, out, …)`，**sigma 为 0 或非法访存**【实测】；`casscf*`、`sacasscf_grad`、`mcpdft_gpu`、`nevpt2_gpu` 都受牵连 | 先对调参数；以后改调上游的 `contract_2e`，只保留 trans-RDM、`contract_ss`、hdiag 这些上游没有的部分 | S / M |
| `mp2_rdm`、`rpa_gpu`、`gw_gpu` | ✅ | — | — | — |
| `dhdft` | 能量 ✅，梯度 ⚠️ | 通过 `dfmp2_grad` 受牵连【实测】 | 随 `dfmp2_grad` 一起修 | — |

### 3.5 PBC

| 模块 | 状态 | 依据 | 动作 | 尺寸 |
|---|---|---|---|---|
| `fastpbcgrad` | 🔧 | #910 统一了梯度和应力，`krhf.grad_elec`、`rks_stress` 都没有了【实测】；新的目标是 `pbc/grad/krhf.py:get_nuc_fftdf`（multigrid v3 下会被绕过）；上游新增了解析的 ppnl 导数（#912） | J7 重新定目标；J5 计一次时后删除 | S |
| `fastpbcdf` | 🔧 | #924 用 `cderi_k.get(out=host)` 加主机 OpenMP recontract 替换了 `store_col_segment`【实测：import 失败】；上游仍在映射的主机内存里建 cderi，**没有被覆盖** | 改写成设备端的 CSR scatter-add（包装 `_create_pair_recontractor`） | M |
| `pbc_tdscf` | 🔧 | multigrid v3（已是默认）的 `get_j` 断言只能有一个密度【实测】；GHF 参考没有检查 | 逐个密度调用 J，或改走 FFTDF/GDF 的 J；加 GHF 守卫 | S |
| `fastfftk` | ✅（测试 🔧） | `get_k_kpts` 未变；1.9.0 的 SCF 要求 k 网格包含 Γ 点，测试里平移网格那个算例要改【实测】 | 平移网格的算例改为直接调用 `get_k_kpts` | S |
| `kmp2_gpu` | ✅ | `kk_adapted_iter` 换了新的默认值，产出的元组相同 | 补一个 KRHF 检查 | S |
| `pbc_stability` | 需要决定 | Γ 点的 `pbc.scf.hf.SCF` 现在继承了分子版的 `stability`（#1005），会在晶胞上静默运行 | 加一个 `absent=`；在金刚石上对比 | S |

### 3.6 新增功能、`fixes` 和 `guards`

| 项 | 状态 | 依据 | 动作 |
|---|---|---|---|
| `sap` | ♻️ | 1.9.1 的 #997 提供了 `SCF.init_guess_by_sap` 和 vSAP | 退役；把 `geneig` 移到别处 |
| `stability` | 需要决定，建议改成补丁 | #1005 覆盖了 RHF/RKS 的内、外部稳定性，UHF 的内部稳定性，以及 HF 的 UHF→GHF 外部稳定性；ROHF 仍是 `NotImplemented`，UKS 外部也没有；上游还带了 `gpu4pyscf.scf.stability`，`register_gpu_module` 会把它遮住 | 只把 ROHF 的槽位标为 absent；UKS 外部稳定性和批处理用 `replace_broken` 补上 |
| `casscf*`、`sacasscf_grad`、`mcpdft_gpu`、`nevpt2_gpu` | 保留，作为超集 | #869 只有 `gpu4pyscf.mcscf.DFCASCI/DFCASSCF`：仅 RHF、仅 DF、对角 Hessian，没有态平均、梯度和冻结轨道，也不占 `SCF.CASSCF` 这个槽位 | 修好 `fci_gpu` 和 `dfmp2_grad` 后重测 |
| `rohf_grad`、`lo_gpu`、`mp2_rdm` | ✅ | 槽位仍空 | — |
| `fixes` D1（vSAP） | ♻️，并且会让目标检查失败 | #997 删掉了 `UKS.get_vasp`（1.9.1） | 删除这一项，并从 `_TARGETS` 中拿掉 |
| `fixes` D2（ROHF 初猜） | 在 ≥1.9.0 上为 no-op | #975 | 退役 |
| `fixes` D14（X2C eigh） | 在 ≥1.9.1 上为 no-op | #1004 捕获了 `CuSolverError` | 只对 1.9.0 生效，或者退役 |
| `fixes` D3、D4、D7、D10，V3 | 仍然需要 | 上游对应代码未变 | 保留；`fixes` 要按条拆分目标，以免一条缺失拖垮整个模块 |
| `fixes` D8、D9 | 上游已在处理 | #937 的转换，#942/#965 的 PBC GHF | 重新探测 |
| `guards` | 仍然需要 | ROHF 仍然继承 `RHF.Hessian`、`RHF.TDA`；`_DFHF.Gradients/Hessian` 把 ROHF 分派到 RHF 的类；QMMMSCF 没有 Hessian | 保留 |

---

## 4. 与 `docs/UPSTREAM_BUGS.md` 对照：上游修了哪些

| 条目 | 1.9.x 的状态 | 对 Electrolites 的影响 |
|---|---|---|
| A1 原地 cuTENSOR 收缩 | **1.9.0 修复**（#877） | `fastxchess` 的别名守卫变得多余，但无害 |
| B1 RIJONX 的 `only_dfj` | **1.9.1 修复**（#1004） | `fastdfj` 的修复只对 1.9.1 之前的版本生效 |
| B6 `multi_gpu.py:99` | **1.9.1 修复**（#1004） | 不能再作为上游 PR 提了 |
| §9 GHF 批量 J/K | **1.9.1 修复**（#1004；1.9.2 的 `scf/ghf.py` 实数分支改为按 `reshape(4, n_dm, …)` 读回） | 在 ≥1.9.1 上，`ghf_td_gpu._vresp` 的绕行可以改回直接用 `mf.get_jk` |
| §9 sfX2C1e | **1.9.1 修复** | D14 变成 no-op |
| §9 冻结核 CCSD | 1.9.0 的 #961 声称已修复 | 需复核后再决定 D 系列的去留 |
| B4 grid-response `d2w` 的 int32 断言 | **仍然存在**（1.9.2 `hessian/rks.py:2011`） | — |
| C3 Krylov 预留 0.4 倍显存 | **仍然存在**（`df/hessian/rhf.py:1326`） | `fastdfhess` 的修复仍有意义 |
| E2 `empty_mapped` 用 `hostAlloc` | **仍然存在** | `fastdfcderi` 的锁页池仍有意义 |
| E3 IR 重复求解 CPHF | **仍然存在**（"TODO: compact with hessian"） | `fastir` 仍有意义 |
| E4 DF 梯度中的 `with_df.reset()` | **仍然存在** | `fastdfgrad` 的 cderi 复用仍有意义 |
| 新发现：#914 的 `_e_nuc_ext` 在 MM 电荷移动后不失效 | 1.9.0–1.9.2 都有 | 可以报给上游 |

**审计中顺带发现、在 1.8.1 上就存在的 Electrolites 自身问题**（与升级无关，建议先修）：

1. `myjkip1/unrolled_rys_jk_ip1.cu` 在写 `aij_cache` 之后缺 `__syncthreads()`，约 37 处，
   与上游 #1024 修的是同一个问题。直接 SCF Hessian 的 h1 项在 1.8.1 上就会走到这里（读码，未复现）。
2. `myejk2` 的 8 个 kernel 写死了 `constexpr nsq_per_block = 256`，而 `adjust_threads`
   可能把 block 加倍到 512。这有可能解释 `fastejk2.py` 文档中那句至今没有解释的
   "`__launch_bounds__(512,1)` 误差 9–37 倍"（推断，未证实）。
3. `fastxcgrad` 的 VV10 梯度遇到空 AO 块时会失败（读码）。
4. `fastir` 在 PCM 下给出的强度是否正确，任何版本上都没有验证过。
5. `fastxcfxc` 中 UKS Hessian 的 RSH 分支，对负 ω（纯短程）会算错（读码）。

---

## 5. 可以借鉴上游的地方（按收益排序）

1. **单遍的 lr/sr 内核与 `range_separated_mode`**（#859）。
   * `fill_int3c2e` 一次调用就生成 `lr·erf + sr·erfc`。
   * 用 `range_separated_mode='mix_inside_kernel'` 作为开关，梯度和 Hessian 会自动跟着走，保持一致。
   * 收益：`fastdfrsh` 变成一个策略开关；DF Hessian 的 RSH 交换项（ip2、h1、CPHF）
     大约减半；直接 SCF 的 ip1/ip2 kernel 也接受混合因子，`fastgrad` 的两遍交换
     有机会合成一次调用。
2. **`scf/int4c2e._ao2mo_general` 和 `SortedMole(group_size)` + `bas_pair_cache` + `RYS_fill_int4c2e`**：
   * `fastmp2` 的 `ovpq`、`fastc6` 直接取 (ov|ov)，不再需要 nao⁴ 的张量和 2.2·nao⁴ 的内存门；
   * CCSD、EOM、Λ 共用的块引擎也建在这上面。
3. **`nsq_per_block = blockDim.x`**（#1024）。这样 `__launch_bounds__` 和
   `adjust_threads` 都变得安全，`fastejk2` 放弃的 P3 寄存器上限实验可以重新做。
4. **按真实 scheme 估算工作区**（#879，`grad/tdrhf.py:368-373`），用于 `fasttdejk`。
5. **DF 方面**：
   * 一次 launch 处理多个辅助基组（`splits_by_blocksize`），用于 `fastdfgrad`；
   * `_gen_metric_solver`（#1019），用于无 cderi 的路径；
   * 特征值过滤的度量回退，用于 `fastdfj3c`；
   * 按 stream 排序的 `transpose_write`；
   * `_split_l_ctr_groups(..., [16]*(laux+1))`，替换各处私有的拆分。
6. **网格方面**：`add_sparse` 替换 `numpy.ix_ +=`（`fastvnlch.py:456`）；
   `eval_xc_eff(..., inplace=True)` 每次 libxc 调用省一次分配。
7. **功能方面**：
   * 上游 `fci/direct_spin1` 的设备端 Davidson 和 `make_rdm12`，其中 Davidson 按空闲显存估算大小；
   * #937 的 SCF 转换，替换 `stability.py` 里手写的 UHF→GHF；
   * cuSOLVER 的 `devInfo` 检查；
   * `lib.utils.to_cpu(method, out=…)`，用于 D3。

---

## 6. 工作计划

验收的总原则：

* 正确性：每个模块都以**同一版本、未打补丁的进程**作参照，容差沿用 1.8.1 的 suite。
* 性能：在两个尺度上测（`PfPMT` 284 原子，`HcgC` 545 原子），并且至少测一个开壳层
  过渡金属配合物（UKS，基组到 f/g）。
* 1.8.1 的回归测试要同时通过。
* 每次测量都设计成能在一次 interactive 分配内跑完。

### Phase 0：基础设施与止血（1–2 天）

1. **显式版本门**：遇到未测试过的 GPU4PySCF 版本，`patch_all()` 默认拒绝，除非设了
   `ELECTROLITES_ALLOW_UNTESTED=1`。这一项立即就做，用来替代目前"碰巧安全"的状态。
2. `patch()` 改为逐模块处理：目标缺失或 import 失败只跳过该模块并发出警告；只有显式
   点名的模块才报错。`fixes` 按条拆分目标。`check_absent` 改为逐槽位判断。
3. `compat`：
   * 能力探测：`RSH_ABI`、`EVAL_AO_SCREENING_INFO`、`BLOCK_LOOP_STRICT`、`INT4C2E_V2`；
   * 一个 helper，返回 `(omega, lr, sr)` 三个 `c_double`。
4. **ABI 守卫**：把所有通过 ctypes 按位置调用的上游 C 符号登记成一张表，至少包括
   `RYS_build_jk`、`RYS_per_atom_jk_ip1(_multidm)`、`RYS_per_atom_jk_ip2_*`、
   `RYS_build_jk_ip1`、`RYS_build_k`、`MD_build_j`、`sum_ejk_int3c2e_ip1`、
   `RYS_fill_int4c2e`、`transpose_write`、`_gather`。
   * 再写一个 `tests/test_abi.py`：每个符号跑一次"Electrolites 的交还调用 vs 上游
     自己的调用"，在很小的分子上逐元素比较，并且必须包含 f/g 壳层的 TM 算例。
     **这是防止静默 ABI 错误的唯一手段。**
5. 测试框架：把 `benchmarks/upgrade_audit/` 的双版本运行方式固化下来，即用
   `PYTHONPATH` 前缀切换版本，4 卡并行，按根因归类。在 1.9.2 上复核 R11 的 GKS mcol 差异。

### Phase 1：消除静默错误（2–3 天，全部 S）

`fastdfgrad`、`dfmp2_grad`、`fastjk`、`fastejk`、`fasttdejk` 的 ABI；
`fastc6` 的双重对称化；`fci_gpu` 的 `_gather`；`fastdfj3c` 的 grid.z 和 `'eig'` 标签；
`fastdfhess`、`_dfstore` 的 `transpose_write`；`fastir` 的 PCM 守卫。

**验收**：`test_dfgrad`、`test_dfmp2_grad`、`test_dhdft`、`test_prim_pair`、`test_c6`、
`test_casscf_gpu`、`test_nevpt2_gpu`、新的 `test_abi`，在 1.9.2 上达到 1.8.1 的容差。

### Phase 2：消除报错（5–7 天）

* DF 的 lr/sr 签名与转发：`fastdf`、`fastdfj`、`fastdfcderi`、`_dfj2c`、`fastdffp32`、
  `fastdfhess`、`fastdfhessip`、`_dfhessip_u`、`dflocal`、`_FusedDF`；先做"非 `None`
  就交还"，完整转发放到 Phase 6。
* `_spmult`；`_create_pair_recontractor` 改名垫片。
* `eval_ao` 改用 `screening_info`；去掉 `strict_grid_order`；`fastxchess` 处理空块；
  修 `fastxcgrad` 的空块 VV10 问题。
* `fastrshresp` 的命中计数。
* `fastpbcgrad` 重新定目标；`fastpbcdf` 改为设备端 scatter（M）；`pbc_tdscf` 逐个密度做 J；
  修 `test_fftk` 的平移网格算例。
* 上游修复的补齐：#1036 的 event 等待（`fastdf._blocks`、`fastdffp32.loop`）；
  #951 的 `lindep`；#992 的 `blksize`。
* `fastxcfxc` 同步 1.9 的 RSH 分支。
* 测试脚本跟进：`test_dfhess` 改用 `range_coulomb(omega, lr, sr)`；`test_fftk` 的平移网格算例；
  `test_qmmm` 取参考值前先 `reset()`；`run_upstream_int1e_suites.py` 在全套运行时补上 `--src`。
* `fixes`、`sap`、`stability` 的槽位调整（与 Phase 5 合并做）。

**验收**：`tests/` 全套在 1.9.2 上的通过集合等于 1.8.1 的通过集合；1.8.1 上无回归。

### Phase 3：四中心积分引擎（4–6 天，M）

共用 AO 块引擎，服务 `fastccsd`、`eom_gpu._ladder*`、Λ、RDM、(T)、EOM 系列；
`fastmp2` 改写在 `_ao2mo_general` 之上。

**验收**：CCSD/EOM/RR-EOM 的现有测试，加上 TeraChem RR-EOM 基准中的一个中等算例
（`$PSCRATCH/eomcc_bench`），能量误差 ≤1e-10 Eh。

### Phase 4：重新启用不生效的 kernel（3–5 天，M）

从 1.9.2 重新生成 `myejk2`、`myjkip1`（ABI、#909 屏蔽、#1024 同步、`nsq_per_block`
取自 blockDim）；版本门改为能力门。**#1024 的同步修复同时回补到 1.8.1 的构建。**

**验收**：`test_ejk_ip2`、`test_jkip1`；直接 SCF Hessian 在两个尺度上重测加速。

### Phase 5：与上游功能重叠的部分（1–2 天）

`sap` 退役；`stability` 只填 ROHF 槽位，用 `replace_broken` 补 UKS 外部稳定性；
`pbc_stability` 加 `absent`；D1、D2、D14 退役或加门；`casscf` 作为超集保留，并做一次
与上游 `DFCASSCF` 的对照测试。

### Phase 6：重新评估被上游追上的优化（3–5 天）

逐项测量。只有在两个尺度上、并且在 TM 配合物上都还赢，才保留，否则退役：

* `fastdfrsh` 对 `mix_inside_kernel`（注意要和**这个模式**比，而不是和默认模式比）；
* `fastdfhessip` W31、W29 对 #934、#970；
* `fastdfcderi` 的 `sp_mult` 对 #970；
* `fastqmmm` 的能量部分对 #914；
* `fastmp2` 对 `_ao2mo_general`；
* `fastk`、`fastjk`、`fastrshresp` 的 RSH 对 #909（上游现在只要 `lr_factor != 0` 就用
  全程屏蔽，基线变慢但更准；1.8.1 时 RSH 的参考值本身就屏蔽不足，精度数字也要重验）。

### Phase 7：重新测量与文档（4–6 天，以 GPU 时间为主）

* `README.md` 里所有加速比都在 1.9.2 上重测，并标明版本：
  * `PfPMT`、`HcgC` 固定密度逐项计时；
  * DF 梯度和 Hessian；
  * 响应性质；
  * 开壳层 TM 那一节。
* 更新 `TESTED_GPU4PYSCF`、`NOTICE`（vendored 源码换成 1.9.2）、`pyproject.toml` 的注释。
* 写结果文档 `docs/UPGRADE_1.9.md`。
* 把 #914 的缓存问题、B4、C3、E2、E3、E4 整理后提交上游。

### 不在本计划内，但会受影响

未合并的分支 `dlpno-gpu`、`dlpno-analytic-gradient`、`dlpno-gradient-multigpu-20261104`、
`w39-41` 依赖 DF 积分接口（`int3c2e_bdiv`、`_cholesky_eri` 等）。`main` 完成 Phase 2 后，
要用同一套 `compat` 再审一遍。

### 总量

| Phase | 天数 |
|---|---|
| 0 | 1–2 |
| 1 | 2–3 |
| 2 | 5–7 |
| 3 | 4–6 |
| 4 | 3–5 |
| 5 | 1–2 |
| 6 | 3–5 |
| 7 | 4–6 |
| **合计** | **23–36** |

Phase 0 和 Phase 1 不依赖其他阶段，可以先做。Phase 3、Phase 4 与 Phase 2 可以并行。

---

## 7. 复现

```bash
# 并排安装（不碰 barw-dev）
G=/pscratch/sd/w/weiliang/g4pver
pip install --no-deps --target $G/1.9.2 gpu4pyscf-cuda12x==1.9.2
pip install --no-deps --target $G/_libxc090 gpu4pyscf-libxc-cuda12x==0.9.0
export PYTHONPATH=$G/1.9.2:$G/_libxc090        # 切到 1.9.2

# 目标 / 槽位
python -c "import electrolites as e; print(e.check_targets()); print(e.check_absent())"

# 符号级 diff
cd benchmarks/upgrade_audit
python extract_refs.py ../../electrolites refs.json
python dump_syms.py refs.json d181.json                                # 默认环境 1.8.1
PYTHONPATH=$G/1.9.2:$G/_libxc090 python dump_syms.py refs.json d192.json
python cmp_syms.py d181.json d192.json st.json > cmp192.txt

# 1.9.x 上逐模块测量（只用于测量）
PYTHONPATH=$PWD/benchmarks/upgrade_audit:$G/1.9.2:$G/_libxc090 python tests/test_df.py
#   stderr 中的 SHIM-SKIP 行列出打不上的模块
```
