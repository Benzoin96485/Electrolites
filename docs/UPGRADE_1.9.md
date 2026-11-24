# GPU4PySCF 1.8.1 → 1.9.2：移植记录

分支 `port/gpu4pyscf-1.9.2`。计划与审计见 `docs/UPGRADE_1.9_PLAN.md`；本文记录实际做了什么、
怎么验证的、还剩什么。`barw-dev` 仍是 1.8.1，1.9.2 只装在旁路前缀里
（`PYTHONPATH=$PSCRATCH/g4pver/1.9.2:$PSCRATCH/g4pver/_libxc090`）。

---

## 1. 基础设施（Phase 0）

* **显式版本门**：`patch()`/`patch_all()` 只在 `TESTED_GPU4PYSCF` 列出的版本上打补丁，
  其他版本直接拒绝，除非设 `ELECTROLITES_ALLOW_UNTESTED=1`。这样"1.9.x 上不出错"不再依赖
  `fastpbcgrad` 的目标碰巧缺失。
* **逐模块处理**：`patch_all()` 遇到目标缺失或打不上的模块，跳过它并发出警告
  （`electrolites.skipped()` 列出原因）；显式点名的模块（`patch('x')`、`patch_all(only=...)`）
  仍然报错。目标可以标成 `'optional'`（`fixes` 的 D1，`fastpbcgrad` 的两代目标）。
* **`compat` 的 1.9 能力探测**：`RSH_CABI`、`EVAL_AO_SCREENING_INFO`、`BLOCK_LOOP_STRICT`、
  `INT4C2E_FULL`、`TRANSPOSE_WRITE_STREAM`，以及对应的 helper：`rsh_cargs`/`rsh_cdoubles`
  （C kernel 里插入的三个 double）、`eval_ao_screened`、`block_loop_kwargs`、`transpose_write`、
  `rsh_df_key`、`ip2_envs_and_operator`、`bind_tail`/`rsh_kwargs`（同一个替换函数同时接受两代
  调用方的参数表）。

## 2. 静默错误（Phase 1）

| 模块 | 原因 | 修正 |
|---|---|---|
| `fastdfgrad`、`dfmp2_grad` | `sum_ejk_int3c2e_ip1` 在 `n_dm` 后多了 `(omega, lr, sr)` | 传入；`fastdfgrad` 另按 #1019 用右端求解方式施加拟合度量（`_MetricSolve`），并改用 1.9 的 `_rsh_df` 键 |
| `fastjk` | 交还上游的 `RYS_build_jk` 少三个 double（所有 g 壳层类都走这里） | 传入 |
| `fastejk`、`fasttdejk` | 同上（`RYS_per_atom_jk_ip1(_multidm)`），且上游按位置多传参数 | 传入；签名同时接受两代；`fasttdejk` 的工作区改按 scheme 估算（#879） |
| `fastc6` | 1.9 的 `get_int4c2e` 已返回完整对称张量，又被对称化一次 | 1.9 上不再对称化；内存门放宽 |
| `fci_gpu` | `_gather` 的前两个指针参数对调 | 按上游 kernel 源码判断顺序 |
| `fastdfj3c` | 丢了 #864 的 grid.z 分块；1.9.1+ 的 `'eig'` 度量标签 | 补上分块；缓存的度量因子按上游的格式存 |
| `fastdfhess`、`_dfstore` | `transpose_write` 第一个参数改成 stream | 经 `compat.transpose_write`，并检查返回码 |
| `fastir` | 抄的 1.8.1 驱动在 PCM 下静默给出强度 | 与 1.9 一样拒绝 PCM |

## 3. 报错（Phase 2）

* DF 栈的 `lr_factor`/`sr_factor`：`fastdf`、`fastdfj`、`fastdfcderi`、`_dfj2c`、`fastdffp32`、
  `fastdfhess`、`fastdfhessip`、`_dfhessip_u`、`dflocal`、`fastdfrsh._FusedDF`。因子非 `None`
  只出现在 `range_separated_mode='mix_inside_kernel'`，那种张量只有上游的 `build` 能做，所以交还上游；
  1.8 的上游永远收不到它不认识的关键字。
* `_dfaux._spmult` 认得 1.9 的聚合规则 `min(4n, max(n//4, 16))`（1.9 上不再改写它，留待 Phase 6 重测）；
  `_create_pair_recontraction` → `_create_pair_recontractor`。
* `fastdfhessip` 的 ip1 evaluator 跟随 1.9 的多组辅助基批（`splits_by_blocksize`），批内逐组 launch，
  保住按 `lk` 的分块。
* 网格：`fastxcgrad` 改用 `screening_info`，并修了 1.8.1 就有的空块 VV10 梯度问题；
  `fastvnlch`/`fastfnlc`/`fastxcfxc` 不再传 `strict_grid_order`；`fastxchess` 的块列表与 1.9 产出的空块对齐；
  `fastxcfxc` 的 UKS 响应按 1.9 处理 `omega != 0` 与 `mix_inside_kernel`。
* `fastrshresp` 接受 1.9.1+ 的命中数 (2, 3)。
* 上游修复补齐：`fastdf._blocks`、`fastdffp32.loop` 的 #1036 等待；`fastcphf` 跟随 #951 的 `lindep`。
* PBC：`fastpbcgrad` 的 J7 改为替换 1.9 的 `krhf.get_nuc_fftdf`（J5 在 1.9 上由上游的解析 ppnl 导数取代）；
  `pbc_tdscf` 在 multigrid v3 下逐个密度做 J；`fastpbcdf` 移植到 1.9 的 GDF k 点构建器上（§5b）。
* 新增功能：`stability` 只填上游留空的槽位（1.9.1 起只剩 ROHF），不再遮住 `gpu4pyscf.scf.stability`；
  `enable_all()` 遇到上游已提供的功能（`sap`）跳过并警告。`fixes` 的 D1 在 1.9.1+ 上改用上游的
  `get_vsap`，只把初猜里的广义对角化换成 `geneig`（上游走 cuSOLVER `sygvd`，在这些计算节点上失败）。
* 测试脚本跟进：`test_dfhess`、`test_fftk`、`test_qmmm`、`test_pbcgrad`、`test_pbcdf`、
  `test_dfhessip`、`test_dfuhessip`。

## 4. 四中心积分引擎（Phase 3）

1.9 删掉了 1.8 的 `int4c2e._VHFOpt`/`loop_int4c2e_general`/`ccsd_incore._fill_eri_block`。
新模块 `electrolites/_i4blocks.py` 的 `BlockOpt` 在 1.9 的 `RYS_fill_int4c2e` 上重建了 1.8 的接口
（按 `group_size` 切分的形状组、`coeff`、`l_ctr_offsets`、`log_qs`、每个 bra 组对的稠密块）。
`fastccsd`、`eom_gpu`（以及 Λ、RDM、(T)、RR-EOM、ADC）和 `fastmp2` 都改到它上面。

上游 1.9.2 `cc/ccsd_incore.py` 的 `fint` 在非对角组对上先多写了一行 `eri[j, :, i] = eri_tmp`
（用第二组的偏移去索引第一组的轴），再写正确的 `eri[i, :, j]`。第二次写入覆盖不到的元素——
被筛掉的壳层对——留下错值。在按 `group_size` 切分的组上，glycine 的 (ia|jb) 因此偏差 1e-3；
`BlockOpt` 没有抄这一行。`tests/test_i4blocks.py` 拿 PySCF CPU 积分对照块和 `fastmp2`。

**上游自己的 CCSD 也受影响**（本文早先的版本说"上游自己用时没有问题"，那是错的——
当时的 8e-9 就是这个错误）。紧凑分子没有被筛掉的壳层对，所以看不出来；有远距离壳层对时：

| 体系（def2-SVP） | 上游 1.9.2 CCSD | 去掉这一行 |
|---|---|---|
| 两个水分子相距 8 Å | 100 轮不收敛，偏离 PySCF 53 mEh；单次振幅更新相对误差 0.3–7.6 | 1e-14 |
| glycine | 200 轮不收敛，偏离 4.5e-9 Eh；单次振幅更新相对误差 4e-3 | 5.8e-10（42 轮） |

`fixes` 新增 **D15**：在有这一行的版本上，从上游源码重建 `_direct_ovvv_vvvv`，只删掉这一行。
`test_fixes` 用水二聚体上的一次振幅更新对照 PySCF CPU（D15 后 4.8e-12，上游原样 0.32）；
1.8.1 上这一项 2.1e-11，D15 不生效。截至 2026-11-24 上游 master 仍有这一行，值得报给上游。

## 5. Hessian ip1/ip2 kernel（Phase 4）

`codegen/gen_ejk_ip2.py --release 192` 和 `codegen/gen_jk_ip1.py` 从 1.9.2 的 `lib/gvhf-rys`
重新生成了 `fastejk2` 与 `fastjkip1` 的 kernel（`kernels/cuda/gvhf-rys192`、`myejk2_192`、`myjkip1_192`）。
它们带有 1.9 的参数表、#909 的屏蔽和 #1024 的同步。`_nvcc` 按安装的版本构建对应的一套，
两个模块在 1.9.2 上重新生效。

## 5b. GDF k 点 cderi 留在设备上（`fastpbcdf`）

1.9（#924）的 `compressed_cderi_kk` 改成按**原始**壳层对算积分块，每块拷到 pinned 主机内存，
由一个主机线程用 OpenMP 把它重缩并累加进 cderi（`_create_pair_recontractor` →
`PBCzrecontract_cderi`：`cderi[a, out_idx[k]] += coef[k] * j3c[i, a]`）；cderi 本身仍在映射的主机内存里，
每次 J/K 都要隔着 PCIe 读回。1.8 那套 `store_col_segment` 锚点没有了，模块此前在 1.9 上让位。

现在：cderi 放得下时（不超过空闲显存的 ⅓，且加上构建器自己的工作缓冲不超过其预算的 80%）
分配在设备上，同一个重缩映射在设备上执行（`_DeviceRecontract`：按输出列分组的 CSR，
每个输出元素由一个线程写一次，结果确定），弥散壳层对（dd）部分也走它；设备分配在构建器的
工作缓冲尺寸确定之后决定，并从它的显存预算里扣掉。放不下时走上游原路径。1.8 的路径不变。

`test_pbcdf`：cderi 与上游构建差 5.6e-17，J/K 4.5e-16（2×2×2、3×3×3，dd 部分都走了设备路径）；
另外四种晶胞/基组（含 aug-cc-pVDZ 水盒子、LiH 6-31G）≤1.1e-16。其余 PBC 测试在新模块下全部通过。

计时（`benchmarks/gdf_k_probe.py`，PBE0 KRKS + GDF，`patch_all()`，只切换 `FASTPBCDF_OFF`，
单块 A100-40GB 独占）：

| 晶胞，k 网格 | 版本 | cderi 构建 | `get_k_kpts` | SCF |
|---|---|---|---|---|
| diamond8，2×2×2（4.1 GB） | 1.9.2 | 4.42 → **1.84 s** | 0.220 → **0.074 s** | 19.04 → **9.65 s（1.97x）** |
| | 1.8.1 | 6.09 → 4.19 s | 0.203 → 0.069 s | 15.14 → 12.14 s（1.25x） |
| diamond，4×4×4（2.5 GB） | 1.9.2 | 2.52 → **0.90 s** | 0.155 → **0.065 s** | 15.43 → **13.33 s（1.16x）** |
| | 1.8.1 | 2.73 → 1.54 s | 0.150 → 0.068 s | 10.02 → 8.89 s（1.13x） |

1.9 上构建省得更多（diamond8 2.4x，1.8 上 1.45x）：省掉的是映射内存分配、逐块拷回主机和主机端重缩，
三者各占多少没有单独剖析。cderi 与 SCF 能量在开关前后一致（同版本内 SCF 能量逐位相同）。
另：**不打 `fastpbcdf` 时 1.9.2 的 k 点 SCF 比 1.8.1 慢**（diamond 4×4×4 15.4 vs 10.0 s，diamond8 19.0 vs 15.1 s），
差在 cderi 之外，原因尚未剖析（列入 Phase 7）。

## 6. 验证

* 同一分支、同一套 `tests/`（99 个脚本，`test_full` 分 `all`/`none` 两种跑法），每个脚本单独一个进程、
  单独一块 GPU，在 1.9.2（旁路前缀）和 1.8.1（`barw-dev`）上各跑一遍。移植过程中跑了三轮
  （`r1`–`r3`，加上各处的重跑）；**最终回归**在提交 `f69bbcc`（`'1.9.2'` 已在 `TESTED_GPU4PYSCF` 里，
  1.9.2 上**不设** `ELECTROLITES_ALLOW_UNTESTED`，版本门本身也受检）上把全部脚本重跑一遍：
  1.9.2 在一个 A100-40GB 节点上，1.8.1 在一个 A100-80GB 节点上，单脚本限时 1500 s；四个慢脚本
  （`test_rshresp`、`test_ufxc`、`test_uxc`、`test_uxchess`）在 80GB 节点上以 3600 s 限时两个版本各跑一遍。
  之后 `fastpbcdf` 的移植（§5b）与全部 PBC 测试、以及最终回归里失败的脚本，在一个 80GB 节点上
  逐卡独占重跑（`verify_*`）。日志：`$PSCRATCH/g4p192_port/{final_192,final_181,r4_192,r4_181,verify_192,verify_181,…}/`。
* 端到端：`test_full`（全部补丁 vs 不打补丁）在 1.9.2 上 max|ΔE| 2.9e-11、max|ΔG| 8.4e-10，与 1.8.1 上 main 的数一致。
  其余代表性的数：DF 梯度 `ejk` 3e-13、j3c 6e-14–2e-13、`test_ejk_ip2` 6.6e-13、
  CCSD 与上游驱动（D15 后）差 5.3e-12、多态 NAC 与串行差 ≤1e-7（相对）、GDF k 点 cderi 5.6e-17。

因为上游行为变化而改的测试（都不是本仓库的回归）：

| 测试 | 上游变化 | 改法 |
|---|---|---|
| `test_nacms` | #951：`cphf.solve` 显式传 `lindep=max(tol², eps)`，测试改 `krylov` 默认值降不下收敛下限，两条路径又差到 1e-5 | 同时包一层 `cphf.krylov` 降下限。顺带：1.9.2 上串行的基态–激发态 NACV 自身可复现了（glycine 4e-13；1.8.1 上 2.6e-4） |
| `test_mcfun` | 两次 SCF 落到不同的极小 | 打补丁的一次从未打补丁的解重启 |
| `test_features` | 1.9.1 填上了 `RHF.stability` | 示例槽位换成 `ROHF.stability` |
| `test_stability`、`test_sap` | 1.9.1 自带稳定性分析与 SAP/vSAP | 直接调本仓库的驱动，或在上游已提供时跳过 |
| `test_pbc_ktda` | multigrid v3 断言超胞密度为实数 | J 改走 FFTDF |
| `test_qmmm` | #914 把 QM 核–MM 电荷能缓存在 `mf._e_nuc_ext`，MM 电荷移动（`mm_mol.set_geom_`）后不失效 | 参考值在清掉缓存后计算。**这是上游的 bug**：移动电荷后上游自己的 `energy_nuc` 是旧值 |
| `test_fixes` | 1.9.1 自带 vSAP；这些节点上 cuSOLVER `sygvd` 失败 | D1 在 1.9.1+ 只换掉对角化；新增 D15 一项 |
| `test_ccsd` | 上游 CCSD 的 D15 错误 | 参考用的上游驱动带 D15 运行 |
| `test_vv10ft` | 1.8.1 上游的 wB97M-V Hessian 两次运行之间差到 2.3e-6（1.9.2 上 1e-13），固定的 1e-6 门槛过不去 | 门槛取 1e-6 与 3 倍上游自身散布中的大者（与 `test_nacms` 同法） |
| `test_casscf_gpu` | 不是上游变化：SA-4 单个态能量与 PySCF 之差在同一代码的多次运行间为 3e-9–1.2e-7，`conv_tol_grad` 收紧到 1e-7 也降不到 7e-9 以下 | 平均能量（驻定量）仍卡 1e-9，单个态卡 1e-6 |
| 其他 | 见 §3 末尾 | |

## 7. 结果

| | 1.9.2 | 1.8.1（同一分支） |
|---|---|---|
| 全部 99 个脚本 | **全部通过** | **全部通过** |
| 只在 80GB 卡上能跑（40GB 上 OOM，两版本相同） | `test_cuplan`、`test_planguard` | 同左（另有 `test_ccsd` 的 TZVP 一项） |
| 超过 1500 s（3600 s 下通过） | `test_rshresp` 2299 s、`test_ufxc` 1605 s、`test_uxc` 2302 s、`test_uxchess` 1544 s | 2356 s、1633 s、2131 s、1540 s |
| 最终回归中的失败 | 无（除上面的 OOM） | `test_dfuhessip`：与一个探针共用 GPU 0 时 6.5e-7，独占重跑 1.1e-11 通过；`test_vv10ft`、`test_casscf_gpu`：测试门槛问题，见 §6 表 |

结论：1.9.2 的通过集合与 1.8.1 相同，`'1.9.2'` 已加入 `TESTED_GPU4PYSCF`。1.9.0、1.9.1 没有全套验证，
仍需 `ELECTROLITES_ALLOW_UNTESTED=1`。1.9.2 上所有模块都生效（`fastpbcdf` 见 §5b）。

发现的上游问题：
* `cc/ccsd_incore._direct_ovvv_vvvv` 多写一个块 → 有远距离壳层对时 CCSD 错（§4，`fixes` D15）。
  2026-11-24 核对过，上游 master 仍有这一行；
* #914 的 `_e_nuc_ext` 缓存在 MM 电荷移动后不失效（§6 表；未核对 master）；
* 不是上游代码的错，而是本站点：cuSOLVER `sygvd` 在这些计算节点上失败，1.9.1 的 vSAP 初猜因此失败（`fixes` D1）。

## 8. 尚未完成

* Phase 6：重新评估被上游追上的优化（`fastdfrsh` 对 `mix_inside_kernel`、`fastdfhessip` W29/W31
  对 #934/#970、`fastdfcderi` 的 `sp_mult`、`fastqmmm` 的能量部分、`fastmp2` 对 `_ao2mo_general`、
  RSH 对 #909）。
* Phase 7：在 1.9.2 上重测 README 的全部加速比并标明版本；先剖析 1.9.2 上 k 点 SCF 比 1.8.1 慢的部分（§5b）。
