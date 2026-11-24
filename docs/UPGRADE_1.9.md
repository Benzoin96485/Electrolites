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
  `pbc_tdscf` 在 multigrid v3 下逐个密度做 J；`fastpbcdf` 在 1.9 的 GDF 构建器上**暂时让位**（未移植）。
* 新增功能：`stability` 只填上游留空的槽位（1.9.1 起只剩 ROHF），不再遮住 `gpu4pyscf.scf.stability`；
  `enable_all()` 遇到上游已提供的功能（`sap`）跳过并警告。`fixes` 的 D1 只在没有上游 vSAP 的版本上生效。
* 测试脚本跟进：`test_dfhess`、`test_fftk`、`test_qmmm`、`test_pbcgrad`、`test_pbcdf`、
  `test_dfhessip`、`test_dfuhessip`。

## 4. 四中心积分引擎（Phase 3）

1.9 删掉了 1.8 的 `int4c2e._VHFOpt`/`loop_int4c2e_general`/`ccsd_incore._fill_eri_block`。
新模块 `electrolites/_i4blocks.py` 的 `BlockOpt` 在 1.9 的 `RYS_fill_int4c2e` 上重建了 1.8 的接口
（按 `group_size` 切分的形状组、`coeff`、`l_ctr_offsets`、`log_qs`、每个 bra 组对的稠密块）。
`fastccsd`、`eom_gpu`（以及 Λ、RDM、(T)、RR-EOM、ADC）和 `fastmp2` 都改到它上面。

上游 1.9.2 `cc/ccsd_incore.py` 的 `fint` 在非对角组对上还多写了一行 `eri[j, :, i] = ...`。
上游自己用时（组从不切分）没有问题：实测其 CCSD 在 glycine/def2-SVP 上与 PySCF 差 8e-9 Eh。
但在按 `group_size` 切分的组上，这行留下了第二次写入覆盖不到的元素，glycine 的 (ia|jb) 因此偏差 1e-3。
`BlockOpt` 没有抄这一行。`tests/test_i4blocks.py` 拿 PySCF CPU 积分对照块和 `fastmp2`。

## 5. Hessian ip1/ip2 kernel（Phase 4）

`codegen/gen_ejk_ip2.py --release 192` 和 `codegen/gen_jk_ip1.py` 从 1.9.2 的 `lib/gvhf-rys`
重新生成了 `fastejk2` 与 `fastjkip1` 的 kernel（`kernels/cuda/gvhf-rys192`、`myejk2_192`、`myjkip1_192`）。
它们带有 1.9 的参数表、#909 的屏蔽和 #1024 的同步。`_nvcc` 按安装的版本构建对应的一套，
两个模块在 1.9.2 上重新生效。

## 6. 验证

（见 §7 的结果表。）

## 7. 结果

待本轮全套测试完成后填写。

## 8. 尚未完成

* `fastpbcdf`：1.9 的 GDF k 点构建器（#924）上的设备端存储，尚未移植（目前让位给上游）。
* Phase 6：重新评估被上游追上的优化（`fastdfrsh` 对 `mix_inside_kernel`、`fastdfhessip` W29/W31
  对 #934/#970、`fastdfcderi` 的 `sp_mult`、`fastqmmm` 的能量部分、`fastmp2` 对 `_ao2mo_general`、
  RSH 对 #909）。
* Phase 7：在 1.9.2 上重测 README 的全部加速比并标明版本。
