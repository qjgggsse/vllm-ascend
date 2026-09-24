# DeepSeek V4 TurboQuant：Model Runner V2 迁移说明

本实现以当前 vLLM-Ascend 的 Model Runner V2、DSA 执行计划和上游 KV cache descriptor 为基础，
复用旧 GLM 分支的配置入口与旋转量化流程，以及旧 DeepSeek 分支的紧凑 KV 布局思路。
目前完成代码接入、CPU 单元测试及新增算子的 A3 kernel 编译；NPU 精度、整模型运行及实际服务并发仍待目标环境验证。

## 1. 对比基线与设计差异

本次检查的工作树：

| 用途 | 目录 | HEAD |
| --- | --- | --- |
| 当前实现 | `/vllm-workspace/vllm-ascend` | `cfbd3ac72` |
| 旧 DeepSeek 实现（用户所指 025 基线） | `/mnt/share_space/w00502534/DeepSeek/Flash/vllm-ascend` | `1ef6fbca3` |
| 旧 GLM 实现 | `/mnt/share_space/w00502534/DeepSeek/Flash/vllm-ascend-glm` | `f2f74a16c` |

两个参考目录已有未提交改动，参考内容包含其工作树，不能仅用上述提交号复现。本次未修改参考目录。

| 关注点 | 旧实现 | 当前框架与本次处理 |
| --- | --- | --- |
| 框架边界 | 通过 Ascend 平台、patch、attention backend 和自定义算子接入 vLLM | 保持这一插件架构；不修改上游 vLLM |
| 执行器 | 旧 TQ 主要沿 V1 路径接入；GLM 配置明确拒绝 V2 | 以 `worker/v2/model_runner.py` 为入口，复用当前 V2 执行流程 |
| V2 职责划分 | 旧版 V2 与当前依赖的上游接口不同，不能直接复制 runner 改动 | 当前继承上游 GPUModelRunner，并拆分 `attn_utils`、model states、图执行和设备元数据等职责；仅扩展 `attn_utils` 的缓存视图 |
| Attention 选择 | 旧 TQ 分支分散处理量化开关和 attention 调用 | 由 `DsaAttnKvPlan` 根据设备、cache dtype、层压缩率选择 TQ C4 分支 |
| 逻辑块与物理块 | 旧版使用 `compress_ratio` 等字段 | 当前使用 `tokens_per_state`；C4 的逻辑 block size 为物理行数的 4 倍，调度仍按原始 token 计数 |
| 缓存描述 | 旧 DS 使用自定义 `AscendPackedKVCacheTensor` 和 V1 分配逻辑 | 使用上游 `KVCacheTensor(size, layers, offset, layer_stride, block_stride)`，所有视图共用一次底层分配 |
| 配置 | GLM 使用 `--kv-cache-dtype turboquant_4bit_nc`，内部派生相关行为 | 保留该入口；不增加 TQ 环境变量或 additional-config 开关 |
| 量化 | GLM 的有符号 Hadamard、固定码本、逐行归一化；DS 的紧凑 KV 与 scale 修正 | Python 适配原样复制的 ops-nn ABI，生成 258 字节 C4 行 |
| 容量 | 旧 DS 对压缩后的物理 KV 大小及打包进行专门处理 | 当前分组、分配、跨 rank 块数归一化和 admission 共用同一物理字节计算 |

当前 BF16 planner 将 C4/indexer 页作为规范尺寸，对 SWA/state 补齐并按层 tuple 数分组。
TQ 后继续使用这套规则，会产生不必要的补齐，甚至使 state 页大于压缩后的规范页。
因此 TQ 分支按未补齐页的最大字节数切分逻辑组，并将组内各层页打包。
非 TQ 分支保留原规划方式。

### 1.1 原有功能兼容约束

不影响 vLLM-Ascend 原有功能是本次接入的验收要求。只有显式设置
`--kv-cache-dtype turboquant_4bit_nc` 时才启用新行为；普通配置不执行 TurboQuant 的
模型、设备或 opbase 能力校验，也不要求 Model Runner V2。

非 TQ attention 继续选择原算子和参数；缓存写入保持原 scatter 调用，包括空输入的处理。
V2 的非连续页布局仅用于带 TurboQuant 标记的缓存组，普通 DSV4 保留连续页校验和默认缓存视图。
公共头文件的 `inline` 修复保持函数体不变。

回归覆盖包括普通 cache dtype 配置、A2/A3/A5 上 C1/C4/C128 的原 attention 计划、
非 TQ 缓存写入及 V2 非连续页拒绝行为。CPU 测试不能替代默认模型路径在 NPU 上的
精度、图执行和性能回归；这些项目仍是目标环境验收的一部分。

本轮兼容性检查：attention 计划测试 40 项通过，TurboQuant 独立 CPU 测试 36 项通过。
V2 新增及原有测试在收集阶段受当前安装的 vLLM 接口不匹配阻塞
（缺少 `vllm.v1.attention.ops.pcp`），尚未执行，不能计作通过。

## 2. 数据流和实现位置

实现文件：

- `vllm_ascend/quantization/turboquant/config.py`：启动参数与算子能力校验。
- `vllm_ascend/quantization/turboquant/latent.py`：旋转、量化、scale 修正和输出逆旋转。
- `vllm_ascend/attention/dsa_attn_kv_plan.py`：C4 attention/metadata 算子选择。
- `vllm_ascend/attention/dsa_v1.py`：当前 V2 同样使用此 backend；负责实际 KV 写入与 attention 调用。
- `vllm_ascend/attention/mixed_quant_sparse_flash_mla.py`：参数名、压缩长度和 layout 适配。
- `vllm_ascend/quantization/turboquant/cache.py`：分组、物理布局、descriptor 与容量计算。
- `vllm_ascend/patch/platform/patch_kv_cache_utils.py`：在现有规划入口选择 TQ 实现。
- `vllm_ascend/worker/v2/attn_utils.py`：一次底层分配，按 offset/block stride 创建实际缓存视图。

### 2.1 量化范围

| 数据 | TQ 模式下存储 | 处理 |
| --- | --- | --- |
| C4 压缩 attention KV | `uint8[..., 258]` | 512 维的 4-bit code 共 256 字节，加 2 字节 FP16 scale |
| C4 层 SWA KV | BF16，512 维 | 写入前进行相同旋转，使两类 KV 与 query 位于同一坐标系 |
| C128 压缩 attention KV | BF16，512 维 | 保留当前非 TQ attention 路径 |
| C128/SWA-only 层 SWA KV | BF16，512 维 | 保留当前处理 |
| Indexer KV 和 compressor state | 原有类型 | 保留其量化/状态语义，按实际页大小规划内存 |

每个 C4 attention 实例持有自己的变换常量，固定随机种子为 0，码本使用 GLM 分支的 16 个 centroid。
在 profiling forward 中初始化，使其内存被计入 profile，并在图捕获前固定下来。
没有新增可变全局张量缓存。

量化写入顺序为：RoPE 后的 compressor 输出 → 有符号归一化 Hadamard → ops-nn TurboQuant → scale 修正 → 紧凑缓存。
ops-nn 返回 packed code 和原始 L2 norm；融合 attention 直接将 centroid 乘以缓存 scale。
为保留旧 DS 的范数修正，适配层计算：

```text
scale = norm / sqrt(sum(selected_centroid ** 2))
slot  = packed_codes[256 bytes] + fp16(scale)[2 bytes]
```

scale 修正发生在 Python 张量层，未修改量化算子源码。
写缓存复用 `npu_scatter_nd_update_sk`；A2/A3 的 packed byte 写入使用 `int8` 视图，存储字节保持不变。
该算子读取实际 stride，并跳过负 slot，支持图执行中的 padding token。

attention 前 query 做同样旋转；`MixedQuantSparseFlashMla` 在 attention 内融合反量化；
结果逆旋转后继续当前的逆 RoPE 与输出投影。原有 per-head sinks 由 attention 实例持有。

### 2.2 完整复制与源包同步

四个目录均先完整复制源包工作树，后续必要的算子修复同步两侧。
一致性检查覆盖文件集合和文件内容，不允许仅覆盖同名文件而遗留旧目录内容。

| 源路径（相对 `/vllm-workspace`） | 目标路径（相对仓库） | 文件数 |
| --- | --- | --- |
| `ops-nn/quant/turbo_quant` | `csrc/quant/turbo_quant` | 25 |
| `ops-transformer/attention/sparse_flash_mla` | `csrc/attention/sparse_flash_mla` | 112 |
| `ops-transformer/attention/mixed_quant_sparse_flash_mla` | `csrc/attention/mixed_quant_sparse_flash_mla` | 75 |
| `ops-transformer/attention/mixed_quant_sparse_flash_mla_metadata` | `csrc/attention/mixed_quant_sparse_flash_mla_metadata` | 15 |

Sparse MLA 的完整副本包含 arch35 公共 metadata、host 公共头文件、tests 和 torch_extension。
不再逐个补入 Sparse MLA 公共头文件，也不再手工拼接 `CheckContext`。
原 Ascend 专用 `sparse_flash_mla_torch_adpt.h` 原样移到 `csrc/aclnn_torch_adapter/`，
`torch_binding.cpp` 的 include 随之更新，原 Torch 注册及函数体保留。
外部公共依赖 `ops-nn/common/inc/error_util.h` 仍原样复制到 `csrc/common/include/error_util.h`。
ops-nn 的 TurboQuant 源码未修改。

```bash
python tools/check_turboquant_sources.py --source-root /vllm-workspace
```

该检查覆盖 227 个算子目录文件及 1 个外部公共头文件。

### 2.3 替换 Sparse MLA 后的修复复核

| 先前修复 | 当前处理 | 原因 |
| --- | --- | --- |
| MixedQuant metadata 改用旧 Sparse MLA 的入口头 | 撤回，两侧恢复 `arch35/common/smla_metadata_common.h` | 完整 Sparse MLA 副本现在提供原依赖 |
| 删除 MixedQuant host 的 `op_host/tiling_util.h` include | 保留 | 此文件属于源工程的 common 目录，不在 Sparse MLA 包中；该 include 未使用，Ascend 仍不提供它 |
| Metadata AICPU CMake 兼容两种注册函数 | 保留 | 两套工程的函数签名、目标名称及 JSON 注册方式仍不同 |
| Metadata ACLNN 日志宏及 `CHECK_RET` 兼容 | 保留 | CANN 公共日志宏与 opdev 日志宏的参数约定仍不同，与 Sparse MLA 目录是否完整无关 |
| Ascend `CheckContext` 手工补字段 | 由完整复制替代 | 当前结构及其头文件来自同步后的 Sparse MLA 源包 |

为保留 Ascend 已有运行行为，完整复制后保留两项局部兼容处理，并同步回 ops-transformer：

- `op_host/checkers/sparse_compression_checker.cpp`：仅普通 SparseFlashMla、无 cmp KV 的 SWA
  接受历史 `cmp_ratio=0`。带 cmp KV 的零倍率和 MixedQuant/Quant 的零倍率仍拒绝。
  源包的倍率 1 默认值、TQ 专用检查及其他检查逻辑保留；对应 C++ checker UT 已补充。
- `op_kernel/arch22/sparse_flash_mla_csa_kernel.h`：保留原 Ascend 的有效稀疏索引前缀计数，
  避免索引存在间隔并以 `-1` 填充时，gather 的有效长度与后续计算长度不一致。
  同时保留源包对每个压缩块使用自身读取切片的修复，没有整体回退旧 kernel。

Sparse MLA 使用源包的完整模板集合：host、A2/A3、A5 的头文件均声明 320 个组合，
包含原 Ascend 模型所需的全部 key，参数顺序及编码保持一致；不再保留原目录的 6/12 个组合裁剪。
源码 opdef 同时提供 FP16/BF16，因此编译成本会高于旧的 BF16 裁剪版本。
这不等于所有模板组合都能通过 host 的运行时参数检查。

此前 Torch 扩展的重复符号修复继续保留：`op_api_common.h` 中
`IsOpInputBaseFormat` 添加 `inline`，函数体不变；它属于适配层，不属于算子内部算法修改。

本次替换前的 Ascend 算子目录及首次生成物备份位于
`/tmp/sparse-mla-before-copy-tfmhv9fr`。构建时重新生成 opdef、编译参数、源码副本和 kernel，
避免 CMake/Ninja 的旧生成文件掩盖此次替换。

本次替换的验证记录：

- `ascend910_93` 定向构建退出码为 0：Sparse MLA 的 FP16/BF16 完整模板内核和
  MixedQuant 两个内核均重新生成 `.o/.json`，host tiling、ACLNN、proto、metadata AICPU
  及 `cust_aicpu_kernels` 目标构建通过。Sparse MLA 每种 dtype 分别编译了 320 个 Vector Core
  和 320 个 Cube Core 模板；此次完整内核编译约耗时 45 分钟。
  构建日志：`/tmp/sparse-mla-copy-rebuild-unrestricted.log`。
- 8 项 tiling key 回归通过，检查完整组合集合、原模型 key 子集和编码。
- 使用实际 CANN 头文件及库编译运行 host checker 回归，通过普通 SWA 的倍率 0/1、
  压缩 KV 非法倍率、其他算子变体倍率 0 拒绝及 TQ residual 参数校验。
- `torch_binding.cpp` 使用实际 Torch/CANN 头文件的 C++ 编译语法检查通过。
- 四个算子目录的 227 个文件及外部公共头文件逐字节一致。

这些检查不代表已完成整个 Torch 扩展的链接、editable 安装或 NPU 精度与性能验证。

## 3. KV 压缩与 Maximum concurrency

单个 C4 attention KV 行从 BF16 的 `512 × 2 = 1024` 字节变成 `258` 字节，压缩比约 `3.969×`。
整模型还包含 SWA、C128、indexer 和 state，因此整模型可支持并发不会等于该压缩比。

物理布局遵守以下约束：

1. 同组不同层的页互不重叠。
2. 不同组可以复用同一物理 slot；同一 block ID 只能归属一个组。
3. 跨组共享 slot 时使用相同物理 stride，因此不同 block ID 永不别名。
4. offset 满足数据类型对齐，TQ block stride 保持 258 字节行的整倍数。
5. 全部 descriptor 共用一个 backing，禁止按每个 descriptor 的 size 重复分配整个池。

容量计算使用：

```text
pool_bytes_per_block = sum(physical_slot_stride)
num_blocks = available_KV_bytes // pool_bytes_per_block
request_blocks = sum(group.max_memory_usage_pages(config))
Maximum concurrency = num_blocks / request_blocks
```

实际分配、单请求 admission 和跨 rank 重新归一化都使用相同的 `pool_bytes_per_block`。
日志仍调用上游 `get_max_concurrency_for_kv_cache_config`，没有人为放大结果。
额外输出 `DeepSeek V4 TurboQuant KV: C4 slot=258 bytes, pool bytes/block=..., groups=..., blocks=...`，
便于将规划值与底层内存分配对照。

### 3.1 CPU 容量回归结果

测试使用上游真实并发计算函数，以独立 PageSpec 描述物理字节几何，绕开本机旧 vLLM 的字段差异。
BF16 对照复现当前原 planner 的 tuple 与 state 补齐方式。
模型几何为 43 层（21 个 C4、20 个 C128、43 个 SWA，以及对应 indexer/state），
固定 KV 预算 32 GiB、max in-flight tokens 8192，不包含 MTP/CP。

| 每请求 token 上限 | 物理 block 行数 | BF16 并发容量 | TQ 并发容量 |
| --- | --- | --- | --- |
| 131072 | 32 | 6.347 | 10.265 |
| 131072 | 64 | 6.342 | 10.257 |
| 131072 | 128 | 6.333 | 10.282 |
| 1048576 | 32 | 3.083 | 5.634 |
| 1048576 | 64 | 3.082 | 5.631 |
| 1048576 | 128 | 3.079 | 5.649 |

这是固定内存预算下的容量计算回归，**不是 NPU 实测日志、吞吐或精度结果**。
真实运行的内存 profile、图捕获、模型并行划分、MTP 和块取整都会影响结果。
增加分组还会影响调度开销，需与旋转、scale 修正开销一起在目标硬件测量。

## 4. 启动与目标环境验证

### 4.1 当前接入范围

- DeepSeek V4、Model Runner V2、BF16、Ascend A2/A3。
- `head_dim=512`、`qk_rope_head_dim=64`、`index_topk=512/1024`，每 rank query head 数为 4 的倍数。
- opbase 提供 `NnopbaseSupportTensorV2`，以原始 strided cache 调用融合算子，避免每次 attention 复制整个 KV 池。
- 当前拒绝 PCP/DCP、KV transfer 和多 rank KV layer parallelism；这些路径需要单独适配缓存写入/分配。

MTP 未指定压缩率时沿用当前框架的 dense/SWA 语义；本次未完成 MTP 整模型验证。
多模态与其他 speculative decoding 组合也未完成验证。

需使用与当前 Ascend 分支接口匹配的 vLLM，然后按仓库常规流程重新编译安装自定义 CANN 算子和 Torch 扩展。
用户此前构建通过的完整算子包对应此次 Sparse MLA 替换前的代码；本次替换的验证范围见 2.3。
整个 Torch 扩展及 editable 安装仍需重新构建通过，不能将源码一致性或局部编译检查理解为完整安装成功。

启用参数示例（其余参数沿用已验证的同模型 V2 启动配置）：

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /path/to/DeepSeek-V4 \
  --dtype bfloat16 \
  --kv-cache-dtype turboquant_4bit_nc \
  --tensor-parallel-size 8 \
  --max-model-len 131072
```

### 4.2 验证命令

独立 CPU 测试共 31 项通过，覆盖旋转与范数修正、算子参数/schema、配置校验、
跨组缓存别名、容量计算，以及 PP rank 不含 C4 的规划识别。
Python 语法、`git diff --check`、构建脚本语法和算子源码一致性检查通过。

可独立运行的 CPU 测试：

```bash
VLLM_PLUGINS='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python -m pytest --confcutdir=tests/ut/quantization/turboquant \
  tests/ut/quantization/turboquant -q -s
```

在匹配的框架环境运行实际 V2 descriptor/allocator/view 回归：

```bash
python -m pytest tests/ut/worker/test_turboquant_v2.py -q
```

编译后在 NPU 上运行量化 → strided scatter → 融合 attention 测试，覆盖 eager 和 NPUGraph，
以 CPU 独立反量化及带 sinks 的 attention 为 oracle：

```bash
python -m pytest \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_turboquant_dsv4.py -q
```

本机安装的 vLLM 为 0.27.1，标准 UT conftest 在 `EagleModelMixin.AUX_HIDDEN_STATE_KEY` 处报接口不兼容，
所以实际 V2 集成 UT 尚未执行。独立 CPU 测试不能替代这一验证。
本机无可用 NPU；标准格式检查也因缺少 pre-commit/ruff 未完成。

### 4.3 服务验收

用同一模型、权重量化、TP/PP、max-model-len、批处理上限和显存预算分别启动 BF16 KV 与 TQ KV，
只改变 `--kv-cache-dtype`。不要用固定 `num_gpu_blocks_override` 掩盖物理容量差异。
先验证 eager 正确性，再验证图执行；覆盖 prefill、decode、chunked prefill、不同请求长度和 padding。

记录并对照：

- KV 可用内存、TQ 物理 bytes/block、分组数、num_blocks 和实际 backing 分配大小。
- `Maximum concurrency for ... tokens per request: ...x`，相同 token 上限下应体现压缩收益。
- 并发请求达到容量边界时的峰值显存、抢占/OOM、吞吐和延迟。
- 相对 BF16 的输出质量回归；TQ 属于有损量化，不能要求逐 token 输出完全相同。

若日志没有合理提升，应先核对实际预算、是否走到 TQ C4 spec、物理页是否仍被 BF16 补齐、
是否重复分配 backing、是否在算子调用前发生全缓存 contiguous 转换，再检查并发日志。
