# HiSparse on Ascend

HiSparse keeps selected sparse MLA KV rows in NPU Resident/Hot storage and
refills missing rows from a Host mirror. This experimental adapter reuses
vLLM Core's cache allocation, scheduler, index groups and connector lifecycle.
The A3 implementation supports INT8 main KV and an INT8 indexer.
BF16 remains available through the ordinary non-HiSparse backend on A3.
Full-model accuracy, capacity and performance qualification is still required.

## Architecture

Core owns logical source pages, Resident/Hot allocation, request state,
prefix-cache admission/eviction, prefetch and Host mirror completion. Ascend
provides the backend through `NPUPlatform.get_hisparse_backend_cls`; it does not
add another scheduler or a CPU TopK/miss planner.

1. The existing indexer produces logical TopK rows. A shared index group owns
   the resolver workspace and refill plan for its participating MLA layers.
2. The device resolver handles the complete contiguous query group for each
   request, including all MTP verification rows. Resident rows take priority.
3. Before eviction, the resolver deduplicates cold Host rows and protects every
   selected Hot hit. It checks the full union against Hot/UB capacity, assigns
   LRU victims, then publishes updated tags, LRU and compact refill descriptors.
   Attention indices retain selection order and duplicate multiplicity; only
   transfers are deduplicated. An oversized union leaves cache state unchanged
   and reports an error through the shared error workspace.
4. Triton gathers missing payloads from explicitly mapped, pinned Host memory
   into the existing shared device allocation. Per-layer page strides are
   preserved. The existing SFA computation consumes the resolved physical rows.
5. New KV writes are mirrored through Core's connector and the existing Ascend
   `swap_blocks_batch` ABI. Stream/event dependencies and completion-before-reuse
   remain under Core ownership. Graph capture retains fixed workspaces and does
   not submit a second Host mirror update.

The shared Torch ABI selects `csrc/hisparse/op_kernel/arch22` on A3 and
`arch35` on A5. This follows the repository's architecture-directory convention
for sparse Attention operators and the existing direct AscendC launch/build
path. A3 uses ordinary AscendC UB hash/tag buffers, scalar control and ordered
MTE transfers. A5 retains the reference SIMT kernel. An ordered A3 statistics
kernel avoids cross-core int64 atomics. A3 scratch checks include the complete
UB tag row and reject unsupported geometry before launch.

When the request's source pages are fully Resident, A3 query rows resolve
independently without modifying Hot/LRU state. Integer vector mapping and
stable compaction preserve physical indices above FP32's exact integer range.
Mixed requests retain the leader-owned union path.

The A5 Host mirror connector wrapper is also reused on A3. Eligible unpadded
decode steps can overlap Target D2H mirroring with drafting after proving that
their source pages and Host destinations are disjoint. Output completion,
post-draft finalization and subsequent page reuse retain the required event
dependencies. Other steps use Core's original completion fence.

Two narrow worker adaptations retain the reference design: a pinned Host
allocator compatible with both DMA and AIV mapping, and slot-staging dtype /
grouped-invalidation hooks. They should move behind Core backend hooks when
those extension points are available. The coordinator patch delegates HiSparse
back to Core instead of applying Ascend's unrelated coordinator policy.
The recycled-page grouped-invalidation optimization remains A5-only; A3 uses
Core's existing per-runtime path for those pages. Large written-token batches
use the shared sorted-key invalidation path with a bounded search loop that
compiles on A3; small decode batches retain direct invalidation.

## Requirements and configuration

- Use the Core revision pinned in `.github/vllm-main-verified.commit`. Core must
  expose `vllm.v1.hisparse.backend`; older CUDA-only HiSparse is insufficient.
- Build vLLM Ascend for the actual A3 SoC (`ascend910_93*`) using the normal
  source installation flow. No additional operator registration package is
  required for the resolver. A5 builds need the reference SIMT-capable toolkit.
- Use Model Runner V2, BF16 activations, INT8 main KV and INT8 indexer on A3.
  A5 C8 code is shared from the reference, but is not qualified by A3 tests.
- Use block size 128 and the existing sparse MLA/SFA TopK constraints.
- Use a private Host pool per TP rank and eager Host mirroring. The runtime
  checks AIV Host mapping and compiled resolver availability explicitly.
- Context parallelism, external P/D transport, legacy
  `sparse_kv_offload_config`, TurboQuant and A3 FP8/C4/BF16 HiSparse KV are unsupported.

Set `VLLM_USE_V2_MODEL_RUNNER=1` and add these options to a compatible sparse
MLA model's serving command:

```bash
--dtype bfloat16 \
--block-size 128 \
--kv-cache-dtype int8 \
--attention-config '{"indexer_kv_dtype":"int8","hisparse_config":{"eager_host_mirror":true}}' \
--kv-transfer-config '{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":4}}'
```

`host_pool_gib` is per rank. Core's `device_buffer_size` must cover the selected
query union; its int16 LRU representation also limits Hot capacity to 32768
rows, and the compiled UB budget can impose a lower limit. Tested A3 geometry
includes TopK 2048 with four query rows, union 8192 and Hot capacity 10240.
These limits do not reduce TopK or silently discard selected tokens.

## KV quantization

HiSparse reuses the existing SFA quantizer and Attention operators. On A3,
512 NoPE elements are quantized to INT8 in groups of 128, while 64 RoPE
elements remain BF16 and four scales remain FP32. Each main KV row occupies
656 bytes. The matching INT8 weight path reuses PROLOG_V3 to write this layout
directly into the current Resident mapping. Separate preprocessing uses the
existing `npu_dynamic_block_quant` path. Host/Hot movement copies the packed
bytes without dequantization or requantization.

The Indexer retains its existing Hadamard transform, per-token INT8 quantizer
and FP16 scale. Its K and scale views retain each shared-slab page's stride;
cache writes must not flatten these views into separate allocations. A5 keeps
its FP8 E4M3FN main KV and FP8/FP32 Indexer scheme.

## Validation status

The production extension was built and installed on `Ascend910_9382` using
CANN 9.2.1, torch 2.10.0+cpu, torch-npu 2.10.0.post7 and Python 3.12. The tests
use an isolated installation of pinned Core revision
`24420b7a341af37a9bc3186e6fc6062512e0aa58`; the original Core checkout is unchanged.

Focused hardware tests cover the native resolver, INT8 packed Host refill,
MTP query unions, cross-stream graph replay, page gaps, invalidation, overflow
and D2H. INT8 Attention is checked against a CPU FP32 reference, including
duplicates and empty queries. The production C8 resolver ABI is tested through
refill and Attention, including capacity filtering, stable compaction, padded
rows and graph replay. PROLOG output and page-strided Resident writes
are compared with ordinary INT8 SFA in eager and graph modes. Indexer stores
are checked for neighboring-byte preservation and its selection agrees with
the corresponding contiguous INT8 cache.

```bash
pytest tests/ut/hisparse
pytest tests/e2e/pull_request/one_card/test_hisparse_a3.py
pytest tests/e2e/pull_request/one_card/test_hisparse_a3_attention.py
pytest tests/e2e/pull_request/one_card/test_hisparse_a3_c8.py
pytest tests/e2e/pull_request/one_card/test_hisparse_a3_prolog.py
```

GLM-5.3 W8A8C8 completed real-weight startup and FULL_DECODE_ONLY graph capture
on A3 with TP16/DP1/EP and MTP3. Short and 16K retrieval smoke requests passed.
Single-concurrency 1K/1K and 15K/1K comparisons ran three rounds per mode with
no failed requests. After the second Resident-resolver optimization, using
rounds 2 and 3, HiSparse increased TPOT by 19.93% and 16.55%, increased ITL by
17.26% and 16.55%, and reduced output throughput by 16.24% and 11.16%,
respectively. Short-input MTP acceptance varied between rounds;
these are preliminary service measurements, not an isolated operator profile.
Neither scenario triggered Host-to-Device refill or preemption, so system
capacity and offload benefits remain untested.

The Host mirror wrapper initialized on all ranks, but deferred mirror waits
did not activate in these padded single-request graph runs. Its eligible
overlap path still needs A3 service validation without padding. The measured
improvement over the preceding implementation does not establish a benefit
from mirror overlap.

Model accuracy is not signed off: the ordinary INT8 baseline also produced
incorrect or unstable arithmetic answers, and the two smoke outputs were not
both token-identical across modes. Earlier BF16 runs do not qualify this INT8
path. A5 compilation and device regression were not run. The adapter remains
experimental; no capacity benefit or production accuracy qualification is claimed.
