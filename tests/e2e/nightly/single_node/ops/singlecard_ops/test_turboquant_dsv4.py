# SPDX-License-Identifier: Apache-2.0
"""Run after rebuilding the CANN operators and torch extension."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.dsa_attn_kv_plan import get_dsa_attn_kv_plan
from vllm_ascend.attention.mixed_quant_sparse_flash_mla import (
    mixed_quant_sparse_flash_mla,
    mixed_quant_sparse_flash_mla_metadata,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.quantization.turboquant.latent import CENTROIDS, TurboQuantLatent
from vllm_ascend.utils import bootstrap_custom_op_env


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_tokens", [0, 7, 28, 8192])
def test_tq_batched_slot_metadata_is_exact(dtype, num_tokens):
    bootstrap_custom_op_env(include_vendor_lib=True, include_turboquant=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    plan = get_dsa_attn_kv_plan(SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="turboquant_4bit_nc")))
    sizes = [8, 32, 128] * 40
    torch.manual_seed(45)
    slots_cpu = torch.randint(-2, 1000000, (len(sizes), num_tokens + 3), dtype=dtype)
    slots_cpu[:, :2] = -1
    slots_cpu = slots_cpu[:, :num_tokens]
    expected = torch.stack([plan.format_dsa_slot_mapping(row, size) for row, size in zip(slots_cpu, sizes)])
    slots = torch.zeros((len(sizes), num_tokens + 3), dtype=dtype, device="npu:0")[:, :num_tokens]
    slots.copy_(slots_cpu)
    block_sizes = torch.tensor(sizes, dtype=dtype, device="npu:0").unsqueeze(1)
    output = plan.format_dsa_slot_mapping(slots, block_sizes)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    if num_tokens:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = plan.format_dsa_slot_mapping(slots, block_sizes)
        slots_cpu.add_(17)
        slots.copy_(slots_cpu)
        graph.replay()
        torch.npu.synchronize()
        expected = torch.stack([plan.format_dsa_slot_mapping(row, size) for row, size in zip(slots_cpu, sizes)])
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("page_stride", [33024, 1049028, 1056768])
def test_tq_quant_scatter_fused_attention(capture, page_stride):
    if not torch.npu.is_available():
        pytest.skip("Requires Ascend A2/A3")
    bootstrap_custom_op_env(include_vendor_lib=True, include_turboquant=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch.manual_seed(0)
    device = "npu:0"
    store = TurboQuantLatent()
    q = torch.randn(1, 4, 512, device=device, dtype=torch.bfloat16)
    ori = torch.randn(160, 1, 512, device=device, dtype=torch.bfloat16)
    cmp = torch.randn(33, 1, 512, device=device, dtype=torch.bfloat16)
    ori_cache = store.forward(ori).view(5, 32, 1, 512)
    # Use a larger physical stride than the logical 32-row page.
    backing = torch.zeros(2 * page_stride, dtype=torch.uint8, device=device)
    cmp_cache = torch.as_strided(backing, (2, 32, 1, 258), (page_stride, 258, 258, 1))
    positions = torch.arange(33, device=device, dtype=torch.int32)
    slots = torch.stack((positions // 32, positions % 32), dim=-1)
    indices = torch.full((1, 1, 512), -1, dtype=torch.int32, device=device)
    indices[0, 0, :33] = positions
    sinks = torch.zeros(4, dtype=torch.float32, device=device)
    lengths = torch.tensor([132], dtype=torch.int32, device=device)
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    ori_table = torch.arange(5, dtype=torch.int32, device=device).view(1, -1)
    cmp_table = torch.arange(2, dtype=torch.int32, device=device).view(1, -1)
    metadata = mixed_quant_sparse_flash_mla_metadata(
        num_heads_q=4,
        num_heads_kv=1,
        head_dim=512,
        cu_seqlens_q=cu_q,
        seqused_kv=lengths,
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_kv=132,
        cmp_topk=512,
        cmp_ratio=4,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        has_ori_kv=True,
        has_cmp_kv=True,
    )

    def run():
        packed = store.compress(cmp)
        torch.ops._C_ascend.npu_scatter_nd_update_sk(cmp_cache.view(torch.int8), slots, packed.view(torch.int8))
        result, _ = mixed_quant_sparse_flash_mla(
            store.forward(q),
            ori_kv=ori_cache,
            cmp_kv=cmp_cache,
            ori_block_table=ori_table,
            cmp_block_table=cmp_table,
            cmp_sparse_indices=indices,
            cu_seqlens_q=cu_q,
            seqused_kv=lengths,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=512**-0.5,
            cmp_ratio=4,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=127,
            ori_win_right=0,
            layout_q="TND",
        )
        return store.inverse(result)

    for _ in range(3):
        result = run()
    if capture:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = run()
        graph.replay()
    torch.npu.synchronize()
    # Independent CPU dequantization and attention oracle includes the sink.
    packed = cmp_cache.cpu().reshape(-1, 258)[:33]
    codes = torch.stack((packed[:, :256] & 15, packed[:, :256] >> 4), dim=-1).flatten(1)
    scale = packed[:, 256:].contiguous().view(torch.float16).float()
    cmp_dequant = (torch.tensor(CENTROIDS)[codes.long()] * scale).to(torch.bfloat16).float()
    kv = torch.cat((ori_cache.cpu().reshape(-1, 512)[4:132].float(), cmp_dequant))
    rotation = store.rotation.cpu()
    rotated_q = (q.cpu().float() @ rotation).to(torch.bfloat16).float()[0]
    logits = rotated_q @ kv.T / 512**0.5
    probs = torch.cat((logits, sinks.cpu().view(4, 1)), dim=-1).softmax(-1)[:, :-1]
    expected = (probs @ kv) @ rotation.T
    torch.testing.assert_close(result.cpu().float()[0], expected, rtol=0.04, atol=0.04)


@pytest.mark.parametrize(
    "batch,query_len,key_len,topk",
    [(1, 17, 260, 512), (16, 7, 20480, 512), (16, 7, 20480, 1024), (1, 257, 20480, 512)]
    + [(2, 1, rows * 4, 1024) for rows in (1, 15, 16, 17, 127, 128, 129, 511, 512, 513)],
)
def test_tq_batched_attention_reference(batch, query_len, key_len, topk):
    bootstrap_custom_op_env(include_vendor_lib=True, include_turboquant=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch.manual_seed(71)
    device, heads, page = "npu:0", 8, 32
    store = TurboQuantLatent()
    ori_rows = (key_len + page - 1) // page * page
    cmp_rows = (key_len // 4 + page - 1) // page * page
    ori = store.forward(torch.randn(ori_rows, 1, 512, device=device, dtype=torch.bfloat16))
    packed = store.compress(torch.randn(cmp_rows, 1, 512, device=device, dtype=torch.bfloat16))
    q = store.forward(torch.randn(batch * query_len, heads, 512, device=device, dtype=torch.bfloat16))
    # Requests share read-only pages but use different queries and sparse selections.
    ori_table = torch.arange(ori_rows // page, device=device, dtype=torch.int32).repeat(batch, 1)
    cmp_table = torch.arange(cmp_rows // page, device=device, dtype=torch.int32).repeat(batch, 1)
    indices = torch.full((batch * query_len, 1, topk), -1, dtype=torch.int32)
    for row in range(batch * query_len):
        available = (key_len - query_len + row % query_len + 1) // 4
        count = min(topk, available)
        indices[row, 0, :count] = torch.randperm(available)[:count].sort().values
    lengths = torch.full((batch,), key_len, device=device, dtype=torch.int32)
    cu_q = torch.arange(batch + 1, device=device, dtype=torch.int32) * query_len
    sinks = torch.randn(heads, device=device, dtype=torch.float32)
    metadata = mixed_quant_sparse_flash_mla_metadata(
        num_heads_q=heads, num_heads_kv=1, head_dim=512, cu_seqlens_q=cu_q,
        seqused_kv=lengths, batch_size=batch, max_seqlen_q=query_len,
        max_seqlen_kv=key_len, cmp_topk=topk, cmp_ratio=4,
        ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=127, ori_win_right=0,
        layout_q="TND", has_ori_kv=True, has_cmp_kv=True,
    )
    indices_npu = indices.to(device)

    def run():
        return mixed_quant_sparse_flash_mla(
            q, ori_kv=ori.view(-1, page, 1, 512), cmp_kv=packed.view(-1, page, 1, 258),
            ori_block_table=ori_table, cmp_block_table=cmp_table,
            cmp_sparse_indices=indices_npu, cu_seqlens_q=cu_q,
            seqused_kv=lengths, sinks=sinks, metadata=metadata,
            softmax_scale=512**-0.5, cmp_ratio=4, ori_mask_mode=4, cmp_mask_mode=3,
            ori_win_left=127, ori_win_right=0, layout_q="TND",
        )[0]

    eager = run().cpu().float()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run()
    for _ in range(25 if batch == 16 else 1):
        graph.replay()
        actual = output.cpu().float()
        torch.testing.assert_close(actual, eager, rtol=0, atol=0)
    packed_cpu = packed.cpu().reshape(-1, 258)
    codes = torch.stack((packed_cpu[:, :256] & 15, packed_cpu[:, :256] >> 4), dim=-1).flatten(1)
    scales = packed_cpu[:, 256:].contiguous().view(torch.float16).float()
    cmp = torch.tensor(CENTROIDS)[codes.long()] * scales
    ori_cpu, q_cpu = ori.cpu().reshape(-1, 512).float(), q.cpu().float()
    for row in range(batch * query_len):
        position = key_len - query_len + row % query_len
        selected = indices[row, 0]
        kv = torch.cat((ori_cpu[max(0, position - 127):position + 1], cmp[selected[selected >= 0]]))
        logits = q_cpu[row] @ kv.T / 512**0.5
        probs = torch.cat((logits, sinks.cpu().view(heads, 1)), dim=-1).softmax(-1)[:, :-1]
        torch.testing.assert_close(actual[row], probs @ kv, rtol=0.04, atol=0.04)


@pytest.mark.parametrize("dtype,width", [(torch.bfloat16, 512), (torch.int8, 258)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("paired", [False, True])
def test_tq_cache_write_padding_graph(dtype, width, index_dtype, paired):
    from vllm_ascend.quantization.turboquant.cache_write import write_cache

    init_device_properties_triton()
    page_size, page_stride = 32, 1056768
    backing = torch.full((3 * page_stride,), 7, dtype=dtype, device="npu:0")
    cache = backing.as_strided((3, page_size, 1, width), (page_stride, width, width, 1))
    values = torch.arange(9, device="npu:0").to(dtype).view(-1, 1, 1).expand(-1, 1, width).contiguous()
    flat_slots = torch.tensor([-1, 0, 1, 31, -2, 32, 33, 95, -1], dtype=index_dtype, device="npu:0")
    slots = torch.full((9, 2), -1, dtype=index_dtype, device="npu:0") if paired else flat_slots.clone()
    write_cache(cache, values, slots)
    torch.npu.synchronize()
    backing.fill_(7)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        write_cache(cache, values, slots)
    if paired:
        slots.copy_(torch.stack((flat_slots // page_size, flat_slots % page_size), dim=-1))
    graph.replay()
    expected = torch.full_like(backing.cpu(), 7)
    for row, slot in enumerate(flat_slots.cpu().tolist()):
        if slot >= 0:
            start = slot // page_size * page_stride + slot % page_size * width
            expected[start:start + width] = row
    torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
    slots.fill_(-1)
    graph.replay()
    torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,width", [(torch.bfloat16, 512), (torch.int8, 258)])
def test_tq_cache_write_runtime_token_counts(dtype, width):
    from vllm_ascend.quantization.turboquant.cache_write import write_cache

    init_device_properties_triton()
    cache = torch.zeros((256, 32, 1, width), dtype=dtype, device="npu:0")
    for tokens in (1, 7, 112, 8192):
        slots = torch.arange(tokens, dtype=torch.int32, device="npu:0")
        slots[::7] = -1
        values = (torch.arange(tokens, device="npu:0") % 101).to(dtype)
        updates = values[:, None].expand(-1, width).contiguous()
        write_cache(cache, updates, slots)
        torch.npu.synchronize()
        cache.zero_()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            write_cache(cache, updates, slots)
        graph.replay()
        actual = cache.cpu().view(-1, width)
        expected = torch.zeros_like(actual)
        valid = slots.cpu() >= 0
        expected[:tokens][valid] = updates.cpu()[valid]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
