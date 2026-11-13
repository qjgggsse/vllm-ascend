# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the actual SFA bridge against an independent FP32 CPU reference."""

import math
from functools import partial
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
import torch_npu

pytest.importorskip("vllm.v1.hisparse.backend")
from vllm.v1.hisparse.runtime import HiSparseRuntime, PagedCacheView  # noqa: E402

from vllm_ascend.attention.sfa_hisparse import build_hisparse_prefill_staging_plan  # noqa: E402
from vllm_ascend.hisparse.backend import AscendHiSparseBackend  # noqa: E402
from vllm_ascend.hisparse.cache_ops import HiSparseNPUOps, scatter_packed_kv  # noqa: E402
from vllm_ascend.hisparse.host_pool import allocate_mapped_host  # noqa: E402


@pytest.fixture(autouse=True)
def _require_npu_sfa():
    from vllm_ascend.utils import bootstrap_custom_op_env

    # Match platform initialization: publish the OPP path before device queries
    # or set_device can initialize the runtime and its tiling registry.
    bootstrap_custom_op_env()
    if not torch.npu.is_available():
        pytest.skip("HiSparse SFA tests require an NPU")
    if not 250 <= torch_npu.npu.get_soc_version() <= 256:
        pytest.skip("requires A3")
    torch.npu.set_device(0)
    # Load the required SFA extension without changing custom-op selection.
    import_module("vllm_ascend.vllm_ascend_C")


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("num_queries", [1, 7, 35])
def test_sparse_attention_reference(num_queries, strided, capture):
    from vllm_ascend.attention.sfa_hisparse import _ATTENTION_CHUNK_ROWS, AscendHiSparseImpl

    torch.manual_seed(47)
    device = torch.device("npu:0")
    topk, latent_dim, rope_dim, heads = 2048, 512, 64, 8
    block_size, blocks = 128, 16
    capacity = _ATTENTION_CHUNK_ROWS
    scale = 1 / math.sqrt(latent_dim + rope_dim)
    workspace = (
        torch.empty((capacity, topk, 1, latent_dim), device=device, dtype=torch.bfloat16),
        torch.empty((capacity, topk, 1, rope_dim), device=device, dtype=torch.bfloat16),
        torch.arange(capacity * topk // block_size, dtype=torch.int32, device=device).view(capacity, -1),
        torch.arange(topk, dtype=torch.int32, device=device).view(1, 1, topk),
        torch.arange(1, capacity + 1, dtype=torch.int32, device=device),
    )
    impl = SimpleNamespace(kv_lora_rank=latent_dim, qk_rope_head_dim=rope_dim, scale=scale, enable_sparse_sfa_c8=False)
    impl._workspace = lambda dtype: workspace
    # Quantize before taking the FP32 reference so both paths see the same inputs.
    source = torch.randn((blocks, block_size, latent_dim + rope_dim)).to(torch.bfloat16)
    query = torch.randn((num_queries, heads, latent_dim)).to(torch.bfloat16)
    query_rope = torch.randn((num_queries, heads, rope_dim)).to(torch.bfloat16)
    physical = torch.randint(0, blocks * block_size, (num_queries, topk), dtype=torch.int32)
    physical[:, 3::7] = -1
    physical[:, 1] = physical[:, 0]
    # A short row makes duplicate multiplicity and invalid-padding mistakes large.
    physical[0] = -1
    physical[0, :3] = torch.tensor([0, 0, 1], dtype=torch.int32)
    if num_queries > 1:
        physical[-1] = -1
    if strided:
        backing = torch.empty((blocks, 2, block_size, latent_dim + rope_dim), device=device, dtype=source.dtype)
        cache = backing[:, 1]
        cache.copy_(source)
    else:
        cache = source.to(device)
    query_npu, rope_npu, physical_npu = query.to(device), query_rope.to(device), physical.to(device)
    forward = partial(AscendHiSparseImpl._attention_physical, impl, query_npu, rope_npu, cache, physical_npu)

    graph = None
    try:
        if capture:
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for _ in range(2):
                    forward()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                captured = forward()
            # Change both payload and selection length after graph capture.
            query.neg_()
            physical[0, :5] = torch.tensor([3, 3, 8, 12, 14], dtype=torch.int32)
            query_npu.copy_(query)
            physical_npu.copy_(physical)
            for _ in range(3):
                graph.replay()
            actual = captured.float().cpu()
        else:
            actual = forward().float().cpu()

        selected = source.flatten(0, 1)[physical.clamp_min(0).long()].float()
        logits = torch.einsum("thd,tkd->thk", query.float(), selected[..., :latent_dim])
        logits += torch.einsum("thd,tkd->thk", query_rope.float(), selected[..., latent_dim:])
        logits *= scale
        logits.masked_fill_(~(physical >= 0)[:, None, :], -torch.inf)
        probabilities = torch.softmax(logits, dim=-1).nan_to_num(0)
        expected = torch.einsum("thk,tkd->thd", probabilities, selected[..., :latent_dim])
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.025)
        if num_queries > 1:
            assert torch.count_nonzero(actual[-1]) == 0
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()




@pytest.mark.parametrize("resident_block_size", [128, 64, 32])
@pytest.mark.parametrize("host_pages", [1, 2])
def test_resident_prefill_plan_shared_layers(resident_block_size, host_pages):
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    torch.npu.set_device(0)
    device = "npu:0"
    block_size, width, layers, blocks = 128, 576, 4, 16
    page_elements = block_size * width
    block_stride = 6 * page_elements
    raw = torch.zeros(blocks * block_stride * 2, dtype=torch.int8, device=device)
    # Match the shared HMA slab: create and retain every layer view before use.
    views = [
        PagedCacheView.bind(
            raw,
            dtype=torch.bfloat16,
            row_width=width,
            byte_offset=layer * resident_block_size * width * 2,
            block_stride=block_stride * 2,
            num_blocks=blocks,
            block_size=resident_block_size,
        )
        for layer in range(layers)
    ]
    host = allocate_mapped_host(((host_pages + 1) * block_size, width), torch.bfloat16)
    host.zero_()
    host[block_size:].fill_(-23)
    ops = HiSparseNPUOps(host._hisparse_registration)
    try:
        subpages = block_size // resident_block_size
        resident_ids = torch.arange(2, 2 + host_pages * subpages, dtype=torch.int32)
        if host_pages == 2:
            # The last subpage is a real host miss; other rows must stay on NPU.
            resident_ids[-1] = 0
        resident_table = resident_ids[None, :].to(device)
        host_table = torch.arange(1, host_pages + 1, dtype=torch.int32, device=device)[None, :]
        plan = build_hisparse_prefill_staging_plan(
            host_table, torch.tensor([host_pages * block_size - 7], device=device), block_size, host_pages
        )
        state = torch.tensor([0], dtype=torch.int32, device=device)
        logical_rows = torch.arange(host_pages * block_size)
        physical = resident_ids.long()[logical_rows // resident_block_size] * resident_block_size
        physical += logical_rows % resident_block_size
        hits = resident_ids[logical_rows // resident_block_size] > 0
        physical[~hits] = -1
        expected_gpu = torch.cat((torch.full((block_size,), -1), physical)).int()
        expected_mask = (expected_gpu < 0).int()
        original_rows = None
        for layer, view in enumerate(views):
            columns = torch.arange(width)[None, :] % 16
            payload = (layer * 16 + logical_rows[:, None].float() / 8 + columns / 16 + 1).bfloat16()
            on_device = payload.to(device)
            scatter_packed_kv(on_device[:, :512], on_device[:, 512:], view.cache, physical.int().to(device))
            plan.ensure_gpu_sources(resident_table, state, resident_block_size)
            torch.testing.assert_close(plan.gpu_row_ids.cpu().flatten(), expected_gpu, rtol=0, atol=0)
            torch.testing.assert_close(plan.miss_mask.cpu().flatten(), expected_mask, rtol=0, atol=0)
            if original_rows is None:
                original_rows = plan.gpu_row_ids
            else:
                # Layers in an index group reuse the same resolved page plan.
                assert plan.gpu_row_ids is original_rows
            actual = HiSparseRuntime.gather_prefill_cache(
                SimpleNamespace(cache_ops=ops, backend=AscendHiSparseBackend()),
                host.view(-1, block_size, width),
                plan,
                view.cache,
            )
            expected = host.clone()
            expected[block_size:][hits] = payload[hits]
            torch.testing.assert_close(actual.cpu().flatten(0, 1), expected, rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        host._hisparse_registration.close()


@pytest.mark.parametrize("dtype,width", [(torch.bfloat16, 576)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_rows", [0, 1, 4, 9])
def test_backend_resident_payload_graph_replay(dtype, width, index_dtype, num_rows):
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    torch.npu.set_device(0)
    stride = 3 * width
    raw = torch.arange(7 * width * dtype.itemsize, dtype=torch.int64).to(torch.uint8).to("npu:0")
    source = raw.view(dtype).as_strided((2, 2, width), (stride, width, 1), storage_offset=width)
    rows = torch.empty(num_rows, dtype=index_dtype, device="npu:0")
    destination = torch.empty((num_rows, width), dtype=dtype, device="npu:0")
    rows.copy_((torch.arange(num_rows, dtype=index_dtype, device="npu:0") % 5) - 1)
    backend = AscendHiSparseBackend()
    backend.gather_resident_rows(source, destination, rows)
    torch.npu.synchronize()
    if not num_rows:
        assert destination.numel() == 0
        return
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            backend.gather_resident_rows(source, destination, rows)
        for offset in (0, 2):
            values = (torch.arange(num_rows, dtype=index_dtype) + offset) % 5 - 1
            if index_dtype == torch.int64:
                values[0] = -(1 << 34)
            rows.copy_(values.to("npu:0"))
            raw.bitwise_xor_(1)
            graph.replay()
            torch.npu.synchronize()
            safe = values.long().clamp_min(0)
            expected = source.view(torch.uint8).cpu()[safe // 2, safe % 2]
            assert torch.equal(destination.view(torch.uint8).cpu(), expected)
    finally:
        torch.npu.synchronize()
        graph.reset()


def test_prefill_singleton_column_mapping_tracks_current_residency():
    if not torch.npu.is_available():
        pytest.skip("HiSparse prefill mapping requires an NPU")
    torch.npu.set_device(0)
    block = 128
    host_table = torch.tensor([[1, 2]], dtype=torch.int32, device="npu:0")
    lengths = torch.tensor([2 * block - 7], dtype=torch.int32, device="npu:0")
    resident = torch.tensor([[2, 0], [0, 5]], dtype=torch.int32, device="npu:0")
    state = torch.tensor([0], dtype=torch.int32, device="npu:0")

    def resolve():
        plan = build_hisparse_prefill_staging_plan(host_table, lengths, block, 2)
        plan.ensure_gpu_sources(resident, state, block)
        return plan.gpu_row_ids, plan.miss_mask

    # FULL_DECODE_ONLY builds Prefill metadata outside graph capture; Core's
    # representative-page scatter_reduce is not a capturable NPU operation.
    for state_row, pages in ((0, [2, 0]), (1, [0, 5]), (0, [2, 0])):
        state.fill_(state_row)
        gpu_rows, mask = resolve()
        expected = torch.full((3 * block,), -1, dtype=torch.int32)
        for logical, physical in enumerate(pages, start=1):
            if physical:
                expected[logical * block : (logical + 1) * block] = torch.arange(
                    physical * block, (physical + 1) * block, dtype=torch.int32
                )
        torch.testing.assert_close(gpu_rows.cpu().flatten(), expected, rtol=0, atol=0)
        torch.testing.assert_close(mask.cpu().flatten(), (expected < 0).int(), rtol=0, atol=0)
