# SPDX-License-Identifier: Apache-2.0
"""Run after rebuilding both the custom CANN package and torch extension."""
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.mixed_quant_sparse_flash_mla import (
    mixed_quant_sparse_flash_mla,
    mixed_quant_sparse_flash_mla_metadata,
)
from vllm_ascend.quantization.turboquant.latent import CENTROIDS, TurboQuantLatent
from vllm_ascend.utils import bootstrap_custom_op_env


@pytest.mark.parametrize("capture", [False, True])
def test_tq_quant_scatter_fused_attention(capture):
    if not torch.npu.is_available():
        pytest.skip("Requires Ascend A2/A3")
    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch.manual_seed(0)
    device = "npu:0"
    store = TurboQuantLatent()
    q = torch.randn(1, 4, 512, device=device, dtype=torch.bfloat16)
    ori = torch.randn(160, 1, 512, device=device, dtype=torch.bfloat16)
    cmp = torch.randn(33, 1, 512, device=device, dtype=torch.bfloat16)
    ori_cache = store.forward(ori).view(5, 32, 1, 512)
    # Use a larger physical stride than the logical 32-row page.
    backing = torch.zeros(2 * 33024, dtype=torch.uint8, device=device)
    cmp_cache = torch.as_strided(backing, (2, 32, 1, 258), (33024, 258, 258, 1))
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
        num_heads_q=4, num_heads_kv=1, head_dim=512, cu_seqlens_q=cu_q,
        seqused_kv=lengths, batch_size=1, max_seqlen_q=1, max_seqlen_kv=132,
        cmp_topk=512, cmp_ratio=4, ori_mask_mode=4, cmp_mask_mode=3,
        ori_win_left=127, ori_win_right=0, layout_q="TND", has_ori_kv=True, has_cmp_kv=True,
    )

    def run():
        packed = store.compress(cmp)
        torch.ops._C_ascend.npu_scatter_nd_update_sk(cmp_cache.view(torch.int8), slots, packed.view(torch.int8))
        result, _ = mixed_quant_sparse_flash_mla(
            store.forward(q), ori_kv=ori_cache, cmp_kv=cmp_cache,
            ori_block_table=ori_table, cmp_block_table=cmp_table, cmp_sparse_indices=indices,
            cu_seqlens_q=cu_q, seqused_kv=lengths, sinks=sinks, metadata=metadata,
            softmax_scale=512**-0.5, cmp_ratio=4, ori_mask_mode=4, cmp_mask_mode=3,
            ori_win_left=127, ori_win_right=0, layout_q="TND",
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
