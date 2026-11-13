# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod


@pytest.mark.parametrize("weight_format", [2, 29])
def test_prolog_splits_int8_weights_without_changing_values(weight_format):
    """Exercise GLM5.3's real fused projection shape in ND and NZ storage."""
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    torch.npu.config.allow_internal_format = True
    try:
        weight = torch.arange(6144 * 2624, dtype=torch.int32).remainder(127).to(torch.int8).view(6144, 2624)
        scales = torch.arange(1, 2625, dtype=torch.float32)
        source = torch_npu.npu_format_cast(weight.npu(), weight_format)
        impl = SimpleNamespace(
            fused_qkv_a_proj=SimpleNamespace(weight=source, weight_scale=scales.npu()),
            q_proj=SimpleNamespace(weight=source, weight_scale=scales.npu()),
            q_lora_rank=2048,
            _quant_type=AscendW8A8DynamicLinearMethod,
            use_rope=True,
            enable_sparse_sfa_c8=False,
            is_kv_consumer=False,
        )
        AscendSFAImpl._process_weights_for_fused_prolog_v3(impl)
        torch.npu.synchronize()
        assert torch.equal(impl.weight_dq.cpu(), weight[:, :2048])
        assert torch.equal(impl.weight_dkv_kr.cpu(), weight[:, 2048:])
        assert torch.equal(impl.dequant_scale_w_dq.cpu(), scales[:2048].view(1, -1))
        assert torch.equal(impl.dequant_scale_w_dkv_kr.cpu(), scales[2048:].view(1, -1))
        assert torch.equal(source.cpu(), weight)
    finally:
        torch.npu.config.allow_internal_format = previous == b"enable"
