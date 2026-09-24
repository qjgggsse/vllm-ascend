# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapt DSA arguments to the copied MixedQuantSparseFlashMla operators."""

import torch

from vllm_ascend.attention.sparse_flash_mla import (
    _add_compressed_kv_lengths,
    _drop_paged_kv_cu_seqlens,
    _ensure_sinks,
)


def _adapt(kwargs):
    kwargs.pop("device", None)
    kwargs.pop("kv_quant_mode", None)
    kwargs.pop("tile_size", None)
    kwargs.update(quant_mode=3, rope_head_dim=64, layout_kv="PA_BBND")
    if "seqused_kv" in kwargs:
        kwargs["seqused_ori_kv"] = kwargs.pop("seqused_kv")
    if "max_seqlen_kv" in kwargs:
        kwargs["max_seqlen_ori_kv"] = kwargs.pop("max_seqlen_kv")
    _drop_paged_kv_cu_seqlens(kwargs)
    _add_compressed_kv_lengths(kwargs)


def mixed_quant_sparse_flash_mla_metadata(**kwargs):
    _adapt(kwargs)
    return torch.ops._C_ascend.mixed_quant_sparse_flash_mla_metadata(**kwargs)


def mixed_quant_sparse_flash_mla(q, **kwargs):
    _adapt(kwargs)
    _ensure_sinks(kwargs)
    return torch.ops._C_ascend.mixed_quant_sparse_flash_mla(q, **kwargs)
