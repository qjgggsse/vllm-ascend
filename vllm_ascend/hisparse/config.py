# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the HiSparse contract without creating an NPU context."""

from importlib.util import find_spec

import torch

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile


def validate_hisparse_config(config) -> None:
    hisparse = config.attention_config.hisparse_config
    if hisparse is None:
        return
    try:
        has_backend = find_spec("vllm.v1.hisparse.backend") is not None
    except ModuleNotFoundError:
        has_backend = False
    if not has_backend:
        raise ValueError(
            "Ascend HiSparse requires Core's pluggable HiSparse backend. Install the vLLM revision in "
            ".github/vllm-main-verified.commit; older CUDA-only HiSparse runtimes are not supported."
        )
    profile = get_current_hardware_profile()
    if not profile.supports(HardwareCapability.HISPARSE_RESIDENCY):
        raise ValueError("This Ascend build does not support the HiSparse residency backend")
    if not config.use_v2_model_runner:
        raise ValueError("Ascend HiSparse requires Model Runner V2")
    if not hisparse.eager_host_mirror:
        raise ValueError("Ascend HiSparse requires eager_host_mirror=true")
    parallel = config.parallel_config
    if parallel.prefill_context_parallel_size != 1 or parallel.decode_context_parallel_size != 1:
        raise ValueError("Ascend HiSparse does not support context parallelism")
    transfer = config.kv_transfer_config
    if transfer is None or transfer.kv_connector != "HiSparseConnector":
        raise ValueError("Ascend HiSparse requires HiSparseConnector without external P/D transport")
    if config.additional_config.get("sparse_kv_offload_config", {}).get("enabled", False):
        raise ValueError("HiSparse and sparse_kv_offload_config are mutually exclusive")
    if config.cache_config.block_size not in (None, 128):
        raise ValueError("Ascend HiSparse requires block-size 128")
    if not profile.supports(HardwareCapability.FP8_ATTENTION):
        cache_dtype = config.cache_config.cache_dtype
        if cache_dtype != "int8" or config.model_config.dtype != torch.bfloat16:
            raise ValueError("A3 HiSparse requires INT8 main KV and BF16 model activations")
        if config.attention_config.indexer_kv_dtype != "int8":
            raise ValueError("A3 HiSparse requires an INT8 indexer cache")
    elif config.cache_config.cache_dtype == "int8" or config.attention_config.indexer_kv_dtype == "int8":
        raise ValueError("A5 HiSparse retains its BF16/FP8 cache support; INT8 HiSparse is enabled only on A3")
