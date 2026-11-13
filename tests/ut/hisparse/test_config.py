# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.hisparse import config as module


def make_config():
    return SimpleNamespace(
        use_v2_model_runner=True,
        attention_config=SimpleNamespace(
            hisparse_config=SimpleNamespace(eager_host_mirror=True), indexer_kv_dtype="int8"
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        kv_transfer_config=SimpleNamespace(kv_connector="HiSparseConnector"),
        additional_config={},
        cache_config=SimpleNamespace(block_size=128, cache_dtype="int8"),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setattr(module, "find_spec", lambda name: object())
    monkeypatch.setattr(module, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A3))
    return make_config()


def test_a3_int8(config):
    module.validate_hisparse_config(config)


@pytest.mark.parametrize("dtype,indexer", [("bfloat16", "bf16"), ("fp8", "fp8"), ("int8", "int8")])
def test_a5_preserves_existing_cache_support(config, monkeypatch, dtype, indexer):
    monkeypatch.setattr(module, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A5))
    config.cache_config.cache_dtype = dtype
    config.attention_config.indexer_kv_dtype = indexer
    if dtype == "int8":
        with pytest.raises(ValueError, match="only on A3"):
            module.validate_hisparse_config(config)
    else:
        module.validate_hisparse_config(config)


@pytest.mark.parametrize("section,field,value,match", [
    ("cache_config", "cache_dtype", "fp8", "INT8"),
    ("model_config", "dtype", torch.float16, "BF16"),
    ("attention_config", "indexer_kv_dtype", "bf16", "INT8 indexer"),
    ("cache_config", "cache_dtype", "bfloat16", "INT8"),
    ("cache_config", "block_size", 64, "block-size"),
    ("parallel_config", "prefill_context_parallel_size", 2, "context parallelism"),
    ("parallel_config", "decode_context_parallel_size", 2, "context parallelism"),
    ("kv_transfer_config", "kv_connector", "SfaRemoteD2HConnector", "HiSparseConnector"),
])
def test_invalid_combination(config, section, field, value, match):
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match=match):
        module.validate_hisparse_config(config)


def test_lazy_mirror(config):
    config.attention_config.hisparse_config.eager_host_mirror = False
    with pytest.raises(ValueError, match="eager_host_mirror"):
        module.validate_hisparse_config(config)


def test_v1_runner(config):
    config.use_v2_model_runner = False
    with pytest.raises(ValueError, match="Model Runner V2"):
        module.validate_hisparse_config(config)


def test_legacy_offload(config):
    config.additional_config = {"sparse_kv_offload_config": {"enabled": True}}
    with pytest.raises(ValueError, match="mutually exclusive"):
        module.validate_hisparse_config(config)


@pytest.mark.parametrize("missing_parent", [False, True])
def test_unsupported_core(config, monkeypatch, missing_parent):
    def lookup(name):
        if missing_parent:
            raise ModuleNotFoundError(name)
        return None

    monkeypatch.setattr(module, "find_spec", lookup)
    with pytest.raises(ValueError, match="pluggable HiSparse backend"):
        module.validate_hisparse_config(config)


def test_disabled_does_not_probe_core(config, monkeypatch):
    config.attention_config.hisparse_config = None
    monkeypatch.setattr(module, "find_spec", lambda _: pytest.fail("unexpected Core lookup"))
    module.validate_hisparse_config(config)


def test_unsupported_hardware(config, monkeypatch):
    monkeypatch.setattr(module, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A2))
    with pytest.raises(ValueError, match="does not support"):
        module.validate_hisparse_config(config)
