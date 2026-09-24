# SPDX-License-Identifier: Apache-2.0
"""Exercise the real vLLM concurrency formula with mixed DS cache geometry.

PageSpec supplies physical byte sizes independently of version-specific MLA
storage fields. The upstream grouping, admission and concurrency methods are
used without mocks.
"""
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.quantization.turboquant import cache


@dataclass(frozen=True, kw_only=True)
class PageSpec(MLAAttentionSpec):
    payload_bytes: int
    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.float16

    @property
    def real_page_size_bytes(self):
        return self.payload_bytes


def mixed_specs(tq, block_size=128):
    cache_dtype = "turboquant_4bit_nc" if tq else "bfloat16"
    c4 = {}
    for i in range(21):
        c4[f"c4.{i}"] = PageSpec(
            block_size=block_size * 4, num_kv_heads=1, head_size=258 if tq else 512,
            dtype=torch.uint8 if tq else torch.bfloat16,
            cache_dtype_str=cache_dtype, payload_bytes=block_size * (258 if tq else 1024),
        )
        c4[f"index.{i}"] = PageSpec(
            block_size=block_size * 4, num_kv_heads=1, head_size=128,
            dtype=torch.int8, cache_dtype_str=cache_dtype,
            payload_bytes=block_size * 130, scale_dim=1,
        )
    c128 = {
        f"c128.{i}": PageSpec(
            block_size=block_size * 128, num_kv_heads=1, head_size=512,
            dtype=torch.bfloat16, cache_dtype_str=cache_dtype, payload_bytes=block_size * 1024,
        ) for i in range(20)
    }
    swa = {
        f"swa.{i}": SlidingWindowMLASpec(
            block_size=block_size, num_kv_heads=1, head_size=512,
            dtype=torch.bfloat16, sliding_window=128,
        ) for i in range(43)
    }
    state4 = {
        f"state4.{i}": SlidingWindowMLASpec(
            block_size=block_size // 16, num_kv_heads=1, head_size=2048,
            dtype=torch.float32, sliding_window=8,
        ) for i in range(21)
    }
    index_state = {
        f"index_state.{i}": SlidingWindowMLASpec(
            block_size=block_size // 16, num_kv_heads=1, head_size=512,
            dtype=torch.float32, sliding_window=8,
        ) for i in range(21)
    }
    state128 = {
        f"state128.{i}": SlidingWindowMLASpec(
            block_size=block_size // 4, num_kv_heads=1, head_size=1024,
            dtype=torch.float32, sliding_window=128,
        ) for i in range(20)
    }
    grouped = [
        UniformTypeKVCacheSpecs.from_specs(specs) for specs in (c4, c128, swa, {**state4, **index_state}, state128)
    ]
    assert all(group is not None for group in grouped)
    return grouped


def test_tq_planner_detects_pipeline_rank_without_c4():
    c128_group = mixed_specs(tq=True)[1]
    groups = cache.group_specs([c128_group])
    assert cache.uses_turboquant_groups(groups)
    assert all(spec.head_size == 512 for group in groups for spec in group.kv_cache_spec.kv_cache_specs.values())
    baseline = cache.group_specs([mixed_specs(tq=False)[1]])
    assert not cache.uses_turboquant_groups(baseline)


@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("max_model_len", [131072, 1048576])
def test_mixed_cache_capacity_and_normalization(block_size, max_model_len):
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        max_in_flight_tokens=8192,
    )
    available_memory = 32 * 1024**3
    capacities = []
    for tq in (False, True):
        grouped = mixed_specs(tq, block_size)
        if tq:
            groups = cache.group_specs(grouped)
            divisor = cache.pool_bytes_per_block(groups)
        else:
            # Existing BF16 planner: 21 tuples, each with a main and index
            # page. State pages are padded to those two canonical widths.
            groups = [KVCacheGroupSpec(layer_names=list(g.kv_cache_specs), kv_cache_spec=g) for g in grouped[:2]]
            for index, group in enumerate(grouped[2:], start=2):
                names = list(group.kv_cache_specs)
                chunk_size = len(names) if index == 3 else 21
                for start in range(0, len(names), chunk_size):
                    specs = {
                        name: replace(
                            group.kv_cache_specs[name],
                            page_size_padded=block_size * (130 if index == 3 and name.startswith("index_state") else 1024),
                        )
                        for name in names[start:start + chunk_size]
                    }
                    groups.append(
                        KVCacheGroupSpec(layer_names=list(specs), kv_cache_spec=UniformTypeKVCacheSpecs.from_specs(specs))
                    )
            divisor = 21 * block_size * (1024 + 130)
        num_blocks = available_memory // divisor
        runtime = SimpleNamespace(num_blocks=num_blocks, kv_cache_groups=groups)
        capacities.append(get_max_concurrency_for_kv_cache_config(config, runtime))
        blocks_per_request = sum(group.kv_cache_spec.max_memory_usage_pages(config) for group in groups)
        assert capacities[-1] == num_blocks / blocks_per_request
        if tq:
            assert cache.max_memory_usage(config, groups) == divisor * blocks_per_request
        # Rank normalization must reproduce the chosen block count exactly.
        if tq:
            assert (num_blocks * divisor) // cache.pool_bytes_per_block(groups) == num_blocks
    assert capacities[1] > capacities[0] * 1.5
    print(f"length={max_model_len}, block={block_size}: baseline={capacities[0]:.3f}x, TQ={capacities[1]:.3f}x")
