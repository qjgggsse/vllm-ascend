# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pack DeepSeek C4 TQ pages using upstream byte-offset cache descriptors.

Each physical slot has the same page stride for every group that shares it.
Block IDs can therefore never alias a different block ID in another group.
"""

import math
from dataclasses import replace

from vllm.logger import logger
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.kv_cache_utils import may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheTensor,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from . import SLOT_BYTES, TURBOQUANT_CACHE_DTYPE


def uses_turboquant_groups(groups):
    # A pipeline rank may own only C128/SWA layers. Its grouping and physical
    # allocation must still use the same TQ planner as ranks containing C4.
    return any(
        getattr(spec, "cache_dtype_str", None) == TURBOQUANT_CACHE_DTYPE
        for group in groups
        for spec in (
            group.kv_cache_spec.kv_cache_specs.values()
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            else (group.kv_cache_spec,)
        )
    )


def group_specs(grouped_specs):
    # Use the largest unpadded page as a byte budget for every logical
    # group. A BF16 layer-count budget would keep reserving BF16-sized pools
    # for small TQ/indexer pages and mask the actual compression benefit.
    normalized = [
        {
            name: replace(spec, page_size_padded=None) if isinstance(spec, SlidingWindowMLASpec) else spec
            for name, spec in group.kv_cache_specs.items()
        }
        for group in grouped_specs
    ]
    budget = max(spec.page_size_bytes for specs in normalized for spec in specs.values())
    groups = []
    for specs in normalized:
        chunk = {}
        used = 0
        for name, spec in specs.items():
            if chunk and used + spec.page_size_bytes > budget:
                merged = UniformTypeKVCacheSpecs.from_specs(chunk)
                assert merged is not None
                groups.append(KVCacheGroupSpec(layer_names=list(chunk), kv_cache_spec=merged))
                chunk, used = {}, 0
            chunk[name] = spec
            used += spec.page_size_bytes
        if chunk:
            merged = UniformTypeKVCacheSpecs.from_specs(chunk)
            assert merged is not None
            groups.append(KVCacheGroupSpec(layer_names=list(chunk), kv_cache_spec=merged))
    return groups


def _alignment(spec):
    alignment = get_dtype_size(spec.dtype)
    if getattr(spec, "scale_dim", 0):
        alignment = math.lcm(alignment, get_dtype_size(spec.scale_dtype))
    if getattr(spec, "cache_dtype_str", None) == TURBOQUANT_CACHE_DTYPE and spec.head_size == SLOT_BYTES:
        # The fused kernel computes the physical row stride from the page
        # stride. Preserve an integral number of compact 258-byte rows.
        alignment = math.lcm(alignment, SLOT_BYTES)
    return alignment


def packed_layout(groups):
    specs = {name: spec for group in groups for name, spec in group.kv_cache_spec.kv_cache_specs.items()}
    page_alignment = math.lcm(*(_alignment(spec) for spec in specs.values()))
    page_limit = round_up(max(spec.page_size_bytes for spec in specs.values()), page_alignment)
    group_slots = []
    for group in groups:
        slots, used = [], []
        for name in sorted(group.layer_names, key=lambda name: -specs[name].page_size_bytes):
            spec = specs[name]
            for index, size in enumerate(used):
                offset = round_up(size, _alignment(spec))
                if offset + spec.page_size_bytes <= page_limit:
                    slots[index][name] = offset
                    used[index] = offset + spec.page_size_bytes
                    break
            else:
                slots.append({name: 0})
                used.append(spec.page_size_bytes)
        group_slots.append(slots)
    result = []
    for index in range(max(map(len, group_slots))):
        offsets = {
            name: offset for slots in group_slots if index < len(slots) for name, offset in slots[index].items()
        }
        size = max(offset + specs[name].page_size_bytes for name, offset in offsets.items())
        alignment = math.lcm(*(_alignment(specs[name]) for name in offsets))
        result.append((round_up(size, alignment), offsets))
    return result


def pool_bytes_per_block(groups):
    return sum(size for size, _ in packed_layout(groups))


def max_memory_usage(vllm_config, groups):
    request_blocks = sum(group.kv_cache_spec.max_memory_usage_pages(vllm_config) for group in groups)
    return pool_bytes_per_block(groups) * request_blocks


def cache_config(vllm_config, groups, available_memory):
    layout = packed_layout(groups)
    page_bytes = sum(size for size, _ in layout)
    num_blocks = may_override_num_blocks(vllm_config, available_memory // page_bytes)
    logger.info(
        "DeepSeek V4 TurboQuant KV: C4 slot=%d bytes, pool bytes/block=%d, groups=%d, blocks=%d",
        SLOT_BYTES,
        page_bytes,
        len(groups),
        num_blocks,
    )
    backing_size = num_blocks * page_bytes
    descriptors = []
    slot_base = 0
    for stride, offsets in layout:
        for name, offset in offsets.items():
            descriptors.append(
                KVCacheTensor(
                    size=backing_size,
                    layers=[name],
                    offset=slot_base + offset,
                    layer_stride=0,
                    block_stride=stride,
                )
            )
        slot_base += stride * num_blocks
    return num_blocks, descriptors
