# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Padding-safe writes for the paged caches used alongside TurboQuant."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["tokens"])
def _write_cache(
    cache,
    updates,
    slots,
    tokens,
    width: tl.constexpr,
    pages: tl.constexpr,
    page_size: tl.constexpr,
    page_stride: tl.constexpr,
    row_stride: tl.constexpr,
    update_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    slot_column_stride: tl.constexpr,
    paired: tl.constexpr,
    columns: tl.constexpr,
):
    columns_idx = tl.arange(0, columns)
    for token in range(tl.program_id(0), tokens, tl.num_programs(0)):
        first = tl.load(slots + token * slot_stride).to(tl.int64)
        if paired:
            block = first
            offset = tl.load(slots + token * slot_stride + slot_column_stride).to(tl.int64)
        else:
            block = first // page_size
            offset = first % page_size
        if (first >= 0) & (block < pages) & (offset >= 0) & (offset < page_size):
            value = tl.load(updates + token * update_stride + columns_idx, columns_idx < width, other=0)
            address = block * page_stride + offset * row_stride + columns_idx
            tl.store(cache + address, value, columns_idx < width)


def write_cache(cache: torch.Tensor, updates: torch.Tensor, slots: torch.Tensor) -> None:
    """Write unique valid slots, preserving page strides and skipping padding."""
    if cache.ndim != 4 or cache.shape[2] != 1 or cache.stride(-1) != 1:
        raise ValueError("Cache must be [pages, page_size, 1, width] with contiguous rows")
    if updates.ndim not in (2, 3) or (updates.ndim == 3 and updates.shape[1] != 1):
        raise ValueError("Updates must be [tokens, width] or [tokens, 1, width]")
    if updates.shape[-1] != cache.shape[-1] or updates.stride(-1) != 1 or updates.dtype != cache.dtype:
        raise ValueError("Update rows must match the cache width and dtype")
    if slots.ndim not in (1, 2) or (slots.ndim == 2 and slots.shape[1] != 2):
        raise ValueError("Slots must be flat or [tokens, 2] block/offset pairs")
    if slots.shape[0] != updates.shape[0] or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("Slots must have one int32/int64 entry per update row")
    if not (cache.device == updates.device == slots.device):
        raise ValueError("Cache, updates and slots must be on the same device")
    tokens = updates.shape[0]
    if tokens == 0:
        return
    _write_cache[(min(get_vectorcore_num(), tokens),)](
        cache,
        updates,
        slots,
        tokens,
        cache.shape[-1],
        cache.shape[0],
        cache.shape[1],
        cache.stride(0),
        cache.stride(1),
        updates.stride(0),
        slots.stride(0),
        slots.stride(1) if slots.ndim == 2 else 0,
        slots.ndim == 2,
        triton.next_power_of_2(cache.shape[-1]),
        multibuffer=False,
    )
