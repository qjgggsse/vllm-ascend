# SPDX-License-Identifier: Apache-2.0
"""Host durability fences must survive drafting, fallback and exceptional exits."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.hisparse.host_mirror import (
    HostMirrorConnector,
    _draft_groups_disjoint,
    _host_disjoint,
)


@pytest.fixture
def mirror(monkeypatch):
    compute, output = Mock(), Mock()
    monkeypatch.setattr(torch.accelerator, "current_stream", lambda: compute)
    worker = NS(
        _forward_ready_event=Mock(), host_write_event=Mock(),
        _pending_dma_descriptors=[], _dma_submitted=False,
        _release_completed_dma_descriptors=Mock(),
    )

    def submit(ready):
        worker._pending_dma_descriptors.append(object())
        worker._dma_submitted = True

    worker._finish_mirror_phase = Mock(side_effect=submit)
    connector = Mock()
    connector._disabled = False
    sources = Mock()
    sources.can_defer.return_value = True
    wrapped = HostMirrorConnector(connector, worker, output, sources)
    wrapped.begin_execution(True)
    wrapped.pre_forward(NS(has_sync_kv_loads=True), NS(has_prefill=False, num_tokens=4,
                                                     num_tokens_after_padding=4), attn_metadata={})
    return wrapped, worker, connector, compute, output


@pytest.mark.parametrize("exit_path", ["finalize", "next_step", "no_forward", "disable", "reset"])
def test_deferred_mirror_joins_before_reuse(mirror, exit_path):
    wrapped, worker, connector, compute, output = mirror
    wrapped.finish_forward()
    output.wait_event.assert_called_once_with(worker.host_write_event)
    compute.wait_event.assert_not_called()
    wrapped.post_forward(set(), defer_finalize=True)
    compute.wait_event.assert_not_called()
    if exit_path == "finalize":
        wrapped.finalize_step()
    elif exit_path == "next_step":
        wrapped.begin_execution(True)
    elif exit_path == "no_forward":
        wrapped.no_forward(NS())
    elif exit_path == "disable":
        wrapped.set_disabled(True)
    else:
        wrapped.reset_capture_state()
    compute.wait_event.assert_called_once_with(worker.host_write_event)
    wrapped.join_pending()
    compute.wait_event.assert_called_once()
    assert wrapped._pending_event is None


@pytest.mark.parametrize("reason", ["prefill", "padding", "profile", "ownership", "disabled"])
def test_ineligible_mirror_uses_core_fence(mirror, reason):
    wrapped, worker, connector, compute, output = mirror
    wrapped.begin_execution(reason != "profile")
    wrapped.pre_forward(NS(has_sync_kv_loads=True), NS(has_prefill=reason == "prefill", num_tokens=3,
                        num_tokens_after_padding=4 if reason == "padding" else 3), attn_metadata={})
    wrapped._sources.can_defer.return_value = reason != "ownership"
    connector._disabled = reason == "disabled"
    wrapped.finish_forward()
    connector.finish_forward.assert_called_once()
    worker._finish_mirror_phase.assert_not_called()
    output.wait_event.assert_not_called()


def test_partial_mirror_submission_is_joined_on_failure(mirror):
    wrapped, worker, connector, compute, output = mirror

    def fail(ready):
        worker._dma_submitted = True
        raise RuntimeError("partial submission")

    worker._finish_mirror_phase.side_effect = fail
    with pytest.raises(RuntimeError, match="partial submission"):
        wrapped.finish_forward()
    compute.wait_event.assert_called_once_with(worker.host_write_event)
    assert wrapped._pending_event is None


def test_mirror_requires_disjoint_host_storage_and_draft_groups():
    storage = torch.empty(32, 8)
    assert _host_disjoint([storage[:16], storage[16:]], {1})
    assert not _host_disjoint([storage[:17], storage[16:]], {1})
    target, draft = NS(), NS()
    handles = [NS(runtime=target), NS(runtime=draft)]
    draft.index_group = NS(leader=draft, followers=[])
    assert _draft_groups_disjoint(handles, {1})
    draft.index_group.followers.append(target)
    assert not _draft_groups_disjoint(handles, {1})
