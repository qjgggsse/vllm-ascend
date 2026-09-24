# SPDX-License-Identifier: Apache-2.0
import re
from pathlib import Path

import torch


def test_dispatcher_schemas_parse():
    source = Path(__file__).resolve().parents[4] / "csrc/aclnn_torch_adapter/turboquant.cpp"
    schemas = re.findall(r'R"schema\((.*?)\)schema"', source.read_text(), re.DOTALL)
    assert len(schemas) == 3
    for schema in schemas:
        parsed = torch._C.parse_schema(schema)
        assert parsed.name in {"turbo_quant", "mixed_quant_sparse_flash_mla", "mixed_quant_sparse_flash_mla_metadata"}
