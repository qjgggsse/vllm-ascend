#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify that vendored TurboQuant operators are byte-identical to their sources."""
import argparse
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    pairs = (
        ("ops-nn/quant/turbo_quant", "csrc/quant/turbo_quant"),
        ("ops-transformer/attention/sparse_flash_mla", "csrc/attention/sparse_flash_mla"),
        ("ops-transformer/attention/mixed_quant_sparse_flash_mla", "csrc/attention/mixed_quant_sparse_flash_mla"),
        (
            "ops-transformer/attention/mixed_quant_sparse_flash_mla_metadata",
            "csrc/attention/mixed_quant_sparse_flash_mla_metadata",
        ),
    )
    for source, target in pairs:
        source, target = args.source_root / source, repo / target
        files = {p.relative_to(source) for p in source.rglob("*") if p.is_file()}
        copied = {p.relative_to(target) for p in target.rglob("*") if p.is_file()}
        if files != copied:
            raise SystemExit(f"File set mismatch: {source} -> {target}: {files ^ copied}")
        digest = hashlib.sha256()
        for path in sorted(files):
            data = (source / path).read_bytes()
            if data != (target / path).read_bytes():
                raise SystemExit(f"Content mismatch: {path}")
            digest.update(str(path).encode() + b"\0" + data)
        print(f"{target.relative_to(repo)}: {len(files)} files identical; sha256={digest.hexdigest()}")

    # These shared headers are required by the operators but live outside
    # their source directories. Keep the standalone package self-contained.
    dependencies = (
        ("ops-nn/common/inc/error_util.h", "csrc/common/include/error_util.h"),
    )
    for source, target in dependencies:
        if (args.source_root / source).read_bytes() != (repo / target).read_bytes():
            raise SystemExit(f"Shared dependency mismatch: {source} -> {target}")
    print(f"{len(dependencies)} shared dependency headers identical")


if __name__ == "__main__":
    main()
