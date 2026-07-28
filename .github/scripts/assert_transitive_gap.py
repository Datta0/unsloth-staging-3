#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Decide, from probe facts, whether the real-import check earns its place.

The state under test is a venv whose every requirement line is satisfied and
whose install manifest is complete, but which is missing a transitively
installed package the backend imports at module scope.

Three things have to hold for the answer to mean anything:

1. the backend really is broken -- otherwise nothing was proven;
2. `verify-install` (requirement lines + manifest) reports ok, i.e. it is blind
    to this class;
3. `desktop-runtime-check` (real import) reports failed, i.e. it is not.

(2) is reported rather than enforced: if a future change teaches the manifest
path to see this too, that is good news, not a regression, and the job should
say so instead of failing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <verdict.json>", file=sys.stderr)
        return 2

    data = json.loads(Path(argv[1]).read_text())
    facts = data.get("facts", data)
    verify = facts.get("verify_install")
    runtime = facts.get("desktop_runtime_check")
    backend_ok = facts.get("backend_ok")

    print(f"[gap] verify_install         = {verify}")
    print(f"[gap] desktop_runtime_check  = {runtime}")
    print(f"[gap] backend_ok             = {backend_ok}")
    print(f"[gap] verdict                = {data.get('verdict')}")

    if backend_ok is not False:
        print("::error::pruning the transitive dep did not stop the backend booting; "
              "the premise of this job is wrong, so neither check is being tested")
        return 1

    if runtime == "absent":
        print("::error::desktop-runtime-check is not present in this tree, so this job "
              "cannot answer the question it exists to answer")
        return 1

    if runtime != "failed":
        print(f"::error::desktop-runtime-check did NOT catch a missing transitive "
              f"dependency (reported {runtime!r}) even though the backend cannot boot")
        return 1

    if verify == "ok":
        print("CONFIRMED: the manifest/requirements check reports a healthy install "
              "while the backend cannot import. Only the real-import check sees it.")
    else:
        print(f"::warning::verify-install also caught this (reported {verify!r}); the "
              f"two checks overlap more than the requirement-line reasoning predicts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
