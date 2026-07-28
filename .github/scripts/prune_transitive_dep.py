#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Remove one installed distribution from the managed venv, in place.

Used to construct the state a requirements-line check cannot detect: a package
that no requirements file names (it arrives transitively) but that the backend
imports at module scope. `starlette` is the canonical one -- it comes in via
fastapi and is imported at studio/backend/main.py:296.

Deliberately not `uv pip uninstall`: that is a clean, recorded removal. What an
interrupted or partially rolled back install leaves is a missing package tree,
so remove the tree and its dist-info directly.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def site_packages(venv: Path) -> Path:
    # Windows keeps site-packages under Lib\, POSIX under lib/pythonX.Y/.
    for pattern in ("lib/python*/site-packages", "Lib/site-packages"):
        for candidate in venv.glob(pattern):
            return candidate
    raise SystemExit(f"no site-packages under {venv}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <distribution-name>", file=sys.stderr)
        return 2
    name = argv[1]

    home = os.environ.get("UNSLOTH_STUDIO_HOME")
    if not home:
        raise SystemExit("UNSLOTH_STUDIO_HOME is not set")
    venv = Path(home) / "unsloth_studio"
    if not venv.is_dir():
        raise SystemExit(f"no managed venv at {venv}")

    sp = site_packages(venv)
    targets = sorted(sp.glob(name)) + sorted(sp.glob(f"{name}-*.dist-info"))
    if not targets:
        # Not a soft failure: if it was never installed, the whole premise of the
        # job is wrong and a later "the backend still boots" would be misread as
        # the runtime check being unnecessary.
        print(f"::error::{name} is not installed under {sp}; nothing to prune")
        return 1

    for path in targets:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
        print(f"[prune] removed {path.relative_to(sp)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
