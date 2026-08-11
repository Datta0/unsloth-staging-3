#!/usr/bin/env python3
"""Resolve and install studio/backend/requirements on the host platform.

Staging-CI only: proves every pin in studio/backend/requirements has an
installable candidate on this OS + interpreter, and that the torch-free
install path actually builds and imports.

Three phases, mirroring install_python_stack.py's own flags:
  1. resolve  -- uv pip compile every requirements file (--no-deps where the
                 installer uses it, -c constraints.txt where it constrains).
  2. dry-run  -- full dependency resolution for the two with-deps files.
  3. install  -- real venv for the no-torch path, then an import smoke test.

usage: python check_requirement_pins.py <python-version>
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REQ = REPO / "studio" / "backend" / "requirements"
CONSTRAINTS = REQ / "single-env" / "constraints.txt"
MLX_OVERRIDES = REQ / "single-env" / "overrides-darwin-arm64.txt"

IS_MAC_ARM = sys.platform == "darwin" and os.uname().machine == "arm64" if hasattr(os, "uname") else False

# (label, file, no_deps) exactly as install_python_stack.py applies them.
FILES = [
    ("extras",           REQ / "extras.txt",                            False),
    ("extras-no-deps",   REQ / "extras-no-deps.txt",                    True),
    ("studio",           REQ / "studio.txt",                            False),
    ("no-torch-runtime", REQ / "no-torch-runtime.txt",                  True),
    ("dd-deps",          REQ / "single-env" / "data-designer-deps.txt", False),
    ("data-designer",    REQ / "single-env" / "data-designer.txt",      True),
    ("diffusers-pin",    REQ / "diffusers-pin.txt",                     False),
]

IMPORTS = [
    "fastapi", "uvicorn", "pydantic", "pandas", "gguf", "typer", "transformers",
    "numpy", "PIL", "yaml", "structlog", "pymupdf", "docx", "jwt", "diceware", "av",
]

failures: list[str] = []


def run(label: str, cmd: list[str], *, soft: bool = False) -> bool:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    ok = proc.returncode == 0
    if not ok and not soft:
        failures.append(label)
    print(f"--> {label}: {'ok' if ok else ('SOFT-FAIL' if soft else 'FAIL')}", flush=True)
    return ok


def uv(*args: str) -> list[str]:
    return [shutil.which("uv") or "uv", *args]


def main() -> int:
    py = sys.argv[1] if len(sys.argv) > 1 else f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"::group::environment\npython={py} platform={sys.platform} repo={REPO}\n::endgroup::", flush=True)

    common = ["-c", str(CONSTRAINTS)]
    if IS_MAC_ARM:
        common += ["--override", str(MLX_OVERRIDES)]

    print("\n==================== phase 1: resolve every file ====================", flush=True)
    for label, path, no_deps in FILES:
        cmd = uv("pip", "compile", "-q", "--no-header", "--no-annotate",
                 "--python-version", py, str(path), *common)
        if no_deps:
            cmd.append("--no-deps")
        run(f"resolve/{label}", cmd)

    print("\n==================== phase 2: full dep dry-run ====================", flush=True)
    venv = REPO / ".pinvenv"
    if not run("venv", uv("venv", "--python", py, str(venv))):
        return 1
    vpy = venv / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    for label in ("extras", "studio"):
        path = dict((f[0], f[1]) for f in FILES)[label]
        run(f"dry-run/{label}",
            uv("pip", "install", "-p", str(vpy), "--dry-run", "--no-cache-dir", "-r", str(path), *common))

    print("\n==================== phase 3: real no-torch install ====================", flush=True)
    # install_python_stack.py resolves pydantic WITH deps before the --no-deps file.
    run("install/pydantic", uv("pip", "install", "-p", str(vpy), "--no-cache-dir", "pydantic", *common))
    run("install/no-torch-runtime",
        uv("pip", "install", "-p", str(vpy), "--no-cache-dir", "--no-deps",
           "-r", str(REQ / "no-torch-runtime.txt"), *common))
    run("install/studio",
        uv("pip", "install", "-p", str(vpy), "--no-cache-dir", "-r", str(REQ / "studio.txt"), *common))

    print("\n==================== phase 4: import smoke ====================", flush=True)
    snippet = (
        "import importlib, sys\n"
        f"mods = {IMPORTS!r}\n"
        "bad = []\n"
        "for m in mods:\n"
        "    try: importlib.import_module(m)\n"
        "    except Exception as e: bad.append((m, type(e).__name__ + ': ' + str(e)[:160]))\n"
        "print('IMPORT FAILURES:', bad or 'none')\n"
        "sys.exit(1 if bad else 0)\n"
    )
    run("import-smoke", [str(vpy), "-c", snippet])

    print("\n==================== summary ====================", flush=True)
    if failures:
        print(f"FAILED steps ({len(failures)}): {', '.join(failures)}", flush=True)
        return 1
    print("all steps passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
