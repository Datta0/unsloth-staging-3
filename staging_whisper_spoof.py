#!/usr/bin/env python3
"""Real-runner whisper.cpp prebuilt pairing check for unslothai/unsloth#8379.

CI runners have no AMD or NVIDIA GPU, so the accelerator is spoofed two ways:

* the hardware hints the installer already exposes for setup
  (``--has-rocm`` / ``--rocm-gfx`` / ``--backend``), and
* the llama.cpp side, by planting a REAL published llama bundle as the managed
  install (``UNSLOTH_LLAMA_CPP_PATH``) so the pairing sees a genuine ROCm, CUDA
  or Vulkan runtime layout on a genuine filesystem.

That covers what a Linux-hosted simulation cannot: real Windows path semantics,
the build/bin/Release layout, and case-insensitive globbing on NTFS.

Exits non-zero on the first expectation that does not hold.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

LLAMA_REPO = "unslothai/llama.cpp"
LLAMA_TAG = "b10333-mix-e34b418"
DL = f"https://github.com/{LLAMA_REPO}/releases/download/{LLAMA_TAG}"

# (key, asset, published bundle_profile)  -- profiles as the llama manifest carries
# them; the windows rocm artifacts genuinely carry none.
BUNDLES = {
    "windows": [
        ("cpu", f"app-{LLAMA_TAG}-windows-x64-cpu.zip", "windows-cpu-x64"),
        ("vulkan", f"app-{LLAMA_TAG}-windows-x64-vulkan.zip", "windows-vulkan-x64"),
        ("cuda", f"app-{LLAMA_TAG}-windows-x64-cuda12-legacy.zip", "cuda12-legacy"),
        ("rocm", f"app-{LLAMA_TAG}-windows-x64-rocm-gfx1150.zip", ""),
    ],
    "linux": [
        ("cpu", f"app-{LLAMA_TAG}-linux-x64-cpu.tar.gz", "linux-cpu-x64"),
        ("vulkan", f"app-{LLAMA_TAG}-linux-x64-vulkan.tar.gz", "linux-vulkan-x64"),
        ("cuda", f"app-{LLAMA_TAG}-linux-x64-cuda12-legacy.tar.gz", "cuda12-legacy"),
    ],
    "macos": [
        ("metal", f"llama-{LLAMA_TAG}-bin-macos-arm64.tar.gz", "macos-arm64"),
    ],
}

# (label, bundle, spoof args, mutations, expect_available, expect_backend)
# mutations: (files to delete, files to create) inside the planted bin dir.
CHECKS = {
    "windows": [
        ("rocm bundle, rocm spoof", "rocm", ["--has-rocm", "--rocm-gfx", "gfx1150"],
         ((), ()), True, "rocm"),
        ("rocm bundle, forced cpu", "rocm", ["--backend", "cpu"], ((), ()), True, "cpu"),
        ("cuda bundle, forced cuda", "cuda", ["--backend", "cuda"], ((), ()), True, "cuda"),
        ("cuda bundle, auto (no gpu on runner)", "cuda", ["--backend", "auto"],
         ((), ()), True, "cpu"),
        ("vulkan bundle, forced vulkan", "vulkan", ["--backend", "vulkan"],
         ((), ()), True, "vulkan"),
        ("cpu bundle, forced cpu", "cpu", ["--backend", "cpu"], ((), ()), True, "cpu"),
        # a cpu llama install on an AMD box must fall back, not claim rocm
        ("cpu bundle, rocm spoof falls back", "cpu", ["--has-rocm", "--rocm-gfx", "gfx1150"],
         ((), ()), True, "cpu"),
        # damage: the cpu bundle's ggml really imports libomp
        ("cpu bundle minus libomp", "cpu", ["--backend", "cpu"],
         (("libomp140.x86_64.dll",), ()), False, None),
        # damage: rocm runtime without its ggml backend module
        ("rocm bundle minus ggml-hip", "rocm", ["--has-rocm", "--rocm-gfx", "gfx1150"],
         (("ggml-hip.dll",), ()), False, None),
        # The glob fix in isolation: libomp is present so the soname gate cannot
        # do the rejecting. ROCm must be refused for want of the ggml module; the
        # CPU retry then legitimately succeeds on this runtime's ggml-cpu.dll,
        # so the proof is that cpu is chosen and rocm is never claimed.
        ("rocm bundle minus ggml-hip, libomp present", "rocm",
         ["--has-rocm", "--rocm-gfx", "gfx1150"],
         (("ggml-hip.dll",), ("libomp140.x86_64.dll",)), True, "cpu"),
    ],
    "linux": [
        ("cpu bundle, forced cpu", "cpu", ["--backend", "cpu"], ((), ()), True, "cpu"),
        ("cuda bundle, forced cuda", "cuda", ["--backend", "cuda"], ((), ()), True, "cuda"),
        ("vulkan bundle, forced vulkan", "vulkan", ["--backend", "vulkan"],
         ((), ()), True, "vulkan"),
        ("cuda bundle minus ggml-cuda falls back", "cuda", ["--backend", "cuda"],
         (("libggml-cuda.so",), ()), True, "cpu"),
    ],
    "macos": [
        ("metal bundle, auto", "metal", ["--backend", "auto"], ((), ()), True, "metal"),
        ("metal bundle, forced cpu", "metal", ["--backend", "cpu"], ((), ()), True, "cpu"),
    ],
}


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch(asset: str, dest: Path) -> Path:
    out = dest / asset
    if out.exists():
        return out
    log(f"  downloading {asset}")
    req = urllib.request.Request(f"{DL}/{asset}", headers={"User-Agent": "unsloth-ci"})
    with urllib.request.urlopen(req) as r, open(out, "wb") as f:
        shutil.copyfileobj(r, f)
    return out


def plant(archive: Path, root: Path, os_token: str, profile: str) -> Path:
    """Lay a real llama bundle out as a managed install and write its marker."""
    if root.exists():
        shutil.rmtree(root)
    bin_dir = root / "build" / "bin"
    if os_token == "windows":
        bin_dir = bin_dir / "Release"
    bin_dir.mkdir(parents=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(bin_dir)
    else:
        with tarfile.open(archive) as t:
            try:
                t.extractall(bin_dir, filter="data")
            except TypeError:  # filter= predates 3.11.4
                t.extractall(bin_dir)
    # some archives carry a single top-level dir; flatten it
    entries = list(bin_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for item in list(inner.iterdir()):
            shutil.move(str(item), str(bin_dir / item.name))
        inner.rmdir()
    # llama sometimes nests build/bin inside the archive
    nested = bin_dir / "build" / "bin"
    if nested.is_dir():
        for item in list(nested.iterdir()):
            shutil.move(str(item), str(bin_dir / item.name))
    marker = {"release_tag": LLAMA_TAG, "component": "llama.cpp"}
    if profile:
        marker["bundle_profile"] = profile
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps(marker))
    return bin_dir


def mutate(bin_dir: Path, drop, add) -> None:
    for name in drop:
        for path in list(bin_dir.glob(name)):
            path.unlink()
    for name in add:
        (bin_dir / name).write_bytes(b"planted")


def resolve(repo_root: Path, spoof: list[str], env: dict) -> dict:
    cmd = [sys.executable, str(repo_root / "studio" / "install_whisper_prebuilt.py"),
           "--resolve-prebuilt", "--output-format", "json", *spoof]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"prebuilt_available": False, "_stdout": proc.stdout[-400:],
                "_stderr": proc.stderr[-400:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    repo_root = Path(args.repo_root).resolve()

    system = platform.system()
    os_token = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}[system]
    log(f"== whisper prebuilt spoof matrix on {system} ({platform.machine()}) ==")

    work = Path(os.environ.get("RUNNER_TEMP") or ".").resolve() / "whisper_spoof"
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "dl"
    cache.mkdir(exist_ok=True)

    archives, profiles = {}, {}
    for key, asset, profile in BUNDLES[os_token]:
        archives[key] = fetch(asset, cache)
        profiles[key] = profile

    failures = []
    log(f"\n{'check':<46}{'expected':<22}{'actual'}")
    for label, bundle, spoof, (drop, add), exp_ok, exp_backend in CHECKS[os_token]:
        root = work / f"llama_{bundle}"
        bin_dir = plant(archives[bundle], root, os_token, profiles[bundle])
        mutate(bin_dir, drop, add)
        env = dict(os.environ, UNSLOTH_LLAMA_CPP_PATH=str(root))
        payload = resolve(repo_root, spoof, env)
        ok = bool(payload.get("prebuilt_available"))
        backend = payload.get("backend")
        expected = f"{'available' if exp_ok else 'unavailable'}" + (
            f"/{exp_backend}" if exp_backend else "")
        actual = f"{'available' if ok else 'unavailable'}" + (f"/{backend}" if ok else "")
        good = ok == exp_ok and (not exp_ok or backend == exp_backend)
        log(f"{label:<46}{expected:<22}{actual}  {'OK' if good else 'MISMATCH'}")
        if not good:
            failures.append((label, expected, actual, payload))

    if failures:
        log("\n== FAILURES ==")
        for label, expected, actual, payload in failures:
            log(f"  {label}: expected {expected}, got {actual}")
            log(f"    payload: {json.dumps(payload)[:500]}")
        return 1
    log(f"\nall {len(CHECKS[os_token])} spoofed pairings behaved as expected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
