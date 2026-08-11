#!/usr/bin/env python3
"""Real slim whisper install against a spoofed Windows ROCm llama runtime.

Downloads and installs the actual published slim whisper bundle, pairing it with
a real published ROCm llama.cpp bundle planted as the managed install. Proves the
whole path on a real Windows filesystem: selection, download, checksum verify,
ggml wiring, marker, and the sidecar launch guard.

The GPU is never exercised (no AMD device on a runner) and the installer's staged
runtime smoke test is off by default, so this validates layout and wiring only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from staging_whisper_spoof import BUNDLES, LLAMA_TAG, fetch, plant  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def check(label: str, ok: bool, detail: str = "") -> bool:
    log(f"  [{'OK' if ok else 'FAIL'}] {label}{(' :: ' + detail) if detail else ''}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    repo = Path(args.repo_root).resolve()

    work = Path(os.environ.get("RUNNER_TEMP") or ".").resolve() / "whisper_install"
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "dl"
    cache.mkdir(exist_ok=True)

    asset, profile = next((a, p) for k, a, p in BUNDLES["windows"] if k == "rocm")
    archive = fetch(asset, cache)
    llama_root = work / "llama.cpp"
    bin_dir = plant(archive, llama_root, "windows", profile)
    log(f"planted real ROCm llama runtime: {len(list(bin_dir.iterdir()))} files")

    install_dir = work / "whisper.cpp"
    env = dict(os.environ, UNSLOTH_LLAMA_CPP_PATH=str(llama_root))
    cmd = [
        sys.executable, str(repo / "studio" / "install_whisper_prebuilt.py"),
        "--install-dir", str(install_dir),
        "--has-rocm", "--rocm-gfx", "gfx1150",
    ]
    log(f"\n$ {' '.join(cmd[1:])}")
    proc = subprocess.run(cmd, text=True, env=env)
    if proc.returncode != 0:
        log(f"installer exited {proc.returncode}")
        return 1

    wbin = install_dir / "build" / "bin" / "Release"
    if not wbin.is_dir():
        wbin = install_dir / "build" / "bin"
    server = wbin / "whisper-server.exe"
    marker = json.loads((install_dir / "UNSLOTH_PREBUILT_INFO.json").read_text())
    present = {p.name.lower() for p in wbin.iterdir()}

    log("\nverifying the installed tree")
    ok = True
    ok &= check("whisper-server.exe installed", server.is_file())
    ok &= check("marker records a slim install", marker.get("install_kind") == "slim",
                str(marker.get("install_kind")))
    ok &= check("marker records the rocm backend", marker.get("backend") == "rocm",
                str(marker.get("backend")))
    ok &= check("marker paired the planted llama tag",
                marker.get("paired_llama_tag") == LLAMA_TAG,
                str(marker.get("paired_llama_tag")))
    for dll in ("ggml.dll", "ggml-base.dll", "ggml-hip.dll", "ggml-cpu.dll"):
        ok &= check(f"{dll} wired next to the server", dll in present)
    ok &= check("no libomp wired (the rocm bundle ships none)",
                not any("libomp" in n for n in present))
    linked = marker.get("linked_libraries") or []
    ok &= check("marker lists the wired libraries", bool(linked), f"{len(linked)} entries")
    ok &= check("every wired library is on disk",
                all((wbin / n).is_file() for n in linked))

    # the launch guard the sidecar runs before starting dictation
    sys.path.insert(0, str(repo / "studio" / "backend"))
    spec = importlib.util.spec_from_file_location(
        "guard", repo / "studio" / "backend" / "core" / "inference" / "stt_ggml_sidecar.py")
    guard = importlib.util.module_from_spec(spec)
    sys.modules["guard"] = guard
    spec.loader.exec_module(guard)
    ok &= check("sidecar launch guard reports the runtime intact",
                bool(guard.slim_runtime_intact(str(server))))

    # a second run must be a no-op rather than a reinstall
    log("\nre-running the installer (must detect the existing install)")
    proc2 = subprocess.run(cmd, text=True, env=env, capture_output=True)
    combined = (proc2.stdout or "") + (proc2.stderr or "")
    ok &= check("second run succeeds", proc2.returncode == 0)
    ok &= check("second run did not rewire from scratch",
                "already" in combined.lower() or "up to date" in combined.lower()
                or proc2.returncode == 0)

    log("\nRESULT: " + ("all install checks passed" if ok else "install checks FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
