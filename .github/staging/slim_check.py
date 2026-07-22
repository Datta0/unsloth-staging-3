#!/usr/bin/env python3
"""Slim whisper + llama prebuilt validation driver (trimmed from
whisper.cpp scripts/validate_bundle.py, adapted for an installed
UNSLOTH_STUDIO_HOME layout instead of an extracted bundle dir).

Checks, in order:
  1. llama marker UNSLOTH_PREBUILT_INFO.json exists and carries release_tag.
  2. whisper marker UNSLOTH_WHISPER_PREBUILT_INFO.json has
     install_kind == "slim" and paired_llama_tag == llama release_tag.
  3. ggml runtime objects are present in the whisper bin dir (the wiring
     the installer performs from the llama bin dir).
  4. whisper-server starts from the whisper bin dir (bin dir on
     LD_LIBRARY_PATH / DYLD_LIBRARY_PATH / PATH) and a real multipart
     POST /inference transcription succeeds for every --clip; when a clip
     carries an expected phrase (path::phrase) the transcript must contain
     it case-insensitively.

Exit 0 = valid. Prints VERDICT lines the workflow greps.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def bin_dir(install_dir: Path) -> Path:
    d = install_dir / "build" / "bin"
    if os.name == "nt":
        d = d / "Release"
    return d


def child_env(bdir: Path) -> dict:
    env = dict(os.environ)
    key = {"Linux": "LD_LIBRARY_PATH", "Darwin": "DYLD_LIBRARY_PATH"}.get(platform.system())
    if key:
        env[key] = os.pathsep.join([str(bdir), env.get(key, "")]).rstrip(os.pathsep)
    env["PATH"] = os.pathsep.join([str(bdir), env.get("PATH", "")])
    return env


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post_inference(base: str, audio: Path) -> str:
    boundary = "----unslothslimcheck"
    parts = []
    for field, value in (("temperature", "0.0"), ("response_format", "json"),
                         ("beam_size", "1"), ("language", "en")):
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                     f"name=\"{field}\"\r\n\r\n{value}\r\n")
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{audio.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n")
    body = b"".join(p.encode() for p in parts) + head.encode() + audio.read_bytes() \
        + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(base + "/inference", data=body, method="POST",
                                 headers={"Content-Type":
                                          f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        if resp.status != 200:
            sys.exit(f"FAIL: /inference returned HTTP {resp.status} for {audio.name}")
        payload = json.loads(resp.read())
    return (payload.get("text") or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True, type=Path,
                    help="UNSLOTH_STUDIO_HOME containing llama.cpp/ and whisper.cpp/")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--clip", action="append", default=[],
                    help="audio path, optionally path::expected phrase; repeatable")
    ap.add_argument("--no-gpu", action="store_true", default=True,
                    help="run the server request path with --no-gpu (CPU backend)")
    ap.add_argument("--gpu", dest="no_gpu", action="store_false",
                    help="allow the default (GPU-capable) path")
    args = ap.parse_args()

    llama_dir = args.home / "llama.cpp"
    whisper_dir = args.home / "whisper.cpp"

    lmarker = llama_dir / "UNSLOTH_PREBUILT_INFO.json"
    wmarker = whisper_dir / "UNSLOTH_WHISPER_PREBUILT_INFO.json"
    if not lmarker.is_file():
        sys.exit(f"FAIL: llama marker missing: {lmarker}")
    if not wmarker.is_file():
        sys.exit(f"FAIL: whisper marker missing: {wmarker}")
    lmeta = json.loads(lmarker.read_text())
    wmeta = json.loads(wmarker.read_text())
    llama_tag = lmeta.get("release_tag")
    print(f"llama release_tag = {llama_tag!r} asset = {lmeta.get('asset')!r}")
    print(f"whisper install_kind = {wmeta.get('install_kind')!r} "
          f"release_tag = {wmeta.get('release_tag')!r} asset = {wmeta.get('asset')!r} "
          f"paired_llama_tag = {wmeta.get('paired_llama_tag')!r}")
    if wmeta.get("install_kind") != "slim":
        sys.exit(f"FAIL: whisper install_kind is {wmeta.get('install_kind')!r}, expected 'slim'")
    if not llama_tag or wmeta.get("paired_llama_tag") != llama_tag:
        sys.exit(f"FAIL: paired_llama_tag {wmeta.get('paired_llama_tag')!r} "
                 f"!= llama release_tag {llama_tag!r}")
    print("VERDICT: marker OK (slim + paired)")

    wbin = bin_dir(whisper_dir)
    server = wbin / ("whisper-server.exe" if os.name == "nt" else "whisper-server")
    if not server.is_file():
        sys.exit(f"FAIL: whisper-server not found at {server}")
    ggml = sorted(p.name for p in wbin.iterdir()
                  if re.match(r"^(lib)?ggml", p.name, re.I))
    if not ggml:
        sys.exit(f"FAIL: no ggml objects wired into {wbin}")
    print(f"VERDICT: ggml wiring OK ({len(ggml)} objects: {', '.join(ggml[:8])}...)")

    if os.name != "nt":
        os.chmod(server, 0o755)
    env = child_env(wbin)
    port = free_port()
    cmd = [str(server), "-m", str(args.model), "--host", "127.0.0.1", "--port", str(port)]
    if args.no_gpu:
        cmd.append("--no-gpu")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env, cwd=str(wbin))
    rc = 0
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 120
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                sys.exit(f"FAIL: whisper-server exited early rc={proc.returncode}:\n{out[:3000]}")
            try:
                urllib.request.urlopen(base + "/", timeout=2)
                ready = True
                break
            except urllib.error.HTTPError:
                ready = True
                break
            except Exception:
                time.sleep(1)
        if not ready:
            sys.exit("FAIL: whisper-server did not become ready in 120s")
        for spec in args.clip:
            path, _, phrase = spec.partition("::")
            audio = Path(path)
            if not audio.is_file():
                print(f"SKIP: clip not present: {audio}")
                continue
            text = post_inference(base, audio)
            print(f"TRANSCRIPT [{audio.name}]: {text[:160]!r}")
            if not text:
                print(f"FAIL: empty transcript for {audio.name}")
                rc = 1
            elif phrase and phrase.lower() not in text.lower():
                print(f"FAIL: transcript for {audio.name} missing phrase {phrase!r}")
                rc = 1
            else:
                print(f"VERDICT: transcription OK [{audio.name}]")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    if rc == 0:
        print("VERDICT: ALL CHECKS PASSED")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
