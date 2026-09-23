"""Which llama-server invocation survives the head's macOS Seatbelt validation profile?"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HEAD = Path(sys.argv[1]).resolve()
INSTALL = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(HEAD / "studio"))
spec = importlib.util.spec_from_file_location("ilp", HEAD / "studio" / "install_llama_prebuilt.py")
m = importlib.util.module_from_spec(spec)
sys.modules["ilp"] = m
spec.loader.exec_module(m)

host = m.detect_host()
server = INSTALL / "build" / "bin" / "llama-server"
work = Path(tempfile.mkdtemp(prefix = "pr6739-metal-"))
probe = work / "stories260K.gguf"
m.download_validation_model(probe, m.validation_model_cache_path(INSTALL))
env = m.binary_env(server, INSTALL, host)

BROAD = "(allow iokit-open)(allow iokit-get-properties)(allow mach-lookup)(allow sysctl-read)"
NARROW = (
    "(allow iokit-open (iokit-user-client-class \"AGXDeviceUserClient\")"
    " (iokit-user-client-class \"IOSurfaceRootUserClient\")"
    " (iokit-user-client-class \"AppleParavirtDeviceUserClient\")"
    " (iokit-user-client-class \"IOAccelDevice2\") (iokit-user-client-class \"IOAccelSharedUserClient2\")"
    " (iokit-user-client-class \"IOAccelCommandQueue2\") (iokit-user-client-class \"AGXCommandQueue\"))"
    "(allow iokit-get-properties)"
    "(allow mach-lookup (global-name \"com.apple.MTLCompilerService\"))"
)
VARIANTS = [
    ("bare_default", [], None, False),
    ("sandbox_default", [], "", True),
    ("sandbox_ngl0", ["--n-gpu-layers", "0"], "", True),
    ("sandbox_device_none", ["--device", "none"], "", True),
    ("sandbox_ngl0_device_none", ["--n-gpu-layers", "0", "--device", "none"], "", True),
    ("sandbox_default_metal_broad", [], BROAD, True),
    ("sandbox_default_metal_narrow", [], NARROW, True),
]
results = {}
for name, extra, rules, sandboxed in VARIANTS:
    port = m.free_local_port()
    command = [str(server), "-m", str(probe), "--host", "127.0.0.1", "--port", str(port),
               "-c", "32", "--parallel", "1", "--threads", "1", "--ubatch-size", "32",
               "--batch-size", "32", *extra]
    if sandboxed:
        prefix = m._macos_validation_sandbox_prefix(
            command, binary_path = server, install_dir = INSTALL,
            purpose = m._VALIDATION_PURPOSE_SERVER, env = env,
            adapter_path = "/usr/bin/sandbox-exec",
        )
        prefix[2] = prefix[2] + rules
        full = [*prefix, "/usr/bin/env", "-i", *[f"{k}={v}" for k, v in sorted(env.items())], *command]
    else:
        full = ["/usr/bin/env", "-i", *[f"{k}={v}" for k, v in sorted(env.items())], *command]
    log = work / f"{name}.log"
    with log.open("w") as handle:
        proc = subprocess.Popen(full, stdout = handle, stderr = subprocess.STDOUT)
    outcome = "timeout"
    deadline = time.time() + 45
    while time.time() < deadline:
        if proc.poll() is not None:
            outcome = f"exited {proc.returncode}"
            break
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/completion",
                data = json.dumps({"prompt": "Hi", "n_predict": 1}).encode(),
                headers = {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout = 5) as resp:
                if resp.status == 200:
                    outcome = "completion_ok"
                    break
        except Exception:
            time.sleep(0.5)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
    text = log.read_text(errors = "replace")
    offload = [l for l in text.splitlines() if "offloaded" in l or "using device" in l.lower() or "MTL" in l]
    errors = [l for l in text.splitlines() if " E " in l or "error" in l.lower()]
    results[name] = {"outcome": outcome, "offload": offload[-2:], "errors": errors[:3]}
    print("VARIANT", name, json.dumps(results[name]), flush = True)
print("METAL_PROBE " + json.dumps({k: v["outcome"] for k, v in results.items()}))
