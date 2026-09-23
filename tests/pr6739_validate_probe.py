"""Call the head installer's validate_quantize / validate_server on a real install,
then on truncated copies of the same binaries (negative control: must be rejected)."""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HEAD = Path(sys.argv[1]).resolve()
INSTALL = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(HEAD / "studio"))
spec = importlib.util.spec_from_file_location("ilp", HEAD / "studio" / "install_llama_prebuilt.py")
m = importlib.util.module_from_spec(spec)
sys.modules["ilp"] = m
spec.loader.exec_module(m)

host = m.detect_host()
ext = ".exe" if host.is_windows else ""
bin_dir = INSTALL / "build" / "bin"
server = bin_dir / f"llama-server{ext}"
quantize = bin_dir / f"llama-quantize{ext}"
work = Path(tempfile.mkdtemp(prefix = "pr6739-probe-"))
probe = work / "stories260K.gguf"
m.download_validation_model(probe, m.validation_model_cache_path(INSTALL))

plan = m.build_validation_sandbox_plan(
    [str(quantize)], binary_path = quantize, install_dir = INSTALL,
    purpose = m._VALIDATION_PURPOSE_QUANTIZE, env = {}, host = host,
)
results = {"plan_kind": plan.sandbox_kind, "plan_action": plan.action}


def attempt(name, fn):
    start = time.monotonic()
    try:
        fn()
        results[name] = "passed"
    except m.PrebuiltFallback as exc:
        results[name] = f"rejected: {str(exc)[:160]}"
    except Exception as exc:
        results[name] = f"error {type(exc).__name__}: {str(exc)[:160]}"
    results[name + "_secs"] = round(time.monotonic() - start, 1)


attempt("quantize_real", lambda: m.validate_quantize(
    quantize, probe, work / "q.gguf", INSTALL, host, require_launch = True))
attempt("server_real", lambda: m.validate_server(
    server, probe, host, INSTALL, require_launch = True))

# Negative control: same install, binaries truncated to 4 KiB.
broken = work / "broken"
shutil.copytree(INSTALL, broken, symlinks = True)
for name in (server.name, quantize.name):
    target = broken / "build" / "bin" / name
    data = target.resolve().read_bytes()[:4096]
    target.unlink()
    target.write_bytes(data)
    target.chmod(0o755)
bb = broken / "build" / "bin"
attempt("quantize_truncated", lambda: m.validate_quantize(
    bb / quantize.name, probe, work / "q2.gguf", broken, host, require_launch = True))
attempt("server_truncated", lambda: m.validate_server(
    bb / server.name, probe, host, broken, require_launch = True))

print("PROBE " + json.dumps(results))
ok = (
    results["quantize_real"] == "passed"
    and results["server_real"] == "passed"
    and results["quantize_truncated"].startswith("rejected")
    and results["server_truncated"].startswith("rejected")
)
print("PROBE_VERDICT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
