#!/usr/bin/env python3
"""Does the interface actually freeze on a real Wayland session?

The reporter's symptom is precise: the app stays up and the backend stays reachable from
another device, but the local window stops updating. That gives a clean oracle, because the
two poll loops have different owners:

  /api/inference/monitor   frontend (studio/frontend/src/features/chat/api/chat-api.ts)
                           -- only fires while the webview's JS is running
  /api/liveness            native Rust watchdog (desktop_backend_owner.rs)
                           -- keeps firing even if the webview is dead

So a run where liveness keeps ticking while monitor goes quiet IS the reported freeze,
observed rather than inferred. Pixels are not used as the signal: a healthy idle UI is a
static image, so frame differencing cannot tell "not moving" from "not alive".
"""
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import time

# Overridable because the two callers ask different questions. The released-vs-fixed
# comparison wants a long window (a slow leak into a freeze still counts). The escape-hatch
# survey only asks "does the webview come up and keep polling under this variable", which a
# short window answers just as well, and it has five candidates to get through.
WARMUP = int(os.environ.get("FREEZE_PROBE_WARMUP", 60))
WINDOW = int(os.environ.get("FREEZE_PROBE_WINDOW", 240))
MONITOR = re.compile(r"/api/inference/monitor")
LIVENESS = re.compile(r"/api/liveness")


def wait_for_port_free(port=8888, timeout=90):
    """The previous candidate's backend has to be gone or the next one adopts it."""
    for _ in range(timeout):
        out = subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout
        if f":{port}" not in out:
            return True
        time.sleep(1)
    return False


# Where the poll lines actually are, established the hard way over three runs that each
# measured nothing for a different reason:
#
#   1. stdout           -- the backend runs uvicorn with access_log=False
#                          (studio/backend/run.py) and logs requests itself, so stdout has
#                          none of them. Counted zero for a healthy app.
#   2. tauri.log        -- the desktop shell's own log. Carries the renderer decision and
#                          the preflight lines, but not request lines.
#   3. backend-*.log    -- correct, BUT only written when the desktop shell SPAWNS and
#                          supervises the backend. A backend started by hand logs to its
#                          own stdout, so the app attaches to something healthy that is
#                          logging nowhere and the counters still read zero.
#
# So the probe needs an app-managed backend, not merely a reachable one, and there is no
# log path that rescues a run where the Studio stack is not installed at all: preflight
# reports NotInstalled / port=None and neither poll loop exists to be counted.
TAURI_LOG = pathlib.Path.home() / ".unsloth" / "studio" / "tauri.log"
BACKEND_LOGS = pathlib.Path.home() / ".unsloth" / "studio" / "logs"


def _tail_from(path, offset):
    try:
        with path.open("rb") as fh:
            fh.seek(offset)
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _offset(path):
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _backend_offsets():
    """Sizes of every supervised backend log now, so a candidate counts only its own."""
    try:
        return {p: p.stat().st_size for p in BACKEND_LOGS.glob("backend-*.log")}
    except OSError:
        return {}


def _backend_tail(before):
    """Everything appended since `before`, including logs that did not exist then."""
    out = []
    try:
        logs = list(BACKEND_LOGS.glob("backend-*.log"))
    except OSError:
        return ""
    for p in logs:
        out.append(_tail_from(p, before.get(p, 0)))
    return "".join(out)


def run_candidate(name, binary, extra_env):
    print(f"\n===== {name} =====", flush=True)
    print(f"binary: {binary}", flush=True)
    wait_for_port_free()
    log = pathlib.Path(tempfile.mkstemp(suffix=".log", prefix="freeze-")[1])
    tauri_at = _offset(TAURI_LOG)
    backend_at = _backend_offsets()
    env = {**os.environ, **extra_env}
    env["APPIMAGE_EXTRACT_AND_RUN"] = "1"
    proc = subprocess.Popen([binary], env=env, stdout=log.open("w"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    started = time.time()
    last_report = 0
    try:
        while time.time() - started < WARMUP + WINDOW:
            time.sleep(5)
            if proc.poll() is not None:
                print(f"  process EXITED rc={proc.returncode} after "
                      f"{time.time() - started:.0f}s", flush=True)
                break
            now = time.time() - started
            if now - last_report >= 30:
                text = (log.read_text(errors="replace")
                        + _tail_from(TAURI_LOG, tauri_at) + _backend_tail(backend_at))
                print(f"  t={now:6.0f}s  monitor={len(MONITOR.findall(text)):4}"
                      f"  liveness={len(LIVENESS.findall(text)):4}", flush=True)
                last_report = now
    finally:
        alive = proc.poll() is None
        if alive:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=60)

    text = (log.read_text(errors="replace")
            + _tail_from(TAURI_LOG, tauri_at) + _backend_tail(backend_at))
    n_mon, n_live = len(MONITOR.findall(text)), len(LIVENESS.findall(text))
    decision = [line for line in text.splitlines() if "WebKitGTK compatibility" in line]
    print(f"  renderer: {decision[0].strip() if decision else '(no workaround applied)'}",
          flush=True)
    print(f"  alive at end: {alive}   monitor polls: {n_mon}   liveness polls: {n_live}",
          flush=True)
    pre = [l for l in text.splitlines() if "desktop_preflight completed" in l]
    print(f"  preflight: {pre[-1].strip() if pre else '(none)'}", flush=True)
    if not alive:
        verdict = "CRASHED"
    elif n_mon == 0 and n_live == 0:
        # Neither loop was seen at all. A frozen app and a probe that is looking in the
        # wrong place produce identical counters, so this can never be reported as a pass;
        # it means the measurement failed and the run has nothing to say about the app.
        verdict = "NO SIGNAL (probe saw neither poll loop; measurement failed, not a pass)"
    elif n_live > 0 and n_mon == 0:
        verdict = "FROZE (webview never polled while the shell kept running)"
    elif n_live >= 3 and n_mon * 3 < n_live:
        verdict = "SUSPECT (webview polled far less than the native watchdog)"
    else:
        verdict = "OK (webview kept polling)"
    print(f"  VERDICT: {verdict}", flush=True)
    for line in text.splitlines():
        if re.search(r"Error 71|Protocol error|Gdk-Message|EGL|wayland", line, re.I):
            print(f"    ! {line.strip()[:200]}", flush=True)
    return name, verdict


if __name__ == "__main__":
    # each argument is  name|binary|K=V,K=V
    results = []
    for arg in sys.argv[1:]:
        name, binary, raw = arg.split("|")
        extra = dict(pair.split("=", 1) for pair in raw.split(",") if pair)
        results.append(run_candidate(name, binary, extra))
    print("\n===== summary =====", flush=True)
    for name, verdict in results:
        print(f"  {verdict:70} {name}", flush=True)
    # Fail the step on a dead oracle. A green step whose every verdict was produced without
    # a single observation is worse than a red one, because it gets quoted as evidence.
    if any(v.startswith("NO SIGNAL") for _, v in results):
        print("::error::freeze probe collected no signal; verdicts above are not evidence",
              flush=True)
        sys.exit(1)
