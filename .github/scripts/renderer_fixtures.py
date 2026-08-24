#!/usr/bin/env python3
"""Launch the built AppImage under four display fixtures and assert the Wayland
classification it logs. Studio logs to stderr before it needs a window, so each launch runs
until the startup line appears and is then killed; a GUI app never exits on its own, so the
process outliving the check is the expected outcome, not a failure."""
import os, pathlib, signal, socket, subprocess, sys, tempfile, time

APPIMAGE = sys.argv[1]
STARTED = "Unsloth desktop app starting"
WAYLAND = "Wayland session"


def launch(env_overrides):
    """Return the renderer decision line, or "" when the app applied no workaround."""
    env = dict(os.environ)
    for name in ("WAYLAND_DISPLAY", "XDG_RUNTIME_DIR"):
        env.pop(name, None)
    env.update(env_overrides)
    env["APPIMAGE_EXTRACT_AND_RUN"] = "1"
    log = pathlib.Path(tempfile.mkstemp(suffix=".log")[1])
    with log.open("w") as sink:
        proc = subprocess.Popen(["xvfb-run", "-a", APPIMAGE], env=env, stdout=sink,
                                stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + 150
        while time.time() < deadline and proc.poll() is None:
            if STARTED in log.read_text(errors="replace"):
                time.sleep(3)   # the decision is logged immediately after the startup line
                break
            time.sleep(2)
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=60)
    text = log.read_text(errors="replace")
    if STARTED not in text:
        sys.exit(f"the AppImage never reached its startup log:\n{text[-3000:]}")
    hits = [line for line in text.splitlines() if "for WebKitGTK compatibility" in line]
    return hits[0] if hits else ""


def runtime_dir(state):
    """A directory holding a listening, a stale, or no wayland-0 socket."""
    d = tempfile.mkdtemp()
    if state == "none":
        return d
    sock = socket.socket(socket.AF_UNIX)
    sock.bind(os.path.join(d, "wayland-0"))
    sock.listen(1)
    if state == "stale":
        sock.close()        # the inode stays behind, as a crashed compositor leaves it
    else:
        KEEP.append(sock)
    return d


KEEP = []


def check(name, overrides, wayland_expected):
    got = launch(overrides)
    # Only the Wayland classification is asserted. A runner whose AppImage cannot load GLES
    # legitimately applies the same variable for a different reason, and reading the reason
    # rather than the variable keeps that from passing or failing for the wrong cause.
    ok = (WAYLAND in got) == wayland_expected
    print(f"[{'ok' if ok else 'FAIL'}] {name}: wayland={WAYLAND in got}, "
          f"expected {wayland_expected}, log line: {got or '(none)'}", flush=True)
    return ok


results = [
    check("plain X11, no wayland socket",
          {"XDG_RUNTIME_DIR": runtime_dir("none")}, False),
    check("live socket named by WAYLAND_DISPLAY",
          {"XDG_RUNTIME_DIR": runtime_dir("live"), "WAYLAND_DISPLAY": "wayland-0"}, True),
    check("stale socket named by WAYLAND_DISPLAY",
          {"XDG_RUNTIME_DIR": runtime_dir("stale"), "WAYLAND_DISPLAY": "wayland-0"}, False),
    check("live default socket, no WAYLAND_DISPLAY",
          {"XDG_RUNTIME_DIR": runtime_dir("live")}, True),
]
sys.exit(0 if all(results) else 1)
