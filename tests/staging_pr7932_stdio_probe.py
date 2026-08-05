"""Staging-only probe for PR #7932: does the no-console launch actually work?

Two parts, both platform-aware:

  PART A (Windows only) -- the question that cannot be answered on Linux.
    Spawn a child with exactly the flags `unsloth studio` uses at
    unsloth_cli/commands/studio.py:1599 -- CREATE_NO_WINDOW, no stdout=/stderr=
    redirection, so close_fds defaults to True (bInheritHandles=FALSE) -- and have the
    child report whether the interpreter left sys.stdout / sys.stderr as None.
    This is the premise the whole PR rests on.

  PART B (all OSes) -- replay the startup order of run_server() with
    sys.stdout = sys.stderr = sys.stdin = None, against the run.py in THIS checkout,
    aborting at the first exception exactly as run_server does:
        run.py:1748  logger.info(...)              (structlog, BEFORE the tee)
        run.py:1784  _setup_server_disk_logging()  (installs _TeeStream)
        run.py:1955  uvicorn.Config(app, ...)      (AFTER the tee)
        run.py:2316  the failure handler's sys.stderr.write(...)

Exit code is always 0: this probe REPORTS, it does not gate. Read the log.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def emit(msg: str = "") -> None:
    os.write(1, (msg + "\n").encode("utf-8", "replace"))


# --------------------------------------------------------------------------- PART A

CHILD_A = textwrap.dedent(
    """
    import ctypes, os, sys
    # Report through a file, because by construction we may have no stdout.
    lines = []
    for name in ("stdin", "stdout", "stderr"):
        s = getattr(sys, name, "MISSING")
        lines.append(f"sys.{name} = {'None' if s is None else type(s).__name__}")
        d = getattr(sys, f"__{name}__", "MISSING")
        lines.append(f"sys.__{name}__ = {'None' if d is None else type(d).__name__}")
    if sys.platform == "win32":
        k = ctypes.windll.kernel32
        for label, nstd in (("STD_INPUT", -10), ("STD_OUTPUT", -11), ("STD_ERROR", -12)):
            h = k.GetStdHandle(nstd)
            # GetFileType: 0 = FILE_TYPE_UNKNOWN, 1 = DISK, 2 = CHAR, 3 = PIPE
            t = k.GetFileType(h) if h not in (0, None) else -1
            lines.append(f"GetStdHandle({label}) = {h}   GetFileType = {t}")
        lines.append(f"GetConsoleWindow() = {k.GetConsoleWindow()}")
    # print() is a no-op when stdout is None, so write the report to a file.
    open(sys.argv[1], "w", encoding="utf-8").write("\\n".join(lines))
    """
)


def part_a() -> None:
    emit("=" * 78)
    emit("PART A -- real CREATE_NO_WINDOW handle state (the PR's premise)")
    emit("=" * 78)
    if sys.platform != "win32":
        emit(f"  SKIPPED: not Windows (sys.platform={sys.platform!r}).")
        emit("  CREATE_NO_WINDOW does not exist off Windows; PART B still runs everywhere.")
        return

    report = REPO / "_pr7932_child_report.txt"
    child_py = REPO / "_pr7932_child_a.py"
    child_py.write_text(CHILD_A, encoding = "utf-8")

    kwargs: dict = {}
    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if create_no_window:
        kwargs["creationflags"] = create_no_window
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = subprocess.SW_HIDE
    kwargs["startupinfo"] = si

    # Deliberately NO stdout=/stderr=/stdin= -- this is studio.py:1599 verbatim, so
    # close_fds defaults to True and CreateProcess gets bInheritHandles=FALSE.
    emit("  parent: Popen(child, creationflags=CREATE_NO_WINDOW, startupinfo=SW_HIDE)")
    emit("          no stdout=/stderr=/stdin= -> close_fds=True -> bInheritHandles=FALSE")
    proc = subprocess.Popen([sys.executable, str(child_py), str(report)], **kwargs)
    proc.wait(timeout = 120)
    emit(f"  child exit code: {proc.returncode}")
    if report.is_file():
        emit("  child reported:")
        for line in report.read_text(encoding = "utf-8").splitlines():
            emit(f"      {line}")
    else:
        emit("  child produced no report file")
    emit()


# --------------------------------------------------------------------------- PART B

CHILD_B = textwrap.dedent(
    """
    import os, sys
    repo, out_path = sys.argv[1], sys.argv[2]
    results = []

    sys.path.insert(0, os.path.join(repo, "studio", "backend"))
    # Set BEFORE importing run.py: structlog binds `from sys import stdout` at import time.
    sys.stdout = None
    sys.stderr = None
    sys.stdin = None

    def describe(e):
        c = e.__cause__ or e.__context__
        t = f"{type(e).__name__}: {e}"
        if c is not None:
            t += f"  <- {type(c).__name__}: {c}"
        return t

    try:
        import run as run_mod
        results.append(("import run.py", "OK"))
    except BaseException as e:
        results.append(("import run.py", describe(e)))
        open(out_path, "w", encoding="utf-8").write(
            "\\n".join(f"{a}\\t{b}" for a, b in results))
        raise SystemExit(0)

    aborted = False
    try:
        run_mod.logger.info("run_server startup begin api_only=%s", False)
        results.append(("run.py:1748 logger.info (pre-tee)", "OK"))
    except BaseException as e:
        results.append(("run.py:1748 logger.info (pre-tee)", describe(e)))
        aborted = True   # run_server() would abort here

    if aborted:
        results.append(("run.py:1784 _setup_server_disk_logging", "NEVER REACHED"))
        results.append(("run.py:1955 uvicorn.Config", "NEVER REACHED"))
    else:
        try:
            p = run_mod._setup_server_disk_logging()
            results.append(("run.py:1784 _setup_server_disk_logging",
                            "OK (session log: %s)" % ("written" if p else "DISABLED/None")))
        except BaseException as e:
            results.append(("run.py:1784 _setup_server_disk_logging", describe(e)))
        try:
            import uvicorn
            async def _app(scope, receive, send):
                return None
            uvicorn.Config(_app, host="127.0.0.1", port=2718, log_level="info",
                           access_log=False, server_header=False)
            results.append(("run.py:1955 uvicorn.Config", "OK"))
        except BaseException as e:
            results.append(("run.py:1955 uvicorn.Config", describe(e)))

    try:
        sys.stderr.write("ERROR: Unsloth Studio failed to start.\\n")
        sys.stderr.flush()
        results.append(("run.py:2316 failure handler sys.stderr.write", "OK"))
    except BaseException as e:
        results.append(("run.py:2316 failure handler sys.stderr.write",
                        describe(e) + "  [masks the real error; silent death]"))

    open(out_path, "w", encoding="utf-8").write(
        "\\n".join(f"{a}\\t{b}" for a, b in results))
    """
)


def part_b() -> None:
    emit("=" * 78)
    emit(f"PART B -- run_server() startup order with no std streams  ({sys.platform})")
    emit("=" * 78)
    out = REPO / "_pr7932_partb_report.txt"
    child_py = REPO / "_pr7932_child_b.py"
    child_py.write_text(CHILD_B, encoding = "utf-8")

    proc = subprocess.run(
        [sys.executable, str(child_py), str(REPO), str(out)],
        capture_output = True, text = True, timeout = 600,
    )
    if out.is_file():
        for line in out.read_text(encoding = "utf-8").splitlines():
            step, _, outcome = line.partition("\t")
            emit(f"  {step:<46} {outcome}")
        rows = [l.split("\t") for l in out.read_text(encoding = "utf-8").splitlines()]
        first = next((s for s, o in rows if not o.startswith("OK")), "none")
        emit(f"  first failure: {first}")
    else:
        emit(f"  child produced no report (rc={proc.returncode})")
        emit(f"  stderr: {proc.stderr[:800]}")
    emit()


def main() -> int:
    emit(f"python {sys.version.split()[0]} on {sys.platform}")
    try:
        import structlog
        emit(f"structlog {structlog.__version__}")
    except Exception as exc:
        emit(f"structlog unavailable: {exc}")
    try:
        import uvicorn
        emit(f"uvicorn {uvicorn.__version__}")
    except Exception as exc:
        emit(f"uvicorn unavailable: {exc}")
    emit()
    part_a()
    part_b()
    emit("PROBE COMPLETE (reporting only; never gates the job)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
