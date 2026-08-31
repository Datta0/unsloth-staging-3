#!/usr/bin/env python3
"""Run the Hub selection resolver inside Chromium, Firefox and WebKit.

Starts the project's own Vite dev server against the harness page, so each
engine executes the real source modules with the real alias resolution, then
reads the result payload the harness leaves on `window`.

Chromium covers Edge, which shares the engine. WebKit covers Safari.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

FRONTEND = Path(sys.argv[1]).resolve()
# One engine per CI leg, all three locally.
ENGINES = tuple(
    e.strip()
    for e in os.environ.get("ENGINES", "chromium,firefox,webkit").split(",")
    if e.strip()
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(port: int, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise SystemExit(f"vite never came up on :{port}")


def main() -> int:
    from playwright.sync_api import sync_playwright

    port = free_port()
    server = subprocess.Popen(
        ["npx", "vite", "--port", str(port), "--strictPort", "--host", "127.0.0.1"],
        cwd=FRONTEND,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for(port)
        url = f"http://127.0.0.1:{port}/smoke-hub-selection.html"
        overall_ok = True
        summary: dict[str, object] = {}

        with sync_playwright() as pw:
            for engine in ENGINES:
                browser = getattr(pw, engine).launch()
                page = browser.new_page()
                errors: list[str] = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.on(
                    "console",
                    lambda m: errors.append(m.text) if m.type == "error" else None,
                )
                page.goto(url, wait_until="load", timeout=120_000)
                page.wait_for_function(
                    "() => window.__hubSelectionResults !== undefined",
                    timeout=120_000,
                )
                payload = page.evaluate("() => window.__hubSelectionResults")
                browser.close()

                ok = not payload["failed"] and not errors
                overall_ok &= ok
                summary[engine] = {
                    "passed": payload["passed"],
                    "total": payload["total"],
                    "failed": payload["failed"],
                    "console_errors": errors,
                    "ua": payload["userAgent"][:90],
                }
                mark = "PASS" if ok else "FAIL"
                print(f"[{engine:9s}] {mark} {payload['passed']}/{payload['total']}")
                for case in payload["failed"]:
                    print(f"             - {case['name']}: {case.get('detail')}")
                for err in errors:
                    print(f"             ! console: {err}")

        out = FRONTEND.parent.parent / "cross_browser_results.json"
        try:
            out.write_text(json.dumps(summary, indent=2))
        except OSError:
            pass
        print(json.dumps(summary, indent=2))
        return 0 if overall_ok else 1
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
