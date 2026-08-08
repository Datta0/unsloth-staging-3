# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Browser regression + screenshot job for the desktop closing overlay.

Renders the REAL built frontend (studio/frontend/dist) in Chromium with
`_tauri_webview_shim.js` standing in for the Tauri runtime and main.rs, clicks
the titlebar close button, and asserts the user gets feedback:

  1. clicking close raises "Closing Unsloth Desktop..." over the shell;
  2. the shell is still underneath (a declined quit must hand it back);
  3. past FORCE_QUIT_AFTER_MS the overlay offers a Force quit that reaches the
     force_quit command;
  4. app-closing-cancelled takes the overlay back down.

No Studio backend: the frontend is served statically, so this needs a node
build and a browser and nothing else. The frontend unit tests cover the store
and assert the call sites by reading the source; nothing else in the repo puts
ClosingScreen on a screen.

Standalone script, like the other tests/studio/playwright_*.py:

    (cd studio/frontend && npm ci && npm run build)
    python tests/studio/playwright_desktop_closing_overlay.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import chromium_launch_args  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DIST = REPO / "studio/frontend/dist"
SHIM = (Path(__file__).resolve().parent / "_tauri_webview_shim.js").read_text()
ART = Path(os.environ.get("PW_ART_DIR", "logs/closing-overlay"))

# closing-signal.ts FORCE_QUIT_AFTER_MS, plus room for the render.
FORCE_QUIT_WAIT_MS = 22_000
VIEWPORT = {"width": 1440, "height": 900}


class _SpaHandler(SimpleHTTPRequestHandler):
    """Static dist with the SPA fallback the backend does for unknown routes."""

    def send_head(self):  # noqa: ANN201
        if not os.path.exists(self.translate_path(self.path)) and (
            "." not in os.path.basename(self.path)
        ):
            self.path = "/index.html"
        return super().send_head()

    def log_message(self, *args: object) -> None:
        return


def _serve(directory: Path) -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(_SpaHandler, directory=str(directory))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_port


def _overlay_title(page) -> str | None:  # noqa: ANN001
    return page.evaluate(
        """() => {
            const el = [...document.querySelectorAll('p')]
                .find(n => n.textContent.trim().startsWith('Closing Unsloth Desktop'));
            return el ? el.textContent.trim() : null;
        }"""
    )


def main() -> int:
    if not (DIST / "index.html").exists():
        print(f"FAIL: no frontend build at {DIST}; run `npm run build` first")
        return 1
    ART.mkdir(parents=True, exist_ok=True)

    server, port = _serve(DIST)
    failures: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=chromium_launch_args())
            context = browser.new_context(viewport=VIEWPORT)
            cfg = json.dumps(
                {
                    "backendPort": port,
                    "accessToken": "closing-overlay-test",
                    "refreshToken": "closing-overlay-test",
                    "width": VIEWPORT["width"],
                    "height": VIEWPORT["height"],
                }
            )
            context.add_init_script(f"window.__UNSLOTH_SHIM_CFG__ = {cfg};")
            context.add_init_script(SHIM)
            page = context.new_page()
            page.goto(f"http://127.0.0.1:{port}/chat", wait_until="domcontentloaded")

            close_btn = page.locator('button[aria-label="Close window"]').first
            close_btn.wait_for(state="visible", timeout=60_000)
            page.wait_for_timeout(1_500)
            page.screenshot(path=str(ART / "01_desktop_shell.png"))

            if page.evaluate("() => window.__SHIM__.listenerCount('app-closing')") != 1:
                failures.append("nothing listens for app-closing")

            close_btn.click()
            page.wait_for_timeout(1_000)
            page.screenshot(path=str(ART / "02_closing_overlay.png"))

            if _overlay_title(page) != "Closing Unsloth Desktop...":
                failures.append("close left the window with no closing overlay")
            if not page.locator("text=Shutting down the backend.").count():
                failures.append("the overlay does not name the wait it covers")
            if not page.evaluate(
                "() => !!document.querySelector('form:has(textarea) textarea')"
            ):
                failures.append("the overlay replaced the app instead of covering it")

            page.wait_for_timeout(FORCE_QUIT_WAIT_MS)
            page.screenshot(path=str(ART / "03_wedged_force_quit.png"))
            force_quit = page.locator("button", has_text="Force quit").first
            if not force_quit.count():
                failures.append("a wedged reap offers no way out of the app")
            else:
                force_quit.click()
                page.wait_for_timeout(500)
                if not page.evaluate("() => window.__SHIM_FORCE_QUIT__ || 0"):
                    failures.append("Force quit never reached the force_quit command")

            page.evaluate("() => window.__SHIM__.emit('app-closing-cancelled', null)")
            page.wait_for_timeout(1_000)
            page.screenshot(path=str(ART / "04_cancelled_shell_restored.png"))
            if _overlay_title(page) is not None:
                failures.append("a cancelled quit left the overlay up")

            context.close()
            browser.close()
    finally:
        server.shutdown()

    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        return 1
    print(f"OK -- closing overlay renders, wedges and clears; shots in {ART}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
