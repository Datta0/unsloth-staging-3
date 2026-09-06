# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth studio update` must reuse the cache the install filled, not uv's default.

Three places pick a uv cache and only two of them agreed. install.sh and install.ps1
choose one for the install (#10204), and storage_roots._setup_cache_env seeds
cache_root()/uv for the backend server, which is also where install.sh repoints
UV_CACHE_DIR before autostart. An update reached neither: it runs setup.sh/setup.ps1
straight from the CLI process, nothing there set the variable, and uv fell back to
its own user-wide default.

Measured on a real uv 0.10.7 with the merged selector, one package, --reinstall:

    new user   (installer picked "studio")
        Studio cache after install : 24 files
        default cache after update : 0 -> 23 files      <- re-downloaded

The install's bytes were then dead weight under the Studio root, and the update's
copy landed where scripts/uninstall.* cannot reclaim it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


def _setup_tree(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    (repo_root / "studio").mkdir(parents = True)
    (repo_root / "studio" / "setup.sh").write_text("")
    (repo_root / "studio" / "setup.ps1").write_text("")
    return repo_root


class _Result:
    returncode = 0


def _run_posix(monkeypatch, tmp_path: Path) -> dict:
    """Run the POSIX branch of _run_setup_script and return the child's env."""
    studio = _studio()
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    seen: dict = {}

    def _fake_run(
        argv,
        env = None,
        **kwargs,
    ):
        seen["argv"] = list(argv)
        seen["env"] = env
        return _Result()

    monkeypatch.setattr(studio.subprocess, "run", _fake_run)
    studio._run_setup_script(repo_root = _setup_tree(tmp_path))
    return seen


def test_the_update_uses_the_studio_cache_when_the_caller_set_none(monkeypatch, tmp_path):
    studio = _studio()
    monkeypatch.delenv("UV_CACHE_DIR", raising = False)
    seen = _run_posix(monkeypatch, tmp_path)

    expected = str(studio.STUDIO_HOME / "cache" / "uv")
    assert seen["env"] is not None, "env must be materialised, not left as inherit-everything"
    assert seen["env"]["UV_CACHE_DIR"] == expected, seen["env"].get("UV_CACHE_DIR")


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_a_blank_uv_cache_dir_counts_as_unset(monkeypatch, tmp_path, blank):
    """storage_roots.py:373 treats blank as unset; the update path must agree, or an
    inherited UV_CACHE_DIR= pins uv's cache to the empty string."""
    studio = _studio()
    monkeypatch.setenv("UV_CACHE_DIR", blank)
    seen = _run_posix(monkeypatch, tmp_path)

    expected = str(studio.STUDIO_HOME / "cache" / "uv")
    assert seen["env"]["UV_CACHE_DIR"] == expected, seen["env"].get("UV_CACHE_DIR")


def test_an_explicit_uv_cache_dir_still_wins(monkeypatch, tmp_path):
    """Same precedence the installers use: a nonblank caller value is preserved
    (install.sh:626, install.ps1:1232), so CI images that pin a cache keep it."""
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "caller cache"))
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"] is None or seen["env"]["UV_CACHE_DIR"] == str(
        tmp_path / "caller cache"
    ), seen["env"]


def test_verbose_keeps_its_own_flag_alongside_the_cache(monkeypatch, tmp_path):
    """The verbose branch builds env first; the cache seeding must extend it, not
    replace it."""
    studio = _studio()
    monkeypatch.delenv("UV_CACHE_DIR", raising = False)
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    seen: dict = {}

    def _fake_run(
        argv,
        env = None,
        **kwargs,
    ):
        seen["env"] = env
        return _Result()

    monkeypatch.setattr(studio.subprocess, "run", _fake_run)
    studio._run_setup_script(verbose = True, repo_root = _setup_tree(tmp_path))

    assert seen["env"]["UNSLOTH_VERBOSE"] == "1", seen["env"].get("UNSLOTH_VERBOSE")
    assert seen["env"]["UV_CACHE_DIR"] == str(studio.STUDIO_HOME / "cache" / "uv")


def test_the_windows_branch_gets_the_same_cache(monkeypatch, tmp_path):
    """setup.ps1 runs the same uv pip installs, so the PowerShell spawn needs it too."""
    studio = _studio()
    monkeypatch.delenv("UV_CACHE_DIR", raising = False)
    monkeypatch.setattr(studio.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        studio._studio_runtime_gate, "resolve_windows_powershell", lambda: "powershell.exe"
    )
    monkeypatch.setattr(studio, "_probe_profile_proxy_defaults", lambda hosts: None)
    monkeypatch.setattr(studio, "_wait_for_windows_setup_process", lambda process: 0)
    seen: dict = {}

    class _Process:
        pass

    def _fake_popen(
        argv,
        env = None,
        **kwargs,
    ):
        seen["env"] = env
        return _Process()

    monkeypatch.setattr(studio.subprocess, "Popen", _fake_popen)
    studio._run_setup_script(repo_root = _setup_tree(tmp_path))

    assert seen["env"]["UV_CACHE_DIR"] == str(studio.STUDIO_HOME / "cache" / "uv")


def test_the_seeding_does_not_leak_into_this_process(monkeypatch, tmp_path):
    """_ensure_studio_env_exported mutates os.environ on purpose; this must not, or a
    later `unsloth studio` in the same process would look like a caller override to
    storage_roots._setup_cache_env."""
    monkeypatch.delenv("UV_CACHE_DIR", raising = False)
    _run_posix(monkeypatch, tmp_path)

    assert "UV_CACHE_DIR" not in os.environ, os.environ.get("UV_CACHE_DIR")
