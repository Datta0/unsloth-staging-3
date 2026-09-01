"""Exercise studio/backend/core/_msvc_env.py on a real windows-latest runner with AMD spoofed.

The runner is genuinely Windows/win_amd64 with a genuinely installed `triton-windows` wheel and
genuinely installed Visual Studio 2022 Build Tools. What is faked is only the AMD side: a
zero-byte `_rocm_sdk_core/lib/llvm/bin/clang-cl.exe` standing in for the ROCm wheel's compiler,
and, for the "no Visual Studio" half, an environment that hides VS from Triton's own discovery
instead of uninstalling it. There is no AMD GPU and no ROCm runtime here, so no Triton HIP
compile ever runs; this measures the gate's decision, not the crash the gate exists to avoid.

Each scenario runs in a FRESH child process: `find_msvc_winsdk` and `get_cc`/`_find_compiler`
are `functools.lru_cache`d and `TORCHDYNAMO_DISABLE` is process state, so re-using one
interpreter would measure the first scenario forever.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

MARK = "@@GATEJSON@@"
GATE_REL = Path("studio") / "backend" / "core" / "_msvc_env.py"


def _load_gate(repo_root: Path):
    """Load the gate by file path: it imports only os/sys/logging, so nothing else is needed and
    the studio package's heavy __init__ never runs."""
    import importlib.util

    path = repo_root / GATE_REL
    spec = importlib.util.spec_from_file_location("_msvc_env_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rocm_cc_path() -> Path:
    return (Path(sysconfig.get_paths()["platlib"]) / "_rocm_sdk_core" / "lib" / "llvm" / "bin"
            / "clang-cl.exe")


def _triton_facts() -> dict:
    """What Triton itself reports, before the gate is asked anything."""
    out: dict = {}
    try:
        import triton

        out["triton_version"] = getattr(triton, "__version__", None)
        out["triton_file"] = getattr(triton, "__file__", None)
    except Exception as e:
        out["triton_import_error"] = f"{type(e).__name__}: {e}"
        return out
    try:
        import importlib.metadata as md

        dists = md.packages_distributions().get("triton") or []
        out["owning_distributions"] = sorted(dists)
        for d in dists:
            try:
                out[f"dist_version_{d}"] = md.version(d)
            except Exception:
                pass
    except Exception as e:
        out["dist_error"] = f"{type(e).__name__}: {e}"

    # Which compiler Triton picks, via whichever entry point this release ships.
    try:
        from triton.runtime import build as tb

        if hasattr(tb, "get_cc"):
            out["compiler_entry_point"] = "get_cc"
            cc = tb.get_cc()
        else:
            out["compiler_entry_point"] = "_find_compiler"
            cc = tb._find_compiler("c")
        out["cc"] = cc
        out["cc_basename"] = os.path.basename(str(cc))
        out["cc_is_rocm_wheel"] = "_rocm_sdk_core" in str(cc).replace("/", "\\")
        for pred in ("is_msvc", "is_clang_cl", "is_tcc"):
            if hasattr(tb, pred):
                try:
                    out[pred] = bool(getattr(tb, pred)(cc))
                except Exception:
                    out[pred] = None
    except Exception as e:
        out["cc_error"] = f"{type(e).__name__}: {e}"

    # Triton's own MSVC/WinSDK search, which is the only evidence the gate may act on.
    try:
        from triton.windows_utils import find_msvc_winsdk

        bin_path, inc_dirs, _lib = find_msvc_winsdk()
        inc_dirs = list(inc_dirs)
        out["msvc_bin_path"] = bin_path
        out["include_dirs"] = inc_dirs
        out["include_dir_count"] = len(inc_dirs)
        for hdr in ("stdlib.h", "vcruntime.h"):
            out[f"found_{hdr.replace('.', '_')}"] = any(
                d and os.path.isfile(os.path.join(d, hdr)) for d in inc_dirs
            )
    except Exception as e:
        out["find_msvc_winsdk_error"] = f"{type(e).__name__}: {e}"
    return out


def run_child(repo_root: Path, scenario: str) -> dict:
    res: dict = {"scenario": scenario, "platform": sys.platform,
                 "python": sys.version.split()[0],
                 "rocm_cc_present": _rocm_cc_path().is_file(),
                 "CC_env": os.environ.get("CC"),
                 "INCLUDE_env": os.environ.get("INCLUDE"),
                 "VCINSTALLDIR_env": os.environ.get("VCINSTALLDIR")}
    res["triton"] = _triton_facts()

    gate = _load_gate(repo_root)
    log = logging.getLogger("probe")
    # crt_headers_reachable() is the decision; gate_torch_compile_on_windows() is the effect.
    try:
        res["crt_headers_reachable"] = gate.crt_headers_reachable()
    except Exception as e:
        res["crt_headers_reachable_error"] = f"{type(e).__name__}: {e}"
    for helper in ("_needs_msvc_headers", "_have_crt_headers", "_triton_finds_crt_headers",
                   "_rocm_clang_cl_present", "_triton_is_triton_windows"):
        if hasattr(gate, helper):
            try:
                res[helper] = getattr(gate, helper)()
            except Exception as e:
                res[helper] = f"ERROR {type(e).__name__}: {e}"
    try:
        res["toolchain_summary"] = gate._toolchain_summary()
    except Exception as e:
        res["toolchain_summary"] = f"ERROR {type(e).__name__}: {e}"

    os.environ.pop("TORCHDYNAMO_DISABLE", None)
    gate.gate_torch_compile_on_windows(log)
    res["TORCHDYNAMO_DISABLE"] = os.environ.get("TORCHDYNAMO_DISABLE")
    res["dynamo_disabled"] = os.environ.get("TORCHDYNAMO_DISABLE") == "1"
    return res


# Scenario environments.
_VS_ENV_VARS = ("VCINSTALLDIR", "VCToolsVersion", "VCToolsInstallDir", "WindowsSdkDir",
                "WindowsSDKVersion", "WindowsSDKVer", "INCLUDE", "LIB", "LIBPATH")


def _hide_vs(env: dict, empty_dir: Path) -> dict:
    """Hide Visual Studio from triton.windows_utils without uninstalling it.

    find_msvc_env reads VCINSTALLDIR/VCToolsVersion; find_msvc_vswhere and find_msvc_hardcoded
    and find_winsdk_hardcoded all go through find_in_program_files, which reads the
    ProgramFiles(x86)/ProgramW6432 env vars; find_msvc_envpath scans PATH for \\VC\\Tools\\MSVC\\.
    All four are therefore reachable from the environment. find_winsdk_registry reads HKLM
    directly and is NOT hidden this way, which is the point: it leaves a standalone Windows SDK
    (stdlib.h, no vcruntime.h) visible, exactly the partial toolchain the gate must refuse.
    """
    for k in _VS_ENV_VARS:
        env.pop(k, None)
    for k in ("ProgramFiles(x86)", "ProgramW6432", "ProgramFiles"):
        env[k] = str(empty_dir)
    path = env.get("PATH", "")
    kept = [p for p in path.split(os.pathsep)
            if not re.search(r"\\VC\\Tools\\MSVC\\", p.replace("/", "\\"), re.I)]
    env["PATH"] = os.pathsep.join(kept)
    return env


def _no_triton_dir(root: Path) -> Path:
    """A `triton` that raises ImportError, so the gate's no-Triton branch is exercised for real
    rather than by patching the gate."""
    d = root / "no_triton"
    (d / "triton").mkdir(parents=True, exist_ok=True)
    (d / "triton" / "__init__.py").write_text(
        'raise ImportError("triton hidden by the staging probe")\n')
    return d


SCENARIOS = {
    # id: (plant rocm clang-cl, force CC to it, hide Visual Studio, hide triton)
    "s1_baseline_no_amd_vs_present":   (False, False, False, False),
    "s2_amd_wheel_vs_present":         (True,  False, False, False),
    "s3_amd_wheel_vs_absent":          (True,  False, True,  False),
    "s4_no_amd_vs_absent":             (False, False, True,  False),
    "s5_cc_clang_cl_vs_present":       (True,  True,  False, False),
    "s6_cc_clang_cl_vs_absent":        (True,  True,  True,  False),
    "s7_no_triton":                    (False, False, False, True),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--child")
    ap.add_argument("--out", default="win_gate_report.json")
    a = ap.parse_args()
    repo_root = Path(a.repo_root).resolve()

    if a.child:
        print(MARK + json.dumps(run_child(repo_root, a.child)))
        return 0

    if sys.platform != "win32":
        print(json.dumps({"skipped": "not win32", "platform": sys.platform}, indent=2))
        return 0

    scratch = repo_root / "_gate_scratch"
    empty = scratch / "empty_program_files"
    empty.mkdir(parents=True, exist_ok=True)
    no_triton = _no_triton_dir(scratch)
    rocm_cc = _rocm_cc_path()

    report: dict = {"runner": {"platform": sys.platform, "python": sys.version.split()[0],
                               "platlib": sysconfig.get_paths()["platlib"],
                               "rocm_cc_path": str(rocm_cc)},
                    "scenarios": {}}

    for sid, (plant, force_cc, hide_vs, hide_triton) in SCENARIOS.items():
        if plant:
            rocm_cc.parent.mkdir(parents=True, exist_ok=True)
            rocm_cc.write_bytes(b"")  # a placeholder, not a working compiler
        elif rocm_cc.is_file():
            rocm_cc.unlink()

        env = dict(os.environ)
        env.pop("TORCHDYNAMO_DISABLE", None)
        env.pop("CC", None)
        if force_cc:
            env["CC"] = str(rocm_cc)
        if hide_vs:
            env = _hide_vs(env, empty)
        if hide_triton:
            env["PYTHONPATH"] = str(no_triton) + os.pathsep + env.get("PYTHONPATH", "")

        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--repo-root", str(repo_root), "--child", sid]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith(MARK)), None)
        if line is None:
            report["scenarios"][sid] = {"scenario": sid, "child_failed": True,
                                        "returncode": p.returncode,
                                        "stdout": p.stdout[-3000:], "stderr": p.stderr[-3000:]}
        else:
            report["scenarios"][sid] = json.loads(line[len(MARK):])

    if rocm_cc.is_file():
        rocm_cc.unlink()

    report["checks"] = _check(report["scenarios"])
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    failed = [c for c in report["checks"] if c["status"] == "FAIL"]
    print("\n=== gate checks ===")
    for c in report["checks"]:
        print(f"{c['status']:5} {c['id']}: {c['detail']}")
    return 1 if failed else 0


def _clang_cl_selected(s: dict) -> bool:
    t = s.get("triton") or {}
    return bool(t.get("is_clang_cl")) or t.get("cc_basename", "").lower() == "clang-cl.exe"


def _check(sc: dict) -> list:
    """Expectations. The gate can only ever DISABLE torch.compile, so a wrong `True` degrades a
    machine that compiles fine; the false-positive guards below are the ones that matter most."""
    out = []

    def add(cid, status, detail):
        out.append({"id": cid, "status": status, "detail": detail})

    for sid, s in sc.items():
        if s.get("child_failed"):
            add(sid, "FAIL", f"child process did not report (rc={s.get('returncode')})")

    # Deterministic pair: CC is Triton's own documented override, so clang-cl is definitely picked.
    s5, s6 = sc.get("s5_cc_clang_cl_vs_present", {}), sc.get("s6_cc_clang_cl_vs_absent", {})
    if _clang_cl_selected(s5):
        add("s5_vs_present_leaves_compile_on",
            "PASS" if s5.get("dynamo_disabled") is False else "FAIL",
            f"clang-cl + Visual Studio present -> TORCHDYNAMO_DISABLE={s5.get('TORCHDYNAMO_DISABLE')!r} "
            f"(want unset); {s5.get('toolchain_summary')}")
    else:
        add("s5_vs_present_leaves_compile_on", "FAIL",
            f"CC override did not make Triton pick clang-cl: cc={s5.get('triton', {}).get('cc')!r}")
    if _clang_cl_selected(s6):
        add("s6_vs_absent_disables_compile",
            "PASS" if s6.get("dynamo_disabled") is True else "FAIL",
            f"clang-cl + Visual Studio hidden -> TORCHDYNAMO_DISABLE={s6.get('TORCHDYNAMO_DISABLE')!r} "
            f"(want '1'); {s6.get('toolchain_summary')}")
    else:
        add("s6_vs_absent_disables_compile", "FAIL",
            f"CC override did not make Triton pick clang-cl: cc={s6.get('triton', {}).get('cc')!r}")

    # False-positive guards: no AMD compiler means the gate must never fire, VS or no VS.
    for sid, label in (("s1_baseline_no_amd_vs_present", "stock runner"),
                       ("s4_no_amd_vs_absent", "no AMD compiler, Visual Studio hidden")):
        s = sc.get(sid, {})
        add(sid, "PASS" if s.get("dynamo_disabled") is False else "FAIL",
            f"{label} -> TORCHDYNAMO_DISABLE={s.get('TORCHDYNAMO_DISABLE')!r} (want unset); "
            f"cc={s.get('triton', {}).get('cc_basename')!r}")

    s7 = sc.get("s7_no_triton", {})
    add("s7_no_triton_disables_compile",
        "PASS" if s7.get("dynamo_disabled") is True else "FAIL",
        f"triton absent -> TORCHDYNAMO_DISABLE={s7.get('TORCHDYNAMO_DISABLE')!r} (want '1')")

    # Higher-fidelity pair: only asserted when the ROCm wheel layout actually moved Triton's
    # choice on this release. If it did not, that is a fact about the wheel, not a gate failure,
    # so it is reported rather than failed -- s5/s6 already pin the behaviour.
    s2, s3 = sc.get("s2_amd_wheel_vs_present", {}), sc.get("s3_amd_wheel_vs_absent", {})
    if _clang_cl_selected(s2) and _clang_cl_selected(s3):
        add("s2_amd_wheel_vs_present", "PASS" if s2.get("dynamo_disabled") is False else "FAIL",
            f"ROCm wheel clang-cl + VS present -> {s2.get('TORCHDYNAMO_DISABLE')!r} (want unset)")
        add("s3_amd_wheel_vs_absent", "PASS" if s3.get("dynamo_disabled") is True else "FAIL",
            f"ROCm wheel clang-cl + VS hidden -> {s3.get('TORCHDYNAMO_DISABLE')!r} (want '1')")
    else:
        add("s2_s3_rocm_wheel_layout", "INFO",
            "this triton-windows release did not prefer the planted ROCm clang-cl "
            f"(cc={s2.get('triton', {}).get('cc_basename')!r}); s5/s6 cover the compiler choice")

    # Did hiding Visual Studio actually work? Reported, and required, because a VS-absent
    # scenario that silently still sees VS would make s6 vacuous.
    t6 = s6.get("triton", {})
    hidden = t6.get("msvc_bin_path") is None and t6.get("found_vcruntime_h") is False
    add("vs_hidden_effectively", "PASS" if hidden else "FAIL",
        f"msvc_bin_path={t6.get('msvc_bin_path')!r}, include_dirs={t6.get('include_dir_count')}, "
        f"stdlib.h={t6.get('found_stdlib_h')}, vcruntime.h={t6.get('found_vcruntime_h')}")

    # The state the PR makes reachable for the first time: Triton importable AND dynamo disabled.
    add("triton_present_and_dynamo_disabled",
        "PASS" if (s6.get("dynamo_disabled") and "triton_import_error" not in t6) else "FAIL",
        f"triton {t6.get('triton_version')!r} imported and TORCHDYNAMO_DISABLE="
        f"{s6.get('TORCHDYNAMO_DISABLE')!r}")
    return out


if __name__ == "__main__":
    sys.exit(main())
