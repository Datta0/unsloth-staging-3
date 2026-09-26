# Spoofed-GPU detection matrix for unslothai/unsloth#11944 (staging-only overlay, never on the PR).
"""One subprocess per cell: {Linux, Windows, Darwin} x {nvidia, amd_rocm, intel_xpu, apple_mps,
cpu_only} x {healthy, each defect #11944 guards}. Every cell runs the PR's real Studio entrypoint
(detect_hardware, _devices_that_can_establish_a_mismatch over _linux_drm_sysfs_records,
_gpu_present_but_unusable_message, _nvidia_smi_executable, _query_gpu_inventory) twice:
HEAD (the checkout) and BASE (merge-base copies of the touched files, vendored under
tests/_spoof_base_11944/, overlaid on a temp copy of studio/backend/utils).

Cell ids: `<OS>-patched|...` = platform.system()/machine() patched in the child (cross-OS logic,
runs on every host); `<OS>-real|...` = needs the real OS (PATH lookup, nvidia-smi[.bat] exec,
Windows ProgramFiles/SystemRoot paths, WSL path exec), skipped elsewhere with the reason.

test_head_verdict  : head output == the verdict the PR intends.
test_base_vs_head  : EXPECT 'same' -> base output == head output;
                     EXPECT 'changed:<why>' -> base != head AND base == the known old behaviour.
torch is a stub module injected per child (no real torch needed; identical on every runner).
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
HEAD_ROOT = Path(os.environ.get("SPOOF_HEAD_ROOT") or HERE.parent).resolve()
HEAD_BACKEND = HEAD_ROOT / "studio" / "backend"
BASE_COPIES = HERE / "_spoof_base_11944" / "studio" / "backend"
HOST = platform.system()
OSES = ("Linux", "Windows", "Darwin")
VENDORS = ("nvidia", "amd_rocm", "intel_xpu", "apple_mps", "cpu_only")

# --------------------------------------------------------------------------- child runner

CHILD = textwrap.dedent(r'''
import json, os, sys, types
from types import SimpleNamespace as NS
cell = json.loads(sys.argv[1]); paths = json.loads(sys.argv[2])
for p in reversed(paths):
    sys.path.insert(0, p)

def boom(*_a, **_k):
    raise RuntimeError("spoofed probe failure")

v, d = cell.get("vendor"), cell.get("defect")
if cell.get("os_mode") == "patched":
    import platform
    platform.system = lambda: cell["os"]
    platform.machine = lambda: ("arm64" if (cell["os"] == "Darwin" and v == "apple_mps") else "x86_64")

# ---- stub torch
t = types.ModuleType("torch")
cuda_ok = v in ("nvidia", "amd_rocm")
xpu_ok = v == "intel_xpu"
t.__version__ = {"amd_rocm": "2.11.0+rocm7.2", "intel_xpu": "2.11.0+xpu", "cpu_only": "2.11.0+cpu",
                 "apple_mps": "2.11.0"}.get(v, "2.11.0+cu128")
t.version = NS(hip="7.2.1" if v == "amd_rocm" else None,
               cuda="12.8" if v == "nvidia" else None,
               xpu="20250101" if v == "intel_xpu" else None)
t.cuda = NS(is_available=boom if d == "cuda_is_available_raises" else (lambda: cuda_ok),
            device_count=lambda: 1 if cuda_ok else 0,
            get_device_properties=boom if d == "get_device_properties_raises"
            else (lambda _i: NS(name="AMD Radeon RX 7900 XTX" if v == "amd_rocm" else "NVIDIA RTX A4000")))
t.xpu = NS(is_available=boom if d == "xpu_is_available_raises" else (lambda: xpu_ok),
           device_count=lambda: 1 if xpu_ok else 0,
           get_device_name=boom if d in ("xpu_get_device_name_raises", "force_xpu_get_device_name_raises")
           else (lambda _i: "Intel(R) Arc(TM) B580 Graphics"))
t.backends = NS(mps=NS(is_available=lambda: v == "apple_mps", is_built=lambda: v == "apple_mps"))
sys.modules["torch"] = t

for k in ("UNSLOTH_FORCE_XPU", "ZE_AFFINITY_MASK", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
          "ROCR_VISIBLE_DEVICES"):
    os.environ.pop(k, None)
if d == "force_xpu_get_device_name_raises":
    os.environ["UNSLOTH_FORCE_XPU"] = "1"

import io, contextlib
fam = cell["family"]
out = {}
with contextlib.redirect_stdout(io.StringIO()):
    import utils.hardware.hardware as hw
    from utils.hardware import nvidia
    out["hw_file"] = hw.__file__
    if fam == "detect":
        hw.TORCH_IMPORT_ERROR = None
        hw._has_usable_mlx_stack = lambda: v == "apple_mps"
        hw._installed_without_torch = lambda: False
        hw._mlx_stack_detail = lambda: None
        hw._mismatch_verdict_for_this_host = lambda *_a, **_k: (None, None)  # no host inventory
        try:
            dev = hw.detect_hardware()
            out["verdict"] = [getattr(dev, "value", str(dev)), hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM]
        except Exception as e:
            out["verdict"] = "raised:" + type(e).__name__
    elif fam == "mismatch":
        # /sys/class/drm redirected to a temp tree; realpath stubbed (Windows cannot name a dir "0000:03:00.0").
        root = cell["tmp"]; card = os.path.join(root, "card0", "device"); os.makedirs(card)
        with open(os.path.join(card, "vendor"), "w") as fh:
            fh.write({"amd": "0x1002"}.get(cell["kind"], "0x8086") + "\n")
        if cell["kind"] == "amd":
            with open(os.path.join(card, "mem_info_vram_total"), "w") as fh:
                fh.write(str(16 * 1024**3))
        real_listdir, real_open, real_realpath = os.listdir, open, os.path.realpath
        def redirect(p):
            p = str(p)
            return p.replace("/sys/class/drm", root, 1) if p.startswith("/sys/class/drm") else p
        os.listdir = lambda p=".": real_listdir(redirect(p))
        hw.open = lambda p, *a, **k: real_open(redirect(p), *a, **k)
        addr = {"intel_arc_bus03": "0000:03:00.0", "intel_igpu_bus00": "0000:00:02.0",
                "intel_unparseable_address": "platform-gpu", "amd": "0000:0c:00.0"}[cell["kind"]]
        os.path.realpath = lambda p, *a, **k: (f"/sys/devices/pci0000:00/{addr}"
                                               if str(p).replace("\\", "/").endswith("card0/device")
                                               else real_realpath(p, *a, **k))
        hw._expected_xpu_flavor_was_chosen = lambda: False
        hw._torch_reports_an_xpu_runtime = lambda: False
        hw._vendors_masked_off = lambda **_k: set()
        hw._expected_rocm_flavor_was_chosen = lambda: True
        recs = hw._linux_drm_sysfs_records()
        out["verdict"] = {"records": [[r["vendor"], r.get("discrete", "absent")] for r in recs],
                          "kept": [r["vendor"] for r in hw._devices_that_can_establish_a_mismatch(recs)]}
    elif fam == "hint":
        hw.CHAT_ONLY_MISMATCH_VENDORS = frozenset(cell["vendors"])
        hw._torch_reports_a_hip_runtime = lambda: False
        hw._expected_rocm_flavor_was_chosen = lambda: False
        hw._torch_reports_another_vendors_runtime = lambda: False
        import utils.hardware.amd as amd
        amd.amd_closed_nodes_block_the_runtime = lambda *_a, **_k: False
        out["verdict"] = hw._gpu_present_but_unusable_message("training", (cell["reason"], "2.11.0+cpu"))
    elif fam == "nvsmi":
        import shutil as _sh
        for k in ("ProgramFiles", "SystemRoot"):
            os.environ.pop(k, None)
        nvidia.shutil.which = (lambda _n: "/opt/fake/nvidia-smi") if d == "on_path" else (lambda _n: None)
        real_isfile = os.path.isfile
        wsl = "/usr/lib/wsl/lib/nvidia-smi"
        nvidia.os.path.isfile = lambda p: (d == "wsl_fallback") if p == wsl else real_isfile(p)
        out["verdict"] = nvidia._nvidia_smi_executable()
    elif fam == "nvsmi_real":
        # real OS: real shutil.which / os.path.isfile / subprocess; env prepared by the parent.
        if d == "wsl_nvidia_smi_off_path":
            nvidia._WSL_NVIDIA_SMI = cell["fake"]  # head reads it; base has no such constant
        saved = {k: os.environ.get(k) for k in cell["late_env"]}
        def _apply(vals):
            for k, v in vals.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        _apply(cell["late_env"])
        exe = nvidia._nvidia_smi_executable()
        _apply(saved)
        out["exe"] = exe
        if cell.get("query"):
            inv = nvidia._query_gpu_inventory("spoof-matrix")
            inv = "NVIDIA_SMI_ABSENT" if inv is nvidia.NVIDIA_SMI_ABSENT else inv
        else:
            inv = None
        out["verdict"] = {"exe": os.path.normcase(exe), "inventory": inv}
print("SPOOF_RESULT " + json.dumps(out))
''')

# --------------------------------------------------------------------------- cells


def _detect_model(vendor, defect, osname, base):
    """Expected detect_hardware() verdict: [device, chat_only, reason, is_rocm] or 'raised:X'."""
    cuda = vendor in ("nvidia", "amd_rocm")
    cuda_raises = defect == "cuda_is_available_raises"
    xpu_raises = defect == "xpu_is_available_raises"
    name_raises = defect in ("xpu_get_device_name_raises", "force_xpu_get_device_name_raises")
    force = defect == "force_xpu_get_device_name_raises"
    xpu = vendor == "intel_xpu" and not xpu_raises
    if force and xpu:  # prefer-XPU branch (guarded is_available on both sides)
        return "raised:RuntimeError" if (base and name_raises) else ["xpu", False, None, False]
    if cuda_raises and base:
        return "raised:RuntimeError"  # base re-called torch.cuda.is_available() unguarded
    if cuda and not cuda_raises:
        return ["cuda", False, None, vendor == "amd_rocm"]
    if xpu_raises and base:
        return "raised:RuntimeError"  # base second XPU probe unguarded
    if xpu:
        return "raised:RuntimeError" if (base and name_raises) else ["xpu", False, None, False]
    if osname == "Darwin" and vendor == "apple_mps":
        return ["mlx", False, None, False]
    return ["cpu", True, "intel_mac" if osname == "Darwin" else "no_gpu", False]


DETECT_DEFECTS = ("healthy", "cuda_is_available_raises", "get_device_properties_raises",
                  "xpu_is_available_raises", "xpu_get_device_name_raises",
                  "force_xpu_get_device_name_raises")

REPAIR = ("Reinstall the GPU build: use Repair installation in Settings in the desktop app, "
          "or re-run the Unsloth installer.")
PIN = ("On Linux the Unsloth installer installs the Intel XPU build only when asked: re-run it "
       "with UNSLOTH_TORCH_INDEX_FAMILY=xpu set.")


def _hint_msg(reason, repair):
    if reason == "torch_cpu_build":
        return ("This host has a GPU, but the installed PyTorch is a CPU-only build (installed "
                f"2.11.0+cpu), so training cannot use it. {repair}")
    return ("This host has a GPU, but the installed PyTorch (installed 2.11.0+cpu) cannot initialise "
            "it, so training cannot use it. This is usually a driver or runtime mismatch; "
            "reinstalling a matching PyTorch build fixes it. Use Repair installation in Settings "
            "in the desktop app, or re-run the Unsloth installer.")


def _cells():
    cells = []

    def add(cid, cell, head, base_old=None, why=None, real_os=None, skip=None):
        cell = dict(cell, id=cid)
        cell["expect_head"] = head
        cell["expect"] = "same" if base_old is None else f"changed:{why}"
        cell["expect_base"] = head if base_old is None else base_old
        cell["real_os"] = real_os
        cell["skip"] = skip
        cells.append(cell)

    # 1. detect_hardware: full OS x vendor x defect product (cross-OS logic -> patched).
    for osn in OSES:
        for v in VENDORS:
            for d in DETECT_DEFECTS:
                h, b = _detect_model(v, d, osn, False), _detect_model(v, d, osn, True)
                add(f"{osn}-patched|{v}|detect:{d}",
                    dict(family="detect", os=osn, os_mode="patched", vendor=v, defect=d),
                    h, None if h == b else b,
                    None if h == b else "failing probe no longer raises out of detect_hardware")

    # 2. Linux sysfs DRM record -> mismatch filter (the Intel PCI-bus discrete rule).
    for kind, head, base in (
        ("intel_arc_bus03", {"records": [["intel", True]], "kept": ["intel"]},
         {"records": [["intel", "absent"]], "kept": []}),
        ("intel_igpu_bus00", {"records": [["intel", False]], "kept": []},
         {"records": [["intel", "absent"]], "kept": []}),
        ("intel_unparseable_address", {"records": [["intel", None]], "kept": []},
         {"records": [["intel", "absent"]], "kept": []}),
        ("amd", {"records": [["amd", "absent"]], "kept": ["amd"]}, None),
    ):
        why = ("nameless discrete Arc now establishes a mismatch" if kind == "intel_arc_bus03"
               else "record gains discrete field; mismatch verdict unchanged")
        add(f"Linux-patched|{'amd_rocm' if kind == 'amd' else 'intel_xpu'}|sysfs:{kind}",
            dict(family="mismatch", os="Linux", os_mode="patched", vendor=None, defect=kind, kind=kind),
            head, base, None if base is None else why)

    # 3. repair hint wording: OS x recorded vendors x reason.
    for osn in OSES:
        for vs in (("intel",), ("nvidia",), ("amd",), ("intel", "nvidia")):
            for reason in ("torch_cpu_build", "torch_cuda_unavailable"):
                pinned = osn == "Linux" and vs == ("intel",) and reason == "torch_cpu_build"
                head = _hint_msg(reason, PIN if pinned else REPAIR)
                base = _hint_msg(reason, REPAIR)
                vlabel = "+".join(vs)
                add(f"{osn}-patched|{vlabel}|hint:{reason}",
                    dict(family="hint", os=osn, os_mode="patched", vendor=None, defect=reason,
                         vendors=list(vs), reason=reason),
                    head, None if head == base else base,
                    "Linux Intel-only host told the XPU pin, not a CPU-reinstalling Repair")

    # 4. nvidia-smi resolution logic (which/isfile patched, cross-OS).
    for osn in OSES:
        for d in ("on_path", "absent_from_path", "wsl_fallback"):
            head = {"on_path": "/opt/fake/nvidia-smi"}.get(d, "nvidia-smi")
            if d == "wsl_fallback" and osn == "Linux":
                head = "/usr/lib/wsl/lib/nvidia-smi"
            base = "/opt/fake/nvidia-smi" if d == "on_path" else "nvidia-smi"
            add(f"{osn}-patched|nvidia|nvsmi:{d}",
                dict(family="nvsmi", os=osn, os_mode="patched", vendor="nvidia", defect=d),
                head, None if head == base else base, "WSL nvidia-smi found off PATH")

    # 5. real-OS cells: real PATH lookup + real exec of a fake nvidia-smi, Windows install dirs.
    rows = [{"index": 0, "name": "NVIDIA RTX A4000", "memory_total_gb": 16.0}]
    for osn in OSES:
        add(f"{osn}-real|nvidia|nvsmi_real:on_path_exec",
            dict(family="nvsmi_real", os=osn, os_mode="real", vendor="nvidia",
                 defect="on_path_exec", query=True),
            {"exe": "<fake>", "inventory": rows}, real_os=osn)
        add(f"{osn}-real|cpu_only|nvsmi_real:absent_from_path",
            dict(family="nvsmi_real", os=osn, os_mode="real", vendor="cpu_only",
                 defect="absent_from_path", query=True),
            {"exe": "nvidia-smi", "inventory": "NVIDIA_SMI_ABSENT"}, real_os=osn)
    for d in ("programfiles_nvsmi_off_path", "systemroot_system32_off_path"):
        add(f"Windows-real|nvidia|nvsmi_real:{d}",
            dict(family="nvsmi_real", os="Windows", os_mode="real", vendor="nvidia", defect=d),
            {"exe": "<fake>", "inventory": None}, real_os="Windows")
    add("Linux-real|nvidia|nvsmi_real:wsl_nvidia_smi_off_path",
        dict(family="nvsmi_real", os="Linux", os_mode="real", vendor="nvidia",
             defect="wsl_nvidia_smi_off_path", query=True),
        {"exe": "<fake>", "inventory": rows},
        {"exe": "nvidia-smi", "inventory": "NVIDIA_SMI_ABSENT"},
        "WSL nvidia-smi found off PATH and queried", real_os="Linux")
    return cells


CELLS = _cells()

# --------------------------------------------------------------------------- execution


@pytest.fixture(scope="session")
def base_backend(tmp_path_factory):
    assert (HEAD_BACKEND / "utils" / "hardware" / "hardware.py").is_file(), f"no head tree at {HEAD_ROOT}"
    assert (BASE_COPIES / "utils" / "hardware" / "hardware.py").is_file(), "vendored base copies missing"
    root = tmp_path_factory.mktemp("base_backend")
    shutil.copytree(HEAD_BACKEND / "utils", root / "utils",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"))
    for f in BASE_COPIES.rglob("*.py"):
        dst = root / f.relative_to(BASE_COPIES)
        shutil.copy2(f, dst)
    return root


_RESULTS: dict = {}


def _fake_nvidia_smi(d: Path) -> Path:
    """A nvidia-smi answering the inventory query, as a .bat (Windows) or /bin/sh script."""
    d.mkdir(parents=True, exist_ok=True)
    py = d / "fake_nvidia_smi.py"
    py.write_text("print('0, NVIDIA RTX A4000, 16384')\n")
    if HOST == "Windows":
        exe = d / "nvidia-smi.bat"
        exe.write_text(f'@echo off\r\n"{sys.executable}" "{py}" %*\r\n')
    else:
        exe = d / "nvidia-smi"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{py}" "$@"\n')
        exe.chmod(0o755)
    return exe


def _run(cell, arm, base_root, tmp: Path):
    key = (cell["id"], arm)
    if key in _RESULTS:
        return _RESULTS[key]
    tmp.mkdir(parents=True, exist_ok=True)
    paths = [str(HEAD_BACKEND)] if arm == "head" else [str(base_root), str(HEAD_BACKEND)]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    cell = dict(cell)
    fake_exe = None
    if cell["family"] == "mismatch":
        cell["tmp"] = str(tmp / "drm")
    if cell["family"] == "nvsmi_real":
        empty = tmp / "empty_path"
        empty.mkdir(parents=True, exist_ok=True)
        env["PATH"] = str(empty)
        # Applied in the child just before the lookup: Windows Python needs the real
        # SystemRoot to start (Winsock raises WinError 10106 without it).
        late = {"ProgramFiles": None, "SystemRoot": None}
        if HOST == "Windows":
            late["SystemRoot"] = str(tmp / "no_sysroot")  # isolate from the runner's System32
        d = cell["defect"]
        if d == "on_path_exec":
            fake_exe = _fake_nvidia_smi(tmp / "bin")
            env["PATH"] = str(fake_exe.parent)
            if HOST == "Windows":
                # cmd.exe runs the .bat; the real System32 carries no nvidia-smi on hosted runners.
                env["PATH"] += os.pathsep + os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
                late["SystemRoot"] = os.environ.get("SystemRoot", r"C:\Windows")
        elif d == "programfiles_nvsmi_off_path":
            fake_exe = tmp / "pf" / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
            fake_exe.parent.mkdir(parents=True)
            fake_exe.write_bytes(b"")
            late["ProgramFiles"] = str(tmp / "pf")
        elif d == "systemroot_system32_off_path":
            fake_exe = tmp / "sr" / "System32" / "nvidia-smi.exe"
            fake_exe.parent.mkdir(parents=True)
            fake_exe.write_bytes(b"")
            late["SystemRoot"] = str(tmp / "sr")
        elif d == "wsl_nvidia_smi_off_path":
            fake_exe = _fake_nvidia_smi(tmp / "wsl_lib")
            cell["fake"] = str(fake_exe)
        cell["late_env"] = late
    proc = subprocess.run([sys.executable, "-c", CHILD, json.dumps(cell), json.dumps(paths)],
                          capture_output=True, text=True, env=env, timeout=120, cwd=str(tmp))
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("SPOOF_RESULT ")), None)
    if line is None:
        raise AssertionError(f"{cell['id']} [{arm}] child failed rc={proc.returncode}\n"
                             f"stdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}")
    out = json.loads(line[len("SPOOF_RESULT "):])
    # arm isolation proof: the child imported the tree we meant it to.
    hw_file = Path(out["hw_file"]).resolve()
    want = HEAD_BACKEND if arm == "head" else base_root
    assert str(hw_file).startswith(str(Path(want).resolve())), f"{arm} imported {hw_file}"
    verdict = out["verdict"]
    if fake_exe is not None and isinstance(verdict, dict) and \
            verdict.get("exe") == os.path.normcase(str(fake_exe)):
        verdict = dict(verdict, exe="<fake>")
    _RESULTS[key] = verdict
    return verdict


def _gate(cell):
    if cell["skip"]:
        pytest.skip(cell["skip"])
    if cell["real_os"] and cell["real_os"] != HOST:
        pytest.skip(f"real-OS cell: needs a {cell['real_os']} host (this is {HOST}); "
                    f"its cross-OS logic is covered by the {cell['real_os']}-patched cells")
    if cell["id"].startswith("Linux-real") and os.path.exists("/usr/lib/wsl/lib/nvidia-smi"):
        pytest.skip("host is WSL: a real /usr/lib/wsl/lib/nvidia-smi exists and would answer")


@pytest.mark.parametrize("cell", CELLS, ids=[c["id"] for c in CELLS])
def test_head_verdict(cell, base_backend, tmp_path):
    _gate(cell)
    got = _run(cell, "head", base_backend, tmp_path / "head")
    assert got == cell["expect_head"], f"{cell['id']} head verdict {got!r} != intended {cell['expect_head']!r}"


@pytest.mark.parametrize("cell", CELLS, ids=[c["id"] for c in CELLS])
def test_base_vs_head(cell, base_backend, tmp_path):
    _gate(cell)
    head = _run(cell, "head", base_backend, tmp_path / "head")
    base = _run(cell, "base", base_backend, tmp_path / "base")
    if cell["expect"] == "same":
        assert base == head, f"{cell['id']} expected IDENTICAL base vs head: base={base!r} head={head!r}"
    else:
        assert base != head, f"{cell['id']} expected a change ({cell['expect']}) but base == head == {head!r}"
        assert base == cell["expect_base"], f"{cell['id']} base {base!r} != known old behaviour {cell['expect_base']!r}"


def test_matrix_shape():
    """Counts pinned so a silently shrunk matrix fails."""
    fams = {}
    for c in CELLS:
        fams[c["family"]] = fams.get(c["family"], 0) + 1
    assert fams == {"detect": 90, "mismatch": 4, "hint": 24, "nvsmi": 9, "nvsmi_real": 9}, fams
    assert len({c["id"] for c in CELLS}) == len(CELLS)
