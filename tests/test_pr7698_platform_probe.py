"""Platform probe for unslothai/unsloth PR #7698, run on real Windows / macOS / Linux.

The PR adds a `threading.Lock` to `UnslothVisionDataCollator` and wraps the whole
`__call__` in it. Several consequences are platform-dependent and cannot be settled
on a Linux dev box:

  * the multiprocessing start method a torch DataLoader actually uses (`spawn` on
    Windows and macOS, `fork` on Linux before 3.14, `forkserver` from 3.14), which
    decides whether `collate_fn` must be picklable at all;
  * whether the collator is picklable / deepcopyable / cloudpicklable there;
  * whether `unsloth` takes the torch path at all on Apple Silicon, or diverts to
    the MLX placeholder collator.

To get a before/after on the same runner, both `__call__` bodies are reconstructed
as sibling subclasses of the installed zoo base (`pr7698_collators.py`). Only
`unsloth_zoo` + torch are needed, so macOS still reports real data when `unsloth`
itself will not import.

Diagnostics print unconditionally; only properties that must hold are asserted.
"""

import copy
import multiprocessing
import os
import pickle
import platform
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("unsloth_zoo.vision_utils")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pr7698_collators as models  # noqa: E402

ZooBase = models.ZooBase
CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "pr7698_dataloader_child.py")


def test_platform_report(capsys):
    """Always-printed facts about the runner. Never fails."""
    with capsys.disabled():
        print("\n=== PR7698 PLATFORM REPORT ===")
        print(f"  system            : {platform.system()} {platform.release()} "
              f"({platform.machine()})")
        print(f"  python            : {platform.python_version()} "
              f"({platform.python_implementation()})")
        print(f"  gil enabled       : {getattr(sys, '_is_gil_enabled', lambda: True)()}")
        print(f"  default start     : {multiprocessing.get_start_method(allow_none = False)}")
        print(f"  all start methods : {multiprocessing.get_all_start_methods()}")
        print(f"  torch             : {torch.__version__}")
        try:
            import unsloth
            print(f"  unsloth           : {unsloth.__version__} "
                  f"(DEVICE_TYPE={getattr(unsloth, 'DEVICE_TYPE', '?')})")
        except Exception as e:  # noqa: BLE001
            print(f"  unsloth           : NOT IMPORTABLE "
                  f"({type(e).__name__}: {str(e)[:70]})")
            return
        try:
            from unsloth.trainer import UnslothVisionDataCollator as Real
        except Exception as e:  # noqa: BLE001
            print(f"  unsloth.trainer   : NOT IMPORTABLE "
                  f"({type(e).__name__}: {str(e)[:70]})")
            return
        is_zoo_subclass = isinstance(Real, type) and issubclass(Real, ZooBase)
        print(f"  real collator     : {Real.__module__}.{Real.__qualname__}")
        print(f"  is the zoo subclass: {is_zoo_subclass} "
              f"({'torch path' if is_zoo_subclass else 'MLX placeholder'})")
        print(f"  real __slots__    : {getattr(Real, '__slots__', '<none>')}")


def test_real_class_matches_the_reconstruction():
    """Where the torch path is live, the staged class must have the modelled shape."""
    unsloth_trainer = pytest.importorskip("unsloth.trainer")
    real = unsloth_trainer.UnslothVisionDataCollator
    if not (isinstance(real, type) and issubclass(real, ZooBase)):
        pytest.skip("MLX placeholder collator on this platform, not the torch subclass")
    assert "formatting_func" in ZooBase.__slots__, (
        "the modelled base/subclass relationship no longer matches the installed zoo"
    )


# ─── the race ────────────────────────────────────────────────────────────────

def _race_probe(cls, num_threads = 16, num_examples = 3):
    """Park a leader inside the base exactly inside the mutation window."""
    formatted, flock = [], threading.Lock()
    base_saw, block = [], threading.Lock()
    parked, release = threading.Event(), threading.Event()
    first, entered = threading.Lock(), []

    def formatter(example):
        with flock:
            formatted.append(example["tag"])
        return example

    def fake_base(self, examples):
        with block:
            base_saw.append(self.formatting_func)
        park = False
        with first:
            if not entered:
                entered.append(True)
                park = True
        if park:
            parked.set()
            release.wait(30)
        return examples

    original = ZooBase.__call__
    ZooBase.__call__ = fake_base
    try:
        collator = models.build(cls, formatter)
        work = lambda tid: collator([{"tag": (tid, i)} for i in range(num_examples)])
        with ThreadPoolExecutor(max_workers = num_threads) as pool:
            leader = pool.submit(work, 0)
            assert parked.wait(30), "leader never reached the base collator"
            followers = [pool.submit(work, t) for t in range(1, num_threads)]
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not all(f.done() for f in followers):
                time.sleep(0.01)
            release.set()
            leader.result(timeout = 60)
            for f in followers:
                f.result(timeout = 60)
    finally:
        ZooBase.__call__ = original
        release.set()

    return {"formatted": len(formatted), "expected": num_threads * num_examples,
            "base_saw_none": sum(x is None for x in base_saw), "base_calls": len(base_saw)}


def test_race_exists_before_the_pr(capsys):
    result = _race_probe(models.PreFixCollator)
    with capsys.disabled():
        print(f"\n  pre-PR  race probe: formatted {result['formatted']}/{result['expected']}")
    assert result["formatted"] < result["expected"], (
        "the pre-PR code should lose formatter applications on this platform"
    )


def test_pr_fixes_the_race(capsys):
    result = _race_probe(models.ViewCollator)
    with capsys.disabled():
        print(f"  PR      race probe: formatted {result['formatted']}/{result['expected']}")
    assert result["formatted"] == result["expected"]
    assert result["base_saw_none"] == result["base_calls"]


# ─── serialization ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("with_zoo_lambda", [True, False],
                         ids = ["zoo_size_func_lambda", "picklable_size_func"])
def test_serialization_before_and_after(capsys, with_zoo_lambda):
    """Records, does not gate.

    The serialization regression is already established and is identical on every
    OS, so gating three runners on it would just triplicate one finding and mask
    anything genuinely platform-specific. Regressions are marked in the printout.
    """
    table = {}
    for label, cls in models.VARIANTS.items():
        collator = models.build(cls, models.module_formatter,
                                with_zoo_lambda = with_zoo_lambda)
        outcomes = {}
        probes = [("pickle", lambda c: pickle.dumps(c)),
                  ("deepcopy", lambda c: copy.deepcopy(c))]
        try:
            import cloudpickle
            probes.append(("cloudpickle", lambda c: cloudpickle.loads(cloudpickle.dumps(c))))
        except ImportError:
            outcomes["cloudpickle"] = "not installed"
        for name, op in probes:
            try:
                op(collator)
                outcomes[name] = "ok"
            except Exception as e:  # noqa: BLE001
                outcomes[name] = type(e).__name__
        table[label] = outcomes

    with capsys.disabled():
        print(f"\n  serialization "
              f"({'zoo lambda' if with_zoo_lambda else 'picklable size_func'}):")
        for label, outcomes in table.items():
            print(f"    {label:<7} " + "  ".join(f"{k}={v}" for k, v in outcomes.items()))

    regressed = [k for k, v in table["pre-PR"].items()
                 if v == "ok" and table["view"].get(k) not in ("ok", "not installed")]
    with capsys.disabled():
        print(f"    -> REGRESSED BY THE PR on {platform.system()}: {regressed or 'nothing'}")


# ─── the real DataLoader, in a child with a __main__ guard ───────────────────

@pytest.mark.parametrize("size_mode", ["zoo_lambda", "picklable"])
def test_dataloader_default_start_method(capsys, size_mode):
    """Does a real DataLoader with workers survive this OS's default start method?

    Runs in a child script: under `spawn` the workers re-import `__main__`, so
    doing this inside pytest re-enters pytest in the child and fails for reasons
    that have nothing to do with the collator.
    """
    outcomes = {}
    for label in models.VARIANTS:
        proc = subprocess.run([sys.executable, CHILD, label, size_mode],
                              capture_output = True, text = True, timeout = 300)
        outcomes[label] = (proc.stdout.strip().splitlines() or ["<no output>"])[-1]

    with capsys.disabled():
        method = multiprocessing.get_start_method(allow_none = False)
        print(f"\n  DataLoader(num_workers=2, start={method}, {size_mode}):")
        for label, outcome in outcomes.items():
            print(f"    {label:<7} {outcome}")

    assert not (outcomes["pre-PR"] == "OK" and outcomes["view"] != "OK"), (
        f"PR #7698 breaks DataLoader workers on {platform.system()} "
        f"(start method {multiprocessing.get_start_method(allow_none = False)}): "
        f"pre-PR={outcomes['pre-PR']} view={outcomes['view']}"
    )


# ─── reentrancy (recorded, not gated - it is a Lock property, not an OS one) ──

def test_reentrant_formatter(capsys):
    outcomes = []
    original = ZooBase.__call__
    ZooBase.__call__ = lambda self, examples: examples
    try:
        for label, cls in models.VARIANTS.items():
            collator = models.build(cls, None)
            depth = {"n": 0}

            def recursive(example, _c = collator, _d = depth):
                if _d["n"] == 0:
                    _d["n"] += 1
                    _c([{"inner": True}])
                return example

            collator.formatting_func = recursive
            done = threading.Event()

            def run(_c = collator, _done = done):
                try:
                    _c([{"outer": True}])
                finally:
                    _done.set()

            # daemon, so a deadlocked thread cannot keep the runner alive
            threading.Thread(target = run, daemon = True).start()
            outcomes.append((label, "completed" if done.wait(20) else "DEADLOCK (20s)"))
    finally:
        ZooBase.__call__ = original

    with capsys.disabled():
        print("\n  reentrant formatting_func:")
        for label, outcome in outcomes:
            print(f"    {label:<7} {outcome}")
