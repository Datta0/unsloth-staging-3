"""Platform probe for unslothai/unsloth PR #7698, run on real Windows / macOS / Linux.

The PR adds a `threading.Lock` to `UnslothVisionDataCollator` and wraps the whole
`__call__` in it. Three of the consequences are platform-dependent and cannot be
settled on a Linux dev box:

  * the multiprocessing start method a torch DataLoader will actually use
    (`spawn` on Windows and macOS, `fork` on Linux before 3.14, `forkserver` from
    3.14), which decides whether `collate_fn` has to be picklable at all;
  * whether the collator is picklable / deepcopyable / cloudpicklable there;
  * whether `unsloth` even takes the torch path on Apple Silicon, or diverts to
    the MLX placeholder.

To get a before/after on the same runner, the pre-PR `__call__` is reconstructed
as a sibling subclass of the same unsloth_zoo base. Only `unsloth_zoo` + torch are
required, so macOS still reports real data when `unsloth` itself will not import
(bitsandbytes has no arm64 wheel).

Diagnostics are printed unconditionally; only the properties that must hold are
asserted.
"""

import copy
import multiprocessing
import pickle
import platform
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

torch = pytest.importorskip("torch")
vision_utils = pytest.importorskip("unsloth_zoo.vision_utils")

ZooBase = vision_utils.UnslothVisionDataCollator


def _noop_check(examples, raise_error = True, checked = None):
    """Stand-in for unsloth.models.vision.check_dataset_for_missing_videos.

    The real one is in `unsloth`, which may not import on every runner; the video
    scan is irrelevant to what this file measures.
    """
    return []


class PreFixCollator(ZooBase):
    """unsloth/trainer.py as it is on main - unsynchronised, mutates self."""

    __slots__ = ("_checked_video_paths",)

    def __call__(self, examples):
        formatting_func = self.formatting_func
        if formatting_func is not None:
            examples = [formatting_func(example) for example in examples]

        _noop_check(examples, raise_error = True, checked = self._checked_video_paths)

        if formatting_func is None:
            return super().__call__(examples)

        self.formatting_func = None
        try:
            return super().__call__(examples)
        finally:
            self.formatting_func = formatting_func


class PrCollator(ZooBase):
    """unsloth/trainer.py as PR #7698 makes it - whole call under a Lock."""

    __slots__ = ("_checked_video_paths", "_formatting_lock")

    def __call__(self, examples):
        with self._formatting_lock:
            formatting_func = self.formatting_func

            if formatting_func is not None:
                examples = [formatting_func(example) for example in examples]

            _noop_check(examples, raise_error = True, checked = self._checked_video_paths)

            if formatting_func is None:
                return super().__call__(examples)

            self.formatting_func = None
            try:
                return super().__call__(examples)
            finally:
                self.formatting_func = formatting_func


def module_formatter(example):
    """Module level, so it is not itself a pickling obstacle."""
    return example


def build(cls, formatting_func = None, with_zoo_lambda = True):
    """Populate the slots the real __init__ would, without a model or processor."""
    collator = cls.__new__(cls)
    for slot in ZooBase.__slots__:
        setattr(collator, slot, None)
    collator.formatting_func = formatting_func
    collator._checked_video_paths = set()
    if "_formatting_lock" in cls.__slots__:
        collator._formatting_lock = threading.Lock()
    if with_zoo_lambda:
        # unsloth_zoo/vision_utils.py:858-862 always assigns a local lambda here.
        resize_dimension = 0
        collator.size_func = lambda x: x.size[resize_dimension]
    return collator


# ─────────────────────────────────────────────────────────────────────────────


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
            from unsloth.trainer import UnslothVisionDataCollator as Real
            print(f"  real __slots__    : {Real.__slots__}")
            print(f"  real is MLX stub  : "
                  f"{Real.__module__ == 'unsloth' or not issubclass(Real, ZooBase)}")
        except Exception as e:  # noqa: BLE001
            print(f"  unsloth           : NOT IMPORTABLE ({type(e).__name__}: {str(e)[:70]})")


def test_real_class_matches_the_reconstruction():
    """If unsloth imports here, the PR's real class must have the shape this file models."""
    unsloth_trainer = pytest.importorskip("unsloth.trainer")
    real = unsloth_trainer.UnslothVisionDataCollator
    if not issubclass(real, ZooBase):
        pytest.skip("MLX placeholder collator on this platform, not the torch subclass")
    assert "_formatting_lock" in real.__slots__, (
        "the staged branch should contain PR #7698's lock slot"
    )


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
        collator = build(cls, formatter)
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

    return {
        "formatted": len(formatted),
        "expected": num_threads * num_examples,
        "base_saw_none": sum(x is None for x in base_saw),
        "base_calls": len(base_saw),
    }


def test_race_exists_before_the_pr(capsys):
    result = _race_probe(PreFixCollator)
    with capsys.disabled():
        print(f"\n  pre-PR  race probe: formatted {result['formatted']}/{result['expected']}")
    assert result["formatted"] < result["expected"], (
        "the pre-PR code should lose formatter applications on this platform"
    )


def test_pr_fixes_the_race(capsys):
    result = _race_probe(PrCollator)
    with capsys.disabled():
        print(f"  PR      race probe: formatted {result['formatted']}/{result['expected']}")
    assert result["formatted"] == result["expected"]
    assert result["base_saw_none"] == result["base_calls"]


@pytest.mark.parametrize("with_zoo_lambda", [True, False],
                         ids = ["zoo_size_func_lambda", "picklable_size_func"])
def test_serialization_before_and_after(capsys, with_zoo_lambda):
    """Records, does not gate: what each serializer does on this OS, pre-PR vs PR."""
    rows = []
    for label, cls in (("pre-PR", PreFixCollator), ("PR", PrCollator)):
        collator = build(cls, module_formatter, with_zoo_lambda = with_zoo_lambda)
        if not with_zoo_lambda:
            collator.size_func = max
        outcomes = {}
        for name, op in (
            ("pickle", lambda c = collator: pickle.dumps(c)),
            ("deepcopy", lambda c = collator: copy.deepcopy(c)),
        ):
            try:
                op()
                outcomes[name] = "ok"
            except Exception as e:  # noqa: BLE001
                outcomes[name] = type(e).__name__
        try:
            import cloudpickle
            try:
                cloudpickle.loads(cloudpickle.dumps(collator))
                outcomes["cloudpickle"] = "ok"
            except Exception as e:  # noqa: BLE001
                outcomes["cloudpickle"] = type(e).__name__
        except ImportError:
            outcomes["cloudpickle"] = "not installed"
        rows.append((label, outcomes))
    with capsys.disabled():
        print(f"\n  serialization ({'zoo lambda' if with_zoo_lambda else 'picklable size_func'}):")
        for label, outcomes in rows:
            print(f"    {label:<7} " + "  ".join(f"{k}={v}" for k, v in outcomes.items()))


class _PassThrough:
    """Top level so spawn can pickle the wrapper itself."""

    def __init__(self, collator):
        self.collator = collator

    def __call__(self, rows):
        return self.collator(rows)


@pytest.mark.parametrize("with_zoo_lambda", [True, False],
                         ids = ["zoo_size_func_lambda", "picklable_size_func"])
def test_dataloader_default_start_method(capsys, with_zoo_lambda):
    """The decisive platform question: does a real DataLoader with workers survive?

    Uses whatever start method this OS defaults to, i.e. spawn on Windows/macOS.
    """
    from torch.utils.data import DataLoader

    rows_out = []
    for label, cls in (("pre-PR", PreFixCollator), ("PR", PrCollator)):
        collator = build(cls, module_formatter, with_zoo_lambda = with_zoo_lambda)
        if not with_zoo_lambda:
            collator.size_func = max
        original = ZooBase.__call__
        ZooBase.__call__ = lambda self, examples: {"n": len(examples)}
        try:
            loader = DataLoader(
                [{"tag": i} for i in range(4)], batch_size = 2, num_workers = 2,
                collate_fn = _PassThrough(collator),
            )
            for _ in loader:
                pass
            rows_out.append((label, "ok"))
        except Exception as e:  # noqa: BLE001
            rows_out.append((label, f"{type(e).__name__}: {str(e)[:60]}"))
        finally:
            ZooBase.__call__ = original

    with capsys.disabled():
        method = multiprocessing.get_start_method(allow_none = False)
        print(f"\n  DataLoader(num_workers=2, start={method}, "
              f"{'zoo lambda' if with_zoo_lambda else 'picklable size_func'}):")
        for label, outcome in rows_out:
            print(f"    {label:<7} {outcome}")

    # The gate: the PR must not newly break a configuration that worked before it.
    before, after = dict(rows_out)["pre-PR"], dict(rows_out)["PR"]
    assert not (before == "ok" and after != "ok"), (
        f"PR #7698 breaks DataLoader workers on {platform.system()} "
        f"(start method {multiprocessing.get_start_method(allow_none = False)}): "
        f"pre-PR={before} PR={after}"
    )


def test_reentrant_formatter(capsys):
    """A formatter that re-enters the collator.

    Records only: the outcome is a property of `threading.Lock`, not of the OS, so
    gating three runners on it would just triplicate one finding. Guarded with a
    daemon thread so a deadlock cannot keep the runner alive.
    """
    outcomes = []
    original = ZooBase.__call__
    ZooBase.__call__ = lambda self, examples: examples
    try:
        for label, cls in (("pre-PR", PreFixCollator), ("PR", PrCollator)):
            collator = build(cls, None)
            depth = {"n": 0}

            def recursive(example, _c = collator, _d = depth):
                if _d["n"] == 0:
                    _d["n"] += 1
                    _c([{"inner": True}])
                return example

            collator.formatting_func = recursive
            done = threading.Event()
            errors = []

            def run():
                try:
                    collator([{"outer": True}])
                except Exception as e:  # noqa: BLE001
                    errors.append(e)
                finally:
                    done.set()

            # daemon, so a deadlocked thread cannot keep the runner alive
            threading.Thread(target = run, daemon = True).start()
            finished = done.wait(20)
            outcomes.append((label, "completed" if finished else "DEADLOCK (20s)"))
    finally:
        ZooBase.__call__ = original

    with capsys.disabled():
        print("\n  reentrant formatting_func:")
        for label, outcome in outcomes:
            print(f"    {label:<7} {outcome}")
