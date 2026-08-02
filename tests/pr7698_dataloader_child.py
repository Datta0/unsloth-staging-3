"""Drive a real torch DataLoader with workers, under this OS's default start method.

Run as a script, never imported by the test process. The `if __name__ == "__main__"`
guard is mandatory: on Windows and macOS the default start method is `spawn`, and a
spawned worker re-imports `__main__`. Doing this inside pytest re-enters pytest in
the child, which is why the first iteration of this probe reported
"DataLoader worker exited unexpectedly" for both variants and gated on nothing.

  python pr7698_dataloader_child.py {pre-PR,PR} {zoo_lambda,picklable}
prints one line: OK | <ExceptionType>: <message>
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# unsloth_zoo/__init__.py refuses to import unless it can see unsloth
# (find_spec("unsloth") plus the UNSLOTH_IS_PRESENT env gate), so the repo root
# has to be importable in this child too - it is not, by default, because
# sys.path[0] is the script's own directory.
sys.path.insert(1, os.path.dirname(_HERE))
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

# Same trick as tests/conftest.py: unsloth_zoo.device_type calls get_device_type()
# at import time and raises on a GPU-less runner. Pre-load it under a mocked
# is_available() so its @cache captures "cuda". Must run at module scope, because
# a spawned DataLoader worker re-imports __main__ and needs the same treatment.
try:
    import torch as _torch
    if not _torch.cuda.is_available():
        _real_is_available = _torch.cuda.is_available
        _torch.cuda.is_available = lambda: True
        try:
            import unsloth_zoo.device_type  # noqa: F401
        finally:
            _torch.cuda.is_available = _real_is_available
except Exception:  # noqa: BLE001 - a real accelerator, or zoo not installed yet
    pass


def main():
    from torch.utils.data import DataLoader
    from unsloth_zoo.vision_utils import UnslothVisionDataCollator as ZooBase

    import pr7698_collators as models

    variant, size_mode = sys.argv[1], sys.argv[2]
    collator = models.build(
        models.VARIANTS[variant], models.module_formatter,
        with_zoo_lambda = (size_mode == "zoo_lambda"),
    )
    # Patch on the class so the workers, which re-import this module, see it too.
    ZooBase.__call__ = models.flat_base_call

    loader = DataLoader(
        [{"tag": i} for i in range(4)], batch_size = 2, num_workers = 2,
        collate_fn = models.PassThrough(collator),
    )
    total = 0
    for batch in loader:
        total += batch["n"]
    if total != 4:
        raise AssertionError(f"collated {total} of 4 rows")
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"{type(e).__name__}: {str(e).splitlines()[0][:120]}")
        sys.exit(0)
