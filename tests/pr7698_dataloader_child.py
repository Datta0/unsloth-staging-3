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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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
