"""Shared collator variants for the PR #7698 platform probe.

Kept out of the test module so the DataLoader child process (which must run under
a real `if __name__ == "__main__":` guard for `spawn` to work on Windows/macOS)
can import the same classes the tests use.

The pre-PR and PR `__call__` bodies are reconstructed as sibling subclasses of the
installed unsloth_zoo base, so one runner reports both sides of the change.
"""

import copy as _copy
import threading

from unsloth_zoo.vision_utils import UnslothVisionDataCollator as ZooBase


def _noop_check(examples, raise_error = True, checked = None):
    """Stand-in for unsloth.models.vision.check_dataset_for_missing_videos.

    The real one lives in `unsloth`, which does not import on every runner; the
    video scan is irrelevant to what this probe measures.
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


class ViewCollator(ZooBase):
    """What the PR now carries after the review commit: no lock, a per-call
    shallow view, so self is never mutated."""

    __slots__ = ("_checked_video_paths",)

    def __call__(self, examples):
        formatting_func = self.formatting_func
        if formatting_func is not None:
            examples = [formatting_func(example) for example in examples]

        _noop_check(examples, raise_error = True, checked = self._checked_video_paths)

        if formatting_func is None:
            return super().__call__(examples)

        view = _copy.copy(self)
        view.formatting_func = None
        return super(ViewCollator, view).__call__(examples)


VARIANTS = {"pre-PR": PreFixCollator, "lock": PrCollator, "view": ViewCollator}


def module_formatter(example):
    """Module level, so the formatter is not itself a pickling obstacle."""
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
    else:
        collator.size_func = max
    return collator


class PassThrough:
    """Top level so `spawn` can pickle the collate_fn wrapper itself."""

    def __init__(self, collator):
        self.collator = collator

    def __call__(self, rows):
        return self.collator(rows)


def flat_base_call(self, examples):
    """Replaces the heavy zoo collate; module level so it survives pickling."""
    return {"n": len(examples)}
