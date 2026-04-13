"""Shared helper to extract grpo_compute_loss from unsloth_zoo source."""
import sys, os
import torch

def get_grpo_compute_loss():
    zoo_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..",
        "lib/python3.13/site-packages/unsloth_zoo/rl_replacements.py"
    )
    if not os.path.exists(zoo_path):
        for p in sys.path:
            candidate = os.path.join(p, "unsloth_zoo", "rl_replacements.py")
            if os.path.exists(candidate):
                zoo_path = candidate
                break
    assert os.path.exists(zoo_path), f"Cannot find unsloth_zoo/rl_replacements.py"

    with open(zoo_path) as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("def grpo_compute_loss("):
            start_idx = i
            break
    assert start_idx is not None

    # Find end: stop at RL_REPLACEMENTS line (first non-indented non-pass line)
    end_idx = start_idx + 1
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        if line.startswith("RL_REPLACEMENTS"):
            end_idx = i
            break
        end_idx = i + 1

    func_src = "".join(lines[start_idx:end_idx])
    ns = {"torch": torch}
    exec(compile(func_src, "<grpo_compute_loss>", "exec"), ns)
    return ns["grpo_compute_loss"]
