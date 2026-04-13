"""Test that pre-PR call pattern raises TypeError due to duplicate sampling_per_token_logps."""
import torch

def grpo_compute_loss(ref, new, old, sampling_per_token_logps, input_ids, mask, beta, advantages, **kwargs):
    return "ok"

# Pre-PR call shape: missing 4th positional, duplicate kwarg
def test_pre_pr_raises():
    args = [torch.zeros(2,4)] * 3  # ref, new, old
    sampling = torch.zeros(2,4)
    input_ids = torch.zeros(2,4, dtype=torch.long)
    mask = torch.ones(2,4)
    beta = 0.1
    adv = torch.ones(2)
    try:
        grpo_compute_loss(
            *args,
            input_ids,       # lands in sampling_per_token_logps slot
            mask,
            beta,
            adv,
            sampling_per_token_logps=sampling,  # duplicate!
        )
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "multiple values" in str(e), f"Wrong error: {e}"
    print("PASS: pre-PR call pattern raises TypeError as expected")

# Same crash even when sampling is None
def test_pre_pr_raises_none():
    args = [torch.zeros(2,4)] * 3
    input_ids = torch.zeros(2,4, dtype=torch.long)
    mask = torch.ones(2,4)
    try:
        grpo_compute_loss(
            *args,
            input_ids,
            mask,
            0.1,
            torch.ones(2),
            sampling_per_token_logps=None,
        )
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "multiple values" in str(e)
    print("PASS: pre-PR raises even with sampling=None")

if __name__ == "__main__":
    test_pre_pr_raises()
    test_pre_pr_raises_none()
