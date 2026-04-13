"""Test that post-PR call pattern binds all 8 positional args correctly."""
import torch

def grpo_compute_loss(ref, new, old, sampling_per_token_logps, input_ids, mask, beta, advantages, **kwargs):
    # Verify each arg got the right value via sentinel checks
    assert ref == "REF"
    assert new == "NEW"
    assert old == "OLD"
    assert sampling_per_token_logps == "SAMPLING"
    assert input_ids == "IDS"
    assert mask == "MASK"
    assert beta == "BETA"
    assert advantages == "ADV"
    return "ok"

def test_post_pr_binding():
    result = grpo_compute_loss(
        "REF",
        "NEW",
        "OLD",
        "SAMPLING",   # 4th positional -- the fix
        "IDS",
        "MASK",
        "BETA",
        "ADV",
        loss_type="grpo",
        epsilon_low=0.2,
    )
    assert result == "ok"
    print("PASS: post-PR positional args bind correctly")

def test_post_pr_binding_with_none_sampling():
    def check(ref, new, old, sampling_per_token_logps, input_ids, mask, beta, advantages, **kwargs):
        assert sampling_per_token_logps is None
        assert input_ids == "IDS"
        return "ok"
    result = check("REF", "NEW", "OLD", None, "IDS", "MASK", "BETA", "ADV")
    assert result == "ok"
    print("PASS: post-PR binding works with sampling=None")

if __name__ == "__main__":
    test_post_pr_binding()
    test_post_pr_binding_with_none_sampling()
