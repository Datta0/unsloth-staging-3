"""Static analysis: verify the PR branch call site matches the callee signature."""
import ast, os, re

WORKDIR = os.path.dirname(os.path.abspath(__file__))

def test_callsite_has_sampling_positional():
    """Verify sampling_per_token_logps appears as 4th positional arg in the call."""
    rl_path = os.path.join(WORKDIR, "unsloth", "models", "rl_replacements.py")
    with open(rl_path) as f:
        src = f.read()

    # Find the grpo_compute_loss_slow call block
    match = re.search(r'grpo_compute_loss_slow\(\s*\n', src)
    assert match, "Cannot find grpo_compute_loss_slow call"

    start = match.start()
    # Find matching close paren
    depth = 0
    call_text = ""
    for i, ch in enumerate(src[start:]):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                call_text = src[start:start+i+1]
                break

    # Extract positional args (lines without '=' that aren't the function name)
    lines = call_text.split('\n')
    positional_args = []
    for line in lines[1:]:  # skip function name line
        stripped = line.strip().rstrip(',')
        if not stripped or stripped == ')':
            continue
        if '=' in stripped:
            # Check it's a keyword arg (not inside the value)
            # Simple heuristic: if '=' appears and it's not 'self.xxx'
            before_eq = stripped.split('=')[0].strip()
            if before_eq.isidentifier():
                break  # first keyword arg, stop collecting positionals
        positional_args.append(stripped)

    assert len(positional_args) == 8, f"Expected 8 positional args, got {len(positional_args)}: {positional_args}"
    assert positional_args[0] == "ref_logps", f"arg[0] should be ref_logps, got {positional_args[0]}"
    assert positional_args[1] == "per_token_logps", f"arg[1] should be per_token_logps, got {positional_args[1]}"
    assert positional_args[2] == "old_logps", f"arg[2] should be old_logps, got {positional_args[2]}"
    assert positional_args[3] == "sampling_per_token_logps", f"arg[3] should be sampling_per_token_logps, got {positional_args[3]}"
    assert positional_args[4] == "input_ids", f"arg[4] should be input_ids, got {positional_args[4]}"
    assert positional_args[5] == "completion_mask", f"arg[5] should be completion_mask, got {positional_args[5]}"
    assert "self.beta" in positional_args[6], f"arg[6] should be self.beta, got {positional_args[6]}"
    assert positional_args[7] == "advantages", f"arg[7] should be advantages, got {positional_args[7]}"
    print("PASS: call site has 8 positional args in correct order")

def test_no_duplicate_kwarg():
    """Verify sampling_per_token_logps is NOT also passed as a keyword arg."""
    rl_path = os.path.join(WORKDIR, "unsloth", "models", "rl_replacements.py")
    with open(rl_path) as f:
        src = f.read()

    match = re.search(r'grpo_compute_loss_slow\(\s*\n', src)
    start = match.start()
    depth = 0
    call_text = ""
    for i, ch in enumerate(src[start:]):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                call_text = src[start:start+i+1]
                break

    # Check no keyword version of sampling_per_token_logps
    kwarg_pattern = re.compile(r'sampling_per_token_logps\s*=\s*sampling_per_token_logps')
    matches = kwarg_pattern.findall(call_text)
    assert len(matches) == 0, f"Found duplicate kwarg sampling_per_token_logps in call: {matches}"
    print("PASS: no duplicate sampling_per_token_logps kwarg")

def test_callee_signature_matches():
    """Verify the callee signature order matches what we expect."""
    zoo_path = os.path.join(WORKDIR, "..", "..",
        "lib/python3.13/site-packages/unsloth_zoo/rl_replacements.py")
    if not os.path.exists(zoo_path):
        import sys
        for p in sys.path:
            candidate = os.path.join(p, "unsloth_zoo", "rl_replacements.py")
            if os.path.exists(candidate):
                zoo_path = candidate
                break

    with open(zoo_path) as f:
        src = f.read()

    match = re.search(r'def grpo_compute_loss\((.*?)\):', src, re.DOTALL)
    assert match, "Cannot find grpo_compute_loss definition"
    params_str = match.group(1)
    params = [p.strip().strip('*') for p in params_str.split(',') if p.strip() and p.strip() != '**kwargs']
    expected = ["ref", "new", "old", "sampling_per_token_logps", "input_ids", "mask", "beta", "advantages"]
    assert params == expected, f"Signature mismatch: {params} != {expected}"
    print("PASS: callee signature matches expected order")

if __name__ == "__main__":
    test_callsite_has_sampling_positional()
    test_no_duplicate_kwarg()
    test_callee_signature_matches()
