"""Operator eligibility only: never reset GCIM states at a pool boundary."""

POLICIES = {"unique", "cycle", "repeat_after_first", "unrestricted"}


def eligible_indices(pool_size, selection_count, used, policy):
    """Return candidates, refreshing only the used-index set when requested."""
    if policy not in POLICIES:
        raise ValueError("Unknown repetition policy")
    if policy == "cycle" and selection_count % pool_size == 0:
        used.clear()
    reuse = policy == "unrestricted" or (
        policy == "repeat_after_first" and selection_count >= pool_size)
    return [i for i in range(pool_size) if reuse or i not in used]
