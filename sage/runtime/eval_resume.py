"""Validate an interrupted evaluation before restoring its proposal RNG."""

import copy


def retry_failed_tail(existing, protocol, query_ids, initial_state):
    """Retry only an error, retaining its evidence and exact pre-query RNG."""
    result = copy.deepcopy(existing)
    rows = result.get('results', [])
    if not rows or 'error' not in rows[-1] or any('error' in r for r in rows[:-1]):
        raise ValueError('Retry requires exactly one trailing error row')
    state = rows[-1].get('proposal_rng_state_before')
    if state is None:
        # Old Gaussian runs used independent query RNGs, leaving this RNG untouched.
        if protocol.get('controller') != 'gaussian_lewm' or existing.get('proposal_rng_state') != initial_state:
            raise ValueError('No certified pre-query RNG state for retry')
        state = initial_state
    result.setdefault('failed_attempts', []).append(rows.pop())
    result['proposal_rng_state'] = state
    validate_resume(result, protocol, query_ids)
    return result


def validate_resume(existing, protocol, query_ids):
    if existing.get("protocol") != protocol:
        raise ValueError("Resume protocol or input identity mismatch")
    rows = existing.get("results", [])
    if [row.get("query_id") for row in rows] != query_ids[:len(rows)]:
        raise ValueError("Resume results must be an ordered prefix of the selected queries")
    if any("error" in row for row in rows):
        raise ValueError("Cannot resume an error-containing result as completed evaluation")
    state = existing.get("proposal_rng_state")
    if rows and (not isinstance(state, list) or not state):
        raise ValueError("Missing proposal RNG state; start a fresh evaluation")
    if state is not None and (not isinstance(state, list) or
                             any(type(v) is not int or not 0 <= v <= 255 for v in state)):
        raise ValueError("Invalid proposal RNG state")
    return state
