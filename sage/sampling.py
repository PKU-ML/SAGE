"""Memory-bounded, joint-balanced sampler for dense temporal supervision."""

from __future__ import annotations

from collections import Counter

import numpy as np


def sample_dense_unique_pairs(
    dataset,
    specs,
    *,
    history_len: int,
    frameskip: int,
    goal_offsets: list[int],
    action_offsets: list[int],
    limit: int,
    seed: int,
    allow_repeats: bool = False,
    balance_by_goal: bool = False,
    goal_quotas: dict[int, int] | None = None,
):
    """Joint-balance temporal pairs while maximizing unique source windows.

    Cells with the longest far goal are allocated first. A spec can therefore
    appear at most once in the first allocation stage, protecting rare
    long-horizon windows from the repeated-pair behavior of an expanded pair
    pool. When ``allow_repeats`` is enabled and ``limit`` exceeds the number
    of available windows, the sampler completes this broad unique cover first,
    then adds a second, cell-balanced pass. This is useful for Cube, whose
    valid window pool is smaller than a desired 400k training budget.
    """
    if limit <= 0:
        raise ValueError("dense sampling requires a positive example limit")

    goals = sorted({int(value) for value in goal_offsets})
    actions = sorted({int(value) for value in action_offsets})
    cells = [(goal, action) for goal in goals for action in actions if action <= goal]
    if not cells:
        raise ValueError("No valid dense (goal_offset, action_offset) cells")

    if int(limit) > len(specs) and not allow_repeats:
        raise ValueError(
            f"Requested {limit} dense pairs but only {len(specs)} source windows; "
            "pass allow_repeats=True to fill the remaining budget after a unique pass"
        )

    def allocate_quotas(total: int) -> dict[tuple[int, int], int]:
        """Allocate either uniformly over cells or uniformly over far goals."""
        if goal_quotas is not None:
            if set(goal_quotas) != set(goals):
                raise ValueError(
                    f"goal_quotas must specify exactly {goals}, got {sorted(goal_quotas)}"
                )
            if sum(int(value) for value in goal_quotas.values()) != int(total):
                raise ValueError(
                    f"goal_quotas total {sum(goal_quotas.values())} does not match {total}"
                )
            quotas: dict[tuple[int, int], int] = {}
            for goal in goals:
                group_actions = [action for action in actions if action <= goal]
                cell_base, cell_remainder = divmod(int(goal_quotas[goal]), len(group_actions))
                for action_index, action in enumerate(group_actions):
                    quotas[(goal, action)] = cell_base + int(action_index < cell_remainder)
            return quotas
        if not balance_by_goal:
            base, remainder = divmod(total, len(cells))
            return {
                cell: base + int(index < remainder)
                for index, cell in enumerate(cells)
            }

        goal_base, goal_remainder = divmod(total, len(goals))
        quotas: dict[tuple[int, int], int] = {}
        for goal_index, goal in enumerate(goals):
            goal_total = goal_base + int(goal_index < goal_remainder)
            group_actions = [action for action in actions if action <= goal]
            cell_base, cell_remainder = divmod(goal_total, len(group_actions))
            for action_index, action in enumerate(group_actions):
                quotas[(goal, action)] = cell_base + int(action_index < cell_remainder)
        return quotas

    unique_limit = min(int(limit), len(specs))
    quotas = allocate_quotas(unique_limit)
    current = np.asarray(
        [int(spec.start) + (int(history_len) - 1) * int(frameskip) for spec in specs],
        dtype=np.int64,
    )
    final = np.asarray(
        [int(dataset.lengths[spec.local_episode]) - 1 for spec in specs],
        dtype=np.int64,
    )
    used = np.zeros(len(specs), dtype=bool)
    rng = np.random.default_rng(int(seed))
    records: list[tuple[int, int, int]] = []
    available_by_goal: dict[int, int] = {}

    # Long horizons have the narrowest support, so reserve their windows first.
    for goal in reversed(goals):
        group_actions = [action for action in actions if action <= goal]
        required = sum(quotas[(goal, action)] for action in group_actions)
        candidates = np.flatnonzero((current + int(goal) <= final) & ~used)
        available_by_goal[int(goal)] = int(len(candidates))
        if len(candidates) < required:
            raise ValueError(
                f"Dense sampler needs {required} unique windows for Delta={goal}, "
                f"but only {len(candidates)} remain"
            )
        selected = rng.choice(candidates, size=required, replace=False)
        rng.shuffle(selected)
        offset = 0
        for action in group_actions:
            count = quotas[(goal, action)]
            chunk = selected[offset : offset + count]
            records.extend((int(spec_index), int(goal), int(action)) for spec_index in chunk)
            offset += count
        used[selected] = True

    order = rng.permutation(len(records))
    records = [records[int(index)] for index in order]
    pair_spec_indices = np.asarray([record[0] for record in records], dtype=np.int64)
    pair_goal_offsets = np.asarray([record[1] for record in records], dtype=np.int64)
    pair_action_offsets = np.asarray([record[2] for record in records], dtype=np.int64)
    repeated_examples = int(limit) - len(records)
    repeat_quotas: dict[tuple[int, int], int] = {}
    if repeated_examples:
        repeat_quotas = allocate_quotas(repeated_examples)
        # The coverage stage above already used every available source window.
        # Repeats are therefore intentionally balanced by cell, not by episode.
        for goal in goals:
            candidates = np.flatnonzero(current + int(goal) <= final)
            for action in actions:
                if action > goal:
                    continue
                count = repeat_quotas[(goal, action)]
                if not len(candidates):
                    raise RuntimeError(f"No valid windows for repeated H={goal}, tau={action}")
                chosen = rng.choice(candidates, size=count, replace=count > len(candidates))
                records.extend((int(spec_index), int(goal), int(action)) for spec_index in chosen)

        order = rng.permutation(len(records))
        records = [records[int(index)] for index in order]
        pair_spec_indices = np.asarray([record[0] for record in records], dtype=np.int64)
        pair_goal_offsets = np.asarray([record[1] for record in records], dtype=np.int64)
        pair_action_offsets = np.asarray([record[2] for record in records], dtype=np.int64)

    diagnostics = {
        "num_cells": len(cells),
        "balance_by_goal": bool(balance_by_goal),
        "goal_quotas": None if goal_quotas is None else dict(sorted(goal_quotas.items())),
        "num_examples": len(records),
        "unique_windows": int(len(np.unique(pair_spec_indices))),
        "coverage_stage_examples": int(unique_limit),
        "repeated_fill_examples": int(repeated_examples),
        "goal_counts": dict(sorted(Counter(pair_goal_offsets.tolist()).items())),
        "joint_counts": {
            f"H{goal}_A{action}": int(quotas[(goal, action)] + repeat_quotas.get((goal, action), 0))
            for goal, action in cells
        },
        "available_before_allocation": available_by_goal,
    }
    return pair_spec_indices, pair_goal_offsets, pair_action_offsets, diagnostics
