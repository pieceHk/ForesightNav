import ast
import json
from pathlib import Path

import numpy as np
import pytest

from pbsp.navigation.cooldown import (
    CategorySwitchCooldown,
    cooldown_for_category,
    normalize_category,
    parse_category_cooldowns,
)


COOLDOWNS = {
    "bed": 30,
    "couch": 20,
    "toilet": 10,
    "chair": 10,
    "plant": 20,
    "tv": 10,
}

A = [0.0, 0.0, 0.0]
B = [5.0, 0.0, 0.0]


def _load_reconcile_spatial_candidates():
    path = (
        Path(__file__).parents[2]
        / "Navigation/runtime/lagmemo/src/lagmemo/lagmemo/agent/lagmemo_agent/glue_agent.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "reconcile_spatial_candidates"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["reconcile_spatial_candidates"]


RECONCILE = _load_reconcile_spatial_candidates()


def _reconcile(
    old_goals,
    new_goals,
    active_indices,
    visited_positions,
    *,
    preserve_active=False,
    protected=None,
):
    return RECONCILE(
        old_goals,
        new_goals,
        active_indices,
        visited_positions,
        match_radius=1.0,
        blacklist_radius=1.0,
        preserve_active=preserve_active,
        protected_active_by_task=protected,
    )


def _apply(gate, task_idx, category, step, active, latest):
    return gate.apply(
        task_idx=task_idx,
        category=category,
        step=step,
        active_position=active,
        latest_ranked_candidates=latest,
    )


def test_category_aliases_and_configuration_validation():
    assert normalize_category("sofa") == "couch"
    assert normalize_category("tv_monitor") == "tv"
    assert parse_category_cooldowns(json.dumps(COOLDOWNS)) == COOLDOWNS
    assert cooldown_for_category(COOLDOWNS, "sofa") == 20
    with pytest.raises(ValueError):
        parse_category_cooldowns('{"bed":-1}')
    with pytest.raises(ValueError):
        parse_category_cooldowns('{"bed":true}')
    with pytest.raises(KeyError):
        cooldown_for_category(COOLDOWNS, "sink")


@pytest.mark.parametrize(
    ("category", "cooldown"),
    [
        ("bed", 30),
        ("couch", 20),
        ("plant", 20),
        ("toilet", 10),
        ("chair", 10),
        ("tv", 10),
    ],
)
def test_category_cooldown_holds_then_uses_latest_rank_at_boundary(
    category, cooldown
):
    gate = CategorySwitchCooldown(COOLDOWNS)
    task_idx = list(COOLDOWNS).index(category)
    gate.observe_active(task_idx, A, 0)

    held, event = _apply(
        gate, task_idx, category, cooldown - 1, A, [B, A]
    )
    assert held == [A, B]
    assert event["reason"] == "KEEP_ACTIVE_COOLDOWN"
    assert event["held"] == 1

    latest, event = _apply(
        gate, task_idx, category, cooldown, A, [B, A]
    )
    assert latest == [B, A]
    assert event["reason"] == "ACCEPT_LATEST_RANK"
    assert event["held"] == 0


def test_realized_post_reconcile_switch_restarts_clock():
    gate = CategorySwitchCooldown(COOLDOWNS)
    gate.observe_active(0, A, 0)

    proposed, event = _apply(gate, 0, "bed", 30, A, [B, A])
    assert proposed == [B, A]
    assert event["proposed_switch"] == 1
    assert gate._active_by_task[0] == tuple(A)

    assert gate.observe_active(0, B, 30) is True
    held, event = _apply(gate, 0, "bed", 40, B, [A, B])
    assert held == [B, A]
    assert event["elapsed_since_switch"] == 10
    assert event["held"] == 1


def test_post_reconcile_rejection_does_not_restart_clock():
    gate = CategorySwitchCooldown(COOLDOWNS)
    gate.observe_active(0, A, 0)
    _apply(gate, 0, "bed", 30, A, [B, A])

    assert gate.observe_active(0, A, 30) is False
    assert gate._last_switch_step[0] == 0


def test_empty_then_recovered_active_starts_a_new_cooldown_window():
    gate = CategorySwitchCooldown(COOLDOWNS)
    gate.observe_active(0, A, 0)
    gate.observe_active(0, None, 5)
    assert gate.observe_active(0, B, 10) is True

    held, event = _apply(gate, 0, "bed", 20, B, [A, B])
    assert held == [B, A]
    assert event["elapsed_since_switch"] == 10
    assert event["held"] == 1


def test_state_is_isolated_by_subtask_and_reset_by_episode():
    gate = CategorySwitchCooldown(COOLDOWNS)
    gate.observe_active(0, A, 0)
    gate.observe_active(1, A, 0)
    gate.observe_active(0, B, 30)

    task_zero, event_zero = _apply(gate, 0, "bed", 35, B, [A, B])
    task_one, event_one = _apply(gate, 1, "chair", 10, A, [B, A])
    assert task_zero == [B, A]
    assert event_zero["held"] == 1
    assert task_one == [B, A]
    assert event_one["held"] == 0

    gate.reset()
    assert not gate._active_by_task
    assert not gate._last_switch_step
    assert not gate._initialized_tasks


def test_missing_active_candidate_bypasses_cooldown():
    gate = CategorySwitchCooldown(COOLDOWNS)
    gate.observe_active(0, A, 0)
    latest, event = _apply(gate, 0, "bed", 10, A, [B])
    assert latest == [B]
    assert event["reason"] == "ACTIVE_MISSING_ACCEPT_LATEST"


def test_top_coordinate_drift_within_track_radius_is_not_a_switch():
    gate = CategorySwitchCooldown(COOLDOWNS, match_radius_m=1.0)
    gate.observe_active(0, A, 0)
    drifted = [0.8, 0.0, 0.0]
    latest, event = _apply(gate, 0, "bed", 10, A, [drifted, B])
    assert latest == [drifted, B]
    assert event["proposed_switch"] == 0


def test_gate_rejects_invalid_step_and_unconfigured_category():
    gate = CategorySwitchCooldown(COOLDOWNS)
    with pytest.raises(ValueError):
        _apply(gate, 0, "bed", -1, A, [B, A])
    with pytest.raises(KeyError):
        _apply(gate, 0, "sink", 10, A, [B, A])


def test_keep_survives_visited_blacklist_and_does_not_realize_switch():
    gate = CategorySwitchCooldown(COOLDOWNS)
    gate.observe_active(0, A, 0)
    gated, event = _apply(gate, 0, "bed", 10, A, [B, A])
    reconciled, indices, _ = _reconcile(
        {0: [A]},
        {0: gated},
        {0: 0},
        {0: [A]},
        protected={0: A},
    )
    assert event["reason"] == "KEEP_ACTIVE_COOLDOWN"
    assert reconciled[0][0] == A
    assert indices[0] == 0
    assert gate.observe_active(0, reconciled[0][0], 10) is False


def test_latest_unchanged_drift_is_protected_through_reconciliation():
    drifted = [0.8, 0.0, 0.0]
    gate = CategorySwitchCooldown(COOLDOWNS)
    gate.observe_active(0, A, 0)
    gated, event = _apply(gate, 0, "bed", 10, A, [drifted, B])
    reconciled, _, _ = _reconcile(
        {0: [A]},
        {0: gated},
        {0: 0},
        {0: [A]},
        protected={0: A},
    )
    assert event["reason"] == "LATEST_UNCHANGED"
    assert np.allclose(reconciled[0][0], drifted)
    assert gate.observe_active(0, reconciled[0][0], 10) is False


def test_latest_unchanged_at_boundary_is_not_protected():
    drifted = [0.8, 0.0, 0.0]
    gate = CategorySwitchCooldown(COOLDOWNS)
    gate.observe_active(0, A, 0)
    gated, event = _apply(gate, 0, "bed", 30, A, [drifted, B])
    protected = (
        {0: A}
        if event["elapsed_since_switch"] < event["cooldown_steps"]
        else None
    )
    reconciled, _, _ = _reconcile(
        {0: [A]},
        {0: gated},
        {0: 0},
        {0: [A]},
        protected=protected,
    )
    assert event["reason"] == "LATEST_UNCHANGED"
    assert event["elapsed_since_switch"] == 30
    assert protected is None
    assert reconciled[0][0] == B


def test_accept_at_boundary_remains_free_to_switch():
    gate = CategorySwitchCooldown(COOLDOWNS)
    gate.observe_active(0, A, 0)
    gated, event = _apply(gate, 0, "bed", 30, A, [B, A])
    reconciled, _, _ = _reconcile(
        {0: [A]}, {0: gated}, {0: 0}, {}, protected=None
    )
    assert event["reason"] == "ACCEPT_LATEST_RANK"
    assert reconciled[0][0] == B


def test_missing_and_absent_active_paths_remain_unprotected_bypasses():
    gate = CategorySwitchCooldown(COOLDOWNS)
    gate.observe_active(0, A, 0)
    gated, event = _apply(gate, 0, "bed", 10, A, [B])
    reconciled, _, _ = _reconcile(
        {0: [A]}, {0: gated}, {0: 0}, {}, protected=None
    )
    assert event["reason"] == "ACTIVE_MISSING_ACCEPT_LATEST"
    assert reconciled[0][0] == B

    absent_gate = CategorySwitchCooldown(COOLDOWNS)
    absent, event = _apply(absent_gate, 0, "bed", 10, None, [B])
    reconciled, _, _ = _reconcile(
        {0: []}, {0: absent}, {0: 0}, {}, protected=None
    )
    assert event["reason"] == "NO_ACTIVE_ACCEPT_LATEST"
    assert reconciled[0][0] == B


def test_external_switch_restarts_clock_before_protected_refresh():
    gate = CategorySwitchCooldown(COOLDOWNS)
    gate.observe_active(0, A, 0)
    assert gate.observe_active(0, B, 5) is True
    gated, event = _apply(gate, 0, "bed", 10, B, [A, B])
    reconciled, _, _ = _reconcile(
        {0: [B]},
        {0: gated},
        {0: 0},
        {0: [B]},
        protected={0: B},
    )
    assert event["reason"] == "KEEP_ACTIVE_COOLDOWN"
    assert event["elapsed_since_switch"] == 5
    assert reconciled[0][0] == B


def test_protection_is_scoped_to_only_the_current_subtask():
    D = [0.0, 0.0, 5.0]
    E = [5.0, 0.0, 5.0]
    reconciled, _, _ = _reconcile(
        {0: [A], 1: [D]},
        {0: [A, B], 1: [D, E]},
        {0: 0, 1: 0},
        {0: [A], 1: [D]},
        protected={0: A},
    )
    assert reconciled[0][0] == A
    assert reconciled[1] == [E]


def test_legacy_reconciliation_is_unchanged_without_protection():
    unprotected, _, _ = _reconcile(
        {0: [A]}, {0: [B, A]}, {0: 0}, {}, protected=None
    )
    preserved, _, _ = _reconcile(
        {0: [A]},
        {0: [B, A]},
        {0: 0},
        {},
        preserve_active=True,
        protected=None,
    )
    assert unprotected[0] == [B, A]
    assert preserved[0] == [A, B]
