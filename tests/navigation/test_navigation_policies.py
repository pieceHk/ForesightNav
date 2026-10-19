import ast
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from pbsp.navigation.types import Action, Candidate, Verification
from pbsp.navigation.candidates import ActiveCandidateTracker, project_candidates
from pbsp.navigation.evidence import (
    CandidateEvidenceBank,
    passes_dual_score_verification,
    select_independent_evidence,
)
from pbsp.navigation.policy import GoalHierarchy, StopBoundary
from pbsp.navigation.ranking import (
    CandidateActionReranker,
    CandidateRerankConfig,
    FrontierUtilityRanker,
)
from pbsp.navigation.recovery import NoveltyGoalSelector, TrappedDetector


class FlatNavmesh:
    def project(self, point):
        point = np.array(point, dtype=np.float32)
        point[1] = 0
        return point


def test_candidate_projection_and_continuity():
    candidates = [Candidate(np.array([1.0, 0.2, 1.0]), 0), Candidate(np.array([3.0, 0.2, 3.0]), 1)]
    projected = project_candidates(candidates, FlatNavmesh())
    tracker = ActiveCandidateTracker()
    assert tracker.refresh(projected).rank == 0
    refreshed = [Candidate(np.array([1.2, 0.0, 1.1]), 2), projected[1]]
    assert tracker.refresh(refreshed).rank == 2


def test_stop_and_goal_boundaries():
    failed = Verification(True, False, 100)
    confirmed = Verification(True, True, 100)
    assert StopBoundary.resolve(Action.STOP, failed, False) == Action.TURN_RIGHT
    assert StopBoundary.resolve(Action.MOVE_FORWARD, confirmed, False) == Action.STOP
    assert GoalHierarchy.choose(False, False, False) == GoalHierarchy.RECOVERY


def test_trapped_detector():
    detector = TrappedDetector(warmup_steps=3, window_steps=3, radius_m=0.05)
    assert not detector.update(np.zeros(3))
    assert not detector.update(np.array([0.01, 0, 0]))
    assert detector.update(np.array([0.02, 0, 0]))


def test_candidate_evidence_promotes_recurring_candidate():
    bank = CandidateEvidenceBank(match_radius_m=0.5, promotion_margin=1)
    transient = [0.0, 0.0, 0.0]
    persistent = [3.0, 0.0, 3.0]
    ranked, _ = bank.refresh({0: [transient, persistent]})
    assert ranked[0][0] == transient
    bank.refresh({0: [persistent]}, {0: transient})
    bank.refresh({0: [persistent]}, {0: transient})
    ranked, diagnostics = bank.refresh({0: [transient, persistent]}, {0: transient})
    assert ranked[0][0] == persistent
    assert diagnostics[0]["promoted"] == 1


def test_disabled_candidate_evidence_preserves_source_order():
    bank = CandidateEvidenceBank(enabled=False)
    source = [[3.0, 0.0, 3.0], [0.0, 0.0, 0.0]]
    ranked, diagnostics = bank.refresh({0: source})
    assert ranked[0] == source
    assert diagnostics[0] == {
        "candidates": 2,
        "max_observations": 0,
        "promoted": 0,
    }


def test_candidate_evidence_can_track_without_overriding_source_order():
    bank = CandidateEvidenceBank(match_radius_m=0.5)
    persistent = [3.0, 0.0, 3.0]
    bank.refresh({0: [persistent]})
    bank.refresh({0: [persistent]})
    transient = [0.0, 0.0, 0.0]
    ranked, _ = bank.refresh(
        {0: [transient, persistent]}, preserve_source_order=True
    )
    assert ranked[0] == [transient, persistent]


def test_candidate_evidence_source_order_does_not_restore_active_candidate():
    bank = CandidateEvidenceBank(match_radius_m=0.5)
    far = [8.0, 0.0, 0.0]
    near = [1.0, 0.0, 0.0]
    bank.refresh({0: [far, near]})
    ranked, _ = bank.refresh(
        {0: [near, far]},
        active_by_task={0: far},
        preserve_source_order=True,
    )
    assert ranked[0] == [near, far]


def _record(position, raw_rank, semantic, support=1.0, revision=1):
    return {
        "position": position,
        "raw_rank": raw_rank,
        "semantic_score": semantic,
        "support_count": int(support),
        "support_score": support,
        "map_revision": revision,
    }


def test_disabled_action_reranker_preserves_records_exactly():
    records = [
        _record([3.0, 0.0, 3.0], 0, 0.5),
        _record([1.0, 0.0, 1.0], 1, 0.4),
    ]
    ranked, diagnostics = CandidateActionReranker(enabled=False).rank(
        0, records, [0.0, 0.0, 0.0]
    )
    assert ranked == records
    assert diagnostics["reordered"] == 0


def test_action_reranker_balances_semantics_and_travel_with_stable_ties():
    reranker = CandidateActionReranker(
        enabled=True,
        config=CandidateRerankConfig(
            semantic=0.2,
            support=0.0,
            recurrence=0.0,
            travel=1.0,
            novelty=0.0,
            revisit=0.0,
            failure=0.0,
            switch_margin=0.0,
        ),
    )
    records = [
        _record([8.0, 0.0, 0.0], 0, 0.9),
        _record([1.0, 0.0, 0.0], 1, 0.7),
    ]
    ranked, diagnostics = reranker.rank(
        0,
        records,
        [0.0, 0.0, 0.0],
        travel_cost=lambda start, end: np.linalg.norm(end - start),
    )
    assert ranked[0]["raw_rank"] == 1
    assert diagnostics["reordered"] == 1


def test_action_reranker_hysteresis_holds_active_candidate():
    reranker = CandidateActionReranker(
        enabled=True,
        config=CandidateRerankConfig(
            semantic=1.0,
            support=0.0,
            recurrence=0.0,
            travel=0.0,
            novelty=0.0,
            revisit=0.0,
            failure=0.0,
            switch_margin=2.0,
        ),
    )
    records = [
        _record([0.0, 0.0, 0.0], 0, 0.5),
        _record([3.0, 0.0, 3.0], 1, 0.9),
    ]
    ranked, diagnostics = reranker.rank(
        0, records, [0.0, 0.0, 0.0], active_position=records[0]["position"]
    )
    assert ranked[0]["raw_rank"] == 0
    assert diagnostics["hysteresis_holds"] == 1


def test_action_reranker_counts_recurrence_once_per_map_revision():
    reranker = CandidateActionReranker(
        enabled=True,
        config=CandidateRerankConfig(
            semantic=0.0,
            support=0.0,
            recurrence=1.0,
            travel=0.0,
            novelty=0.0,
            revisit=0.0,
            failure=0.0,
            switch_margin=0.0,
        ),
    )
    first = _record([0.0, 0.0, 0.0], 0, 0.5, revision=10)
    second = _record([3.0, 0.0, 3.0], 1, 0.5, revision=10)
    reranker.rank(0, [first, second], [0.0, 0.0, 0.0])
    reranker.rank(0, [first], [0.0, 0.0, 0.0])
    reranker.rank(0, [first, second], [0.0, 0.0, 0.0])
    tracks = reranker._tracks[0]
    assert [track.observations for track in tracks] == [1, 1]
    first_new_revision = dict(first, map_revision=20)
    reranker.rank(0, [first_new_revision, second], [0.0, 0.0, 0.0])
    assert tracks[0].observations == 2


def test_action_reranker_rejects_nonfinite_positions():
    reranker = CandidateActionReranker(enabled=True)
    with np.testing.assert_raises(ValueError):
        reranker.rank(
            0,
            [_record([np.nan, 0.0, 0.0], 0, 1.0)],
            [0.0, 0.0, 0.0],
        )


def test_action_reranker_rejects_nonfinite_scores_and_weights():
    reranker = CandidateActionReranker(enabled=True)
    with np.testing.assert_raises(ValueError):
        reranker.rank(
            0,
            [_record([0.0, 0.0, 0.0], 0, np.inf)],
            [0.0, 0.0, 0.0],
        )
    with np.testing.assert_raises(ValueError):
        CandidateRerankConfig(travel=-1.0)


def test_frontier_ranker_selects_reachable_component_and_resets():
    frontier = np.zeros((12, 12), dtype=np.uint8)
    frontier[2, 4:6] = 1
    frontier[9, 7:9] = 1
    obstacle = np.zeros_like(frontier)
    obstacle[:, 6] = 1
    explored = np.ones_like(frontier)
    ranker = FrontierUtilityRanker(enabled=True)
    ranked, diagnostics = ranker.rank(frontier, obstacle, explored, [1, 1])
    assert ranked[:, 7:].sum() == 0
    assert diagnostics["reachable"] == 1
    ranker.reset()
    assert ranker._active is None


def test_frontier_ranker_does_not_cross_unknown_space():
    frontier = np.zeros((20, 20), dtype=np.uint8)
    frontier[2, 4] = 1
    frontier[15, 15] = 1
    obstacle = np.zeros_like(frontier)
    explored = np.zeros_like(frontier)
    explored[:6, :6] = 1
    ranker = FrontierUtilityRanker(enabled=True)
    ranked, diagnostics = ranker.rank(frontier, obstacle, explored, [1, 1])
    assert ranked[2, 4] == 1
    assert ranked[15, 15] == 0
    assert diagnostics["reachable"] == 1


def test_verification_requires_independent_viewpoint():
    observations = [
        {"observer_position": [0.0, 0.0], "observer_yaw": 0.0},
        {"observer_position": [0.0, 0.0], "observer_yaw": np.deg2rad(30)},
        {"observer_position": [0.0, 0.0], "observer_yaw": np.deg2rad(60)},
    ]
    selected = select_independent_evidence(observations, 0.2, 45.0)
    assert len(selected) == 2


def test_dual_score_verification_vetoes_only_jointly_weak_evidence():
    assert passes_dual_score_verification(True, 1.40, 0.05, 1.35, 0.12)
    assert passes_dual_score_verification(True, 1.20, 0.15, 1.35, 0.12)
    assert not passes_dual_score_verification(True, 1.20, 0.10, 1.35, 0.12)
    assert not passes_dual_score_verification(False, 1.50, 0.20, 1.35, 0.12)


def test_recovery_goal_selector_prefers_novel_basin():
    selector = NoveltyGoalSelector(history_size=4, novelty_weight=0.8)
    origin = np.zeros(3)
    first = selector.select(
        [np.array([3.0, 0.0, 0.0]), np.array([0.0, 0.0, 2.5])], origin
    )
    second = selector.select(
        [np.array([3.1, 0.0, 0.0]), np.array([0.0, 0.0, 2.5])], origin
    )
    assert np.allclose(first, [3.0, 0.0, 0.0])
    assert np.allclose(second, [0.0, 0.0, 2.5])


GLUE_AGENT = (
    Path(__file__).resolve().parents[2]
    / "Navigation/runtime/lagmemo/src/lagmemo/lagmemo/agent/lagmemo_agent/glue_agent.py"
)


def _glue_method_source(method_name):
    owner = GLUE_AGENT
    class_name = "GLUEAgent"
    if method_name in {"_validate_external_ranking_metadata", "refresh_gs_poses", "_mark_current_candidate_visited"}:
        owner = Path(__file__).resolve().parents[2] / "pbsp/navigation/candidates.py"
        class_name = "CandidateTracking"
    source = owner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    child.decorator_list = []
                    ast.fix_missing_locations(child)
                    return ast.unparse(child)
    raise AssertionError(f"missing GLUEAgent.{method_name}")


def _external_ranking_metadata(valid_gate=True):
    return {
        "mode": "ordered_suffix_mvp",
        "source": "suffix",
        "task_idx": 0,
        "version_token": {
            "scene_id": "scene-a",
            "episode_id": "episode-1",
            "map_revision": 7,
            "last_fused_keyframes": 7,
            "query_serial": 3,
        },
        "ordered_raw_ranks": [2, 0],
        "dispatch_version_gate": {
            "would_invalidate": not valid_gate,
            "reason": "none" if valid_gate else "stale_version_token",
            "scene_id": "scene-a",
            "episode_id": "episode-1",
            "query_serial": 3,
            "batch_map_revision": 7,
            "status_last_fused_keyframes": 7,
            "status_rebuilding": False,
        },
    }


def test_glue_external_ranking_contract_is_atomic_and_preserves_supplied_order():
    validator_namespace = {"Mapping": dict, "json": json}
    exec(
        compile(
            _glue_method_source("_validate_external_ranking_metadata"),
            "<external-validator>",
            "exec",
        ),
        validator_namespace,
    )
    validator = validator_namespace["_validate_external_ranking_metadata"]
    calls = {}

    class Evidence:
        def refresh(
            self,
            goals,
            active_goals,
            *,
            preserve_source_order,
        ):
            calls["evidence"] = {
                "goals": goals,
                "active": active_goals,
                "preserve_source_order": preserve_source_order,
            }
            return goals, {0: {"tracked": 2}}

    def reconcile(
        old_goals,
        ranked_goals,
        frontier_flag,
        visited,
        match_radius,
        blacklist_radius,
        *,
        preserve_active,
        protected_active_by_task,
    ):
        calls["reconcile"] = {
            "preserve_active": preserve_active,
            "protected": protected_active_by_task,
        }
        return ranked_goals, {0: 0}, visited

    refresh_namespace = {
        "deepcopy": deepcopy,
        "reconcile_spatial_candidates": reconcile,
    }
    exec(
        compile(
            _glue_method_source("refresh_gs_poses"),
            "<external-refresh>",
            "exec",
        ),
        refresh_namespace,
    )
    refresh = refresh_namespace["refresh_gs_poses"]
    agent = type("FakeAgent", (), {})()
    agent._validate_external_ranking_metadata = validator
    agent.gs_poses = {0: [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]}
    agent.frontier_flag = {0: 0}
    agent.visited_candidate_positions = {}
    agent.candidate_match_radius = 0.75
    agent.candidate_blacklist_radius = 1.5
    agent.query_action_rerank_enabled = False
    agent.current_task_idx = 0
    agent.candidate_evidence = Evidence()
    agent.candidate_evidence_diagnostics = {}
    agent.stop_by_seem = True
    agent.verification_history = [{"verified": True}]
    goals = {0: [[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
    metadata = _external_ranking_metadata()

    refresh(agent, goals, external_ranking_metadata=metadata)

    assert agent.gs_poses == goals
    assert calls["evidence"]["preserve_source_order"] is True
    assert calls["reconcile"]["preserve_active"] is False
    assert agent.last_external_ranking_metadata == metadata
    assert agent.stop_by_seem is True
    assert agent.verification_history == [{"verified": True}]


def test_glue_rejects_stale_external_order_before_candidate_mutation():
    namespace = {"Mapping": dict, "json": json}
    exec(
        compile(
            _glue_method_source("_validate_external_ranking_metadata"),
            "<external-validator>",
            "exec",
        ),
        namespace,
    )
    validator = namespace["_validate_external_ranking_metadata"]
    goals = {0: [[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
    with np.testing.assert_raises_regex(ValueError, "valid dispatch version gate"):
        validator(_external_ranking_metadata(valid_gate=False), goals)

    contradictory = _external_ranking_metadata()
    contradictory.update({"mode": "idea1_only", "source": "suffix"})
    with np.testing.assert_raises_regex(ValueError, "mode/source"):
        validator(contradictory, goals)

    mismatched = _external_ranking_metadata()
    mismatched["dispatch_version_gate"]["query_serial"] = 4
    with np.testing.assert_raises_regex(ValueError, "does not match"):
        validator(mismatched, goals)


def test_glue_external_refresh_rolls_back_evidence_on_reconcile_failure():
    validator_namespace = {"Mapping": dict, "json": json}
    exec(
        compile(
            _glue_method_source("_validate_external_ranking_metadata"),
            "<external-validator>",
            "exec",
        ),
        validator_namespace,
    )

    class Evidence:
        def __init__(self):
            self.refreshes = 0

        def refresh(self, goals, active, *, preserve_source_order):
            self.refreshes += 1
            return goals, {0: {"mutated": True}}

    def fail_reconcile(*_args, **_kwargs):
        raise RuntimeError("reconcile failed")

    namespace = {
        "deepcopy": deepcopy,
        "reconcile_spatial_candidates": fail_reconcile,
    }
    exec(
        compile(
            _glue_method_source("refresh_gs_poses"),
            "<external-refresh>",
            "exec",
        ),
        namespace,
    )
    agent = type("FakeAgent", (), {})()
    agent._validate_external_ranking_metadata = validator_namespace[
        "_validate_external_ranking_metadata"
    ]
    original_goals = {0: [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]}
    agent.gs_poses = deepcopy(original_goals)
    agent.frontier_flag = {0: 0}
    agent.visited_candidate_positions = {}
    agent.candidate_match_radius = 0.75
    agent.candidate_blacklist_radius = 1.5
    agent.query_action_rerank_enabled = False
    agent.current_task_idx = 0
    agent.candidate_evidence = Evidence()
    agent.candidate_evidence_diagnostics = {0: {"before": True}}

    with np.testing.assert_raises_regex(RuntimeError, "reconcile failed"):
        namespace["refresh_gs_poses"](
            agent,
            {0: [[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]]},
            external_ranking_metadata=_external_ranking_metadata(),
        )

    assert agent.gs_poses == original_goals
    assert agent.frontier_flag == {0: 0}
    assert agent.candidate_evidence.refreshes == 0
    assert agent.candidate_evidence_diagnostics == {0: {"before": True}}
