"""PBSP navigation state and candidate execution, configured by the caller."""

# Portions adapted from LagMemo, Copyright (c) Meta Platforms, Inc. and affiliates.
# Upstream MIT terms and provenance are retained in THIRD_PARTY_NOTICES.md.

from dataclasses import dataclass
from typing import Optional

from pbsp.navigation.evidence import CandidateEvidenceBank
from pbsp.navigation.planning import (
    active_candidate,
    raw_rank_for_position,
    raw_ranks_for_positions,
)
from pbsp.navigation.navigator import Navigator
from pbsp.navigation.ranking import FrontierRerankConfig, FrontierUtilityRanker
from pbsp.settings import NavigationSettings


@dataclass(frozen=True)
class CandidateDispatch:
    projected_goals: Optional[dict]
    metadata: Optional[dict]
    cooldown_event: Optional[dict]
    refresh_applied: bool
    refresh_error: Optional[str]
    realized_active_position: object
    realized_raw_rank: Optional[int]


class PBSPAgent(Navigator):
    """Execute the online policy with perception and map layers supplied by an adapter."""

    def __init__(self, *, settings: NavigationSettings, actions):
        super().__init__(actions=actions)
        self.settings = settings
        self.visited_candidate_positions = {}
        self.candidate_match_radius = settings.candidate_track_radius_m
        self.candidate_blacklist_radius = settings.candidate_blacklist_radius_m
        self.two_stage_verification = settings.two_stage_verification
        self.verification_window_steps = settings.verification_window_steps
        self.verification_min_confirmations = settings.verification_min_confirmations
        self.verification_pose_radius = settings.verification_pose_radius_m
        self.verification_min_peak_seem = settings.verification_min_peak_seem
        self.verification_dual_score_gate = settings.verification_dual_score_gate
        self.verification_strong_peak_seem = settings.verification_strong_peak_seem
        self.verification_min_peak_clip = settings.verification_min_peak_clip
        self.verification_min_observer_translation = (
            settings.verification_min_observer_translation_m
        )
        self.verification_min_observer_yaw_deg = (
            settings.verification_min_observer_yaw_deg
        )
        self.candidate_evidence = CandidateEvidenceBank(
            match_radius_m=settings.candidate_evidence_radius_m,
            promotion_margin=settings.candidate_promotion_margin,
            enabled=settings.candidate_evidence_ranking,
        )
        self.query_action_rerank_enabled = settings.query_action_rerank_enabled
        self.frontier_reranker = FrontierUtilityRanker(
            enabled=settings.frontier_rerank_enabled,
            config=FrontierRerankConfig(
                gain=settings.frontier_w_gain,
                clearance=settings.frontier_w_clearance,
                travel=settings.frontier_w_travel,
                novelty=settings.frontier_w_novelty,
                revisit=settings.frontier_w_revisit,
                failure=settings.frontier_w_failure,
                switch_margin=settings.frontier_switch_margin,
            ),
        )
        self.frontier_rerank_diagnostics = {}
        self.candidate_evidence_diagnostics = {}
        self.last_external_ranking_metadata = None
        self.runtime_scene_id = None
        self.runtime_episode_id = None
        self.trapped_warmup_steps = settings.trapped_warmup_steps
        self.trapped_window_steps = settings.trapped_window_steps
        self.trapped_radius = settings.trapped_radius_m
        self.random_exploration_warmup_steps = settings.random_exploration_warmup_steps
        self.trapped_position_history = []
        self.trapped_active = False
        self.trapped_events = []
        self.navmesh_override_goal = None
        self.navmesh_override_mode = None
        self.navmesh_override_task = None

    def dispatch_initial_candidates(
        self, projected_goals, dispatch_metadata, *, task_idx, candidate_batch
    ) -> CandidateDispatch:
        """Install the first version-checked order before the first action."""
        ordered_ranks = dispatch_metadata["ordered_raw_ranks"]
        dispatch_metadata["pre_cooldown_selected_raw_rank"] = (
            ordered_ranks[0] if ordered_ranks else None
        )
        dispatch_metadata["post_cooldown_selected_raw_rank"] = (
            ordered_ranks[0] if ordered_ranks else None
        )
        initial_refresh_kwargs = {}
        if dispatch_metadata["externally_ranked"]:
            initial_refresh_kwargs["external_ranking_metadata"] = dispatch_metadata
        refresh_applied = False
        refresh_error = None
        try:
            if projected_goals is not None:
                self.refresh_gs_poses(projected_goals, **initial_refresh_kwargs)
                refresh_applied = True
        except Exception as exc:
            refresh_error = f"{type(exc).__name__}: {exc}"
            print(f"[P4] initial goal refresh warning: {refresh_error}", flush=True)
        realized_active = active_candidate(self, task_idx)
        realized_raw_rank = raw_rank_for_position(
            candidate_batch, task_idx, realized_active
        )
        return CandidateDispatch(
            projected_goals,
            dispatch_metadata,
            None,
            refresh_applied,
            refresh_error,
            realized_active,
            realized_raw_rank,
        )

    def dispatch_candidates(
        self,
        projected_goals,
        dispatch_metadata,
        *,
        task_idx,
        category,
        local_step,
        category_cooldown,
        candidate_batch,
    ) -> CandidateDispatch:
        """Apply cooldown, install the order once and observe its realized target."""
        cooldown_event = None
        active_before_refresh = None
        if (
            projected_goals is not None
            and category_cooldown.enabled
            and (task_idx in projected_goals)
        ):
            active_before_refresh = active_candidate(self, task_idx)
            if local_step == 0:
                active_before_refresh = None
            (gated, cooldown_event) = category_cooldown.apply(
                task_idx=task_idx,
                category=category,
                step=local_step,
                active_position=active_before_refresh,
                latest_ranked_candidates=projected_goals[task_idx],
            )
            projected_goals = dict(projected_goals)
            projected_goals[task_idx] = gated
            cooldown_event["proposed_active_position"] = gated[0] if gated else None
        if dispatch_metadata is not None and dispatch_metadata["ordered_raw_ranks"]:
            try:
                pre_cooldown_ranks = tuple(dispatch_metadata["ordered_raw_ranks"])
                post_cooldown_ranks = raw_ranks_for_positions(
                    candidate_batch, task_idx, projected_goals[task_idx]
                )
                dispatch_metadata["ordered_raw_ranks"] = list(post_cooldown_ranks)
                dispatch_metadata["pre_cooldown_selected_raw_rank"] = (
                    pre_cooldown_ranks[0] if pre_cooldown_ranks else None
                )
                dispatch_metadata["post_cooldown_selected_raw_rank"] = (
                    post_cooldown_ranks[0] if post_cooldown_ranks else None
                )
                dispatch_metadata["cooldown_vetoed"] = bool(
                    pre_cooldown_ranks
                    and post_cooldown_ranks
                    and (pre_cooldown_ranks[0] != post_cooldown_ranks[0])
                )
            except Exception as exc:
                dispatch_metadata.update(
                    {
                        "source": "legacy",
                        "externally_ranked": False,
                        "reason": "invalid_external_order",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                projected_goals = None
        refresh_applied = False
        refresh_error = None
        if projected_goals is not None:
            try:
                protected_active_by_task = None
                if (
                    cooldown_event is not None
                    and active_before_refresh is not None
                    and (
                        cooldown_event.get("reason")
                        in {"KEEP_ACTIVE_COOLDOWN", "LATEST_UNCHANGED"}
                    )
                    and (
                        int(cooldown_event["elapsed_since_switch"])
                        < int(cooldown_event["cooldown_steps"])
                    )
                ):
                    protected_active_by_task = {int(task_idx): active_before_refresh}
                refresh_kwargs = {"protected_active_by_task": protected_active_by_task}
                if (
                    dispatch_metadata is not None
                    and dispatch_metadata["externally_ranked"]
                ):
                    refresh_kwargs["external_ranking_metadata"] = dispatch_metadata
                self.refresh_gs_poses(projected_goals, **refresh_kwargs)
                refresh_applied = True
            except Exception as e:
                refresh_error = f"{type(e).__name__}: {e}"
                print(f"[P4] goal refresh warning: {e}", flush=True)
        if cooldown_event is not None:
            realized_active = active_candidate(self, task_idx)
            cooldown_event["refresh_applied"] = int(refresh_applied)
            cooldown_event["realized_switch"] = int(
                category_cooldown.observe_active(task_idx, realized_active, local_step)
            )
            cooldown_event["realized_active_position"] = (
                None
                if realized_active is None
                else [float(value) for value in realized_active]
            )
        realized_active = active_candidate(self, task_idx)
        return CandidateDispatch(
            projected_goals,
            dispatch_metadata,
            cooldown_event,
            refresh_applied,
            refresh_error,
            realized_active,
            raw_rank_for_position(candidate_batch, task_idx, realized_active),
        )
