"""Build and record versioned planning events around the PBSP planning core."""

import time

from experiments.recorder import (
    planning_fallback_event as _scsp_shadow_fallback_event,
    record_provider_event as _record_scsp_provider_event,
)
from experiments.telemetry import (
    SHADOW_SCHEMA_VERSION,
    build_scsp_shadow_decision,
    dumps_json_strict,
)
from pbsp.navigation.planning import dispatch_version_gate as _scsp_dispatch_gate
from pbsp.types import FallbackReason, SCSPContractError


class PlanningBridge:
    """Bind geometry and a clock; retain event and fallback semantics."""

    def __init__(self, *, project_goals, geodesic, clock=time):
        self.project_goals = project_goals
        self.geodesic = geodesic
        self.clock = clock

    def _fallback_event(self, **kwargs):
        return _scsp_shadow_fallback_event(clock=self.clock.perf_counter, **kwargs)

    def compute_decision(
        self,
        *,
        phase,
        arm,
        provider_event,
        candidate_batch,
        projected_goals,
        projected_indices,
        current_task,
        origin,
        pathfinder,
        agent,
        reranker,
        distance_cache,
        candidate_cap,
        suffix_weight,
        idea1_config,
        provenance,
        episode_identity,
    ):
        """Compute one suffix decision while leaving legacy candidates untouched."""
        started_at = self.clock.perf_counter()
        version_gate = dict(provider_event.get('version_gate', {}))
        if candidate_batch is None:
            return self._fallback_event(
                phase=phase,
                arm=arm,
                current_task=current_task,
                reason=FallbackReason.INVALID_CANDIDATE_BATCH.value,
                version_gate=version_gate,
                candidate_batch=None,
                origin=origin,
                started_at=started_at,
                failure_stage='candidate_batch',
                provenance=provenance,
                candidate_cap=candidate_cap,
                suffix_weight=suffix_weight,
                idea1_config=idea1_config,
                episode_identity=episode_identity,
            )
        if version_gate.get('would_invalidate', True):
            return self._fallback_event(
                phase=phase,
                arm=arm,
                current_task=current_task,
                reason=version_gate.get(
                    'reason', FallbackReason.PROVIDER_STATUS_UNAVAILABLE.value
                ),
                version_gate=version_gate,
                candidate_batch=candidate_batch,
                origin=origin,
                started_at=started_at,
                failure_stage='version_gate',
                provenance=provenance,
                candidate_cap=candidate_cap,
                suffix_weight=suffix_weight,
                idea1_config=idea1_config,
                episode_identity=episode_identity,
            )
        try:
            from pbsp.navigation.planning import compute_candidate_plan

            calculation = compute_candidate_plan(
                candidate_batch=candidate_batch,
                projected_goals=projected_goals,
                projected_indices=projected_indices,
                current_task=current_task,
                origin=origin,
                geodesic=lambda start, end: self.geodesic(pathfinder, start, end),
                agent=agent,
                reranker=reranker,
                distance_cache=distance_cache,
                candidate_cap=candidate_cap,
                suffix_weight=suffix_weight,
                clock=self.clock.perf_counter,
                started_at=started_at,
            )
            event = build_scsp_shadow_decision(
                phase=phase,
                arm=arm,
                candidate_batch=calculation.candidate_batch,
                suffix_plan=calculation.suffix_plan,
                legacy_order=calculation.legacy_order,
                idea1_order=calculation.idea1_order,
                suffix_order=tuple(calculation.suffix_order),
                idea1_candidates=calculation.idea1_candidates,
                rerank_diagnostics=calculation.rerank_diagnostics,
                version_gate=version_gate,
                origin_position=[float(value) for value in origin],
                timing_ms=calculation.timing_ms,
                distance_cache=calculation.distance_cache,
                candidate_cap=int(candidate_cap),
                suffix_weight=float(suffix_weight),
                idea1_config=idea1_config,
                provenance=provenance,
                episode_identity=episode_identity,
            )
            event['status'] = 'planned'
            event['provider_candidate_batch'] = candidate_batch
            event['identity_sidecar_sha256'] = provenance.get('identity_sidecar_sha256')
            return event
        except Exception as exc:
            reason = (
                FallbackReason.INVALID_CANDIDATE_BATCH.value
                if isinstance(exc, SCSPContractError)
                else FallbackReason.PLANNER_ERROR.value
            )
            return self._fallback_event(
                phase=phase,
                arm=arm,
                current_task=current_task,
                reason=reason,
                version_gate=version_gate,
                candidate_batch=candidate_batch,
                origin=origin,
                started_at=started_at,
                error=exc,
                failure_stage='projection_rerank_or_planning',
                provenance=provenance,
                candidate_cap=candidate_cap,
                suffix_weight=suffix_weight,
                idea1_config=idea1_config,
                episode_identity=episode_identity,
            )

    def record_decision(self, *, output_path=None, decision_event=None, **kwargs):
        """Persist best-effort SCSP telemetry after the dispatch transaction."""
        compute_kwargs = dict(kwargs)
        try:
            if decision_event is not None:
                event = dict(decision_event)
            else:
                provider = compute_kwargs.pop('provider', None)
                if provider is not None and 'candidate_batch' not in compute_kwargs:
                    compute_kwargs['candidate_batch'] = provider.get_candidate_batch()
                event = self.compute_decision(**compute_kwargs)
        except Exception as exc:
            try:
                event = self._fallback_event(
                    phase=kwargs.get('phase', 'unknown'),
                    arm=kwargs.get('arm', 'unknown'),
                    current_task=kwargs.get('current_task', 0),
                    reason=FallbackReason.PLANNER_ERROR.value,
                    version_gate=kwargs.get('provider_event', {}).get('version_gate', {}),
                    candidate_batch=compute_kwargs.get('candidate_batch'),
                    origin=kwargs.get('origin', (0.0, 0.0, 0.0)),
                    started_at=self.clock.perf_counter(),
                    error=exc,
                    failure_stage='unhandled_shadow_compute',
                    provenance=kwargs.get('provenance'),
                    candidate_cap=kwargs.get('candidate_cap'),
                    suffix_weight=kwargs.get('suffix_weight'),
                    idea1_config=kwargs.get('idea1_config'),
                    episode_identity=kwargs.get('episode_identity'),
                )
            except Exception as fallback_exc:
                fallback_mode = str(kwargs.get('runtime_mode', 'shadow'))
                event = {
                    'schema_version': SHADOW_SCHEMA_VERSION,
                    'phase': str(kwargs.get('phase', 'unknown')),
                    'arm': str(kwargs.get('arm', 'unknown')),
                    'mode': fallback_mode,
                    'shadow_only': fallback_mode == 'shadow',
                    'status': 'fallback',
                    'dispatch_order_source': 'not_applied',
                    'current_task': 0,
                    'fallback_reason': FallbackReason.PLANNER_ERROR.value,
                    'error': {'type': type(exc).__name__, 'message': str(exc)},
                    'fallback_construction_error': {
                        'type': type(fallback_exc).__name__,
                        'message': str(fallback_exc),
                    },
                }
        from experiments.recorder import record_planning_decision

        return record_planning_decision(
            event,
            output_path=output_path,
            current_task_hint=kwargs.get('current_task', 0),
            serialize=dumps_json_strict,
        )

    def prepare_refresh(
        self,
        *,
        phase,
        runtime_mode,
        arm,
        provider,
        provider_event,
        raw_goals,
        current_task,
        origin,
        pathfinder,
        agent,
        reranker,
        distance_cache,
        candidate_cap,
        suffix_weight,
        idea1_config,
        provenance,
        episode_identity,
        scene_id,
        episode_id,
        ordered_categories,
        provider_events_path=None,
        preserve_existing_on_missing=False,
    ):
        from pbsp.navigation.planning import prepare_candidate_refresh

        def plan_once(attempt_goals, attempt_provider_event):
            (projected, projected_indices) = self.project_goals(
                attempt_goals, pathfinder, return_indices=True
            )
            decision = self.compute_decision(
                phase=phase,
                arm=arm,
                provider_event=attempt_provider_event,
                candidate_batch=provider.get_candidate_batch(),
                projected_goals=projected,
                projected_indices=projected_indices,
                current_task=current_task,
                origin=origin,
                pathfinder=pathfinder,
                agent=agent,
                reranker=reranker,
                distance_cache=distance_cache,
                candidate_cap=candidate_cap,
                suffix_weight=0.0 if runtime_mode == 'idea1_only' else suffix_weight,
                idea1_config=idea1_config,
                provenance=provenance,
                episode_identity=episode_identity,
            )
            gate = _scsp_dispatch_gate(
                provider,
                decision,
                scene_id=scene_id,
                episode_id=episode_id,
                ordered_categories=ordered_categories,
            )
            return (projected, decision, gate)

        def make_fallback(exc):
            return self._fallback_event(
                phase=phase,
                arm=arm,
                current_task=current_task,
                reason=FallbackReason.PLANNER_ERROR.value,
                version_gate=provider_event.get('version_gate', {}),
                candidate_batch=None,
                origin=origin,
                started_at=self.clock.perf_counter(),
                error=exc,
                failure_stage='unhandled_plan_attempt',
                provenance=provenance,
                candidate_cap=candidate_cap,
                suffix_weight=suffix_weight,
                idea1_config=idea1_config,
                episode_identity=episode_identity,
            )

        def read_retry_event():
            return _record_scsp_provider_event(
                provider,
                phase=f'{phase}_synchronous_requery',
                current_task=current_task,
                scene_id=scene_id,
                episode_id=episode_id,
                ordered_categories=ordered_categories,
                output_path=provider_events_path,
            )

        result = prepare_candidate_refresh(
            runtime_mode=runtime_mode,
            provider=provider,
            provider_event=provider_event,
            raw_goals=raw_goals,
            plan_once=plan_once,
            project_fallback=lambda goals: self.project_goals(goals, pathfinder),
            make_fallback=make_fallback,
            read_dispatch_gate=lambda decision: _scsp_dispatch_gate(
                provider,
                decision,
                scene_id=scene_id,
                episode_id=episode_id,
                ordered_categories=ordered_categories,
            ),
            read_retry_event=read_retry_event,
            preserve_existing_on_missing=preserve_existing_on_missing,
        )
        return (result.projected_goals, result.decision, result.retry_provider_events)
