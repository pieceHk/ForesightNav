"""CPU checks for the Eq.(13) mechanism and runtime contracts."""
import copy
import math
import unittest
from types import SimpleNamespace

import numpy as np

from pbsp.navigation.blp import BeliefParameters, build_snapshot, solve_snapshot
from pbsp.navigation.evidence import CandidateEvidenceBank
from pbsp.navigation.planning import apply_runtime_order
from pbsp.types import CandidateBatch, CandidateRecord, GeodesicDistance


def toy(seed=1, n=3):
    rng = np.random.RandomState(seed)
    points = rng.rand(1 + n * 3, 3) * 10
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1).tolist()
    layers = [[dict(raw_rank=i, E=float(rng.rand()), N_views=int(rng.randint(13)),
                    location=1 + m * 3 + i) for i in range(3)] for m in range(n)]
    return dict(k=0, layers=layers, distances=distances, nearest_frontier=[None] * len(points),
                frontier_available=False, all_raw_ranks=list(range(3)))


def brute(snapshot, theta):
    layers, d = snapshot['layers'], snapshot['distances']
    def value(m, x, remaining):
        if m == len(layers):
            return 0.
        if not remaining:
            return theta.c_exp + value(m + 1, x, tuple(range(3)))
        terms = []
        for i in remaining:
            item = layers[m][i]
            y = item['location']
            b = theta.evidence(item['E'], item['N_views'])[2]
            terms.append(d[x][y] + b * value(m + 1, y, tuple(range(3))) +
                         (1 - b) * value(m, y, tuple(j for j in remaining if j != i)))
        return min(terms)
    result = {}
    for i, item in enumerate(layers[0]):
        y = item['location']
        b = theta.evidence(item['E'], item['N_views'])[2]
        result[i] = d[0][y] + b * value(1, y, tuple(range(3))) + (1 - b) * value(0, y, tuple(j for j in range(3) if j != i))
    return result


class BLPTests(unittest.TestCase):
    def test_memo_matches_independent_brute(self):
        for seed in range(8):
            snapshot, theta = toy(seed), BeliefParameters()
            result = solve_snapshot(snapshot, theta)
            expected = brute(snapshot, theta)
            for item in result['candidates']:
                self.assertAlmostEqual(item['Q'], expected[item['raw_rank']], places=10)

    def test_last_goal_equalities(self):
        snapshot, theta = toy(n=1), BeliefParameters()
        def costs(variant):
            return {r['raw_rank']: r['Q'] for r in solve_snapshot(snapshot, theta, variant)['candidates']}
        self.assertEqual(costs('full'), costs('myopic'))
        self.assertEqual(costs('suffix_only'), costs('nearest'))

    def test_unit_belief_reduces_to_deterministic(self):
        snapshot = toy()
        for layer in snapshot['layers']:
            for item in layer:
                item['N_views'] = 0
        theta = BeliefParameters(pi=1.)
        a, b = [solve_snapshot(snapshot, theta, variant) for variant in ('full', 'suffix_only')]
        self.assertEqual(a['ordered_raw_ranks'], b['ordered_raw_ranks'])
        self.assertEqual([r['Q'] for r in a['candidates']], [r['Q'] for r in b['candidates']])

    def test_myopic_ignores_all_future_layers(self):
        snapshot = toy()
        before = solve_snapshot(snapshot, BeliefParameters(), 'myopic')
        snapshot['layers'][1:] = [[], []]
        after = solve_snapshot(snapshot, BeliefParameters(), 'myopic')
        self.assertEqual([r['Q'] for r in before['candidates']], [r['Q'] for r in after['candidates']])

    def test_permutation_and_hysteresis(self):
        snapshot = toy(n=1)
        snapshot['all_raw_ranks'].extend([5, 12, 19])
        result = solve_snapshot(snapshot, BeliefParameters(), 'full', active_raw_rank=2,
                                switch_delta_m=1000.)
        self.assertEqual(result['ordered_raw_ranks'][0], 2)
        self.assertEqual(sorted(result['ordered_raw_ranks']), snapshot['all_raw_ranks'])
        self.assertEqual(len(result['ordered_raw_ranks']), 6)

    def test_frontier_boundary_and_no_frontier(self):
        snapshot = dict(k=0, layers=[[], []], distances=[[0., 2.], [2., 0.]],
                        nearest_frontier=[1, 1], frontier_available=True, all_raw_ranks=[])
        # Trigger exhaustion through a zero-belief candidate and ensure terminal
        # cost follows the actual frontier location rather than the old position.
        snapshot['layers'][0] = [dict(raw_rank=0, E=.2, N_views=0, location=0)]
        snapshot['all_raw_ranks'] = [0]
        theta = BeliefParameters(pi=0., c_exp=7.)
        self.assertEqual(solve_snapshot(snapshot, theta)['candidates'][0]['Q'], 16.)
        snapshot['nearest_frontier'] = [None, None]
        self.assertEqual(solve_snapshot(snapshot, theta)['candidates'][0]['Q'], 1000.)

    def test_bearing_coverage_rotation_and_full_circle(self):
        bank = CandidateEvidenceBank()
        bank.refresh({0: [[0, 0, 0]]})
        depth = np.ones((120, 160)) * 10
        def observe(angle):
            camera = np.array([2 * math.cos(angle), 0, 2 * math.sin(angle)])
            backward = camera / 2
            right = np.cross([0, 1, 0], backward)
            rotation = np.array([right, [0, 1, 0], backward])
            bank.observe_coverage(camera, rotation, depth, 79.)
            return camera, rotation
        camera, rotation = observe(.1)
        for yaw in np.linspace(-.5, .5, 10):
            turn = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0],
                             [-math.sin(yaw), 0, math.cos(yaw)]])
            bank.observe_coverage(camera, turn @ rotation, depth, 79.)
        self.assertEqual(bank.coverage_views(0, [0, 0, 0]), 1)
        for i in range(12):
            observe((i + .2) * math.pi / 6)
        self.assertEqual(bank.coverage_views(0, [0, 0, 0]), 12)
        bank.reset(preserve_coverage=True)
        bank.refresh({0: [[.05, 0, .05]]})
        self.assertEqual(bank.coverage_views(0, [.05, 0, .05]), 12)

    def test_visibility_occlusion_range_and_fov(self):
        bank = CandidateEvidenceBank()
        bank.refresh({0: [[0, 0, -2], [0, 0, 2], [0, 0, -7]]})
        bank.observe_coverage([0, 0, 0], np.eye(3), np.ones((20, 20)), 79.)
        self.assertEqual([bank.coverage_views(0, x) for x in ([0, 0, -2], [0, 0, 2], [0, 0, -7])], [0, 0, 0])
        bank.observe_coverage([0, 0, 0], np.eye(3), np.ones((20, 20)) * 3, 79.)
        self.assertEqual(bank.coverage_views(0, [0, 0, -2]), 1)

    def test_raw_rank_cap_blacklist_and_unreachable(self):
        records = tuple(CandidateRecord(task_idx=0, category='chair', position=(i, 0, 0),
                        raw_rank=10-i, semantic_score=.2, support_count=3,
                        support_score=0., map_revision=1) for i in range(8))
        points = [list(r.position) for r in records]
        batch = CandidateBatch(scene_id='s', episode_id='0', map_revision=1,
                               last_fused_keyframes=1, query_serial=1,
                               ordered_categories=('chair',), positions_by_task=((0, tuple(map(tuple, points))),),
                               records_by_task=((0, records),))
        agent = SimpleNamespace(gs_poses={}, frontier_flag={}, visited_candidate_positions={0: [[7, 0, 0]]})
        def distance(a, b):
            return GeodesicDistance.unreachable() if b[0] == 5 else GeodesicDistance.available(abs(a[0]-b[0]))
        _, snapshot = build_snapshot(candidate_batch=batch, projected_goals={0: points},
                        projected_indices={0: list(range(8))}, current_task=0, origin=[-1, 0, 0],
                        geodesic=distance, agent=agent, distance_cache={})
        self.assertEqual([r['raw_rank'] for r in snapshot['layers'][0]], [6, 7, 8, 9, 10])
        result = solve_snapshot(snapshot, BeliefParameters())
        self.assertEqual(set(result['ordered_raw_ranks']), set(range(3, 11)))

    def test_frontier_world_round_trip(self):
        from experiments.blp_observer import observe_frontier
        from scipy.spatial.transform import Rotation
        for yaw in (0., .3, -2.):
            rotation = Rotation.from_euler('y', yaw).as_matrix()
            agent = SimpleNamespace(rot_matrix_v2=rotation, tran_v2=np.array([3., 0., 2.]))
            obs = SimpleNamespace(gps=np.array([1., 2.]), globalpose=np.r_[0, 0, 0, 1, 4., 0., 5.])
            sensor = np.array([3., 4., 0., 70., 140., 40., 120.])
            mask = np.zeros((30, 30))
            mask[10, 20] = 1
            observe_frontier(agent, obs, {'frontier_map': mask, 'sensor_pose': sensor})
            world = np.array(agent.blp_frontier_world[0])
            local = rotation @ world - agent.tran_v2
            gps = np.array([-local[2], -local[0]])
            origin_pixel = sensor[:2] * 20 - [sensor[5], sensor[3]]
            actual_pixel = (gps - obs.gps) * 20 + origin_pixel
            np.testing.assert_allclose(actual_pixel, [20, 10], atol=1e-10)

    def test_blp_dispatch_preserves_future_and_gate(self):
        records = tuple(CandidateRecord(task_idx=0, category='chair', position=(i, 0, 0),
                        raw_rank=i, semantic_score=.2, support_count=3,
                        support_score=0., map_revision=1) for i in range(3))
        batch = CandidateBatch(scene_id='s', episode_id='0', map_revision=1,
                               last_fused_keyframes=1, query_serial=1,
                               ordered_categories=('chair',),
                               positions_by_task=((0, tuple(r.position for r in records)),),
                               records_by_task=((0, records),))
        goals = {0: [list(r.position) for r in records], 1: [[90, 0, 0]]}
        decision = dict(current_task=0, status='planned', candidate_batch=batch,
                         dispatch_version_gate={'would_invalidate': False},
                         blp={'ordered_raw_ranks': [2, 0, 1]})
        dispatched, metadata = apply_runtime_order(goals, decision, 'blp_eq13')
        self.assertEqual(dispatched[0], [[2., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
        self.assertEqual(dispatched[1], goals[1])
        self.assertEqual(metadata['source'], 'blp')
        decision['dispatch_version_gate']['would_invalidate'] = True
        dispatched, metadata = apply_runtime_order(goals, decision, 'blp_eq13')
        self.assertEqual(dispatched, goals)
        self.assertFalse(metadata['externally_ranked'])

    def test_uncalibrated_active_mode_is_rejected(self):
        from experiments.blp_observer import initialize_blp
        planning = SimpleNamespace(blp_config={'enabled': True})
        with self.assertRaises(ValueError):
            initialize_blp(planning, SimpleNamespace(), '/tmp')

    def test_myopic_does_not_query_future_categories(self):
        from pbsp.runtime.mapping_client import MappingClient
        client = MappingClient('/tmp/unused', task_reader=lambda *_: [])
        client._tasks = [['chair', 'object'], ['bed', 'object'], ['couch', 'object']]
        client.current_only_task = 1
        calls = []
        def send(request):
            calls.append(request)
            return {'ok': True, 'results': {'bed': []}, 'candidate_records_by_category': {'bed': []}}
        client._send_recv = send
        client._capture_status_view = lambda: ({}, None)
        client._publish_candidate_batch = lambda **kwargs: None
        result = client.get_all_goals()
        self.assertEqual(calls[0]['categories'], ['bed'])
        self.assertEqual(result, {0: [], 1: [], 2: []})
        # The protocol-error fallback must obey the same restriction.
        called_tasks = []
        client.get_goals = lambda task, cat: called_tasks.append(task) or []
        client._get_all_goals_sequential()
        self.assertEqual(called_tasks, [1])

    def test_positive_slope_fit_and_unique_candidate_prior(self):
        from experiments.blp_calibration import fit, diagnose
        events = []
        for i, (e, label) in enumerate([(.1, 0), (.2, 0), (.25, 1), (.3, 0), (.35, 1), (.4, 1)]):
            events.append(dict(event='verification_final', E=e, v=label, y=label,
                               scene_id='s', episode_id=str(i), k=0, candidate_id='0:0'))
        events.append(dict(event='empty_activation', path_to_first_candidate_m=5.))
        events.append(dict(events[0]))  # A second failed visit is not a new candidate.
        theta = fit(events)
        self.assertGreater(theta.a, 0.)
        self.assertEqual(theta.pi, .5)
        self.assertEqual(theta.c_exp, 5.)
        diagnostic = diagnose(events, theta)
        self.assertEqual(diagnostic['fp'], 0)
        self.assertEqual(diagnostic['fn'], 0)
        self.assertEqual(sum(b['n'] for b in diagnostic['reliability_bins']), 7)

    def test_separated_calibration_refuses_invented_finite_fit(self):
        from experiments.blp_calibration import fit, CalibrationDataError
        events = [dict(event='verification_final', E=e, v=v, y=v, scene_id='s',
                       episode_id=str(i), k=0, candidate_id='0:0')
                  for i, (e, v) in enumerate([(.1, 0), (.2, 0), (.3, 1), (.4, 1)])]
        events.append(dict(event='empty_activation', path_to_first_candidate_m=2.))
        with self.assertRaises(CalibrationDataError):
            fit(events)

    def test_negative_check_finalizes_once_on_abandonment(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from experiments.blp_observer import observe_verification, prepare_action
        record = SimpleNamespace(position=[1, 0, 0], semantic_score=.25)
        batch = SimpleNamespace(scene_id='s', episode_id='0', records_for_task=lambda _: [record])
        with TemporaryDirectory() as folder:
            planning = SimpleNamespace(blp_directory=Path(folder), blp_parameters=BeliefParameters(),
                       blp_context={'batch': batch, 't': 1, 'path_m': 0.})
            agent = SimpleNamespace(gs_poses={0: [[1, 0, 0]]}, frontier_flag={0: 0},
                                    candidate_evidence=CandidateEvidenceBank())
            evaluator = SimpleNamespace(get_episode_metrics=lambda: {'goat_sub-task_success': 0},
                         label_candidate=lambda *_: {'y': 0, 'criterion': 'test'})
            prepare_action(planning, agent, 0)
            planning.blp_context['path_m'] = 2.
            metadata = {'verification_event': {'candidate_position': [1, 0, 0], 'task_idx': 0, 'confirmed': False}}
            observe_verification(planning, agent, evaluator, metadata)
            observe_verification(planning, agent, evaluator, metadata)
            path = Path(folder) / 'blp_events.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(sum(r['event'] == 'verification_final' for r in rows), 0)
            agent.gs_poses[0] = [[3, 0, 0]]
            observe_verification(planning, agent, evaluator, {})
            observe_verification(planning, agent, evaluator, {})
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            finalized = [r for r in rows if r['event'] == 'verification_final']
            self.assertEqual(len(finalized), 1)
            self.assertEqual(finalized[0]['v'], 0)
            self.assertEqual(finalized[0]['verification_path_m'], 2.)


if __name__ == '__main__':
    unittest.main()
