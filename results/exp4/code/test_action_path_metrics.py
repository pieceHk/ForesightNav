"""Hand-calculated displacement attribution, including changing GT labels."""
import copy
import unittest

from action_path_metrics import action_paths, GT_CRITERION


def gt(scene, episode, task, position):
    valid = position[0] == 0.
    return {'y': int(valid), 'distance_to_goal_m': .5 if valid else 2., 'criterion': GT_CRITERION}


def final(candidate, frame, y=0, k=0):
    return dict(event='verification_final', scene_id='s', episode_id='0', k=k,
                candidate_id=candidate, t=frame, last_t=frame, y=y, v=0)


def row(frame, x, candidate='b', *, source='query_candidate', point=10., stop=False,
        k=0, confirmed=False, recovery=None):
    target = dict(schema_version='blp_action_target_v1', frame_index=frame, task_idx=k,
                  source=source, position=None if source == 'frontier' else [point, 0., 0.],
                  candidate_id=candidate if source == 'query_candidate' else None,
                  query_identity_status='resolved' if source == 'query_candidate' else 'not_query')
    if confirmed:
        target['pursuit_mode'] = 'confirmed_visual_goal'
    if recovery:
        target['runner_recovery_mode'] = recovery
    return dict(scene_id='s', episode_id='0', frame_index=frame, subtask_index=k,
                world_pose=[1., 0., 0., 0., x, 0., 0.], is_stop=stop,
                blp_action_target=target)


class ActionPathTests(unittest.TestCase):
    def mixed_case(self):
        rows = [row(1, 0.), row(2, 1., confirmed=True),
                row(3, 2., source='navmesh_override', recovery='escape'),
                row(4, 3., point=0.), row(5, 4., candidate='g', point=0.),
                row(6, 4., stop=True)]
        events = [final('b', 2), final('b', 3), final('b', 4, y=1), final('g', 5, y=1)]
        return rows, events

    def test_mixed_labels_confirmation_and_recovery_have_exact_nonoverlapping_cost(self):
        rows, events = self.mixed_case()
        value = action_paths(events, {('s', '0'): rows}, gt, {('s', '0', 0): 4.})
        self.assertEqual(value['invalid_visit_path_m'], 2.)
        self.assertEqual(value['invalid_approach_and_verification_path_m'], 1.)
        self.assertEqual(value['invalid_confirmed_approach_tail_m'], 1.)
        self.assertEqual(value['invalid_identity_pursuit_path_m'], 3.)
        self.assertEqual(value['checked_invalid_identity_but_current_target_GT_valid_path_m'], 1.)
        self.assertEqual(value['mixed_GT_checked_identities'], [['s', '0', 0, 'b']])
        self.assertEqual(value['verification_then_effective_recovery_actions'], 1)
        self.assertEqual(sum(value['accepted_action_path_partition_m'].values()), 4.)
        self.assertEqual(value['native_stage_path_max_abs_error_m'], 0.)
        self.assertTrue(value['invalid_path_complete'])

    def test_recorded_after_pose_is_used_and_checked_against_next_observation(self):
        rows = [row(1, 0.), row(2, .5, stop=True)]
        for r in rows:
            r['world_pose_after'] = [1., 0., 0., 0., .5, 0., 0.]
        value = action_paths([final('b', 1)], {('s', '0'): rows}, gt, {('s', '0', 0): .5})
        self.assertEqual(value['invalid_visit_path_m'], .5)
        self.assertEqual(value['last_STOP_without_after_pose_assumed_stationary'], 0)
        rows[0]['world_pose_after'][4] = .7
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            action_paths([final('b', 1)], {('s', '0'): rows}, gt)

    def test_legacy_is_explicit_and_partial_recording_is_rejected(self):
        rows = [row(1, 0.), row(2, 1., stop=True)]
        legacy = [{k: v for k, v in r.items() if k != 'blp_action_target'} for r in rows]
        self.assertIsNone(action_paths([], {('s', '0'): legacy}, None))
        del rows[0]['blp_action_target']
        with self.assertRaisesRegex(ValueError, 'missing and present'):
            action_paths([], {('s', '0'): rows}, None)

    def test_unresolved_identity_stays_in_the_path_partition_and_blocks_completion(self):
        rows = [row(1, 0.), row(2, 1., stop=True)]
        for r in rows:
            r['blp_action_target'].update(candidate_id=None, query_identity_status='unresolved')
        value = action_paths([final('b', 1)], {('s', '0'): rows}, gt, {('s', '0', 0): 1.})
        self.assertFalse(value['invalid_path_complete'])
        self.assertEqual(value['accepted_action_path_partition_m']['unresolved_query'], 1.)
        self.assertEqual(value['invalid_visit_path_m'], 0.)
        self.assertTrue(value['invalid_path_association_gaps'])

    def test_verification_identity_disagreement_is_not_silently_relabelled(self):
        rows = [row(1, 0., candidate='g'), row(2, 1., candidate='g', stop=True)]
        events = [final('b', 1), dict(event='candidate_first_appearance', scene_id='s',
                  episode_id='0', k=0, candidate_id='g')]
        value = action_paths(events, {('s', '0'): rows}, gt)
        self.assertFalse(value['invalid_path_complete'])
        self.assertEqual(value['invalid_path_association_gaps'][0]['reason'],
                         'verification_and_action_target_identity_disagree')

    def test_native_path_mismatch_is_rejected(self):
        rows, events = self.mixed_case()
        with self.assertRaisesRegex(ValueError, 'Action/native stage path mismatch'):
            action_paths(events, {('s', '0'): rows}, gt, {('s', '0', 0): 5.})

    def test_missing_or_non_native_gt_cannot_populate_cost(self):
        rows, events = self.mixed_case()
        with self.assertRaisesRegex(ValueError, 'post-hoc GT'):
            action_paths(events, {('s', '0'): rows}, None)
        with self.assertRaisesRegex(ValueError, 'pinned native GT'):
            action_paths(events, {('s', '0'): rows}, lambda *args: {'criterion': 'other', 'y': 0})

    def test_valid_unchecked_and_frontier_costs_stay_outside_invalid_cost(self):
        rows = [row(1, 0., source='frontier'), row(2, 1., candidate='u'),
                row(3, 2., candidate='g', point=0.), row(4, 3., source='frontier', stop=True)]
        events = [final('g', 3, y=1), dict(event='candidate_first_appearance', scene_id='s',
                  episode_id='0', k=0, candidate_id='u')]
        value = action_paths(events, {('s', '0'): rows}, None)
        self.assertEqual(value['invalid_visit_path_m'], 0.)
        for key in ('frontier', 'unchecked_query', 'checked_valid'):
            self.assertEqual(value['accepted_action_path_partition_m'][key], 1.)
        self.assertEqual(value['accepted_action_path_m'], 3.)

    def test_stop_transition_keeps_displacement_in_the_pre_stop_stage(self):
        rows = [row(1, 0.), row(2, 1., stop=True),
                row(3, 1., candidate='c', point=0., k=1),
                row(4, 1.5, candidate='c', point=0., k=1, stop=True)]
        events = [final('b', 1), final('c', 3, y=1, k=1)]
        value = action_paths(events, {('s', '0'): rows}, gt,
                             {('s', '0', 0): 1., ('s', '0', 1): .5})
        self.assertEqual(value['invalid_visit_path_m'], 1.)
        self.assertEqual(value['accepted_action_path_m'], 1.5)
        self.assertEqual(value['native_stage_path_check_count'], 2)
        rows[1]['blp_action_target']['task_idx'] = 1
        with self.assertRaisesRegex(ValueError, 'pre-STOP task'):
            action_paths(events, {('s', '0'): rows}, gt)

    def test_missing_terminal_stop_is_not_treated_as_a_complete_path(self):
        rows = [row(1, 0.), row(2, 1.)]
        with self.assertRaisesRegex(ValueError, 'end at STOP'):
            action_paths([], {('s', '0'): rows}, None)


if __name__ == '__main__':
    unittest.main()
