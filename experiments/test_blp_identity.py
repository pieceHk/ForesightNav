"""Target-identity logging checks; candidate grouping and policy stay unchanged."""
import copy
import unittest
from types import SimpleNamespace

from experiments.blp_observer import bind_action_target, register_candidate


def planning():
    return SimpleNamespace(blp_context={'t': 1, 'batch': SimpleNamespace(scene_id='s', episode_id='0')},
                           blp_directory=None)


def register(p, point, task=0):
    # Suppress file output without changing the registration implementation.
    from unittest.mock import patch
    record = SimpleNamespace(position=point, semantic_score=.25)
    with patch('experiments.blp_observer.append'):
        return register_candidate(p, task, record)


def target(point, task=0, **extra):
    return {'schema_version': 'blp_action_target_v1', 'task_idx': task,
            'source': 'query_candidate', 'position': point, **extra}


class IdentityBindingTests(unittest.TestCase):
    def chain(self):
        p = planning()
        for x in (0., .7, 1.4):
            register(p, [x, 0., 0.])
        return p

    def test_chain_drift_retains_exact_registered_point(self):
        p = self.chain()
        self.assertEqual(len(p.blp_context['candidate_registry'][0]), 1)
        before = copy.deepcopy(p.blp_context)
        original = target([0., 0., 0.])
        result = bind_action_target(p, {'blp_action_target': original}, 0, 10)
        self.assertEqual(result['candidate_id'], '0:0')
        self.assertEqual(result['query_identity_resolution'], 'exact_registered_position')
        self.assertEqual(result['query_identity_status'], 'resolved')
        self.assertEqual(p.blp_context, before)
        self.assertNotIn('candidate_id', original)

    def test_existing_geometry_binding_keeps_priority(self):
        p = self.chain()
        register(p, [.1, 0., 0.])
        # The alias for zero still points at 0:0, but the old geometric rule
        # now resolves to 0:1. Preserve that established behavior.
        result = bind_action_target(p, {'blp_action_target': target([0., 0., 0.])}, 0, 11)
        self.assertEqual(result['candidate_id'], '0:1')
        self.assertNotIn('query_identity_resolution', result)

    def test_repeated_exact_position_remembers_latest_registration(self):
        p = self.chain()
        item = register(p, [0., 0., 0.])
        self.assertEqual(item['candidate_id'], '0:1')
        self.assertEqual(p.blp_context['candidate_position_identities'][0][(0., 0., 0.)], '0:1')

    def test_task_identity_cannot_cross(self):
        p = self.chain()
        result = bind_action_target(p, {'blp_action_target': target([0., 0., 0.], task=1)}, 1, 10)
        self.assertIsNone(result['candidate_id'])
        self.assertEqual(result['query_identity_status'], 'unresolved')

    def test_nearby_or_different_height_unregistered_point_stays_unresolved(self):
        p = self.chain()
        for point in ([1e-8, 0., 0.], [0., 1., 0.], [-.2, 0., 0.]):
            result = bind_action_target(p, {'blp_action_target': target(point)}, 0, 10)
            self.assertIsNone(result['candidate_id'])
            self.assertEqual(result['query_identity_status'], 'unresolved')

    def test_recovery_override_has_no_query_identity(self):
        p = self.chain()
        result = bind_action_target(p, {'blp_action_target': target([0., 0., 0.], candidate_id='0:0'),
                  'blp_effective_recovery': {'position': [0., 0., 0.], 'mode': 'escape'}}, 0, 10)
        self.assertEqual(result['source'], 'navmesh_override')
        self.assertIsNone(result['candidate_id'])
        self.assertEqual(result['query_identity_status'], 'not_query')

    def test_confirmed_visual_parent_keeps_priority(self):
        p = self.chain()
        register(p, [.1, 0., 0.])
        result = bind_action_target(p, {'blp_action_target': target([0., 0., 0.],
                 candidate_id='0:0', pursuit_mode='confirmed_visual_goal')}, 0, 10)
        self.assertEqual(result['candidate_id'], '0:0')
        self.assertEqual(result['pursuit_mode'], 'confirmed_visual_goal')
        self.assertNotIn('query_identity_resolution', result)

    def test_episode_reset_clears_remembered_identities(self):
        p = self.chain()
        p.blp_context.clear()
        result = bind_action_target(p, {'blp_action_target': target([0., 0., 0.])}, 0, 1)
        self.assertEqual(result['query_identity_status'], 'unresolved')

    def test_stale_alias_cannot_create_unknown_identity(self):
        p = self.chain()
        p.blp_context['candidate_position_identities'][0][(0., 0., 0.)] = '0:missing'
        result = bind_action_target(p, {'blp_action_target': target([0., 0., 0.])}, 0, 10)
        self.assertIsNone(result['candidate_id'])
        self.assertEqual(result['query_identity_status'], 'unresolved')


if __name__ == '__main__':
    unittest.main()

