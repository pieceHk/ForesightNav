"""Test proposed trace helpers and accepted-action adapter without applying it."""
import copy
from enum import IntEnum
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

BASE = Path(__file__).resolve().parent / 'pending_action_trace/proposed'


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, BASE / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


observer = module('pending_blp_observer', 'experiments/blp_observer.py')
trajectory = module('pending_trajectory', 'experiments/trajectory.py')


class Action(IntEnum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2


class Environment:
    def __init__(self):
        self.habitat_env = SimpleNamespace(current_episode=SimpleNamespace(
            scene_id='scenes/test.basis.glb', episode_id='0'))
        self.reset()

    def reset(self):
        self.task, self.accepted, self.fail = 0, 0, False
        self.pose = [1., 0., 0., 0., 0., 0., 0.]

    def get_observation(self):
        return SimpleNamespace(globalpose=list(self.pose), task_observations={
            'tasks': [{'category': 'chair'}, {'category': 'bed'}]})

    def apply_action(self, action, info=None):
        if self.fail:
            raise RuntimeError('rejected action')
        self.accepted += 1
        if action == Action.STOP:
            self.task += 1
        elif action == Action.MOVE_FORWARD:
            self.pose[4] += .25


class PendingTraceTests(unittest.TestCase):
    def planning(self):
        return SimpleNamespace(blp_context={'candidate_registry': {0: [
            {'candidate_id': '0:0', 'position': [1., 0., 0.]},
            {'candidate_id': '0:1', 'position': [2., 0., 0.]}]}})

    def target(self, position=(1., 0., 0.), task=0):
        return observer.action_target_descriptor(SimpleNamespace(navmesh_override_goal=None), task, position)

    def wrapper(self, directory, metadata):
        env = Environment()
        wrapped = trajectory.TrajectoryEnvironment(env, directory=directory, identities={},
            dataset_manifest_sha256='synthetic', actions=Action, action_metadata=lambda: metadata)
        wrapped.reset()
        return env, wrapped

    def test_descriptor_copies_point_and_distinguishes_nonquery_sources(self):
        point = np.array([1., 2., 3.])
        desc = self.target(point)
        point[0] = 9.
        self.assertEqual(desc['position'], [1., 2., 3.])
        self.assertEqual(desc['source'], 'query_candidate')
        self.assertEqual(self.target(None)['source'], 'frontier')
        agent = SimpleNamespace(navmesh_override_goal=np.array([1., 2., 3.]))
        self.assertEqual(observer.action_target_descriptor(agent, 0, [1., 2., 3.])['source'], 'navmesh_override')
        with self.assertRaises(ValueError):
            self.target([float('nan'), 0., 0.])

    def test_registry_is_read_only_and_unresolved_query_is_explicit(self):
        planning = self.planning()
        original = copy.deepcopy(planning.blp_context)
        result = observer.bind_action_target(planning, {'blp_action_target': self.target()}, 0, 1)
        self.assertEqual(result['candidate_id'], '0:0')
        missing = observer.bind_action_target(planning, {'blp_action_target': self.target([10., 0., 0.])}, 0, 2)
        self.assertIsNone(missing['candidate_id'])
        self.assertEqual(missing['query_identity_status'], 'unresolved')
        self.assertEqual(planning.blp_context, original)

    def test_confirmed_visual_parent_identity_survives_query_centroid_drift(self):
        planning = self.planning()
        target = self.target()
        target.update(candidate_id='0:0', pursuit_mode='confirmed_visual_goal')
        planning.blp_context['candidate_registry'][0][0]['position'] = [20., 0., 0.]
        result = observer.bind_action_target(planning, {'blp_action_target': target}, 0, 99)
        self.assertEqual(result['candidate_id'], '0:0')
        self.assertEqual(result['position'], [1., 0., 0.])

    def test_recovery_does_not_rewrite_the_confirmed_parent(self):
        target = self.target()
        target['candidate_id'] = '0:0'
        original = copy.deepcopy(target)
        result = observer.bind_action_target(self.planning(), {
            'blp_action_target': target,
            'blp_effective_recovery': {'mode': 'escape', 'position': [5., 0., 0.]}}, 0, 9)
        self.assertEqual(result['source'], 'navmesh_override')
        self.assertEqual(result['runner_recovery_mode'], 'escape')
        self.assertIsNone(result['candidate_id'])
        self.assertEqual(target, original)

    def test_stop_uses_environment_task_before_transition_and_snapshots_target(self):
        metadata = {'stop_reason': 'verified_target'}
        parent = self.target()
        metadata['blp_action_target'] = observer.bind_action_target(
            self.planning(), {'blp_action_target': parent}, 0, 1)
        with tempfile.TemporaryDirectory() as directory, patch.object(trajectory, 'task_index', lambda env: env.task):
            env, wrapped = self.wrapper(directory, metadata)
            wrapped.get_observation()
            wrapped.apply_action(Action.STOP)
            self.assertEqual(env.task, 1)
            metadata['blp_action_target']['position'][0] = 999.
            row = json.loads(next(Path(directory).rglob('trajectory.jsonl')).read_text())
            self.assertEqual(row['subtask_index'], 0)
            self.assertEqual(row['blp_action_target']['task_idx'], 0)
            self.assertEqual(row['blp_action_target']['position'], [1., 0., 0.])

    def test_rejected_action_is_not_added_to_the_native_trace(self):
        metadata = {'blp_action_target': observer.bind_action_target(
            self.planning(), {'blp_action_target': self.target()}, 0, 1)}
        with tempfile.TemporaryDirectory() as directory, patch.object(trajectory, 'task_index', lambda env: env.task):
            env, wrapped = self.wrapper(directory, metadata)
            wrapped.get_observation()
            env.fail = True
            with self.assertRaisesRegex(RuntimeError, 'rejected action'):
                wrapped.apply_action(Action.MOVE_FORWARD)
            self.assertEqual(wrapped.frame, 0)
            self.assertEqual(list(Path(directory).rglob('trajectory.jsonl')), [])

    def test_frame_and_task_mismatches_are_rejected(self):
        metadata = {'blp_action_target': observer.bind_action_target(
            self.planning(), {'blp_action_target': self.target()}, 0, 2)}
        with tempfile.TemporaryDirectory() as directory, patch.object(trajectory, 'task_index', lambda env: env.task):
            env, wrapped = self.wrapper(directory, metadata)
            wrapped.get_observation()
            with self.assertRaisesRegex(ValueError, 'trajectory disagree'):
                wrapped.apply_action(Action.MOVE_FORWARD)
            self.assertEqual(env.accepted, 0)
        with self.assertRaisesRegex(ValueError, 'pre-STOP task'):
            observer.bind_action_target(self.planning(), {'blp_action_target': self.target()}, 1, 1)

    def test_old_mode_keeps_native_action_row_without_extension(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(trajectory, 'task_index', lambda env: env.task):
            _, wrapped = self.wrapper(directory, {})
            wrapped.get_observation()
            wrapped.apply_action(Action.MOVE_FORWARD)
            row = json.loads(next(Path(directory).rglob('trajectory.jsonl')).read_text())
            self.assertEqual(row, dict(schema_version='pbsp_trajectory_v1', scene_id='test', episode_id='0',
                subtask_index=0, frame_index=1, task_step=1, action_id=1, action_name='MOVE_FORWARD',
                is_stop=False, stop_reason=None, world_pose=[1., 0., 0., 0., 0., 0., 0.]))


if __name__ == '__main__':
    unittest.main()
