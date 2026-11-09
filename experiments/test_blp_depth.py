"""Exercise the real observation adapter at the BLP depth-unit boundary."""
from types import SimpleNamespace
import unittest

import numpy as np
import quaternion

from experiments.blp_observer import observe_frame
from pbsp.adapters.habitat import HabitatEnvironment, HabitatObservationSettings
from pbsp.navigation.evidence import CandidateEvidenceBank


class BlpDepthUnitTests(unittest.TestCase):
    def check_visibility(self, *, normalized, value):
        sensor = SimpleNamespace(normalize_depth=normalized, min_depth=.5,
                                 max_depth=5., hfov=79.)
        config = SimpleNamespace(habitat=SimpleNamespace(simulator=SimpleNamespace(
            agents=SimpleNamespace(main_agent=SimpleNamespace(sim_sensors=SimpleNamespace(
                depth_sensor=sensor))))))
        camera = SimpleNamespace(position=np.zeros(3), rotation=np.quaternion(1., 0., 0., 0.))
        simulator = SimpleNamespace(get_agent_state=lambda: SimpleNamespace(
            sensor_states={'depth': camera}))
        env = HabitatEnvironment(SimpleNamespace(sim=simulator),
            settings=HabitatObservationSettings(.5, 5., 'disabled'), categories=(),
            observation_factory=SimpleNamespace, action_ids=None)
        env.config = config
        env.current_episode = SimpleNamespace(tasks=[])
        raw = np.full((20, 30, 1), value, dtype=np.float32)
        obs = env._preprocess_obs({'depth': raw, 'rgb': np.zeros((20, 30, 3), dtype=np.uint8),
            'multigoal': [], 'compass': np.zeros(1), 'gps': np.zeros(2),
            'globalpose': np.array([1., 0., 0., 0., 0., 0., 0.])})
        original = obs.depth.copy()
        bank = CandidateEvidenceBank()
        bank.refresh({0: [[0., 0., -2.]], 1: [[0., 0., -4.]]}, preserve_source_order=True)
        planning = SimpleNamespace(blp_context={'activation_task': 0, 'path_m': 0.})
        observe_frame(planning, SimpleNamespace(candidate_evidence=bank), env, obs, 1, 0)
        # A surface at 2.75m leaves the 2m centre visible and occludes the 4m centre.
        self.assertEqual(bank.coverage_views(0, [0., 0., -2.]), 1)
        self.assertEqual(bank.coverage_views(1, [0., 0., -4.]), 0)
        np.testing.assert_array_equal(obs.depth, original)
        np.testing.assert_array_equal(raw[:, :, 0], original)

    def test_normalized_sensor_depth_has_metric_visibility(self):
        self.check_visibility(normalized=True, value=.5)

    def test_metric_sensor_depth_is_not_rescaled(self):
        self.check_visibility(normalized=False, value=2.75)


if __name__ == '__main__':
    unittest.main()
