"""Golden supplementary-metric cases and actual native-artifact checks."""
from pathlib import Path
import math
import unittest

from report_variants import conditional_rates, effects_and_pairs, prefix_metrics, summarize_distribution, interval_union_length, process_metrics


class ReportingTests(unittest.TestCase):
    def test_post_failure_success_is_not_a_successful_prefix(self):
        stages = [{'goat_sub-task_success': s, 'goat_sub-task_path_length_m': p,
                   'goat_sub-task_shortest_path_length_m': d}
                  for s, p, d in [(0, 10., 5.), (1, 20., 10.), (1, 5., 5.)]]
        result = prefix_metrics(stages)
        self.assertEqual(result['PR'], 2 / 3)
        self.assertEqual(result['PPL'], 0.)
        stages[0]['goat_sub-task_success'] = 1
        stages[1]['goat_sub-task_success'] = 0
        result = prefix_metrics(stages)
        self.assertAlmostEqual(result['PPL'], 1 / 3)
        self.assertAlmostEqual(result['PPL_prefix'], 1 / 6)

    def test_conditional_denominators_and_undefined_tail(self):
        result = conditional_rates([[1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0]])
        self.assertEqual(result['conditioned_episodes'], [4, 3, 2])
        self.assertEqual(result['successes'], [3, 2, 1])
        self.assertEqual(result['Joint'], .25)
        self.assertEqual(conditional_rates([[0, 1], [0, 0]])['Joint'], 0.)

    def test_two_by_two_effects_and_paired_discordance(self):
        outcomes = {'nearest': [1, 0, 0, 0], 'suffix_only': [1, 1, 0, 0],
                    'myopic': [1, 0, 1, 0], 'full': [1, 1, 1, 1]}
        variants = {}
        for variant, vector in outcomes.items():
            episodes = [dict(scene_id='s', episode_id=str(i), episode_success=bool(v)) for i, v in enumerate(vector)]
            score = sum(vector) * 25.
            variants[variant] = (episodes, [], {'SR': score, 'SPL': score / 2})
        result = effects_and_pairs(variants)
        self.assertEqual(result['effects']['SR'], {'future_main_effect_pp': 37.5,
                         'belief_main_effect_pp': 37.5, 'interaction_pp': 25.})
        pair = result['paired_episode_success']['suffix_only_vs_myopic']
        self.assertEqual(pair, {'both_success': 1, 'left_only_success': 1,
                               'right_only_success': 1, 'both_failure': 1, 'n_paired': 4})
        variants['full'][0].pop()
        with self.assertRaises(ValueError):
            effects_and_pairs(variants)

    def test_latency_quantile(self):
        result = summarize_distribution([1., 2., 3., 4., 5.])
        self.assertEqual(result['median'], 3.)
        self.assertAlmostEqual(result['p95'], 4.8)

    def test_overlapping_visits_do_not_double_count_the_physical_path(self):
        self.assertEqual(interval_union_length([(0., 2.), (1., 3.), (5., 7.)]), 5.)
        self.assertEqual(interval_union_length([(0., 3.), (1., 2.)]), 3.)

    def test_invalid_confirmed_approach_tail_is_counted_even_at_budget_stop(self):
        event = {'event': 'verification_final', 'scene_id': 's', 'episode_id': '0',
                 'k': 0, 'candidate_id': '0:0', 'v': 1, 'y': 0,
                 'start_path_m': 0., 'verification_path_m': .25, 't': 2, 'last_t': 2}
        trace = [{'frame_index': i + 1, 'subtask_index': 0, 'is_stop': i == 2,
                  'stop_reason': 'max_steps' if i == 2 else None,
                  'world_pose': [0., 0., 0., 1., i * .25, 0., 0.]} for i in range(3)]
        result = process_metrics([event], [], [], {('s', '0'): trace}, {('s', '0')})
        self.assertEqual(result['invalid_visit_path_m'], .5)
        self.assertTrue(result['invalid_path_complete'])


if __name__ == '__main__':
    unittest.main()
