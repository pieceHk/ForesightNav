"""Protocol edge cases for the report's independent parameter-selection check."""
from copy import deepcopy
import unittest

from report_calibration import selection_rule


def fit_record():
    intervals = {k: [0., 2.] for k in ('a', 'b0', 'pi', 'c_exp')}
    return {'per_fold': {str(f): {'bootstrap': {'ci95': deepcopy(intervals)},
                                 'flip_vs_pool': {'rate': .1}} for f in range(3)},
            'all_fold_ci_overlap': {k: True for k in intervals},
            'stable_under_protocol_rule': True}


class SelectionRuleTests(unittest.TestCase):
    def test_ten_percent_and_touching_intervals_are_admissible(self):
        value = fit_record()
        value['per_fold']['2']['bootstrap']['ci95']['c_exp'] = [2., 3.]
        self.assertEqual(selection_rule(value)[0], 'pooled')

    def test_a_flip_rate_above_ten_percent_requires_crossfit(self):
        value = fit_record()
        value['per_fold']['1']['flip_vs_pool']['rate'] = .100001
        value['stable_under_protocol_rule'] = False
        self.assertEqual(selection_rule(value)[0], 'cross_fit')

    def test_disjoint_cost_intervals_require_crossfit_even_with_zero_flips(self):
        value = fit_record()
        for group in value['per_fold'].values():
            group['flip_vs_pool']['rate'] = 0.
        value['per_fold']['2']['bootstrap']['ci95']['c_exp'] = [2.001, 3.]
        value['all_fold_ci_overlap']['c_exp'] = False
        value['stable_under_protocol_rule'] = False
        self.assertEqual(selection_rule(value)[0], 'cross_fit')

    def test_a_stale_summary_flag_cannot_override_the_intervals(self):
        value = fit_record()
        value['per_fold']['2']['bootstrap']['ci95']['c_exp'] = [2.001, 3.]
        with self.assertRaisesRegex(ValueError, 'interval overlap'):
            selection_rule(value)

    def test_a_stale_selection_flag_cannot_override_the_actual_flips(self):
        value = fit_record()
        value['per_fold']['0']['flip_vs_pool']['rate'] = .11
        with self.assertRaisesRegex(ValueError, 'Selection differs'):
            selection_rule(value)


if __name__ == '__main__':
    unittest.main()
