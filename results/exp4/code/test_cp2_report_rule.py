"""Protocol edge cases for the report's independent parameter-selection check."""
from copy import deepcopy
from dataclasses import asdict
import unittest

from report_calibration import parameter_proposal, selection_rule
from pbsp.navigation.blp_parameters import CalibrationDocument


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


class ProposalLoadingTests(unittest.TestCase):
    def setUp(self):
        self.theta = {'a': 10., 'b0': -2.5, 'pi': .3, 'c_exp': 10.}
        self.folds = [[f'scene-{f}-{i}' for i in range(4)] for f in range(5)]
        self.fit = {'theta_pool': self.theta,
                    'cross_fit_proposals': {str(f): dict(self.theta, a=11. + f) for f in range(3)}}

    def test_pooled_proposal_loads_without_crossfit_fields(self):
        proposal = parameter_proposal(self.fit, self.folds, {}, 'pooled')
        self.assertIs(proposal['frozen'], False)
        frozen = dict(proposal, frozen=True, schema_version='exp4_calibration_v1')
        document = CalibrationDocument.from_mapping(frozen)
        for scenes in self.folds:
            for scene in scenes:
                self.assertEqual(asdict(document.resolve(scene)[0]), self.theta)

    def test_crossfit_proposal_preserves_all_twenty_runtime_assignments(self):
        proposal = parameter_proposal(self.fit, self.folds, {}, 'cross_fit')
        self.assertIs(proposal['frozen'], False)
        frozen = dict(proposal, frozen=True, schema_version='exp4_calibration_v1')
        document = CalibrationDocument.from_mapping(frozen)
        for f, scenes in enumerate(self.folds):
            for scene in scenes:
                theta, origin = document.resolve(scene)
                expected = self.fit['cross_fit_proposals'][str(f)] if f < 3 else self.theta
                self.assertEqual(asdict(theta), expected)
                self.assertEqual(origin['training_folds'], [x for x in range(3) if x != f])


if __name__ == '__main__':
    unittest.main()
