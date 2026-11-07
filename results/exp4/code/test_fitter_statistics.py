import unittest

from fit_calibration_reviewed import reviewed_decision_flip_rate
from pbsp.navigation.blp import BeliefParameters


class FitterStatisticsTests(unittest.TestCase):
    def test_empty_roots_are_counted_in_all_snapshot_denominator(self):
        nonempty = {'k': 0, 'layers': [[{'raw_rank': 0, 'E': 0., 'N_views': 12, 'location': 1},
                                      {'raw_rank': 1, 'E': 1., 'N_views': 12, 'location': 2}]],
                    'distances': [[0., 1., 3.], [1., 0., 4.], [3., 4., 0.]],
                    'nearest_frontier': [None] * 3, 'frontier_available': False,
                    'all_raw_ranks': [0, 1]}
        empty = {**nonempty, 'layers': [[]], 'all_raw_ranks': []}
        rows = [{'snapshot': s, 'active_raw_rank': None} for s in (nonempty, empty, nonempty, empty)]
        result = reviewed_decision_flip_rate(rows, BeliefParameters(a=20, b0=-10),
                                             BeliefParameters(a=1, b0=-.5))
        self.assertEqual(result['flips'], 2)
        self.assertEqual(result['all_snapshots'], 4)
        self.assertEqual(result['eligible_snapshots'], 2)
        self.assertEqual(result['rate'], .5)
        self.assertEqual(result['eligible_rate'], 1.)


if __name__ == '__main__':
    unittest.main()
