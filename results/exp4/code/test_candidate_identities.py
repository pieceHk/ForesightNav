"""Exact-position identity grouping is label blind and retains every check."""
import copy
from dataclasses import asdict
import math
import unittest

from candidate_identities import canonicalize
from experiments.blp_calibration import fit as original_fit
from fit_calibration_reviewed import robust_fit


def row(cid, position, **extra):
    return dict(event='verification_final', scene_id='s', episode_id='0', k=0,
                candidate_id=cid, position=position, E=.2, v=0, y=0, **extra)


class CanonicalIdentityTests(unittest.TestCase):
    def test_exact_duplicate_merges_without_dropping_checks(self):
        events = [row('0:1', [1., 2., 3.]), row('0:0', [1., 2., 3.])]
        before = copy.deepcopy(events)
        output, _, stats = canonicalize(events)
        self.assertEqual(len(output), 2)
        self.assertEqual([r['candidate_id'] for r in output], ['0:0', '0:0'])
        self.assertEqual(stats['identities_merged'], 1)
        self.assertEqual(events, before)
        self.assertEqual(canonicalize(output)[0], output)

    def test_near_or_different_height_coordinates_do_not_merge(self):
        _, _, stats = canonicalize([row('a', [0., 0., 0.]), row('b', [1e-8, 0., 0.]),
                                   row('c', [0., 1., 0.])])
        self.assertEqual(stats['identities_merged'], 0)

    def test_scene_episode_task_and_bootstrap_instance_are_separate(self):
        rows = [row('a', [0., 0., 0.])]
        for field, value in [('scene_id', 'other'), ('episode_id', '1'), ('k', 1), ('_resample_instance', 0)]:
            item = row('b', [0., 0., 0.])
            item[field] = value
            rows.append(item)
        _, _, stats = canonicalize(rows)
        self.assertEqual(stats['identities_merged'], 0)

    def test_same_recorded_identity_connects_its_time_varying_points(self):
        rows = [row('a', [0., 0., 0.]), row('a', [1., 0., 0.]),
                row('b', [1., 0., 0.]), row('c', [0., 0., 0.])]
        output, _, stats = canonicalize(list(reversed(rows)))
        self.assertEqual({r['candidate_id'] for r in output}, {'a'})
        self.assertEqual(stats['identities_merged'], 2)

    def test_labels_do_not_choose_grouping(self):
        rows = [row('b', [0., 0., 0.]), row('a', [0., 0., 0.])]
        a = canonicalize(rows)
        rows[0].update(v=1, y=1)
        rows[1].update(v=0, y=1)
        b = canonicalize(rows)
        self.assertEqual([r['candidate_id'] for r in a[0]], [r['candidate_id'] for r in b[0]])
        self.assertEqual(a[2], b[2])

    def test_action_targets_follow_same_mapping_but_unresolved_stays_missing(self):
        events = [row('a', [0., 0., 0.]), row('b', [0., 0., 0.])]
        traces = {('s', '0'): [
            {'scene_id': 's', 'episode_id': '0', 'blp_action_target': {'task_idx': 0, 'candidate_id': 'b', 'query_identity_status': 'resolved'}},
            {'scene_id': 's', 'episode_id': '0', 'blp_action_target': {'task_idx': 0, 'candidate_id': None, 'query_identity_status': 'unresolved'}}]}
        before = copy.deepcopy(traces)
        _, result, _ = canonicalize(events, traces)
        self.assertEqual(result['s', '0'][0]['blp_action_target']['candidate_id'], 'a')
        self.assertIsNone(result['s', '0'][1]['blp_action_target']['candidate_id'])
        self.assertEqual(traces, before)

    def test_bad_coordinate_and_missing_identity_rejected(self):
        for position in (None, [0., 0.], [math.nan, 0., 0.], [0., math.inf, 0.]):
            with self.assertRaises(ValueError):
                canonicalize([row('a', position)])
        with self.assertRaises(ValueError):
            canonicalize([row(None, [0., 0., 0.])])

    def test_prior_corrected_with_identical_logistic_and_exploration_inputs(self):
        pairs = [(.1, 0), (.2, 0), (.25, 1), (.3, 0), (.35, 1), (.4, 1)]
        rows = [row(str(i), [float(i), 0., 0.]) for i in range(6)]
        for item, (e, v) in zip(rows, pairs):
            item.update(E=e, v=v, y=1-v)
        rows.append(dict(rows[0], candidate_id='duplicate'))
        rows.append(dict(event='empty_activation', path_to_first_candidate_m=3.))
        old = original_fit(rows)
        new = robust_fit(rows)
        self.assertEqual((old.a, old.b0, old.c_exp), (new.a, new.b0, new.c_exp))
        self.assertEqual(old.pi, 3/7)
        self.assertEqual(new.pi, .5)
        changed_gt = [dict(r, y=0) if r['event']=='verification_final' else dict(r) for r in rows]
        self.assertEqual(asdict(robust_fit(changed_gt)), asdict(new))

    def test_bootstrap_copies_keep_their_own_labels(self):
        rows = [dict(row('a', [0., 0., 0.]), _resample_instance=0),
                dict(row('b', [0., 0., 0.]), _resample_instance=0),
                dict(row('a', [0., 0., 0.]), _resample_instance=1)]
        output, _, stats = canonicalize(rows)
        self.assertEqual(stats['canonical_checked_identities'], 2)
        self.assertEqual(stats['identities_merged'], 1)
        self.assertEqual(len(output), 3)


if __name__ == '__main__':
    unittest.main()

