import copy
import unittest

import proposed_candidate_identities as proposed
import base_candidate_identities as base


def event(kind, cid, point, **extra):
    return dict(event=kind, scene_id='s', episode_id='2', k=0,
                candidate_id=cid, position=point, **extra)


class UncheckedAliasTests(unittest.TestCase):
    def test_exact_unchecked_alias_rewrites_trace_without_changing_fit_rows(self):
        rows = [event('candidate_first_appearance', '0:0', [1., 0., 2.]),
                event('candidate_first_appearance', '0:4', [1., 0., 2.]),
                event('verification_final', '0:4', [1., 0., 2.], y=0, v=1, E=.2),
                dict(event='empty_activation', scene_id='s', episode_id='2', k=0,
                     path_to_first_candidate_m=1.25)]
        trace = {('s', '2'): [dict(scene_id='s', episode_id='2', blp_action_target={
            'task_idx': 0, 'candidate_id': '0:0', 'query_identity_status': 'resolved'})]}
        original = copy.deepcopy((rows, trace))
        before, _, stats0 = base.canonicalize(rows, trace)
        after, rewritten, stats = proposed.canonicalize(rows, trace)
        relevant = lambda values: [r for r in values if r['event'] in ('verification_final', 'empty_activation')]
        self.assertEqual(relevant(before), relevant(after))
        self.assertEqual(rewritten['s', '2'][0]['blp_action_target']['candidate_id'], '0:4')
        self.assertEqual(stats['unchecked_first_appearance_aliases'], 1)
        self.assertEqual(stats0['canonical_checked_identities'], stats['canonical_checked_identities'])
        self.assertEqual((rows, trace), original)

    def test_scope_height_and_exact_coordinates_are_not_relaxed(self):
        checked = event('verification_final', '0:4', [1., 0., 2.], y=0, v=0)
        for change in [dict(k=1), dict(episode_id='3'), dict(scene_id='other'),
                       dict(_resample_instance=2), dict(position=[1., .01, 2.]),
                       dict(position=[1.00000001, 0., 2.])]:
            with self.subTest(change=change):
                first = event('candidate_first_appearance', '0:0', [1., 0., 2.])
                first.update(change)
                output, _, stats = proposed.canonicalize([first, checked])
                self.assertEqual(output[0]['candidate_id'], '0:0')
                self.assertEqual(stats['unchecked_first_appearance_aliases'], 0)

    def test_checked_identity_is_not_merged_by_its_first_appearance(self):
        rows = [event('candidate_first_appearance', '0:0', [1., 0., 2.]),
                event('verification_final', '0:0', [3., 0., 2.], y=0, v=0),
                event('verification_final', '0:4', [1., 0., 2.], y=1, v=1)]
        old, _, a = base.canonicalize(rows)
        new, _, b = proposed.canonicalize(rows)
        self.assertEqual(old, new)
        self.assertEqual(a['canonical_checked_identities'], b['canonical_checked_identities'])

    def test_labels_do_not_choose_aliases_and_checked_dedup_still_applies(self):
        for value in [0, 1]:
            rows = [event('candidate_first_appearance', '0:0', [1., 0., 2.]),
                    event('verification_final', '0:4', [1., 0., 2.], y=value, v=1-value),
                    event('verification_final', '0:5', [1., 0., 2.], y=1-value, v=value)]
            out, _, stats = proposed.canonicalize(rows)
            self.assertEqual([x['candidate_id'] for x in out], ['0:4'] * 3)
            self.assertEqual(stats['raw_checked_registry_identities'], 2)
            self.assertEqual(stats['canonical_checked_identities'], 1)

    def test_unresolved_null_is_not_filled(self):
        events = [event('candidate_first_appearance', '0:0', [1., 0., 2.]),
                  event('verification_final', '0:4', [1., 0., 2.], y=0, v=0)]
        trace = {('s', '2'): [dict(scene_id='s', episode_id='2', blp_action_target={
            'task_idx': 0, 'candidate_id': None, 'query_identity_status': 'unresolved'})]}
        _, out, _ = proposed.canonicalize(events, trace)
        self.assertIsNone(out['s', '2'][0]['blp_action_target']['candidate_id'])
        self.assertEqual(out['s', '2'][0]['blp_action_target']['query_identity_status'], 'unresolved')

    def test_bootstrap_instances_stay_separate(self):
        rows = []
        for instance in [1, 2]:
            rows.extend([event('candidate_first_appearance', '0:0', [1., 0., 2.], _resample_instance=instance),
                         event('verification_final', '0:4', [1., 0., 2.], y=0, v=0, _resample_instance=instance)])
        _, _, stats = proposed.canonicalize(rows)
        self.assertEqual(stats['canonical_checked_identities'], 2)
        self.assertEqual(stats['unchecked_first_appearance_aliases'], 2)


if __name__ == '__main__':
    unittest.main()
