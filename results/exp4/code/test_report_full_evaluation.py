"""Isolated synthetic publication fixtures; never run simulation or fit data."""
import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import report_full_evaluation as report
import report_variants as analysis
from evaluate_sequence import digest, GateError
from pbsp.navigation.blp_parameters import CalibrationDocument


def save(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    """Only formal-shape test data under TemporaryDirectory, never actual results."""
    fingerprint = report.source_fingerprint()
    save(root, 'meta/calibration.json', {'frozen': True, 'selection': 'pooled',
         'label_source': 'system_verification', 'parameters': {'a': 10., 'b0': -2.5, 'pi': .3, 'c_exp': 10.}})
    parameter_sha = digest(root / 'meta/calibration.json')
    save(root, 'meta/CP2b_acceptance.json', {'accepted': True, 'source_fingerprint': fingerprint,
                                          'parameter_sha256': parameter_sha})
    identities = [{'scene_id': f'synthetic-scene-{s}', 'runtime_episode_id': str(e),
                   'ordered_categories': ['chair', 'bed', 'plant'],
                   'source_episode_ids': [f'{s}-{e}-{k}' for k in range(3)]}
                  for s in range(20) for e in range(20)]
    save(root, 'meta/3mon-identities.json', identities)
    document = CalibrationDocument.from_mapping(json.loads((root / 'meta/calibration.json').read_text()))
    binding = {'selection': 'pooled', 'parameter_sha256': parameter_sha,
               'parameters_by_scene': {r['scene_id']: asdict(document.pooled) for r in identities}}
    versions = {'report_variants.py': analysis.SOURCE_SHA256, 'action_path_metrics.py': analysis.ACTION_SOURCE_SHA256,
                'native_candidate_gt.py': analysis.GT_SOURCE_SHA256, 'blp_parameters.py': analysis.PARAMETER_SOURCE_SHA256}
    variants = {}
    for name, successes_per_scene in zip(report.VARIANTS, (4, 6, 5, 7)):
        records = []
        for identity in identities:
            success = int(identity['runtime_episode_id']) < successes_per_scene
            records.append({'scene_id': identity['scene_id'], 'episode_id': identity['runtime_episode_id'],
                            'goals': identity['ordered_categories'], 'source_episode_ids': identity['source_episode_ids'],
                            'stages': [{'k': k + 1, 'success': k < 2 or success} for k in range(3)],
                            'episode_success': success})
        p = successes_per_scene / 20
        precise = {'SR': 100 * p, 'SPL': 60 * p, 'PR': 100 * (2 + p) / 3,
                   'PPL': 100 * (.4 + .2 * p), 'P_C1': 100., 'P_C2_given_C1': 100.,
                   'P_C3_given_C1C2': 100 * p, 'P_C4_given_C1C2C3': None, 'Joint': 100 * p}
        process = {'invalid_path_complete': True, 'invalid_path_association_gaps': [],
                   'invalid_path_source': 'blp_action_target_v1_native_gt', 'native_stage_path_check_count': 1200,
                   'native_stage_path_max_abs_error_m': 0., 'Invalid_visits_per_ep': 0.,
                   'invalid_visit_path_m_per_ep': 0., 'invalid_identity_pursuit_path_m_per_ep': 0.,
                   'deviation_from_nearest_proposal_pct': 0., 'deviation_from_nearest_after_hysteresis_pct': 0.,
                   'deviation_from_nearest_realized_pct': 0., 'planning_ms': {'median': 1., 'p95': 2.},
                   'fallback_root_decisions': {}, 'fallback_decision_incidence': {}, 'fallback_memo_state_counts': {},
                   'eligible_realized_dispatches': 400}
        summary = analysis.format_summary({'method': 'ForesightNav', 'variant': name, 'task': '3-MON',
                  'n_episodes': 400, 'n_error': 0, 'diagnostic_only': False, **precise,
                  'process': process, 'calibration_binding': binding, 'analysis_sources_sha256': versions,
                  'conditional_counts': {'conditioned_episodes': [400, 400, 400],
                                         'successes': [400, 400, int(400 * p)], 'rates': [1., 1., p]},
                  'source_commits': ['synthetic-test-only']})
        save(root, f'summary/{name}.json', summary)
        (root / 'logs').mkdir(exist_ok=True)
        (root / f'logs/{name}_episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        (root / f'logs/{name}_decisions.jsonl').write_text('')
        jobs = []
        for s in range(20):
            subset = identities[s * 20:(s + 1) * 20]
            run = root / 'synthetic-native' / name / f'scene-{s}'
            save(root, str(run.relative_to(root) / 'status.json'), {'status': 'completed'})
            save(root, str(run.relative_to(root) / 'launch.json'), {'identities': subset})
            save(root, str(run.relative_to(root) / 'mon_metrics.json'), {'counts': {'completed_episodes': 20, 'targets_per_episode': 3}})
            jobs.append({'status': 'completed', 'scene': subset[0]['scene_id'],
                         'identities': subset, 'run_directory': str(run)})
        artifacts = {relative: digest(root / relative) for relative in
                     (f'summary/{name}.json', f'logs/{name}_episodes.jsonl', f'logs/{name}_decisions.jsonl')}
        save(root, f'meta/evaluation-{name}-runs.json', {'status': 'completed', 'n_episodes': 400,
             'source_fingerprint': fingerprint, 'parameter_sha256': parameter_sha,
             'jobs': jobs, 'artifact_sha256': artifacts})
        variants[name] = records, [], summary
    analysis.export_comparison(root, variants)
    (root / 'reports').mkdir()
    (root / 'code').mkdir()
    shutil.copy2(Path(report.__file__), root / 'code/report_full_evaluation.py')


class FullPublicationTests(unittest.TestCase):
    def test_complete_fixture_renders_but_never_authorizes_or_reports_in_chat(self):
        with TemporaryDirectory(prefix='exp4-report-synthetic-only-') as folder:
            root = Path(folder)
            fixture(root)
            with patch.object(report, 'ROOT', root):
                for stage in ('CP4', 'CP5'):
                    proof = report.render(stage, *report.verify(stage))
                    self.assertFalse(proof['reported_in_chat'])
                    self.assertFalse(proof['authorizes_next_stage'])
                    self.assertFalse(proof['launches_workers'])
                    self.assertTrue((root / f'reports/{stage}.md').exists())
                self.assertFalse((root / 'meta/CP4_reported.json').exists())
                self.assertTrue((root / 'figures/CP5/SR_SPL.png').exists())
                self.assertTrue((root / 'figures/CP5/effects.png').exists())

    def test_partial_coverage_is_rejected_even_if_hashes_are_updated(self):
        with TemporaryDirectory(prefix='exp4-report-synthetic-only-') as folder:
            root = Path(folder)
            fixture(root)
            relative = 'logs/full_episodes.jsonl'
            path = root / relative
            path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
            ledger = json.loads((root / 'meta/evaluation-full-runs.json').read_text())
            ledger['artifact_sha256'][relative] = digest(path)
            save(root, 'meta/evaluation-full-runs.json', ledger)
            with patch.object(report, 'ROOT', root), self.assertRaisesRegex(GateError, 'fixed identities'):
                report.verify('CP4')

    def test_stale_effects_are_rejected(self):
        with TemporaryDirectory(prefix='exp4-report-synthetic-only-') as folder:
            root = Path(folder)
            fixture(root)
            value = json.loads((root / 'summary/effects.json').read_text())
            value['effects']['SR']['future_main_effect_pp'] += 1
            save(root, 'summary/effects.json', value)
            with patch.object(report, 'ROOT', root), self.assertRaisesRegex(ValueError, 'Canonical effects'):
                report.verify('CP5')

    def test_stale_csv_scores_are_rejected_with_correct_variant_names(self):
        with TemporaryDirectory(prefix='exp4-report-synthetic-only-') as folder:
            root = Path(folder)
            fixture(root)
            path = root / 'summary/all.csv'
            with path.open(newline='') as handle:
                reader = csv.DictReader(handle)
                values, fields = list(reader), reader.fieldnames
            for key in ('SR', 'Invalid_visits_per_ep'):
                with self.subTest(column=key):
                    original = values[0][key]
                    values[0][key] = '99.00'
                    with path.open('w', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(values)
                    with patch.object(report, 'ROOT', root), self.assertRaisesRegex(ValueError, 'stale ' + key):
                        report.verify('CP5')
                    values[0][key] = original


if __name__ == '__main__':
    unittest.main()
