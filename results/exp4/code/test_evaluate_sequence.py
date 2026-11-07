"""Checkpoint and assignment tests; fixtures never start simulation workers."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluate_sequence import build_jobs, validate_gate, GateError, digest


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    fp = {'frozen_policy.py': 'fixed_source_hash'}
    groups = [[f'synthetic_{f}_{i}' for i in range(4)] for f in range(5)]
    save(root/'meta/folds.json', {'folds': groups})
    jobs = []
    for f in range(3):
        for scene in groups[f]:
            run = root/'native-calibration'/scene
            save(run/'status.json', {'status': 'completed'})
            save(run/'mon_metrics.json', {'counts': {'completed_episodes': 30, 'targets_per_episode': 2}})
            jobs.append({'fold': f, 'scene': scene, 'status': 'completed', 'run_directory': str(run)})
    cal = root/'meta/calibration_runs.json'
    save(cal, {'status': 'completed_three_folds_pending_CP2a', 'jobs': jobs})
    params = root/'meta/calibration.json'
    save(params, {'frozen': True, 'label_source': 'system_verification',
                  'parameters': {'a': 1., 'b0': -1., 'pi': .5, 'c_exp': 2.}})
    save(root/'meta/CP2b_acceptance.json', {'accepted': True, 'source_fingerprint': fp,
        'parameter_sha256': digest(params), 'calibration_ledger_sha256': digest(cal)})
    return fp


def smoke_proof(root, fp):
    identities, jobs, rows, decisions = [], [], [], []
    for i in range(20):
        identity = {'scene_id': f'eval_scene{i}', 'runtime_episode_id': '0',
                    'ordered_categories': ['chair', 'bed', 'plant'],
                    'source_episode_ids': [f'{i}a', f'{i}b', f'{i}c']}
        identities.append(identity)
        run = root/'native-smoke'/identity['scene_id']
        save(run/'status.json', {'status': 'completed'})
        save(run/'mon_metrics.json', {'counts': {'completed_episodes': 1, 'targets_per_episode': 3}})
        save(run/'launch.json', {'identities': [identity]})
        jobs.append({'status': 'completed', 'scene': identity['scene_id'],
                     'identities': [identity], 'run_directory': str(run)})
        rows.append({'scene_id': identity['scene_id'], 'episode_id': '0',
                     'goals': identity['ordered_categories'], 'source_episode_ids': identity['source_episode_ids'],
                     'stages': [{'k': k + 1} for k in range(3)]})
        decisions.append({'scene_id': identity['scene_id'], 'episode_id': '0', 'variant': 'full'})
    save(root/'meta/smoke-20-identities.json', identities)
    destination = root/'diagnostics/CP3'
    save(destination/'summary/full.json', {'n_episodes': 20, 'variant': 'full', 'diagnostic_only': False})
    (destination/'logs').mkdir()
    for name, values in [('episodes', rows), ('decisions', decisions)]:
        (destination/f'logs/full_{name}.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in values))
    artifacts = {relative: digest(destination/relative) for relative in
                 ('summary/full.json', 'logs/full_episodes.jsonl', 'logs/full_decisions.jsonl')}
    ledger = root/'meta/evaluation-smoke-runs.json'
    save(ledger, {'status': 'completed', 'n_episodes': 20, 'jobs': jobs,
                  'artifact_sha256': artifacts, 'source_fingerprint': fp,
                  'parameter_sha256': digest(root/'meta/calibration.json')})
    save(root/'meta/CP3_acceptance.json', {'accepted': True, 'behavior_reviewed': True,
         'source_fingerprint': fp, 'parameter_sha256': digest(root/'meta/calibration.json'),
         'smoke_ledger_sha256': digest(ledger)})


class EvaluationSequenceTests(unittest.TestCase):
    def test_assignments_are_twenty_scenes_and_exact_fixed_coverage(self):
        full = [{'scene_id': f'scene{s}', 'runtime_episode_id': str(e),
                 'ordered_categories': ['chair', 'bed', 'plant'], 'source_episode_ids': ['a', 'b', 'c']}
                for s in range(20) for e in range(20)]
        jobs = build_jobs('full', full)
        self.assertEqual(len(jobs), 20)
        self.assertEqual(sum(len(j['identities']) for j in jobs), 400)
        self.assertTrue(all(len(j['identities']) == 20 for j in jobs))
        smoke = full[::20]
        self.assertTrue(all(len(j['identities']) == 1 for j in build_jobs('smoke', smoke)))
        with self.assertRaises(GateError):
            build_jobs('full', full[:-1])
        with self.assertRaises(GateError):
            build_jobs('smoke', smoke[:-1] + [smoke[0]])

    def test_smoke_accepts_only_completed_source_bound_self_calibration(self):
        with TemporaryDirectory(prefix='exp4_gate_test_') as folder:
            root = Path(folder)
            fp = fixture(root)
            gate = validate_gate('smoke', root, lambda: fp)
            self.assertEqual(gate['parameters']['pi'], .5)
            with self.assertRaises(GateError):
                validate_gate('smoke', root, lambda: {'frozen_policy.py': 'changed'})
            save(root/'meta/calibration.json', {'frozen': False, 'label_source': 'system_verification'})
            with self.assertRaises(GateError):
                validate_gate('smoke', root, lambda: fp)

    def test_parameter_mutation_and_incomplete_calibration_are_rejected(self):
        with TemporaryDirectory(prefix='exp4_gate_test_') as folder:
            root = Path(folder)
            fp = fixture(root)
            value = json.loads((root/'meta/calibration.json').read_text())
            value['parameters']['pi'] = .9
            save(root/'meta/calibration.json', value)
            with self.assertRaises(GateError):
                validate_gate('smoke', root, lambda: fp)
            fp = fixture(root)
            ledger = json.loads((root/'meta/calibration_runs.json').read_text())
            ledger['status'] = 'running'
            save(root/'meta/calibration_runs.json', ledger)
            with self.assertRaises(GateError):
                validate_gate('smoke', root, lambda: fp)

    def test_full_is_not_authorized_by_CP2b_alone(self):
        with TemporaryDirectory(prefix='exp4_gate_test_') as folder:
            root = Path(folder)
            fp = fixture(root)
            with self.assertRaises(GateError):
                validate_gate('full', root, lambda: fp)

    def test_unfinished_calibration_scene_is_not_inferred_complete_from_ledger(self):
        with TemporaryDirectory(prefix='exp4_gate_test_') as folder:
            root = Path(folder)
            fp = fixture(root)
            ledger = json.loads((root/'meta/calibration_runs.json').read_text())
            run = Path(ledger['jobs'][0]['run_directory'])
            save(run/'status.json', {'status': 'running'})
            with self.assertRaises(GateError):
                validate_gate('smoke', root, lambda: fp)

    def test_full_accepts_only_current_native_and_normalized_smoke_evidence(self):
        with TemporaryDirectory(prefix='exp4_gate_test_') as folder:
            root = Path(folder)
            fp = fixture(root)
            smoke_proof(root, fp)
            self.assertEqual(validate_gate('full', root, lambda: fp)['parameters']['pi'], .5)
            path = root/'diagnostics/CP3/logs/full_episodes.jsonl'
            path.write_text(path.read_text().replace('"plant"', '"toilet"', 1))
            with self.assertRaises(GateError):
                validate_gate('full', root, lambda: fp)

    def test_missing_behavior_review_does_not_launch_full(self):
        with TemporaryDirectory(prefix='exp4_gate_test_') as folder:
            root = Path(folder)
            fp = fixture(root)
            smoke_proof(root, fp)
            path = root/'meta/CP3_acceptance.json'
            review = json.loads(path.read_text())
            review['behavior_reviewed'] = False
            save(path, review)
            with self.assertRaises(GateError):
                validate_gate('full', root, lambda: fp)

    def test_crossfit_file_cannot_silently_run_as_pooled_parameters(self):
        with TemporaryDirectory(prefix='exp4_gate_test_') as folder:
            root = Path(folder)
            fp = fixture(root)
            path = root/'meta/calibration.json'
            value = json.loads(path.read_text())
            value['selection'] = 'cross_fit'
            save(path, value)
            with self.assertRaisesRegex(GateError, 'Cross-fitting'):
                validate_gate('smoke', root, lambda: fp)


if __name__ == '__main__':
    unittest.main()
