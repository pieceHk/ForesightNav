"""Synthetic source-binding cases; no actual checkpoint files or workers."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import evaluation_runtime_binding as binding


class RuntimeBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix='exp4_runtime_binding_synthetic_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = {f'fixture/{i}.py': 'a' * 64 for i in range(159)}
        self.base[binding.OBSERVER] = binding.BASE_OBSERVER_SHA
        self.evaluation = dict(self.base, **{
            binding.OBSERVER: binding.EVAL_OBSERVER_SHA,
            binding.IDENTITY_TEST: binding.IDENTITY_TEST_SHA})
        self.cp1 = {'accepted': True, 'collection_attempt': 'a3', 'source_fingerprint': self.base}
        self.ledger = {'source_fingerprint': self.base}
        paths = [binding.TEST_LOG, binding.REGRESSION, 'meta/CP1_acceptance.json',
                 'meta/calibration_runs.json', 'meta/CP2a_review.json', 'meta/calibration.json',
                 'meta/calibration-fit-binding.json', 'code/evaluation_runtime_binding.py',
                 'code/validate_evaluation_runtime.py']
        for relative in paths:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
        (self.root / binding.TEST_LOG).write_text('Synthetic fixture only\nRan 60 tests in 0.5s\n\nOK\n')
        self.regression = {'all_decisions_identical': True, 'coverage_complete': True,
                           'calls_per_episode': {'0.pkl.gz': 33, '1.pkl.gz': 85, '2.pkl.gz': 101}}
        (self.root / binding.REGRESSION).write_text(json.dumps(self.regression))
        self.proof = {'validated': True, 'collection_attempt': 'a3',
                      'logger_commit': binding.LOGGER_COMMIT,
                      'calibration_source_fingerprint': self.base,
                      'evaluation_source_fingerprint': self.evaluation,
                      'unit_tests': 60, 'disabled_regression_calls': 219,
                      'evidence_sha256': {p: binding.digest(self.root / p) for p in paths}}
        self.save()

    def save(self):
        path = self.root / 'meta/evaluation-runtime-validation.json'
        path.write_text(json.dumps(self.proof))

    def check(self):
        return binding.validate_runtime_binding(self.root, self.cp1, self.ledger, self.evaluation)

    def test_complete_exact_change_is_validated(self):
        self.assertEqual(self.check()['unit_tests'], 60)
        self.assertEqual(self.ledger['source_fingerprint'], self.base)

    def test_unchanged_observer_cannot_skip_logging_fix(self):
        with self.assertRaises(ValueError):
            binding.validate_source_change(self.base, self.base)

    def test_other_runtime_change_is_rejected(self):
        self.evaluation['fixture/0.py'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'Only the reviewed'):
            self.check()

    def test_different_observer_bytes_are_rejected(self):
        self.evaluation[binding.OBSERVER] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'differs from the tested'):
            self.check()

    def test_altered_calibration_source_is_rejected(self):
        self.cp1['source_fingerprint'] = dict(self.base)
        self.cp1['source_fingerprint']['fixture/0.py'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'original corrected'):
            self.check()

    def test_missing_validation_does_not_grant_binding(self):
        self.proof['validated'] = False
        self.save()
        with self.assertRaisesRegex(ValueError, 'completed validation'):
            self.check()

    def test_stale_parameter_hash_is_rejected(self):
        (self.root / 'meta/calibration.json').write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, 'Changed runtime validation evidence'):
            self.check()

    def test_missing_evidence_binding_is_rejected(self):
        del self.proof['evidence_sha256'][binding.REGRESSION]
        self.save()
        with self.assertRaisesRegex(ValueError, 'missing required evidence'):
            self.check()

    def test_fewer_checks_rejected_even_with_updated_digest(self):
        (self.root / binding.TEST_LOG).write_text('Ran 59 tests in 0.5s\n\nOK\n')
        self.proof['evidence_sha256'][binding.TEST_LOG] = binding.digest(self.root / binding.TEST_LOG)
        self.save()
        with self.assertRaisesRegex(ValueError, 'sixty passing'):
            self.check()

    def test_same_total_wrong_baseline_coverage_rejected(self):
        self.regression['calls_per_episode'] = {'0.pkl.gz': 34, '1.pkl.gz': 84, '2.pkl.gz': 101}
        (self.root / binding.REGRESSION).write_text(json.dumps(self.regression))
        self.proof['evidence_sha256'][binding.REGRESSION] = binding.digest(self.root / binding.REGRESSION)
        self.save()
        with self.assertRaisesRegex(ValueError, 'all 219'):
            self.check()

    def test_source_fingerprint_proof_cannot_be_reused(self):
        self.proof['evaluation_source_fingerprint'] = dict(self.evaluation)
        self.proof['evaluation_source_fingerprint'][binding.OBSERVER] = 'b' * 64
        self.save()
        with self.assertRaisesRegex(ValueError, 'different source inputs'):
            self.check()

    def test_evidence_cannot_leave_experiment_directory(self):
        self.proof['evidence_sha256']['../outside'] = 'a' * 64
        self.save()
        with self.assertRaisesRegex(ValueError, 'inside the experiment directory'):
            self.check()


if __name__ == '__main__':
    unittest.main()

