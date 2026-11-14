"""Bind the single tested passive logger change to the original calibration.

This never rewrites the calibration ledger or grants parameter approval.
"""
import hashlib
import json
from pathlib import Path
import re

LOGGER_COMMIT = '03242fda76fd6a631596a7269c8e7990c657ea30'
OBSERVER = 'experiments/blp_observer.py'
IDENTITY_TEST = 'experiments/test_blp_identity.py'
BASE_OBSERVER_SHA = 'f3587f64f88b5470f6917258c35c0924eb77dd4f8de22ba57de962b4467dddf1'
EVAL_OBSERVER_SHA = '12dd680b3f4d102fd70277a193d1c85ad5deb219dfb04846e3060df9304af9fb'
IDENTITY_TEST_SHA = 'e7b6de94d6ed893ef53a4930f2f7ddd8a88c788abd7735ecfc2e88d1b9237d7b'
TEST_LOG = 'code/evaluation-runtime-tests.txt'
REGRESSION = 'diagnostics/evaluation-runtime-regression.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def changed_sources(calibration, evaluation):
    return {name for name in set(calibration) | set(evaluation)
            if calibration.get(name) != evaluation.get(name)}


def validate_source_change(calibration, evaluation):
    require(len(calibration) == 160 and len(evaluation) == 161,
            'Expected the preserved 160-input calibration and 161-input logging validation')
    require(calibration.get(OBSERVER) == BASE_OBSERVER_SHA and IDENTITY_TEST not in calibration,
            'Calibration is not the original corrected observer')
    require(changed_sources(calibration, evaluation) == {OBSERVER, IDENTITY_TEST},
            'Only the reviewed observer and identity test may differ')
    require(evaluation.get(OBSERVER) == EVAL_OBSERVER_SHA and
            evaluation.get(IDENTITY_TEST) == IDENTITY_TEST_SHA,
            'The passive logging change differs from the tested source')


def validate_results(out):
    out = Path(out)
    log = (out / TEST_LOG).read_text()
    require(bool(re.search(r'Ran 60 tests in ', log)) and log.rstrip().endswith('OK'),
            'The final evaluation source needs sixty passing runtime checks')
    regression = json.loads((out / REGRESSION).read_text())
    require(regression['all_decisions_identical'] and regression['coverage_complete'] and
            regression['calls_per_episode'] == {'0.pkl.gz': 33, '1.pkl.gz': 85, '2.pkl.gz': 101},
            'The final evaluation source needs all 219 matched-input baseline calls')
    return regression


def validate_runtime_binding(out, cp1, ledger, evaluation):
    out = Path(out)
    require(cp1.get('accepted') is True and cp1.get('collection_attempt') == 'a3' and
            cp1['source_fingerprint'] == ledger['source_fingerprint'],
            'Keep the original corrected CP1 and calibration source binding')
    validate_source_change(ledger['source_fingerprint'], evaluation)
    proof = json.loads((out / 'meta/evaluation-runtime-validation.json').read_text())
    require(proof.get('validated') is True and proof.get('collection_attempt') == 'a3' and
            proof.get('logger_commit') == LOGGER_COMMIT,
            'A completed validation of the passive evaluation logger is required')
    require(proof.get('calibration_source_fingerprint') == ledger['source_fingerprint'] and
            proof.get('evaluation_source_fingerprint') == evaluation,
            'Runtime validation belongs to different source inputs')
    require(proof.get('unit_tests') == 60 and proof.get('disabled_regression_calls') == 219,
            'Runtime validation coverage is incomplete')
    evidence = proof['evidence_sha256']
    required = {TEST_LOG, REGRESSION, 'meta/CP1_acceptance.json', 'meta/calibration_runs.json',
                'meta/CP2a_review.json', 'meta/calibration.json', 'meta/calibration-fit-binding.json',
                'code/evaluation_runtime_binding.py', 'code/validate_evaluation_runtime.py'}
    require(required <= set(evidence), 'Runtime validation is missing required evidence bindings')
    for relative, expected in evidence.items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                'Runtime validation evidence must remain inside the experiment directory')
        require(digest(out / relative) == expected, 'Changed runtime validation evidence: ' + relative)
    validate_results(out)
    return proof

