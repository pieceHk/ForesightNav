"""Validate an already adopted passive logger after approved calibration.

No source patching, parameter fitting/freezing, or native workers occur here.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from evaluate_sequence import validate_calibration_coverage, GateError
from evaluation_runtime_binding import (LOGGER_COMMIT, TEST_LOG, REGRESSION, digest,
    require, validate_results, validate_source_change, validate_runtime_binding)
from experiments.exp4_collect import source_fingerprint, write
from pbsp.navigation.blp_parameters import CalibrationDocument

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'results/exp4'
PROOF = 'meta/evaluation-runtime-validation.json'


def read(relative):
    return json.loads((ROOT / relative).read_text())


def preconditions():
    ledger = read('meta/calibration_runs.json')
    validate_calibration_coverage(ROOT, ledger)
    require(len(ledger['jobs']) == 12, 'This validation is for all three fixed calibration folds')
    post = read('meta/calibration-postprocess.json')
    require(post['status'] == 'completed_pending_CP2a_review' and
            post['source_fingerprint'] == ledger['source_fingerprint'],
            'Complete the bound CPU calibration analysis before changing evaluation sources')
    require(not list((ROOT / 'meta').glob('evaluation-*-runs.json')) and
            not (ROOT / 'meta/CP2b_acceptance.json').exists(),
            'Runtime validation must precede all evaluation jobs and CP2b binding')
    review = read('meta/CP2a_review.json')
    require(review.get('accepted') is True and review.get('collection_attempt') == 'a3',
            'An accepted current CP2a proposal is required')
    document = CalibrationDocument.from_mapping(read('meta/calibration.json'))
    choice = 'three_fold_cross_fit' if document.selection == 'cross_fit' else 'three_fold_pooled'
    require(review.get('chosen_choice') == choice, 'Reviewed and frozen parameter choices differ')
    proposal = ('meta/calibration.crossfit.proposed.json' if document.selection == 'cross_fit'
                else 'meta/calibration.pooled.review-proposed.json')
    require(read('meta/calibration.json').get('approved_proposal_sha256') == digest(ROOT / proposal),
            'The frozen file is not bound to the accepted proposal')
    for relative, expected in review['evidence_sha256'].items():
        require(digest(ROOT / relative) == expected, 'Reviewed evidence changed: ' + relative)
    fit = read('meta/calibration-fit-binding.json')
    require(fit['collection_attempt'] == 'a3', 'Fit binding belongs to an older collection')
    for relative, expected in {**fit['input_sha256'], **fit['output_sha256']}.items():
        require(digest(ROOT / relative) == expected, 'Fitted evidence changed: ' + relative)
    cp1 = read('meta/CP1_acceptance.json')
    require(cp1.get('accepted') is True and cp1['source_fingerprint'] == ledger['source_fingerprint'],
            'Original corrected calibration source binding differs')
    evaluation = source_fingerprint()
    validate_source_change(ledger['source_fingerprint'], evaluation)
    return cp1, ledger, evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        cp1, ledger, fingerprint = preconditions()
    except (ValueError, FileNotFoundError, KeyError, GateError) as error:
        if args.check:
            print(json.dumps({'ready': False, 'reason': str(error), 'launches_workers': False,
                              'patches_sources': False, 'freezes_parameters': False}))
            return
        raise
    if args.check:
        print(json.dumps({'ready': True, 'unit_tests': 60, 'disabled_regression_calls': 219,
                          'launches_workers': False, 'patches_sources': False, 'freezes_parameters': False}))
        return
    require(not (ROOT / PROOF).exists(), 'Inspect existing runtime validation before repeating checks')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    environment = dict(os.environ, PYTHONPATH=str(REPO) + ':' + str(ROOT / 'code'),
                       CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1')
    test_log = ROOT / ('logs/evaluation-runtime-' + stamp + '-tests.log')
    modules = ['experiments.test_blp', 'experiments.test_blp_parameters', 'experiments.test_blp_depth',
               'experiments.test_blp_identity', 'test_report_variants', 'test_evaluate_sequence']
    with test_log.open('x') as output:
        result = subprocess.run([sys.executable, '-m', 'unittest', '-v', *modules],
                                cwd=REPO, env=environment, stdout=output, stderr=subprocess.STDOUT)
    require(result.returncode == 0, 'Runtime checks failed; preserve log ' + str(test_log))
    regression_log = ROOT / ('logs/evaluation-runtime-' + stamp + '-regression.log')
    regression_stderr = regression_log.with_suffix('.stderr.log')
    with regression_log.open('x') as output, regression_stderr.open('x') as errors:
        result = subprocess.run([sys.executable, '-m', 'experiments.exp4_regression', '--require-complete'],
                                cwd=REPO, env=environment, stdout=output, stderr=errors)
    require(result.returncode == 0, 'Runtime regression failed; preserve log ' + str(regression_log))
    shutil.copy2(test_log, ROOT / TEST_LOG)
    regression = json.loads(regression_log.read_text())
    write(ROOT / REGRESSION, regression)
    validate_results(ROOT)
    require(source_fingerprint() == fingerprint, 'Runtime inputs changed during validation')
    evidence = [TEST_LOG, REGRESSION, 'meta/CP1_acceptance.json', 'meta/calibration_runs.json',
                'meta/CP2a_review.json', 'meta/calibration.json', 'meta/calibration-fit-binding.json',
                'code/evaluation_runtime_binding.py', 'code/validate_evaluation_runtime.py']
    proof = {'validated': True, 'collection_attempt': 'a3',
             'validated_at_utc': datetime.now(timezone.utc).isoformat(), 'logger_commit': LOGGER_COMMIT,
             'calibration_source_fingerprint': ledger['source_fingerprint'],
             'evaluation_source_fingerprint': fingerprint, 'unit_tests': 60,
             'disabled_regression_calls': 219,
             'source_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
             'evidence_sha256': {relative: digest(ROOT / relative) for relative in evidence},
             'test_execution_log': str(test_log.relative_to(ROOT)),
             'regression_execution_log': str(regression_log.relative_to(ROOT)),
             'native_workers_launched': False, 'parameters_changed': False, 'raw_calibration_changed': False}
    write(ROOT / PROOF, proof)
    validate_runtime_binding(ROOT, cp1, ledger, fingerprint)
    print(json.dumps({'validated': True, 'runtime_inputs': len(fingerprint),
                      'original_calibration_inputs': len(ledger['source_fingerprint']),
                      'unit_tests': 60, 'disabled_regression_calls': 219,
                      'launches_workers': False, 'freezes_parameters': False}, indent=2))


if __name__ == '__main__':
    main()

