"""Run existing analysis/report checks with the pending module in this process only."""
import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('candidate_identities', HERE / 'proposed_candidate_identities.py')
module = importlib.util.module_from_spec(spec)
sys.modules['candidate_identities'] = module
spec.loader.exec_module(module)

names = ['test_candidate_identities', 'test_report_variants', 'test_evaluate_sequence',
         'experiments.test_blp_identity', 'test_report_full_evaluation', 'test_cp2_report_rule']
suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in names)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
