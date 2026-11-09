"""CPU-only observer regression checks; no simulation or policy execution."""
import contextlib
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import supervise_calibration as supervisor
import watch_calibration as watcher

FAILURE = Path(__file__).resolve().parents[1] / 'diagnostics/observer-initialization-failure-audit.json'


class SupervisionTelemetryTests(unittest.TestCase):
    def setUp(self):
        jobs = json.loads(FAILURE.read_text())['jobs']
        self.pending = next(dict(j) for j in jobs if j['status'] == 'running' and 'completed_episodes' not in j)
        self.ready = next(dict(j) for j in jobs if j['status'] == 'running' and 'completed_episodes' in j)

    def test_actual_failed_snapshot_requires_explicit_initialization_marker(self):
        with self.assertRaisesRegex(ValueError, 'initialization marker'):
            supervisor.compact_active_job(self.pending)
        self.pending['telemetry_status'] = 'awaiting_run_directory'
        result = supervisor.compact_active_job(self.pending)
        self.assertIsNone(result['completed'])
        self.assertIsNone(result['accepted_actions'])
        self.assertIsNone(result['snapshots'])
        self.assertIsNone(result['shadow_error_bytes'])
        self.assertTrue(result['live_matching'])

    def test_actual_ready_counters_and_nonzero_errors_are_preserved(self):
        self.ready['shadow_error_bytes'] = 9
        result = supervisor.compact_active_job(self.ready)
        self.assertEqual(result['completed'], self.ready['completed_episodes'])
        self.assertEqual(result['accepted_actions'], self.ready['accepted_actions'])
        self.assertEqual(result['snapshots'], self.ready['snapshots'])
        self.assertEqual(result['shadow_error_bytes'], 9)

    def test_unmarked_missing_counter_is_not_silently_zero(self):
        del self.ready['snapshots']
        with self.assertRaisesRegex(ValueError, 'snapshots'):
            supervisor.compact_active_job(self.ready)

    def test_initializing_dead_launcher_stays_visible(self):
        self.pending.update(telemetry_status='awaiting_run_directory', launcher_live_and_matching=False)
        result = supervisor.compact_active_job(self.pending)
        self.assertFalse(result['live_matching'])
        self.assertIsNone(result['completed'])

    def test_initialization_marker_cannot_hide_measured_counters(self):
        self.ready['telemetry_status'] = 'awaiting_run_directory'
        with self.assertRaisesRegex(ValueError, 'measured counters'):
            supervisor.compact_active_job(self.ready)

    def test_real_watcher_distinguishes_pending_from_known_empty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'results/exp4'
            (root / 'meta').mkdir(parents=True)
            ready = root / 'runs/ready'
            ready.mkdir(parents=True)
            jobs = [dict(fold=1, scene='pending', gpu=2, pid=101, status='running', run_directory=None),
                    dict(fold=1, scene='ready', gpu=3, pid=102, status='running', run_directory=str(ready))]
            ledger = dict(controller_pid=100, status='running', source_fingerprint={}, jobs=jobs,
                          boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            (root / 'meta/calibration_runs.json').write_text(json.dumps(ledger))
            output = StringIO()
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(watcher, 'ROOT', root))
                stack.enter_context(patch.object(watcher, 'source_fingerprint', return_value={}))
                stack.enter_context(patch.object(watcher, 'process_matches', return_value=True))
                stack.enter_context(contextlib.redirect_stdout(output))
                watcher.main()
            result = json.loads(output.getvalue())
            pending, measured = [supervisor.compact_active_job(j) for j in result['jobs']]
            self.assertEqual(pending['telemetry_status'], 'awaiting_run_directory')
            self.assertIsNone(pending['completed'])
            self.assertEqual(measured['telemetry_status'], 'available')
            self.assertEqual(measured['completed'], 0)
            self.assertEqual(measured['shadow_error_bytes'], 0)
            self.assertTrue(result['frozen_files_unchanged'])


if __name__ == '__main__':
    unittest.main()
