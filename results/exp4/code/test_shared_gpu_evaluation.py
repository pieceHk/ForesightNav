"""Scheduling safety checks; Linux process probes use only spawned toy children."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import shared_gpu_evaluate_sequence as parallel


class SchedulingTests(unittest.TestCase):
    def identities(self):
        return [{'scene_id': f'scene{s:02}', 'runtime_episode_id': str(e),
                 'ordered_categories': ['a','b','c'], 'source_episode_ids': [s*60+e*3+i for i in range(3)]}
                for s in range(20) for e in range(20)]

    def jobs(self):
        return parallel.build_jobs('full', self.identities())

    def test_existing_scenes_are_never_reassigned(self):
        jobs = self.jobs()
        existing = [dict(jobs[0], status='completed'), dict(jobs[1], status='running')]
        with patch.object(parallel, 'expected_identities', return_value=self.identities()):
            pending = parallel.pending_assignments('full', existing)
        self.assertEqual(pending, jobs[2:])
        self.assertEqual(sum(len(job['identities']) for job in existing + pending), 400)

    def test_duplicate_existing_scene_is_rejected(self):
        job = dict(self.jobs()[0], status='completed')
        with patch.object(parallel, 'expected_identities', return_value=self.identities()):
            with self.assertRaises(parallel.GateError):
                parallel.pending_assignments('full', [job, job])

    def test_changed_target_order_is_rejected(self):
        job = dict(self.jobs()[0], status='running')
        job['identities'] = json.loads(json.dumps(job['identities']))
        job['identities'][0]['ordered_categories'] = ['b','a','c']
        with patch.object(parallel, 'expected_identities', return_value=self.identities()):
            with self.assertRaises(parallel.GateError):
                parallel.pending_assignments('full', [job])

    def test_failed_scene_cannot_be_silently_resumed(self):
        with patch.object(parallel, 'expected_identities', return_value=self.identities()):
            with self.assertRaises(parallel.GateError):
                parallel.pending_assignments('full', [dict(self.jobs()[0], status='failed')])

    def test_idle_selection_respects_occupancy_capacity_and_authorization(self):
        inventory = {'gpus': [{'index': str(i), 'uuid': f'g{i}', 'memory.free': str(free)}
                     for i,free in enumerate([45000,45000,38000,45000])],
                     'compute_apps': [{'gpu_uuid':'g1','pid':'123','used_gpu_memory':'4000'}]}
        self.assertEqual(parallel.eligible_gpus(inventory, [0,1,2]), [1])

    def test_pid_birth_mismatch_is_rejected(self):
        with patch.object(parallel, 'process', return_value={'live':True,'start_ticks':'11','state':'S'}):
            with self.assertRaises(parallel.GateError):
                parallel.checked_process(12, '10')

    def test_stopped_parent_is_required(self):
        with patch.object(parallel, 'process', return_value={'live':True,'start_ticks':'11','state':'S'}):
            with self.assertRaises(parallel.GateError):
                parallel.checked_process(12, '11', stopped=True)

    def test_gpu_selection_does_not_signal_any_process(self):
        with patch.object(parallel.os, 'kill', side_effect=AssertionError('No signals permitted')):
            self.assertEqual(parallel.eligible_gpus({'gpus':[],'compute_apps':[]}, list(range(8))), [])

    def process_probe(self, exitcode):
        with tempfile.TemporaryDirectory(prefix='exp4-controller-handoff-test-') as temporary:
            directory = Path(temporary)
            worker = 'import pathlib,sys,time\np=pathlib.Path(sys.argv[1])\nfor i in range(16):\n p.write_text(str(i));time.sleep(.06)\nsys.exit(int(sys.argv[2]))\n'
            parent = ('import pathlib,subprocess,sys,time,json\n'
                'p=subprocess.Popen([sys.executable,"-c",sys.argv[2],sys.argv[3],sys.argv[4]])\n'
                'pathlib.Path(sys.argv[1]).write_text(str(p.pid))\n'
                'time.sleep(60)\n')
            child = subprocess.Popen([sys.executable,'-c',parent,str(directory/'pid'),worker,
                                      str(directory/'heartbeat'),str(exitcode)])
            try:
                deadline = time.monotonic()+5
                while not (directory/'pid').exists() or not (directory/'heartbeat').exists():
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.02)
                pid = int((directory/'pid').read_text())
                ticks = parallel.process(pid)['start_ticks']
                parent_ticks = parallel.process(child.pid)['start_ticks']
                os.kill(child.pid, signal.SIGSTOP)
                while parallel.process(child.pid)['state'] not in ('T','t'):
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.01)
                initial = int((directory/'heartbeat').read_text())
                time.sleep(.2)
                self.assertGreater(int((directory/'heartbeat').read_text()), initial)
                result, proof = parallel.adopted_returncode(pid, ticks)
                while result is None:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.03)
                    result, proof = parallel.adopted_returncode(pid, ticks)
                self.assertEqual(result, exitcode)
                self.assertEqual(proof['wait_status'], exitcode << 8)
                with self.assertRaises(parallel.GateError):
                    parallel.adopted_returncode(pid, str(int(ticks)+1))
                parallel.checked_process(child.pid, parent_ticks, stopped=True)
                os.kill(child.pid, signal.SIGKILL)
                self.assertEqual(child.wait(timeout=5), -signal.SIGKILL)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)

    def test_stopped_parent_does_not_stop_native_child_and_exit0_is_captured(self):
        self.process_probe(0)

    def test_nonzero_native_exit_is_preserved(self):
        self.process_probe(17)


    def test_latest_policy_excludes_gpu0_and_other_cards(self):
        with patch.object(parallel, 'load', return_value={'allowed_gpus':[1,2,3]}):
            self.assertEqual(parallel.policy_allowed([0,1,2,3,4,5,6,7]), [1,2,3])
        with patch.object(parallel, 'load', return_value={'allowed_gpus':[0,1,2,3]}):
            with self.assertRaises(parallel.GateError):
                parallel.policy_allowed([0,1,2,3])

    def test_resource_withdrawal_removes_only_exact_interrupted_launcher(self):
        jobs = [dict(self.jobs()[0], status='running', pid=10,gpu=0),
                dict(self.jobs()[1], status='running', pid=11,gpu=1)]
        proof = {'gpu0_native_completed_at_withdrawal':0, 'raw_logs_retained':True,
                 'job':jobs[0], 'scene':jobs[0]['scene']}
        with patch.object(parallel, 'load', return_value=proof):
            self.assertEqual(parallel.keep_authorized_jobs('full',jobs), [jobs[1]])
            self.assertEqual(parallel.keep_authorized_jobs('nearest',jobs), jobs)

    def test_completed_results_cannot_be_removed_as_initialization_withdrawal(self):
        proof = {'gpu0_native_completed_at_withdrawal':1, 'raw_logs_retained':True}
        with patch.object(parallel, 'load', return_value=proof):
            with self.assertRaises(parallel.GateError):
                parallel.keep_authorized_jobs('full',[])

    def test_resumed_scene_keeps_all_twenty_frozen_identities(self):
        jobs = self.jobs()
        existing = [dict(jobs[1],status='running',pid=11,gpu=1)]
        with patch.object(parallel,'expected_identities',return_value=self.identities()):
            pending = parallel.pending_assignments('full',existing)
        self.assertEqual(pending[0],jobs[0])
        self.assertEqual(len(pending[0]['identities']),20)
        self.assertEqual(sum(len(job['identities']) for job in existing+pending),400)

    def memory_inventory(self, free=38912, occupied='8192'):
        return {'gpus':[{'index':'2','uuid':'g2','memory.free':str(free)}],
                'compute_apps':[{'gpu_uuid':'g2','pid':'123','used_gpu_memory':occupied}]}

    def test_shared_boundary_keeps_original_thirty_eight_GiB_headroom(self):
        self.assertEqual(parallel.eligible_gpus(self.memory_inventory(),[2]),[2])
        self.assertEqual(parallel.eligible_gpus(self.memory_inventory(free=38911),[2]),[])

    def test_external_over_eight_GiB_is_not_admitted(self):
        self.assertEqual(parallel.eligible_gpus(self.memory_inventory(occupied='8193'),[2]),[])

    def test_unknown_or_negative_memory_is_not_admitted(self):
        for value in ('N/A','-1',None):
            self.assertEqual(parallel.eligible_gpus(self.memory_inventory(occupied=value),[2]),[])

    def test_four_external_workers_still_need_original_headroom(self):
        inventory = self.memory_inventory(free=36633)
        inventory['compute_apps'] = [{'gpu_uuid':'g2','pid':str(i),'used_gpu_memory':'2008'} for i in range(4)]
        self.assertEqual(parallel.eligible_gpus(inventory,[1,2,3]),[])
        inventory['gpus'][0]['memory.free']='40000'
        self.assertEqual(parallel.eligible_gpus(inventory,[1,2,3]),[2])

    def test_current_two_external_workers_fit_original_headroom(self):
        inventory=self.memory_inventory(free=41181)
        inventory['compute_apps']=[{'gpu_uuid':'g2','used_gpu_memory':'2008'} for _ in range(2)]
        self.assertEqual(parallel.eligible_gpus(inventory,[1,2,3]),[2])

    def test_all_predecessor_parent_generations_are_bound(self):
        old={'pid':30,'start_ticks':'300','boot_id':'boot','controller_sha256':'sha','scheduling_plan':'plan',
             'predecessor_controller':{'pid':20,'start_ticks':'200'}}
        plan={'boot_id':'boot','controller_sha256':'sha','parent_controllers':[{'pid':10,'start_ticks':'100'},{'pid':20,'start_ticks':'200'}]}
        with patch.object(parallel,'load',return_value=plan):
            self.assertEqual(parallel.recorded_parent_births(old),{30:'300',20:'200',10:'100'})

    def test_conflicting_ancestor_birth_is_rejected(self):
        old={'pid':30,'start_ticks':'300','boot_id':'boot','controller_sha256':'sha','scheduling_plan':'plan',
             'predecessor_controller':{'pid':20,'start_ticks':'200'}}
        plan={'boot_id':'boot','controller_sha256':'sha','parent_controllers':[{'pid':20,'start_ticks':'201'}]}
        with patch.object(parallel,'load',return_value=plan):
            with self.assertRaises(parallel.GateError):
                parallel.recorded_parent_births(old)

    def test_foreign_boot_plan_is_rejected(self):
        old={'pid':30,'start_ticks':'300','boot_id':'boot','controller_sha256':'sha','scheduling_plan':'plan'}
        with patch.object(parallel,'load',return_value={'boot_id':'other','controller_sha256':'sha'}):
            with self.assertRaises(parallel.GateError):
                parallel.recorded_parent_births(old)

    def test_public_inventory_excludes_other_cards_and_process_ids(self):
        inventory={'gpus':[{'index':str(i),'uuid':f'g{i}','memory.free':'45000'} for i in range(8)],
                   'compute_apps':[{'gpu_uuid':f'g{i}','pid':'123','used_gpu_memory':'2008'} for i in range(8)]}
        public=parallel.public_inventory(inventory)
        self.assertEqual([int(g['index']) for g in public['gpus']],[1,2,3])
        self.assertEqual(len(public['compute_apps']),3)
        self.assertTrue(all('pid' not in row for row in public['compute_apps']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
