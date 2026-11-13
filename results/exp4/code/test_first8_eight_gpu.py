import copy
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch
import first8_evaluate8 as engine

class EightGpuTests(unittest.TestCase):
    def inventory(self,free='32768',occupied='11264'):
        return {'gpus':[{'index':str(i),'uuid':f'g{i}','memory.free':free} for i in range(8)],
                'compute_apps':[{'gpu_uuid':f'g{i}','used_gpu_memory':occupied} for i in range(8)]}

    def test_latest_authorization_includes_physical_zero_and_three(self):
        self.assertEqual(engine.eligible(self.inventory()),list(range(8)))

    def test_low_free_memory_rejected(self):
        self.assertEqual(engine.eligible(self.inventory(free='32767')),[])

    def test_foreign_growth_rejected(self):
        self.assertEqual(engine.eligible(self.inventory(occupied='11265')),[])

    def test_unknown_and_negative_memory_are_closed(self):
        self.assertEqual(engine.eligible(self.inventory(occupied='N/A')),[])
        self.assertEqual(engine.eligible(self.inventory(occupied='-1')),[])

    def test_unapproved_gpu_not_admitted(self):
        self.assertEqual(engine.eligible(self.inventory(),[5,6,7]),[5,6,7])

    @patch.object(engine.base,'audit_scene')
    def test_native_completion_without_real_exit_is_rejected(self,audit):
        scope={'ordered_scenes':['scene'],'identities':[{'scene_id':'scene','runtime_episode_id':'0'}]}
        ledger={'stage':'myopic','status':'running','jobs':[{'scene':'scene','identities':scope['identities'],
                'status':'completed','run_directory':'/unused'}]}
        with self.assertRaises(AssertionError):engine.validate_retained(ledger,scope)
        ledger['jobs'][0]['returncode']=0
        self.assertEqual(engine.validate_retained(ledger,scope),[])
        audit.assert_called_once()

    def test_duplicate_or_changed_assignment_rejected(self):
        scope={'ordered_scenes':['scene'],'identities':[{'scene_id':'scene','runtime_episode_id':'0'}]}
        ledger={'stage':'myopic','status':'running','jobs':[{'scene':'outside','identities':scope['identities'],'status':'completed','returncode':0}]}
        with self.assertRaises(AssertionError):engine.validate_retained(ledger,scope)

    @patch.object(engine.os,'kill')
    def test_parent_is_kept_while_any_adopted_native_remains_running(self,kill):
        plan={'adopted_launchers':[{'scene':'a'}],'old_controller':{'pid':9,'start_ticks':'10'}}
        with tempfile.TemporaryDirectory() as directory:
            engine.retire_parent(plan,[{'scene':'a','status':'running'}],Path(directory))
        kill.assert_not_called()

    @patch.object(engine,'save')
    @patch.object(engine,'assert_no_live_children')
    @patch.object(engine,'checked_cpu')
    @patch.object(engine.os,'kill')
    def test_parent_retired_only_after_actual_success_and_child_audit(self,kill,check,children,save):
        old={'pid':9,'start_ticks':'10'}
        plan={'adopted_launchers':[{'scene':'a'}],'old_controller':old}
        with tempfile.TemporaryDirectory() as directory:
            engine.retire_parent(plan,[{'scene':'a','status':'completed','returncode':0}],Path(directory))
        check.assert_called_once_with(old,stopped=True)
        children.assert_called_once_with(9)
        kill.assert_called_once_with(9,signal.SIGKILL)

    @patch.object(engine.os,'kill')
    def test_nonzero_native_exit_cannot_retire_parent_as_success(self,kill):
        plan={'adopted_launchers':[{'scene':'a'}],'old_controller':{'pid':9,'start_ticks':'10'}}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AssertionError):
                engine.retire_parent(plan,[{'scene':'a','status':'completed','returncode':1}],Path(directory))
        kill.assert_not_called()

if __name__=='__main__':unittest.main()
