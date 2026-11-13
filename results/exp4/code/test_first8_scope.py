import copy
import unittest
from unittest.mock import patch
import first8_evaluate as scope

class ScopeTests(unittest.TestCase):
    def identities(self):
        return [{'scene_id':f's{s:02d}','runtime_episode_id':str(e),
                 'source_episode_ids':[str(3*e+k) for k in range(3)],
                 'ordered_categories':['bed','chair','couch']} for s in range(20) for e in range(20)]

    def test_original_order_including_complete_failures_is_fixed(self):
        rows=self.identities()
        for row in rows:row['irrelevant_observed_success']=int(row['scene_id']>'s07')
        scenes,selected=scope.select_first8(rows)
        self.assertEqual(scenes,[f's{s:02d}' for s in range(8)])
        self.assertEqual(selected,rows[:160])

    def test_duplicate_episode_is_rejected(self):
        rows=self.identities();rows[1]=copy.deepcopy(rows[0])
        with self.assertRaises(scope.base.GateError):scope.select_first8(rows)

    def test_partial_scene_is_rejected(self):
        rows=self.identities();del rows[1]
        with self.assertRaises(scope.base.GateError):scope.select_first8(rows)

    @patch.object(scope.base,'audit_scene')
    def test_completed_status_without_actual_zero_exit_is_rejected(self,audit):
        scenes,rows=scope.select_first8(self.identities())
        target={'ordered_scenes':scenes,'identities':rows}
        jobs=[{'scene':s,'identities':[r for r in rows if r['scene_id']==s],
               'status':'completed','returncode':0,'run_directory':'/unused'} for s in scenes]
        scope.validate_jobs(jobs,target)
        self.assertEqual(audit.call_count,8)
        del jobs[2]['returncode']
        with self.assertRaises(scope.base.GateError):scope.validate_jobs(jobs,target)

    @patch.object(scope.base,'audit_scene')
    def test_same_size_different_scene_cannot_replace_a_selected_scene(self,audit):
        scenes,rows=scope.select_first8(self.identities())
        target={'ordered_scenes':scenes,'identities':rows}
        jobs=[{'scene':s,'identities':[r for r in rows if r['scene_id']==s],
               'status':'completed','returncode':0,'run_directory':'/unused'} for s in scenes]
        jobs[-1]['scene']='s08'
        with self.assertRaises(scope.base.GateError):scope.validate_jobs(jobs,target)

if __name__=='__main__':unittest.main()
