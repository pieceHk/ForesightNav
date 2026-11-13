import copy, unittest
from first8_infrastructure_recovery import validated_receipt, validate_retry_assignment, FAILED_SCENE

class RecoveryTests(unittest.TestCase):
    def receipt(self,code=0):
        return {'pid':23,'start_ticks':'90','uid':1001,'actual_wait_status':code<<8,'exit_event_status':code<<8,
                'returncode':code,'source':'real os.waitpid of traced launcher after PTRACE_EVENT_EXIT'}
    def test_actual_zero_exit(self):self.assertEqual(validated_receipt(self.receipt(),23,'90'),0)
    def test_actual_failed_exit_is_not_success(self):self.assertEqual(validated_receipt(self.receipt(7),23,'90'),7)
    def test_reused_pid_is_rejected(self):
        with self.assertRaises(ValueError):validated_receipt(self.receipt(),23,'91')
    def test_wrong_user_is_rejected(self):
        r=self.receipt();r['uid']=1008
        with self.assertRaises(ValueError):validated_receipt(r,23,'90')
    def test_non_os_success_marker_rejected(self):
        r=self.receipt();r['source']='status.json completed'
        with self.assertRaises(ValueError):validated_receipt(r,23,'90')
    def test_exit_event_must_match_wait(self):
        r=self.receipt();r['exit_event_status']=256
        with self.assertRaises(ValueError):validated_receipt(r,23,'90')
    def test_claimed_zero_must_match_actual_nonzero(self):
        r=self.receipt(7);r['returncode']=0
        with self.assertRaises(ValueError):validated_receipt(r,23,'90')
    def assignment(self):
        scenes=['s'+str(i) for i in range(7)]+[FAILED_SCENE]
        scope={'ordered_scenes':scenes,'identities':[{'scene_id':s,'runtime_episode_id':str(i)} for s in scenes for i in range(20)]}
        jobs=[{'scene':s,'identities':[r for r in scope['identities'] if r['scene_id']==s]} for s in scenes[:-1]]
        return jobs,scope
    def test_seven_scenes_retained_entire_failed_scene_retried(self):
        jobs,scope=self.assignment();validate_retry_assignment(jobs,scope)
    def test_duplicate_retained_scene_rejected(self):
        jobs,scope=self.assignment();jobs.append(copy.deepcopy(jobs[0]))
        with self.assertRaises(ValueError):validate_retry_assignment(jobs,scope)
    def test_partial_scene_or_changed_ids_rejected(self):
        jobs,scope=self.assignment();jobs[0]['identities'].pop()
        with self.assertRaises(ValueError):validate_retry_assignment(jobs,scope)
    def test_retrying_another_scene_is_rejected(self):
        jobs,scope=self.assignment();jobs=jobs[1:]+[{'scene':FAILED_SCENE,'identities':[r for r in scope['identities'] if r['scene_id']==FAILED_SCENE]}]
        with self.assertRaises(ValueError):validate_retry_assignment(jobs,scope)
    def test_full_twenty_ids_required(self):
        jobs,scope=self.assignment();scope['identities']=scope['identities'][:-1]
        with self.assertRaises(ValueError):validate_retry_assignment(jobs,scope)

if __name__=='__main__':unittest.main()
