import copy,unittest
from first8_infrastructure_recovery_v2 import validated_receipt,validate_retry_assignment,FAILED_SCENES,startup_host_ready,mapping_is_unusable,cleanup_target

class RecoveryV2Tests(unittest.TestCase):
    def receipt(self,code=0):
        return {'pid':23,'start_ticks':'90','uid':1001,'actual_wait_status':code<<8,'exit_event_status':code<<8,'returncode':code,
            'source':'real os.waitpid of traced launcher after PTRACE_EVENT_EXIT'}
    def test_actual_zero_exit(self):self.assertEqual(validated_receipt(self.receipt(),23,'90'),0)
    def test_actual_nonzero_is_not_success(self):self.assertEqual(validated_receipt(self.receipt(7),23,'90'),7)
    def test_reused_pid_rejected(self):
        with self.assertRaises(ValueError):validated_receipt(self.receipt(),23,'91')
    def test_wrong_user_rejected(self):
        data=self.receipt();data['uid']=1008
        with self.assertRaises(ValueError):validated_receipt(data,23,'90')
    def test_native_status_cannot_replace_actual_os_wait(self):
        data=self.receipt();data['source']='status.json completed'
        with self.assertRaises(ValueError):validated_receipt(data,23,'90')
    def test_disagreeing_exit_event_rejected(self):
        data=self.receipt();data['exit_event_status']=256
        with self.assertRaises(ValueError):validated_receipt(data,23,'90')
    def test_claimed_zero_cannot_override_actual_failure(self):
        data=self.receipt(7);data['returncode']=0
        with self.assertRaises(ValueError):validated_receipt(data,23,'90')
    def assignment(self):
        scenes=['s'+str(i) for i in range(6)]+sorted(FAILED_SCENES)
        scope={'ordered_scenes':scenes,'identities':[{'scene_id':s,'runtime_episode_id':str(i)} for s in scenes for i in range(20)]}
        jobs=[{'scene':s,'identities':[r for r in scope['identities'] if r['scene_id']==s]} for s in scenes[:6]]
        return jobs,scope
    def test_both_entire_failed_scenes_are_retried(self):
        jobs,scope=self.assignment();validate_retry_assignment(jobs,scope)
    def test_duplicate_retained_scene_rejected(self):
        jobs,scope=self.assignment();jobs.append(copy.deepcopy(jobs[0]))
        with self.assertRaises(ValueError):validate_retry_assignment(jobs,scope)
    def test_changed_or_partial_assignment_rejected(self):
        jobs,scope=self.assignment();jobs[0]['identities'].pop()
        with self.assertRaises(ValueError):validate_retry_assignment(jobs,scope)
    def test_wrong_scene_retry_rejected(self):
        jobs,scope=self.assignment();scene=sorted(FAILED_SCENES)[0]
        jobs=jobs[1:]+[{'scene':scene,'identities':[r for r in scope['identities'] if r['scene_id']==scene]}]
        with self.assertRaises(ValueError):validate_retry_assignment(jobs,scope)
    def test_twenty_identities_per_retry_required(self):
        jobs,scope=self.assignment();scope['identities']=scope['identities'][:-1]
        with self.assertRaises(ValueError):validate_retry_assignment(jobs,scope)
    def test_startup_reservation_adds_to_existing_floor(self):
        self.assertFalse(startup_host_ready(64,0));self.assertFalse(startup_host_ready(79.99,0));self.assertTrue(startup_host_ready(80,0))
    def test_unsettled_launch_reservations_accumulate(self):
        self.assertFalse(startup_host_ready(95.99,1));self.assertTrue(startup_host_ready(96,1))
        self.assertFalse(startup_host_ready(191.99,7));self.assertTrue(startup_host_ready(192,7))
    def test_unknown_invalid_startup_readings_fail_closed(self):
        for value in (None,'128',float('nan'),float('inf'),-1,True):self.assertFalse(startup_host_ready(value,0))
        for recent in (-1,True,None):self.assertFalse(startup_host_ready(192,recent))
    def test_fatal_mapper_oom_detected_even_if_frames_continue(self):
        self.assertTrue(mapping_is_unusable("RuntimeError: Mapping rebuild failed; incomplete maps cannot be used: OutOfMemoryError('CUDA out of memory. Tried to allocate 1.59 GiB.')"))
    def test_general_allocator_advice_does_not_trigger_cleanup(self):
        self.assertFalse(mapping_is_unusable('CUDA out of memory guidance: change allocator if required'))
        self.assertFalse(mapping_is_unusable('Mapping rebuild completed successfully'))
    def job(self):return {'pid':23,'pid_start_ticks':'90','command':['python','run.py','--scene','scene']}
    def test_only_matching_own_live_launcher_can_be_cleanup_target(self):
        job=self.job();state={'live':True,'start_ticks':'90'}
        self.assertEqual(cleanup_target(job,state,1001,job['command']),23)
        self.assertIsNone(cleanup_target(job,state,1008,job['command']))
        self.assertIsNone(cleanup_target(job,{'live':True,'start_ticks':'91'},1001,job['command']))
        self.assertIsNone(cleanup_target(job,{'live':False,'start_ticks':'90'},1001,job['command']))
        self.assertIsNone(cleanup_target(job,state,1001,['python','other.py']))

if __name__=='__main__':unittest.main()
