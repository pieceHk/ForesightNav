import copy,unittest
from first8_scene_pipeline import APPROVAL_SCOPE,approved,scene_ready,suffix_ready_scenes

class PipelineTests(unittest.TestCase):
    def consent(self):return {'approved':True,'scope':APPROVAL_SCOPE,'scope_sha256':'scope','parameter_sha256':'parameter','boot_id':'boot','source':'explicit_user_chat_reply','user_reply_text':'允许按场景接续'}
    def assignment(self):
        identities=[{'scene_id':'s','runtime_episode_id':str(i)} for i in range(20)]
        job={'scene':'s','identities':identities,'status':'completed','returncode':0}
        proof={'scene':'s','completed_native_episodes':20,'audited_stage_paths':60,'n_error':0,'invalid_paths_complete':True,'source_and_parameters_unchanged':True}
        return identities,job,proof
    def test_explicit_bound_reply_required(self):
        self.assertTrue(approved(self.consent(),'scope','parameter','boot'))
        self.assertFalse(approved({},'scope','parameter','boot'))
        row=self.consent();row['approved']=False
        self.assertFalse(approved(row,'scope','parameter','boot'))
    def test_old_or_implicit_authority_rejected(self):
        for key,value in (('scope','GPU3_cleanup'),('source','proposal_only'),('user_reply_text',''),('parameter_sha256','changed'),('boot_id','other')):
            row=self.consent();row[key]=value
            self.assertFalse(approved(row,'scope','parameter','boot'))
    def test_only_complete_twenty_with_sixty_verified_paths_are_ready(self):
        ids,job,proof=self.assignment();self.assertTrue(scene_ready(job,proof,ids))
        for key,value in (('completed_native_episodes',19),('audited_stage_paths',59),('n_error',1),('invalid_paths_complete',False),('source_and_parameters_unchanged',False)):
            changed=copy.deepcopy(proof);changed[key]=value
            self.assertFalse(scene_ready(job,changed,ids))
    def test_missing_actual_exit_is_rejected(self):
        ids,job,proof=self.assignment();del job['returncode']
        self.assertFalse(scene_ready(job,proof,ids))
    def test_failed_or_live_launcher_cannot_unlock_suffix(self):
        ids,job,proof=self.assignment()
        job['returncode']=1;self.assertFalse(scene_ready(job,proof,ids))
        job['returncode']=0;job['status']='running';self.assertFalse(scene_ready(job,proof,ids))
    def test_changed_fixed_assignment_rejected(self):
        ids,job,proof=self.assignment();job['identities']=job['identities'][:-1]
        self.assertFalse(scene_ready(job,proof,ids))
    def test_same_scene_is_enqueued_once_only(self):
        ids,job,proof=self.assignment()
        self.assertEqual(suffix_ready_scenes(['s'],[job],[],{'s':proof},ids),['s'])
        self.assertEqual(suffix_ready_scenes(['s'],[job],[{'scene':'s','status':'running'}],{'s':proof},ids),[])
    def test_no_GT_audit_cannot_unlock_suffix(self):
        ids,job,proof=self.assignment();self.assertEqual(suffix_ready_scenes(['s'],[job],[],{},ids),[])
    def test_success_score_is_not_a_scheduling_filter(self):
        ids,job,proof=self.assignment();proof['SR']=0
        self.assertEqual(suffix_ready_scenes(['s'],[job],[],{'s':proof},ids),['s'])
        proof['SR']=100;self.assertEqual(suffix_ready_scenes(['s'],[job],[],{'s':proof},ids),['s'])
    def test_duplicate_scene_assignments_rejected(self):
        ids,job,proof=self.assignment()
        with self.assertRaises(ValueError):suffix_ready_scenes(['s'],[job,copy.deepcopy(job)],[],{'s':proof},ids)

if __name__=='__main__':unittest.main()
