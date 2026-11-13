import copy,unittest
from dataclasses import asdict
from audit_first8_snapshot_replay import verify_row
from pbsp.navigation.blp import BeliefParameters,solve_snapshot

class ReplayAuditTests(unittest.TestCase):
    def fixture(self,variant='full'):
        params=BeliefParameters();binding={'calibration_sha256':'fixture'}
        snapshot={'k':0,'layers':[[{'raw_rank':0,'location':1,'position':[1,0,0],'E':.2,'N_views':2}],[],[]],
            'distances':[[0,1],[1,0]],'frontier_available':False,'nearest_frontier':[None,None],'all_raw_ranks':[0,1]}
        row={'scene_id':'s','episode_id':'0','t':1,'k':0,'shadow':False,'active_raw_rank':None,
            'snapshot':snapshot,'parameters':asdict(params),'parameter_binding':binding}
        observed={**{k:row[k] for k in ('scene_id','episode_id','t','k')},**solve_snapshot(snapshot,params,variant)}
        return row,observed,variant,asdict(params),binding
    def check(self,fixture):return verify_row(*fixture,solve_snapshot,BeliefParameters)
    def test_valid_online_row(self):self.assertEqual(self.check(self.fixture())['complete_order_count'],2)
    def test_missing_unscored_candidate_rejected(self):
        fixture=self.fixture();fixture[1]['ordered_raw_ranks']=[0]
        with self.assertRaisesRegex(ValueError,'complete input permutation'):self.check(fixture)
    def test_recursive_failure_value_corruption_rejected(self):
        fixture=self.fixture();fixture[1]['candidates'][0]['V_fail']+=1
        with self.assertRaisesRegex(ValueError,'V_fail'):self.check(fixture)
    def test_recursive_Q_corruption_rejected(self):
        fixture=self.fixture();fixture[1]['candidates'][0]['Q']+=1
        with self.assertRaisesRegex(ValueError,'Q'):self.check(fixture)
    def test_changed_identity_rejected(self):
        fixture=self.fixture();fixture[1]['episode_id']='1'
        with self.assertRaisesRegex(ValueError,'identities'):self.check(fixture)
    def test_shadow_snapshot_rejected(self):
        fixture=self.fixture();fixture[0]['shadow']=True
        with self.assertRaisesRegex(ValueError,'Shadow'):self.check(fixture)
    def test_changed_frozen_parameters_rejected(self):
        fixture=self.fixture();fixture[0]['parameters']=dict(fixture[0]['parameters'],pi=.9)
        with self.assertRaisesRegex(ValueError,'Frozen'):self.check(fixture)
    def test_corrupt_fallback_counts_rejected(self):
        fixture=self.fixture();fixture[1]['fallback_counts']['empty_candidates']+=1
        with self.assertRaisesRegex(ValueError,'fallback_counts'):self.check(fixture)
    def test_myopic_cannot_query_future(self):
        fixture=self.fixture('myopic');fixture[0]['snapshot']['layers'][1]=copy.deepcopy(fixture[0]['snapshot']['layers'][0])
        with self.assertRaisesRegex(ValueError,'queried future'):self.check(fixture)
    def test_altered_memo_state_count_rejected(self):
        fixture=self.fixture();fixture[1]['memo_states']+=1
        with self.assertRaisesRegex(ValueError,'memo_states'):self.check(fixture)

if __name__=='__main__':unittest.main()
