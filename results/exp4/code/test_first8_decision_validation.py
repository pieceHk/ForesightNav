import copy,unittest
from validate_first8_decisions import validate_decision_rows

class DecisionAuditTests(unittest.TestCase):
    def fixture(self,variant='full'):
        candidates=[]
        for rank,distance in enumerate((1.,2.)):
            b=1. if variant=='suffix_only' else .3
            succ=0. if variant in ('nearest','myopic') else 2.
            fail=0. if variant in ('nearest','suffix_only') else 3.
            candidates.append({'raw_rank':rank,'E':0.,'N_views':0,'S':0.,'q':.5,'B':b,'d':distance,
                'V_succ':succ,'V_fail':fail,'Q':distance+b*succ+(1-b)*fail})
        row={'scene_id':'scene','episode_id':'0','t':1,'k':0,'variant':variant,'candidates':candidates,
            'chosen_raw_rank':0,'nearest_raw_rank':0,'unconstrained_raw_rank':0,'ordered_raw_ranks':[0,1],
            'switched':False,'plan_ms':1.,'fallback':None,'fallback_counts':{}}
        return row
    def validate(self,row):
        return validate_decision_rows([row],row['variant'],[{'scene_id':'scene','runtime_episode_id':'0'}],{'scene':{'a':1.,'b0':0.,'pi':.3}})
    def test_all_four_valid_formula_switches(self):
        for variant in ('full','nearest','myopic','suffix_only'):self.assertEqual(self.validate(self.fixture(variant))['n_decisions'],1)
    def test_missing_required_candidate_field_rejected(self):
        row=self.fixture();del row['candidates'][0]['q']
        with self.assertRaises(ValueError):self.validate(row)
    def test_changed_episode_identity_rejected(self):
        row=self.fixture();row['episode_id']='other'
        with self.assertRaises(ValueError):self.validate(row)
    def test_coverage_cannot_exceed_twelve_views(self):
        row=self.fixture();row['candidates'][0]['N_views']=13
        with self.assertRaises(ValueError):self.validate(row)
    def test_wrong_self_calibration_belief_rejected_even_if_Q_consistent(self):
        row=self.fixture();c=row['candidates'][0];c['B']=.5;c['Q']=c['d']+.5*c['V_succ']+.5*c['V_fail']
        with self.assertRaises(ValueError):self.validate(row)
    def test_nearest_extra_failure_cost_rejected(self):
        row=self.fixture('nearest');c=row['candidates'][0];c['V_fail']=1.;c['Q']=c['d']+.7
        with self.assertRaises(ValueError):self.validate(row)
    def test_future_leak_in_myopic_rejected(self):
        row=self.fixture('myopic');c=row['candidates'][0];c['V_succ']=1.;c['Q']+=c['B']
        with self.assertRaises(ValueError):self.validate(row)
    def test_duplicate_full_permutation_rejected(self):
        row=self.fixture();row['ordered_raw_ranks']=[0,0,1]
        with self.assertRaises(ValueError):self.validate(row)
    def test_nonminimal_proposal_rejected(self):
        row=self.fixture();row['unconstrained_raw_rank']=1
        with self.assertRaises(ValueError):self.validate(row)
    def test_last_target_cannot_use_future_value(self):
        row=self.fixture();row['k']=2
        with self.assertRaises(ValueError):self.validate(row)
    def test_nonfinite_cost_and_invalid_latency_rejected(self):
        row=self.fixture();row['candidates'][0]['Q']=float('nan')
        with self.assertRaises(ValueError):self.validate(row)
        row=self.fixture();row['plan_ms']=-1
        with self.assertRaises(ValueError):self.validate(row)
    def test_six_candidates_rejected(self):
        row=self.fixture()
        for rank in range(2,6):
            candidate=copy.deepcopy(row['candidates'][0]);candidate['raw_rank']=rank
            row['candidates'].append(candidate);row['ordered_raw_ranks'].append(rank)
        with self.assertRaises(ValueError):self.validate(row)

if __name__=='__main__':unittest.main()
