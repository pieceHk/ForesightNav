"""Counterexamples whose aggregate counts or means can look correct."""
import copy,unittest
from validate_first8_pairs import validate_pair_rows

def episode(index,success,spl):
    return {'scene_id':'fixed_scene','episode_id':str(index),'goals':['chair','bed','tv'],
            'source_episode_ids':[str(index*3+j) for j in range(3)],
            'episode_success':success,'native_mon_metrics':{'sr':int(success),'spl':spl},'PR':1.0,'PPL':spl}

class PairValidationTests(unittest.TestCase):
    def setUp(self):
        self.left=[episode(0,True,.2),episode(1,True,.4),episode(2,False,0)]
        self.right=[episode(0,False,0),episode(1,True,.3),episode(2,False,0)]
        self.rows=[]
        for a,b in zip(self.left,self.right):
            row={'scene_id':a['scene_id'],'episode_id':a['episode_id'],'left':'nearest','right':'full'}
            for side,value in (('left',a),('right',b)):
                row[side+'_success']=str(int(value['episode_success']))
                row.update({side+'_'+metric:str(value['native_mon_metrics'][metric.lower()] if metric in ('SR','SPL') else value[metric])
                            for metric in ('SR','SPL','PR','PPL')})
            self.rows.append(row)
    def check(self):return validate_pair_rows(self.rows,self.left,self.right,'nearest','full',expected_n=3)
    def test_complete_pair_matches(self):
        counts,means=self.check()
        self.assertEqual(counts,{'both_success':1,'left_only_success':1,'right_only_success':0,'both_failure':1,'n_paired':3})
        self.assertAlmostEqual(means['left']['SPL'],20)
    def test_same_outcome_duplicate_is_rejected(self):
        # Same successes and length; a simple contingency/count check cannot catch this.
        self.rows[1]['episode_id']='0'
        with self.assertRaises(AssertionError):self.check()
    def test_swapped_scores_preserve_mean_but_are_rejected(self):
        self.rows[0]['left_SPL'],self.rows[1]['left_SPL']=self.rows[1]['left_SPL'],self.rows[0]['left_SPL']
        with self.assertRaises(AssertionError):self.check()
    def test_mixed_variant_name_is_rejected(self):
        self.rows[2]['right']='myopic'
        with self.assertRaises(AssertionError):self.check()
    def test_source_task_mismatch_is_rejected(self):
        self.right[1]['source_episode_ids']=['other','source','ids']
        with self.assertRaises(AssertionError):self.check()
    def test_nonfinite_metric_is_rejected(self):
        self.rows[0]['right_PR']='nan'
        with self.assertRaises(AssertionError):self.check()

if __name__=='__main__':unittest.main()
