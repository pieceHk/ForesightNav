"""Validate paired CSV rows against the actual normalized episode records."""
import math

METRICS=('SR','SPL','PR','PPL')

def episode_key(row):
    return row['scene_id'],str(row['episode_id'])

def episode_map(rows):
    indexed={episode_key(row):row for row in rows}
    assert len(indexed)==len(rows),'Duplicate normalized episode identity'
    return indexed

def metric_value(row,metric):
    return float(row['native_mon_metrics'][metric.lower()] if metric in ('SR','SPL') else row[metric])

def validate_pair_rows(rows,left_rows,right_rows,left,right,expected_n=160):
    a,b=episode_map(left_rows),episode_map(right_rows)
    assert set(a)==set(b) and len(a)==expected_n,'Pair must use the exact same episode pool'
    assert len(rows)==expected_n,'Incomplete paired CSV'
    seen=set();counts={'both_success':0,'left_only_success':0,'right_only_success':0,'both_failure':0,'n_paired':expected_n}
    values={side:{metric:[] for metric in METRICS} for side in ('left','right')}
    for row in rows:
        key=episode_key(row)
        assert key in a and key not in seen,'Missing, duplicate or foreign paired identity'
        seen.add(key)
        assert row['left']==left and row['right']==right,'Variant names changed within the CSV'
        assert a[key]['goals']==b[key]['goals'] and a[key]['source_episode_ids']==b[key]['source_episode_ids'],'Pair task/source identity mismatch'
        success={}
        for side,expected in (('left',a[key]),('right',b[key])):
            assert str(row[side+'_success']) in ('0','1'),'Success must be0 or1'
            success[side]=bool(int(row[side+'_success']))
            assert success[side]==bool(expected['episode_success']),'Success belongs to a different episode'
            for metric in METRICS:
                observed=float(row[side+'_'+metric]);wanted=metric_value(expected,metric)
                assert math.isfinite(observed) and math.isfinite(wanted)
                assert math.isclose(observed,wanted,rel_tol=0,abs_tol=1e-12),f'Wrong {side} {metric} for {key}'
                values[side][metric].append(observed)
        count=('both_success' if success['left'] and success['right'] else
               'left_only_success' if success['left'] else 'right_only_success' if success['right'] else 'both_failure')
        counts[count]+=1
    assert seen==set(a)
    means={side:{metric:100*math.fsum(values[side][metric])/expected_n for metric in METRICS} for side in values}
    return counts,means
