"""Review observed CP3 latency tails and confirmation failures; no fitting."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ledger = load(ROOT / 'meta/evaluation-smoke-runs.json')
    assert ledger['status'] == 'completed' and ledger['n_episodes'] == 20 and ledger['n_error'] == 0
    behavior = load(ROOT / 'diagnostics/CP3/completed-behavior-review.json')
    summary = load(ROOT / 'diagnostics/CP3/summary/full.json')
    decisions_path = ROOT / 'diagnostics/CP3/logs/full_decisions.jsonl'
    decisions = [json.loads(line) for line in decisions_path.read_text().splitlines() if line]
    tail = load(ROOT / 'diagnostics/CP3/latency-tail-review.json')
    assert digest(decisions_path) == tail['source_decisions_sha256'] and tail['snapshots'] == len(decisions)
    by_scene = {job['scene']: job for job in ledger['jobs']}
    categories, rows = Counter(), []
    for episode in behavior['episodes']:
        run = Path(by_scene[episode['scene_id']]['run_directory'])
        events = [json.loads(line) for path in run.rglob('blp_events.jsonl')
                  for line in path.read_text().splitlines() if line]
        for stage in episode['stages']:
            if stage['outcome_description'] != 'system_confirmed_but_native_failed':
                continue
            confirmations = [event for event in events if event['event'] == 'verification_final'
                             and event['k'] == stage['k'] - 1 and event['v'] == 1]
            last = max(confirmations, key=lambda event: event.get('last_t', event['t'])) if confirmations else None
            label = str(last['y']) if last else 'not_recorded'
            categories[label] += 1
            rows.append({'scene_id': episode['scene_id'], 'episode_id': episode['episode_id'],
                         'stage': stage['k'], 'native_success': stage['success'],
                         'last_observed_confirmation': {key: last[key] for key in
                             ('t', 'last_t', 'candidate_id', 'E', 'S', 'v', 'y', 'position') if key in last} if last else None})
    assert len(rows) == 10
    review = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': digest(Path(__file__)),
              'definition': 'Last recorded v=1 candidate check in each confirmed-stop/native-failure stage; coordinate GT at check is distinct from final agent success or moving candidate location',
              'confirmed_stop_native_failure_stages': len(rows),
              'last_confirmed_candidate_y_at_check_counts': dict(categories), 'stages': rows,
              'evaluation_used_for_fitting_or_selection': False, 'native_data_source_or_parameters_changed': False}
    (ROOT / 'diagnostics/CP3/confirmation-failure-review.json').write_text(json.dumps(review, indent=2) + '\n')
    process = summary['process']
    archive = load(ROOT / 'meta/evaluation-smoke-log-archive.json')
    lines = ['# CP3审阅入口：固定20条Full', '',
             '[完整CP3指标、20条逐例结果与两张俯视图](CP3.md)。本次3/20条全部目标成功，21/60个阶段成功；当前尚无四变体在线对照，无法从这些结果判断机制增益。', '',
             '| 指标 | 结果 |', '| --- | --- |',
             f'| SR / SPL | {summary["SR"]:.2f}% / {summary["SPL"]:.2f}% |',
             f'| PR / PPL | {summary["PR"]:.2f}% / {summary["PPL"]:.2f}% |',
             f'| 实际下发偏离最近 | {process["deviation_from_nearest_realized_pct"]:.2f}% |',
             f'| 规划耗时中位数 / P95 | {process["planning_ms"]["median"]:.2f} / {process["planning_ms"]["p95"]:.2f} ms |', '',
             f'已核对全部60阶段路径、1999个快照及场景/episode/目标顺序；源码与冻结参数未变。20份原始日志归档共{archive["archive_bytes_total"] / 1024**2:.2f} MiB，逐文件与压缩往返SHA均匹配。[原始日志及SHA清单](../meta/evaluation-smoke-log-archive.json)。', '',
             '## 全量前需要看到的限制', '',
             f'规划耗时有长尾：1999次中{tail["over_100ms"]}次超过100ms，{tail["over_1000ms"]}次超过1秒，最大{tail["max_ms"] / 1000:.3f}秒。所有记录均保留在统计和图中，CDF采用对数横轴；目前未确定长尾原因。[具体快照](../diagnostics/CP3/latency-tail-review.json)。', '',
             f'60个阶段中21个原生成功、29个预算耗尽后失败、10个在系统确认后停止但原生失败。后10个阶段的最后一次v=1检查，其候选坐标GT为0的有{categories.get("0", 0)}个、GT为1的有{categories.get("1", 0)}个、未记录的有{categories.get("not_recorded", 0)}个。候选检查坐标与最终agent位置不同，不能据此直接归因为导航器或递归排序错误。[逐阶段复核](../diagnostics/CP3/confirmation-failure-review.json)。', '',
             '固定任务池经过历史配对结果筛选，结论限于该池内受控消融；此外有部分目标存在高度或navmesh连通性疑点。所有回合保留，未据本次3-MON结果改参数或选样。[原始来源与可达性静态审计](dataset_scope.md)。', '',
             '建议按已冻结协议完成四变体对照，以检验belief、suffix及交互项。若需要独立盲测或跨数据泛化结论，应另建独立任务池并统一重新标定、评估。按实验设计CP3【需确认】，当前未创建CP3接受记录，也未启动Full400。', '']
    (ROOT / 'reports/CP3_review.md').write_text('\n'.join(lines))
    print(json.dumps({'review_report': 'reports/CP3_review.md', 'confirmation_GT_counts': dict(categories),
                      'max_planning_ms': tail['max_ms'], 'native_or_parameters_changed': False}))


if __name__ == '__main__':
    main()
