"""Disclose the frozen benchmark provenance without altering its protocol."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    scope = json.loads((ROOT / 'meta/dataset-scientific-scope.json').read_text())
    height = json.loads((ROOT / 'diagnostics/CP3/dataset-height-availability.json').read_text())
    navmesh = json.loads((ROOT / 'diagnostics/CP3/dataset-navmesh-availability.json').read_text())
    ledger = json.loads((ROOT / 'meta/evaluation-smoke-runs.json').read_text())
    launch = json.loads((Path(ledger['jobs'][0]['run_directory']) / 'launch.json').read_text())
    config = launch['configuration']['dataset']
    dataset = Path(launch['resources']['data']) / config['root']
    assert scope['manifest_sha256'] == height['data_manifest_sha256'] == navmesh['manifest_sha256'] == config['manifest_sha256']
    assert scope['source_trace_sha256'] == config['source_trace_sha256']
    copies = [('dataset_manifest.json', '3mon-manifest.json', scope['manifest_sha256']),
              ('source_trace.json', '3mon-source-trace.json', scope['source_trace_sha256'])]
    destination = ROOT / 'diagnostics/dataset'
    destination.mkdir(parents=True, exist_ok=True)
    for original, copied, expected in copies:
        assert digest(dataset / original) == expected
        shutil.copyfile(dataset / original, destination / copied)
        assert digest(destination / copied) == expected
    historical = scope['historical_source_outcome_counts']
    lines = ['# Exp4固定任务池的来源与适用范围', '',
             f'实际数据集为 `{scope["dataset_name"]}`，包含20个场景、400条3-MON、1200个不重复的原始ObjectNav任务。原始manifest明确记录 `not_independent_held_out=true`、`not_outcome_blind=true` 和 `selection_type=historical paired-outcome-calibrated`。', '',
             '任务池经过历史1-MON基线与候选方法的配对结果筛选。这些历史标签不代表本次Full、Nearest、Belief-myopic或Suffix-only的成绩。当前实验继续采用已冻结清单及参数，不根据本次3-MON结果删选、重排回合或拟合参数。', '',
             '因此，结果适用于这个固定筛选池内的受控消融；不能据此声称独立盲测、未见场景或未见底层任务的泛化性能。交叉拟合排除了对应标定组，但不能消除任务池既有的选择偏差。标定/评估共享来源的审计另见[既有记录](../meta/calibration-overlap-audit.json)。', '',
             '| 历史源任务配对结局 | 数量 |', '| --- | --- |']
    lines += [f'| {name} | {historical[name]} |' for name in sorted(historical)]
    lines += ['', '这些计数仅用于说明数据来源，没有用于本次belief拟合或在线决策。', '',
              '## 起点、高度与navmesh静态检查', '',
              '一条MON只执行第一个源任务的起始姿态；后续源任务的起点只作来源记录，不会重置agent。后续阶段应从真实转换位置计算路径和SPL。目标按类别匹配；高度检查覆盖该类别的全部记录viewpoint。', '',
              '| 检查 | 固定20条 | 完整400条 |', '| --- | --- | --- |']
    rows = [('三个目标都存在距初始高度≤0.5m的viewpoint', height['smoke20']['all_targets_have_height_compatible_viewpoint_0.5m'], height['full400']['all_targets_have_height_compatible_viewpoint_0.5m']),
            ('三个目标从初始点到viewpoint均有有限navmesh测地距离', navmesh['smoke20']['episodes_all_targets_have_finite_initial_geodesic'], navmesh['full400']['episodes_all_targets_have_finite_initial_geodesic']),
            ('三个目标均有高度相近且测地距离有限的viewpoint', navmesh['smoke20']['episodes_all_targets_have_finite_geodesic_to_height_compatible_viewpoint'], navmesh['full400']['episodes_all_targets_have_finite_geodesic_to_height_compatible_viewpoint'])]
    lines += [f'| {label} | {small} / 20 | {large} / 400 |' for label, small, large in rows]
    lines += ['', '高度相近只是代理条件，不能证明同楼层或策略可达；navmesh连通也不能证明当前2D导航策略能走通楼梯。以上距离均从回合初始位置计算，不能替代后续阶段真实起点的指标，也不足以解释某一回合的成败。未找到有限路径的项目保留原始记录，未剔除或修补。', '',
              '[高度逐条检查](../diagnostics/CP3/dataset-height-availability.json)、[CPU原生navmesh逐条检查](../diagnostics/CP3/dataset-navmesh-availability.json)。后者复用固定版本的HabitatSim测地距离与PathFinder，不创建renderer、模型或策略，不把GT反馈给规划或拟合。', '',
              '## 可核对的原始文件', '',
              f'- [原始manifest副本](../diagnostics/dataset/3mon-manifest.json)，SHA256 `{scope["manifest_sha256"]}`。',
              f'- [原始source trace副本](../diagnostics/dataset/3mon-source-trace.json)，SHA256 `{scope["source_trace_sha256"]}`。',
              '- [来源汇总](../meta/dataset-scientific-scope.json)和[固定400条身份](../meta/3mon-identities.json)。', '',
              '本次新增审计与说明没有修改数据、原生执行源码或冻结参数。CP3报告将同时提交这些限制，进入主评测仍需按既定CP3检查点审阅。', '']
    (ROOT / 'reports/dataset_scope.md').write_text('\n'.join(lines))
    print(json.dumps({'report': 'reports/dataset_scope.md', 'report_sha256': digest(ROOT / 'reports/dataset_scope.md'),
                      'native_inputs_or_parameters_changed': False, 'historical_counts': historical}))


if __name__ == '__main__':
    main()
