"""Prepare and inspect all four main stages; never launch or accept a gate."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import yaml
from evaluate_sequence import build_jobs, command, expected_identities, validate_gate, GateError
from experiments.exp4_collect import source_fingerprint

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
STAGES = ('full', 'nearest', 'myopic', 'suffix_only')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    cp2b = load(ROOT / 'meta/CP2b_acceptance.json')
    assert source_fingerprint() == cp2b['source_fingerprint']
    assert digest(ROOT / 'meta/calibration.json') == cp2b['parameter_sha256']
    identities = expected_identities('full')
    canonical_method, experiment_protocol, plans = None, None, {}
    for stage in STAGES:
        experiment_path = REPO / 'configs/experiments' / ('exp4_' + stage + '.yaml')
        experiment = yaml.safe_load(experiment_path.read_text())
        method_path = (experiment_path.parent / experiment['method']).resolve()
        method = yaml.safe_load(method_path.read_text())
        assert method['suffix']['blp_variant'] == stage
        shared_method = deepcopy(method)
        shared_method.pop('name', None)
        del shared_method['suffix']['blp_variant']
        shared_experiment = deepcopy(experiment)
        del shared_experiment['name'], shared_experiment['method']
        if canonical_method is None:
            canonical_method, experiment_protocol = shared_method, shared_experiment
        assert shared_method == canonical_method, 'Non-variant method setting differs: ' + stage
        assert shared_experiment == experiment_protocol, 'Data/evaluation/seed differs: ' + stage
        jobs = build_jobs(stage, identities)
        plans[stage] = {'episodes': sum(len(job['identities']) for job in jobs), 'scenes': len(jobs),
                        'episodes_per_scene': sorted({len(job['identities']) for job in jobs}),
                        'experiment_sha256': digest(experiment_path), 'method_sha256': digest(method_path),
                        'first_scene_argv_example': command(stage, jobs[0]['scene'], jobs[0]['identities'], 2, 'a3'),
                        'ledger_exists': (ROOT / f'meta/evaluation-{stage}-runs.json').exists()}
    gate_ready, gate_reason = True, None
    try:
        validate_gate('full')
    except GateError as error:
        gate_ready, gate_reason = False, str(error)
    gpu_rows = [line for line in subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,name,memory.total,memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()
                if line.split(',')[0].strip() in ('2', '3')]
    assert len(gpu_rows) == 2
    smoke = load(ROOT / 'meta/evaluation-smoke-runs.json')
    seen, logical, allocated = set(), 0, 0
    for job in smoke['jobs']:
        for directory, _, filenames in os.walk(job['run_directory'], followlinks=False):
            for filename in filenames:
                path = Path(directory) / filename
                stat = path.lstat()
                if path.is_symlink() or (stat.st_dev, stat.st_ino) in seen:
                    continue
                seen.add((stat.st_dev, stat.st_ino))
                logical += stat.st_size
                allocated += stat.st_blocks * 512
    disk = os.statvfs(ROOT)
    evidence = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'plan_prepared': True,
                'launches_workers': False, 'creates_acceptance': False,
                'source_sha256': digest(Path(__file__)), 'parameter_sha256': cp2b['parameter_sha256'],
                'native_source_files': len(cp2b['source_fingerprint']), 'native_source_unchanged': True,
                'identity_sha256': digest(ROOT / 'meta/3mon-identities.json'), 'stages': plans,
                'same_data_evaluation_and_seed': True, 'only_method_variant_and_name_differ': True,
                'gpu2_3_observation': gpu_rows, 'gpu_units': 'index, name, total_MiB, free_MiB',
                'filesystem_available_bytes': disk.f_bavail * disk.f_frsize,
                'smoke_owned_file_logical_bytes': logical, 'smoke_owned_file_allocated_bytes': allocated,
                'linear_four_stage_owned_storage_estimate_bytes': allocated * 80,
                'estimate_limitation': 'Linear planning estimate from twenty smoke runs; not a bound on later episodes or external outputs. Recheck GPU and disk at actual launch.',
                'full_gate_ready': gate_ready, 'full_gate_reason': gate_reason,
                'long_observer_seconds': 7 * 24 * 3600,
                'native_launch_environment': 'Remove inherited CUDA_VISIBLE_DEVICES= from CPU transport before spawning native controller; controller assigns physical GPU2/3.',
                'recovery_rule': 'Existing ledger or matching live PID/start/boot must be inspected; observation timeout never authorizes a duplicate native launch.'}
    assert source_fingerprint() == cp2b['source_fingerprint']
    assert digest(ROOT / 'meta/calibration.json') == cp2b['parameter_sha256']
    (ROOT / 'diagnostics/CP3/main-preflight.json').write_text(json.dumps(evidence, indent=2) + '\n')
    lines = ['# Exp4四变体全量执行准备', '',
             '此检查不创建CP3接受记录，不启动原生任务。当前Full门禁状态：' + ('已满足。' if gate_ready else '未满足，' + gate_reason + '。'), '',
             '已在实际服务器核对四组均为固定20场景、每场景20条，共400条3-MON；数据、评估和seed=1一致，方法设置仅变体和名称不同。执行顺序为Full→Nearest→Belief-myopic→Suffix-only。Full完成后生成并汇报CP4，再自动继续其余三组。', '',
             '当前GPU观测（卡号、型号、总显存MiB、空闲MiB）：', '']
    lines += ['- `' + row + '`' for row in gpu_rows]
    lines += ['', f'当前文件系统可用{evidence["filesystem_available_bytes"] / 1024**3:.1f} GiB。20条冒烟run目录占用{allocated / 1024**2:.1f} MiB，按1600条线性外推约{allocated * 80 / 1024**3:.2f} GiB；这只是当前保存策略下的规划估计，不保证后续回合或目录外输出体积。实际启动时重新读取资源状态。', '',
              '原生控制器和只读观察器均应在服务器独立会话运行，观察窗口设为7天并在阶段结束时退出；以后阶段各自创建新观察会话。需要持续核对PID/start_ticks/boot_id、冻结源码和参数、候选绑定及动作目标记录；断开SSH后先重查原进程，不重复启动。', '',
              'SSH CPU后处理通道设置了空CUDA_VISIBLE_DEVICES；启动原生控制器时必须移除此继承值，由控制器指定物理GPU2/3。四阶段固定源、参数与清单保持一致，不根据本次3-MON结果删选或调参。', '',
              '[逐组指令和SHA](../diagnostics/CP3/main-preflight.json)、[CP3审阅材料](CP3_review.md)。', '']
    (ROOT / 'reports/CP3_main_execution_plan.md').write_text('\n'.join(lines))
    print(json.dumps({key: evidence[key] for key in ('plan_prepared', 'full_gate_ready', 'full_gate_reason',
        'same_data_evaluation_and_seed', 'only_method_variant_and_name_differ', 'gpu2_3_observation',
        'filesystem_available_bytes', 'linear_four_stage_owned_storage_estimate_bytes', 'launches_workers')}))


if __name__ == '__main__':
    main()
