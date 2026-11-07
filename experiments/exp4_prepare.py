"""Freeze scene folds, native episode identities, configs and server baseline."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess

import yaml

from pbsp.runtime.config import load_experiment
from pbsp.runtime.plan import prepare_plan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/exp4'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    runtime = yaml.safe_load((ROOT / 'runtime.local.yaml').read_text())
    runtime['paths']['outputs'] = str(OUT / 'runs')
    runtime['gpus'] = {'mapping': 2, 'navigation': 3}
    (ROOT / 'runtime.exp4.local.yaml').write_text(yaml.safe_dump(runtime, sort_keys=False))
    plans = {}
    for n in (2, 3):
        config = load_experiment(ROOT / f'configs/experiments/hm3d_{n}mon.yaml', ROOT / 'runtime.exp4.local.yaml')
        plans[n] = prepare_plan(config, root=ROOT, source_commit='7c7e9f4')
        write(OUT / f'meta/{n}mon-identities.json', [asdict(i) for i in plans[n].identities])
    scenes = sorted(set(i.scene_id for i in plans[3].identities))
    assert len(scenes) == 20
    assert set(scenes) == set(i.scene_id for i in plans[2].identities)
    rng = random.Random(20261107)
    rng.shuffle(scenes)
    folds = {'seed': 20261107, 'algorithm': 'Python random.Random.shuffle on sorted scene IDs',
             'folds': [scenes[i:i + 4] for i in range(0, 20, 4)], 'first_collect_folds': [0, 1, 2]}
    path = OUT / 'meta/folds.json'
    if path.exists():
        assert json.loads(path.read_text()) == folds
    else:
        write(path, folds)
    for scene in scenes:
        assert sum(i.scene_id == scene for i in plans[2].identities) == 30
        assert sum(i.scene_id == scene for i in plans[3].identities) == 20
    # One deterministic episode per scene; this smoke list is held out from
    # calibration, which uses only the distinct 2-MON episode manifest.
    smoke = [next(asdict(i) for i in plans[3].identities if i.scene_id == scene)
             for scene in sorted(scenes)]
    write(OUT / 'meta/smoke-20-identities.json', smoke)
    base = yaml.safe_load((ROOT / 'configs/methods/full_pbsp.yaml').read_text())
    for variant in ('full', 'nearest', 'myopic', 'suffix_only', 'calibration'):
        method = json.loads(json.dumps(base))
        method['name'] = f'exp4_{variant}'
        method['suffix'].update(blp_variant='full' if variant == 'calibration' else variant,
                                blp_shadow=variant == 'calibration',
                                blp_calibration='uncalibrated' if variant == 'calibration' else 'results/exp4/meta/calibration.json')
        if variant != 'calibration':
            method['suffix']['mode'] = 'blp_eq13'
        (ROOT / f'configs/methods/exp4_{variant}.yaml').write_text(yaml.safe_dump(method, sort_keys=False))
        experiment = {'name': f'exp4_{variant}', 'method': f'../methods/exp4_{variant}.yaml',
                       'dataset': f'../datasets/hm3d_{2 if variant == "calibration" else 3}mon.yaml',
                       'evaluation': '../evaluation/standard.yaml', 'seed': 1}
        (ROOT / f'configs/experiments/exp4_{variant}.yaml').write_text(yaml.safe_dump(experiment, sort_keys=False))
    write(OUT / 'meta/preparation.json', {'server_baseline_commit': '7c7e9f4',
          'original_head': 'ef9ecd7b3674673791746c9a6a31c8e512ddeec7',
          'dataset_counts': {'2mon': 600, '3mon': 400},
          'calibration_episode_count_per_fold': 120,
          'gpu_indices': [2, 3], 'scene_fold_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    env = subprocess.check_output(['nvidia-smi'], text=True)
    (OUT / 'meta/env.txt').write_text(env)
    print(json.dumps(folds, indent=2))


if __name__ == '__main__':
    main()
