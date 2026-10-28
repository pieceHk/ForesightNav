"""Connect an existing official ObjectNav split and freeze its experiment inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from experiments.objectnav_data import audit_split
from experiments.prepare_data import connect

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {'hm3dv1': (2000, 20), 'hm3dv2': (1000, 36)}


def _save(path, text):
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError('Prepared ObjectNav input differs: ' + str(path))
    else:
        with path.open('x') as stream:
            stream.write(text)


def prepare(dataset, source, scenes, scene_dataset, destination, *, root=ROOT, expected=None):
    episodes, count = SPLITS[dataset] if expected is None else expected
    manifest = audit_split(source, expected_episodes=episodes, expected_scenes=count)
    source, scenes, destination = Path(source).resolve(strict=True), Path(scenes).resolve(strict=True), Path(destination).resolve()
    scene_config = Path(scene_dataset)
    if scene_config.is_absolute() or '..' in scene_config.parts:
        raise ValueError('Scene configuration must be relative to the scene directory')
    if not (scenes/scene_config).is_file():
        raise FileNotFoundError(scenes/scene_config)
    relative = Path('episodes/objectnav')/dataset
    target = destination/relative
    if target.is_symlink():
        raise ValueError('Prepared ObjectNav metadata requires its own directory')
    target.mkdir(parents=True, exist_ok=True)
    connect(source, target/'val.json.gz')
    connect(source.parent/'content', target/'content')
    scene_relative = Path('scenes')/dataset
    connect(scenes, destination/scene_relative)
    text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n'
    _save(target/'dataset_manifest.json', text)
    config = dict(format='objectnav', name=dataset+'_objectnav_val', split='val', root=str(relative),
        manifest='dataset_manifest.json', manifest_sha256=hashlib.sha256(text.encode()).hexdigest(),
        expected_episodes=episodes, expected_scenes=count,
        scene_directory=str(scene_relative), scene_dataset=str(scene_config))
    _save(target/'dataset.yaml', yaml.safe_dump(config, sort_keys=False))
    _save(target/'evaluation.yaml', yaml.safe_dump(dict(max_steps_per_task=500, metrics=['SR', 'SPL'],
                                                      record_decisions=True, visualize=False), sort_keys=False))
    experiment = dict(name=dataset+'_objectnav_full_pbsp', method=str(Path(root)/'configs/methods/full_pbsp.yaml'),
                      dataset='dataset.yaml', evaluation='evaluation.yaml', seed=1)
    _save(target/'experiment.yaml', yaml.safe_dump(experiment, sort_keys=False))
    return target/'experiment.yaml'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=SPLITS, required=True)
    parser.add_argument('--source', type=Path, required=True, help='Official val.json.gz')
    parser.add_argument('--scenes', type=Path, required=True, help='Parent directory of original scene paths')
    parser.add_argument('--scene-dataset', required=True, help='Scene configuration relative to --scenes')
    parser.add_argument('--destination', type=Path, default=ROOT/'data')
    args = parser.parse_args(argv)
    experiment = prepare(args.dataset, args.source, args.scenes, args.scene_dataset, args.destination)
    print(json.dumps({'experiment': str(experiment), 'status': 'prepared'}, indent=2))


if __name__ == '__main__':
    main()
