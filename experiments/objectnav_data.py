"""Audit complete native ObjectNav splits and bind immutable source identities."""

import gzip
import hashlib
import json
import math
from pathlib import Path

from pbsp.adapters.objectnav import policy_category
from pbsp.types import EpisodeIdentity


def _read(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data)), hashlib.sha256(data).hexdigest()


def _vector(value, length, name):
    if not isinstance(value, list) or len(value) != length or any(
        type(item) not in (int, float) or not math.isfinite(item) for item in value
    ):
        raise ValueError('Invalid ' + name)
    return value


def audit_split(header, *, expected_episodes, expected_scenes):
    header = Path(header).resolve(strict=True)
    data, header_hash = _read(header)
    if data.get('episodes') != []:
        raise ValueError('Expected an empty split header with per-scene episode content')
    files = sorted((header.parent/'content').glob('*.json.gz'), key=lambda p: p.name.casefold())
    entries, sources, seen, scenes = [], [], set(), set()
    for path in files:
        data, checksum = _read(path)
        if not data.get('episodes') or not isinstance(data.get('goals_by_category'), dict):
            raise ValueError('Missing ObjectNav content or goal definitions: ' + str(path))
        file_scenes = set()
        for index, episode in enumerate(data['episodes']):
            scene = episode['scene_id']
            source_id = episode['episode_id']
            if (not isinstance(scene, str) or not scene.strip()
                    or type(source_id) not in (int, str) or not str(source_id).strip()):
                raise ValueError('Invalid source episode identity')
            source_id = str(source_id)
            if (scene, index) in seen:
                raise ValueError('Duplicate source episode identity')
            seen.add((scene, index))
            category = episode['object_category']
            policy_category(category)
            if not data['goals_by_category'].get(Path(scene).name + '_' + category):
                raise ValueError('Missing native ObjectNav goals')
            position = _vector(episode['start_position'], 3, 'start position')
            rotation = _vector(episode['start_rotation'], 4, 'start rotation')
            if not math.isclose(sum(item*item for item in rotation), 1., abs_tol=1e-4):
                raise ValueError('Start rotation is not a unit quaternion')
            distance = episode['info']['geodesic_distance']
            if type(distance) not in (int, float) or not math.isfinite(distance) or distance < 0:
                raise ValueError('Invalid source geodesic distance')
            scenes.add(scene)
            file_scenes.add(scene)
            entries.append(dict(global_index=len(entries), scene_id=scene,
                source_episode_id=str(index), raw_episode_id=source_id, content_index=index, category=category,
                start_position=position, start_rotation=rotation,
                source_geodesic_distance=distance, source_file=path.name))
        if len(file_scenes) != 1:
            raise ValueError('A content file contains multiple scenes')
        scene_id = Path(next(iter(file_scenes))).name.split('.')[0]
        if path.name != scene_id + '.json.gz':
            raise ValueError('ObjectNav scene content name differs from its scene')
        sources.append(dict(path='content/' + path.name, sha256=checksum, episodes=len(data['episodes'])))
    if len(entries) != expected_episodes or len(scenes) != expected_scenes or len(files) != expected_scenes:
        raise ValueError('Official split coverage mismatch: ' + repr((len(entries), len(scenes))))
    payload = dict(schema_version='pbsp_objectnav_dataset_v1', header='val.json.gz',
        header_sha256=header_hash, source_files=sources, episodes=len(entries), scenes=len(scenes), entries=entries)
    return payload


def load_dataset(directory, config):
    directory = Path(directory)
    raw = (directory/config.manifest).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config.manifest_sha256:
        raise ValueError('Native ObjectNav manifest checksum mismatch')
    manifest = json.loads(raw)
    current = audit_split(directory/'val.json.gz', expected_episodes=config.expected_episodes,
                          expected_scenes=config.expected_scenes)
    if current != manifest:
        raise ValueError('Native ObjectNav source differs from its frozen manifest')
    return manifest


def episode_identities(manifest, checksum):
    records, seen = [], set()
    for row in manifest['entries']:
        scene = Path(row['scene_id']).name.split('.')[0]
        runtime = str(row['content_index'])
        if (scene, runtime) in seen:
            raise ValueError('Ambiguous native ObjectNav runtime identity')
        seen.add((scene, runtime))
        records.append(EpisodeIdentity(dataset_manifest_sha256=checksum, scene_id=scene,
            runtime_episode_id=runtime, source_episode_ids=(row['source_episode_id'],),
            ordered_categories=(policy_category(row['category']),)))
    return records
