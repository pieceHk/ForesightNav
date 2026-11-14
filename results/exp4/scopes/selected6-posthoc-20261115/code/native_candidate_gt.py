"""Offline candidate labels from the pinned GOAT measures on a CPU navmesh.

No simulator, renderer, model, policy or training parameter is constructed.
The native dataset parser, HabitatSim geodesic wrapper and GoatEvaluator are
reused; the small simulator facade supplies only the loaded PathFinder.
"""
import gzip
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NativeCandidateGT:
    def __init__(self, run):
        self.run = Path(run)
        self.plan = json.loads((self.run / 'launch.json').read_text())
        from pbsp.runtime.resources import ResourcePaths
        from pbsp.runtime.source_provenance import habitat_source, verify_source
        self.resources = ResourcePaths(**{k: Path(v) for k, v in self.plan['resources'].items()})
        nav = self.resources.navigation_resources()
        snapshot = json.loads((self.run / 'source_snapshot.json').read_text())
        config_paths = ('configs/habitat.yaml', 'configs/navigation.yaml',
                        'configs/environments/navigation/linux-64.lock.json')
        dirty = {line[3:].split(' -> ')[-1] for line in snapshot['status']}
        for relative in config_paths:
            if relative in dirty:
                raise ValueError('Recover the recorded config snapshot before offline GT: ' + relative)
            recorded = subprocess.run(['git', '-C', str(self.resources.root), 'show',
                self.plan['source_commit'] + ':' + relative], check=True, capture_output=True).stdout
            if hashlib.sha256(recorded).hexdigest() != sha256(self.resources.root / relative):
                raise ValueError('Current config differs from the recorded run: ' + relative)
        lock = json.loads((self.resources.root / config_paths[-1]).read_text())
        self.source_proof = verify_source(self.resources.habitat, habitat_source(lock))
        sys.path[:0] = [str(self.resources.habitat / 'habitat-lab'),
                       str(self.resources.habitat / 'habitat-baselines')]
        import numpy as np
        import habitat_sim
        from habitat.datasets.goat.goat_dataset import GoatDatasetV1
        from habitat.tasks.nav.nav import GOATDistanceToSubGoal, GOATSubTaskSuccess
        from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
        from pbsp.adapters.habitat_config import get_config
        from experiments.evaluator import GoatEvaluator
        self.np, self.PathFinder, self.Dataset = np, habitat_sim.PathFinder, GoatDatasetV1
        self.Distance, self.Success, self.Evaluator = GOATDistanceToSubGoal, GOATSubTaskSuccess, GoatEvaluator
        self.geodesic = HabitatSim.geodesic_distance
        expected_source = self.resources.habitat / 'habitat-lab/habitat/tasks/nav/nav.py'
        if sha256(inspect.getsourcefile(self.Distance)) != sha256(expected_source):
            raise ValueError('Imported GOAT measures differ from the pinned source')
        self.config = get_config(str(nav.habitat_config), str(nav.agent_config))
        from omegaconf import OmegaConf
        measures = self.config.habitat.task.measurements
        self.distance_config = measures['goat_distance_to_sub-goal']
        self.success_config = measures['goat_sub-task_success']
        self.measure_config = {
            'distance': OmegaConf.to_container(self.distance_config, resolve=True),
            'success': OmegaConf.to_container(self.success_config, resolve=True),
        }
        dataset = self.plan['configuration']['dataset']
        self.dataset_root = self.resources.data / dataset['root']
        manifest_path = self.dataset_root / dataset['manifest']
        if sha256(manifest_path) != dataset['manifest_sha256']:
            raise ValueError('Dataset manifest changed since the native run')
        self.manifest = json.loads(manifest_path.read_text())
        self.identities = {(str(r['scene_id']), str(r['runtime_episode_id'])): r
                           for r in self.plan['identities']}
        self.scene_cache, self.label_cache = {}, {}
        self.queries = self.cache_hits = 0

    def scene(self, scene):
        if scene in self.scene_cache:
            return self.scene_cache[scene]
        rows = [r for r in self.manifest['scene_summary'] if r['scene_id'] == scene]
        if len(rows) != 1:
            raise ValueError('Scene is absent or duplicated in the source manifest')
        content = self.dataset_root / 'content' / (scene + '.json.gz')
        content_hash = sha256(content)
        if content_hash != rows[0]['output_sha256']:
            raise ValueError('Scene episode/goal content changed since the native run')
        dataset = self.Dataset()
        dataset.episodes = []
        dataset.from_json(gzip.decompress(content.read_bytes()).decode(),
                          scenes_dir=str(self.resources.scene_directory))
        episodes = {str(e.episode_id): e for e in dataset.episodes}
        if len(episodes) != len(dataset.episodes) or not episodes:
            raise ValueError('Native dataset parser produced invalid episode identities')
        aliases = self.manifest['runtime_category_aliases']
        for (claimed_scene, episode), identity in self.identities.items():
            if claimed_scene != scene:
                continue
            observed = episodes[episode]
            categories = [aliases.get(t[0], t[0]) for t in observed.tasks]
            if categories != identity['ordered_categories']:
                raise ValueError('Native parsed task order differs from the launch identity')
        mesh = Path(next(iter(episodes.values())).scene_id).with_suffix('.navmesh')
        pathfinder = self.PathFinder()
        if not pathfinder.load_nav_mesh(str(mesh)):
            raise ValueError('Cannot load the native scene navmesh: ' + str(mesh))
        facade = SimpleNamespace(pathfinder=pathfinder)
        facade.geodesic_distance = self.geodesic.__get__(facade, type(facade))
        distance = self.Distance(sim=facade, config=self.distance_config)
        success = self.Success(sim=facade, config=self.success_config)
        env = SimpleNamespace(sim=facade, current_episode=None,
            task=SimpleNamespace(measurements=SimpleNamespace(measures={
                'goat_distance_to_sub-goal': distance, 'goat_sub-task_success': success})))
        value = {'episodes': episodes, 'env': env, 'evaluator': self.Evaluator(env),
                 'content_sha256': content_hash, 'navmesh': str(mesh),
                 'navmesh_sha256': sha256(mesh)}
        self.scene_cache[scene] = value
        return value

    def __call__(self, scene, episode, task, position):
        scene, episode, task = str(scene), str(episode), int(task)
        if (scene, episode) not in self.identities:
            raise ValueError('Offline GT requested an episode outside this native run')
        point = self.np.asarray(position, dtype=self.np.float32)
        if point.shape != (3,) or not self.np.all(self.np.isfinite(point)):
            raise ValueError('Offline GT requires a finite XYZ target')
        key = (scene, episode, task, *map(float, point))
        self.queries += 1
        if key in self.label_cache:
            self.cache_hits += 1
            return dict(self.label_cache[key])
        bundle = self.scene(scene)
        current = bundle['episodes'][episode]
        if not 0 <= task < len(current.tasks):
            raise ValueError('Offline GT task index is outside the episode')
        bundle['env'].current_episode = current
        value = bundle['evaluator'].label_candidate(point, task)
        self.label_cache[key] = value
        return dict(value)

    def provenance(self):
        return {'method': 'native GOAT measures and geodesic wrapper on CPU PathFinder',
                'habitat_source': self.source_proof, 'measurements': self.measure_config,
                'dataset_manifest_sha256': self.plan['configuration']['dataset']['manifest_sha256'],
                'queries': self.queries, 'cache_hits': self.cache_hits,
                'distinct_positions': len(self.label_cache),
                'scenes': {scene: {k: value[k] for k in ('content_sha256', 'navmesh', 'navmesh_sha256')}
                           for scene, value in self.scene_cache.items()}}

    def verify_events(self, events):
        counts = {'candidate_verifications': 0, 'recovery_verifications': 0}
        max_error = 0.
        for event in events:
            kind = event['event']
            if kind not in ('verification_final', 'noncandidate_verification'):
                continue
            expected = event['gt_diagnostic']
            if expected.get('criterion') != 'pinned_GOAT_measures_at_candidate_position':
                raise ValueError('Recorded candidate GT uses a different criterion')
            actual = self(event['scene_id'], event['episode_id'], event['k'], event['position'])
            if actual['y'] != expected['y'] or (kind == 'verification_final' and event['y'] != expected['y']):
                raise ValueError('Offline and recorded native candidate GT disagree')
            a, b = actual['distance_to_goal_m'], expected['distance_to_goal_m']
            if a is None or b is None:
                if a != b:
                    raise ValueError('Offline and recorded goal reachability disagree')
            else:
                error = abs(a - b)
                max_error = max(error, max_error)
                if not math.isclose(a, b, abs_tol=1e-5, rel_tol=0.):
                    raise ValueError('Offline and recorded native geodesic distances disagree')
            counts['candidate_verifications' if kind == 'verification_final' else 'recovery_verifications'] += 1
        return {**counts, 'max_distance_error_m': max_error}
