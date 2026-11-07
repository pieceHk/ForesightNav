"""PBSP adapter for the author's ordered HM3D-MFMON benchmark."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from experiments.mfmon_protocol import World, aggregate, atomic, category_name, episode_metrics
from pbsp.types import EpisodeIdentity


def load_dataset(path, config):
    raw = (Path(path)/'dataset_manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == config.manifest_sha256
    data = json.loads(raw)
    assert len(data['entries']) == 927
    from experiments.mfmon_protocol import sha
    for row in data['input_files']:
        assert sha(row['path']) == row['sha256']
    return data


def episode_identities(manifest, digest):
    return [EpisodeIdentity(dataset_manifest_sha256=digest,
             scene_id=Path(e['scene_id']).name.split('.')[0], runtime_episode_id=str(e['global_id']),
             source_episode_ids=tuple(str(e['global_id'])+':'+str(i) for i in range(3)),
             ordered_categories=tuple(category_name(c) for c in e['goals'])) for e in manifest['entries']]


def project_detection(mask, observation):
    """Record the policy's confirmed mask in world coordinates, for evaluation."""
    import numpy as np
    import quaternion
    camera = observation.task_observations.get('_mfmon_camera')
    if camera is None: return None
    mask = np.asarray(mask)
    z = np.asarray(observation.depth)
    if mask.ndim != 2 or mask.shape != z.shape or not np.any(mask): return None
    # The native PBSP sensor returns raw metric depth beyond 5 m. Preserve
    # the confirmed measurement; terminal geometry is evaluated separately.
    y, x = np.nonzero((mask > 0) & np.isfinite(z) & (z > .01))
    if not len(x): return None
    f = z.shape[1]/(2*np.tan(np.deg2rad(camera['hfov']/2)))
    points = np.stack(((x-z.shape[1]/2)*z[y,x]/f,
                       -(y-z.shape[0]/2)*z[y,x]/f, -z[y,x]), axis=1)
    rotation = quaternion.as_rotation_matrix(np.quaternion(*camera['rotation']))
    return (np.median(points @ rotation.T, axis=0)+camera['position']).tolist()


def build_services(request, stack):
    import numpy as np
    from functools import partial
    from pbsp.adapters.habitat_config import get_config
    from pbsp.adapters.habitat import HabitatEnvironment, HabitatObservationSettings
    from pbsp.adapters.goat import load_categories
    from pbsp.adapters.detic import create_segmentation
    from pbsp.adapters.lagmemo_agent import LagMemoAgent
    from pbsp.runtime.mapping_client import MappingClient
    from pbsp.runtime.navigation_worker import NavigationServices, enable_planning_maps
    from lagmemo.core.interfaces import DiscreteNavigationAction, Observations
    from experiments.metrics import summarize_frame_times

    raw = (request.dataset/'dataset_manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == request.planning_settings.provenance['dataset_manifest_sha256']
    manifest = json.loads(raw)
    entries = [r for r in manifest['entries'] if str(r['global_id']) in request.episode_ids]
    assert len(entries) == len(request.episode_ids)
    current_only = manifest['goal_information'] == 'current_goal_only'
    cfr = set(manifest['cfr_ids'])
    world = World(step=.25,depth_min=.5)
    stack.callback(world.close)
    cfg = get_config(str(request.resources.habitat_config), str(request.resources.agent_config))
    cfg.GROUND_TRUTH_SEMANTICS = 0
    cfg.NUM_ENVIRONMENTS, cfg.PRINT_IMAGES, cfg.VISUALIZE = 1, 0, 0
    cfg.AGENT.max_steps, cfg.AGENT.radius = 1500, .1
    cfg.ENVIRONMENT.max_num_sub_task_episodes = 3
    cfg.EXP_FIRST = False
    cfg.DUMP_LOCATION, cfg.EXP_NAME = str(request.output), 'hm3d_mfmon'

    class Environment(HabitatEnvironment):
        current_goal_only = current_only
        def reset(self):
            self.entry = entries[self.index]; self.index += 1
            self.raw = world.reset(self.entry)
            self.habitat_env.sim = world.sim
            self.current_episode = self.habitat_env.current_episode = SimpleNamespace(
                episode_id=str(self.entry['global_id']), scene_id=self.entry['scene_id'],
                start_position=self.entry['start_position'], start_rotation=self.entry['start_rotation'])
            self.last_capture = None
            self.agent._mfmon_detection_position = None
            self._last_obs = self._preprocess_obs(self.raw)
            return self._last_obs

        @property
        def task_index(self): return world.stage
        @property
        def episode_over(self): return world.done
        @property
        def action_end_pose(self): return self._last_obs.globalpose

        def policy_categories(self):
            return [category_name(c) if (not current_only or i <= world.stage) else '__unrevealed__'
                    for i, c in enumerate(self.entry['goals'])]

        def policy_identity(self, identity):
            return dict(identity, ordered_categories=self.policy_categories())

        def activate_policy(self, provider):
            provider._tasks = [[c, 'object'] for c in self.policy_categories()]
            provider.invalidate_cache()

        def _preprocess_obs(self, raw):
            tasks = [dict(category=c, description='', image=None, semantic_id=(self.categories.index(c)+1
                     if c != '__unrevealed__' else 0), type='objectnav') for c in self.policy_categories()]
            obs = self.observation_factory(rgb=raw['rgb'], depth=raw['depth'][:,:,0],
                    compass=raw['compass'], gps=self._preprocess_xy(raw['gps']),
                    globalpose=raw['globalpose'], camera_pose=None, third_person_image=None,
                    task_observations={'tasks':tasks, '_mfmon_camera':dict(position=raw['camera_position'],
                        rotation=raw['camera_rotation'], hfov=world.hfov)})
            return self._preprocess_semantic(obs, None)

        def apply_action(self, action, info=None, **kwargs):
            before = len(world.stages)
            name = action.name.lower()
            md = self.agent.last_action_metadata
            self.raw = world.act(name, called_found=(name == 'stop' and md.get('stop_reason') == 'verified_target'),
                                 detection=md.get('mfmon_detection_position'))
            if len(world.stages) > before: self.last_capture = world.stages[before]
            self._last_obs = self._preprocess_obs(self.raw)

    facade = SimpleNamespace(gpu_device=0, number_of_episodes=len(entries), sim=None)
    env = Environment(facade, settings=HabitatObservationSettings(.5,5.,'detic'),
          categories=load_categories(request.resources.categories), observation_factory=Observations,
          action_ids=DiscreteNavigationAction,
          segmentation_factory=partial(create_segmentation, gpu_id=0,
            config_file=str(request.resources.detic_config), checkpoint_file=str(request.resources.detic_checkpoint),
            category_frequency=request.resources.detic_category_frequency,
            clip_checkpoint=request.resources.matching.clip))
    env.index = 0

    assert not current_only, "This comparison explicitly uses the complete sequence"
    provider = MappingClient(socket_path=str(request.socket_path),top_k=request.top_k,
        connect_timeout=120., live_mode=True, strict_errors=True,
        task_reader=lambda scene,episode: [[c,'object'] for c in env.policy_categories()])
    stack.callback(provider.close)
    agent = LagMemoAgent(config=cfg, seem_cfg=dict(conf_path=str(request.resources.seem_config),
        ckpt_path=str(request.resources.seem_checkpoint),tokenizer_path=str(request.resources.seem_tokenizer)),
        clip_cfg=str(request.resources.mobileclip_checkpoint), settings=request.settings,
        matching_resources=request.resources.matching)
    env.agent = agent
    stack.callback(agent.close)
    agent.top_down_map_reader = lambda: None
    enable_planning_maps(agent,scsp_enabled=True,print_images=False)

    class Protocol:
        metric_keys=('success','spl'); distance_key=None
        capture_on_episode_end=True; allows_terminal_prefix=True
        def capture(self, evaluator, *, steps, stop_reason, frame_times_ms):
            assert env.last_capture is not None
            return dict(env.last_capture)
        def metric_summary(self, records, name):
            vals=[r[name] for r in records]
            from statistics import median
            return dict(mean=sum(vals)/len(vals),median=median(vals))
        def summarize_episode(self, records, **kwargs):
            assert records == world.stages
            entry=dict(env.entry, cfr=env.entry['global_id'] in cfr,
                       stages=records, total_steps=world.steps,
                       goal_information=manifest['goal_information'])
            path=request.output/'episodes'/('episode_%03d.json'%entry['global_id'])
            assert not path.exists()
            atomic(path,entry)
            return {'mfmon_metrics':episode_metrics(records)}
        def summarize_records(self, records, **kwargs): return {}
    return NavigationServices(env,env,agent,provider,DiscreteNavigationAction,cfg,len(entries),Protocol())


def aggregate_run(path, require_complete=True):
    path=Path(path)
    entries=[json.loads(p.read_text()) for p in sorted((path/'episodes').glob('episode_*.json'))]
    if require_complete:
        expected=json.loads((path/'launch.json').read_text())['identities']
        assert {r['global_id'] for r in entries} == {int(r['runtime_episode_id']) for r in expected}
    return {**aggregate(entries), 'counts':dict(completed_episodes=len(entries),
            completed_subtasks=sum(len(r['stages']) for r in entries))}
