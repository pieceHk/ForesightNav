"""HM3D-MFMON evaluation only. No target geometry is supplied to policies.

The target taxonomy and posterior metric definition follow the frozen LCN
release-v2.0 evaluation. Navigation code is supplied by each independent project.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path

DATA = Path('/data/3DONav_datasets/LifelongCrossNav/objectnav_hm3d_MFMON')
SCENES = Path('/data/3DONav_datasets/LifelongCrossNav/scene_datasets')
NAVMESH = dict(agent_height=1.5, agent_radius=.1, agent_max_climb=.3,
               agent_max_slope=60., cell_size=.03, cell_height=.05)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def manifest(data=DATA):
    rows, files = [], []
    for p in sorted(Path(data).glob('*.json.gz')):
        d = json.loads(gzip.decompress(p.read_bytes()))
        files.append(dict(path=str(p), sha256=sha(p)))
        for i, ep in enumerate(d['episodes']):
            assert len(ep['object_goals']) == 3
            rows.append(dict(global_id=len(rows), content_index=i,
                             source_file=str(p), scene_id=ep['scene_id'],
                             start_position=ep['start_position'],
                             start_rotation=ep['start_rotation'],
                             goals=ep['object_goals']))
    assert len(rows) == 927 and len({r['scene_id'] for r in rows}) == 36
    return dict(schema='hm3d_mfmon_comparison_v1', entries=rows, input_files=files)


def box_center(box):
    return box.center() if callable(box.center) else box.center


def box_sizes(box):
    sizes = getattr(box, "sizes", None)
    if sizes is None: sizes = box.size
    return sizes() if callable(sizes) else sizes


def category_name(value):
    return {'sofa': 'couch', 'tv_monitor': 'tv'}.get(value, value)


def episode_metrics(stages):
    assert 1 <= len(stages) <= 3
    assert all(s['success'] == 1 for s in stages[:-1])
    assert len(stages) == 3 or stages[-1]['success'] == 0 or stages[-1].get('budget_exhausted', False)
    success = int(len(stages) == 3 and all(s['success'] for s in stages))
    good = [s for s in stages if s['success']]
    for s in good:
        assert s['shortest_m'] is not None and math.isfinite(s['shortest_m'])
    ds = math.fsum(s['shortest_m'] for s in good)
    dr = math.fsum(s['traveled_m'] for s in good)
    eff = ds / max(ds, dr) if max(ds, dr) > 0 else 0.
    return dict(SR=success, SPL=success*eff, PR=len(good)/3., PPL=len(good)/3.*eff)


def aggregate(rows):
    def group(values):
        if not values:
            return None
        n = len(values)
        stages = [[r['stages'][k] for r in values if len(r['stages']) > k] for k in range(3)]
        return {**{k: math.fsum(episode_metrics(r['stages'])[k] for r in values)/n
                   for k in ('SR', 'SPL', 'PR', 'PPL')},
                'episodes': n, 'stages_reached': [len(s) for s in stages],
                'stage_successes': [sum(t['success'] for t in s) for s in stages],
                'conditional_SR': [sum(t['success'] for t in s)/len(s) if s else None for s in stages],
                'conditional_SPL': [math.fsum(t['spl'] for t in s)/len(s) if s else None for s in stages],
                'global_SR': [sum(t['success'] for t in s)/n for s in stages],
                'global_SPL': [math.fsum(t['spl'] for t in s)/n for s in stages]}
    assert len({r['global_id'] for r in rows}) == len(rows)
    return dict(all=group(rows), cross_floor_required=group([r for r in rows if r['cfr']]))


class World:
    """Small Habitat-Sim interface; preserves simulator state across targets."""
    def __init__(self, *, width=640, height=480, hfov=79., camera_height=.88,
                 step=.25, depth_max=5., depth_min=0., seed=1):
        self.width, self.height, self.hfov = width, height, hfov
        self.camera_height, self.step_size = camera_height, step
        self.depth_max, self.depth_min, self.seed = depth_max, depth_min, seed
        self.sim, self.scene = None, None

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def reset(self, entry):
        import habitat_sim as hs
        import numpy as np
        from habitat_sim.utils.common import quat_from_coeffs
        if self.scene != entry['scene_id']:
            self.close()
            c = hs.SimulatorConfiguration()
            c.scene_id = str(SCENES/entry['scene_id'])
            c.scene_dataset_config_file = str(SCENES/'hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json')
            c.gpu_device_id = 0
            c.allow_sliding = False
            a = hs.agent.AgentConfiguration()
            a.height, a.radius = 1.5, .1
            specs = []
            for name, st in [('rgb', hs.SensorType.COLOR), ('depth', hs.SensorType.DEPTH)]:
                s = hs.CameraSensorSpec()
                s.uuid, s.sensor_type, s.resolution = name, st, [self.height, self.width]
                s.position, s.hfov = [0., self.camera_height, 0.], self.hfov
                # Preserve the native sensor frustum. A 5 m render far plane
                # returns zeros for distant geometry, unlike metric depth.
                specs.append(s)
            a.sensor_specifications = specs
            a.action_space = {k: hs.agent.ActionSpec(k, hs.agent.ActuationSpec(amount=v))
                              for k, v in [('move_forward', self.step_size), ('turn_left', 30.),
                                           ('turn_right', 30.), ('look_up', 30.), ('look_down', 30.)]}
            self.sim = hs.Simulator(hs.Configuration(c, [a]))
            self.sim.seed(self.seed)
            nm = hs.NavMeshSettings(); nm.set_defaults()
            for k, v in NAVMESH.items(): setattr(nm, k, v)
            try:
                ok = self.sim.recompute_navmesh(self.sim.pathfinder, nm, include_static_objects=False)
            except TypeError:
                # Habitat-Sim 0.2.3 has no static-object argument.
                ok = self.sim.recompute_navmesh(self.sim.pathfinder, nm)
            assert ok
            self.scene = entry['scene_id']
            from experiments.mfmon_categories import TARGET_CATEGORIES, matches_target_category
            self.objects = {cat: [o for o in self.sim.semantic_scene.objects
                                  if o is not None and matches_target_category(cat, o.category.name())]
                            for cat in TARGET_CATEGORIES}
        self.entry = entry
        self.start = np.asarray(entry['start_position'], dtype=np.float32)
        self.rotation = quat_from_coeffs(entry['start_rotation'])
        self.sim.initialize_agent(0, hs.AgentState(self.start, self.rotation))
        d = json.loads(gzip.decompress(Path(entry['source_file']).read_bytes()))
        self.goal_data = d['goals_by_category']
        self.stage, self.steps, self.stages = 0, 0, []
        self.done = False
        self.stage_poses = [self.position().tolist()]
        self.shortest = self.shortest_distance(entry['goals'][0], self.position())
        return self.observation()

    def position(self):
        return self.sim.get_agent(0).get_state().position.copy()

    def observation(self):
        import numpy as np
        import quaternion
        from habitat_sim.utils.common import quat_rotate_vector
        o = self.sim.get_sensor_observations()
        s = self.sim.get_agent(0).get_state()
        sensor = s.sensor_states['depth']
        relative = quat_rotate_vector(self.rotation.inverse(), s.position-self.start)
        heading = quat_rotate_vector(s.rotation.inverse()*self.rotation, np.array([0., 0., -1.]))
        q = quaternion.as_float_array(sensor.rotation)
        # This installed PBSP HabitatSimDepthSensor returns unnormalized,
        # unclipped metric depth, including zero-valued invalid pixels.
        depth = np.asarray(o['depth']).astype(np.float32)
        return dict(rgb=o['rgb'][:, :, :3], depth=depth[:, :, None],
                    gps=np.asarray([-relative[2], relative[0]], dtype=np.float32),
                    compass=np.asarray([np.arctan2(heading[0], -heading[2])], dtype=np.float32),
                    globalpose=np.r_[quaternion.as_float_array(s.rotation), s.position],
                    agent_position=s.position.copy(), agent_rotation=quaternion.as_float_array(s.rotation),
                    camera_position=sensor.position.copy(), camera_rotation=q)

    def shortest_distance(self, cat, start):
        import habitat_sim as hs
        import numpy as np
        start = np.asarray(start, dtype=np.float32)
        pf = self.sim.pathfinder
        if not pf.is_navigable(start): start = pf.snap_point(start)
        if not np.isfinite(start).all(): return None
        best = float('inf')
        for obj in self.objects[cat]:
            views = next((v.get('view_points', []) for values in self.goal_data.values()
                          for v in values if v['object_id'] == obj.semantic_id), [])
            if views:
                p = hs.MultiGoalShortestPath()
                p.requested_start = start
                p.requested_ends = np.asarray([v['agent_state']['position'] for v in views], dtype=np.float32)
            else:
                end = pf.snap_point(np.asarray(box_center(obj.aabb), dtype=np.float32))
                if not np.isfinite(end).all(): continue
                p = hs.ShortestPath(); p.requested_start = start; p.requested_end = end
            pf.find_path(p)
            best = min(best, p.geodesic_distance)
        return best if math.isfinite(best) else None

    def distance(self, point):
        import numpy as np
        if point is None: return None
        p = np.asarray(point, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all(): return None
        # Native LCN: planar AABB distance plus a 1.1 m center-height gate.
        ds = []
        for o in self.objects[self.entry['goals'][self.stage]]:
            center = np.asarray(box_center(o.aabb))
            size = np.asarray(box_sizes(o.aabb))
            if abs(p[1]-center[1]) > 1.1: continue
            ds.append(float(np.linalg.norm(np.maximum(np.abs(p[[0,2]]-center[[0,2]])-size[[0,2]]/2., 0.))))
        return min(ds) if ds else None

    def act(self, name, *, called_found=False, detection=None):
        import numpy as np
        assert not self.done and self.steps < 1500
        self.steps += 1
        if name != 'stop': self.sim.step(name)
        self.stage_poses.append(self.position().tolist())
        obs = self.observation()
        if name == 'stop' or self.steps >= 1500:
            ad, dd = self.distance(obs['camera_position']), self.distance(detection)
            success = int(called_found and ad is not None and ad < 1.5 and dd is not None and dd < 1.5)
            dist = float(np.linalg.norm(np.diff(np.asarray(self.stage_poses), axis=0), axis=1).sum())
            ds = self.shortest
            spl = success*ds/max(ds, dist) if ds is not None and max(ds, dist) > 0 else 0.
            self.stages.append(dict(stage=self.stage, category=self.entry['goals'][self.stage],
                                    success=success, spl=spl, shortest_m=ds, traveled_m=dist,
                                    agent_distance_m=ad, detection_distance_m=dd,
                                    detection_position=None if detection is None else list(map(float, detection)),
                                    steps=len(self.stage_poses)-1, terminal_action=name,
                                    called_found=called_found, budget_exhausted=self.steps >= 1500, poses=self.stage_poses))
            self.done = not success or self.stage == 2 or self.steps >= 1500
            if not self.done:
                self.stage += 1
                self.stage_poses = [self.position().tolist()]
                self.shortest = self.shortest_distance(self.entry['goals'][self.stage], self.position())
        return obs
