"""Build an unapplied, source-bound passive trace patch outside frozen sources.

This script only writes proposed copies and a unified diff under results/code.
Do not apply the patch while the current calibration controller is running.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent / 'pending_action_trace'
REPO = Path(__file__).resolve().parents[3]
BASE_COMMIT = 'd94ca7d0f53b419a81027b50b45e26f9523e2f8f'


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f'Patch anchor not unique: {before[:100]!r}')
    return text.replace(before, after, 1)


HELPERS = '''

def action_target_descriptor(agent, task_idx, position):
    """Copy the selected target before STOP can advance the policy task."""
    point = None if position is None else np.asarray(position, dtype=float)
    if point is not None and (point.shape != (3,) or not np.all(np.isfinite(point))):
        raise ValueError('Action target must be a finite world XYZ point')
    override = getattr(agent, 'navmesh_override_goal', None)
    source = ('frontier' if point is None else 'navmesh_override'
              if override is not None and np.allclose(point, override, rtol=0., atol=1e-6)
              else 'query_candidate')
    return {'schema_version': 'blp_action_target_v1', 'task_idx': int(task_idx),
            'source': source, 'position': None if point is None else point.tolist()}


def bind_action_target(planning, metadata, task_idx, frame_index):
    """Read existing identities without registering candidates or changing rank."""
    target = dict(metadata['blp_action_target'])
    if target['task_idx'] != int(task_idx):
        raise ValueError('Action target belongs to a different pre-STOP task')
    recovery = metadata.get('blp_effective_recovery')
    if recovery is not None:
        target.update(source='navmesh_override', position=list(recovery['position']),
                      runner_recovery_mode=recovery['mode'])
    saved_id = target.get('candidate_id') if recovery is None else None
    target.update(frame_index=int(frame_index), candidate_id=None,
                  query_identity_status='not_query')
    if target['source'] == 'query_candidate':
        point = np.asarray(target['position'], dtype=float)
        registry = planning.blp_context.get('candidate_registry', {}).get(int(task_idx), [])
        if saved_id is not None:
            if not any(r['candidate_id'] == saved_id for r in registry):
                raise ValueError('Confirmed visual parent is absent from episode registry')
            target.update(candidate_id=saved_id, query_identity_status='resolved')
            return target
        distances = [(float(np.linalg.norm((np.asarray(r['position']) - point)[[0, 2]])), r)
                     for r in registry]
        matches = [(distance, r) for distance, r in distances if distance <= .75]
        if matches:
            _, match = min(matches, key=lambda item: (item[0], item[1]['candidate_id']))
            target.update(candidate_id=match['candidate_id'], query_identity_status='resolved')
        else:
            target['query_identity_status'] = 'unresolved'
    return target
'''


def main():
    manifest = json.loads((HERE / 'base-sha256.json').read_text())
    for path, wanted in manifest.items():
        base = HERE / 'base' / path
        if not base.exists():
            current = REPO / path
            source = current.read_bytes()
            if hashlib.sha256(source).hexdigest() != wanted:
                source = subprocess.run(['git', 'show', BASE_COMMIT + ':' + path],
                                        cwd=REPO, check=True, capture_output=True).stdout
            if hashlib.sha256(source).hexdigest() != wanted:
                raise ValueError(f'Cannot recover the reviewed base: {path}')
            base.parent.mkdir(parents=True, exist_ok=True)
            base.write_bytes(source)
    originals = {p: (HERE / 'base' / p).read_text() for p in manifest}
    for p, source in originals.items():
        if hashlib.sha256(source.encode()).hexdigest() != manifest[p]:
            raise ValueError(f'Base source changed: {p}')
    proposed = dict(originals)
    nav = 'pbsp/navigation/navigator.py'
    proposed[nav] = replace_once(proposed[nav],
        '        if frontier_global_pose is None and self.frontier_reranker.enabled:\n',
        "        if getattr(self, 'blp_enabled', False):\n"
        '            from experiments.blp_observer import action_target_descriptor\n'
        '            if not self.stop_by_seem:\n'
        '                self.blp_visual_parent = None\n'
        '            blp_action_target = action_target_descriptor(\n'
        '                self, self.current_task_idx, frontier_global_pose)\n'
        '        if frontier_global_pose is None and self.frontier_reranker.enabled:\n')
    proposed[nav] = replace_once(proposed[nav],
        '        if self.stop_by_seem:\n            min_dist = 1000000.0\n',
        '        if self.stop_by_seem:\n'
        "            if getattr(self, 'blp_enabled', False):\n"
        "                blp_action_target = getattr(self, 'blp_visual_parent', None) or blp_action_target\n"
        "                blp_action_target['pursuit_mode'] = 'confirmed_visual_goal'\n"
        '            min_dist = 1000000.0\n')
    proposed[nav] = replace_once(proposed[nav],
        "                            if confirmed:\n                                if '_mfmon_camera' in obs.task_observations:\n",
        '                            if confirmed:\n'
        "                                if getattr(self, 'blp_enabled', False):\n"
        "                                    blp_action_target['pursuit_mode'] = 'confirmed_visual_goal'\n"
        '                                    self.blp_visual_parent = blp_action_target\n'
        "                                if '_mfmon_camera' in obs.task_observations:\n")
    proposed[nav] = replace_once(proposed[nav],
        '        self.last_action_metadata = {\n',
        '        self.last_action_metadata = {\n'
        "            **({'blp_action_target': blp_action_target}\n"
        "               if getattr(self, 'blp_enabled', False) else {}),\n")
    proposed['experiments/blp_observer.py'] += HELPERS
    runner = 'experiments/runner.py'
    for variable, mode in (('escape_goal', 'escape'), ('random_goal', 'random')):
        anchor = f"                    agent.set_navmesh_override_goal({variable}, '{mode}')\n"
        proposed[runner] = replace_once(proposed[runner], anchor, anchor +
            "                    if getattr(agent, 'blp_enabled', False):\n"
            "                        action_metadata['blp_effective_recovery'] = {\n"
            f"                            'mode': '{mode}', 'position': list(map(float, {variable}))}}\n")
    proposed[runner] = replace_once(proposed[runner],
        '            env.apply_action(action, info=info)\n',
        "            if getattr(agent, 'blp_enabled', False):\n"
        '                from experiments.blp_observer import bind_action_target\n'
        "                original_target = action_metadata['blp_action_target']\n"
        '                bound_target = bind_action_target(\n'
        '                    planning, action_metadata, current_task_idx, t)\n'
        "                if bound_target['source'] == 'query_candidate' and bound_target['candidate_id'] is not None:\n"
        "                    original_target['candidate_id'] = bound_target['candidate_id']\n"
        "                action_metadata['blp_action_target'] = bound_target\n"
        '            env.apply_action(action, info=info)\n')
    trajectory = 'experiments/trajectory.py'
    anchor = '        encoded = json.dumps(row, allow_nan=False, separators=(",", ":")) + "\\n"\n'
    proposed[trajectory] = replace_once(proposed[trajectory], anchor,
        "        target = self.action_metadata().get('blp_action_target')\n"
        '        if target is not None:\n'
        "            if target['task_idx'] != task or target['frame_index'] != self.frame + 1:\n"
        "                raise ValueError('BLP target and accepted-action trajectory disagree')\n"
        "            row['blp_action_target'] = json.loads(json.dumps(target, allow_nan=False))\n"
        + anchor)
    patch = []
    for path, source in proposed.items():
        ast.parse(source, filename=path)
        compile(source, path, 'exec')
        target = HERE / 'proposed' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
        patch.extend(difflib.unified_diff(originals[path].splitlines(True), source.splitlines(True),
                                        fromfile='a/' + path, tofile='b/' + path))
    (HERE / 'action-trace-after-calibration.patch').write_text(''.join(patch))
    print(json.dumps({'applied': False, 'proposed_files': list(proposed),
                      'compiled_files': len(proposed), 'base_sha256': manifest}, indent=2))


if __name__ == '__main__':
    main()
