"""Plot actual trajectories and candidate beliefs over the dataset navmesh.

The PathFinder loads the static navmesh on CPU; no simulator/rendering context
is created and no observation is passed to a policy. Snapshot choice is fixed:
the first snapshot with two candidates, otherwise the first nonempty snapshot.
"""
import argparse
import json
from pathlib import Path

import habitat_sim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def plot(run, scene, episode, output, diagnostic=False):
    plan = json.loads((run / 'launch.json').read_text())
    if not diagnostic and plan['configuration']['method']['suffix']['mode'] != 'blp_eq13':
        raise ValueError('Formal figures require an active BLP trajectory')
    identities = [p for p in run.rglob('identity.json') if p.parent.name == str(episode)
                  and p.parent.parent.name == scene]
    if len(identities) != 1:
        raise ValueError('Unknown or duplicated episode identity')
    target = identities[0].parent
    trace = jsonl(target / 'trajectory.jsonl')
    identity = json.loads(identities[0].read_text())['identity']
    candidates = [r for p in run.rglob('blp_decisions.jsonl') for r in jsonl(p)
                  if r['scene_id'] == scene and str(r['episode_id']) == str(episode)]
    dispatches = [r for p in run.rglob('scsp_decisions.jsonl') for r in jsonl(p)
                  if r.get('episode_identity', {}).get('scene_id') == scene
                  and str(r.get('episode_identity', {}).get('runtime_episode_id')) == str(episode)]
    records = [json.loads(p.read_text()) for p in run.rglob('per_episode_metrics.json')]
    matches = [r[scene + '_' + str(episode)] for r in records if scene + '_' + str(episode) in r]
    if len(matches) != 1:
        raise ValueError('Complete native episode metrics are required')
    record = matches[0]
    data = Path(plan['resources']['data'])
    meshes = list((data / 'scenes/hm3dv1/hm3d/val').glob(f'*-{scene}/{scene}.basis.navmesh'))
    if len(meshes) != 1:
        raise ValueError('Cannot identify dataset navmesh')
    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(meshes[0])):
        raise ValueError('Cannot load navmesh')
    lower, upper = [np.asarray(p) for p in pathfinder.get_bounds()]
    n = len(identity['ordered_categories'])
    fig = plt.figure(figsize=(4.8 * n + .6, 7.3), constrained_layout=True)
    grid = fig.add_gridspec(3, n + 1, height_ratios=(4.4, 1.6, .4),
                          width_ratios=[1.] * n + [.045])
    axes = [fig.add_subplot(grid[0, k]) for k in range(n)]
    tables = [fig.add_subplot(grid[1, k]) for k in range(n)]
    color_axis = fig.add_subplot(grid[0, n])
    legend_axis = fig.add_subplot(grid[2, :])
    legend_axis.axis('off')
    norm = Normalize(0., 1.)
    metadata = {'scene_id': scene, 'episode_id': str(episode), 'diagnostic_only': diagnostic,
                'navmesh': str(meshes[0]), 'agent_radius_m': float(pathfinder.nav_mesh_settings.agent_radius),
                'snapshot_selection': 'first applied snapshot with >=2 candidates, otherwise first applied nonempty; proposals only in diagnostic mode', 'stages': []}
    for k, (ax, table_axis) in enumerate(zip(axes, tables)):
        table_axis.axis('off')
        rows = [r for r in trace if r['subtask_index'] == k]
        positions = np.asarray([r['world_pose'][4:] for r in rows])
        if not len(positions):
            raise ValueError('Missing stage trajectory')
        meters = .05
        floor = float(positions[0, 1])
        mask = pathfinder.get_topdown_view(meters, floor)
        ax.imshow(np.where(mask, 1., .82), cmap='gray', vmin=0., vmax=1., origin='lower',
                  extent=(lower[0], lower[0] + mask.shape[1] * meters,
                          lower[2], lower[2] + mask.shape[0] * meters))
        ax.plot(positions[:, 0], positions[:, 2], color='#303030', linewidth=1.3, label='Actual trajectory')
        ax.scatter(*positions[0, [0, 2]], s=40, marker='s', color='#1565c0', zorder=6, label='Stage start')
        ax.scatter(*positions[-1, [0, 2]], s=55, marker='x', color='#c62828', zorder=6, label='STOP')
        available = [r for r in candidates if r['k'] == k and r['candidates']]
        if not diagnostic:
            applied_times = {e['blp']['plan_ms'] for e in dispatches if e.get('refresh_applied')
                             and e.get('blp') is not None and e['current_task'] == k}
            available = [r for r in available if r['plan_ms'] in applied_times]
        snapshot = next((r for r in available if len(r['candidates']) >= 2), available[0] if available else None)
        extent = [positions[:, [0, 2]]]
        stage_meta = {'k': k + 1, 'floor_height_m': floor, 'native_success': bool(record['metrics'][k]['goat_sub-task_success'])}
        if snapshot:
            points = np.asarray([r['position'] for r in snapshot['candidates']])
            beliefs = [r['B'] for r in snapshot['candidates']]
            ax.scatter(points[:, 0], points[:, 2], c=beliefs, cmap='viridis', norm=norm,
                       s=65, edgecolor='black', linewidth=.6, zorder=5)
            extent.append(points[:, [0, 2]])
            offsets = [(-14, 12), (12, 12), (-14, -16), (12, -16), (0, 26)]
            for index, row in enumerate(snapshot['candidates']):
                point = row['position']
                ax.annotate(str(index + 1), (point[0], point[2]),
                            xytext=offsets[index % len(offsets)], textcoords='offset points',
                            fontsize=8, ha='center', va='center',
                            arrowprops={'arrowstyle': '-', 'color': '#666666', 'lw': .6},
                            bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': '#aaaaaa',
                                  'boxstyle': 'round,pad=.2'}, zorder=8)
            selected = next(r for r in snapshot['candidates'] if r['raw_rank'] == snapshot['chosen_raw_rank'])
            ax.scatter(selected['position'][0], selected['position'][2], marker='*', s=210,
                       facecolor='none', edgecolor='#e65100', linewidth=1.5, zorder=7,
                       label='BLP proposal' if diagnostic else 'BLP after hysteresis')
            applied = [e for e in dispatches if e.get('refresh_applied') and e.get('blp', {}).get('plan_ms') == snapshot['plan_ms']]
            if not diagnostic and len(applied) != 1:
                raise ValueError('Cannot align selected snapshot with one actual dispatch')
            actual_rank = applied[0]['selected_raw_ranks']['dispatched'] if applied else None
            if applied and actual_rank is not None:
                batch_records = next(r for task, r in applied[0]['candidate_batch']['records_by_task'] if task == k)
                actual = next(r for r in batch_records if r['raw_rank'] == actual_rank)
                ax.scatter(actual['position'][0], actual['position'][2], marker='o', s=150,
                           facecolor='none', edgecolor='#d32f2f', linewidth=1.3, zorder=7,
                           label='Actual dispatched candidate')
            stage_meta.update(t=snapshot['t'], candidates=snapshot['candidates'],
                              chosen_raw_rank=snapshot['chosen_raw_rank'], actual_dispatched_raw_rank=actual_rank)
            values = [[str(index + 1), str(row['raw_rank']), f'{row["B"]:.3f}',
                       f'{row["position"][1] - floor:+.2f}']
                      for index, row in enumerate(snapshot['candidates'])]
            table = table_axis.table(cellText=values,
                                     colLabels=['#', 'Raw rank', 'Belief B', 'Δheight (m)'],
                                     colWidths=[.1, .3, .3, .3], cellLoc='center',
                                     bbox=[0., .18, 1., .67])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            for (row_index, column), cell in table.get_celld().items():
                cell.set_linewidth(.35)
                cell.set_edgecolor('#cccccc')
                if row_index == 0:
                    cell.set_facecolor('#eeeeee')
            table_axis.text(0., .98, f'Applied snapshot: t={snapshot["t"]}',
                            transform=table_axis.transAxes, fontsize=9, va='top')
            table_axis.text(0., .04,
                            f'BLP: r{snapshot["chosen_raw_rank"]}   |   Dispatched: r{actual_rank}',
                            transform=table_axis.transAxes, fontsize=8)
        else:
            table_axis.text(0., .8, 'No applied nonempty candidate snapshot',
                            transform=table_axis.transAxes, fontsize=8)
        cloud = np.concatenate(extent)
        minimum, maximum = cloud.min(axis=0) - 1., cloud.max(axis=0) + 1.
        ax.set(xlim=(minimum[0], maximum[0]), ylim=(minimum[1], maximum[1]),
               xlabel='World x (m)', ylabel='World z (m)', aspect='equal',
               title=f'C{k+1}: {identity["ordered_categories"][k]} | {"success" if stage_meta["native_success"] else "failure"}')
        metadata['stages'].append(stage_meta)
    handles, labels = axes[0].get_legend_handles_labels()
    legend_axis.legend(handles, labels, loc='center', fontsize=8, ncol=5, frameon=False)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=color_axis,
                 label='Candidate belief B')
    subtitle = 'Shadow BLP proposal; native legacy trajectory (diagnostic only)' if diagnostic else 'Full BLP: candidate beliefs, dispatch and actual trajectory'
    fig.suptitle(f'{scene}/{episode}\n{subtitle}', fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--episode', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-only', action='store_true')
    args = parser.parse_args()
    plot(args.run, args.scene, args.episode, args.output, args.diagnostic_only)


if __name__ == '__main__':
    main()
