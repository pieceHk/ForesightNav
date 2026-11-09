"""Canonical checked identities using exact coordinates and recorded IDs.

No distance threshold, label, or GT participates. Raw files remain untouched.
An identity keeps its recorded time-varying points; two checked identities are
unified only if a final check records exactly the same full XYZ point.
"""
import hashlib
import math
from pathlib import Path

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DEFINITION = ('recorded registry identity, merging checked IDs sharing an exact full-XYZ '
              'final-check coordinate within one scene/episode/task/resample')


def key(row):
    return (row.get('_resample_instance'), str(row['scene_id']), str(row['episode_id']),
            int(row['k']), str(row['candidate_id']))


def canonicalize(events, traces=None):
    parent, by_position = {}, {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    checked = [r for r in events if r['event'] == 'verification_final']
    for row in checked:
        if row.get('candidate_id') is None:
            raise ValueError('A final candidate check requires a recorded identity')
        point = row.get('position')
        if point is None or len(point) != 3 or not all(math.isfinite(float(x)) for x in point):
            raise ValueError('A final candidate check requires finite full XYZ coordinates')
        node = key(row)
        find(node)
        coordinate = node[:4] + (tuple(map(float, point)),)
        if coordinate in by_position:
            left, right = find(node), find(by_position[coordinate])
            if left != right:
                # Scope fields agree by construction; lexical ID order is stable.
                first, second = sorted((left, right), key=lambda n: n[-1])
                parent[second] = first
        by_position[coordinate] = node
    mapping = {node: find(node)[-1] for node in parent}
    output = []
    for row in events:
        out = dict(row)
        if row.get('candidate_id') is not None:
            canonical = mapping.get(key(row), str(row['candidate_id']))
            if canonical != str(row['candidate_id']):
                out.setdefault('raw_registry_candidate_id', row['candidate_id'])
                out['candidate_id'] = canonical
        output.append(out)
    rewritten = None
    if traces is not None:
        rewritten = {}
        for pair, rows in traces.items():
            rewritten[pair] = []
            for row in rows:
                out = dict(row)
                if 'blp_action_target' in row:
                    target = dict(row['blp_action_target'])
                    if target.get('candidate_id') is not None:
                        node = (None, str(row['scene_id']), str(row['episode_id']),
                                int(target['task_idx']), str(target['candidate_id']))
                        canonical = mapping.get(node, str(target['candidate_id']))
                        if canonical != str(target['candidate_id']):
                            target.setdefault('raw_registry_candidate_id', target['candidate_id'])
                            target['candidate_id'] = canonical
                    out['blp_action_target'] = target
                rewritten[pair].append(out)
    stats = {'definition': DEFINITION, 'source_sha256': SOURCE_SHA256,
             'raw_checked_registry_identities': len(parent),
             'canonical_checked_identities': len(set((node[:4], cid) for node, cid in mapping.items())),
             'identities_merged': len(parent) - len(set((node[:4], cid) for node, cid in mapping.items())),
             'verification_rows_retained': len(checked),
             'GT_or_labels_used_to_group': False,
             'raw_events_modified': False}
    return output, rewritten, stats

