"""Identity-only overlap audit; does not inspect evaluation outcomes."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    meta = root / 'meta'
    two = json.loads((meta / '2mon-identities.json').read_text())
    three = json.loads((meta / '3mon-identities.json').read_text())
    folds = json.loads((meta / 'folds.json').read_text())['folds']
    selected = {scene for group in folds[:3] for scene in group}
    calibration = [r for r in two if r['scene_id'] in selected]
    source_two = {(r['scene_id'], str(s)) for r in calibration for s in r['source_episode_ids']}
    source_three = {(r['scene_id'], str(s)) for r in three for s in r['source_episode_ids']}
    pairs = {(r['scene_id'], tuple(r['source_episode_ids'])) for r in calibration}
    prefix_match = [r for r in three if (r['scene_id'], tuple(r['source_episode_ids'][:2])) in pairs]
    value = {'calibration_2mon_episodes': len(calibration), 'evaluation_3mon_episodes': len(three),
             'calibration_scenes': len(selected),
             'shared_scenes': len(selected & {r['scene_id'] for r in three}),
             'unique_calibration_source_episodes': len(source_two),
             'unique_evaluation_source_episodes': len(source_three),
             'shared_source_episodes': len(source_two & source_three),
             'evaluation_ordered_first_two_targets_equal_calibration_sequence': len(prefix_match),
             'matched_prefix_identities': [{'scene_id': r['scene_id'], 'episode_id': r['runtime_episode_id'],
                                            'source_episode_ids': r['source_episode_ids']} for r in prefix_match],
             'scope': 'Follows supplied 2-MON calibration / 3-MON evaluation design; independent ordered episode files do not imply held-out scenes or underlying source goals.',
             'uses_evaluation_outcomes': False}
    (meta / 'calibration-overlap-audit.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({k: v for k, v in value.items() if k != 'matched_prefix_identities'}, indent=2))


if __name__ == '__main__':
    main()
