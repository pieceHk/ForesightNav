"""CPU-only schema/identity tests; frozen files live only in temporary folders."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from experiments.blp_observer import initialize_blp
from pbsp.navigation.blp_parameters import CalibrationDocument, load_frozen_calibration


def frozen_fixture(n_folds=3):
    pool = dict(a=30., b0=-8., pi=.2, c_exp=2.)
    return {'schema_version': 'exp4_calibration_v1', 'frozen': True,
            'label_source': 'system_verification', 'selection': 'cross_fit',
            'collected_folds': list(range(n_folds)),
            'scene_to_fold': {f'synthetic_{f}_{i}': f for f in range(5) for i in range(4)},
            'parameters_for_uncollected_folds': pool,
            'parameters_excluding_fold': {str(f): dict(a=40. + f, b0=-10. - f,
                pi=.3 + .01*f, c_exp=1. + f) for f in range(n_folds)}}


class BlpParameterTests(unittest.TestCase):
    def test_all_twenty_scenes_use_their_held_out_group_or_uncollected_fallback(self):
        value = frozen_fixture()
        document = CalibrationDocument.from_mapping(value)
        for f in range(5):
            for i in range(4):
                with self.subTest(fold=f, scene=i):
                    parameters, binding = document.resolve(f'synthetic_{f}_{i}')
                    expected = value['parameters_excluding_fold'][str(f)] if f < 3 else value['parameters_for_uncollected_folds']
                    self.assertEqual(asdict(parameters), expected)
                    self.assertEqual(binding['excluded_fold'], f if f < 3 else None)
                    self.assertEqual(binding['training_folds'], [x for x in range(3) if x != f])

    def test_five_fold_mode_excludes_each_target_group_without_pooled_fallback(self):
        value = frozen_fixture(5)
        document = CalibrationDocument.from_mapping(value)
        for f in range(5):
            parameters, binding = document.resolve(f'synthetic_{f}_0')
            self.assertEqual(parameters.a, 40. + f)
            self.assertEqual(binding['training_folds'], [x for x in range(5) if x != f])
            self.assertEqual(binding['excluded_fold'], f)

    def test_unknown_scene_cannot_silently_fall_back_or_be_guessed_from_a_path(self):
        document = CalibrationDocument.from_mapping(frozen_fixture())
        for scene in (None, 'unknown', '/tmp/synthetic_0_0.basis.glb'):
            with self.subTest(scene=scene), self.assertRaisesRegex(ValueError, 'exactly match'):
                document.resolve(scene)

    def test_runtime_initialization_uses_scene_and_records_exact_file_hash_for_every_variant(self):
        value = frozen_fixture()
        with TemporaryDirectory() as folder:
            path = Path(folder)/'test-only-calibration.json'
            path.write_text(json.dumps(value))
            for variant in ('nearest', 'myopic', 'suffix_only', 'full'):
                planning = SimpleNamespace(blp_config={'enabled': True, 'calibration': str(path), 'variant': variant})
                agent = SimpleNamespace()
                initialize_blp(planning, agent, folder, scene_id='synthetic_1_2')
                self.assertEqual(asdict(planning.blp_parameters), value['parameters_excluding_fold']['1'])
                self.assertEqual(planning.blp_parameter_binding['training_folds'], [0, 2])
                self.assertEqual(planning.blp_parameter_binding['scene_id'], 'synthetic_1_2')
                self.assertEqual(planning.blp_parameter_binding['calibration_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_legacy_pooled_schema_preserves_exact_parameters(self):
        parameters = dict(a=2., b0=-3., pi=.4, c_exp=5.)
        value = {'frozen': True, 'label_source': 'system_verification', 'parameters': parameters}
        document = CalibrationDocument.from_mapping(value)
        self.assertEqual(asdict(document.resolve('any_scene')[0]), parameters)
        value['parameters_by_scene'] = {'some_scene': parameters}
        with self.assertRaisesRegex(ValueError, 'silently'):
            CalibrationDocument.from_mapping(value)

    def test_unfrozen_gt_label_and_proposal_schema_cannot_activate_evaluation(self):
        cases = [('frozen', False), ('label_source', 'ground_truth'),
                 ('schema_version', 'exp4_crossfit_proposal_v1')]
        for key, data in cases:
            value = frozen_fixture(); value[key] = data
            with self.subTest(key=key), self.assertRaises(ValueError):
                CalibrationDocument.from_mapping(value)
        proposal = Path(__file__).resolve().parents[1]/'results/exp4/meta/calibration.crossfit.proposed.json'
        if proposal.exists():
            with self.assertRaisesRegex(ValueError, 'explicitly frozen'):
                load_frozen_calibration(proposal, '6s7QHgap2fW')

    def test_malformed_group_coverage_and_missing_held_out_parameters_are_rejected(self):
        cases = []
        value = frozen_fixture(); value['collected_folds'] = [0, 1, 1]; cases.append(value)
        value = frozen_fixture(); value['scene_to_fold']['synthetic_0_0'] = 1; cases.append(value)
        value = frozen_fixture(); del value['parameters_excluding_fold']['1']; cases.append(value)
        value = frozen_fixture(); value['parameters_excluding_fold']['4'] = value['parameters_for_uncollected_folds']; cases.append(value)
        value = frozen_fixture(); value['scene_to_fold']['synthetic_0_0'] = True; cases.append(value)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                CalibrationDocument.from_mapping(value)

    def test_invalid_parameter_values_are_rejected_in_every_parameter_set(self):
        for field, bad in (('a', 0.), ('b0', float('nan')), ('pi', 1.1), ('c_exp', -1.), ('a', True)):
            value = frozen_fixture(); value['parameters_excluding_fold']['1'][field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                CalibrationDocument.from_mapping(value)

    def test_loaded_plan_is_independent_of_mutations_to_the_input_mapping(self):
        value = frozen_fixture(); document = CalibrationDocument.from_mapping(value)
        value['parameters_excluding_fold']['0']['a'] = 999.
        value['scene_to_fold']['synthetic_0_0'] = 2
        self.assertEqual(document.resolve('synthetic_0_0')[0].a, 40.)
        with self.assertRaises(TypeError):
            document.scene_to_fold['synthetic_0_0'] = 1


if __name__ == '__main__':
    unittest.main()
