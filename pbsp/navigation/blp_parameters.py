"""Resolve frozen self-calibration by the exact benchmark scene identity."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from .blp import BeliefParameters

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PARAMETER_KEYS = frozenset(('a', 'b0', 'pi', 'c_exp'))


def parameter_values(value):
    if not isinstance(value, dict) or set(value) != PARAMETER_KEYS:
        raise ValueError('Belief parameters require exactly a, b0, pi, and c_exp')
    if any(type(x) not in (int, float) for x in value.values()):
        raise ValueError('Belief parameters must be numeric, without boolean values')
    return BeliefParameters(**value)


@dataclass(frozen=True)
class CalibrationDocument:
    selection: str
    pooled: BeliefParameters
    collected_folds: tuple
    scene_to_fold: object
    excluding_fold: object

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or value.get('frozen') is not True:
            raise ValueError('Calibration must be explicitly frozen')
        if value.get('label_source') != 'system_verification':
            raise ValueError('Calibration must use system verification labels')
        mode = value.get('selection', 'pooled')
        if mode == 'pooled':
            crossfit_fields = {'scene_to_fold', 'parameters_excluding_fold',
                               'parameters_for_uncollected_folds', 'parameters_by_scene'}
            if crossfit_fields & set(value):
                raise ValueError('Cross-fit fields cannot silently load as pooled parameters')
            return cls(mode, parameter_values(value.get('parameters')), (),
                       MappingProxyType({}), MappingProxyType({}))
        if mode != 'cross_fit' or value.get('schema_version') != 'exp4_calibration_v1':
            raise ValueError('Unknown calibration selection or cross-fit schema')
        if 'parameters_by_scene' in value:
            raise ValueError('Use one canonical fold mapping, not a second per-scene parameter map')
        collected = value.get('collected_folds')
        if (not isinstance(collected, list) or any(type(f) is not int for f in collected)
                or len(set(collected)) != len(collected)
                or set(collected) not in ({0, 1, 2}, {0, 1, 2, 3, 4})):
            raise ValueError('Cross-fit calibration requires the fixed three or five complete folds')
        mapping = value.get('scene_to_fold')
        if (not isinstance(mapping, dict) or len(mapping) != 20
                or any(not isinstance(s, str) or not s or type(f) is not int or f not in range(5)
                       for s, f in mapping.items())
                or any(sum(f == i for f in mapping.values()) != 4 for i in range(5))):
            raise ValueError('Cross-fit mapping must cover five disjoint four-scene folds')
        excluding = value.get('parameters_excluding_fold')
        if not isinstance(excluding, dict) or set(excluding) != {str(f) for f in collected}:
            raise ValueError('Cross-fit parameter sets must cover exactly the collected held-out folds')
        pooled = parameter_values(value.get('parameters_for_uncollected_folds'))
        if 'parameters' in value and parameter_values(value['parameters']) != pooled:
            raise ValueError('Ambiguous pooled fallback in cross-fit calibration')
        return cls(mode, pooled, tuple(sorted(collected)), MappingProxyType(dict(mapping)),
                   MappingProxyType({int(f): parameter_values(p) for f, p in excluding.items()}))

    def resolve(self, scene_id=None):
        if self.selection == 'pooled':
            return self.pooled, {'selection': 'pooled', 'scene_id': scene_id,
                                 'scene_fold': None, 'excluded_fold': None, 'training_folds': None}
        if scene_id not in self.scene_to_fold:
            raise ValueError('Cross-fit scene must exactly match the frozen benchmark mapping')
        fold = self.scene_to_fold[scene_id]
        held_out = fold if fold in self.collected_folds else None
        training = [f for f in self.collected_folds if f != held_out]
        parameters = self.excluding_fold[fold] if held_out is not None else self.pooled
        return parameters, {'selection': 'cross_fit', 'scene_id': scene_id, 'scene_fold': fold,
                            'excluded_fold': held_out, 'training_folds': training}


def load_frozen_calibration(path, scene_id=None):
    """Parse and hash the same bytes once; never silently fall back by scene."""
    blob = Path(path).read_bytes()
    document = CalibrationDocument.from_mapping(json.loads(blob))
    parameters, binding = document.resolve(scene_id)
    return parameters, dict(binding, calibration_sha256=hashlib.sha256(blob).hexdigest())
