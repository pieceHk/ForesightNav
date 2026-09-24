"""Model-independent contracts for suffix-conditioned sequence planning (SCSP).

The objects in this module deliberately use only the Python standard library.  They
are safe to import from unit tests, mapping adapters, and evaluator entrypoints
without importing Habitat or either runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Type, TypeVar


XYZ = Tuple[float, float, float]
QuantizedXYZ = Tuple[int, int, int]
CandidateIdentity = Tuple[int, QuantizedXYZ]


class SCSPContractError(ValueError):
    """Raised when an SCSP value violates a cross-runtime contract."""


class EvidenceState(str, Enum):
    """Evidence states reserved by the protocol.

    MVP producers emit only ``VERIFIED_POSITIVE`` and ``UNKNOWN``.  The negative
    state is reserved now so later producers cannot silently reinterpret absence of
    evidence as negative evidence.
    """

    VERIFIED_POSITIVE = "verified_positive"
    CERTIFIED_NEGATIVE = "certified_negative"
    UNKNOWN = "unknown"


class BatchValidationStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"


class ExperimentArm(str, Enum):
    EXACT_474A752 = "exact_474a752"
    IDEA1_ONLY = "idea1_only"
    ORDERED_SUFFIX_MVP = "ordered_suffix_mvp"


class GeodesicStatus(str, Enum):
    AVAILABLE = "available"
    UNREACHABLE = "unreachable"
    UNAVAILABLE = "unavailable"


class FallbackReason(str, Enum):
    """Frozen vocabulary for falling back to the Idea-1 ordering."""

    NONE = "none"
    SCSP_DISABLED = "scsp_disabled"
    WEIGHT_ZERO = "weight_zero"
    LAST_TASK = "last_task"
    NO_CANDIDATES = "no_candidates"
    FUTURE_LAYER_EMPTY = "future_layer_empty"
    ALL_EDGES_UNREACHABLE = "all_edges_unreachable"
    EQUAL_SUFFIX_COSTS = "equal_suffix_costs"
    REBUILDING = "rebuilding"
    EPISODE_MISMATCH = "episode_mismatch"
    PROVIDER_TOKEN_CHANGED = "provider_token_changed"
    PROVIDER_STATUS_UNAVAILABLE = "provider_status_unavailable"
    INVALID_CANDIDATE_BATCH = "invalid_candidate_batch"
    STALE_VERSION_TOKEN = "stale_version_token"
    GEODESIC_UNAVAILABLE = "geodesic_unavailable"
    NO_FEASIBLE_SUFFIX = "no_feasible_suffix"
    LATENCY_BUDGET_EXCEEDED = "latency_budget_exceeded"
    PLANNER_ERROR = "planner_error"


class InterventionReason(str, Enum):
    """Frozen vocabulary for an SCSP selection that differs from Idea 1."""

    NONE = "none"
    LOWER_SUFFIX_COST = "lower_suffix_cost"
    IDEA1_CANDIDATE_UNREACHABLE = "idea1_candidate_unreachable"


EnumT = TypeVar("EnumT", bound=Enum)


def _enum(value: Any, enum_type: Type[EnumT], name: str) -> EnumT:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(repr(item.value) for item in enum_type)
        raise SCSPContractError(
            f"{name} must be one of {allowed}, got {value!r}"
        ) from exc


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SCSPContractError(f"{name} must be a non-empty string")
    return value


def _sha256(value: Any, name: str) -> str:
    digest = _nonempty_string(value, name).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise SCSPContractError(f"{name} must be a 64-character hexadecimal SHA-256")
    return digest


def _optional_reason(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _nonempty_string(value, name)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SCSPContractError(f"{name} must be an integer, got {value!r}")
    result = int(value)
    if result < 0:
        raise SCSPContractError(f"{name} must be >= 0, got {result!r}")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SCSPContractError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _finite_float(value: Any, name: str, *, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SCSPContractError(f"{name} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise SCSPContractError(f"{name} must be finite, got {result!r}")
    if minimum is not None and result < minimum:
        raise SCSPContractError(f"{name} must be >= {minimum}, got {result!r}")
    return result


def _xyz(value: Any, name: str) -> XYZ:
    if isinstance(value, (str, bytes)):
        raise SCSPContractError(f"{name} must be a finite xyz vector")
    try:
        coordinates = tuple(value)
    except TypeError as exc:
        raise SCSPContractError(f"{name} must be a finite xyz vector") from exc
    if len(coordinates) != 3:
        raise SCSPContractError(
            f"{name} must contain exactly three coordinates, got {len(coordinates)}"
        )
    return tuple(
        _finite_float(coordinate, f"{name}[{index}]")
        for index, coordinate in enumerate(coordinates)
    )  # type: ignore[return-value]


def quantize_xyz(position: Sequence[float], quantum_m: float = 0.001) -> QuantizedXYZ:
    """Return the revision-local coordinate component of a candidate identity."""

    point = _xyz(position, "position")
    quantum = _finite_float(quantum_m, "quantum_m", minimum=0.0)
    if quantum == 0.0:
        raise SCSPContractError("quantum_m must be > 0")
    return tuple(int(round(coordinate / quantum)) for coordinate in point)  # type: ignore[return-value]


def _candidate_identity(value: Any, name: str) -> CandidateIdentity:
    if isinstance(value, (str, bytes)):
        raise SCSPContractError(f"{name} must be (raw_rank, quantized_xyz)")
    try:
        raw_rank, quantized = value
    except (TypeError, ValueError) as exc:
        raise SCSPContractError(f"{name} must be (raw_rank, quantized_xyz)") from exc
    if isinstance(quantized, (str, bytes)):
        raise SCSPContractError(f"{name}.quantized_xyz must contain three integers")
    try:
        coordinates = tuple(quantized)
    except TypeError as exc:
        raise SCSPContractError(
            f"{name}.quantized_xyz must contain three integers"
        ) from exc
    if len(coordinates) != 3:
        raise SCSPContractError(
            f"{name}.quantized_xyz must contain exactly three integers"
        )
    return (
        _nonnegative_int(raw_rank, f"{name}.raw_rank"),
        tuple(
            _integer(coordinate, f"{name}.quantized_xyz[{index}]")
            for index, coordinate in enumerate(coordinates)
        ),  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class EpisodeIdentity:
    """Complete identity of one three-source MON runtime episode."""

    dataset_manifest_sha256: str
    scene_id: str
    runtime_episode_id: str
    source_episode_ids: Tuple[str, ...]
    ordered_categories: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_manifest_sha256",
            _sha256(self.dataset_manifest_sha256, "dataset_manifest_sha256"),
        )
        object.__setattr__(
            self, "scene_id", _nonempty_string(self.scene_id, "scene_id")
        )
        object.__setattr__(
            self,
            "runtime_episode_id",
            _nonempty_string(self.runtime_episode_id, "runtime_episode_id"),
        )
        if isinstance(self.source_episode_ids, (str, bytes)) or isinstance(
            self.ordered_categories, (str, bytes)
        ):
            raise SCSPContractError(
                "source_episode_ids and ordered_categories must be sequences"
            )
        try:
            source_ids = tuple(self.source_episode_ids)
            categories = tuple(self.ordered_categories)
        except TypeError as exc:
            raise SCSPContractError(
                "source_episode_ids and ordered_categories must be sequences"
            ) from exc
        source_ids = tuple(
            _nonempty_string(source_id, f"source_episode_ids[{index}]")
            for index, source_id in enumerate(source_ids)
        )
        categories = tuple(
            _nonempty_string(category, f"ordered_categories[{index}]")
            for index, category in enumerate(categories)
        )
        if len(source_ids) != 3:
            raise SCSPContractError(
                "a MON episode identity requires exactly three source_episode_ids"
            )
        if len(categories) != len(source_ids):
            raise SCSPContractError(
                "source_episode_ids and ordered_categories must have equal length"
            )
        object.__setattr__(self, "source_episode_ids", source_ids)
        object.__setattr__(self, "ordered_categories", categories)


@dataclass(frozen=True)
class VersionToken:
    """Identity of the exact episode/query/map snapshot consumed by a plan."""

    scene_id: str
    episode_id: str
    map_revision: int
    last_fused_keyframes: int
    query_serial: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scene_id", _nonempty_string(self.scene_id, "scene_id")
        )
        object.__setattr__(
            self, "episode_id", _nonempty_string(self.episode_id, "episode_id")
        )
        for name in ("map_revision", "last_fused_keyframes", "query_serial"):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        if self.map_revision != self.last_fused_keyframes:
            raise SCSPContractError(
                "map_revision must equal last_fused_keyframes for the mapper snapshot"
            )


@dataclass(frozen=True)
class CandidateRecord:
    """One query candidate with provenance retained across adapters."""

    task_idx: int
    category: str
    position: XYZ
    raw_rank: int
    semantic_score: float
    support_count: int
    support_score: float
    map_revision: int

    def __post_init__(self) -> None:
        for name in ("task_idx", "raw_rank", "support_count", "map_revision"):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        object.__setattr__(
            self, "category", _nonempty_string(self.category, "category")
        )
        object.__setattr__(self, "position", _xyz(self.position, "position"))
        for name in ("semantic_score", "support_score"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))

    def identity(self, quantum_m: float = 0.001) -> Tuple[int, QuantizedXYZ]:
        """Return ``(raw_rank, quantized_xyz)`` within episode/task/revision scope."""

        return self.raw_rank, quantize_xyz(self.position, quantum_m)


TaskPositions = Tuple[Tuple[int, Tuple[XYZ, ...]], ...]
TaskRecords = Tuple[Tuple[int, Tuple[CandidateRecord, ...]], ...]


def _task_items(value: Any, name: str) -> Iterable[Any]:
    if isinstance(value, Mapping):
        return value.items()
    if isinstance(value, (str, bytes)):
        raise SCSPContractError(f"{name} must be a mapping or task-indexed pairs")
    try:
        return iter(value)
    except TypeError as exc:
        raise SCSPContractError(
            f"{name} must be a mapping or task-indexed pairs"
        ) from exc


def _normalise_positions_by_task(value: Any) -> TaskPositions:
    groups = []
    for group_index, item in enumerate(_task_items(value, "positions_by_task")):
        try:
            task_idx, positions = item
        except (TypeError, ValueError) as exc:
            raise SCSPContractError(
                f"positions_by_task[{group_index}] must be (task_idx, positions)"
            ) from exc
        index = _nonnegative_int(task_idx, f"positions_by_task[{group_index}].task_idx")
        if isinstance(positions, (str, bytes)):
            raise SCSPContractError(
                f"positions_by_task[{group_index}].positions must be a sequence"
            )
        try:
            points = tuple(
                _xyz(
                    point, f"positions_by_task[{group_index}].positions[{point_index}]"
                )
                for point_index, point in enumerate(positions)
            )
        except TypeError as exc:
            raise SCSPContractError(
                f"positions_by_task[{group_index}].positions must be a sequence"
            ) from exc
        groups.append((index, points))
    return tuple(groups)


def _normalise_records_by_task(value: Any) -> TaskRecords:
    groups = []
    for group_index, item in enumerate(_task_items(value, "records_by_task")):
        try:
            task_idx, records = item
        except (TypeError, ValueError) as exc:
            raise SCSPContractError(
                f"records_by_task[{group_index}] must be (task_idx, records)"
            ) from exc
        index = _nonnegative_int(task_idx, f"records_by_task[{group_index}].task_idx")
        if isinstance(records, (str, bytes)):
            raise SCSPContractError(
                f"records_by_task[{group_index}].records must be a sequence"
            )
        try:
            normalised_records = tuple(records)
        except TypeError as exc:
            raise SCSPContractError(
                f"records_by_task[{group_index}].records must be a sequence"
            ) from exc
        for record_index, record in enumerate(normalised_records):
            if not isinstance(record, CandidateRecord):
                raise SCSPContractError(
                    f"records_by_task[{group_index}].records[{record_index}] "
                    "must be a CandidateRecord"
                )
        groups.append((index, normalised_records))
    return tuple(groups)


def _duplicate_task_index(groups: Sequence[Tuple[int, Any]], name: str) -> None:
    seen = set()
    for task_idx, _ in groups:
        if task_idx in seen:
            raise SCSPContractError(f"duplicate task index {task_idx} in {name}")
        seen.add(task_idx)


@dataclass(frozen=True)
class CandidateBatch:
    """Complete, ordered candidates for one versioned MON episode query.

    A batch marked ``INVALID`` is a structurally safe sentinel carrying a reason;
    malformed records are never admitted.  The provider keeps its legacy coordinate
    payload outside this contract when constructing such a sentinel.  Consumers must
    call :func:`validate_candidate_batch` before SCSP planning.
    """

    scene_id: str
    episode_id: str
    map_revision: int
    last_fused_keyframes: int
    query_serial: int
    ordered_categories: Tuple[str, ...]
    positions_by_task: TaskPositions
    records_by_task: TaskRecords
    validation_status: BatchValidationStatus = BatchValidationStatus.VALID
    validation_reason: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scene_id", _nonempty_string(self.scene_id, "scene_id")
        )
        object.__setattr__(
            self, "episode_id", _nonempty_string(self.episode_id, "episode_id")
        )
        for name in ("map_revision", "last_fused_keyframes", "query_serial"):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        if self.map_revision != self.last_fused_keyframes:
            raise SCSPContractError(
                "map_revision must equal last_fused_keyframes for the mapper snapshot"
            )

        if isinstance(self.ordered_categories, (str, bytes)):
            raise SCSPContractError("ordered_categories must be a sequence")
        try:
            categories = tuple(self.ordered_categories)
        except TypeError as exc:
            raise SCSPContractError("ordered_categories must be a sequence") from exc
        categories = tuple(
            _nonempty_string(category, f"ordered_categories[{index}]")
            for index, category in enumerate(categories)
        )
        object.__setattr__(self, "ordered_categories", categories)
        object.__setattr__(
            self,
            "positions_by_task",
            _normalise_positions_by_task(self.positions_by_task),
        )
        object.__setattr__(
            self, "records_by_task", _normalise_records_by_task(self.records_by_task)
        )

        status = _enum(
            self.validation_status, BatchValidationStatus, "validation_status"
        )
        reason = _optional_reason(self.validation_reason, "validation_reason")
        object.__setattr__(self, "validation_status", status)
        object.__setattr__(self, "validation_reason", reason)
        if status is BatchValidationStatus.VALID and reason is not None:
            raise SCSPContractError(
                "a valid candidate batch cannot have a validation reason"
            )
        if status is BatchValidationStatus.INVALID and reason is None:
            raise SCSPContractError(
                "an invalid candidate batch requires validation_reason"
            )
        _validate_candidate_batch_structure(self)

    @property
    def version_token(self) -> VersionToken:
        return VersionToken(
            scene_id=self.scene_id,
            episode_id=self.episode_id,
            map_revision=self.map_revision,
            last_fused_keyframes=self.last_fused_keyframes,
            query_serial=self.query_serial,
        )

    def positions_for_task(self, task_idx: int) -> Tuple[XYZ, ...]:
        index = _nonnegative_int(task_idx, "task_idx")
        for candidate_idx, positions in self.positions_by_task:
            if candidate_idx == index:
                return positions
        raise KeyError(index)

    def records_for_task(self, task_idx: int) -> Tuple[CandidateRecord, ...]:
        index = _nonnegative_int(task_idx, "task_idx")
        for candidate_idx, records in self.records_by_task:
            if candidate_idx == index:
                return records
        raise KeyError(index)


def _validate_candidate_batch_structure(batch: CandidateBatch) -> None:
    _duplicate_task_index(batch.positions_by_task, "positions_by_task")
    _duplicate_task_index(batch.records_by_task, "records_by_task")

    expected_indices = tuple(range(len(batch.ordered_categories)))
    position_indices = tuple(task_idx for task_idx, _ in batch.positions_by_task)
    record_indices = tuple(task_idx for task_idx, _ in batch.records_by_task)
    if position_indices != expected_indices:
        raise SCSPContractError(
            "positions_by_task must contain every task exactly once in ordered-category order: "
            f"expected {expected_indices!r}, got {position_indices!r}"
        )
    if record_indices != expected_indices:
        raise SCSPContractError(
            "records_by_task must contain every task exactly once in ordered-category order: "
            f"expected {expected_indices!r}, got {record_indices!r}"
        )

    for task_idx in expected_indices:
        category = batch.ordered_categories[task_idx]
        positions = batch.positions_by_task[task_idx][1]
        records = batch.records_by_task[task_idx][1]
        if len(records) != len(positions):
            raise SCSPContractError(
                f"task {task_idx} record/position count mismatch: "
                f"records={len(records)} positions={len(positions)}"
            )

        identities = set()
        raw_ranks = set()
        for candidate_idx, (position, record) in enumerate(zip(positions, records)):
            if record.task_idx != task_idx:
                raise SCSPContractError(
                    f"task {task_idx} record {candidate_idx} has task_idx={record.task_idx}"
                )
            if record.category != category:
                raise SCSPContractError(
                    f"task {task_idx} record {candidate_idx} category {record.category!r} "
                    f"does not match {category!r}"
                )
            if record.position != position:
                raise SCSPContractError(
                    f"task {task_idx} record/position alignment mismatch at candidate "
                    f"{candidate_idx}"
                )
            if record.map_revision != batch.map_revision:
                raise SCSPContractError(
                    f"mixed map revisions: batch={batch.map_revision}, "
                    f"task {task_idx} candidate {candidate_idx}={record.map_revision}"
                )
            identity = record.identity()
            if identity in identities:
                raise SCSPContractError(
                    f"duplicate candidate identity in task {task_idx}: {identity!r}"
                )
            if record.raw_rank in raw_ranks:
                raise SCSPContractError(
                    f"duplicate raw_rank {record.raw_rank} in task {task_idx}"
                )
            identities.add(identity)
            raw_ranks.add(record.raw_rank)


def validate_candidate_batch(batch: CandidateBatch) -> CandidateBatch:
    """Validate a batch for SCSP consumption and return it unchanged."""

    if not isinstance(batch, CandidateBatch):
        raise SCSPContractError("batch must be a CandidateBatch")
    _validate_candidate_batch_structure(batch)
    if batch.validation_status is BatchValidationStatus.INVALID:
        raise SCSPContractError(
            f"candidate batch is invalid: {batch.validation_reason}"
        )
    return batch


@dataclass(frozen=True)
class GeodesicDistance:
    """A geodesic distance with an explicit availability/reachability status."""

    status: GeodesicStatus
    distance_m: Optional[float]

    def __post_init__(self) -> None:
        status = _enum(self.status, GeodesicStatus, "status")
        object.__setattr__(self, "status", status)
        if status is GeodesicStatus.AVAILABLE:
            if self.distance_m is None:
                raise SCSPContractError("available distance requires distance_m")
            object.__setattr__(
                self,
                "distance_m",
                _finite_float(self.distance_m, "distance_m", minimum=0.0),
            )
            return

        if status is GeodesicStatus.UNAVAILABLE:
            if self.distance_m is not None:
                raise SCSPContractError(
                    "unavailable distance_m must be None"
                )
            return

        if self.distance_m is None:
            return
        if isinstance(self.distance_m, bool) or not isinstance(self.distance_m, Real):
            raise SCSPContractError(
                "unreachable distance_m must be None or positive infinity"
            )
        distance = float(self.distance_m)
        if distance != math.inf:
            raise SCSPContractError(
                "unreachable distance_m must be None or positive infinity"
            )
        object.__setattr__(self, "distance_m", distance)

    @classmethod
    def available(cls, distance_m: float) -> "GeodesicDistance":
        return cls(GeodesicStatus.AVAILABLE, distance_m)

    @classmethod
    def unreachable(cls) -> "GeodesicDistance":
        return cls(GeodesicStatus.UNREACHABLE, math.inf)

    @classmethod
    def unavailable(cls) -> "GeodesicDistance":
        return cls(GeodesicStatus.UNAVAILABLE, None)


@dataclass(frozen=True)
class GeodesicMatrix:
    """One labeled strict-geodesic transition layer in the reverse DP."""

    source_task_idx: int
    target_task_idx: int
    source_candidate_identities: Tuple[CandidateIdentity, ...]
    target_candidate_identities: Tuple[CandidateIdentity, ...]
    distances: Tuple[Tuple[GeodesicDistance, ...], ...]

    def __post_init__(self) -> None:
        source_task = _nonnegative_int(self.source_task_idx, "source_task_idx")
        target_task = _nonnegative_int(self.target_task_idx, "target_task_idx")
        if target_task != source_task + 1:
            raise SCSPContractError(
                "a geodesic matrix must connect consecutive task indices"
            )
        object.__setattr__(self, "source_task_idx", source_task)
        object.__setattr__(self, "target_task_idx", target_task)

        source_identities = self._normalise_identities(
            self.source_candidate_identities, "source_candidate_identities"
        )
        target_identities = self._normalise_identities(
            self.target_candidate_identities, "target_candidate_identities"
        )
        object.__setattr__(self, "source_candidate_identities", source_identities)
        object.__setattr__(self, "target_candidate_identities", target_identities)

        try:
            matrix = tuple(tuple(row) for row in self.distances)
        except TypeError as exc:
            raise SCSPContractError("distances must be a rectangular matrix") from exc
        if len(matrix) != len(source_identities):
            raise SCSPContractError(
                "distance row count must match source_candidate_identities"
            )
        for row_index, row in enumerate(matrix):
            if len(row) != len(target_identities):
                raise SCSPContractError(
                    f"distance row {row_index} length must match "
                    "target_candidate_identities"
                )
            for column_index, distance in enumerate(row):
                if not isinstance(distance, GeodesicDistance):
                    raise SCSPContractError(
                        f"distances[{row_index}][{column_index}] must be a "
                        "GeodesicDistance"
                    )
        object.__setattr__(self, "distances", matrix)

    @staticmethod
    def _normalise_identities(values: Any, name: str) -> Tuple[CandidateIdentity, ...]:
        if isinstance(values, (str, bytes)):
            raise SCSPContractError(f"{name} must be a sequence")
        try:
            identities = tuple(
                _candidate_identity(value, f"{name}[{index}]")
                for index, value in enumerate(values)
            )
        except TypeError as exc:
            raise SCSPContractError(f"{name} must be a sequence") from exc
        if len(set(identities)) != len(identities):
            raise SCSPContractError(f"{name} contains duplicate candidate identities")
        raw_ranks = tuple(identity[0] for identity in identities)
        if len(set(raw_ranks)) != len(raw_ranks):
            raise SCSPContractError(f"{name} contains duplicate raw ranks")
        return identities


@dataclass(frozen=True)
class CandidateSuffixCost:
    raw_rank: int
    cost: GeodesicDistance

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "raw_rank", _nonnegative_int(self.raw_rank, "raw_rank")
        )
        if not isinstance(self.cost, GeodesicDistance):
            raise SCSPContractError("cost must be a GeodesicDistance")


@dataclass(frozen=True)
class SuffixPlan:
    """Immutable output of a suffix planner for one current task."""

    input_token: VersionToken
    current_task: int
    strict_geodesic_matrices: Tuple[GeodesicMatrix, ...]
    suffix_costs: Tuple[CandidateSuffixCost, ...]
    selected_raw_rank: Optional[int]
    intervention: bool
    fallback_reason: FallbackReason
    intervention_reason: InterventionReason
    compute_time_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.input_token, VersionToken):
            raise SCSPContractError("input_token must be a VersionToken")
        object.__setattr__(
            self, "current_task", _nonnegative_int(self.current_task, "current_task")
        )

        try:
            matrices = tuple(self.strict_geodesic_matrices)
        except TypeError as exc:
            raise SCSPContractError(
                "strict_geodesic_matrices must be a sequence"
            ) from exc
        expected_source_task = self.current_task
        previous_target_identities = None
        transitions = set()
        for index, matrix in enumerate(matrices):
            if not isinstance(matrix, GeodesicMatrix):
                raise SCSPContractError(
                    f"strict_geodesic_matrices[{index}] must be a GeodesicMatrix"
                )
            transition = (matrix.source_task_idx, matrix.target_task_idx)
            if transition in transitions:
                raise SCSPContractError(
                    f"duplicate geodesic transition layer {transition!r}"
                )
            if matrix.source_task_idx != expected_source_task:
                raise SCSPContractError(
                    "strict_geodesic_matrices must form a consecutive task chain "
                    f"starting at current_task={self.current_task}"
                )
            if (
                previous_target_identities is not None
                and matrix.source_candidate_identities != previous_target_identities
            ):
                raise SCSPContractError(
                    "adjacent geodesic layers have misaligned candidate identities"
                )
            transitions.add(transition)
            expected_source_task = matrix.target_task_idx
            previous_target_identities = matrix.target_candidate_identities
        object.__setattr__(self, "strict_geodesic_matrices", matrices)

        try:
            costs = tuple(self.suffix_costs)
        except TypeError as exc:
            raise SCSPContractError("suffix_costs must be a sequence") from exc
        for index, cost in enumerate(costs):
            if not isinstance(cost, CandidateSuffixCost):
                raise SCSPContractError(
                    f"suffix_costs[{index}] must be a CandidateSuffixCost"
                )
        ranks = tuple(cost.raw_rank for cost in costs)
        if len(set(ranks)) != len(ranks):
            raise SCSPContractError("suffix_costs contains duplicate raw_rank values")
        if matrices:
            source_ranks = tuple(
                identity[0] for identity in matrices[0].source_candidate_identities
            )
            if ranks != source_ranks:
                raise SCSPContractError(
                    "suffix_costs must align with the first geodesic layer's "
                    "source candidates"
                )
        object.__setattr__(self, "suffix_costs", costs)

        if self.selected_raw_rank is not None:
            selected = _nonnegative_int(self.selected_raw_rank, "selected_raw_rank")
            if selected not in ranks:
                raise SCSPContractError(
                    "selected_raw_rank must identify an entry in suffix_costs"
                )
            object.__setattr__(self, "selected_raw_rank", selected)

        if not isinstance(self.intervention, bool):
            raise SCSPContractError("intervention must be a bool")

        object.__setattr__(
            self,
            "fallback_reason",
            _enum(self.fallback_reason, FallbackReason, "fallback_reason"),
        )
        object.__setattr__(
            self,
            "intervention_reason",
            _enum(self.intervention_reason, InterventionReason, "intervention_reason"),
        )
        if self.intervention and self.intervention_reason is InterventionReason.NONE:
            raise SCSPContractError(
                "intervention=True requires an explicit intervention_reason"
            )
        if self.intervention and self.selected_raw_rank is None:
            raise SCSPContractError("intervention=True requires a selected_raw_rank")
        if (
            not self.intervention
            and self.intervention_reason is not InterventionReason.NONE
        ):
            raise SCSPContractError(
                "intervention=False requires intervention_reason='none'"
            )
        if self.intervention and self.fallback_reason is not FallbackReason.NONE:
            raise SCSPContractError("an intervention cannot also be a fallback")
        if (
            self.selected_raw_rank is None
            and self.fallback_reason is FallbackReason.NONE
        ):
            raise SCSPContractError(
                "a plan without a selected candidate requires an explicit fallback reason"
            )
        object.__setattr__(
            self,
            "compute_time_ms",
            _finite_float(self.compute_time_ms, "compute_time_ms", minimum=0.0),
        )

    @property
    def version_token(self) -> VersionToken:
        return self.input_token

    @property
    def planner_latency_ms(self) -> float:
        return self.compute_time_ms


__all__ = [
    "BatchValidationStatus",
    "CandidateBatch",
    "CandidateIdentity",
    "CandidateRecord",
    "CandidateSuffixCost",
    "EpisodeIdentity",
    "EvidenceState",
    "ExperimentArm",
    "FallbackReason",
    "GeodesicDistance",
    "GeodesicMatrix",
    "GeodesicStatus",
    "InterventionReason",
    "SCSPContractError",
    "SuffixPlan",
    "VersionToken",
    "quantize_xyz",
    "validate_candidate_batch",
]
