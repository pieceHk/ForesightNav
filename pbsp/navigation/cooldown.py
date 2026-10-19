"""Configuration helpers for category-conditioned candidate cooldowns."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence


_CATEGORY_ALIASES = {
    "sofa": "couch",
    "tv_monitor": "tv",
    "tv monitor": "tv",
}


def normalize_category(value: str) -> str:
    category = str(value).strip().lower()
    return _CATEGORY_ALIASES.get(category, category)


def parse_category_cooldowns(raw: str) -> dict[str, int]:
    """Parse and validate the JSON mapping exported from the TOML profile."""

    if not raw.strip():
        return {}
    values = json.loads(raw)
    if not isinstance(values, dict):
        raise ValueError("category cooldown configuration must be a JSON object")
    output: dict[str, int] = {}
    for raw_category, raw_steps in values.items():
        category = normalize_category(raw_category)
        if not category:
            raise ValueError("category cooldown names must be non-empty")
        if isinstance(raw_steps, bool) or not isinstance(raw_steps, int):
            raise ValueError(f"cooldown for {category!r} must be an integer")
        if raw_steps < 0:
            raise ValueError(f"cooldown for {category!r} must be non-negative")
        if category in output:
            raise ValueError(f"duplicate normalized cooldown category: {category}")
        output[category] = raw_steps
    return output


def cooldown_for_category(
    cooldown_by_category: Mapping[str, int], category: str
) -> int:
    """Return the configured action-step cooldown for one MON subtask."""

    normalized = normalize_category(category)
    if normalized not in cooldown_by_category:
        raise KeyError(f"missing cooldown for MON category {normalized!r}")
    return int(cooldown_by_category[normalized])


def _point(value: Sequence[float]) -> tuple[float, float, float]:
    point = tuple(float(component) for component in value)
    if len(point) != 3 or not all(math.isfinite(component) for component in point):
        raise ValueError("candidate positions must be finite xyz triples")
    return point


def _distance_xz(first: Sequence[float], second: Sequence[float]) -> float:
    a = _point(first)
    b = _point(second)
    return math.hypot(a[0] - b[0], a[2] - b[2])


@dataclass
class CategorySwitchCooldown:
    """Gate switches from the realized active goal to the latest mapper rank."""

    cooldown_by_category: Mapping[str, int]
    match_radius_m: float = 1.0
    _active_by_task: dict[int, tuple[float, float, float]] = field(
        default_factory=dict, init=False
    )
    _last_switch_step: dict[int, int] = field(default_factory=dict, init=False)
    _initialized_tasks: set[int] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.cooldown_by_category = parse_category_cooldowns(
            json.dumps(dict(self.cooldown_by_category))
        )
        if not math.isfinite(self.match_radius_m) or self.match_radius_m <= 0:
            raise ValueError("match_radius_m must be finite and positive")

    @property
    def enabled(self) -> bool:
        return bool(self.cooldown_by_category)

    def reset(self) -> None:
        self._active_by_task.clear()
        self._last_switch_step.clear()
        self._initialized_tasks.clear()

    @staticmethod
    def _step(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("step must be a non-negative integer")
        return int(value)

    def observe_active(
        self,
        task_idx: int,
        position: Sequence[float] | None,
        step: int,
    ) -> bool:
        """Record the active candidate realized by downstream reconciliation."""

        task_idx = int(task_idx)
        step = self._step(step)
        first_observation = task_idx not in self._initialized_tasks
        self._initialized_tasks.add(task_idx)
        if first_observation:
            self._last_switch_step[task_idx] = step

        previous = self._active_by_task.get(task_idx)
        if position is None:
            self._active_by_task.pop(task_idx, None)
            return False

        point = _point(position)
        realized_switch = (
            not first_observation
            and (
                previous is None
                or _distance_xz(previous, point) > self.match_radius_m
            )
        )
        self._active_by_task[task_idx] = point
        if realized_switch:
            self._last_switch_step[task_idx] = step
        return realized_switch

    def apply(
        self,
        *,
        task_idx: int,
        category: str,
        step: int,
        active_position: Sequence[float] | None,
        latest_ranked_candidates: Sequence[Sequence[float]],
    ) -> tuple[list[list[float]], dict[str, object]]:
        """Return the latest mapper rank, holding its active track if required."""

        task_idx = int(task_idx)
        step = self._step(step)
        normalized = normalize_category(category)
        cooldown_steps = cooldown_for_category(
            self.cooldown_by_category, normalized
        )
        latest = [list(_point(value)) for value in latest_ranked_candidates]
        event: dict[str, object] = {
            "task_idx": task_idx,
            "category": normalized,
            "subtask_step": step,
            "cooldown_steps": cooldown_steps,
            "held": 0,
            "proposed_switch": 0,
            "reason": "NO_CANDIDATES" if not latest else "LATEST_UNCHANGED",
        }
        if not latest:
            return latest, event

        active = (
            _point(active_position)
            if active_position is not None
            else self._active_by_task.get(task_idx)
        )
        if active is None:
            event["reason"] = "NO_ACTIVE_ACCEPT_LATEST"
            return latest, event

        distances = [_distance_xz(value, active) for value in latest]
        if distances[0] <= self.match_radius_m:
            event["elapsed_since_switch"] = (
                step - self._last_switch_step.get(task_idx, 0)
            )
            return latest, event
        matches = [
            index
            for index, distance in enumerate(distances)
            if distance <= self.match_radius_m
        ]
        if not matches:
            event.update(
                {
                    "reason": "ACTIVE_MISSING_ACCEPT_LATEST",
                    "proposed_switch": 1,
                }
            )
            return latest, event

        active_match = min(matches, key=distances.__getitem__)
        elapsed = step - self._last_switch_step.get(task_idx, 0)
        event.update(
            {
                "elapsed_since_switch": elapsed,
                "proposed_switch": 1,
            }
        )
        if elapsed < cooldown_steps:
            latest.insert(0, latest.pop(active_match))
            event.update({"reason": "KEEP_ACTIVE_COOLDOWN", "held": 1})
        else:
            event["reason"] = "ACCEPT_LATEST_RANK"
        return latest, event
