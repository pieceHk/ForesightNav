from enum import Enum

from .contracts import Action, Verification


class GoalHierarchy(str, Enum):
    VISUAL = "visual"
    MEMORY = "memory"
    FRONTIER = "frontier"
    RECOVERY = "recovery"

    @classmethod
    def choose(cls, visual: bool, memory: bool, frontier: bool) -> "GoalHierarchy":
        if visual:
            return cls.VISUAL
        if memory:
            return cls.MEMORY
        if frontier:
            return cls.FRONTIER
        return cls.RECOVERY


class StopBoundary:
    @staticmethod
    def resolve(planner_action: Action, verification: Verification, budget_exhausted: bool) -> Action:
        if verification.confirmed or budget_exhausted:
            return Action.STOP
        if planner_action == Action.STOP:
            return Action.TURN_RIGHT
        return planner_action
