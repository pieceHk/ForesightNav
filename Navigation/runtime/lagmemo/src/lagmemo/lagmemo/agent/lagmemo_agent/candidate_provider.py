import json
from typing import Dict, List, Optional


class CandidateProvider:
    """Base class for navigation candidate providers.

    Implementations supply per-task goal poses that GLUEAgent uses to
    navigate towards objects.  The interface is kept intentionally thin so
    that JsonCandidateProvider (S-1), OfflineOpenGaussianProvider (S-3),
    and OnlineESProvider (Phase 3) can all plug in without changes to the
    agent loop.
    """

    def reset_episode(self, scene_id: str, episode_id: str) -> None:
        """Called once at the start of every episode to load goal data."""
        raise NotImplementedError

    def get_goals(self, task_idx: int, category: str = None) -> List:
        """Return list of 3-D world positions (x, y, z) for *task_idx*.

        Each element is a sequence of length 3 in Habitat world coordinates
        (x-z horizontal plane, y vertical), matching the format previously
        stored in GLUEAgent.gs_poses[task_idx].

        category : str, optional
            Object category name (e.g. "bathtub").  Used by
            OfflineESProvider and future ES-based providers; ignored by
            JsonCandidateProvider.
        """
        raise NotImplementedError

    def get_all_goals(self) -> Dict[int, List]:
        """Return the full {task_idx: [poses]} dict for the current episode.

        Used by GLUEAgent.reset() to populate self.gs_poses so that the
        rest of the agent loop requires zero changes for S-1.
        """
        raise NotImplementedError


class JsonCandidateProvider(CandidateProvider):
    """Candidate provider backed by lagmemo_goal.json.

    Replicates the original gs_poses behaviour exactly:
      - Loads the JSON once at construction time.
      - reset_episode() selects the scene/episode slice.
      - get_all_goals() returns {int(task_idx): [poses]} for that episode.
    """

    def __init__(self, goal_path: str):
        with open(goal_path) as f:
            self._all_goals = json.load(f)
        self._cur_goals: Optional[Dict[int, List]] = None

    def reset_episode(self, scene_id: str, episode_id: str) -> None:
        scene_goals = self._all_goals.get(scene_id, {})
        ep_goals = scene_goals.get(episode_id, {})
        self._cur_goals = {int(k): v for k, v in ep_goals.items()}

    def get_goals(self, task_idx: int, category: str = None) -> List:
        if self._cur_goals is None:
            return []
        return self._cur_goals.get(task_idx, [])

    def get_all_goals(self) -> Dict[int, List]:
        return self._cur_goals if self._cur_goals is not None else {}
