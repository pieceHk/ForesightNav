"""Episode-local Mapping orchestration with explicit publication semantics."""

from threading import RLock
from typing import Optional

from .contracts import CameraObservation, MappingBackend, MappingSnapshot
from .keyframes import KeyframeAdmission
from .resources import release_episode_resources


class OnlineMappingPipeline:
    def __init__(self, backend: MappingBackend, admission: Optional[KeyframeAdmission] = None):
        self._backend = backend
        self._admission = admission or KeyframeAdmission()
        self._lock = RLock()
        self._accepted: list[CameraObservation] = []
        self._raw_frames = 0
        self._scene_id = ""
        self._episode_id = ""

    def reset(self, scene_id: str, episode_id: str) -> None:
        with self._lock:
            self._backend.reset(scene_id, episode_id)
            self._admission.reset()
            self._accepted.clear()
            self._raw_frames = 0
            self._scene_id = scene_id
            self._episode_id = episode_id
        release_episode_resources()

    def observe(self, observation: CameraObservation) -> tuple[bool, float]:
        accepted, ratio = self._admission.accept(observation.rgb)
        with self._lock:
            self._raw_frames += 1
            if accepted:
                self._accepted.append(observation)
        return accepted, ratio

    def update(self) -> MappingSnapshot:
        with self._lock:
            return self._backend.update(tuple(self._accepted))

    def query(self, category: str, top_k: int = 10):
        with self._lock:
            return tuple(self._backend.query(category, top_k))

    def close(self) -> None:
        with self._lock:
            self._backend.close()
