"""Record issued actions and sensor poses outside the policy loop."""

import json
import math
import operator
from pathlib import Path
from urllib.parse import quote

from pbsp.adapters.habitat import task_index

SCHEMA = "pbsp_trajectory_v1"


def component(value):
    value = str(value)
    if not value:
        raise ValueError("Trajectory identity cannot be empty")
    encoded = quote(value, safe="")
    return encoded.replace(".", "%2E") if encoded in {".", ".."} else encoded


class TrajectoryEnvironment:
    """Observe existing calls without extra sensor reads or policy changes.

    Poses precede the issued action. A row is appended only after the environment
    accepts that action. Failed runs retain their recorded prefix.
    """

    def __init__(self, environment, *, directory, identities, dataset_manifest_sha256,
                 actions, action_metadata):
        self.environment = environment
        self.directory = Path(directory)
        if (self.directory / "episodes").exists():
            raise FileExistsError("Trajectory recording requires a fresh results directory")
        self.identities = identities
        self.dataset_manifest_sha256 = dataset_manifest_sha256
        self.actions = actions
        self.action_metadata = action_metadata
        self.pose = None
        self.output = None

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def reset(self):
        result = self.environment.reset()
        episode = self.environment.habitat_env.current_episode
        self.scene = episode.scene_id.split("/")[-1].split(".")[0]
        self.episode = str(episode.episode_id)
        self.identity = self.identities.get((self.scene, self.episode))
        self.pose, self.output, self.frame, self.task_steps = None, None, 0, {}
        return result

    def get_observation(self):
        observation = self.environment.get_observation()
        pose = [float(value) for value in observation.globalpose]
        if len(pose) != 7 or not all(math.isfinite(value) for value in pose):
            raise ValueError("Trajectory requires a finite [qw,qx,qy,qz,x,y,z] sensor pose")
        self.pose = pose
        if self.output is None:
            self.action_space = {}
            members = getattr(self.actions, "__members__", None)
            if members is None:
                members = getattr(self.actions, "__dict__", {})
            for name, action in members.items():
                if name.startswith("_"):
                    continue
                try:
                    self.action_space[name] = operator.index(getattr(action, "value", action))
                except TypeError:
                    continue
            if "STOP" not in self.action_space:
                raise ValueError("Trajectory recording requires a discrete STOP action")
            categories = [task["category"] for task in observation.task_observations["tasks"]]
            identity = {"dataset_manifest_sha256": self.dataset_manifest_sha256,
                        "scene_id": self.scene, "runtime_episode_id": self.episode,
                        "source_episode_ids": None, "ordered_categories": categories}
            if self.identity is not None:
                identity.update(self.identity)
            if hasattr(self.environment, "policy_identity"):
                identity = self.environment.policy_identity(identity)
            if (identity["scene_id"], identity["runtime_episode_id"], list(identity["ordered_categories"])) != (
                    self.scene, self.episode, categories):
                raise ValueError("Trajectory identity differs from the observed episode tasks")
            target = self.directory / "episodes" / component(self.scene) / component(self.episode)
            target.mkdir(parents=True, exist_ok=False)
            payload = {"schema_version": SCHEMA, "identity": identity,
                       "coordinate_frame": "habitat_world_xyz_y_up",
                       "pose_format": "qw_qx_qy_qz_x_y_z", "pose_observed": "before_action",
                       "action_space": self.action_space}
            with (target / "identity.json").open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, allow_nan=False)
                stream.write("\n")
            self.output = target / "trajectory.jsonl"
        return observation

    def apply_action(self, action, info=None, **kwargs):
        if self.pose is None or self.output is None:
            raise RuntimeError("Trajectory action requires an observation after reset")
        task = task_index(self.environment)
        value = getattr(action, "value", action)
        if isinstance(value, bool):
            raise ValueError("Trajectory action ID cannot be a boolean")
        action_id = operator.index(value)
        action_name = getattr(action, "name", next(
            (name for name, code in self.action_space.items() if code == action_id), str(action_id)))
        is_stop = bool(action == self.actions.STOP)
        row = {"schema_version": SCHEMA, "scene_id": self.scene, "episode_id": self.episode,
               "subtask_index": task, "frame_index": self.frame + 1,
               "task_step": self.task_steps.get(task, 0) + 1,
               "action_id": action_id, "action_name": str(action_name),
               "is_stop": is_stop, "stop_reason": self.action_metadata().get("stop_reason") if is_stop else None,
               "world_pose": list(self.pose)}
        target = self.action_metadata().get('blp_action_target')
        if target is not None:
            if target['task_idx'] != task or target['frame_index'] != self.frame + 1:
                raise ValueError('BLP target and accepted-action trajectory disagree')
            row['blp_action_target'] = json.loads(json.dumps(target, allow_nan=False))
        encoded = json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
        result = self.environment.apply_action(action, info=info, **kwargs)
        end_pose = getattr(self.environment, 'action_end_pose', None)
        if end_pose is not None:
            end_pose = [float(value) for value in end_pose]
            if len(end_pose) != 7 or not all(math.isfinite(value) for value in end_pose):
                raise ValueError('Native action end pose must be finite')
            row['world_pose_after'] = end_pose
            encoded = json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n'
        with self.output.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
        self.frame += 1
        self.task_steps[task] = row["task_step"]
        self.pose = None
        return result
