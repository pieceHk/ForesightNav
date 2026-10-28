# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skfmm
import skimage
from numpy import ma


class PlannerError(RuntimeError):
    pass


class FMMPlanner:
    """
    Fast Marching Method Planner.
    This is just the core FMM logic.
    """

    def __init__(
        self,
        traversible: np.ndarray,
        scale: int = 1,
        step_size: int = 5,
        goal_tolerance: float = 2.0,
        vis_dir: str = "data/images/planner",
        visualize=False,
        print_images=True,
        debug=False,
        distance_cache=None,
    ):
        """
        Arguments:
            traversible: (M + 1, M + 1) binary map encoding traversible regions
            scale: map scale
            step_size: maximum distance of the short-term goal selected by the
             planner
            vis_dir: folder where to dump visualization
        """
        self.visualize = visualize
        self.print_images = print_images
        self.vis_dir = vis_dir
        os.makedirs(self.vis_dir, exist_ok=True)

        self.scale = scale
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        if scale != 1.0:
            self.traversible = cv2.resize(
                traversible,
                (traversible.shape[1] // scale, traversible.shape[0] // scale),
                interpolation=cv2.INTER_NEAREST,
            )
            self.traversible = np.rint(self.traversible)
        else:
            self.traversible = traversible

        self.du = int(self.step_size / (self.scale * 1.0))
        self.fmm_dist = None
        self.debug = debug
        self.distance_cache = distance_cache
        # self.goal_map = None

    def set_goal(self, goal, auto_improve: bool = False):
        """Set planner goal. Goal should be of size 2, containing x and y positions."""
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        goal_x, goal_y = int(goal[0] / (self.scale * 1.0)), int(
            goal[1] / (self.scale * 1.0)
        )
        if not (
            0 <= goal_x < self.traversible.shape[0]
            and 0 <= goal_y < self.traversible.shape[1]
        ):
            raise PlannerError(f"goal_out_of_bounds:{goal_x},{goal_y}")

        if self.traversible[goal_x, goal_y] == 0.0 and auto_improve:
            goal_x, goal_y = self._find_nearest_goal([goal_x, goal_y])

        traversible_ma[goal_x, goal_y] = 0
        try:
            dd = self._distance(traversible_ma, dx=1)
        except Exception as exc:
            raise PlannerError(f"fmm_distance_failed:{exc}") from exc
        dd = self._filled_distance(dd)
        self.fmm_dist = dd
        return

    def _distance(self, level_set, *, dx):
        if self.distance_cache is None:
            return skfmm.distance(level_set, dx=dx)
        point_source = getattr(self.distance_cache, 'point_source', None)
        values = np.ma.getdata(level_set)
        if point_source is not None and values.ndim == 2 and min(values.shape) >= 3:
            if not np.any(np.ma.getmaskarray(level_set)):
                sources = np.flatnonzero(values == 0)
                if len(sources) == 1 and np.count_nonzero(values != 1) == 1:
                    goal = np.unravel_index(sources[0], values.shape)
                    return point_source(values.shape, goal, dtype=values.dtype,
                                        dx=dx, solver=skfmm.distance)
        return self.distance_cache.compute(level_set, dx=dx, solver=skfmm.distance)

    @staticmethod
    def _filled_distance(dd: np.ndarray) -> np.ndarray:
        if dd is None:
            raise PlannerError("fmm_distance_missing")
        compressed = ma.asarray(dd).compressed()
        if compressed.size == 0:
            raise PlannerError("fmm_distance_all_masked")
        fill_value = float(np.max(compressed)) + 1.0
        filled = np.asarray(ma.filled(dd, fill_value), dtype=np.float32)
        if not np.isfinite(filled).all():
            raise PlannerError("fmm_distance_non_finite")
        return filled

    def set_multi_goal(
        self,
        goal_map: np.ndarray,
        timestep: int = 0,
        dd: Optional[np.ndarray] = None,
        map_downsample_factor: float = 1.0,
        map_update_frequency: int = 1,
    ):
        """Set long-term goal(s) used to compute distance from a binary
        goal map.
        dd: distance map for when we want to reuse previously computed ones (instead of updating at each step)
        map_update_frequency: skfmm.distance call made every n steps
        map_downsample_factor: 1 for no downsampling, 2 for halving both image dimensions.
        """
        assert map_downsample_factor >= 1.0
        if goal_map is None or not np.any(np.asarray(goal_map) > 0):
            raise PlannerError("empty_goal_map")
        traversible = self.traversible
        if map_downsample_factor > 1.0:
            l, w = self.traversible.shape
            traversible = cv2.resize(
                traversible,
                dsize=(int(l / map_downsample_factor), int(w / map_downsample_factor)),
            )
            print(f"Downsampling goal and traversible maps {map_downsample_factor}x.")
            goal_map_copy = goal_map.copy()
            goal_map = cv2.resize(
                goal_map,
                dsize=(int(l / map_downsample_factor), int(w / map_downsample_factor)),
                interpolation=cv2.INTER_NEAREST,
            )

            if goal_map.sum() == 0:
                # dilating goal map so as to not lose pixels when resizing
                kernel = np.ones((2, 2), np.uint8)
                goal_map = cv2.dilate(goal_map_copy, kernel, iterations=1)
                goal_map = cv2.resize(
                    goal_map,
                    dsize=(
                        int(l / map_downsample_factor),
                        int(w / map_downsample_factor),
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )
            if goal_map.sum() == 0:
                raise PlannerError("empty_goal_map_after_downsample")

        traversible_ma = ma.masked_values(traversible * 1, 0)
        traversible_ma[goal_map > 0] = 0

        # This is where we actually call the FMM algorithm!!
        # It will compute the distance from each traversible point to the goal.
        if (timestep - 1) % map_update_frequency == 0 or dd is None:
            try:
                dd = self._distance(traversible_ma, dx=1 * map_downsample_factor)
            except Exception as exc:
                raise PlannerError(f"fmm_distance_failed:{exc}") from exc
            dd = self._filled_distance(dd)
            if self.debug:
                print(f"Computing skfmm.distance (timestep: {timestep})")
        else:
            if self.debug:
                print(f"Reusing previous skfmm.distance value (timestep: {timestep})")

        if map_downsample_factor > 1.0:
            dd = cv2.resize(dd, (l, w))  # upsampling
            if not np.isfinite(dd).all():
                raise PlannerError("upsampled_fmm_distance_non_finite")

        self.fmm_dist = dd
        # self.goal_map = goal_map

        if self.visualize or self.print_images and timestep != 0:
            r, c = traversible.shape  # for visualizing (downsampled) traversible map
            dist_vis = np.zeros((r, c * 3))
            dist_vis[:, :c] = np.flipud(traversible)
            dist_vis[:, c : 2 * c] = np.flipud(goal_map)
            dist_vis[:, 2 * c :] = np.flipud(self.fmm_dist / self.fmm_dist.max())

            if self.visualize:
                cv2.imshow("Planner Distance", dist_vis)
                cv2.waitKey(1)

            if self.print_images and timestep is not None:
                cv2.imwrite(
                    os.path.join(self.vis_dir, f"planner_snapshot_{timestep}.png"),
                    (dist_vis * 255).astype(int),
                )
        return dd

    def get_short_term_goal(self, state: List[float], continuous=True, timestep=0):
        """Compute the short-term goal closest to the current state.

        Arguments:
            state: current location
        """
        if self.fmm_dist is None:
            raise PlannerError("short_term_goal_without_distance_map")
        scale = self.scale * 1.0
        state = [x / scale for x in state]
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
        mask = FMMPlanner.get_mask(
            dx, dy, scale, self.step_size, min_radius=0 if continuous else None
        )
        dist_mask = FMMPlanner.get_dist(dx, dy, scale, self.step_size)

        state = [int(x) for x in state]

        dist = np.pad(
            self.fmm_dist,
            self.du,
            "constant",
            constant_values=self.fmm_dist.shape[0] ** 2,
        )
        subset = dist[
            state[0] : state[0] + 2 * self.du + 1, state[1] : state[1] + 2 * self.du + 1
        ]

        assert (
            subset.shape[0] == 2 * self.du + 1 and subset.shape[1] == 2 * self.du + 1
        ), "Planning error: unexpected subset shape {}".format(subset.shape)

        visualize = False
        if visualize:
            # TODO
            plt.subplot(231)
            plt.imshow(subset)

        subset *= mask
        subset += (1 - mask) * self.fmm_dist.shape[0] ** 2

        if visualize:
            plt.subplot(232)
            plt.imshow(subset)
            plt.subplot(235)
            plt.imshow(mask)

        if self.debug:
            print(
                "[FMM] Distance to fmm navigable goal pt =",
                subset[self.du, self.du] * 5,
            )
        stop = subset[self.du, self.du] < self.goal_tolerance
        if self.debug:
            print("subset[self.du, self.du]", subset[self.du, self.du])
            print("self.goal_tolerance", self.goal_tolerance)
            print("stop", stop)
            print()

        subset -= subset[self.du, self.du]
        ratio1 = subset / dist_mask
        subset[ratio1 < -1.5] = 1

        if visualize:
            plt.subplot(233)
            plt.imshow(subset)
            plt.show()

        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)

        # Subset will contain negative distance to goal
        replan = subset[stg_x, stg_y] > -0.0001

        return (
            (stg_x + state[0] - self.du) * scale,
            (stg_y + state[1] - self.du) * scale,
            replan,
            stop,
        )

    @staticmethod
    def get_mask(sx, sy, scale, step_size, min_radius=None):
        """Set everything in a circle around the agent to 1; else set to zero"""
        if min_radius is None:
            min_radius = (step_size - 1) ** 2
        size = int(step_size // scale) * 2 + 1
        mask = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                cond1 = (
                    ((i + 0.5) - (size // 2 + sx)) ** 2
                    + ((j + 0.5) - (size // 2 + sy)) ** 2
                ) <= step_size**2
                cond2 = (
                    ((i + 0.5) - (size // 2 + sx)) ** 2
                    + ((j + 0.5) - (size // 2 + sy)) ** 2
                ) > min_radius
                if cond1 and cond2:
                    mask[i, j] = 1
        mask[size // 2, size // 2] = 1
        return mask

    @staticmethod
    def get_dist(sx, sy, scale, step_size):
        size = int(step_size // scale) * 2 + 1
        mask = np.zeros((size, size)) + 1e-10
        for i in range(size):
            for j in range(size):
                if (
                    ((i + 0.5) - (size // 2 + sx)) ** 2
                    + ((j + 0.5) - (size // 2 + sy)) ** 2
                ) <= step_size**2:
                    mask[i, j] = max(
                        5,
                        (
                            ((i + 0.5) - (size // 2 + sx)) ** 2
                            + ((j + 0.5) - (size // 2 + sy)) ** 2
                        )
                        ** 0.5,
                    )
        return mask

    def _find_within_distance_to_multi_goal(
        self,
        goal: np.ndarray,
        distance: float,
        min_distance_only=False,
        visualize=False,
        timestep=0,
        vis_dir=None,
    ) -> np.ndarray:
        """
        Find the nearest point to a goal which is traversible
        """

        if vis_dir is not None:
            self.vis_dir = vis_dir
        planner = FMMPlanner(
            np.ones_like(self.traversible),
            print_images=self.print_images,
            vis_dir=self.vis_dir,
            distance_cache=self.distance_cache,
        )
        # Plan to the goal mask
        planner.set_multi_goal(goal, timestep=timestep)

        # Now mask out anything here based on distance to the goal mask
        mask = self.traversible
        dist_map = planner.fmm_dist * mask
        dist_map[dist_map == 0] = dist_map.max()

        if min_distance_only:
            min_dist_idx = dist_map.argmin()
            goal_pt = np.unravel_index(min_dist_idx, dist_map.shape)
            navigable_goal_map = np.zeros_like(goal)
            navigable_goal_map[goal_pt[0], goal_pt[1]] = 1
        else:
            navigable_goal_map = dist_map < distance

        if visualize:
            # Debugging code. Make sure we are properly finding the closest traversible goal.
            plt.subplot(221)
            plt.imshow(self.traversible)
            plt.subplot(222)
            plt.imshow(dist_map)
            plt.subplot(223)
            plt.imshow(navigable_goal_map)
            plt.subplot(224)
            plt.imshow(goal)
            plt.show()
        return navigable_goal_map
