"""Camera, waypoint and local-grid geometry for online navigation."""

# Portions adapted from LagMemo, Copyright (c) Meta Platforms, Inc. and affiliates.
# Upstream MIT terms and provenance are retained in THIRD_PARTY_NOTICES.md.

import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def sample_points(x_offsets, y_offsets, max_points=1000):
    n = len(x_offsets)
    if n <= max_points:
        return (x_offsets, y_offsets)
    indices = np.random.choice(n, max_points, replace=False)
    return (x_offsets[indices], y_offsets[indices])


def compute_transform_params(global_pose, local_pose):
    (x_g, y_g, alpha_g) = global_pose
    (x_l, y_l, alpha_l) = local_pose
    theta_rad = alpha_g - alpha_l
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    A = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    global_pose_rot = (A @ np.array([x_g, y_g]).T).T
    pose_trans = np.array([x_l, y_l]) - global_pose_rot
    (tx, ty) = (pose_trans[0], pose_trans[1])
    return (theta_rad, tx, ty)


def compute_transform_params_v2(global_pose, local_pose):
    (qx, qy, qz, qw) = global_pose[:4]
    quat = np.array([qx, qy, qz, qw])
    q = R.from_quat(quat)
    init_global_pose = np.array(global_pose[4:])
    point_world = q.inv().apply(init_global_pose)
    tran = point_world - local_pose
    rot_matrix = R.from_quat(quat).as_matrix().T
    return (rot_matrix, tran)


def compute_td_params(gp1, gp2, lp1, lp2, theta_rad):
    (x_g1, y_g1) = gp1
    (x_g2, y_g2) = gp2
    (x_l1, y_l1) = lp1
    (x_l2, y_l2) = lp2
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    A = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    gp1_rot = (A @ np.array([x_g1, y_g1]).T).T
    gp2_rot = (A @ np.array([x_g2, y_g2]).T).T
    k = (lp1[1] - lp2[1]) / (gp1_rot[0] - gp2_rot[0])
    k_2 = (lp1[0] - lp2[0]) / (gp1_rot[1] - gp2_rot[1])
    b1 = lp1[0] - k_2 * gp1_rot[1]
    b2 = lp1[1] - k * gp1_rot[0]
    tran_matrix = np.array([[0, k_2, b1], [k, 0, b2], [0, 0, 1]])
    return tran_matrix


def global_to_gps2(global_point, rot_matrix, tran):
    local_pos = rot_matrix @ global_point - tran
    return np.array([-local_pos[2], -local_pos[0]])


def global_to_gps(global_point, transform_parameters):
    (theta_rad, tx, ty, reversed) = transform_parameters
    (x_g, y_g) = global_point
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    A = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    pose_rot = (A @ np.array([x_g, y_g]).T).T
    pose_local = pose_rot + np.array([tx, ty])
    if reversed == 0:
        pose_local_ret = np.array([-pose_local[1], -pose_local[0]])
    else:
        pose_local_ret = np.array([pose_local[1], pose_local[0]])
    return pose_local_ret


def draw_circle(array, x, y, radius):
    """Fill a circle in place, touching only its clipped bounding box."""
    (rows, cols) = array.shape
    if radius < 0 or not rows or not cols:
        return array
    bounds = (x - radius, x + radius, y - radius, y + radius)
    if not all(math.isfinite(value) for value in (*bounds, x, y, radius)):
        xx, yy = np.mgrid[:rows, :cols]
        distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        array[distance <= radius] = 1
        return array
    row_start = max(0, math.floor(bounds[0]))
    row_end = min(rows, math.ceil(bounds[1]) + 1)
    col_start = max(0, math.floor(bounds[2]))
    col_end = min(cols, math.ceil(bounds[3]) + 1)
    if row_start >= row_end or col_start >= col_end:
        return array
    xx, yy = np.ogrid[row_start:row_end, col_start:col_end]
    distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    view = array[row_start:row_end, col_start:col_end]
    view[distance <= radius] = 1
    return array


def draw_goal_circle_to_explored(goal_map, explored_map, x, y, start_radius, min_overlap=20):
    """Draw a finite goal region overlapping the nearest explored pixels.

    The previous callers repeatedly increased the radius until at least
    min_overlap explored pixels were covered. If the transformed goal fell
    outside the local map, or the explored map contained too few pixels,
    that loop had no termination condition and could spin forever.
    """
    if goal_map.shape != explored_map.shape:
        raise ValueError(
            f"goal/explored map shape mismatch: {goal_map.shape} != {explored_map.shape}"
        )
    center = np.asarray([x, y], dtype=np.float64)
    (rows, cols) = goal_map.shape
    if not np.isfinite(center).all():
        center = np.asarray(
            [
                np.clip(np.nan_to_num(center[0]), 0, rows - 1),
                np.clip(np.nan_to_num(center[1]), 0, cols - 1),
            ],
            dtype=np.float64,
        )
    explored_pixels = np.argwhere(np.asarray(explored_map) > 0)
    if explored_pixels.size == 0:
        clipped = np.asarray([np.clip(center[0], 0, rows - 1), np.clip(center[1], 0, cols - 1)])
        used_radius = int(max(1, start_radius))
        goal_map = draw_circle(goal_map, clipped[0], clipped[1], used_radius)
        return (goal_map, used_radius + 2, 0)
    target_overlap = min(int(min_overlap), len(explored_pixels))
    distances = np.linalg.norm(explored_pixels - center, axis=1)
    nearest_radius = np.partition(distances, target_overlap - 1)[target_overlap - 1]
    used_radius = int(np.ceil(max(float(start_radius), nearest_radius)))
    goal_map = draw_circle(goal_map, center[0], center[1], used_radius)
    overlap = int(np.count_nonzero((goal_map > 0) & (np.asarray(explored_map) > 0)))
    return (goal_map, used_radius + 2, overlap)


def get_l2_distance(x1, x2, y1, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_rel_pose_change(pos2, pos1):
    (x1, y1, o1) = pos1
    (x2, y2, o2) = pos2
    theta = np.arctan2(y2 - y1, x2 - x1) - o1
    dist = get_l2_distance(x1, x2, y1, y2)
    dx = dist * np.cos(theta)
    dy = dist * np.sin(theta)
    do = o2 - o1
    return (dx, dy, do)


class GoalProjection:

    def mask_to_relative_2d_vectorized(self, mask, depth_image, hfov_deg, image_width, current_o):
        """
        将seem mask转化为2维坐标偏移
        :param mask: 二维布尔数组，True 表示需要计算的像素
        :param depth_image: 深度图像（H x W），与 mask 同尺寸
        :param hfov_deg: 水平视场角（度）
        :param image_width: 图像宽度（像素）
        :param current_o: 当前朝向（角度）
        :return: 两个数组 (x_offsets, y_offsets)，表示相对于当前坐标系的偏移量
        """
        (v_indices, u_indices) = np.where(mask)
        hfov_rad = np.radians(hfov_deg)
        cx = image_width / 2
        tan_half_hfov = np.tan(hfov_rad / 2)
        u = u_indices.astype(float)
        v = v_indices.astype(float)
        d = depth_image[v_indices, u_indices]
        theta_x_rad = np.arctan2(2 * (u - cx) * tan_half_hfov, image_width)
        x_offsets_global = d * np.cos(theta_x_rad)
        y_offsets_global = d * np.sin(theta_x_rad)
        current_o_rad = current_o
        cos_o = np.cos(current_o_rad)
        sin_o = np.sin(current_o_rad)
        x_offsets = x_offsets_global * cos_o + y_offsets_global * sin_o
        y_offsets = -(-x_offsets_global * sin_o + y_offsets_global * cos_o)
        (x_offsets, y_offsets) = sample_points(x_offsets, y_offsets, max_points=1000)
        mask_pos = np.column_stack((x_offsets, y_offsets))
        norms = np.linalg.norm(mask_pos, axis=1)
        filtered_points = mask_pos[norms > 0.2]
        return filtered_points

    def process_relative_poses(self, mask_pos, planner_inputs):
        """
        对每个 relative_pose 进行处理，并更新 planner_inputs
        :param x_offsets: x 偏移量数组
        :param y_offsets: y 偏移量数组
        :param planner_inputs: 包含目标地图等信息的字典
        :return: 更新后的 planner_inputs
        """
        planner_inputs[0]["goal_map"] = np.zeros_like(planner_inputs[0]["goal_map"])
        min_relative_pose = 0.6
        cur_self_pixel_pose = planner_inputs[0]["sensor_pose"][:2] * 20 - np.array(
            [planner_inputs[0]["sensor_pose"][5], planner_inputs[0]["sensor_pose"][3]]
        )
        for relative_pose in mask_pos:
            planner_goal_pose = np.arctan2(relative_pose[1], relative_pose[0])
            relative_pixel_pose = np.around(relative_pose * (100 / 5)).astype(np.int32)
            goal_pixel_pose = cur_self_pixel_pose + relative_pixel_pose
            planner_inputs[0]["found_goal"] = True
            if np.linalg.norm(relative_pose) < min_relative_pose:
                min_relative_pose = np.linalg.norm(relative_pose)
                planner_inputs[0]["goal_pose"] = math.degrees(planner_goal_pose)
            planner_inputs[0]["goal_map"] = draw_circle(
                planner_inputs[0]["goal_map"], goal_pixel_pose[1], goal_pixel_pose[0], 3
            )
        return planner_inputs
