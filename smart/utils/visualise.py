from typing import Any, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops.math_ops import Sum
import torch
from matplotlib import animation
from matplotlib import patches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos.scenario_pb2 import Scenario
from waymo_open_dataset.utils.sim_agents import visualizations, submission_specs
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.utils import trajectory_utils
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics


n = 100
COLOR_DICT = {i: np.array(plt.cm.viridis(i / n)) for i in range(n)}

def draw_gif(
        predicted_num: int, traj: np.ndarray, real_yaw: np.ndarray,
        data_dict: Dict[str, Any], map: np.array, image_path: str, return_animations=False
):
    fig, axis = plt.subplots(1, 1, figsize=(10, 10))
    # the scenario is the map. it is made of positions and we plot all ofthem
    axis.plot(map[:, 0], map[:, 1], 'k--', alpha=0.5)
    # visualizations.get_bbox_patch()
    # axis.axis('equal')  # 横纵坐标比例相等
    x_list = list()
    y_list = list()
    yaw_list = list()
    for i in range(predicted_num):
        real_traj_x = traj[i, :, 0]
        real_traj_y = traj[i, :, 1]
        real_traj_yaw = real_yaw[i]
        x_list.append(real_traj_x)
        y_list.append(real_traj_y)
        yaw_list.append(real_traj_yaw)
    # [num, step]
    x_list = np.stack(x_list, axis=0)
    y_list = np.stack(y_list, axis=0)
    yaw_list = np.stack(yaw_list, axis=0)

    def animate(t: int) -> list[patches.Rectangle]:
        # At each animation step, we need to remove the existing patches. This can
        # only be done using the `pop()` operation.
        for _ in range(len(axis.patches)):
            axis.patches.pop()
        bboxes = list()
        for j in range(x_list.shape[0]):
            bboxes.append(axis.add_patch(
                get_bbox_patch(
                    x_list[:, t][j], y_list[:, t][j], yaw_list[:, t][j],
                    5,2, COLOR_DICT[j]
                )
            ))
        return bboxes

    animations = animation.FuncAnimation(
        fig, animate, frames=x_list.shape[1], interval=100,
        blit=True)
    axis.set_xticks([])
    axis.set_yticks([])
    # plt.show()
    if return_animations:
        return animations
    animations.save(image_path, writer='ffmpeg', fps=30)
    plt.close('all')  # 避免内存泄漏
    print(f"{image_path} save success")
    return animations, fig

def get_bbox_patch(
        x: float, y: float, bbox_yaw: float, length: float, width: float,
        color: np.ndarray
) -> patches.Rectangle:
    left_rear_object = np.array([-length / 2, -width / 2])

    rotation_matrix = np.array([[np.cos(bbox_yaw), -np.sin(bbox_yaw)],
                                [np.sin(bbox_yaw), np.cos(bbox_yaw)]])
    left_rear_rotated = rotation_matrix.dot(left_rear_object)
    left_rear_global = np.array([x, y]) + left_rear_rotated
    color = list(color)
    rect = patches.Rectangle(
        left_rear_global, length, width, angle=np.rad2deg(bbox_yaw), color=color)
    return rect
