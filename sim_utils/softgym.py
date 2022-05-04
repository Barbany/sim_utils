from typing import Tuple

import numpy as np
from softgym.envs.flex_env import FlexEnv

from sim_utils.camera import (
    get_camera_extrinsic_matrix_euler,
    get_camera_intrinsic_matrix,
)


def get_camera_transform_matrices(
    env: FlexEnv, fov: float = np.pi / 4
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the camera matrices from a mujoco_py camera converting world coordinates to image coordinates.

    Args:
        env (FlexEnv): PyFleX base environment which has the attributes of width, height and camera.
        horizontal_fov (float, optional): Horizontal field of view in radians.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of intrinsic (3x3) and extrinsic (3x4) cameras.
    """
    intrinsic_matrix = get_camera_intrinsic_matrix(
        width=env.camera_width,
        height=env.camera_height,
        vertical_fov=fov,
    )
    cam_pos, cam_angles = env.get_camera_params()
    extrinsic_matrix = get_camera_extrinsic_matrix_euler(
        cam_pos=cam_pos,
        cam_angles=cam_angles,
    )
    return intrinsic_matrix, extrinsic_matrix
