from typing import Tuple

import numpy as np

from sim_utils.camera import get_camera_intrinsic_matrix


def get_camera_transform_matrices(
    width: int, height: int, vertical_fov: float, xpos: np.ndarray, xmat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the camera matrices.

    Args:
        width (int): Width of the rendered image.
        height (int): Height of the rendered image.
        vertical_fov (float): Vertical field of view in radians.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of intrinsic (3x3) and
            extrinsic (3x4) cameras.
    """
    intrinsic_matrix = get_camera_intrinsic_matrix(
        width=width,
        height=height,
        vertical_fov=np.radians(vertical_fov),
    )
    rotation = xmat.reshape(3, 3)
    extrinsic_matrix = np.c_[rotation, -rotation @ xpos.reshape(-1, 1)]
    return intrinsic_matrix, extrinsic_matrix
