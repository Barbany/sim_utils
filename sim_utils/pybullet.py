from typing import Tuple

import numpy as np
from scipy.linalg import rq


def get_camera_transform_matrices(
    width: int, height: int, view_matrix: tuple, projection_matrix: tuple
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the camera matrices.

    Args:
        width (int): Width of the rendered image.
        height (int): Height of the rendered image.
        view_matrix (tuple): Tuple of floats with length 16 representing the
            OpenGL view matrix.
        projection_matrix (tuple): Tuple of floats with length 16 representing the
            OpenGL projection matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of intrinsic (3x3) and
            extrinsic (3x4) cameras.

    References:
        Process to convert camera matrix to intrinsic and extrinsic matrices
        taken from the blogpost
        https://ksimek.github.io/2012/08/14/decompose/
    """
    view_matrix = np.array(view_matrix).reshape(4, 4).T
    projection_matrix = np.array(projection_matrix).reshape(4, 4).T
    # The transformation given by projection_matrix @ view_matrix yields [x, y, z, w]
    # w is the normalization factor and x/w and y/w are the normalized pixel
    # coordinates in [-1, 1]
    # extra_matrix gets rid of z (depth unwanted) and scales pixel coordinates to
    # [0, height/width] as well as flips the values along the horizontal axis
    extra_matrix = np.array(
        [[height / 2, 0, 0, height / 2], [0, -width / 2, 0, width / 2], [0, 0, 0, 1]]
    )
    camera_matrix = extra_matrix @ projection_matrix @ view_matrix

    # Express camera_matrix as [m | -mc] with m a 3x3 matrix
    # and c the camera center
    m = camera_matrix[:, :3]
    camera_center = np.linalg.inv(m) @ camera_matrix[:, -1]

    # Find intrinsic and extrinsic matrix with RQ decomposition
    # RQ decomposition is not unique; swap axes s.t. intrinsic values are positive
    r, q = rq(m)
    for i in range(len(r)):
        if any(r[:, i] < 0):
            r[:, i] = -r[:, i]
            q[i, :] = -q[i, :]
    intrinsic_matrix = r
    extrinsic_matrix = np.c_[q, q @ camera_center]
    assert np.allclose(intrinsic_matrix @ extrinsic_matrix, camera_matrix)
    return intrinsic_matrix, extrinsic_matrix
