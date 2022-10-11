from typing import List

import matplotlib.pyplot as plt
import numpy as np


def project_points_to_image(
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    points: np.ndarray,
    image: np.ndarray,
    c=None,
):
    """Project points to image given the camera intrinsic and extrinsic matrices.

    Args:
       intrinsics (np.ndarray): 3x3 intrinsic camera matrix. 
       extrinsics (np.ndarray): 3x4 extrinsic camera matrix. 
       points (np.ndarray): Nx3 array of points.
       image (np.ndarray): Image.
    """
    plt.imshow(image)
    proj_points = intr @ ext @ np.r_[points.T, np.ones((1, points.shape[0]))]
    proj_points = proj_points / proj_points[-1]
    plt.scatter(proj_points[0], proj_points[1], c=c)


def visualize_extrinsics(
    extrinsics: List[np.ndarray], lookat: np.ndarray = None, scale: float = 1
):
    """Visualize frames of cameras given their extrinsic matrices.
    RGB indicates XYZ, respectively.
    Args:
        extrinsics (List[np.ndarray]): List of 4x3 extrinsic matrices.
        lookat (np.ndarray, optional): Point where cameras are looking at.
        scale (float, optional): Scale of the frame (<1 to make axes shorter, >1 to make them longer).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if lookat is not None:
        ax.scatter(lookat[0], lookat[1], lookat[2])

    origin = np.zeros((4, 1))
    origin[-1] = 1
    basis = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale], [1, 1, 1]])

    colors = ["r", "g", "b"]

    for ext in extrinsics:
        c2w = np.r_[ext, origin.copy().T]
        inv_c2w = np.linalg.inv(c2w)
        new_origin = inv_c2w @ origin
        new_basis = inv_c2w @ basis

        for i in range(3):
            plt.plot(
                [new_origin[0, 0], new_basis[0, i]],
                [new_origin[1, 0], new_basis[1, i]],
                [new_origin[2, 0], new_basis[2, i]],
                colors[i],
                linewidth=3,
            )

    plt.show()
