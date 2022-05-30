import numpy as np


def get_camera_extrinsic_matrix_lookat(
    azimuth: float, distance: float, elevation: float, lookat: np.ndarray
) -> np.ndarray:
    """Get the camera extrinsic matrix converting world coordinates to the camera referential
    according to the OpenGL convention (see references).
    Compute the parameters using a look at camera.

    Args:
        azimuth (float): Azimuth angle in radians.
        distance (float): Distance to point to which the camera is looking at.
        elevation (float): Elevation angle in radians.
        lookat (np.ndarray): 3D position (expressed in world coordinates)
            of the point at which the camera is looking at.

    Returns:
        np.ndarray: 3x4 extrinsic matrix.

    References:
        https://learnopengl.com/Getting-started/Camera
    """
    rotation_matrix = (
        get_rotation_matrix(np.pi, [0, 0, 1])
        @ get_rotation_matrix(elevation + np.pi / 2, [1, 0, 0])
        @ get_rotation_matrix(azimuth + np.pi / 2, [0, 0, 1])
    )
    # We want the lookat point to be [0, 0, -distance] w.r.t. the camera frame
    # Note that in OpenGL convention, camera is pointing towards the -z axis.
    translation_vector = np.array([0, 0, -distance]) - rotation_matrix @ lookat
    return np.c_[rotation_matrix, translation_vector]


def get_camera_extrinsic_matrix_euler(
    cam_pos: np.ndarray, cam_angles: np.ndarray
) -> np.ndarray:
    """Get the camera extrinsic matrix converting world coordinates to the camera referential
    according to the OpenGL convention (see references).
    Compute the parameters using the euler angles and camera position.

    Args:
        cam_pos (np.ndarray): Position of the camera in terms of the world coordinates.
            NumPy array of shape 3 containing [x, y, z].
        cam_angles (np.ndarray): Angle between each axis in the camera frame and the world frame.
            NumPy array of shape 3 containing [cam_x_angle, cam_y_angle, cam_z_angle].

    Returns:
        np.ndarray: Camera extrinsics matrix

    References:
        https://learnopengl.com/Getting-started/Camera
    """
    cam_x_angle, cam_y_angle, _ = cam_angles
    matrix1 = get_rotation_matrix(-cam_x_angle, [0, 1, 0])
    matrix2 = get_rotation_matrix(-cam_y_angle - np.pi, [1, 0, 0])
    rotation_matrix = matrix2 @ matrix1
    return np.c_[rotation_matrix, -cam_pos]


def get_camera_intrinsic_matrix(
    width: int, height: int, horizontal_fov: float = None, vertical_fov: float = None
) -> np.ndarray:
    """Get the camera intrinsic matrix converting camera coordinates to image coordinates.
    If one of the fields of view is not specified, it will be computed using the image dimensions.

    Args:
        width (int): Width of the rendered image.
        height (int): Height of the rendered image.
        horizontal_fov (float, optional): Horizontal field of view in radians.
        vertical_fov (float, optional): Vertical field of view in radians.

    Returns:
        np.ndarray: 3x3 intrinsic matrix.

    Raises:
        AttributeError: If any of the field of views are specified.
    """
    if horizontal_fov is None:
        if vertical_fov is None:
            raise AttributeError("You must specify the FOV in one or the two axes")
        else:

            horizontal_fov = 2.0 * np.arctan(np.tan(vertical_fov / 2) * width / height)
    else:
        if vertical_fov is None:
            vertical_fov = 2.0 * np.arctan(np.tan(horizontal_fov / 2) * height / width)

    px, py = (width / 2, height / 2)
    fy = height / (2.0 * np.tan(vertical_fov / 2.0))
    fx = width / (2.0 * np.tan(horizontal_fov / 2.0))
    return np.array([[fx, 0, px], [0, fy, py], [0, 0, 1.0]])


def get_rotation_matrix(angle: float, axis) -> np.ndarray:
    """Get the rotation matrix encoding a rotation along an arbitrary axis.

    Args:
        angle (float): Angle of rotation in radians.
        axis (array-like): Axis of rotation (possibly unnormalized).

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle)
    c = np.cos(angle)

    return np.array(
        [
            [
                axis[0] * axis[0] + (1.0 - axis[0] * axis[0]) * c,
                axis[0] * axis[1] * (1.0 - c) - axis[2] * s,
                axis[0] * axis[2] * (1.0 - c) + axis[1] * s,
            ],
            [
                axis[0] * axis[1] * (1.0 - c) + axis[2] * s,
                axis[1] * axis[1] + (1.0 - axis[1] * axis[1]) * c,
                axis[1] * axis[2] * (1.0 - c) - axis[0] * s,
            ],
            [
                axis[0] * axis[2] * (1.0 - c) - axis[1] * s,
                axis[1] * axis[2] * (1.0 - c) + axis[0] * s,
                axis[2] * axis[2] + (1.0 - axis[2] * axis[2]) * c,
            ],
        ]
    )
