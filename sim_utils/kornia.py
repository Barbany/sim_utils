import torch

from kornia.geometry.camera.pinhole import PinholeCamera


def get_kornia_pinhole_camera(
    intrinsics_np, extrinsics_np, height=None, width=None, device="cpu"
):
    if height is None:
        height = intrinsics_np[1][2] * 2
    if width is None:
        width = intrinsics_np[0][2] * 2

    height = torch.tensor([height])
    width = torch.tensor([width])

    intrinsics = torch.eye(4)
    intrinsics[:3, :3] = torch.from_numpy(intrinsics_np)

    extrinsics = torch.eye(4)
    extrinsics[:3, :4] = torch.from_numpy(extrinsics_np)

    return PinholeCamera(
        *float_in_device(
            device, intrinsics.unsqueeze(0), extrinsics.unsqueeze(0), height, width
        )
    )


def float_in_device(device, *args):
    for arg in args:
        yield arg.float().to(device)
