import torch
import torch.nn.functional as F


def pose_to_d9(pose: torch.Tensor) -> torch.Tensor:
    nbatch = pose.shape[0]
    R = pose[:, :3, :3]  # [N, 3, 3]
    t = pose[:, :3, -1]  # [N, 3]

    r6 = R[:, :2, :3].reshape(nbatch, -1)  # [N, 6]

    d9 = torch.cat((t, r6), -1)  # [N, 9]

    return d9


def r6d2mat(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def look_at_rotation(camera_position, at=(0, 0, 0), up=(0., 0, 1), device: str = "cpu") -> torch.Tensor:
    # Format input and broadcast
    nbatch = camera_position.shape[0]
    camera_position = camera_position.to(device)
    if not torch.is_tensor(at):
        at = torch.tensor(at, dtype=torch.float32, device=device)
    at = at.expand(nbatch, 3)
    if not torch.is_tensor(up):
        up = torch.tensor(up, dtype=torch.float32, device=device)
    up = up.expand(nbatch, 3)

    for t, n in zip([camera_position, at, up], ["camera_position", "at", "up"]):
        if t.shape[-1] != 3:
            msg = "Expected arg %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))
    z_axis = F.normalize(camera_position - at, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(dim=1, keepdim=True)
    if is_close.any():
        # print(f'warning: up vector {up[0].detach()} is close to x_axis {z_axis[0].detach()}')
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)

