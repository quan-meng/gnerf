import torch
import torch.nn as nn
from vit_pytorch import ViT

from tools.ray_utils import look_at_rotation, r6d2mat, pose_to_d9


class PoseParameters(nn.Module):
    def __init__(self, length, pose_mode, data):
        super(PoseParameters, self).__init__()
        self.length = length
        self.pose_mode = pose_mode
        self.data = data
        self.up = (0., 0, 1)
        # [N, 9]: (x, y, z, r1, r2) or [N, 3]: (x, y, z)
        self.poses_embed = nn.Parameter(self.init_poses_embed())

    def init_poses_embed(self):
        if self.pose_mode == '3d':
            poses_embed = torch.tensor([[0., 0, 1]]).repeat(self.length, 1)  # [N, 3]
        elif self.pose_mode == '6d':
            t = torch.tensor([[0., 0, 1]]).repeat(self.length, 1)  # [N, 3]
            R = look_at_rotation(t, up=self.up)  # [N, 3, 3]
            poses = torch.cat((R, t[..., None]), -1)
            poses_embed = pose_to_d9(poses)
        else:
            raise NotImplementedError

        return poses_embed

    @property
    def poses(self):
        if self.pose_mode == '3d':
            t = self.poses_embed[:, :3]  # [N, 3]
            R = look_at_rotation(t, device=t.device)  # [N, 3, 3]
        elif self.pose_mode == '6d':
            t = self.poses_embed[:, :3]  # [N, 3]
            r = self.poses_embed[:, 3:]
            R = r6d2mat(r)[:, :3, :3]  # [N, 3, 3]
        else:
            raise NotImplementedError

        poses = torch.cat((R, t[..., None]), -1)  # [N, 3, 4]

        return poses

    def forward(self, pose_indices=None):
        if pose_indices is None:
            return self.poses
        return self.poses[pose_indices]


class InversionNet(nn.Module):
    def __init__(self, imsize, pose_mode):
        super(InversionNet, self).__init__()
        self.imsize = imsize
        self.pose_mode = pose_mode

        if pose_mode == '3d':
            final_dims = 3  # [N, 3]
        elif pose_mode == '6d':
            final_dims = 9  # [N, 9]
        else:
            raise NotImplementedError

        self.main = ViT(
            image_size=self.imsize,
            patch_size=self.imsize // 16,
            num_classes=final_dims,
            dim=256,
            depth=6,
            heads=16,
            mlp_dim=256
        )

    def forward(self, img):
        em = self.main(img)

        return em
