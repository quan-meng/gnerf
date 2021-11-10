import torch
import math
import torch.nn.functional as F

from tools.ray_utils import look_at_rotation


class RaySampler(object):
    def __init__(self, near, far, azim_range, elev_range, radius, look_at_origin, ndc, intrinsics):
        self.near = near
        self.far = far
        self.azim_range = azim_range
        self.elev_range = elev_range
        self.radius = radius
        self.look_at_origin = look_at_origin
        self.up = (0., 0, 1)
        self.ndc = ndc
        self.scale = 1.0
        self.start_intrinsics = intrinsics

    def update_intrinsic(self, scale):
        self.intrinsics = self.start_intrinsics.clone().detach()
        self.intrinsics[:2] = self.intrinsics[:2] * scale[:, None]

        return self.intrinsics

    def random_poses(self, nbatch, device='cpu'):
        raes = torch.rand(nbatch, 3, device=device)

        azims = raes[:, 0:1] * (self.azim_range[1] - self.azim_range[0]) + self.azim_range[0]
        elevs = raes[:, 1:2] * (self.elev_range[1] - self.elev_range[0]) + self.elev_range[0]

        azims = math.pi / 180.0 * azims
        elevs = math.pi / 180.0 * elevs

        cx = torch.cos(elevs) * torch.cos(azims)
        cy = torch.cos(elevs) * torch.sin(azims)
        cz = torch.sin(elevs)
        T = torch.cat([cx, cy, cz], -1)  # [N, 3]

        radius = raes[:, 2:] * (self.radius[1] - self.radius[0]) + self.radius[0]

        T = T * radius

        if self.look_at_origin:
            lookat = (0, 0, 0)
        else:
            xy = torch.randn((nbatch, 2), device=device) * self.radius[0] * 0.01
            z = torch.zeros((nbatch, 1), device=device)

            lookat = torch.cat((xy, z), dim=-1)

        R = look_at_rotation(T, at=lookat, up=self.up, device=device)  # [N, 3, 3]
        RT = torch.cat((R, T[..., None]), -1)  # [N, 3, 4]

        return RT

    def spheric_poses(self, N=120, device='cpu'):
        elevs = torch.ones([N, 1], device=device) * sum(self.elev_range) * 0.5 * math.pi / 180.0
        azims = torch.linspace(self.azim_range[0], self.azim_range[1], N, device=device)[:, None] * math.pi / 180.0
        radius = torch.mean(torch.tensor(self.radius))

        cx = torch.cos(elevs) * torch.cos(azims)
        cy = torch.cos(elevs) * torch.sin(azims)
        cz = torch.sin(elevs)
        t = torch.cat([cx, cy, cz], -1) * radius

        R = look_at_rotation(t, at=(0, 0, 0), device=device)  # [N, 3, 3]
        c2w = torch.cat((R, t[..., None]), -1)  # [N, 3, 4]

        return c2w

    def get_rays(self, coords, c2ws, img_wh, device):
        n, h, w, _ = coords.shape
        i, j = torch.meshgrid(torch.linspace(0, img_wh[0] - 1, img_wh[0], device=device),
                              torch.linspace(0, img_wh[1] - 1, img_wh[1], device=device))
        i = i.t()[None, None].repeat(n, 1, 1, 1)  # [N, 1, H, W]
        j = j.t()[None, None].repeat(n, 1, 1, 1)  # [N, 1, H, W]

        u = F.grid_sample(i, coords, mode='bilinear', align_corners=True)[:, 0]  # [N, h, w]
        v = F.grid_sample(j, coords, mode='bilinear', align_corners=True)[:, 0]  # [N, h, w]

        dirs = torch.stack(
            [(u - self.intrinsics[0, 2]) / self.intrinsics[0, 0],
             -(v - self.intrinsics[1, 2]) / self.intrinsics[1, 1],
             -torch.ones_like(u)], -1)  # [N, H, W, 3]

        rays_d = torch.einsum('abcd, ade -> abce', dirs, c2ws[:, :3, :3].permute(0, 2, 1))  # [N, H, W, 3]
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        rays_o = c2ws[:, None, None, :3, -1].repeat(1, h, w, 1)  # [N, H, W, 3]

        rays = torch.cat([rays_o, rays_d,
                          self.near * torch.ones_like(rays_o[..., :1]),
                          self.far * torch.ones_like(rays_o[..., :1])], -1)  # [N, H, W, 8]

        return rays
