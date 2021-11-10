import torch
import torch.nn as nn
from collections import defaultdict
from einops import rearrange

from model.rendering import sample_pdf, Embedding, inference


class GNeRF(nn.Module):
    def __init__(self, ray_sampler, xyz_freq=10, dir_freq=4, fc_depth=8, fc_dim=256, skips=(4,),
                 N_samples=64, N_importance=64, chunk=1024 * 32, white_back=False):
        super(GNeRF, self).__init__()
        self.ray_sampler = ray_sampler
        self.chunk = chunk
        self.N_samples = N_samples
        self.N_importance = N_importance
        self.white_back = white_back
        self.noise_std = 1.0

        self.nerf = NeRF(xyz_freq=xyz_freq, dir_freq=dir_freq, fc_depth=fc_depth, fc_dim=fc_dim, skips=skips)

    def forward(self, coords, img_wh, poses=None):
        nbatch, h, w, _ = coords.shape
        device = coords.device

        noise_std = self.noise_std if self.training else 0.0
        perturb = 1.0 if self.training else 0.0

        poses = self.ray_sampler.random_poses(nbatch, device) if poses is None else poses

        rays = self.ray_sampler.get_rays(coords, poses, img_wh, device)
        rays = rearrange(rays, 'n h w c -> (n h w) c')

        results = {'coarse': defaultdict(list), 'fine': defaultdict(list)}
        for i in range(0, rays.shape[0], self.chunk):
            rendered_ray_chunks = self.render_rays(rays=rays[i:i + self.chunk], perturb=perturb, noise_std=noise_std)

            for k_1, v_1 in rendered_ray_chunks.items():
                for k_2, v_2 in v_1.items():
                    results[k_1][k_2] += [v_2]

        for k_1, v_1 in results.items():
            for k_2, v_2 in v_1.items():
                v_2 = torch.cat(v_2, 0)
                v_2 = rearrange(v_2, '(n h w) c -> n c h w', n=nbatch, h=h, w=w)
                results[k_1][k_2] = v_2 * 2.0 - 1.0

        if self.training:
            return results['coarse']['rgb'], results['fine']['rgb'], poses
        else:
            return results['fine']

    def render_rays(self, rays, use_disp=False, perturb=0.0, noise_std=1.0):
        N_rays = rays.shape[0]
        device = rays.device
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both [N_rays, 3]
        near, far = rays[:, 6:7], rays[:, 7:8]  # both [N_rays, 1]

        rets = {'coarse': {}, 'fine': {}}
        for i, type in enumerate(rets.keys()):
            if type == 'coarse':
                z_steps = torch.linspace(0, 1, self.N_samples, device=device)  # [N_samples]

                if not use_disp:  # use linear sampling in depth space
                    z_vals = near * (1 - z_steps) + far * z_steps
                else:  # use linear sampling in disparity space
                    z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

                z_vals = z_vals.expand(N_rays, self.N_samples)  # [N_rays, N_samples]
            else:
                z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], self.N_importance, det=(perturb == 0)).detach()
                # detach so that grad doesn't propogate to weights_coarse from here
                z_vals, _ = torch.sort(torch.cat([z_vals, new_z_vals], -1), -1)  # [N_rays, N_samples + N_importance]

            xyz_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # [(N_rays, N_samples, 3]

            rgb, depth, weights = inference(self.nerf, xyz_sampled, rays_d, z_vals,
                                            far, self.white_back, self.chunk, noise_std,
                                            weights_only=False)

            rets[type].update({
                'rgb': rgb,
                'depth': depth.detach()[:, None],
                'opacity': weights.sum(1).detach()[:, None]
            })

        return rets

    def decrease_noise(self, it):
        end_it = 5000
        if it < end_it:
            self.noise_std = 1.0 - float(it) / end_it


class NeRF(nn.Module):
    def __init__(self, xyz_freq=10, dir_freq=4, fc_depth=8, fc_dim=256, skips=(4,)):
        super(NeRF, self).__init__()
        self.fc_depth = fc_depth
        self.fc_dim = fc_dim
        self.skips = skips

        self.embedding_xyz = Embedding(3, xyz_freq)
        self.embedding_dir = Embedding(3, dir_freq)
        self.in_channels_xyz = self.embedding_xyz.out_channels
        self.in_channels_dir = self.embedding_dir.out_channels

        # xyz encoding layers
        for i in range(fc_depth):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, fc_dim)
            elif i in skips:
                layer = nn.Linear(fc_dim + self.in_channels_xyz, fc_dim)
            else:
                layer = nn.Linear(fc_dim, fc_dim)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i + 1}", layer)
        self.xyz_encoding_final = nn.Linear(fc_dim, fc_dim)

        # output layers
        self.sigma = nn.Linear(fc_dim, 1)

        # direction encoding layers
        self.rgb = nn.Sequential(
            nn.Linear(fc_dim + self.in_channels_dir, fc_dim // 2),
            nn.ReLU(True),
            nn.Linear(fc_dim // 2, 3),
            nn.Sigmoid()
        )

    def forward(self, x, sigma_only=False):
        if not sigma_only:
            input_xyz, input_dir = torch.split(x, [3, 3], dim=-1)

            input_xyz = self.embedding_xyz(input_xyz)
            input_dir = self.embedding_dir(input_dir)
        else:
            input_xyz = x

            input_xyz = self.embedding_xyz(input_xyz)

        xyz_ = input_xyz
        for i in range(self.fc_depth):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i + 1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding = torch.cat([xyz_encoding_final, input_dir], -1)

        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out
