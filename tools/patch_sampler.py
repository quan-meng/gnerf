import torch
import torch.nn.functional as F
from math import exp


class PatchSampler(object):
    def __init__(self):
        self.full_indices = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def image2patch(self, imgs, wh, device):
        nbatch = imgs.shape[0]
        patch_coord, scale = self(nbatch, wh, device)

        if not self.full_indices:
            imgs = F.grid_sample(imgs, patch_coord, mode='bilinear', align_corners=True)

        return imgs, patch_coord, scale


class FullImageSampler(PatchSampler):
    def __init__(self):
        super(FullImageSampler, self).__init__()

        self.full_indices = True

    def __call__(self, nbatch, wh, device):
        w, h = torch.meshgrid([torch.linspace(-1, 1, wh[1]), torch.linspace(-1, 1, wh[0])])
        h = h[None, ..., None]
        w = w[None, ..., None]

        coords = torch.cat([h, w], dim=-1)  # [1, H, W, 2]

        coords = coords.repeat(nbatch, 1, 1, 1).to(device)
        scales = torch.ones((nbatch, 1, 1, 1), device=device)
        return coords.contiguous(), scales.contiguous()


class RescalePatchSampler(PatchSampler):
    def __init__(self, scale=1.0):
        super(RescalePatchSampler, self).__init__()
        self.scale = scale

        self.full_indices = False

    def __call__(self, nbatch, patch_size, device):
        w, h = torch.meshgrid([torch.linspace(-1, 1, patch_size), torch.linspace(-1, 1, patch_size)])
        h = h[None, ..., None]
        w = w[None, ..., None]

        h = h * self.scale
        w = w * self.scale

        coords = torch.cat([h, w], dim=-1)  # [1, H, W, 2]

        coords = coords.repeat(nbatch, 1, 1, 1).to(device)
        scales = torch.ones((nbatch, 1, 1, 1), device=device)
        return coords.contiguous(), scales.contiguous()


class FlexPatchSampler(PatchSampler):
    def __init__(self, random_shift=True, random_scale=True, min_scale=0.25, max_scale=1., scale_anneal=-1):
        super(FlexPatchSampler, self).__init__()

        self.random_shift = random_shift
        self.random_scale = random_scale

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scales_curr = (min_scale, max_scale)

        self.iterations = 0
        self.scale_anneal = scale_anneal

        self.full_indices = False

    def __call__(self, nbatch, patch_size, device):
        w, h = torch.meshgrid([torch.linspace(-1, 1, patch_size, device=device),
                               torch.linspace(-1, 1, patch_size, device=device)])
        h = h[None, ..., None]
        w = w[None, ..., None]

        if self.scale_anneal > 0:
            min_scale = max(self.min_scale, self.max_scale * exp(-self.iterations * self.scale_anneal))
            min_scale = min(0.8, min_scale)
        else:
            min_scale = self.min_scale

        max_scale = self.max_scale

        self.scales_curr = (min_scale, max_scale)

        if self.random_scale:
            scales = torch.rand((nbatch, 1, 1, 1), device=device) * (max_scale - min_scale) + min_scale
        else:
            scales = torch.ones((nbatch, 1, 1, 1), device=device) * min_scale

        h = h * scales
        w = w * scales

        if self.random_shift:
            max_offset = 1 - scales
            h_offset = (torch.rand((nbatch, 1, 1, 1), device=device) * 2.0 - 1.0) * max_offset  # [nbatch, 1, 1, 1]
            w_offset = (torch.rand((nbatch, 1, 1, 1), device=device) * 2.0 - 1.0) * max_offset  # [nbatch, 1, 1, 1]

            h += h_offset
            w += w_offset

        coords = torch.cat([h, w], dim=-1)

        return coords.contiguous(), scales.contiguous()
