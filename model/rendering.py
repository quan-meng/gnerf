import torch.nn as nn
import torch


class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x, dim=-1):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, dim)


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf.detach(), u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def inference(model, xyz_, dir_, z_vals_, far,
              white_back, chunk, noise_std, weights_only=False):
    """
    Helper function that performs model inference.

    Inputs:
        model: NeRF model (coarse or fine)
        xyz_: (N_rays, N_samples_, 3) sampled positions
              N_samples_ is the number of sampled points in each ray;
                         = N_samples for coarse model
                         = N_samples+N_importance for fine model
        dir_: (N_rays, 3) ray directions
        rays_d: (N_rays, embed_dir_channels) embedded directions
        xyz_noise_: (N_rays, s_dim) shape code
        dir_noise_: (N_rays, a_dim) appearance code
        z_vals: (N_rays, N_samples_) depths of the sampled positions
        weights_only: do inference on sigma only or not

    Outputs:
        if weights_only:
            weights: (N_rays, N_samples_): weights of each sample
        else:
            rgb_final: (N_rays, 3) the final rgb image
            depth_final: (N_rays) depth map
            weights: (N_rays, N_samples_): weights of each sample
    """
    N_rays, N_samples = xyz_.shape[:2]
    rays_d_ = torch.repeat_interleave(dir_, repeats=N_samples, dim=0)  # [N_rays*N_samples, 3]

    # Convert these values using volume rendering (Section 4)
    xyz_ = xyz_.view(-1, 3)  # [N_rays*N_samples, 4]

    deltas = z_vals_[:, 1:] - z_vals_[:, :-1]  # (N_rays, N_samples_-1)
    deltas = torch.cat([deltas, far - z_vals_[:, -1:]], -1)  # (N_rays, N_samples_)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

    # Perform model inference to get rgb and raw sigma
    B = xyz_.shape[0]
    out_chunks = []
    for i in range(0, B, chunk):
        # Embed positions by chunk
        xyzdir = torch.cat([xyz_[i:i + chunk], rays_d_[i:i + chunk]], 1)
        rgb = model(xyzdir, sigma_only=False)
        out_chunks += [rgb]
    out_chunks = torch.cat(out_chunks, 0)

    if weights_only:
        sigmas = out_chunks.view(N_rays, N_samples)
    else:
        rgbsigma = out_chunks.view(N_rays, N_samples, 4)
        rgbs = rgbsigma[..., :3]  # (N_rays, N_samples_, 3)
        sigmas = rgbsigma[..., 3]  # (N_rays, N_samples_)

    noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std
    # compute alpha by the formula (3)
    alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples_)
    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, a1, a2, ...]

    T = torch.cumprod(alphas_shifted, -1)
    weights = alphas * T[:, :-1]  # (N_rays, N_samples_)
    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    if weights_only:
        return weights
    else:
        # compute final weighted outputs
        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # [N_rays, 3]
        depth_final = torch.sum(weights * z_vals_, -1)  # (N_rays)

        if white_back:
            weights_sum = weights.sum(1)  # (N_rays), the accumulated opacity along the rays
            rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights
