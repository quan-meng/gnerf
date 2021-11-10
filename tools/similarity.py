import torch
from kornia.losses import ssim as dssim
from lpips_pytorch import LPIPS

lpips_fn = LPIPS(net_type='alex', version='0.1')
lpips_fn.eval()


def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred - image_gt) ** 2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value


def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    image_pred = image_pred / 2 + 0.5
    image_gt = image_gt / 2 + 0.5
    return -10 * torch.log10(mse(image_pred, image_gt, valid_mask, reduction))


def ssim(image_pred, image_gt, reduction='mean'):
    image_pred = image_pred / 2 + 0.5
    image_gt = image_gt / 2 + 0.5
    dssim_ = dssim(image_pred, image_gt, 3, reduction)  # dissimilarity in [0, 1]
    return 1 - 2 * dssim_  # in [-1, 1]


def lpips(image_pred, image_gt, device='cpu'):
    lpips_fn.to(device)
    with torch.no_grad():
        lpips_ = lpips_fn(image_pred, image_gt)

    return lpips_.mean().item()
