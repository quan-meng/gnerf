import os
import time
import itertools
import torch
import numpy as np
import imageio
from tqdm import tqdm
from torch import autograd
from torchvision.transforms import Resize
import torch.nn.functional as F
from torchvision import utils

from tools.utils import plot_camera_scene
from tools.ray_utils import pose_to_d9
from tools.similarity import psnr, ssim, lpips


class Trainer(object):
    def __init__(self, cfg, generator, discriminator, inv_net, train_pose_params, val_pose_params,
                 optim_g, optim_d, optim_i, optim_t, optim_v,
                 scheduler_g, scheduler_d, scheduler_i, scheduler_t, scheduler_v,
                 train_loader, eval_loader, dynamic_patch_sampler, static_patch_sampler, full_img_sampler,
                 writer, device, it=-1, epoch=-1, psnr_best=-float('inf')):
        self.cfg = cfg
        self.generator = generator
        self.discriminator = discriminator
        self.inv_net = inv_net
        self.train_pose_params = train_pose_params
        self.val_pose_params = val_pose_params
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.optim_i = optim_i
        self.optim_t = optim_t
        self.optim_v = optim_v
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.scheduler_i = scheduler_i
        self.scheduler_t = scheduler_t
        self.scheduler_v = scheduler_v
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.dynamic_patch_sampler = dynamic_patch_sampler
        self.static_patch_sampler = static_patch_sampler
        self.full_img_sampler = full_img_sampler
        self.writer = writer
        self.device = device
        self.it = it
        self.epoch = epoch
        self.psnr_best = psnr_best

        self.out_dir = cfg.out_dir
        self.ckpt_dir = cfg.ckpt_dir
        self.log_dir = cfg.log_dir
        self.video_dir = cfg.video_dir

        self.batch_size = self.cfg.batch_size
        self.patch_size = self.cfg.patch_size
        self.start_img_wh = torch.tensor([self.patch_size, self.patch_size] if self.cfg.progressvie_training
                                         else self.cfg.img_wh, device=self.device)
        self.img_wh_curr = self.start_img_wh
        self.img_wh_end = torch.tensor(self.cfg.img_wh, device=device)

        self.phase = 'A'

    def train(self):
        val_imgs = []
        for rgb_i, _ in self.eval_loader:
            val_imgs.append(rgb_i)
        val_imgs_raw = torch.cat(val_imgs).to(self.device)  # [N, 3, H, W]

        for self.epoch in itertools.count(self.epoch + 1, 1):
            if self.epoch > self.cfg.num_epoch:
                break

            for img_real, pose_indices in self.train_loader:
                t_it = time.time()
                self.it += 1

                loss_dict, metrics, imgs = {}, {}, {}
                nbatch = img_real.shape[0]
                img_real = img_real.to(self.device)

                if self.it <= self.cfg.begin_refine:
                    self.phase = 'A'
                elif self.cfg.begin_refine < self.it <= self.cfg.progressive_end:
                    self.phase = 'ABAB'
                else:
                    self.phase = 'B'

                if self.cfg.progressvie_training:
                    img_real = self.progressvie_training(img_real)
                    val_imgs = self.progressvie_training(val_imgs_raw)

                    self.generator.ray_sampler.update_intrinsic(self.img_wh_curr / self.img_wh_end)

                if self.cfg.decrease_noise:
                    self.generator.decrease_noise(self.it)

                self.dynamic_patch_sampler.iterations = self.it

                self.generator.train()
                self.discriminator.train()
                self.inv_net.train()
                self.train_pose_params.train()
                self.val_pose_params.train()

                patch_real, coords_real, scales_real = self.dynamic_patch_sampler.image2patch(
                    img_real, self.cfg.patch_size, self.device)
                coords_fake, scales_fake = self.dynamic_patch_sampler(
                    nbatch, self.cfg.patch_size, self.device)
                imgs['Dynamic_patch/real'] = patch_real

                if self.phase == 'A' or self.phase == 'ABAB':
                    # Train the generator
                    patch_fake, poses = self.generator_trainstep(coords_fake, scales_fake, loss_dict)
                    imgs['Dynamic_patch_fake/coarse'] = patch_fake[:4]
                    imgs['Dynamic_patch_fake/fine'] = patch_fake[self.batch_size:self.batch_size + 4]

                    # Train the discriminator
                    self.discriminator_trainstep(patch_real, patch_fake, scales_real, scales_fake, loss_dict)

                    # Train the inversion network
                    coords_fake, _ = self.static_patch_sampler(nbatch, self.cfg.inv_size, self.device)
                    img_fake_fine = self.inversion_net_trainstep(coords_fake, loss_dict)
                    imgs['Static_patch_fake'] = img_fake_fine

                    # regularize poses of training images along with the generator
                    img_real, _, _ = self.static_patch_sampler.image2patch(img_real, self.cfg.inv_size, self.device)
                    self.training_pose_regularization(img_real, pose_indices, loss_dict)

                    # regularize poses of evaluation images
                    img_real, _, _ = self.static_patch_sampler.image2patch(val_imgs, self.cfg.inv_size, self.device)
                    self.val_pose_regularization(img_real, loss_dict)

                # Refine the camera poses and NeRF
                if self.phase == 'ABAB' or self.phase == 'B':
                    patch_fake_fine = self.training_refine_step(patch_real, coords_real, pose_indices, loss_dict)
                    imgs['Dynamic_patch/fake'] = patch_fake_fine

                    if self.it % 8 == 0:
                        img_real, coords_real, _ = self.dynamic_patch_sampler.image2patch(
                            val_imgs, self.cfg.patch_size, self.device)
                        self.val_refine_step(img_real, coords_real, loss_dict)

                fps = nbatch / (time.time() - t_it)

                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'Training/{k}', v, self.it)

                # Update learning rate
                self.scheduler_g.step()
                self.scheduler_d.step()
                self.scheduler_i.step()
                self.scheduler_t.step()
                self.scheduler_v.step()

                # Evaluation
                with torch.no_grad():
                    self.generator.eval()

                    if (self.it % self.cfg.print_every) == 0:
                        print(f"{self.out_dir.split('/')[-1]}: {self.epoch:05d}/{self.it:05d}, phase={self.phase}, " +
                              f", ".join("{}={:.4f}".format(k, v) for k, v in loss_dict.items()) +
                              f", fps: {fps:.2f}")

                    if ((self.it % self.cfg.sample_every) == 0) or ((self.it <= 500) and (self.it % 100 == 0)):
                        poses_val = self.val_pose_params()
                        coords_val, _ = self.full_img_sampler(val_imgs.shape[0], self.img_wh_curr, self.device)
                        results = self.generator(coords_val, self.img_wh_curr, poses_val)

                        if self.img_wh_end[0] > self.img_wh_curr[0] or self.img_wh_end[1] > self.img_wh_curr[1]:
                            rescale_func = Resize((self.img_wh_end[1], self.img_wh_end[0]))
                            results = {k: rescale_func(v) for k, v in results.items()}

                        for k, v in results.items():
                            if k == 'depth':
                                v = (v + 1.0) / 2.0
                                v = (v - v.min()) / (v.max() - v.min() + 1e-8)
                                v = v * 2.0 - 1.0
                            v = utils.make_grid(v[:4], nrow=4,
                                                normalize=True,
                                                value_range=(-1, 1))
                            self.writer.add_image(k, v, self.it)

                        real_image = utils.make_grid(val_imgs_raw[:4],
                                                     normalize=True,
                                                     value_range=(-1, 1))
                        self.writer.add_image('real_image', real_image, self.it)

                        metrics['psnr'] = psnr(results['rgb'], val_imgs_raw)
                        metrics['ssim'] = ssim(results['rgb'], val_imgs_raw)
                        lpips_ = []
                        for rgb_pred, rgb_gt in zip(results['rgb'], val_imgs_raw):
                            lpips_.append(lpips(rgb_pred, rgb_gt, device=self.device))
                        metrics['lpips'] = sum(lpips_) / len(lpips_)

                        for k, v in metrics.items():
                            self.writer.add_scalar(f'Val/{k}', v, self.it)

                        self.writer.add_scalar('lr/discriminator', self.optim_d.param_groups[0]['lr'], self.it)
                        self.writer.add_scalar('lr/generator', self.optim_g.param_groups[0]['lr'], self.it)
                        self.writer.add_scalar('lr/inversion_net', self.optim_i.param_groups[0]['lr'], self.it)
                        self.writer.add_scalar('lr/poses_training', self.optim_t.param_groups[0]['lr'], self.it)
                        self.writer.add_scalar('lr/poses_val', self.optim_v.param_groups[0]['lr'], self.it)
                        self.writer.add_scalar('img_wh_curr/w', self.img_wh_curr[0], self.it)
                        self.writer.add_scalar('img_wh_curr/h', self.img_wh_curr[1], self.it)
                        self.writer.add_scalar(
                            'scales_curr/min_scale', self.dynamic_patch_sampler.scales_curr[0], self.it)
                        self.writer.add_scalar(
                            'scales_curr/max_scale', self.dynamic_patch_sampler.scales_curr[1], self.it)
                        self.writer.add_scalar('noise_std', self.generator.noise_std, self.it)
                        self.writer.add_scalar('Val/psnr_best', self.psnr_best, self.it)

                        for k, v in imgs.items():
                            v = utils.make_grid(v, nrow=v.shape[0] // int(v.size(0) ** 0.5),
                                                normalize=True,
                                                value_range=(-1, 1))
                            self.writer.add_image(k, v, self.it)

                        cams_img = plot_camera_scene(self.train_pose_params.poses, self.val_pose_params.poses,
                                                     max(self.cfg.radius), f'Iteration_{self.it}')

                        self.writer.add_image('poses', cams_img, self.it)

                    if (self.it % self.cfg.save_every) == 0:
                        state_dict = {
                            'g': self.generator.state_dict(), 'optim_g': self.optim_g.state_dict(),
                            'd': self.discriminator.state_dict(), 'optim_d': self.optim_d.state_dict(),
                            'i': self.inv_net.state_dict(), 'optim_i': self.optim_i.state_dict(),
                            't': self.train_pose_params.state_dict(), 'optim_t': self.optim_t.state_dict(),
                            'v': self.val_pose_params.state_dict(), 'optim_v': self.optim_v.state_dict(),
                            'args': vars(self.cfg), 'it': self.it, 'epoch': self.epoch, 'psnr_best': self.psnr_best}

                        torch.save(state_dict, os.path.join(self.ckpt_dir, f'{str(self.it).zfill(6)}.pt'))

                        if metrics['psnr'] >= self.psnr_best:
                            self.psnr_best = metrics['psnr']
                            torch.save(state_dict, os.path.join(self.ckpt_dir, 'model_best.pt'))

                    if ((self.it + 1) % self.cfg.video_every) == 0:
                        self.make_video(120)

                    if ((self.it + 1) % self.cfg.empty_cache_every) == 0:
                        torch.cuda.empty_cache()

        print("End of Training")

    def progressvie_training(self, img):
        if self.phase == 'A':
            scale = 1.0 / self.cfg.begin_refine * self.it

            self.img_wh_curr = self.start_img_wh + ((128.0 - self.start_img_wh) * scale).int()
        elif self.phase == 'ABAB':
            img_scale_base = self.cfg.begin_refine / self.cfg.progressive_end
            scale = img_scale_base + (1.0 - img_scale_base) / (self.cfg.progressive_end - self.cfg.begin_refine) * (
                    self.it - self.cfg.begin_refine)

            self.img_wh_curr = 128 + ((self.img_wh_end - 128.0) / (1.0 - img_scale_base) * (
                    scale - img_scale_base)).int()
        else:
            return img

        downsample_func = Resize((self.img_wh_curr[1], self.img_wh_curr[0]))
        img = downsample_func(img)

        return img

    def generator_trainstep(self, coords, scales, loss_dict):
        self.toggle_grad(self.generator, True)
        self.toggle_grad(self.discriminator, False)
        self.toggle_grad(self.train_pose_params, False)
        self.toggle_grad(self.val_pose_params, False)
        self.toggle_grad(self.inv_net, False)
        self.generator.zero_grad()

        fake_patch_coarse, fake_patch_fine, poses = self.generator(coords, self.img_wh_curr)
        patch_fake = torch.cat((fake_patch_coarse, fake_patch_fine), dim=0)
        d_fake = self.discriminator(patch_fake, scales.repeat(2, 1, 1, 1))
        gloss = self.compute_loss(d_fake, 1)

        gloss.backward()

        self.optim_g.step()

        loss_dict['generator'] = gloss.detach()

        return patch_fake.detach(), poses.detach()

    def discriminator_trainstep(self, patch_real, patch_fake, scales_real, scales_fake, loss_dict):
        self.toggle_grad(self.generator, False)
        self.toggle_grad(self.discriminator, True)
        self.toggle_grad(self.train_pose_params, False)
        self.toggle_grad(self.val_pose_params, False)
        self.toggle_grad(self.inv_net, False)
        self.optim_d.zero_grad()

        # On real data
        patch_real.requires_grad_()

        d_real = self.discriminator(patch_real, scales_real)
        dloss_real = self.compute_loss(d_real, 1)

        if self.cfg.reg_type == 'real' or self.cfg.reg_type == 'real_fake':
            dloss_real.backward(retain_graph=True)
            reg = self.cfg.reg_param * self.compute_grad2(d_real, patch_real).mean()
            reg.backward()
        else:
            dloss_real.backward()

        # On fake data
        patch_fake.requires_grad_()
        d_fake = self.discriminator(patch_fake.contiguous(), scales_fake.repeat(2, 1, 1, 1).contiguous())
        dloss_fake = self.compute_loss(d_fake, 0)

        if self.cfg.reg_type == 'fake' or self.cfg.reg_type == 'real_fake':
            dloss_fake.backward(retain_graph=True)
            reg = self.cfg.reg_param * self.compute_grad2(d_fake, patch_fake).mean()
            reg.backward()
        else:
            dloss_fake.backward()

        if self.cfg.reg_type == 'wgangp':
            reg = self.cfg.reg_param * self.wgan_gp_reg(patch_real.repeat(2, 1, 1, 1), patch_fake,
                                                        scales_fake.repeat(2, 1, 1, 1))
            reg.backward()
        elif self.cfg.reg_type == 'wgangp0':
            reg = self.cfg.reg_param * self.wgan_gp_reg(patch_real.repeat(2, 1, 1, 1), patch_fake,
                                                        scales_fake.repeat(2, 1, 1, 1), center=0.)
            reg.backward()

        self.optim_d.step()

        # Output
        dloss = (dloss_real + dloss_fake)

        if self.cfg.reg_type == 'none':
            reg = torch.tensor(0.)

        loss_dict['discriminator'] = dloss.detach()
        loss_dict['regularizer'] = reg.detach()

    def inversion_net_trainstep(self, coords, loss_dict):
        self.toggle_grad(self.generator, False)
        self.toggle_grad(self.discriminator, False)
        self.toggle_grad(self.train_pose_params, False)
        self.toggle_grad(self.val_pose_params, False)
        self.toggle_grad(self.inv_net, True)
        self.optim_i.zero_grad()

        img_fake_coarse, img_fake_fine, poses = self.generator(coords, self.img_wh_curr)

        if self.cfg.pose_mode == '3d':
            d_fake = poses[:, :, -1]  # [N, 3]
        elif self.cfg.pose_mode == '6d':
            d_fake = pose_to_d9(poses)  # [N, 9]
        else:
            raise NotImplementedError

        x_fake_full = torch.cat((img_fake_coarse, img_fake_fine), dim=0)
        x_trans_em = self.inv_net(x_fake_full)  # [N, D]
        eloss = torch.mean((x_trans_em - d_fake.repeat(2, 1)) ** 2)

        eloss.backward()

        self.optim_i.step()

        loss_dict['inversion'] = eloss.detach()

        return img_fake_fine.detach()

    def training_pose_regularization(self, img_real, pose_indices, loss_dict):
        self.toggle_grad(self.train_pose_params, True)
        self.toggle_grad(self.val_pose_params, False)
        self.toggle_grad(self.generator, False)
        self.toggle_grad(self.discriminator, False)
        self.toggle_grad(self.inv_net, False)

        self.optim_t.zero_grad()
        # On real data
        train_poses_real = self.train_pose_params(pose_indices)
        if self.cfg.pose_mode == '3d':
            train_d_real = train_poses_real[:, :, -1]  # [N, 3]
        elif self.cfg.pose_mode == '6d':
            train_d_real = pose_to_d9(train_poses_real)  # [N, 9]
        else:
            raise NotImplementedError

        train_x_trans_em = self.inv_net(img_real)
        ploss = torch.mean((train_x_trans_em - train_d_real) ** 2) * 0.1

        ploss.backward()
        self.optim_t.step()

        loss_dict['pose_training'] = ploss.detach()

    def val_pose_regularization(self, patch_real, loss_dict):
        self.toggle_grad(self.train_pose_params, False)
        self.toggle_grad(self.val_pose_params, True)
        self.toggle_grad(self.generator, False)
        self.toggle_grad(self.discriminator, False)
        self.toggle_grad(self.inv_net, False)
        self.optim_v.zero_grad()

        val_poses_real = self.val_pose_params()
        if self.cfg.pose_mode == '3d':
            val_d_real = val_poses_real[:, :, -1]
        elif self.cfg.pose_mode == '6d':
            val_d_real = pose_to_d9(val_poses_real)  # [N, 9]
        else:
            raise NotImplementedError

        val_x_trans_em = self.inv_net(patch_real)
        val_ploss = torch.mean((val_x_trans_em - val_d_real) ** 2) * 0.1

        val_ploss.backward()
        self.optim_v.step()

        loss_dict['pose_val'] = val_ploss.detach()

    def training_refine_step(self, patch_real, coords_real, pose_indices, loss_dict):
        self.toggle_grad(self.generator, True)
        self.toggle_grad(self.train_pose_params, True)
        self.toggle_grad(self.val_pose_params, False)
        self.toggle_grad(self.discriminator, False)
        self.toggle_grad(self.inv_net, False)
        self.optim_g.zero_grad()
        self.optim_t.zero_grad()

        # On real data
        poses = self.train_pose_params(pose_indices)
        patch_fake_coarse, patch_fake_fine, _ = self.generator(coords_real, self.img_wh_curr, poses)
        patch_fake = torch.cat((patch_fake_coarse, patch_fake_fine), dim=0)
        rloss = F.mse_loss(patch_fake, patch_real.repeat(2, 1, 1, 1), reduction='mean') * 50.0

        rloss.backward()

        self.optim_t.step()
        self.optim_g.step()

        loss_dict['refine_training'] = rloss.detach()

        return patch_fake_fine.detach()

    def val_refine_step(self, img_real, coords_real, loss_dict):
        self.toggle_grad(self.generator, False)
        self.toggle_grad(self.train_pose_params, False)
        self.toggle_grad(self.val_pose_params, True)
        self.toggle_grad(self.discriminator, False)
        self.toggle_grad(self.inv_net, False)
        self.optim_v.zero_grad()

        # On real data
        poses = self.val_pose_params()
        img_fake_coarse, img_fake_fine, _ = self.generator(coords_real, self.img_wh_curr, poses)
        img_fake = torch.cat((img_fake_coarse, img_fake_fine), dim=0)
        rloss = F.mse_loss(img_fake, img_real.repeat(2, 1, 1, 1), reduction='mean') * 50.0

        rloss.backward()

        self.optim_v.step()

        loss_dict['refine_val'] = rloss.detach()

    def toggle_grad(self, model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    def compute_grad2(self, d_outs, x_in):
        d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
        reg = 0
        for d_out in d_outs:
            batch_size = x_in.size(0)
            grad_dout = autograd.grad(
                outputs=d_out.sum(), inputs=x_in,
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad_dout2 = grad_dout.pow(2)
            assert (grad_dout2.size() == x_in.size())
            reg += grad_dout2.view(batch_size, -1).sum(1)
        return reg / len(d_outs)

    def compute_loss(self, d_outs, target):

        d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
        loss = torch.tensor(0.0, device=d_outs[0].device)

        for d_out in d_outs:

            targets = d_out.new_full(size=d_out.size(), fill_value=target)

            if self.cfg.gan_type == 'standard':
                loss += F.binary_cross_entropy_with_logits(d_out, targets)
            elif self.cfg.gan_type == 'wgan':
                loss += (2 * target - 1) * d_out.mean()
            else:
                raise NotImplementedError

        return loss / len(d_outs)

    def wgan_gp_reg(self, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(x_interp, y)

        reg = (self.compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg

    def make_video(self, nframes=40):
        chunk = self.batch_size
        coords, _ = self.full_img_sampler(chunk, self.img_wh_curr, self.device)
        poses = self.generator.ray_sampler.spheric_poses(nframes).to(self.device)
        rescale_func = Resize((self.img_wh_end[1], self.img_wh_end[0]))

        rgbs, depths = [], []
        for i in tqdm(range(0, nframes, chunk)):
            results = self.generator(coords, self.img_wh_curr, poses[i:i + chunk])

            if self.img_wh_end[0] > self.img_wh_curr[0] or self.img_wh_end[1] > self.img_wh_curr[1]:
                results = {k: rescale_func(v) for k, v in results.items()}

            rgbs.append(results['rgb'])
            depths.append(results['depth'])
        rgbs = torch.cat(rgbs)  # [N, 3, h, w]
        depths = torch.cat(depths)

        rgbs = ((rgbs.cpu().permute(0, 2, 3, 1) / 2 + 0.5).numpy().clip(0, 1) * 255).astype(np.uint8)
        imageio.mimwrite(os.path.join(self.video_dir, f'rgb_{self.it:04}.gif'), rgbs, fps=24)

        depths = (depths.cpu().permute(0, 2, 3, 1).numpy().clip(0, 1) * 255).astype(np.uint8)
        imageio.mimwrite(os.path.join(self.video_dir, f'depth_{self.it:04}.gif'), depths, fps=24)
