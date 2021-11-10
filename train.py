import os
import torch
import subprocess
import atexit
import signal
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from config import config
from model import *
from trains.trainer import Trainer
from dataset.dataloader import create_dataloader
from tools.patch_sampler import RescalePatchSampler, FlexPatchSampler, FullImageSampler
from tools.ray_sampler import RaySampler
from tools.utils import count_trainable_parameters


def open_tensorboard(log_dir):
    p = subprocess.Popen(
        ["tensorboard", "--logdir", log_dir, '--bind_all', '--reload_multifile', 'True', '--load_fast', 'false']
    )

    def killme():
        os.kill(p.pid, signal.SIGTERM)

    atexit.register(killme)


def build_scheduler(optimizer, lr_anneal_every, lr_anneal, last_epoch=-1):
    if isinstance(lr_anneal_every, str):
        milestones = [int(m) for m in lr_anneal_every.split(',')]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=lr_anneal,
            last_epoch=last_epoch)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_anneal_every,
            gamma=lr_anneal,
            last_epoch=last_epoch
        )
    return lr_scheduler


def build_optimizers(model, optim_cfg, it):
    params = model.parameters()

    # Optimizers
    if optim_cfg['type'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=optim_cfg['lr'], alpha=0.99, eps=1e-8)
    elif optim_cfg['type'] == 'adam':
        optimizer = torch.optim.Adam(params, lr=optim_cfg['lr'], betas=(0.9, 0.99), eps=1e-8)
    elif optim_cfg['type'] == 'sgd':
        optimizer = torch.optim.SGD(params, lr=optim_cfg['lr'], momentum=0.)
    else:
        raise NotImplementedError

    # Learning rate anneling
    lr = optimizer.param_groups[0]['lr']
    # create learning reate scheduler
    scheduler = build_scheduler(optimizer, optim_cfg['lr_anneal_every'], optim_cfg['lr_anneal'], it)
    # ensure lr is not decreased again
    optimizer.param_groups[0]['lr'] = lr

    return optimizer, scheduler


if __name__ == '__main__':
    args = config.load_config()
    device = 'cuda'

    current_time = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    args.out_dir = os.path.join(args.out_dir, "training-runs", f"{args.name}-{current_time}")

    # sampler
    args.min_scale = args.patch_size / max(args.img_wh[0], args.img_wh[1])

    # Create dataloaders
    print("Making dataloader...")
    train_loader = create_dataloader(args.data, 'train', args.data_dir, args.img_wh, args.batch_size, args.num_workers)
    eval_loader = create_dataloader(args.data, 'val', args.data_dir, args.img_wh, args.batch_size, args.num_workers)
    print(f'Data: {args.data}, train: {len(train_loader.dataset)} val: {len(eval_loader.dataset)}')

    # ray sampler
    dynamic_patch_sampler = FlexPatchSampler(
        random_scale=args.random_scale,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        scale_anneal=args.scale_anneal,
    )

    static_patch_sampler = RescalePatchSampler()

    full_img_sampler = FullImageSampler()

    ray_sampler = RaySampler(near=args.near, far=args.far, azim_range=args.azim_range, elev_range=args.elev_range,
                             radius=args.radius, look_at_origin=args.look_at_origin, ndc=args.ndc,
                             intrinsics=train_loader.dataset.intrinsics.clone().detach().to(device))

    # Create models
    generator = GNeRF(
        ray_sampler=ray_sampler, xyz_freq=args.xyz_freq, dir_freq=args.xyz_freq, fc_depth=args.fc_depth,
        fc_dim=args.fc_dim, chunk=args.chunk, white_back=args.white_back).to(device)

    discriminator = Discriminator(
        conditional=args.conditional, policy=args.policy, ndf=args.ndf, imsize=args.patch_size).to(device)

    inv_net = InversionNet(imsize=args.inv_size, pose_mode=args.pose_mode).to(device)

    train_pose_params = PoseParameters(
        length=len(train_loader.dataset), pose_mode=args.pose_mode, data=args.data).to(device)
    val_pose_params = PoseParameters(
        length=len(eval_loader.dataset), pose_mode=args.pose_mode, data=args.data).to(device)

    print(f'Generator params: {count_trainable_parameters(generator)}, '
          f'Discriminator params: {count_trainable_parameters(discriminator)}'
          f'InversionNet params: {count_trainable_parameters(inv_net)}'
          f'Optimizable poses: {len(train_pose_params.poses_embed)}')

    optim_g, scheduler_g = build_optimizers(generator, args.generator, -1)
    optim_d, scheduler_d = build_optimizers(discriminator, args.discriminator, -1)
    optim_i, scheduler_i = build_optimizers(inv_net, args.inv_net, -1)
    optim_t, scheduler_t = build_optimizers(train_pose_params, args.train_pose_params, -1)
    optim_v, scheduler_v = build_optimizers(val_pose_params, args.val_pose_params, -1)

    if args.ckpt is not None:
        print('load model:', args.ckpt)
        ckpt = torch.load(args.ckpt)

        generator.load_state_dict(ckpt['g'])
        discriminator.load_state_dict(ckpt['d'])
        inv_net.load_state_dict(ckpt['i'])
        train_pose_params.load_state_dict(ckpt['t'])
        val_pose_params.load_state_dict(ckpt['v'])

        optim_g.load_state_dict(ckpt['optim_g'])
        optim_d.load_state_dict(ckpt['optim_d'])
        optim_i.load_state_dict(ckpt['optim_i'])
        optim_t.load_state_dict(ckpt['optim_t'])
        optim_v.load_state_dict(ckpt['optim_v'])

        args.it = ckpt.get('it', -1)
        args.epoch = ckpt.get('epoch', -1)
        args.psnr_best = ckpt.get('psnr_best', -1)

        del ckpt
    else:
        args.it = -1
        args.epoch = -1
        args.psnr_best = -float('inf')

    os.makedirs(args.out_dir, exist_ok=True)
    args.ckpt_dir = os.path.join(args.out_dir, 'ckpt')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    args.log_dir = os.path.join(args.out_dir, 'logs')
    os.makedirs(args.log_dir, exist_ok=True)
    args.video_dir = os.path.join(args.out_dir, 'videos')
    os.makedirs(args.video_dir, exist_ok=True)
    print(f'args: {args}')

    writer = SummaryWriter(log_dir=args.log_dir)
    if args.open_tensorboard:
        open_tensorboard(args.log_dir)

    trainer = Trainer(args, generator, discriminator, inv_net, train_pose_params, val_pose_params,
                      optim_g, optim_d, optim_i, optim_t, optim_v,
                      scheduler_g, scheduler_d, scheduler_i, scheduler_t, scheduler_v,
                      train_loader, eval_loader, dynamic_patch_sampler, static_patch_sampler, full_img_sampler,
                      writer, device, it=args.it, epoch=args.epoch, psnr_best=args.psnr_best)
    trainer.train()
