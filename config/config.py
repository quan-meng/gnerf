import argparse
import yaml


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('conf_file', type=str, help='path to the config file')
    parser.add_argument('--data_dir', type=str, help='patch to the dataset')
    parser.add_argument('--out_dir', type=str, default='.', help='path to the output directory')

    parser.add_argument('--name', type=str, default='GNeRF', help='name of the project')

    # training
    parser.add_argument('--num_epoch', type=int, default=40000, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=6, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    parser.add_argument('--ckpt', type=str, default=None, help='pretrained checkpoint path to load')
    parser.add_argument('--chunk', type=int, default=1024 * 32, help='chunk size to split the input to avoid OOM')
    parser.add_argument('--gan_type', type=str, default='standard', choices=['standard', 'wgan'])
    parser.add_argument('--reg_type', type=str, default='real', choices=['real', 'fake', 'wgangp', 'wgangp0'])
    parser.add_argument('--reg_param', type=float, default=10., help='weight of the discriminator regularization')
    parser.add_argument('--progressvie_training', type=bool, default=True, help='whether use progressive training')
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--video_every', type=int, default=10000)
    parser.add_argument('--empty_cache_every', type=int, default=1000)

    # nerf
    parser.add_argument('--xyz_freq', type=int, default=10)
    parser.add_argument('--dir_freq', type=int, default=4)
    parser.add_argument('--N_samples', type=int, default=64, help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=64, help='number of fine samples')
    parser.add_argument('--fc_depth', type=int, default=8)
    parser.add_argument('--fc_dim', type=int, default=360)
    parser.add_argument('--decrease_noise', type=bool, default=True, help='whether decrease the noise added to sigma')

    # patch sampler
    parser.add_argument('--min_scale', type=float, default=0.0, help='')
    parser.add_argument('--max_scale', type=float, default=1.0, help='')
    parser.add_argument('--scale_anneal', type=float, default=0.0002, help='')
    parser.add_argument('--patch_size', type=int, default=16, help='size of image patch')
    parser.add_argument('--random_scale', type=bool, default=True, help='')

    # inversion network
    parser.add_argument('--inv_size', type=int, default=64, help='input image size of inversion network')

    # discriminator
    parser.add_argument('--ndf', type=int, default=64, help='network dimension of the discriminator')
    parser.add_argument('--conditional', type=int, default=True, help='whether require scale condition')
    parser.add_argument('--policy', nargs='+', default=['color', 'cutout'], help='differentiable augmentation policies')

    # log
    parser.add_argument('--open_tensorboard', action='store_true', help='whether open tensorboard')

    return parser.parse_args()


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively.
    :param dict1: first dictionary to be updated
    :param dict2: second dictionary which entries should be used
    :return:
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def load_config():
    cfg = {}
    # load default config
    args = get_opts()
    cfg_default = {}
    if args.conf_file is not None:
        with open('./config/default.yaml', 'r') as f:
            cfg_default.update(yaml.load(f, Loader=yaml.FullLoader))
    update_recursive(cfg, cfg_default)

    # load specific config
    cfg_special = {}
    if args.conf_file is not None:
        with open(args.conf_file, 'r') as f:
            cfg_special.update(yaml.load(f, Loader=yaml.FullLoader))
    update_recursive(cfg, cfg_special)

    cfg.update(vars(args))

    return argparse.Namespace(**cfg)
