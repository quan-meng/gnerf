__all__ = ['NormalizeForGAN', 'BlendAToRGB', 'Blender', 'DTU']

import os
import glob
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# normalize the range of input image from [0, 1] to [-1, 1]
class NormalizeForGAN(object):
    def __call__(self, x):
        return x * 2 - 1

    def __repr__(self):
        return self.__class__.__name__ + '()'


# blend A to RGB
class BlendAToRGB(object):
    def __call__(self, x):
        if x.shape[0] == 4:
            x = x[:3, ...] * x[-1:, ...] + (1 - x[-1:, ...])
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Blender(Dataset):
    def __init__(self, split, data_dir, img_wh, transforms, sort_key=None):
        super(Blender, self).__init__()
        self.split = split
        self.data_dir = data_dir
        self.img_wh = img_wh
        self.transforms = transforms
        self.sort_key = sort_key

        self.filenames = self.get_filenames(self.data_dir)
        assert len(self.filenames) > 0, 'File dir is empty'
        self.img_wh_original = Image.open(self.filenames[0]).size
        assert self.img_wh_original[1] * self.img_wh[0] == self.img_wh_original[0] * self.img_wh[1], \
            f'You must set @img_wh to have the same aspect ratio as ' \
            f'({self.img_wh_original[0]}, {self.img_wh_original[1]}) !'

        self.imgs = self.load_imgs(self.filenames)
        self.intrinsics, self.poses = self.get_camera_params()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        return img, idx

    def load_imgs(self, filenames):
        imgs = []
        for p in filenames:
            img = Image.open(p)

            if self.transforms is not None:
                img = self.transforms(img)
            imgs.append(img)

        return imgs

    def get_filenames(self, root):
        filenames = glob.glob(f'{root}/{self.split}/*.png')

        if self.sort_key is not None:
            filenames.sort(key=self.sort_key)
        else:
            filenames.sort()

        if self.split == 'val':  # only validate 8 images
            filenames = filenames[:8]

        return filenames

    def get_camera_params(self):
        file_path = os.path.join(self.data_dir, f'transforms_{self.split}.json')

        with open(file_path, 'r') as f:
            meta = json.load(f)

        poses = []
        for frame in meta['frames']:
            pose = torch.tensor(frame['transform_matrix'])[:3, :4]
            poses.append(pose[None])

        poses = torch.cat(poses)  # [N, 3, 4]s

        cx, cy = [x // 2 for x in self.img_wh_original]
        focal = 0.5 * 800 / np.tan(0.5 * meta['camera_angle_x'])  # original focal length

        intrinsics = torch.tensor([
            [focal, 0, cx], [0, focal, cy], [0, 0, 1.]
        ])

        scale = torch.tensor([self.img_wh[0] / self.img_wh_original[0], self.img_wh[1] / self.img_wh_original[1]])
        intrinsics[:2] *= scale[:, None]

        return intrinsics, poses


class DTU(Dataset):
    def __init__(self, split, data_dir, img_wh, transforms, sort_key=None):
        super(DTU, self).__init__()
        self.split = split
        self.data_dir = data_dir
        self.img_wh = img_wh
        self.transforms = transforms
        self.sort_key = sort_key

        self.filenames = self.get_filenames(self.data_dir)
        assert len(self.filenames) > 0, 'File dir is empty'
        self.img_wh_original = Image.open(self.filenames[0]).size
        assert self.img_wh_original[1] * self.img_wh[0] == self.img_wh_original[0] * self.img_wh[1], \
            f'You must set @img_wh to have the same aspect ratio as ' \
            f'({self.img_wh_original[0]}, {self.img_wh_original[1]}) !'

        self.imgs = self.load_imgs(self.filenames)
        self.intrinsics, self.poses = self.get_camera_params()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        return img, idx

    def load_imgs(self, filenames):
        imgs = []
        for p in filenames:
            img = Image.open(p)

            if self.transforms is not None:
                img = self.transforms(img)
            imgs.append(img)

        return imgs

    def get_filenames(self, data_dir):
        filenames = glob.glob(f'{data_dir}/*_3_*.png')  # choose images with a same light condition

        if self.sort_key is not None:
            filenames.sort(key=self.sort_key)
        else:
            filenames.sort()

        # choose every 8 images as evaluation images, the rest as training images
        val_indices = list(np.arange(7, len(filenames), 8))
        if self.split == 'train':
            filenames = [filenames[x] for x in np.arange(0, len(filenames)) if x not in val_indices]
        elif self.split == 'val':
            filenames = [filenames[idx] for idx in val_indices]

        return filenames

    def get_camera_params(self):
        prefix = '/'.join(self.data_dir.split('/')[:-2] + ['Cameras', 'train'])
        id_list = [os.path.join(prefix, str(int(name.split('/')[-1][5:8]) - 1).zfill(8) + '_cam.txt') for name in
                   self.filenames]

        intrinsics, poses = [], []
        for id in id_list:
            with open(id) as f:
                text = f.read().splitlines()

                pose_text = text[text.index('extrinsic') + 1:text.index('extrinsic') + 5]
                pose_text = torch.tensor([[float(b) for b in a.strip().split(' ')] for a in pose_text])
                pose_text = torch.inverse(pose_text)

                intrinsic_text = text[text.index('intrinsic') + 1:text.index('intrinsic') + 4]
                intrinsic_text = torch.tensor([[float(b) for b in a.strip().split(' ')] for a in intrinsic_text])
                intrinsic_text[:2, :] *= 4.0  # rescale with image size

                poses.append(pose_text[None, :3, :4])
                intrinsics.append(intrinsic_text[None])

        poses = torch.cat(poses)  # [N, 3, 4]
        intrinsics = torch.cat(intrinsics, 0)

        intrinsics = intrinsics.mean(dim=0)  # assume intrinsics of all cameras are the same
        poses[:, :, 3] /= 200.0

        scale = torch.tensor([self.img_wh[0] / self.img_wh_original[0], self.img_wh[1] / self.img_wh_original[1]])
        intrinsics[:2] *= scale[:, None]

        return intrinsics, poses
