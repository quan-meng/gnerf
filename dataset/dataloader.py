from torch.utils.data import DataLoader
from torchvision.transforms import *

from dataset.datasets import *


def create_dataloader(data, split, data_dir, img_wh, batch_size, num_workers):
    dataset = get_dataset(data, split, data_dir, img_wh)

    if split == 'train':
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    elif split == 'val':
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
    else:
        raise NotImplementedError


def get_dataset(data, split, data_dir, img_wh):
    trans = Compose([
        Resize((img_wh[1], img_wh[0])),
        ToTensor(),
        NormalizeForGAN(),
    ])

    kwargs = {'split': split, 'data_dir': data_dir, 'img_wh': img_wh}

    if data == 'blender':
        trans.transforms.insert(2, BlendAToRGB())
        # sort images by file names: keep training and evaluation split unchanged
        sort_key = lambda x: int(x.split('/')[-1][x.split('/')[-1].index('_') + 1:  x.split('/')[-1].index('.')])
        dset = Blender(**kwargs, transforms=trans, sort_key=sort_key)
    elif data == 'dtu':
        # sort images by file names: keep training and evaluation split unchanged
        sort_key = lambda x: int(x.split('/')[-1][5:8])
        dset = DTU(**kwargs, transforms=trans, sort_key=sort_key)
    else:
        raise NotImplementedError

    return dset
