import os
from os.path import join as pjoin
import collections
import json
import torch
import scipy.misc as misc
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
from loader_utils import png_reader_32bit, png_reader_uint8
from PIL import Image
from tqdm import tqdm
from torch.utils import data


class sevenScenesLoader(data.Dataset):
    """Data loader for the Seven Scenes dataset.

    """

    def __init__(self, root, split='test', img_size=(480, 640), img_norm=True):
        self.root = os.path.expanduser(root)
        self.split = split
        self.img_norm = img_norm
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) \
            else (img_size, img_size)

        # for split in ['train', 'test', 'testsmall','small_100']:
        for split in ['test']:
            for folder, _, items in os.walk(self.root):
                if os.path.basename(folder).startswith('seq-'):
                    for file in items:
                        if file.endswith('.txt') and file.startswith('frame'):
                            self.files[split].append(pjoin(folder, file.rstrip('.pose.txt')))
                else:
                    print folder

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name_base = self.files[self.split][index]

        im_path = im_name_base + '.color.png'
        depth_path = im_name_base + '.depth.png'

        im = misc.imread(im_path)

        # depth = png_reader_32bit(depth_path, self.img_size)  # 32bit uint
        depth = misc.imread(depth_path)
        depth = depth.astype(float)
        if depth.ndim == 3:  # to dim 2
            depth = depth[:, :, 0]

        if self.img_norm:
            # Resize scales images from -0.5 ~ 0.5
            im = (im.astype(float) - 128) / 255
            # Get valid from rawdepth
            valid = (depth < 65530).astype(float)
            # Resize scales valid, devide by mean value
            depth = depth * valid
            depth = depth / 1000

        # NHWC -> NCHW
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float()

        valid = torch.from_numpy(valid).float()

        depth = depth[np.newaxis, :, :]
        depth = torch.from_numpy(depth).float()

        # input: im, 3*h*w
        # gt: lb, h*w*3
        # mask: gt!=0,h*w
        # valid: rawdepth!=0, h*w
        # rawdepth: depth with hole, 1*h*w
        # meshdepth: depth with hole, 1*h*w
        return im, torch.ones((3, self.img_size[0], self.img_size[1])), torch.ones(
            self.img_size), valid, depth, depth


# Leave code for debugging purposes
if __name__ == '__main__':
    # Config your local data path
    from loader import get_data_path

    local_path = get_data_path('sevenscenes')
    bs = 5
    dst = sevenScenesLoader(root=local_path)
    testloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(testloader):
        imgs, labels, masks, valids, depths, meshdepths = data

        imgs = imgs.numpy()
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        imgs = imgs + 0.5

        labels = labels.numpy()
        labels = 0.5 * (labels + 1)

        masks = masks.numpy()
        masks = np.repeat(masks[:, :, :, np.newaxis], 3, axis=3)

        valids = valids.numpy()
        valids = np.repeat(valids[:, :, :, np.newaxis], 3, axis=3)

        depths = depths.numpy()
        depths = np.transpose(depths, [0, 2, 3, 1])
        depths = np.repeat(depths, 3, axis=3)

        meshdepths = meshdepths.numpy()
        meshdepths = np.transpose(meshdepths, [0, 2, 3, 1])
        meshdepths = np.repeat(meshdepths, 3, axis=3)

        f, axarr = plt.subplots(bs, 6)
        for j in range(bs):
            # print(im_name[j])
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(labels[j])
            axarr[j][2].imshow(masks[j])
            axarr[j][3].imshow(valids[j])
            axarr[j][4].imshow(depths[j])
            axarr[j][5].imshow(meshdepths[j])

        plt.show()
        a = raw_input()
        if a == 'ex':
            break
        else:
            plt.close()
