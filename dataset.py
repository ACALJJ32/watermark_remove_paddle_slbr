import os
import glob
import paddle
from transforms import Compose
import numpy as np
import cv2


def random_crop(im1, im2, patch_size = 512):
    assert im1.shape == im2.shape, "Value Error!"
    _, h, w = im1.shape

    # random choose top coordinated for im1, im2 patch
    top = np.random.randint(h - patch_size + 1)
    left = np.random.randint(w - patch_size + 1)

    im1 = im1[:, top:top+patch_size, left:left+patch_size]
    im2 = im2[:, top:top+patch_size, left:left+patch_size]

    return im1, im2


class TrainDataset(paddle.io.Dataset):
    def __init__(self, dataset_root, transforms, random_crop=True):
        if dataset_root is None:
            raise ValueError("dataset_root is None")
            
        self.dataset_root = dataset_root

        if transforms is not None:
            self.transforms = Compose(transforms)
        else:
            self.transforms = None
        
        self.random_crop = random_crop

        self.lq_list = glob.glob(os.path.join(self.dataset_root, "images", "*.jpg"))
        self.bg_path = os.path.join(self.dataset_root, "bg_images")

        self.lq_list.sort()

    def get_back_ground_index(self, lq):
        frame_name = os.path.basename(lq)

        bg_index_list = frame_name.split("_")  # e.g. [bg, images, 00033, 0001.jpg]
        bg_index_list.pop()

        gt_name = "_".join(bg_index_list) + ".jpg"
        gt = os.path.join(self.bg_path, gt_name)
        return gt

    def __getitem__(self, index):
        lq = self.lq_list[index]
        gt = self.get_back_ground_index(lq)

        if self.transforms is not None:
            lq, gt = self.transforms(lq, gt)
        
        if self.random_crop:
            lq, gt = random_crop(lq, gt, patch_size=512)
            mask = cv2.absdiff(lq, gt)
        
        return lq, gt, mask
    
    def __len__(self):
        return len(self.lq_list)