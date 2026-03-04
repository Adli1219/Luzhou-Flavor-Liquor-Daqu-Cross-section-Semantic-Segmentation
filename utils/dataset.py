import os
import os.path as osp
import logging
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1.0, fixed_size=(1024, 1024)):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.fixed_size = fixed_size
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.img_names = os.listdir(imgs_dir)
        logging.info(f'Creating dataset with {len(self.img_names)} examples')

    def __len__(self):
        return len(self.img_names)

    @classmethod
    def preprocess(cls, pil_img, is_mask=False, fixed_size=(1024, 1024)):
        pil_img = pil_img.resize(fixed_size, resample=Image.NEAREST if is_mask else Image.BILINEAR)
        img_nd = np.array(pil_img)

        if img_nd.ndim == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        if not is_mask:
            img_nd = img_nd / 255.0  # 图像归一化

        img_trans = img_nd.transpose((2, 0, 1))  # HWC -> CHW
        return img_trans.astype(np.float32 if not is_mask else np.uint8)

    def __getitem__(self, i):
        img_name = self.img_names[i]
        img_path = osp.join(self.imgs_dir, img_name)

        mask_name = osp.splitext(img_name)[0] + '.png'
        mask_path = osp.join(self.masks_dir, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        img = self.preprocess(img, is_mask=False, fixed_size=self.fixed_size)
        mask = self.preprocess(mask, is_mask=True, fixed_size=self.fixed_size)

        img = torch.from_numpy(img).float().clone()
        mask = torch.from_numpy(mask).long().squeeze().clone()  # squeeze for 2D mask

        return {'image': img, 'mask': mask}
