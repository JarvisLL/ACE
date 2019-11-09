"""Base segmentation dataset"""
import torch
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import PIL
from data.base import VisionDataset,gamma_trans
from torchvision import transforms as tfs

class SegmentationDataset(VisionDataset):
    """Segmentation Base Dataset"""

    # pylint: disable=abstract-method
    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480):
        super(SegmentationDataset, self).__init__(root)
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        #mask = mask.resize((96,96),resample=PIL.Image.NEAREST,box=None) #if is upsample check it
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        
        if random.random() <0.6:
            w,h = img.size
            #print('00')
            noise_mask = (np.random.random([h,w,3])*3-1.5).astype(np.int8)
            img = np.asarray(img)
            img = img + img*0.05*noise_mask
            img = Image.fromarray(np.uint8(img))
        
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        
        if random.random() < 0.8:
            img = tfs.ColorJitter(brightness=0.3)(img)
        if random.random() < 0.8:
            img = tfs.ColorJitter(contrast=0.3)(img)
        if random.random() < 0.8:
            img = tfs.ColorJitter(hue=0.2)(img)
        if random.random() < 0.8:
            img = tfs.ColorJitter(saturation=0.3)(img)
        
        # final transform
        #mask = mask.resize((64,64),resample=PIL.Image.NEAREST,box=None)#if is_upsample check it
        img, mask = self._img_transform(img), self._mask_transform(mask)
        img = gamma_trans(img)
        return img, mask

    def _img_transform(self, img):
        # return torch.from_numpy(np.array(img))
        return np.array(img)

    def _mask_transform(self, mask):
        # return torch.from_numpy(np.array(mask).astype('int32'))
        return np.array(mask).astype('int64')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0


def ms_batchify_fn(data):
    """Multi-size batchify function"""
    if isinstance(data[0], (str, torch.Tensor)):
        return list(data)
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [ms_batchify_fn(i) for i in data]
    raise RuntimeError('unknown datatype')
