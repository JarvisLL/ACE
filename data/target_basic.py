import torch
import random
import numpy as np
import PIL
from PIL import Image,ImageOps,ImageFilter
from data.base import VisionDataset,gamma_trans
from torchvision import transforms as tfs
import numpy as np
import cv2

class TargetDataset(VisionDataset):
    def __init__(self,root,split,mode,transform,base_size=1024,crop_size=512):
        super(TargetDataset,self).__init__(root)
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
    def _val_sync_transform(self,img):
        outsize = self.crop_size
        short_size = outsize
        w,h = img.size
        if w > h:
            oh = short_size
            ow = int( 1.0*w*oh / h)
        else:
            ow = short_size
            oh = int( 1.0*h*ow / w)
        img = img.resize((ow,oh),Image.BILINEAR)
        w,h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1,y1,x1+outsize,y1+outsize))
        img = self._img_transform(img)
        return img
    def _sync_transform(self,img):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        short_size = random.randint(int(self.base_size*0.5),int(self.base_size*2.0))
        w,h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0*h*ow / w)
        else:
            oh = short_size
            ow = int(1.0*w*oh / h)
        img = img.resize((ow,oh),Image.BILINEAR)
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
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

        img = self._img_transform(img)
        return img

    def _img_transform(self, img):
        # return torch.from_numpy(np.array(img))
        return np.array(img)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0