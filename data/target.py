import os
import numpy as np
from PIL import Image
from data.target_basic import TargetDataset
import torch
import utils as ptutil
import random
def _get_targetdataset_pairs(folder,split='train'):
    def get_path_pairs(img_folder):
        img_path = []
        for root,_,files in os.walk(img_folder):
            for filename in files:
                if filename.endswith('.png'):
                    imgpath = os.path.join(root,filename)
                    #print(imgpath)
                if os.path.isfile(imgpath):
                    img_path.append(imgpath)
                else:
                    print('cannot find the img :',imgpath)
        print('Found {} images in the folder {}'.format(len(img_path),img_folder))
        return img_path
    if split in ('train','val'):
        if split == 'train':
            split_root = 'training'
        else:
            split_root = 'validation'
        img_folder = os.path.join(folder,split_root+'/GT/COLOR')
        img_paths = get_path_pairs(img_folder)
        #print(img_paths)
        return img_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        train_img_folder = os.path.join(folder,'training/GT/COLOR')
        val_img_folder = os.path.join(folder,'validation/GT/COLOR')
        train_img_paths = get_path_pairs(train_img_folder)
        val_img_paths = get_path_pairs(val_img_folder)
        img_paths = train_img_paths + val_img_paths
        return img_paths

class TargetDataLoader(TargetDataset):
    def __init__(self,root=os.path.expanduser('./SYNTHIA'),split='train',
                 mode='train',transform=None,**kwargs):
        super(TargetDataLoader,self).__init__(root,split,mode,transform,**kwargs)
        self.images = _get_targetdataset_pairs(self.root,self.split)
        if len(self.images) == 0:
            raise RuntimeError('Found 0 images in subfolders of : \
                ' + self.root + '\n')
    def __getitem__(self,index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'train':
            img = self._sync_transform(img)
        else:
            assert self.mode == 'val'
            img = self._val_sync_transform(img)
        if self.transform is not None:
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.images)