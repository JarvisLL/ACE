'''Mapillary Dataloader'''
import os
import numpy as np
from PIL import Image
from data.base_seg import SegmentationDataset
import torch
import utils as ptutil
import random

def _get_mapillary_pairs(folder,split='train'):
    def get_path_pairs(img_folder,mask_folder):
        img_paths = []
        mask_paths = []
        for root,_,files in os.walk(img_folder):
            for filename in files:
                if filename.endswith('.jpg'):
                    imgpath = os.path.join(root,filename)
                    # foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('.jpg','.png')
                    maskpath = os.path.join(mask_folder,maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or img:',imgpath,maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths),img_folder))
        return img_paths,mask_paths
    if split in ('train','val'):
        if split == 'train':
            split_root = 'training'
        else:
            split_root = 'validation'
        img_folder = os.path.join(folder,split_root+'/images')
        mask_folder = os.path.join(folder,split_root+'/instances')
        img_paths,mask_paths = get_path_pairs(img_folder,mask_folder)
        return img_paths,mask_paths
    else:
        assert split == 'trainval'
        print('traintest set')
        train_img_folder = os.path.join(folder,'training/images')
        train_mask_folder = os.path.join(folder,'training/instances')
        val_img_folder = os.path.join(folder,'validation/images')
        val_mask_folder = os.path.join(folder,'validation/instances')
        train_img_paths,train_mask_paths = get_path_pairs(train_img_folder,train_mask_folder)
        val_img_paths,val_mask_paths = get_path_pairs(val_img_folder,val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths,mask_paths
class MapillarySegmentation(SegmentationDataset):
    '''Mapillary Dataloader'''
    NUM_CLASS = 17
    def __init__(self,root=os.path.expanduser('/data/mapillary'),split='train',
                mode='train',transform=None,**kwargs):
        super(MapillarySegmentation,self).__init__(
            root,split,mode,transform,**kwargs)
        self.images,self.mask_paths = _get_mapillary_pairs(self.root,self.split)
        assert(len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError('Found 0 images in subfolders of : \
            ' + self.root + '\n')
        self._key = np.array([-1,16,16,6,10,10,10,16,5,8,
                              6,16,7,9,5,9,7,16,16,16,
                              0,0,0,0,8,4,16,16,16,16,
                              9,16,16,16,16,16,16,16,16,16,
                              16,16,11,16,11,16,12,16,16,2,
                              13,3,14,15,16,1,1,1,15,16,
                              1,16,1,16,16,16,-1])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')
    def _class_to_index(self, mask):
        #print(mask)
        mask[mask>65] = 65
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        # mask = self.masks[index]
        mask = Image.open(self.mask_paths[index])
        np_mask = (np.array(mask) / 256).astype(np.int64)
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask
    def _mask_transform(self,mask):
        new_mask = (np.array(mask) / 256).astype(np.int64)
        target = self._class_to_index(new_mask)
        return torch.from_numpy(target)
    def __len__(self):
        return len(self.images)
if __name__ == '__main__':
    data = MapillarySegmentation(split='val',mode='val')
    print(data[0][0].shape,data[0][1].shape)