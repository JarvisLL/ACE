import os
import numpy as np
from PIL import Image
from data.base_seg import SegmentationDataset
import torch
import utils as ptutil
import random

cls_dict = [
    [0,0,0],
    [128,128,128],
    [128,0,0],
    [128,64,128],
    [0,0,192],
    [64,64,128],
    [128,128,0],
    [192,192,128],
    [64,0,128],
    [192,128,128],
    [64,0,128],
    [192,128,128],
    [64,64,0],
    [0,128,192],
    [0,172,0],
    [0,128,128],
]

def convert_color_map(gt_color,cls_dict):
    semantic_map = np.zeros(gt_color.shape[:2]).astype(np.int64)
    for i in range(len(cls_dict)):
        color = cls_dict[i]
        eq = np.equal(gt_color,color)
        eq = np.all(eq, axis=-1)
        semantic_map = semantic_map + eq * i
    return semantic_map

def _get_synthia_pairs(folder,split='train'):
	def get_path_pairs(img_folder,mask_folder):
		img_paths = []
		mask_paths = []
		for root,_,files in os.walk(img_folder):
			for filename in files:
				if filename.endswith('.png'):
					imgpath = os.path.join(root,filename)
					maskpath = imgpath.replace('/RGB', '/GT/COLOR')
					# foldername = os.path.basename(os.path.dirname(imgpath))
					# maskpath = os.path.join(mask_folder,foldername,filename)
					if os.path.isfile(imgpath) and os.path.isfile(maskpath):
						img_paths.append(imgpath)
						mask_paths.append(maskpath)
					else:
						print('cannot find the mask or img',imgpath,maskpath)
		print('Found {} images in the folder {}'.format(len(img_paths),img_folder))
		return img_paths,mask_paths
	if split in ('train','val'):
		if split == 'train':
			split_root = 'training'
		else:
			split_root = 'validation'
		img_folder = os.path.join(folder,split_root+'/RGB')
		mask_folder = os.path.join(folder,split_root+'/GT/COLOR')
		img_paths,mask_paths = get_path_pairs(img_folder,mask_folder)
		return img_paths,mask_paths
	else:
		assert split == 'trainval'
		print('traintest set')
		train_img_folder = os.path.join(folder,'training/RGB')
		train_mask_folder = os.path.join(folder,'training/GT/COLOR')
		val_img_folder = os.path.join(folder,'validation/RGB')
		val_mask_folder = os.path.join(folder,'validation/RGB/COLOR')
		train_img_paths,train_mask_paths = get_path_pairs(train_img_folder,train_mask_folder)
		val_img_paths,val_mask_paths = get_path_pairs(val_img_folder,val_mask_folder)
		img_paths = train_img_paths + val_img_paths
		mask_paths = train_mask_paths + val_mask_paths
	return img_paths,mask_paths

class SynthiaSegmentation(SegmentationDataset):

	NUM_CLASS = 14
	def __init__(self,root=os.path.expanduser('./SYNTHIA'),split='train',
		        mode='train',transform=None,**kwargs):
	    super(SynthiaSegmentation,self).__init__(
	    	root,split,mode,transform,**kwargs)
	    self.images,self.mask_paths = _get_synthia_pairs(self.root,self.split)
	    assert(len(self.images) == len(self.mask_paths))
	    if len(self.images) == 0:
	    	raise RuntimeError('Found 0 images in subfolder of : \
	    		' + self.root + '\n')
	
	def __getitem__(self,index):
	    img = Image.open(self.images[index]).convert('RGB')
	    if self.mode == 'test':
	    	if self.transform is not None:
	    		img = self.transform(img)
	    	return img,os.path.basename(self.images[index])
	    mask = Image.open(self.mask_paths[index])
	    np_mask = np.array(mask).astype(np.int64)
	    if self.mode == 'train':
	    	img, mask = self._sync_transform(img, mask)
	    elif self.mode == 'val':
	    	img,mask = self._val_sync_transform(img, mask)
	    else:
	    	assert self.mode == 'testval'
	    	img, mask = self._img_transform(img), self._mask_transfrom(mask)
	    if self.transform is not None:
	    	img = self.transform(img)
	    return img, mask
	def _mask_transform(self,mask):
		new_mask = np.array(mask).astype(np.int64)
		new_mask = convert_color_map(new_mask[:,:,:3],cls_dict)
		return torch.from_numpy(new_mask)
	def __len__(self):
		return len(self.images)

if __name__ == '__main__':
	data = SynthiaSegmentation(split='train',mode='train')
	print(data[0][0].shape)