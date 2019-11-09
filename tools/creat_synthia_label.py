import cv2
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import utils as ptutil

cls_dict = [
    [0  ,0  ,0  ],
    [128,128,128],
    [128,0  ,0  ],
    [128,64 ,128],
    [0  ,0  ,192],
    [64 ,64 ,128],
    [128,128,0  ],
    [192,192,128],
    [64 ,0  ,128],
    [192,128,128],
    [64 ,64, 0  ],
    [0  ,128,192],
    [0  ,172,0  ],
    [0  ,128,128],
]

def _get_synthia_color_pairs(folder):
    def get_path_pairs(label_folder):
        label_path = []
        filenames = []
        for root,_,files in os.walk(label_folder):
            for filename in files:
                if filename.endswith('.png'):
                    imgpath = os.path.join(root,filename)
                if os.path.isfile(imgpath):
                    label_path.append(imgpath)
                    filenames.append(filename)
                else:
                    print('cannot find the img :',imgpath)
        print('Found {} images in the folder {}'.format(len(label_path),label_folder))
        return label_path
    train_label_folder = os.path.join(folder,'training/GT/COLOR')
    train_label_paths = get_path_pairs(train_label_folder)
    val_label_folder = os.path.join(folder,'validation/GT/COLOR')
    val_label_paths = get_path_pairs(val_label_folder)
    return train_label_paths,val_label_paths

def convert_color_map(gt_color,cls_dict):
    semantic_map = np.zeros(gt_color.shape[:2]).astype(np.int64)
    for i in range(len(cls_dict)):
        color = cls_dict[i]
        eq = np.equal(gt_color,color)
        eq = np.all(eq, axis=-1)
        semantic_map = semantic_map + eq * i
    return semantic_map

def creat_single_channel_label(label_paths):
    tbar = tqdm(label_paths)
    for i,label_path in enumerate(tbar):
        img = Image.open(label_path)
        img = np.array(img).astype(np.int64)
        img = img[:,:,:3]
        semantic = convert_color_map(img,cls_dict)
        save_path = label_path.replace('COLOR','SINGLE')
        cv2.imwrite(save_path,semantic)
    print('finished')

if __name__ == "__main__":
	train_label_paths,val_label_paths = _get_synthia_color_pairs('./SYNTHIA')
	creat_single_channel_label(train_label_paths)