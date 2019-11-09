import os
import sys
sys.path.append('./')
import argparse
import torch
from config import update_config,get_config
import utils as ptutil
from data.sampler import make_data_sampler
from core import get_segmentation_model
from data import get_segmentation_dataset
from utils.metric_seg import SegmentationMetric
from torch.utils import data
from torchvision import transforms

def parse_args():
    ''' Testing Options for Semantic Segmentation '''
    parser = argparse.ArgumentParser(description='Eval Segmentation')
    parser.add_argument('--framework',
                        default='base',
                        type=str,
                        help='framework name')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    config = get_config(args.framework)
    update_config(config,args)
    is_distributed,device = ptutil.init_test_environment(config,args)
    if config.DATASET.IMG_TRANSFORM:
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406],[.229, .224, .255]),
            ])
        data_kwargs = {'base_size':config.DATASET.BASE_SIZE, 'crop_size':config.DATASET.CROP_SIZE, 'transform':input_transform}
    else:
        data_kwargs = {'base_size':config.DATASET.BASE_SIZE, 'crop_size':config.DATASET.CROP_SIZE, 'transform':None}

    val_dataset = get_segmentation_dataset(config.DATASET.NAME,split=config.TEST.TEST_SPLIT,mode=config.TEST.MODE,**data_kwargs)
    sampler = make_data_sampler(val_dataset,False,is_distributed)
    batch_sampler = data.BatchSampler(sampler=sampler,batch_size=config.TEST.TEST_BATCH_SIZE,drop_last=False)
    val_data = data.DataLoader(val_dataset,shuffle=False,batch_sampler=batch_sampler,
                               num_workers=config.DATASET.WORKERS)
    metric = SegmentationMetric(val_dataset.NUM_CLASS)

    model = get_segmentation_model(config.TEST.MODEL_NAME,nclass=val_dataset.NUM_CLASS)
    if not os.path.exists(config.TEST.PRETRAINED):
        raise RuntimeError('cannot found the pretrained file in {}'.format(config.TEST.PRETRAINED))
    model.load_state_dict(torch.load(config.TEST.PRETRAINED))
    model.keep_shape = True if config.TEST.MODE == 'testval' else False
    model.to(device)
    metric = ptutil.validate(model,val_data,metric,device,config)
    ptutil.synchronize()
    pixAcc,mIoU = ptutil.accumulate_metric(metric)
    if ptutil.is_main_process():
        print('pixAcc: %.4f, mIoU: %.4f'%(pixAcc,mIoU))