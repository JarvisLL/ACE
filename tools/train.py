import os
import sys
sys.path.append('./')
import argparse
from config import update_config,get_config
import utils as ptutil
from framework import get_framework
import torch

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='lednet-pelee Segmentation')
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
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    # the parser
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args = parse_args()
    assert args.framework in ('ace','base','distillation','gan'), 'cannot support this framework: {}'.format(args.framework)
    config = get_config(args.framework)
    update_config(config,args)
    assert num_gpus == len(config.GPUS),'GPUS config error'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.GPUS)
    log_dir = os.path.join(config.TRAIN.SAVE_DIR, 'log')
    ptutil.mkdir(log_dir)
    logger = ptutil.setup_logger('TRAIN',log_dir,ptutil.get_rank(), 'log_{}.txt'.format(args.framework),'w')
    logger.info('Using {} GPUs'.format(len(config.GPUS)))
    logger.info(args)
    logger.info(config)
    trainer = get_framework(args.framework,config=config,args=args,logger=logger)
    trainer.training()
    torch.cuda.empty_cache()