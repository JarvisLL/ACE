import argparse
import os
import torch
from tqdm import tqdm
from torch.utils import data

def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def mkdir(path):

    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---new folder '{}'---".format(path))

def init_environment(config,args):
    is_DISTRIBUTED = len(config.GPUS) > 1
    if not config.NO_CUDA and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
        DEVICE = 'cuda'
    else:
        is_DISTRIBUTED = False
        DEVICE = 'cpu'
    if is_DISTRIBUTED:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method=config.INIT_METHOD)
    return is_DISTRIBUTED,DEVICE

def init_test_environment(config,args):
    is_DISTRIBUTED = len(config.GPUS) > 1
    if not config.NO_CUDA and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False if config.TEST.MODE == 'testval' else config.CUDNN.BENCHMARK
        DEVICE = 'cuda'
    else:
        is_DISTRIBUTED = False
        DEVICE = 'cpu'
    if is_DISTRIBUTED:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',init_method=config.INIT_METHOD)
    return is_DISTRIBUTED,DEVICE


def save_model(model,model_path,logger):
    if isinstance(model,torch.nn.parallel.DistributedDataParallel):
        net = model.module
    else:
        net = model
    torch.save(net.state_dict(),model_path)
    logger.info('Saved checkpoint to {}'.format(model_path))

def validate(model,dataloader,metric,device,config):
    metric.reset()
    torch.cuda.empty_cache()
    if isinstance(model,torch.nn.parallel.DistributedDataParallel):
        net = model.module
    else:
        net = model
    net.eval()
    tbar = tqdm(dataloader)
    for i, (image,target) in enumerate(tbar):
        image, target = image.to(device),target.to(device)
        image = image.float()
        if not config.DATASET.IMG_TRANSFORM:
            image = image.permute(0,3,1,2)
        with torch.no_grad():
            outputs = net(image)
        metric.update(target, outputs)
    return metric

def model_resume(net,resume,logger):
    if os.path.isfile(resume):
        checkpoints = torch.load(resume,map_location=torch.device('cpu'))
        net_state_dict = net.state_dict()
        new_load_dict = {}
        for k in net_state_dict:
            if k in checkpoints and net_state_dict[k].size() == checkpoints[k].size():
                new_load_dict[k] = checkpoints[k]
            else:
                logger.info('Skipped loading parameter: %s',k)
        net_state_dict.update(new_load_dict)
        net.load_state_dict(net_state_dict)
        logger.info('===> load the pretrained model !!!')
        return net
    else:
        raise RuntimeError('===> no checkpoint found at {}'.format(resume))