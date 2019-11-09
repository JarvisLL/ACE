import os
import sys
import time
import datetime
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch
from torch import optim
from torch.backends import cudnn
from torch.utils import data
from torchvision import transforms
import numpy as np
import utils as ptutil
from utils.metric_seg import SegmentationMetric
from data import get_segmentation_dataset
from data.sampler import make_data_sampler, IterationBasedBatchSampler
from core.lr_scheduler import WarmupPolyLR
from core import get_segmentation_model,get_loss
from apex import amp
import apex

# def fix_bn(m):
#     classname = m.__class__.__name__
#     if classname.find('BatchNorm') != -1:
#         m.eval()
class TrainerBase(object):
    def __init__(self, config,args,logger):
        self.DISTRIBUTED,self.DEVICE = ptutil.init_environment(config,args)
        self.LR = config.TRAIN.LR * len(config.GPUS)
        self.device = torch.device(self.DEVICE)
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        
        if config.DATASET.IMG_TRANSFORM:
            data_kwargs = {'transform':input_transform, 'base_size':config.DATASET.BASE_SIZE,
                           'crop_size':config.DATASET.CROP_SIZE}
        else:
            data_kwargs = {'transform':None, 'base_size':config.DATASET.BASE_SIZE,
                           'crop_size':config.DATASET.CROP_SIZE}
        trainset = get_segmentation_dataset(
            config.DATASET.NAME, split=config.TRAIN.TRAIN_SPLIT, mode='train', **data_kwargs)
        self.per_iter = len(trainset) // (len(config.GPUS) * config.TRAIN.BATCH_SIZE)
        self.max_iter = config.TRAIN.EPOCHS * self.per_iter
        if self.DISTRIBUTED:
            sampler = data.DistributedSampler(trainset)
        else:
            sampler = data.RandomSampler(trainset)
        train_sampler = data.sampler.BatchSampler(sampler, config.TRAIN.BATCH_SIZE, True)
        train_sampler = IterationBasedBatchSampler(train_sampler, num_iterations=self.max_iter)
        self.train_loader = data.DataLoader(trainset, batch_sampler=train_sampler, pin_memory=config.DATASET.PIN_MEMORY,
                                            num_workers=config.DATASET.WORKERS)
        if not config.TRAIN.SKIP_EVAL or 0 < config.TRAIN.EVAL_EPOCHS < config.TRAIN.EPOCHS:
            valset = get_segmentation_dataset(config.DATASET.NAME, split='val', mode='val', **data_kwargs)
            val_sampler = make_data_sampler(valset, False, self.DISTRIBUTED)
            val_batch_sampler = data.sampler.BatchSampler(val_sampler, config.TEST.TEST_BATCH_SIZE, False)
            self.valid_loader = data.DataLoader(valset, batch_sampler=val_batch_sampler,
                                                num_workers=config.DATASET.WORKERS, pin_memory=config.DATASET.PIN_MEMORY)
        # create network
        self.net = get_segmentation_model(config.MODEL.NAME,nclass=trainset.NUM_CLASS).cuda()
        if self.DISTRIBUTED:
            if config.TRAIN.MIXED_PRECISION:
                self.net = apex.parallel.convert_syncbn_model(self.net)
            else:
                self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        if config.TRAIN.RESUME != '':
            self.net = ptutil.model_resume(self.net,config.TRAIN.RESUME,logger).to(self.device)
        # self.net.to(self.device)
        assert config.TRAIN.SEG_LOSS in ('focalloss2d', 'mixsoftmaxcrossentropyohemloss', 'mixsoftmaxcrossentropy'), 'cannot support {}'.format(config.TRAIN.SEG_LOSS)
        if config.TRAIN.SEG_LOSS == 'focalloss2d':
            self.criterion = get_loss(config.TRAIN.SEG_LOSS,gamma=2., use_weight=False, size_average=True, ignore_index=config.DATASET.IGNORE_INDEX)
        elif config.TRAIN.SEG_LOSS == 'mixsoftmaxcrossentropyohemloss':
            min_kept = int(config.TRAIN.BATCH_SIZE // len(config.GPUS) * config.DATASET.CROP_SIZE ** 2 // 16)
            self.criterion = get_loss(config.TRAIN.SEG_LOSS,min_kept=min_kept,ignore_index =config.DATASET.IGNORE_INDEX).to(self.device)
        else:
            self.criterion = get_loss(config.TRAIN.SEG_LOSS,ignore_index=config.DATASET.IGNORE_INDEX)

        self.optimizer = optim.SGD(self.net.parameters(), lr=self.LR, momentum=config.TRAIN.MOMENTUM,
                                   weight_decay=config.TRAIN.WEIGHT_DECAY)
        self.scheduler = WarmupPolyLR(self.optimizer, T_max=self.max_iter, warmup_factor=config.TRAIN.WARMUP_FACTOR,
                                      warmup_iters=config.TRAIN.WARMUP_ITERS, power=0.9)
        # self.net.apply(fix_bn)
        if config.TRAIN.MIXED_PRECISION:
            self.dtype = torch.half
            self.net,self.optimizer = amp.initialize(self.net,self.optimizer,opt_level=config.TRAIN.MIXED_OPT_LEVEL)
        else:
            self.dtype = torch.float
        if self.DISTRIBUTED:
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net, device_ids=[args.local_rank], output_device=args.local_rank)

        # evaluation metrics
        self.metric = SegmentationMetric(trainset.NUM_CLASS)
        self.config = config
        self.logger = logger
        ptutil.mkdir(self.config.TRAIN.SAVE_DIR)
        model_path = os.path.join(self.config.TRAIN.SAVE_DIR,"{}_{}_{}_init.pth"
                                  .format(config.MODEL.NAME,  config.TRAIN.SEG_LOSS, config.DATASET.NAME))
        ptutil.save_model(self.net,model_path,self.logger)
        
    def training(self):
        self.net.train()
        save_to_disk = ptutil.get_rank() == 0
        start_training_time = time.time()
        trained_time = 0
        mIoU = 0
        best_miou = 0
        tic = time.time()
        end = time.time()
        iteration, max_iter = 0, self.max_iter
        save_iter, eval_iter = self.per_iter * self.config.TRAIN.SAVE_EPOCH, self.per_iter * self.config.TRAIN.EVAL_EPOCHS
        self.logger.info("Start training, total epochs {:3d} = total iteration: {:6d}".format(self.config.TRAIN.EPOCHS, max_iter))
        for i, (image, target) in enumerate(self.train_loader):
            iteration += 1
            self.scheduler.step()
            self.optimizer.zero_grad()
            image, target = image.to(self.device,dtype=self.dtype), target.to(self.device)
            if self.config.DATASET.IMG_TRANSFORM == False:
                image = image.permute(0,3,1,2)
            outputs = self.net(image)
            loss_dict = self.criterion(outputs, target)
            loss_dict_reduced = ptutil.reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss = sum(loss for loss in loss_dict.values())
            if self.config.TRAIN.MIXED_PRECISION:
                with amp.scale_loss(loss,self.optimizer) as scale_loss:
                    scale_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()
            trained_time += time.time() - end
            end = time.time()
            if iteration % self.config.TRAIN.LOG_STEP == 0:
                eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
                log_str = ["Iteration {:06d} , Lr: {:.5f}, Cost: {:.2f}s, Eta: {}"
                               .format(iteration, self.optimizer.param_groups[0]['lr'], time.time() - tic,
                                       str(datetime.timedelta(seconds=eta_seconds))),
                           "total_loss: {:.3f}".format(losses_reduced.item())]
                log_str = ', '.join(log_str)
                self.logger.info(log_str)
                tic = time.time()
            if save_to_disk and iteration % save_iter == 0:
                model_path = os.path.join(self.config.TRAIN.SAVE_DIR, "{}_{}_{}_iter_{:06d}.pth"
                                          .format(self.config.MODEL.NAME, self.config.TRAIN.SEG_LOSS, self.config.DATASET.NAME, iteration))
                ptutil.save_model(self.net,model_path,self.logger)
            if self.config.TRAIN.EVAL_EPOCHS > 0 and iteration % eval_iter == 0 and not iteration == max_iter:
                metrics = ptutil.validate(self.net,self.valid_loader,self.metric,self.device,self.config)
                ptutil.synchronize()
                pixAcc, mIoU = ptutil.accumulate_metric(metrics)
                if mIoU !=None and mIoU >= best_miou:
                    best_miou = mIoU
                    model_path = os.path.join(self.config.TRAIN.SAVE_DIR, "{}_{}_{}_best.pth"
                                          .format(self.config.MODEL.NAME, self.config.TRAIN.SEG_LOSS, self.config.DATASET.NAME))
                    ptutil.save_model(self.net,model_path,self.logger)
                if pixAcc is not None:
                    self.logger.info('pixAcc: {:.4f}, mIoU: {:.4f}'.format(pixAcc, mIoU))
                self.net.train()
        if save_to_disk:
            model_path = os.path.join(self.config.TRAIN.SAVE_DIR, "{}_{}_{}_iter_{:06d}.pth"
                                      .format(self.config.MODEL.NAME, self.config.TRAIN.SEG_LOSS, self.config.DATASET.NAME, max_iter))
            ptutil.save_model(self.net,model_path,self.logger)
        total_training_time = int(time.time() - start_training_time)
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        self.logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
        # eval after training
        if not self.config.TRAIN.SKIP_EVAL:
            metrics = ptutil.validate(self.net,self.valid_loader,self.metric,self.device,self.config)
            ptutil.synchronize()
            pixAcc, mIoU = ptutil.accumulate_metric(metrics)
            if pixAcc is not None:
                self.logger.info('After training, pixAcc: {:.4f}, mIoU: {:.4f}'.format(pixAcc, mIoU))