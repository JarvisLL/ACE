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
from core.VGG19 import vgg19 as vgg19
from apex import amp
import apex

class TrainerACE(object):
    def __init__(self, config,args,logger):
        self.DISTRIBUTED,self.DEVICE = ptutil.init_environment(config,args)
        self.LR = config.TRAIN.LR * len(config.GPUS)  # scale by num gpus
        self.GENERATOR_LR = config.TRAIN.GENERATOR_LR * len(config.GPUS)
        self.device = torch.device(self.DEVICE)
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        if config.DATASET.IMG_TRANSFORM:
            data_kwargs = {"transform": input_transform, "base_size": config.DATASET.BASE_SIZE,
                           "crop_size": config.DATASET.CROP_SIZE}
        else:
            data_kwargs = {"transform": None, "base_size": config.DATASET.BASE_SIZE,
                           "crop_size": config.DATASET.CROP_SIZE}
        # target dataset 
        targetdataset = get_segmentation_dataset(
            'targetdataset',split='train',mode='train',**data_kwargs)
        trainset = get_segmentation_dataset(
            config.DATASET.NAME, split=config.TRAIN.TRAIN_SPLIT, mode='train', **data_kwargs)
        self.per_iter = len(trainset) // (len(config.GPUS) * config.TRAIN.BATCH_SIZE)
        targetset_per_iter = len(targetdataset) // (len(config.GPUS) * config.TRAIN.BATCH_SIZE)
        targetset_max_iter = config.TRAIN.EPOCHS * targetset_per_iter
        self.max_iter = config.TRAIN.epochs * self.per_iter
        if self.DISTRIBUTED:
            sampler = data.DistributedSampler(trainset)
            target_sampler = data.DistributedSampler(targetdataset)
        else:
            sampler = data.RandomSampler(trainset)
            target_sampler = data.RandomSampler(targetdataset)
        train_sampler = data.sampler.BatchSampler(sampler, config.TRAIN.BATCH_SIZE, True)
        train_sampler = IterationBasedBatchSampler(train_sampler, num_iterations=self.max_iter)
        self.train_loader = data.DataLoader(trainset, batch_sampler=train_sampler, pin_memory=config.DATASET.PIN_MEMORY,
                                            num_workers=config.DATASET.WORKERS)
        target_train_sampler = data.sampler.BatchSampler(target_sampler,config.TRAIN.BATCH_SIZE,True)
        target_train_sampler = IterationBasedBatchSampler(target_train_sampler,num_iterations=targetset_max_iter)
        self.target_loader = data.DataLoader(targetdataset,batch_sampler=target_train_sampler,pin_memory=False,
                                             num_workers=config.DATASET.WORKERS)
        self.target_trainloader_iter = enumerate(self.target_loader)
        if not config.TRAIN.SKIP_EVAL or 0 < config.TRAIN.EVAL_EPOCH < config.TRAIN.EPOCHS:
            valset = get_segmentation_dataset(config.DATASET.NAME, split='val', mode='val', **data_kwargs)
            val_sampler = make_data_sampler(valset, False, self.DISTRIBUTED)
            val_batch_sampler = data.sampler.BatchSampler(val_sampler, config.TEST.TEST_BATCH_SIZE, False)
            self.valid_loader = data.DataLoader(valset, batch_sampler=val_batch_sampler,
                                                num_workers=config.DATASET.WORKERS, pin_memory=False)

        # create network
        self.seg_net = get_segmentation_model(config.MODEL.SEG_NET,nclass=trainset.NUM_CLASS).cuda()
        self.feature_extracted = vgg19(pretrained=True)
        self.generator = get_segmentation_model(config.MODEL.TARGET_GENERATOR)

        if self.DISTRIBUTED:
            if config.TRAIN.MIXED_PRECISION:
                self.seg_net = apex.parallel.convert_syncbn_model(self.seg_net)
                self.feature_extracted = apex.parallel.convert_syncbn_model(self.feature_extracted)
                self.generator = apex.parallel.convert_syncbn_model(self.generator)
            else:
                self.seg_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.seg_net)
                self.feature_extracted = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.feature_extracted)
                self.generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.generator)

        # resume checkpoint if needed
        if config.TRAIN.RESUME != '':
            logger.info('loading {} parameter ...'.format(config.MODEL.SEG_NET))
            self.seg_net = ptutil.model_resume(self.seg_net,config.TRAIN.RESUME,logger).to(self.device)
        if config.TRAIN.RESUME_GENERATOR != '':
            logger.info('loading {} parameter ...'.format(config.MODEL.TARGET_GENERATOR))
            self.generator = ptutil.model_resume(self.generator,config.TRAIN.RESUME_GENERATOR,logger).to(self.device)

        self.feature_extracted.to(self.device)
        # create criterion
        assert config.TRAIN.SEG_LOSS in ('focalloss2d', 'mixsoftmaxcrossentropyohemloss', 'mixsoftmaxcrossentropy'), 'cannot support {}'.format(config.TRAIN.SEG_LOSS)
        if config.TRAIN.SEG_LOSS == 'focalloss2d':
            self.criterion = get_loss(config.TRAIN.SEG_LOSS,gamma=2., use_weight=False, size_average=True, ignore_index=config.DATASET.IGNORE_INDEX)
        elif config.TRAIN.SEG_LOSS == 'mixsoftmaxcrossentropyohemloss':
            min_kept = int(config.TRAIN.BATCH_SIZE // len(config.GPUS) * config.DATASET.CROP_SIZE ** 2 // 16)
            self.criterion = get_loss(config.TRAIN.SEG_LOSS,min_kept=min_kept,ignore_index=-1).to(self.device)
        else:
            self.criterion = get_loss(config.TRAIN.SEG_LOSS,ignore_index=-1)

        self.gen_criterion = get_loss('mseloss')
        self.kl_criterion = get_loss('criterionkldivergence')
        # optimizer and lr scheduling
        self.optimizer = optim.SGD(self.seg_net.parameters(), lr=self.LR, momentum=config.TRAIN.MOMENTUM,
                                   weight_decay=config.TRAIN.WEIGHT_DECAY)
        self.scheduler = WarmupPolyLR(self.optimizer, T_max=self.max_iter, warmup_factor=config.TRAIN.WARMUP_FACTOR,
                                      warmup_iters=config.TRAIN.WARMUP_ITERS, power=0.9)
        self.gen_optimizer = optim.SGD(self.generator.parameters(),lr=self.GENERATOR_LR,momentum=config.TRAIN.MOMENTUM,
                                       weight_decay=config.TRAIN.WEIGHT_DECAY)
        self.gen_scheduler = WarmupPolyLR(self.gen_optimizer,T_max=self.max_iter,warmup_factor=config.TRAIN.WARMUP_FACTOR,
                                          warmup_iters=config.TRAIN.WARMUP_ITERS,power=0.9)

        if config.TRAIN.MIXED_PRECISION:
            [self.seg_net,self.generator],[self.optimizer,self.gen_optimizer] = amp.initialize(
                [self.seg_net,self.generator],[self.optimizer,self.gen_optimizer],opt_level=config.TRAIN.MIXED_OPT_LEVEL)
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        if self.DISTRIBUTED:
            self.seg_net = torch.nn.parallel.DistributedDataParallel(
                self.seg_net, device_ids=[args.local_rank], output_device=args.local_rank)
            self.generator = torch.nn.parallel.DistributedDataParallel(
                self.generator,device_ids=[args.local_rank],output_device=args.local_rank)
            self.feature_extracted = torch.nn.parallel.DistributedDataParallel(
                self.feature_extracted,device_ids=[args.local_rank],output_device=args.local_rank)

        # evaluation metrics
        self.metric = SegmentationMetric(trainset.NUM_CLASS)
        self.config = config
        self.logger =logger
        self.seg_dir = os.path.join(self.config.TRAIN.SAVE_DIR,'seg')
        ptutil.mkdir(self.seg_dir)
        self.generator_dir = os.path.join(self.config.TRAIN.SAVE_DIR,'generator')
        ptutil.mkdir(self.generator_dir)
        
    def training(self):
        self.seg_net.train()
        self.generator.train()
        self.feature_extracted.eval()
        for param in self.feature_extracted.parameters():
            param.requires_grad = False
        
        save_to_disk = ptutil.get_rank() == 0
        start_training_time = time.time()
        trained_time = 0
        best_miou = 0
        mean = torch.tensor([0.485,0.456,0.406]).float().cuda().view(1,3,1,1)
        std = torch.tensor([0.229,0.224,0.225]).float().cuda().view(1,3,1,1)
        tic = time.time()
        end = time.time()
        iteration, max_iter = 0, self.max_iter
        save_iter, eval_iter = self.per_iter * self.config.TRAIN.SAVE_EPOCH, self.per_iter * self.config.TRAIN.EVAL_EPOCH
        # save_iter, eval_iter = 10, 10
        self.logger.info("Start training, total epochs {:3d} = total iteration: {:6d}".format(self.config.TRAIN.EPOCHS, max_iter))
        for i, (source_image, label) in enumerate(self.train_loader):
            iteration += 1
            self.scheduler.step()
            # self.optimizer.zero_grad()
            self.gen_scheduler.step()
            # self.gen_optimizer.zero_grad()
            source_image, label = source_image.to(self.device,dtype=self.dtype), label.to(self.device)
            try:
                _,batch = self.target_trainloader_iter.__next__()
            except:
                self.target_trainloader_iter = enumerate(self.target_loader)
                _,batch = self.target_trainloader_iter.__next__()
            target_image = batch.to(self.device,dtype=self.dtype)
            if self.config.DATASET.IMG_TRANSFORM == False:
                source_image = source_image.permute(0,3,1,2)
                target_image = target_image.permute(0,3,1,2)
                source_image_norm = (((source_image / 255) - mean) / std)
                target_image_norm = (((target_image / 255) - mean) / std)
            else:
                source_image_norm = source_image
                target_image_norm = target_image
            source_feature = self.feature_extracted(source_image_norm)
            target_feature = self.feature_extracted(target_image_norm)

            target_feature_mean = torch.mean(target_feature,(2,3),keepdim=True)
            target_feature_var = torch.std(target_feature,(2,3),keepdim=True)
            source_feature_mean = torch.mean(source_feature,(2,3),keepdim=True)
            source_feature_var = torch.std(source_feature,(2,3),keepdim=True)

            adain_feature = ((source_feature - source_feature_mean) / (source_feature_var +0.00001)) * (target_feature_var +0.00001) + target_feature_mean
            gen_image_norm = self.generator(adain_feature)
            gen_image = ((gen_image_norm * std) + mean) * 255

            gen_image_feature = self.feature_extracted(gen_image_norm)
            gen_image_feature_mean = torch.mean(gen_image_feature,(2,3),keepdim=True)
            gen_image_feature_var = torch.std(gen_image_feature,(2,3),keepdim=True)
            #adain_feature <--> gen_image_feature gen_image_feature gen_image_feature_mean <--> target_feature_mean
            #gen_image_feature_var <--> target_feature_var
            loss_feature_dict = self.gen_criterion(gen_image_feature,adain_feature)
            loss_mean_dict = self.gen_criterion(gen_image_feature_mean,target_feature_mean)
            loss_var_dict = self.gen_criterion(gen_image_feature_var,target_feature_var)
            
            loss_feature = sum(loss for loss in loss_feature_dict.values())
            loss_feature_dict_reduced = ptutil.reduce_loss_dict(loss_feature_dict)
            loss_feature_reduced = sum(loss for loss in loss_feature_dict_reduced.values())
            
            loss_mean = sum(loss for loss in loss_mean_dict.values())
            loss_mean_dict_reduced = ptutil.reduce_loss_dict(loss_mean_dict)
            loss_mean_reduced = sum(loss for loss in loss_mean_dict_reduced.values())
            
            loss_var = sum(loss for loss in loss_var_dict.values())
            loss_var_dict_reduced = ptutil.reduce_loss_dict(loss_var_dict)
            loss_var_reduced = sum(loss for loss in loss_var_dict_reduced.values())

            loss_gen = loss_feature + loss_mean + loss_var
            # train source image
            outputs = self.seg_net(source_image)
            source_seg_loss_dict = self.criterion(outputs, label)
            # train gen image
            gen_outputs = self.seg_net(gen_image)
            gen_seg_loss_dict = self.criterion(gen_outputs,label)
            # reduce losses over all GPUs for logging purposes
            outputs = outputs.detach()
            kl_loss_dict = self.kl_criterion(gen_outputs,outputs)

            source_seg_loss_dict_reduced = ptutil.reduce_loss_dict(source_seg_loss_dict)
            # print(type(loss_dict_reduced))
            source_seg_losses_reduced = sum(loss for loss in source_seg_loss_dict_reduced.values())
            source_seg_loss = sum(loss for loss in source_seg_loss_dict.values())
            # source_seg_loss.backward()
            gen_seg_loss_dict_reduced = ptutil.reduce_loss_dict(gen_seg_loss_dict)
            gen_seg_losses_reduced = sum(loss for loss in gen_seg_loss_dict_reduced.values())
            gen_seg_loss = sum(loss for loss in gen_seg_loss_dict.values())
            kl_loss_dict_reduced = ptutil.reduce_loss_dict(kl_loss_dict)
            kl_losses_reduced = sum(loss for loss in kl_loss_dict_reduced.values())
            kl_loss = sum(loss for loss in kl_loss_dict.values())
            loss_seg = source_seg_loss + gen_seg_loss + kl_loss*10
            # loss_seg.backward(retain_graph=True)
            # loss = loss_gen + loss_seg
            # loss.backward()
            if config.TRAIN.MIXED_PRECISION:
                with amp.scale_loss(loss_gen,self.gen_optimizer,loss_id=1) as errGen_scale:
                    errGen_scale.backward()
                with amp.scale_loss(loss_seg,self.optimizer,loss_id=2) as errSeg_scale:
                    errSeg_scale.backward()
            else:
                loss = loss_gen + loss_seg
                loss.backward()

            if iteration % 8 == 0:
                self.optimizer.step()
                self.gen_optimizer.step()
                self.optimizer.zero_grad()
                self.gen_optimizer.zero_grad()
            trained_time += time.time() - end
            end = time.time()
            if iteration % self.config.TRAIN.LOG_STEP == 0:
                eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
                log_str = ["Iteration {:06d} , Lr: {:.5f}, Cost: {:.2f}s, Eta: {}"
                               .format(iteration, self.optimizer.param_groups[0]['lr'], time.time() - tic,
                                       str(datetime.timedelta(seconds=eta_seconds))),
                           "source_seg_loss: {:.6f}, gen_seg_loss:{:.6f}, kl_loss:{:.6f}".format(source_seg_losses_reduced.item(),
                            gen_seg_losses_reduced.item(),kl_losses_reduced.item()*10),
                           "feature_loss:{:.6f}, mean_loss:{:.6f}, var_loss:{:.6f}".format(loss_feature_reduced.item(),loss_mean_reduced.item(),
                            loss_var_reduced.item())]
                log_str = ', '.join(log_str)
                self.logger.info(log_str)
                tic = time.time()
            if save_to_disk and iteration % save_iter == 0:
                model_path = os.path.join(self.seg_dir, "{}_{}_{}_iter_{:06d}.pth"
                                          .format(self.config.MODEL.SEG_NET, self.config.TRAIN.SEG_LOSS, self.config.DATASET.NAME, iteration))
                # self.save_model(model_path)
                ptutil.save_model(self.seg_net,model_path,self.logger)
                generator_path = os.path.join(self.generator_dir,'{}_{}_{}_iter_{:06d}.pth'
                                              .format(self.config.MODEL.TARGET_GENERATOR, self.config.TRAIN.SEG_LOSS, self.config.DATASET.NAME, iteration))
                # self.save_model_generator(generator_path)
                ptutil.save_model(self.generator,generator_path,self.logger)
            # Do eval when training, to trace the mAP changes and see performance improved whether or nor
            if self.config.TRAIN.EVAL_EPOCH > 0 and iteration % eval_iter == 0 and not iteration == max_iter:
                metrics = ptutil.validate(self.seg_net,self.valid_loader,self.metric,self.device,self.config)
                ptutil.synchronize()
                pixAcc, mIoU = ptutil.accumulate_metric(metrics)
                if mIoU !=None and mIoU >= best_miou:
                    best_miou = mIoU
                    model_path = os.path.join(self.seg_dir, "{}_{}_{}_best.pth"
                                          .format(self.config.MODEL.SEG_NET, self.config.TRAIN.SEG_LOSS, self.config.DATASET.NAME))
                    ptutil.save_model(self.seg_net,model_path,self.logger)
                    generator_path = os.path.join(self.generator_dir,'{}_{}_{}_best.pth'
                                                 .format(self.config.TRAIN.TARGET_GENERATOR, self.config.TRAIN.SEG_LOSS, self.config.DATASET.NAME))
                    ptutil.save_model(self.generator,generator_path,self.logger)
                if pixAcc is not None:
                    self.logger.info('pixAcc: {:.4f}, mIoU: {:.4f}'.format(pixAcc, mIoU))
                self.seg_net.train()
        if save_to_disk:
            model_path = os.path.join(self.seg_dir, "{}_{}_{}_iter_{:06d}.pth"
                                      .format(self.config.TRAIN.SEG_NET, self.config.TRAIN.SEG_LOSS, self.config.DATASET.NAME, max_iter))
            ptutil.save_model(self.seg_net,model_path,self.logger)
            generator_path = os.path.join(self.generator_dir,'{}_{}_{}_iter_{:06d}.pth'
                                          .format(self.config.MODEL.TARGET_GENERATOR, self.config.TRAIN.SEG_LOSS, self.config.DATASET.NAME, max_iter))
            ptutil.save_model(self.generator,generator_path,self.logger)
        # compute training time
        total_training_time = int(time.time() - start_training_time)
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        self.logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
        # eval after training
        if not self.config.TRAIN.SKIP_EVAL:
            metrics = ptutil.validate(self.seg_net,self.valid_loader,self.metric,self.device,self.config)
            ptutil.synchronize()
            pixAcc, mIoU = ptutil.accumulate_metric(metrics)
            if pixAcc is not None:
                self.logger.info('After training, pixAcc: {:.4f}, mIoU: {:.4f}'.format(pixAcc, mIoU))
