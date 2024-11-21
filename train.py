import os
import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
import subprocess
from datetime import timedelta
from torch.autograd import Variable
from torch.amp import GradScaler, autocast


from config import Config
from utils.loss import PixLoss
from dataset import MyData
from models.baseline import SANet
from utils.utils import Logger, AverageMeter, set_seed, check_state_dict
from evaluation.valid import valid

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, get_rank

parser = argparse.ArgumentParser(description='')
parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
parser.add_argument('--pip install opencv-python', default=120, type=int)
parser.add_argument('--trainset', default='COD', type=str, help="Options: 'DIS5K'")
parser.add_argument('--ckpt_dir', default=None, help='Temporary folder')
parser.add_argument('--testsets', default='CAMO_TestingDataset+CHAMELEON_TestingDataset+COD10K_TestingDataset+NC4K_TestingDataset', type=str)
parser.add_argument('--dist', default=False, type=lambda x: x == 'True')
parser.add_argument('--epochs', default=200, type=int)
args = parser.parse_args()

config = Config()
'''
if config.rand_seed:
    set_seed(config.rand_seed)
'''

# DDP
to_be_distributed = args.dist
if to_be_distributed:
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 10))
    device = int(os.environ["LOCAL_RANK"])
else:
    device = config.device


# make dir for ckpt
os.makedirs(args.ckpt_dir, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.ckpt_dir, "log.txt"))
logger_loss_idx = 1

# log model and optimizer params
# logger.info("Model details:"); logger.info(model)
#logger.info("datasets: load_all={}, compile={}.".format(config.load_all, config.compile))
logger.info("Other hyperparameters:");
logger.info(args)
print('batch size:', config.batch_size)

# 'DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4'
if os.path.exists(os.path.join(config.dataset_root, config.task, args.testsets.strip('+').split('+')[0])):
    args.testsets = args.testsets.strip('+').split('+')
else:
    args.testsets = []


# Init model
def prepare_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, to_be_distributed=False, is_train=True):
    if to_be_distributed:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size), pin_memory=True,
            shuffle=False, sampler=DistributedSampler(dataset), drop_last=True
        )
    else:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size, 0), pin_memory=True,
            shuffle=is_train, drop_last=True
        )


def init_data_loaders(to_be_distributed):
    # Prepare dataset, 返回一个dataloader 对象
    train_loader = prepare_dataloader(
        MyData(datasets=config.training_set, image_size=config.train_size, is_train=True),
        config.batch_size, to_be_distributed=to_be_distributed, is_train=True
    )
    print(len(train_loader), "batches of train dataloader {} have been created.".format(config.training_set))
    test_loaders = {}
    for testset in args.testsets:
        _data_loader_test = prepare_dataloader(
            MyData(datasets=testset, image_size=config.train_size, is_train=False),
            config.batch_size_valid, is_train=False
        )
        print(len(_data_loader_test), "batches of valid dataloader {} have been created.".format(testset))
        test_loaders[testset] = _data_loader_test
    return train_loader, test_loaders


def init_models_optimizers(epochs, to_be_distributed):
    model = SANet(backbone_pretrained=True)
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            state_dict = torch.load(args.resume, map_location='cpu')
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Distributed training
    if to_be_distributed:
        model = DDP(model, device_ids=[device])
    else:
        model = model.to(device)
    # Setting optimizer
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=1e-2)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[lde if lde > 0 else epochs + lde + 1 for lde in config.lr_decay_epochs],
        gamma=config.lr_decay_rate
    )

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()
    logger.info("Optimizer details:")
    logger.info(optimizer)
    logger.info("Scheduler details:")
    logger.info(lr_scheduler)

    return model, optimizer, lr_scheduler, scaler


class Trainer:
    def __init__(self, data_loaders, model_opt_lrsch):
        self.model, self.optimizer, self.lr_scheduler, self.scaler = model_opt_lrsch
        self.train_loader, self.test_loaders = data_loaders
        if config.out_ref:
            self.criterion_gdt = nn.BCELoss()

        # Setting Losses
        self.pix_loss = PixLoss()
        self.lambdas_pix_multi = config.lambdas_pix_multi

        # Best Epoch
        self.best_epoch = 0
        self.lowest_loss = float('inf')

        # Others
        self.start_time = time.time()  # 记录训练开始时间
        self.loss_log = AverageMeter()
        self.last_log_time = time.time()  # 记录上一次日志输出的时间
        self.total_iterations = len(self.train_loader) * args.epochs

        # tg bot
        self.notified_start = False  # 添加一个标志位，表示是否已发送开始通知


    def _train_batch(self, batch):
        inputs = batch[0].to(device)
        gts = batch[1].to(device)

        # Define weights for each loss component
        weight_pred_m = self.lambdas_pix_multi['weight_pred_m']
        weight_mid_pred1 = self.lambdas_pix_multi['weight_mid_pred1']
        weight_mid_pred2 = self.lambdas_pix_multi['weight_mid_pred2']

        # Use autocast to enable mixed precision training
        with torch.amp.autocast(device_type='cuda'):
            pred_m, mid_pred1, mid_pred2 = self.model(inputs)

            # Loss
            loss_pix1 = self.pix_loss(pred_m, torch.clamp(gts, 0, 1)) * weight_pred_m
            loss_pix2 = self.pix_loss(mid_pred1, torch.clamp(gts, 0, 1)) * weight_mid_pred1
            loss_pix3 = self.pix_loss(mid_pred2, torch.clamp(gts, 0, 1)) * weight_mid_pred2
            total_weight = weight_pred_m + weight_mid_pred1 + weight_mid_pred2
            loss_pix = (loss_pix1 + loss_pix2 + loss_pix3)/total_weight
            self.loss_dict['loss_pix'] = loss_pix.item()
            loss = loss_pix

        # Backward pass and optimization step
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.loss_log.update(loss.item())


    def train_epoch(self, epoch):
        global logger_loss_idx
        self.model.train()
        self.loss_dict = {}
        self.loss_log.reset()
        if epoch > args.epochs + config.IoU_finetune_last_epochs:
            self.pix_loss.lambdas_pix_last['bce'] *= 0.5
            self.pix_loss.lambdas_pix_last['ssim'] *= 1
            self.pix_loss.lambdas_pix_last['iou'] *= 0.5
            # self.pix_loss.lambdas_pix_last['ual'] += 0.1

        if epoch < config.Prior_finetune_first_epochs:
            self.lambdas_pix_multi['weight_mid_pred1'] = 1.0
            self.lambdas_pix_multi['weight_mid_pred2'] = 1.0

        if epoch > args.epochs + config.NonPrior_finetune_last_epochs:
            self.lambdas_pix_multi['weight_mid_pred1'] *= 0
            self.lambdas_pix_multi['weight_mid_pred2'] *= 0

        for batch_idx, batch in enumerate(self.train_loader):
            self._train_batch(batch)
            # tg通知代码块
            # 发送开始通知
            if (epoch == 1 or (config.resume and epoch == int(
                    args.resume.rstrip('.pth').split('ep')[-1]) + 1)) and batch_idx == 40 and not self.notified_start:
                self.notified_start = True  # 设置标志位，表示已发送开始通知
                current_time = time.time()
                elapsed_time = current_time - self.last_log_time
                # 计算剩余的迭代次数
                completed_iterations = (epoch-1) * len(self.train_loader) + batch_idx
                remaining_iterations = self.total_iterations - completed_iterations

                # 估算剩余时间
                remaining_time = elapsed_time * (remaining_iterations / 20)
                estimated_end_time = current_time + remaining_time
                estimated_end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(estimated_end_time))
                subprocess.run(['python', 'train_start_notice.py', str(self.start_time), estimated_end_time_str])

            # Logger
            if batch_idx % 20 == 0:
                current_time = time.time()
                elapsed_time = current_time - self.last_log_time
                self.last_log_time = current_time

                # 计算剩余的迭代次数
                completed_iterations = (epoch-1) * len(self.train_loader) + batch_idx
                remaining_iterations = self.total_iterations - completed_iterations

                # 估算剩余时间
                remaining_time = elapsed_time * (remaining_iterations / 20)
                estimated_end_time = current_time + remaining_time
                estimated_end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(estimated_end_time))

                info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}].'.format(epoch, args.epochs, batch_idx,
                                                                       len(self.train_loader))
                info_loss = 'Training Losses'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += ', {}: {:.3f}'.format(loss_name, loss_value)
                if (epoch == 1 and batch_idx == 0):
                    logger.info(' '.join((info_progress, info_loss)))
                else:
                    info_time = ' | Estimated end time: {}'.format(estimated_end_time_str)
                    logger.info(' '.join((info_progress, info_loss, info_time)))

        info_loss = '@==Final== Epoch[{0}/{1}]  Training Loss: {loss.avg:.3f}  '.format(epoch, args.epochs,
                                                                                        loss=self.loss_log)
        logger.info(info_loss)

        if self.loss_log.avg < self.lowest_loss:
            self.lowest_loss = self.loss_log.avg
            self.best_epoch = epoch
            if self.best_epoch != args.epochs:
                # 保存最低损失对应的模型参数
                best_model_path = os.path.join(args.ckpt_dir, f'ep-1.pth')
                torch.save(self.model.state_dict(), best_model_path)
                # 单独记录最低损失对应的日志
                best_loss_info = f"Best loss found at epoch {self.best_epoch}: {self.lowest_loss:.3f}, saved success!"
                logger.info(best_loss_info)
            else:
                last_best_epoch_model_path = os.path.join(args.ckpt_dir, 'ep-1.pth')
                if os.path.exists(last_best_epoch_model_path):
                    os.remove(last_best_epoch_model_path)
                    logger.info(f"The last epoch is the best one. Skipping save the best epoch.")

        self.lr_scheduler.step()
        return self.loss_log.avg

    def validate_model(self, epoch):
        num_image_testset_all = {'DIS-VD': 470, 'DIS-TE1': 500, 'DIS-TE2': 500, 'DIS-TE3': 500, 'DIS-TE4': 500}
        num_image_testset = {}
        for testset in args.testsets:
            if 'DIS-TE' in testset:
                num_image_testset[testset] = num_image_testset_all[testset]
        weighted_scores = {'f_max': 0, 'f_mean': 0, 'f_wfm': 0, 'sm': 0, 'e_max': 0, 'e_mean': 0, 'mae': 0}
        len_all_data_loaders = 0
        self.model.epoch = epoch
        # data_loader_test 是 test_loader 字典中的第二个元素
        for testset, data_loader_test in self.test_loaders.items():
            print('Validating {}...'.format(testset))
            self.valid = valid(self.model, data_loader_test[:2], pred_dir='.',
                               method=args.ckpt_dir.split('/')[-1] if args.ckpt_dir.split('/')[-1].strip('.').strip(
                                   '/') else 'tmp_val', testset=testset, only_S_MAE=config.only_S_MAE, device=device)
            performance_dict = self.valid
            print('Test set: {}:'.format(testset))
            if config.only_S_MAE:
                print('Smeasure: {:.4f}, MAE: {:.4f}'.format(
                    performance_dict['sm'], performance_dict['mae']
                ))
            else:
                print('Fmax: {:.4f}, Fwfm: {:.4f}, Smeasure: {:.4f}, Emean: {:.4f}, MAE: {:.4f}'.format(
                    performance_dict['f_max'], performance_dict['f_wfm'], performance_dict['sm'],
                    performance_dict['e_mean'], performance_dict['mae']
                ))
            if '-TE' in testset:
                for metric in ['sm', 'mae'] if config.only_S_MAE else ['f_max', 'f_mean', 'f_wfm', 'sm', 'e_max',
                                                                       'e_mean', 'mae']:
                    weighted_scores[metric] += performance_dict[metric] * len(data_loader_test)
                len_all_data_loaders += len(data_loader_test)
        print('Weighted Scores:')
        for metric, score in weighted_scores.items():
            if score:
                print('\t{}: {:.4f}.'.format(metric, score / len_all_data_loaders))

    def record_total_training_time(self):
        end_time = time.time()
        total_time = end_time - self.start_time
        total_time_str = str(timedelta(seconds=round(total_time)))
        logger.info(f"Total training time: {total_time_str}")


def main():
    trainer = Trainer(
        data_loaders=init_data_loaders(to_be_distributed),
        model_opt_lrsch=init_models_optimizers(args.epochs, to_be_distributed)
    )

    epoch_st = 1
    if args.resume:
        if os.path.isfile(args.resume):
            epoch_st = int(args.resume.rstrip('.pth').split('ep')[-1]) + 1

    epoch = epoch_st - 1  # 在进入循环前初始化epoch变量
    try:
        for epoch in range(epoch_st, args.epochs + 1):
            train_loss = trainer.train_epoch(epoch)
            # Save checkpoint
            # DDP
            # 没保存最后一个epoch
            if epoch >= config.save_last and epoch % config.save_step == 0:
                torch.save(
                    trainer.model.module.state_dict() if to_be_distributed else trainer.model.state_dict(),
                    os.path.join(args.ckpt_dir, 'ep{}.pth'.format(epoch))
                )
            if config.val_step and epoch >= args.epochs - config.save_last and (
                    args.epochs - epoch) % config.val_step == 0:

                if to_be_distributed:
                    if get_rank() == 0:
                        print('Validating at rank-{}...'.format(get_rank()))
                        trainer.validate_model(epoch)
                else:
                    trainer.validate_model(epoch)
    except KeyboardInterrupt:
        print("Training interrupted. Saving last epoch...")
        torch.save(
            trainer.model.module.state_dict() if to_be_distributed else trainer.model.state_dict(),
            os.path.join(args.ckpt_dir, 'ep{}.pth'.format(epoch))
        )

    finally:
        trainer.record_total_training_time()

        if to_be_distributed:
            destroy_process_group()

if __name__ == '__main__':
    main()
