import time
import torch.optim
import numpy as np
import random

from common.train import start_epoch
from utils.utils import AverageMeter
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
criterion_ra = nn.MSELoss()


def train(P, epoch, model, criterion, optimizer, scheduler, loader, adversary, logger=None):

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['ra'] = AverageMeter()

    check = time.time()
    for n, (images, labels) in enumerate(loader):
        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        batch_size = images.size(0)
        benign_images = images.to(device)
        labels = labels.to(device)

        ### adv supervised loss ###
        adv_images = adversary(benign_images, labels)
        adv_outputs = model(adv_images)
        loss_adv_ce = criterion(adv_outputs, labels)
        loss = loss_adv_ce

        ### Robust Perception loss ###
        if epoch >= P.RA_start:
            interpolation_rate = P.RA_ip_rate
            interpolation_images = interpolation_rate * benign_images + (1 - interpolation_rate) * adv_images
            interpolation_output_1 = model(interpolation_images)

            benign_outputs = model(benign_images)
            interpolation_output_2 = interpolation_rate * benign_outputs + (1 - interpolation_rate) * adv_outputs

            loss_ra = criterion_ra(interpolation_output_1, interpolation_output_2)
            loss += P.lam * loss_ra

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        ### Log losses ###
        losses['cls'].update(loss_adv_ce.item(), batch_size)
        if epoch >= P.RA_start:
            losses['ra'].update(loss_ra.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossC %f] [LossRA %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value, losses['ra'].value))

        check = time.time()

    if P.optimizer == 'sgd':
        scheduler.step()

    log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossRA %f]' %
         (batch_time.average, data_time.average,
          losses['cls'].average, losses['ra'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_cls', losses['cls'].average, epoch)
        logger.scalar_summary('train/loss_ra', losses['ra'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)
