# The source code is from: https://github.com/YisenWang/MART
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from copy import deepcopy

criterion_ra = nn.MSELoss()


def _batch_l2norm(x):
    x_flat = x.view(x.size(0), -1)
    return torch.norm(x_flat, dim=1)


def mart_loss(model,
              x_natural,
              y,
              optimizer,
              eps_iter=0.007,
              eps=0.031,
              nb_iter=10,
              beta=6.0,
              clip_min=0.0,
              clip_max=1.0,
              distance='Linf',
              return_adv=False,
              return_adv_sample=False):

    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'Linf':
        for _ in range(nb_iter):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + eps_iter * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - eps), x_natural + eps)
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    elif distance == 'L2':
        for _ in range(nb_iter):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            grad_norm = _batch_l2norm(grad).view(-1, 1, 1, 1)
            grad = grad / (grad_norm + 1e-8)

            x_adv = x_adv.detach() + eps_iter * grad
            eta_x_adv = x_adv - x_natural
            eta_x_adv = eta_x_adv.renorm(p=2, dim=0, maxnorm=eps)

            x_adv = x_natural + eta_x_adv
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    else:
        raise NotImplementedError()
    model.train()

    x_adv = torch.clamp(x_adv, clip_min, clip_max).clone().detach()

    optimizer.zero_grad()

    logits = model(x_natural)

    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

    if return_adv_sample:
        return loss_adv, beta * loss_robust, logits_adv, x_adv

    if return_adv:
        return loss_adv, beta * loss_robust, logits_adv

    return loss_adv, beta * loss_robust


def mart_loss_ra(model,
              x_natural,
              y,
              optimizer,
              eps_iter=0.007,
              eps=0.031,
              nb_iter=10,
              beta=6.0,
              clip_min=0.0,
              clip_max=1.0,
              distance='Linf',
              return_adv=False,
              return_adv_sample=False,
              interpolation_rate=0.5,
              lam=1.0):

    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'Linf':
        for _ in range(nb_iter):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + eps_iter * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - eps), x_natural + eps)
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    elif distance == 'L2':
        for _ in range(nb_iter):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            grad_norm = _batch_l2norm(grad).view(-1, 1, 1, 1)
            grad = grad / (grad_norm + 1e-8)

            x_adv = x_adv.detach() + eps_iter * grad
            eta_x_adv = x_adv - x_natural
            eta_x_adv = eta_x_adv.renorm(p=2, dim=0, maxnorm=eps)

            x_adv = x_natural + eta_x_adv
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    else:
        raise NotImplementedError()
    model.train()

    x_adv = torch.clamp(x_adv, clip_min, clip_max).clone().detach()

    optimizer.zero_grad()

    logits = model(x_natural)

    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

    ### Robust Perception loss ###
    interpolation_images = interpolation_rate * x_natural + (1 - interpolation_rate) * x_adv
    interpolation_output_1 = model(interpolation_images)

    interpolation_output_2 = interpolation_rate * logits + (1 - interpolation_rate) * logits_adv

    loss_ra = criterion_ra(interpolation_output_1, interpolation_output_2)
    loss_adv += lam * loss_ra

    if return_adv_sample:
        return loss_adv, beta * loss_robust, logits_adv, x_adv

    if return_adv:
        return loss_adv, beta * loss_robust, logits_adv

    return loss_adv, beta * loss_robust
