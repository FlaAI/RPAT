import torch

from utils.utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


criterion_ra = torch.nn.MSELoss()


def error_k(output, target, ks=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(ks)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in ks:
        correct_k = correct[:k].view(-1).float().sum(0)
        results.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return results


def error_k_MSE(output, target, ks=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(ks)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in ks:
        correct_k = correct[:k].view(-1).float().sum(0)
        results.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return results, correct


def test_classifier(P, model, loader, steps, logger=None):
    error_top1 = AverageMeter()

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    # Switch to evaluate mode
    mode = model.training
    model.eval()

    for n, (images, labels) in enumerate(loader):
        batch_size = images.size(0)

        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)

        top1, = error_k(outputs.data, labels, ks=(1,))
        error_top1.update(top1.item(), batch_size)

    log_(' * [Error@1 %.3f] [Acc %.3f]' %
         (error_top1.average, 100. - error_top1.average))

    if logger is not None:
        logger.scalar_summary('eval/clean_error', error_top1.average, steps)

    model.train(mode)

    return error_top1.average


def test_classifier_adv(P, model, loader, steps, adversary=None,
                        logger=None, softmax=False, ret='clean'):
    error_top1 = AverageMeter()
    error_adv_top1 = AverageMeter()

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    if adversary is None:
        adversary = lambda x, y: x

    # Switch to evaluate mode
    mode = model.training
    model.eval()

    for n, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        adv_images = adversary(images, labels)

        outputs = model(images)
        adv_outputs = model(adv_images)

        # Measure accuracy and record loss
        top1, = error_k(outputs.data, labels, ks=(1,))
        adv_top1, = error_k(adv_outputs.data, labels, ks=(1,))

        batch_size = images.size(0)
        error_top1.update(top1.item(), batch_size)
        error_adv_top1.update(adv_top1.item(), batch_size)

    log_(' * [Error@1 %.3f] [AdvError@1 %.3f]' %
         (error_top1.average, error_adv_top1.average))
    log_(' * [Acc@1 %.3f] [AdvAcc@1 %.3f]' %
         (100. - error_top1.average, 100. - error_adv_top1.average))

    if logger is not None:
        logger.scalar_summary('error/clean', error_top1.average, steps)
        logger.scalar_summary('error/adv', error_adv_top1.average, steps)

    model.train(mode)

    if ret == 'clean':
        return error_top1.average
    elif ret == 'adv':
        return error_adv_top1.average
    else:
        raise NotImplementedError()


def test_classifier_adv_weak(P, model, loader, steps, adversary=None,
                             logger=None, softmax=False, ret='clean'):
    error_top1 = AverageMeter()
    error_adv_top1 = AverageMeter()

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    if adversary is None:
        adversary = lambda x, y: x

    # Switch to evaluate mode
    mode = model.training
    model.eval()

    for n, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        adv_images = adversary(images, labels)

        outputs = model(images)

        weak_rate = P.weak / 8.0
        adv_images_weak = (1 - weak_rate) * images + weak_rate * adv_images

        adv_outputs = model(adv_images_weak)

        # Measure accuracy and record loss
        top1, = error_k(outputs.data, labels, ks=(1,))
        adv_top1, = error_k(adv_outputs.data, labels, ks=(1,))

        batch_size = images.size(0)
        error_top1.update(top1.item(), batch_size)
        error_adv_top1.update(adv_top1.item(), batch_size)

    log_(' * [Error@1 %.3f] [AdvError@1 %.3f]' %
         (error_top1.average, error_adv_top1.average))
    log_(' * [Acc@1 %.3f] [AdvAcc@1 %.3f]' %
         (100. - error_top1.average, 100. - error_adv_top1.average))

    if logger is not None:
        logger.scalar_summary('error/clean', error_top1.average, steps)
        logger.scalar_summary('error/adv', error_adv_top1.average, steps)

    model.train(mode)

    if ret == 'clean':
        return error_top1.average
    elif ret == 'adv':
        return error_adv_top1.average
    else:
        raise NotImplementedError()


def test_classifier_adv_MSE(P, model, loader, steps, adversary=None,
                            logger=None, softmax=False, ret='clean'):
    error_top1 = AverageMeter()
    error_adv_top1 = AverageMeter()
    MSE_score_avg = AverageMeter()
    MSE_score_benign_avg = AverageMeter()
    MSE_score_adv_avg = AverageMeter()
    MSE_score_correct_avg = AverageMeter()
    MSE_score_benign_correct_avg = AverageMeter()
    MSE_score_adv_correct_avg = AverageMeter()
    MSE_score_wrong_avg = AverageMeter()
    MSE_score_benign_wrong_avg = AverageMeter()
    MSE_score_adv_wrong_avg = AverageMeter()

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    if adversary is None:
        adversary = lambda x, y: x

    # Switch to evaluate mode
    mode = model.training
    model.eval()

    for n, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        adv_images = adversary(images, labels)

        outputs = model(images)
        adv_outputs = model(adv_images)

        images_inter = 0.5 * images + 0.5 * adv_images
        inter_outputs = model(images_inter)

        MSE_score = criterion_ra(outputs, adv_outputs)
        MSE_score_benign = criterion_ra(outputs, inter_outputs)
        MSE_score_adv = criterion_ra(inter_outputs, adv_outputs)

        # Measure accuracy and record loss
        (top1,), correct_ts = error_k_MSE(outputs.data, labels, ks=(1,))
        (adv_top1,), correct_ts = error_k_MSE(adv_outputs.data, labels, ks=(1,))

        correct = []
        wrong = []
        for idx in range(labels.size(0)):
            if correct_ts[0][idx] == True:
                correct.append(idx)
            else:
                wrong.append(idx)
        # print(correct)
        # print(wrong)
        # print(len(correct)+len(wrong))

        MSE_score_correct = criterion_ra(outputs[correct], adv_outputs[correct])
        MSE_score_benign_correct = criterion_ra(outputs[correct], inter_outputs[correct])
        MSE_score_adv_correct = criterion_ra(inter_outputs[correct], adv_outputs[correct])

        MSE_score_wrong = criterion_ra(outputs[wrong], adv_outputs[wrong])
        MSE_score_benign_wrong = criterion_ra(outputs[wrong], inter_outputs[wrong])
        MSE_score_adv_wrong = criterion_ra(inter_outputs[wrong], adv_outputs[wrong])

        batch_size = images.size(0)
        error_top1.update(top1.item(), batch_size)
        error_adv_top1.update(adv_top1.item(), batch_size)

        MSE_score_avg.update(MSE_score.item(), batch_size)
        MSE_score_benign_avg.update(MSE_score_benign.item(), batch_size)
        MSE_score_adv_avg.update(MSE_score_adv.item(), batch_size)
        MSE_score_correct_avg.update(MSE_score_correct.item(), batch_size)
        MSE_score_benign_correct_avg.update(MSE_score_benign_correct.item(), batch_size)
        MSE_score_adv_correct_avg.update(MSE_score_adv_correct.item(), batch_size)
        MSE_score_wrong_avg.update(MSE_score_wrong.item(), batch_size)
        MSE_score_benign_wrong_avg.update(MSE_score_benign_wrong.item(), batch_size)
        MSE_score_adv_wrong_avg.update(MSE_score_adv_wrong.item(), batch_size)

    log_(' * [Error@1 %.3f] [AdvError@1 %.3f]' %
         (error_top1.average, error_adv_top1.average))
    log_(' * [Acc@1 %.3f] [AdvAcc@1 %.3f]' %
         (100. - error_top1.average, 100. - error_adv_top1.average))

    log_(' * [MSE Score %.12f]' % (MSE_score_avg.average))
    log_(' * [MSE Score Benign %.12f]' % (MSE_score_benign_avg.average))
    log_(' * [MSE Score Adv %.12f]' % (MSE_score_adv_avg.average))
    log_(' * [MSE Score Correct %.12f]' % (MSE_score_correct_avg.average))
    log_(' * [MSE Score Benign Correct %.12f]' % (MSE_score_benign_correct_avg.average))
    log_(' * [MSE Score Adv Correct %.12f]' % (MSE_score_adv_correct_avg.average))
    log_(' * [MSE Score Wrong %.12f]' % (MSE_score_wrong_avg.average))
    log_(' * [MSE Score Benign Wrong %.12f]' % (MSE_score_benign_wrong_avg.average))
    log_(' * [MSE Score Adv Wrong %.12f]' % (MSE_score_adv_wrong_avg.average))

    if logger is not None:
        logger.scalar_summary('error/clean', error_top1.average, steps)
        logger.scalar_summary('error/adv', error_adv_top1.average, steps)

    model.train(mode)

    if ret == 'clean':
        return error_top1.average
    elif ret == 'adv':
        return error_adv_top1.average
    else:
        raise NotImplementedError()


from training.train.ce_rand import batch_multiply, clamp

def test_classifier_random(P, model, loader, steps, adversary=None,
                           logger=None, softmax=False, ret='clean'):
    error_top1 = AverageMeter()
    error_adv_top1 = AverageMeter()

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    if adversary is None:
        adversary = lambda x, y: x

    # Switch to evaluate mode
    mode = model.training
    model.eval()

    for n, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        ##### Adding random noise #####

        delta = torch.zeros_like(images)
        delta = torch.nn.Parameter(delta)

        delta.data.uniform_(-1, 1)
        delta.data = batch_multiply(8 / 255, delta.data)
        delta.data = clamp(images + delta.data, min=0, max=1) - images

        images_rand = images + delta

        ###############################

        outputs = model(images)
        adv_outputs = model(images_rand)

        # Measure accuracy and record loss
        top1, = error_k(outputs.data, labels, ks=(1,))
        adv_top1, = error_k(adv_outputs.data, labels, ks=(1,))

        batch_size = images.size(0)
        error_top1.update(top1.item(), batch_size)
        error_adv_top1.update(adv_top1.item(), batch_size)

    log_(' * [Error@1 %.3f] [AdvError@1 %.3f]' %
         (error_top1.average, error_adv_top1.average))
    log_(' * [Acc@1 %.3f] [AdvAcc@1 %.3f]' %
         (100. - error_top1.average, 100. - error_adv_top1.average))

    if logger is not None:
        logger.scalar_summary('error/clean', error_top1.average, steps)
        logger.scalar_summary('error/adv', error_adv_top1.average, steps)

    model.train(mode)

    if ret == 'clean':
        return error_top1.average
    elif ret == 'adv':
        return error_adv_top1.average
    else:
        raise NotImplementedError()


def test_classifier_random_MSE(P, model, loader, steps, adversary=None,
                               logger=None, softmax=False, ret='clean'):
    error_top1 = AverageMeter()
    error_adv_top1 = AverageMeter()
    MSE_score_avg = AverageMeter()
    MSE_score_benign_avg = AverageMeter()
    MSE_score_adv_avg = AverageMeter()
    MSE_score_correct_avg = AverageMeter()
    MSE_score_benign_correct_avg = AverageMeter()
    MSE_score_adv_correct_avg = AverageMeter()
    MSE_score_wrong_avg = AverageMeter()
    MSE_score_benign_wrong_avg = AverageMeter()
    MSE_score_adv_wrong_avg = AverageMeter()

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    if adversary is None:
        adversary = lambda x, y: x

    # Switch to evaluate mode
    mode = model.training
    model.eval()

    for n, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        ##### Adding random noise #####

        delta = torch.zeros_like(images)
        delta = torch.nn.Parameter(delta)

        delta.data.uniform_(-1, 1)
        delta.data = batch_multiply(8 / 255, delta.data)
        delta.data = clamp(images + delta.data, min=0, max=1) - images

        images_rand = images + delta
        images_rand_inter = images + 0.5 * delta

        ###############################

        outputs = model(images)
        inter_outputs = model(images_rand_inter)
        adv_outputs = model(images_rand)

        MSE_score = criterion_ra(outputs, adv_outputs)
        MSE_score_benign = criterion_ra(outputs, inter_outputs)
        MSE_score_adv = criterion_ra(inter_outputs, adv_outputs)

        # Measure accuracy and record loss
        (top1,), correct_ts = error_k_MSE(outputs.data, labels, ks=(1,))
        (adv_top1,), correct_ts = error_k_MSE(adv_outputs.data, labels, ks=(1,))

        correct = []
        wrong = []
        for idx in range(labels.size(0)):
            if correct_ts[0][idx] == True:
                correct.append(idx)
            else:
                wrong.append(idx)
        # print(correct)
        # print(wrong)
        # print(len(correct)+len(wrong))

        MSE_score_correct = criterion_ra(outputs[correct], adv_outputs[correct])
        MSE_score_benign_correct = criterion_ra(outputs[correct], inter_outputs[correct])
        MSE_score_adv_correct = criterion_ra(inter_outputs[correct], adv_outputs[correct])

        MSE_score_wrong = criterion_ra(outputs[wrong], adv_outputs[wrong])
        MSE_score_benign_wrong = criterion_ra(outputs[wrong], inter_outputs[wrong])
        MSE_score_adv_wrong = criterion_ra(inter_outputs[wrong], adv_outputs[wrong])

        batch_size = images.size(0)
        error_top1.update(top1.item(), batch_size)
        error_adv_top1.update(adv_top1.item(), batch_size)

        MSE_score_avg.update(MSE_score.item(), batch_size)
        MSE_score_benign_avg.update(MSE_score_benign.item(), batch_size)
        MSE_score_adv_avg.update(MSE_score_adv.item(), batch_size)
        MSE_score_correct_avg.update(MSE_score_correct.item(), batch_size)
        MSE_score_benign_correct_avg.update(MSE_score_benign_correct.item(), batch_size)
        MSE_score_adv_correct_avg.update(MSE_score_adv_correct.item(), batch_size)
        MSE_score_wrong_avg.update(MSE_score_wrong.item(), batch_size)
        MSE_score_benign_wrong_avg.update(MSE_score_benign_wrong.item(), batch_size)
        MSE_score_adv_wrong_avg.update(MSE_score_adv_wrong.item(), batch_size)

    log_(' * [Error@1 %.3f] [AdvError@1 %.3f]' %
         (error_top1.average, error_adv_top1.average))
    log_(' * [Acc@1 %.3f] [AdvAcc@1 %.3f]' %
         (100. - error_top1.average, 100. - error_adv_top1.average))

    log_(' * [MSE Score %.12f]' % (MSE_score_avg.average))
    log_(' * [MSE Score Benign %.12f]' % (MSE_score_benign_avg.average))
    log_(' * [MSE Score Adv %.12f]' % (MSE_score_adv_avg.average))
    log_(' * [MSE Score Correct %.12f]' % (MSE_score_correct_avg.average))
    log_(' * [MSE Score Benign Correct %.12f]' % (MSE_score_benign_correct_avg.average))
    log_(' * [MSE Score Adv Correct %.12f]' % (MSE_score_adv_correct_avg.average))
    log_(' * [MSE Score Wrong %.12f]' % (MSE_score_wrong_avg.average))
    log_(' * [MSE Score Benign Wrong %.12f]' % (MSE_score_benign_wrong_avg.average))
    log_(' * [MSE Score Adv Wrong %.12f]' % (MSE_score_adv_wrong_avg.average))

    if logger is not None:
        logger.scalar_summary('error/clean', error_top1.average, steps)
        logger.scalar_summary('error/adv', error_adv_top1.average, steps)

    model.train(mode)

    if ret == 'clean':
        return error_top1.average
    elif ret == 'adv':
        return error_adv_top1.average
    else:
        raise NotImplementedError()


### Eval by C&W / Boundary Attack ###

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import CarliniLInfMethod, CarliniL2Method

def test_classifier_adv_cw(P, model, loader):
    error_adv_top1 = AverageMeter()

    if P.dataset == 'tinyimagenet':
        image_size = (3, 64, 64)
        n_classes = 200
    elif P.dataset == 'cifar100':
        image_size = (3, 32, 32)
        n_classes = 100
    else:
        image_size = (3, 32, 32)
        n_classes = 10

    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay),
        input_shape=image_size,
        nb_classes=n_classes,
        clip_values=(0, 1),
        channels_first=False
    )
    if P.distance == 'L2':
        adversary_cw = CarliniL2Method(estimator=classifier)
    else:
        adversary_cw = CarliniLInfMethod(estimator=classifier)

    # Switch to evaluate mode
    mode = model.training
    model.eval()

    for n, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        adv_images = adversary_cw.generate(x=images, y=labels)

        adv_outputs = model(adv_images)

        # Measure accuracy and record loss
        batch_size = images.size(0)
        adv_top1, = error_k(adv_outputs.data, labels, ks=(1,))
        error_adv_top1.update(adv_top1.item(), batch_size)

    print(' *** C&W Attack *** [AdvAcc@1 %.3f]' % (100. - error_adv_top1.average))

    model.train(mode)

    return error_adv_top1.average


# from art.attacks.evasion import BoundaryAttack, HopSkipJump
#
#
# def test_classifier_adv_boundary_attack(P, model, loader):
#     error_adv_top1 = AverageMeter()
#
#     if P.dataset == 'tinyimagenet':
#         image_size = (3, 64, 64)
#         n_classes = 200
#     elif P.dataset == 'cifar100':
#         image_size = (3, 32, 32)
#         n_classes = 100
#     else:
#         image_size = (3, 32, 32)
#         n_classes = 10
#
#     classifier = PyTorchClassifier(
#         model=model,
#         loss=torch.nn.CrossEntropyLoss(),
#         optimizer=torch.optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay),
#         input_shape=image_size,
#         nb_classes=n_classes,
#         clip_values=(0, 1),
#         channels_first=False
#     )
#     adversary_boundary_attack = BoundaryAttack(estimator=classifier)
#
#     # Switch to evaluate mode
#     mode = model.training
#     model.eval()
#
#     for n, (images, labels) in enumerate(loader):
#         images, labels = images.to(device), labels.to(device)
#         adv_images = adversary_boundary_attack.generate(x=images, y=labels)
#
#         adv_outputs = model(adv_images)
#
#         # Measure accuracy and record loss
#         batch_size = images.size(0)
#         adv_top1, = error_k(adv_outputs.data, labels, ks=(1,))
#         error_adv_top1.update(adv_top1.item(), batch_size)
#
#     print(' *** Boundary Attack *** [AdvAcc@1 %.3f]' % (100. - error_adv_top1.average))
#
#     model.train(mode)
#
#     return error_adv_top1.average
#
#
# def test_classifier_adv_hopskipjump(P, model, loader):
#     error_adv_top1 = AverageMeter()
#
#     if P.dataset == 'tinyimagenet':
#         image_size = (3, 64, 64)
#         n_classes = 200
#     elif P.dataset == 'cifar100':
#         image_size = (3, 32, 32)
#         n_classes = 100
#     else:
#         image_size = (3, 32, 32)
#         n_classes = 10
#
#     classifier = PyTorchClassifier(
#         model=model,
#         loss=torch.nn.CrossEntropyLoss(),
#         optimizer=torch.optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay),
#         input_shape=image_size,
#         nb_classes=n_classes,
#         clip_values=(0, 1),
#         channels_first=False
#     )
#     adversary_hopskipjump = HopSkipJump(estimator=classifier)
#
#     # Switch to evaluate mode
#     mode = model.training
#     model.eval()
#
#     for n, (images, labels) in enumerate(loader):
#         images, labels = images.to(device), labels.to(device)
#         adv_images = adversary_hopskipjump.generate(x=images, y=labels)
#
#         adv_outputs = model(adv_images)
#
#         # Measure accuracy and record loss
#         batch_size = images.size(0)
#         adv_top1, = error_k(adv_outputs.data, labels, ks=(1,))
#         error_adv_top1.update(adv_top1.item(), batch_size)
#
#     print(' *** HopSkipJump Attack *** [AdvAcc@1 %.3f]' % (100. - error_adv_top1.average))
#
#     model.train(mode)
#
#     return error_adv_top1.average
