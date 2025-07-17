from models.wide_resnet import wide_resnet_28_10
from models.resnet import pre_resnet18, resnet18


# def inplace_relu(m):
#     classname = m.__class__.__name__
#     if classname.find('ReLU') != -1:
#         m.inplace = True


def get_classifier(P, n_classes=10):
    if P.model == 'pre_resnet18':
        classifier = pre_resnet18(num_classes=n_classes)
    elif P.model == 'resnet18':
        if P.dataset == 'tinyimagenet':
            classifier = resnet18(num_classes=n_classes, stride=2)
        else:
            classifier = resnet18(num_classes=n_classes)
    elif P.model == 'wrn2810':
        if P.dataset == 'tinyimagenet':
            classifier = wide_resnet_28_10(num_classes=n_classes, stride=2)
        else:
            classifier = wide_resnet_28_10(num_classes=n_classes)
    else:
        raise NotImplementedError()

    # classifier.apply(inplace_relu)
    return classifier
