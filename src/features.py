from __future__ import division, absolute_import, print_function

import sys

import torch
import torch.nn.functional as F
import torchvision

sys.path.append('../pretrained-models.pytorch/pretrainedmodels')
from models import resnext

def last_conv(x, mo):
    if isinstance(mo, torchvision.models.ResNet):
        x = mo.conv1(x)
        x = mo.bn1(x)
        x = mo.relu(x)
        x = mo.maxpool(x)

        x = mo.layer1(x)
        x = mo.layer2(x)
        x = mo.layer3(x)
        x = mo.layer4(x)
    elif isinstance(mo, resnext.ResNeXt101_32x4d):
        x = mo.features(x)
    elif isinstance(mo, resnext.ResNeXt101_64x4d):
        x = mo.features(x)
    else:
        raise 'Not implemented'
    return x

def compact(x, mo):
    x = last_conv(x, mo)
    if isinstance(mo, torchvision.models.ResNet):
        x = mo.avgpool(x)
    elif isinstance(mo, resnext.ResNeXt101_32x4d):
        x = mo.avg_pool(x)
    elif isinstance(mo, resnext.ResNeXt101_64x4d):
        x = mo.avg_pool(x)
    else:
        raise 'Not implemented'
    x = x.view(x.size(0), -1)
    return x
