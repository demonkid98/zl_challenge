from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import os
import argparse
import logging
import glob
import datetime
import sys

import torchvision
from torchvision import transforms, datasets
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import data

from datasets import ZaloUnlabeledLandmarkDataset

sys.path.append('../pretrained-models.pytorch/pretrainedmodels')
from models import resnext
import features as ft

def skip_error_collate(batch):
    batch = filter(lambda x: x is not None, batch)
    return data.dataloader.default_collate(batch)

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--nb_workers', default=2, type=int)
    parser.add_argument('--gpu', action='store_true')
    # parser.add_argument('--compact', action='store_true')
    parser.add_argument('--nb_classes', default=103, type=int)
    parser.add_argument('--base_model', choices=['resnet18', 'resnet50', 'resnet101', 'resnext101_64x4d'])
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--img_pattern', default='*.jpg')
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--model_state_path', required=True)
    parser.add_argument('--out_fname', required=True)

    args = parser.parse_args()
    logging.info('Arguments: %s', args)

    now = datetime.datetime.now()

    MEAN = [0.485, 0.456, 0.406] # imagenet -- 0-1 range
    STD = [0.229, 0.224, 0.225] # imagenet -- 0-1 range
    SCALE = 256
    INPUT_SHAPE = 224

    tensorize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    xform = transforms.Compose([
        transforms.Resize(SCALE),
        transforms.CenterCrop(INPUT_SHAPE),
        tensorize,
    ])

    dataset = ZaloUnlabeledLandmarkDataset(args.img_dir, transform=xform)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.nb_workers, collate_fn=skip_error_collate)
    dataset_size = len(dataset)

    if args.base_model == 'resnet18':
        mo = torchvision.models.resnet18(pretrained=True)
        num_ftrs = mo.fc.in_features
        mo.fc = nn.Linear(num_ftrs, args.nb_classes)
    elif args.base_model == 'resnet50':
        mo = torchvision.models.resnet50(pretrained=True)
        num_ftrs = mo.fc.in_features
        mo.fc = nn.Linear(num_ftrs, args.nb_classes)
    elif args.base_model == 'resnet101':
        mo = torchvision.models.resnet101(pretrained=True)
        num_ftrs = mo.fc.in_features
        mo.fc = nn.Linear(num_ftrs, args.nb_classes)
    elif args.base_model == 'resnext101_64x4d':
        mo = resnext.resnext101_64x4d(pretrained='imagenet')
        num_ftrs = mo.last_linear.in_features
        mo.last_linear = nn.Linear(num_ftrs, args.nb_classes)

    if args.gpu:
        mo.cuda()

    checkpoint = torch.load(args.model_state_path, map_location='cuda:0' if args.gpu else 'cpu')
    mo.load_state_dict(checkpoint['state_dict'])
    logging.info('States loaded from %s', args.model_state_path)


    mo.eval()   # Set model to evaluate mode
    volatile = True

    buffer_ids = []
    buffer_outs = []

    # Iterate over data.
    for i, (inputs, ids) in enumerate(dataloader):
        if args.gpu:
            inputs = Variable(inputs.cuda(), volatile=volatile)
        else:
            inputs = Variable(inputs)

        # forward
        outputs = ft.compact(inputs, mo)

        buffer_ids.append(ids)
        buffer_outs.append(outputs)

        if (i + 1) % args.log_freq == 0:
            logging.info('  %d/%d', i + 1, len(dataloader))

    merged_ids = np.array([idx for tup in buffer_ids for idx in tup])
    merged_outs = torch.cat(buffer_outs, dim=0).data.cpu().numpy()
    np.savez(args.out_fname, id=merged_ids, descriptor=merged_outs)
    logging.info('outputed to %s', args.out_fname)
