from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import os
import argparse
import logging
import glob
import datetime

import torchvision
from torchvision import transforms, datasets
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils import data

from datasets import ZaloLandmarkDataset

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--nb_epochs', default=20, type=int)
    parser.add_argument('--nb_workers', default=2, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--base_model', choices=['resnet18', 'resnet50', 'resnet101'])
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--train_filelist', required=True)
    parser.add_argument('--test_filelist', required=True)
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--model_state_path')
    parser.add_argument('--checkpoint_suffix', required=True)
    parser.add_argument('--checkpoint_dir', default='./models')
    parser.add_argument('--checkpoint_freq', default=2, type=int)

    args = parser.parse_args()
    logging.info('Arguments: %s', args)

    now = datetime.datetime.now()

    MEAN = [0.485, 0.456, 0.406] # imagenet -- 0-1 range
    STD = [0.229, 0.224, 0.225] # imagenet -- 0-1 range
    SCALE = 256
    INPUT_SHAPE = 224

    dftr = pd.read_csv(args.train_filelist)
    dfte = pd.read_csv(args.train_filelist)
    nb_classes = max(dftr['category'].max(), dfte['category'].max()) + 1

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(SCALE),
            transforms.RandomResizedCrop(INPUT_SHAPE),
            transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
        'val': transforms.Compose([
            transforms.Resize(SCALE),
            transforms.CenterCrop(INPUT_SHAPE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
    }

    datasets = {
        'train': ZaloLandmarkDataset(args.img_dir, dftr, transform=data_transforms['train']),
        'val': ZaloLandmarkDataset(args.img_dir, dfte, transform=data_transforms['val']),
    }
    dataloaders = {x: data.DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.nb_workers)
            for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    if args.base_model == 'resnet18':
        mo = torchvision.models.resnet18(pretrained=True)
    elif args.base_model == 'resnet50':
        mo = torchvision.models.resnet50(pretrained=True)
    elif args.base_model == 'resnet101':
        mo = torchvision.models.resnet101(pretrained=True)

    num_ftrs = mo.fc.in_features
    mo.fc = nn.Linear(num_ftrs, nb_classes)

    if args.gpu:
        mo.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, mo.parameters())), lr=0.001, momentum=0.9)

    if args.model_state_path is not None and os.path.isfile(args.model_state_path):
        checkpoint = torch.load(args.model_state_path, map_location='cuda:0' if args.gpu else 'cpu')
        epoch_start = checkpoint['epoch']
        mo.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info('States loaded from %s', args.model_state_path)
    else:
        epoch_start = 0

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=3)

    for epoch in range(epoch_start, args.nb_epochs):
        logging.info('Epoch %d/%d', epoch + 1, args.nb_epochs)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                mo.train()  # Set model to training mode
            else:
                mo.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            batch_count = 0

            # Iterate over data.
            for i, (inputs, _, labels) in enumerate(dataloaders[phase]):
                if args.gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = mo(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                batch_count += 1

                if phase == 'train' and (i + 1) % args.log_freq == 0:
                    logging.info('  %d/%d loss: %.6f, acc: %.6f', i + 1, len(dataloaders[phase]),
                            loss.data[0], nb_corrects / inputs.size(0))

            epoch_loss = running_loss / batch_count
            epoch_acc = running_corrects / dataset_sizes[phase]

            logging.info('%s loss: %.6f, acc: %.6f', phase, epoch_loss, epoch_acc)
            if phase == 'val':
                scheduler.step(epoch_loss)

        if (epoch + 1) % args.checkpoint_freq == 0:
            out_fname = os.path.join(args.checkpoint_dir,
                    '{}_model_{}_{:03d}.pth'.format(now.strftime('%Y%m%d_%H%M%S'), args.checkpoint_suffix, epoch + 1))
            torch.save({
                'epoch': epoch + 1,
                'state_dict': mo.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, out_fname)
            logging.info('Checkpoint to %s', out_fname)

    logging.info('done')
