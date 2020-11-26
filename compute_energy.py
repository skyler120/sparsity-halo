from __future__ import print_function

import argparse
import os
import random
import shutil
import time
import ipdb
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from thop import profile, clever_format

import models.cifar as models

import sys
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


# Create dataset
def make_dataset(): 
    '''
    Arguments: None
    Returns:
        trainloader (pytorch dataloader): pytorch loader for train set
        testloader (pytorch dataloader): pytorch loader for test set
        num_classes (int); number of classes for supervised image classification
    '''
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)

    return trainloader, testloader, num_classes

# Model
def init_model():
    '''
    Arguments: None
    Returns:
        model (pytorch NN): torch neural network module
    '''
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )

    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model.cuda()
    return model


def test(dataloader):
    '''
    Arguments: 
        dataloader (torch.utils.data.dataloader): pytorch dataloader for evaluating
    Returns:
        losses.avg (float): loss over the test set
        top1.avg (float): classification accuracy on test set
    '''
    global best_acc, global_step
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(dataloader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

    return (losses.avg, top1.avg)


def hook_fn(m, i, o):
    '''
    Helper function for getting embeddings from intermediate layers of network
    Arguments: 
        m (nn.module): layer of the network
        i (torch.Tensor): input to layer
        o (torch.Tensor): output of layer
    Returns:
        None
    '''
    if isinstance(m, nn.Conv2d):
        maxpool = nn.MaxPool2d(1, o.size()[2:])
        int_out[m].append(maxpool(o).view(o.size()[0], -1).detach().cpu().numpy())


def get_all_layers(net):
    '''
    Function for extracting embeddings for each layer
    Arguments: 
        net (nn.module): network to extract embeddings from
    Returns:
        None
    '''
    global int_out
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(hook_fn)
            int_out[m] = []


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')

parser.add_argument('--seed', default=1, help='random seed')
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
# Checkpoints
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save', default='', type=str, help='path to save eigs')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=1, help='Compression Rate (theta) for DenseNet.')

# threshold
parser.add_argument('--tau', default=1e-5, type=float, help='threshold to check weights')

args = parser.parse_args()


if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()

    # Random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    # get dataset
    trainloader, testloader, num_classes = make_dataset()

    # get model
    model = init_model()
    model = model.eval()

    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))


    # load checkpoint params so model is now pruned or trained variant
    assert args.checkpoint, 'Error: no checkpoint directory specified'
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.checkpoint), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.checkpoint)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    print(checkpoint['epoch'])


    # register hooks to get intermediate representation
    int_out = {}
    get_all_layers(model)

    # get intermediate outputs with evaluation on training set
    criterion = nn.CrossEntropyLoss()
    train_loss, train_acc = test(trainloader)
    print('Sparse Network Train Loss:  %.8f, Train Acc:  %.2f' % (train_loss, train_acc))

    # compute eigenvals
    eigs_by_layer = []
    for k, v in int_out.items():
        acts = np.concatenate(v)
        acts = np.transpose(acts)

        cov_acts = np.cov(acts)
        w, _ = np.linalg.eig(cov_acts)
        w = np.abs(w) / np.sum(np.abs(w))
        w[::-1].sort()

        eigs_by_layer.append(w)

    with open(args.save, 'wb') as handle:
        pickle.dump(eigs_by_layer, handle, protocol=pickle.HIGHEST_PROTOCOL)

