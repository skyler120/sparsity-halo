from __future__ import print_function

import argparse
import os
import random
import shutil
import time
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from thop import profile, clever_format

import sys

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
# LeNet Model definition
class LeNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class LeNet_300_100(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 784)))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)



class LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc3 = nn.Linear(16 * 5 * 5, 120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.fc3(x.view(-1, 16 * 5 * 5)))
        x = F.relu(self.fc4(x))
        x = F.log_softmax(self.fc5(x))

        return x

# Create dataset
def make_dataset(): 
    '''
    Arguments: None
    Returns:
        testloader (pytorch dataloader): pytorch loader for test set
        num_classes (int); number of classes for supervised image classification
    '''
    num_classes = 10
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),])), batch_size=100, shuffle=False, num_workers=8)

    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    return test_loader, num_classes


# Model
def init_model():
    '''
    Arguments: None
    Returns:
        model (pytorch NN): torch neural network module (default: LeNet-5 architecture)
    '''
    if args.arch == 'LeNet':
        model = LeNet().to(device)
    elif args.arch == 'LeNet31':
        model = LeNet_300_100().to(device) 
    else:
        model = LeNet_5().to(device)

    model.cuda()
    return model


def test():
    '''
    Arguments: None
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

    for batch_idx, (inputs, targets) in enumerate(testloader):

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

parser = argparse.ArgumentParser(description='PyTorch Sparsity Check')

parser.add_argument('--seed', default=1, help='random seed')

# Checkpoints
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='LeNet31', help='model architecture:')
# threshold
parser.add_argument('--tau', default=1e-5, type=float, help='threshold to check weights')

args = parser.parse_args()


if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    # get dataset
    testloader, num_classes = make_dataset()

    # get model
    model = init_model()
    model = model.eval()

    cudnn.benchmark = True
    # print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))


    assert args.checkpoint, 'Error: no checkpoint directory specified'
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.checkpoint), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)


    total_weights = sum(p.numel() for p in model.parameters())
    print(total_weights)
    zero_weights = 0
    for param in model.parameters():
        zero_weights += torch.sum(torch.abs(param) <= args.tau)

    print( (1. * zero_weights) / total_weights)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = test()
    print('Full Network Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))


    num_zero_params = 0
    zero_param_ratio = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            for param in m.parameters():
                num_zero_params += torch.sum(torch.abs(param) <= args.tau).item()
                zero_param_ratio.append(torch.sum(torch.abs(param) <= args.tau).item() / param.numel() )
    
    print("Total number of params: ", total_weights)

    print("Fraction of zero params: ", num_zero_params / total_weights)

    # Update model to have 0 entries.
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            weight_copy = m.weight.data.abs().clone()
            mask = torch.lt(weight_copy, args.tau)
            mask = 1 - 1. * mask 
            m.weight = torch.nn.Parameter(m.weight * mask)

    # Re-evaluate on the test set
    test_loss, test_acc = test()
    print('Sparse Network Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))


    num_true_zero_params = 0
    for m in model.modules():
        for param in m.parameters():
            num_true_zero_params += torch.sum(torch.abs(param) == 0)

    print("Fraction of true zero params: ", num_true_zero_params / total_weights)
    print(zero_param_ratio)
