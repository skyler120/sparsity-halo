from __future__ import print_function

import argparse
import os
import random
import shutil
import time
import numpy as np
import math
from tqdm import tqdm
import ipdb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
from torchsummary import summary

import models.cifar as models
from cifar_dataset import CIFAR10WithIdx, CIFAR100WithIdx

import sys
from utils import AverageMeter, accuracy, mkdir_p, savefig

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-v', '--validation', default=False, const=True, action='store_const', help='use a validation set')
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--rand_frac', default=0.0, type=float, help='rand perm label for gen experiments')


# Optimization options
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=50, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')


# Regularization 
parser.add_argument('--double_train', default=False, const=True, action='store_const', help='train reg params and model params separately')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--ws', default=False, const=True, action='store_const', help='use ws regularization')
parser.add_argument('--halo', default=False, const=True, action='store_const', help='use halo regularization')
parser.add_argument('--l1', default=False, const=True, action='store_const', help='use l1 regularization')
parser.add_argument('--layer', default=False, const=True, action='store_const', help='use halo layer regularization (defunct)')
parser.add_argument('--psi', default=0, type=float, help='penalization coefficient on regularization coefficients')
parser.add_argument('--lambda_init', default=1, type=float, help='initialization for regularization coefficients')
parser.add_argument('--xi', default=0, type=float, help='scaling to make loss functions on the same scale')
parser.add_argument('--reg_scheduler', default=-1, type=int, help='number of epochs to drop reg params')
parser.add_argument('--harch', default='L1', help='whether to use L1 or L2 penalty on reg params')
parser.add_argument('--params', '--p', default=0, type=int, help='number of parameters for penalization')
parser.add_argument('--thresh', default=None, help='threshold file for checking sparsity over time')

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

# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Miscs
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--save_dir', default='results/', type=str)
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')


# ResNet Basic Block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


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
        # transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        if args.rand_frac == 0.0:
            train_dataloader = datasets.CIFAR10
        else:
            train_dataloader = CIFAR10WithIdx

        
        test_dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        if args.rand_frac == 0.0:
            train_dataloader = datasets.CIFAR100
        else:
            train_dataloader = CIFAR100WithIdx

        test_dataloader = datasets.CIFAR100
        num_classes = 100

    if args.rand_frac == 0.0:
        trainset = train_dataloader(root='./data', train=True, download=True,
            transform=transform_train)
    else:
        trainset = train_dataloader(root='/tmp/data', train=True, download=True,
            transform=transform_train, rand_fraction=args.rand_frac)

    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    if not args.validation:
        testset = test_dataloader(root='./data', train=False, download=False, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    else:
        trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000])
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        testloader = data.DataLoader(valset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers)

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
    

def train():
    '''
    Arguments: 
        None
    Returns:
        losses.avg (float): loss over the test set
        top1.avg (float): classification accuracy on test set
    '''
    global global_step
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, batch in enumerate(trainloader):
        if args.rand_frac == 0.0:
            inputs, targets = batch
        else:
            inputs, targets, idx = batch
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # update loss with regularization term
        yes_reg = args.halo or args.l1 or args.ws

        reg_loss = Variable(torch.cuda.FloatTensor([0]), requires_grad=yes_reg)
        reg_reg_loss = Variable(torch.cuda.FloatTensor([0]), requires_grad=args.halo)
        layer_reg_loss = Variable(torch.cuda.FloatTensor([0]), requires_grad=args.layer)        
        if args.halo or args.l1 or args.ws:
            param_idx = 0
            for i, m in enumerate(model.modules()):
                if isinstance(m, nn.Conv2d): # or isinstance(m, nn.Linear):
                    for param in m.parameters():
                        num_layer_params = param.numel()
                        if args.halo or args.l1:
                                reg_loss = reg_loss + torch.sum(1. / reg_params[param_idx:(param_idx + num_layer_params)]**2. * torch.abs(param).view(-1))
                        elif args.ws:
                            reg_loss = reg_loss + 1. / reg_params**2 * param.norm(1)
                        param_idx += num_layer_params

                    if args.layer:
                        layer_reg_loss = layer_reg_loss + 1. / layer_params[i]**2  * param.norm(1)
                    else:
                        layer_reg_loss = 0.

            if args.ws:
                reg_reg_loss = reg_params**2.
            elif args.harch == 'L1':
                reg_reg_loss = torch.abs(torch.sum(torch.abs(reg_params)) - args.params)
            else:
                reg_reg_loss = (torch.sum(torch.abs(reg_params)) - args.params)**2
            if args.layer:
                layer_reg_reg_loss = torch.sum(torch.abs(layer_params))
            else:
                layer_reg_reg_loss = 0.

        # full loss
        if args.halo or args.l1 or args.ws:
            loss = loss + args.xi * (reg_loss + layer_reg_loss)  +  args.psi * (reg_reg_loss + layer_reg_reg_loss)

                
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.halo or args.ws:
            reg_optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if args.halo or args.ws:
            reg_optimizer.step()


        global_step += 1
        if math.isnan(loss.item()) or not np.isfinite(loss.item()):
            ipdb.set_trace()

    return (losses.avg, top1.avg)

def test():
    '''
    Arguments: 
        None
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

        # write progress
        writer.add_scalar('test/loss', losses.avg, global_step)
        writer.add_scalar('test/top1_acc', top1.avg, global_step)
        writer.add_scalar('test/top5_acc', top5.avg, global_step)
        writer.add_scalar('test/mean_reg', global_step, global_step)
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    '''
    Arguments: 
        state (dictionary): model information for saving
        is_best (boolean): whether accuracy is best
        checkpoint (string): save directory
        filename (string): name to save model info (default: checkpoint.pth.tar)
    Returns:
        None
    '''
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_reg_params(epoch):
    '''
    Arguments: 
        epoch (int): epoch number for adjusting reg params
    Returns:
        None
    '''
    if args.reg_scheduler == -1:
        return
    elif epoch % args.reg_scheduler == 0:
        args.psi *= 10./args.gamma
        return


def adjust_learning_rate(optimizer, epoch):
    '''
    Arguments: 
        optimizer (torch.optim.optimizer): optimizer for which to adjust learning rate
        epoch (int): epoch number for adjusting reg params
    Returns:
        None
    '''
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    args = parser.parse_args()

    state = {k: v for k, v in args._get_kwargs()}
  
    # Validate dataset
    assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

    use_cuda = torch.cuda.is_available()

    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    best_acc = 0  # best test accuracy

    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    os.makedirs(args.save_dir, exist_ok=True)


    trainloader, testloader, num_classes = make_dataset()
    print('Number of training samples: %d, test samples: %d'%(len(trainloader), len(testloader)))

    model = init_model()


    save_checkpoint({'state_dict': model.state_dict()}, False, checkpoint=args.save_dir, filename='init.pth.tar')

    print('Before model checkpoint load')
    zero_weights = 0
    for param in model.parameters():
        zero_weights += torch.sum(torch.abs(param) <= 1e-5)
    print(zero_weights)

    if args.resume is not '':
        # Load checkpoint.
        print('==> Getting reference model from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        # args.save_dir = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = args.start_epoch
        model.load_state_dict(checkpoint['state_dict'])

    print('After model checkpoint load')
    zero_weights = 0
    for param in model.parameters():
        zero_weights += torch.sum(torch.abs(param) <= 1e-5)
    print(zero_weights)

    num_model_params = 0
    num_layers = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d): #  or isinstance(m, nn.Linear):
            num_layers += 1
            for param in m.parameters():
                # num_model_params += m.weight.data.numel()
                num_model_params += param.numel()
    print('Total Network params: %.2fM' % (num_model_params/ 1000000.0))
    print('Number of layers: %d' % (num_layers))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # halo regularization parameters
    if args.ws:
        reg_params = torch.tensor(np.ones(1) * args.lambda_init,
            dtype=torch.float32, requires_grad=args.ws, device='cuda')

        reg_optimizer = optim.SGD([reg_params], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
    elif args.halo or args.l1:
        # args.lambda_init = np.exp(1)
        reg_params = torch.tensor(np.ones(num_model_params) * args.lambda_init,
                                        dtype=torch.float32, requires_grad=args.halo, device='cuda')

        reg_optimizer = optim.SGD([reg_params], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.layer:
        layer_params = torch.tensor(np.ones(num_layers) * args.lambda_init,
                        dtype=torch.float32, requires_grad=args.halo, device='cuda')

        layer_optimizer = optim.SGD([layer_params], lr=args.lr, 
            momentum=args.momentum, weight_decay=args.weight_decay)

    title = 'runs/exp-{}-{}-{}-{}-{}-{}'.format(args.dataset, args.arch, args.halo, args.psi, args.lambda_init, args.xi)
    writer = SummaryWriter(comment=title)
    global_step = 0

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test()
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))

        save_checkpoint({'state_dict': model.state_dict()}, False, checkpoint=args.save_dir, filename='init.pth.tar')

    else:
        # Train and val
        train_accs = []
        train_losses = []
        test_accs = []
        test_losses = []
        num_zero_weights5 = []
        num_zero_weights4 = []
        num_zero_weights3 = []
        num_zero_weights2 = []
        num_zero_weights = []
        for epoch in tqdm(range(start_epoch, args.epochs), total=args.epochs-start_epoch, desc="network training epochs"):
            adjust_learning_rate(optimizer, epoch)
            if args.halo or args.ws:
                adjust_learning_rate(reg_optimizer, epoch)
                adjust_reg_params(epoch)
            if args.halo or args.l1 or args.ws:
                tqdm.write('First reg param vals:  %.8f, %.8f, %.8f'%(reg_params[0].item(), args.psi, args.xi))

            tqdm.write('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

            if not args.double_train:
                start_time = time.time()
                train_loss, train_acc = train()
                end_time = time.time()
                print('run time for train {}'.format(end_time - start_time))	
                
                zero_weights5 = 0
                zero_weights4 = 0
                zero_weights3 = 0
                zero_weights2 = 0
                zero_weights = 0
                for param in model.parameters():
                    if args.thresh == None:
                        zero_weights5 += torch.sum(torch.abs(param) <= 1e-5)
                        zero_weights4 += torch.sum(torch.abs(param) <= 1e-4)
                        zero_weights3 += torch.sum(torch.abs(param) <= 1e-3)
                        zero_weights2 += torch.sum(torch.abs(param) <= 1e-2)
                    else:
                        thre = np.load(args.thresh)[0]
                        zero_weights += torch.sum(torch.abs(param) <= thre)

                if args.thresh == None:
                    num_zero_weights5.append(zero_weights5.cpu().item())
                    num_zero_weights4.append(zero_weights4.cpu().item())
                    num_zero_weights3.append(zero_weights3.cpu().item())
                    num_zero_weights2.append(zero_weights2.cpu().item())
                else:
                    tqdm.write('Number of zero weights: %d'%zero_weights.item())
                    num_zero_weights.append(zero_weights.cpu().item())

            else:
                # first freeze reg params and train only model params
                for param in model.parameters():
                    param.requires_grad = True
                reg_params.requires_grad = False

                start_time = time.time()
                train_loss, train_acc = train()
                end_time = time.time()
                print('run time for train {}'.format(end_time - start_time))	

                tqdm.write(' Train Loss:  %.8f, Train Acc:  %.2f' % (train_loss, train_acc))

                for param in model.parameters():
                    param.requires_grad = False
                reg_params.requires_grad = True

                start_time = time.time()
                train_loss, train_acc = train()
                end_time = time.time()
                print('run time for train {}'.format(end_time - start_time))

            tqdm.write(' Train Loss:  %.8f, Train Acc:  %.2f' % (train_loss, train_acc))

            test_loss, test_acc = test()
            tqdm.write(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))

            # save model
            # add save acc
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            test_accs.append(test_acc)
            test_losses.append(test_loss)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint=args.save_dir)


        state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }
        filepath = os.path.join(args.save_dir, 'final.pth.tar')
        torch.save(state, filepath)

        # write loss outputs to a file
        train_accs = np.asarray(train_accs)
        train_losses = np.asarray(train_losses)
        test_accs = np.asarray(test_accs)
        test_losses = np.asarray(test_losses)

        np.save(os.path.join(args.save_dir, 'train_accs'), train_accs)
        np.save(os.path.join(args.save_dir, 'train_losses'), train_losses)
        np.save(os.path.join(args.save_dir, 'test_accs'), test_accs)
        np.save(os.path.join(args.save_dir, 'test_losses'), test_losses)
        if args.thresh == None:
            np.save(os.path.join(args.save_dir, 'num_zeros5'), num_zero_weights5)
            np.save(os.path.join(args.save_dir, 'num_zeros4'), num_zero_weights4)
            np.save(os.path.join(args.save_dir, 'num_zeros3'), num_zero_weights3)
            np.save(os.path.join(args.save_dir, 'num_zeros2'), num_zero_weights2)
        else:
            np.save(os.path.join(args.save_dir, 'num_zeros'), num_zero_weights)

        # save reg params
        if args.halo and args.rand_frac == 0.0:
            np.save(os.path.join(args.save_dir, 'reg_params'), reg_params.detach().cpu().numpy())

            conv_params = []
            for m in model.modules():
                if isinstance(m, nn.Conv2d): #  or isinstance(m, nn.Linear):
                    for param in m.parameters():
                        conv_params.append(param.data.view(-1).detach().cpu().numpy())

            conv_params = np.concatenate(conv_params)
            np.save(os.path.join(args.save_dir, 'conv_params'), conv_params)

        print('Best acc:')
        print(best_acc)
