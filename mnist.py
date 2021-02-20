from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

# general config for training and loading
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--batch', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 250)')
parser.add_argument('--epochs', type=int, default=250, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--stepsize', '-ss', default=25000, type=int, 
                    help='step size lr decay rate')
parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                    help='Learning rate step gamma (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--train', default=True, const=True, action='store_const',
                    help='Train model or only perform evaluation') 

# model config
parser.add_argument('--arch', '-a', default='LeNet31',
                    help='Model default: LeNet 300x100') 

# regularization configs
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W',
                    help='weight decay (default 0)')
parser.add_argument('--l1', default=False, const=True, action='store_const',
                    help='whether to use l1 regularization')
parser.add_argument('--ws', default=False, const=True, action='store_const',
                    help='whether to use WS regularization')
parser.add_argument('--slope', default=False, const=True, action='store_const',
                    help='whether to use slope regularization')
parser.add_argument('--layer', default=False, const=True, action='store_const',
                    help='whether to use slope layer regularization')
parser.add_argument('--xi', default=0., type=float,
                    help='regularization coefficient')
parser.add_argument('--psi', default=0., type=float,
                    help='regularization coefficient')
parser.add_argument('--lambda_init', default=1, type=float, 
                    help='initialization for regularization coefficients')

# other configs
parser.add_argument('--no_cuda', default=False, const=True, action='store_const',
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save_model',default=None,
                    help='File to save model if train is True or load model if train is False')
parser.add_argument('--tau',  default=1e-5, type=float,
                    help='Sparsity check (default: 1e-5)')


args = parser.parse_args()
train_global_step = 0
test_global_step = 0
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
        return self.fc3(x)
        # return F.log_softmax(self.fc3(x), dim=1)



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
        x = self.fc5(x)
        # x = F.log_softmax(self.fc5(x))
        return x


def train():
    '''
    Arguments: 
        None
    Returns:
        None
    '''
    global train_global_step
    model.train()
    train_loss, correct = (0, 0)
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        scheduler.step()
        if args.slope or args.ws:
            reg_scheduler.step()

        # load data
        data, target = data.to(device), target.to(device)

        train_global_step += 1
        output = model(data)

        # l1 regularization
        if args.l1:
            l1_reg = Variable(torch.cuda.FloatTensor([0]), requires_grad=True)
            for W in model.parameters():
                l1_reg = l1_reg + W.norm(1)

            loss = criterion(output, target) + args.xi * l1_reg

        elif args.ws:
            reg_loss = Variable(torch.cuda.FloatTensor([0]), requires_grad=True)
            reg_reg_loss = Variable(torch.cuda.FloatTensor([0]), requires_grad=True)
            for W in model.parameters():
                reg_loss = reg_loss + W.norm(1)

            reg_loss = args.xi * 1 / reg_params**2. * reg_loss
            reg_reg_loss = args.psi * reg_params**2.
            loss = criterion(output, target) + reg_loss +  reg_reg_loss 

        # slope regularization
        elif args.slope:
            param_idx = 0
            reg_loss = Variable(torch.cuda.FloatTensor([0]), requires_grad=True)
            reg_reg_loss = Variable(torch.cuda.FloatTensor([0]), requires_grad=True)
            layer_reg_loss = Variable(torch.cuda.FloatTensor([0]), requires_grad=True)
            counter = 0
            for i, m in enumerate(model.modules()):
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    for param in m.parameters():
                        num_layer_params = param.numel()
                        reg_loss = reg_loss + torch.sum(1. / reg_params[param_idx:(param_idx + num_layer_params)]**2. * torch.abs(param).view(-1))
                        param_idx += num_layer_params

                    if args.layer:
                        layer_reg_loss = layer_reg_loss + 1. / layer_params[counter]**2  * param.norm(1)
                        counter += 1
                    else:
                        layer_reg_loss = 0.

            reg_reg_loss = torch.sum(torch.abs(reg_params))
            if args.layer:
                layer_reg_reg_loss = torch.sum(torch.abs(layer_params))
            else:
                layer_reg_reg_loss = 0.

            loss = criterion(output, target) + args.xi * (reg_loss + layer_reg_loss)  +  args.psi * (reg_reg_loss + layer_reg_reg_loss)

        # no regularization
        else:
            loss = criterion(output, target)


        
        train_loss += loss.item()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        corr = pred.eq(target.view_as(pred)).sum().item()
        correct += corr

        optimizer.zero_grad()
        if args.slope or args.ws:
            reg_optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        if args.slope or args.ws:
            reg_optimizer.step()


        writer.add_scalar('train/loss', loss, train_global_step)
        writer.add_scalar('train/acc', corr / args.batch, train_global_step)

    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss/len(train_loader), correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


def test():
    '''
    Arguments: 
        None
    Returns:
        None
    '''
    global test_global_step
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, total=len(test_loader)):
            test_global_step += 1
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target).item()
            test_loss += loss  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            corr = pred.eq(target.view_as(pred)).sum().item()
            correct += corr

            writer.add_scalar('test/loss', loss, test_global_step)
            writer.add_scalar('test/acc', corr / args.batch, test_global_step)

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # MNIST Test dataset and dataloader declaration
    if args.train:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),])),batch_size=args.batch, shuffle=True, num_workers=args.workers)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),])), batch_size=args.batch, shuffle=False, num_workers=args.workers)

    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # Create the model
    if args.arch == 'LeNet':
        model = LeNet().to(device)
    elif args.arch == 'LeNet31':
        model = LeNet_300_100().to(device) 
    else:
        model = LeNet_5().to(device)

    torch.save(model.state_dict(), 'models2/'+args.arch+'_%d_init.pt'%args.seed)

    # Create optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
     weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.stepsize, gamma=args.gamma)


    num_model_params = 0
    num_layers = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            num_layers += 1
            for param in m.parameters():
                # num_model_params += m.weight.data.numel()
                num_model_params += param.numel()
    print('Total Network params: %.2fM' % (num_model_params/ 1000000.0))
    print('Number of layers: %d' % (num_layers))

    # slope regularization parameters
    if args.ws:
        reg_params = torch.tensor(np.ones(1) * args.lambda_init, 
            dtype=torch.float32, requires_grad=args.ws, device='cuda')

        reg_optimizer = optim.SGD([reg_params], lr=args.lr, momentum=args.momentum,
         weight_decay=args.weight_decay)

    if args.slope or args.l1:
        reg_params = torch.tensor(np.ones(num_model_params) * args.lambda_init,
                                        dtype=torch.float32, requires_grad=args.slope, device='cuda')

        reg_optimizer = optim.SGD([reg_params], lr=args.lr, momentum=args.momentum,
         weight_decay=args.weight_decay)
    if args.layer:
        layer_params = torch.tensor(np.ones(num_layers) * args.lambda_init,
                                        dtype=torch.float32, requires_grad=args.layer, device='cuda')

        layer_optimizer = optim.SGD([layer_params], lr=args.lr, momentum=args.momentum,
         weight_decay=args.weight_decay)

    if args.slope or args.ws:
        reg_scheduler = optim.lr_scheduler.StepLR(optimizer, args.stepsize, gamma=args.gamma)


    if args.train:
        writer = SummaryWriter('runs/{}-wd_{}-l1_{}_{}-slopeh_{}_{}_{}'.format(
            args.arch, args.weight_decay, args.l1, args.xi, args.slope, 
            args.psi, args.xi))

        for epoch in range(1, args.epochs + 1):
            print('Epoch # {}'.format(epoch))
            train()
            test()

        if args.save_model is not None:
            torch.save(model.state_dict(), args.save_model)

