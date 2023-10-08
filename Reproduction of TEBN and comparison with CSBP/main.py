import argparse
import os
import torch.optim as optim
from models import *
import dataloader
import torch
import torchvision.transforms as transforms
import math
import torch.nn as nn
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser(description='TEBN')
parser.add_argument('-w', '--workers', default=10, type=int, metavar='N', help='number of workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of training epochs')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch number for resume models')
parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='number of batch size')
parser.add_argument('--seed', default=1000, type=int, help='seed')
parser.add_argument('-T', '--time', default=10, type=int, metavar='N', help='inference time-step')
parser.add_argument('-out_dir', default='./logs/', type=str, help='log dir')
parser.add_argument('-resume', type=str, help='resume from checkpoint')
parser.add_argument('-method', default='TEBN', type=str, help='BN method')
parser.add_argument('-tau', type=float, default=0.25, help='tau value of LIF neuron')
parser.add_argument('-csbp', action='store_true', help='use CSBP')
parser.add_argument('-z', type=float)
parser.add_argument('-beta', type=float)
parser.add_argument('-layer_nums', type=int, help='layer number of chaotic dynamics introduced')
parser.add_argument('-batch_anneal', type=bool)

args = parser.parse_args()
print(args)

def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    split train set and test set code is from "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)

def chaos_loss_fun(h, z, I0=0.65):
    out = F.sigmoid(h/10) 
    log1, log2 = torch.log(out), torch.log(1 - out)
    return -z * (I0 * log1 + (1 - I0) * log2)

def chaos(hid, z, chaos_loss_f, I0=0.65):
    chaosloss = 0
    if type(hid) == torch.Tensor:
        hid = torch.where(hid<9.9, hid.float(), torch.tensor(9.9).to(hid.device))
        hid = torch.where(hid>-10, hid, torch.tensor(-10.0).to(hid.device))
        chaosloss = chaos_loss_f(hid, z).sum() / hid.numel()
    else: 
        for i in range(len(hid) - 1):
            hid_thresold = torch.where(hid[i+1]<9.9, hid[i+1], torch.tensor(9.9).to(hid[i+1].device))
            hid_thresold = torch.where(hid_thresold>-10.0, hid_thresold, torch.tensor(-10.0).to(hid[i+1].device))
            chaosloss += chaos_loss_f(hid_thresold, z).sum() / hid_thresold.numel()
    
    return chaosloss

def train(model, device, train_loader, criterion, optimizer, epoch, args):
    if args.csbp:
        global z, beta
    running_loss = 0
    model.train()
    total = 0
    correct = 0
    myTransforms = transforms.Compose([transforms.RandomCrop(48, padding=4),
        transforms.RandomHorizontalFlip()])
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        outputs = model(myTransforms(images))
        if args.csbp:
            mean_out = outputs[0].mean(1)
            chaos_loss = chaos(outputs[:args.layer_nums+1], z, chaos_loss_fun)
            loss = criterion(mean_out, labels)
            total_loss = loss + chaos_loss
        else:
            mean_out = outputs.mean(1)
            loss = criterion(mean_out, labels)
            total_loss = loss
        running_loss += loss.item()
        total_loss.backward()
        optimizer.step()
        if args.batch_anneal and args.csbp:
            z *= beta
        total += float(labels.size(0))
        _, predicted = mean_out.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    if args.csbp:
        return running_loss, 100 * correct / total, chaos_loss
    else:
        return running_loss, 100 * correct / total


@torch.no_grad()
def test(model, test_loader, device, args):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        if args.csbp:
            outputs = model(inputs)[0]
        else:
            outputs = model(inputs)
        mean_out = outputs.mean(1)
        _, predicted = mean_out.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    final_acc = 100 * correct / total
    return final_acc


if __name__ == '__main__':
    # set manual seed
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # processing DVS-CIFAR10
    dts_cache = './dts_cache'
    
    train_set_pth = os.path.join(dts_cache, 'train_set.pt')
    test_set_pth = os.path.join(dts_cache, 'test_set.pt')

    if os.path.exists(train_set_pth) and os.path.exists(test_set_pth):
        train_set = torch.load(train_set_pth)
        test_set = torch.load(test_set_pth)
    else:
        origin_set = dataloader.DVSCifar10()
        train_set, test_set = split_to_train_test_set(0.9, origin_set, 10)
        if not os.path.exists(dts_cache):
            os.makedirs(dts_cache)
        torch.save(train_set, train_set_pth)
        torch.save(test_set, test_set_pth)    
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.csbp:
        model = CNN7_chaos(tau=args.tau)
    else:
        model = CNN7(tau=args.tau)

    model = torch.nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    start_epoch = 0

    out_dir = os.path.join(args.out_dir, f'method_{args.method}_tau_{args.tau}_T_{args.time}')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']

    best_acc = 0
    best_epoch = 0

    writer = SummaryWriter(os.path.join(out_dir, 'logs'), purge_step=start_epoch)
    
    if args.csbp:
        z = args.z
        beta = args.beta

    for epoch in range(start_epoch, args.epochs):
        if args.csbp:
            loss, acc, chaosloss = train(model, device, train_loader, criterion, optimizer, epoch, args)
            if not args.batch_anneal:
                z *= beta
        else:
            loss, acc = train(model, device, train_loader, criterion, optimizer, epoch, args)
        print('Epoch {}/{} train loss={:.5f} train acc={:.3f}'.format(epoch, args.epochs, loss, acc))
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('train_acc', acc, epoch)
        test_acc = test(model, test_loader, device, args)
        print('Epoch {}/{} test acc={:.3f}'.format(epoch, args.epochs, test_acc))
        writer.add_scalar('test_acc', test_acc, epoch)
        scheduler.step()

        save_max = False
        if best_acc < test_acc:
            best_acc = test_acc
            save_max = True
            best_epoch = epoch + 1
        print('Best Test acc={:.3f}'.format(best_acc))

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))
        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

