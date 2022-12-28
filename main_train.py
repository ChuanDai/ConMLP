from __future__ import print_function

import os
import argparse
import math

import torch
import torch.backends.cudnn as cudnn

from util import adjust_learning_rate
from util import set_optimizer, save_model
from util import set_dataset, get_batch, shear, reverse

from models import SupConResNet, Mlp
from losses import SupConLoss


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--dataset', type=str, default='NTU60CV',
                        choices=['NTU60CV', 'NTU60CS', 'NTU120CSet', 'NTU120CSub'],
                        help='dataset')

    # method
    parser.add_argument('--method', type=str, default='Supervised',
                        choices=['Supervised', 'Self-supervised'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    opt = parser.parse_args()

    opt.model_path = './models/'
    opt.model_name = '{}/{}_{}'.format(opt.model, opt.method, opt.dataset)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for iteration in iterations:
        opt.lr_decay_epochs.append(int(iteration))

    return opt


def set_model(opt):

    if opt.model == 'mlp':
        model = Mlp()
    else:
        model = SupConResNet(name=opt.model)

    criterion = SupConLoss(temperature=opt.temp)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(model, criterion, optimizer, x_train, y_train, opt):
    model.train()

    x_batch, y_batch = get_batch(opt, x_train, y_train)

    x_sheared = shear(x_batch)
    x_reversed = reverse(x_batch)
    x_batch = torch.cat([x_sheared, x_reversed], dim=0).float()

    if opt.model == 'mlp':
        x_batch = torch.reshape(x_batch, (opt.batch_size * 2, -1)).float()

    if torch.cuda.is_available():
        x_batch = x_batch.cuda(non_blocking=True)
        y_batch = y_batch.cuda(non_blocking=True)

    features = model(x_batch)

    f1, f2 = torch.split(features, [opt.batch_size, opt.batch_size], dim=0)
    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

    if opt.method == 'Supervised':
        loss = criterion(features, y_batch)
    elif opt.method == 'Self-supervised':
        loss = criterion(features)

    # SGD
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def main():
    opt = parse_option()

    model, criterion = set_model(opt)

    optimizer = set_optimizer(opt, model)

    x_train, y_train, _, _ = set_dataset(opt)

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        loss = train(model, criterion, optimizer, x_train, y_train, opt)
        print('Epoch {}, Loss {}'.format(epoch, loss))

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{}.pth'.format(epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    save_file = os.path.join(
        opt.save_folder, 'ckpt_eps{}_lr{}_decay{}_bsz{}_tau{}_cosine{}_.pth'.
        format(opt.epochs, opt.learning_rate, opt.weight_decay, opt.batch_size, opt.temp, opt.cosine)
    )
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
