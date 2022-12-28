from __future__ import print_function

import argparse
import math

import torch
import torch.backends.cudnn as cudnn

from util import adjust_learning_rate, accuracy
from util import set_dataset, set_optimizer
from models import SupCEResNet, Mlp

from util import get_batch


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='mlp',
                        help='model name')
    parser.add_argument('--dataset', type=str, default='NTU60CV',
                        choices=['NTU60CV', 'NTU60CS', 'NTU120CSet', 'NTU120CSub'],
                        help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

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

    if opt.dataset == 'NTU60CV' or opt.dataset == 'NTU60CS':
        opt.num_classes = 60
    elif opt.dataset == 'NTU120CSet' or opt.dataset == 'NTU120CSub':
        opt.num_classes = 120
    else:
        raise ValueError('Dataset {} is not supported.'.format(opt.dataset))

    return opt


def set_model(opt):
    if opt.model == 'mlp':
        model = Mlp()
    else:
        model = SupCEResNet(name=opt.model, num_classes=opt.num_classes)

    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(model, criterion, optimizer, x_train, y_train, opt):
    model.train()

    x_batch, y_batch = get_batch(opt, x_train, y_train)

    if opt.model == 'mlp':
        x_batch = x_batch.reshape(opt.batch_size, -1)
    else:
        x_batch = x_batch.reshape(opt.batch_size, 300, 50, 3)
    x_batch = torch.from_numpy(x_batch).float()

    if torch.cuda.is_available():
        x_batch = x_batch.cuda(non_blocking=True)
        y_batch = y_batch.cuda(non_blocking=True)

    output = model(x_batch)
    loss = criterion(output, y_batch)
    acc = accuracy(output, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), acc


def validate(model, criterion, x_test, y_test, opt):
    model.eval()

    x_batch, y_batch = get_batch(opt, x_test, y_test)

    if opt.model == 'mlp':
        x_batch = x_batch.reshape(opt.batch_size, -1)
    else:
        x_batch = x_batch.reshape(opt.batch_size, 300, 50, 3)
    x_batch = torch.from_numpy(x_batch).float()

    if torch.cuda.is_available():
        x_batch = x_batch.cuda(non_blocking=True)
        y_batch = y_batch.cuda(non_blocking=True)

    with torch.no_grad():
        output = model(x_batch)
        loss = criterion(output, y_batch)
        acc = accuracy(output, y_batch)

    return loss.item(), acc


def main():
    best_acc = 0.0
    opt = parse_option()

    model, criterion = set_model(opt)

    optimizer = set_optimizer(opt, model)

    x_train, y_train, x_test, y_test = set_dataset(opt)

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        train_loss, train_acc = train(model, criterion, optimizer, x_train, y_train, opt)
        print('Epoch {}, Train loss {}, Train accuracy {}'.format(epoch, train_loss, train_acc))

        val_loss, val_acc = validate(model, criterion, x_test, y_test, opt)
        print('Epoch {}, Test loss {}, Test accuracy {}'.format(epoch, val_loss, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc

    print('Best accuracy: {}'.format(best_acc))


if __name__ == '__main__':
    main()
