from __future__ import print_function

import argparse
import math
import torch
import torch.backends.cudnn as cudnn

from util import adjust_learning_rate, accuracy
from util import set_optimizer

from util import set_dataset, get_batch
from models import SupConResNet, LinearClassifier, Mlp


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--dataset', type=str, default='NTU60CV',
                        choices=['NTU60CV', 'NTU60CS', 'NTU120CSet', 'NTU120CSub'],
                        help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for iteration in iterations:
        opt.lr_decay_epochs.append(int(iteration))

    # warm-up for large-batch training,
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
        classifier = LinearClassifier(num_classes=opt.num_classes)
    else:
        model = SupConResNet(name=opt.model)
        classifier = LinearClassifier(dim_in=2048, num_classes=opt.num_classes)

    criterion = torch.nn.CrossEntropyLoss()

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(model, classifier, criterion, optimizer, x_train, y_train, opt):
    model.eval()
    classifier.train()

    x_batch, y_batch = get_batch(opt, x_train, y_train)

    if opt.model == 'mlp':
        x_batch = x_batch.reshape(opt.batch_size, -1)
    else:
        x_batch = x_batch.reshape(opt.batch_size, 300, 50, 3)
    x_batch = torch.from_numpy(x_batch).float()

    if torch.cuda.is_available():
        x_batch = x_batch.cuda(non_blocking=True)
        y_batch = y_batch.cuda(non_blocking=True)

    with torch.no_grad():
        if opt.model == 'mlp':
            features = model(x_batch, is_train=False)
        else:
            features = model.encoder(x_batch)
    output = classifier(features.detach())
    loss = criterion(output, y_batch)
    acc = accuracy(output, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), acc


def validate(model, classifier, criterion, x_test, y_test, opt):
    model.eval()
    classifier.eval()

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
        if opt.model == 'mlp':
            output = classifier(model(x_batch, is_train=False))
        else:
            output = classifier(model.encoder(x_batch))

        loss = criterion(output, y_batch)
        acc = accuracy(output, y_batch)

    return loss.item(), acc


def main():
    best_acc = 0.0
    opt = parse_option()

    model, classifier, criterion = set_model(opt)
    optimizer = set_optimizer(opt, classifier)
    x_train, y_train, x_test, y_test = set_dataset(opt)

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        train_loss, train_acc = train(model, classifier, criterion, optimizer, x_train, y_train, opt)
        val_loss, val_acc = validate(model, classifier, criterion, x_test, y_test, opt)

        if val_acc > best_acc:
            best_acc = val_acc

        print('Epoch {}, Train loss {}, Train acc {}, Val loss {}, Val acc {}'
              .format(epoch, train_loss, train_acc, val_loss, val_acc))

    print('Best accuracy: {}'.format(best_acc))


if __name__ == '__main__':
    main()
