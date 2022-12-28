from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import random


def get_batch(args, x, y):
    rand_idx = torch.tensor(np.random.choice(np.arange(len(y[1])), args.batch_size)).type(torch.long)
    x_batch = x[rand_idx]
    y_tmp = y[rand_idx]
    y_batch = torch.zeros(y_tmp.shape[0], dtype=torch.long)
    for i in range(y_tmp.shape[0]):
        for j in range(y_tmp.shape[1]):
            if y_tmp[i][j] == 1:
                y_batch[i] = j

    return x_batch, y_batch


def shear(features):
    s_xy = random.uniform(-1, 1)
    s_xz = random.uniform(-1, 1)
    s_yx = random.uniform(-1, 1)
    s_yz = random.uniform(-1, 1)
    s_zx = random.uniform(-1, 1)
    s_zy = random.uniform(-1, 1)

    trans_matrix = np.array([[1, s_xy, s_xz],
                             [s_yx, 1, s_yz],
                             [s_zx, s_zy, 1]])

    features_sheared = np.dot(
        features.reshape(features.shape[0], features.shape[1], 50, 3), trans_matrix)

    return torch.as_tensor(features_sheared)


def reverse(features, p=0.5):
    features_reversed = features.reshape(features.shape[0], features.shape[1], 50, 3)
    if random.random() < p:
        _, frames, _, _ = features_reversed.shape
        time_range_order = [i for i in range(frames)]
        time_range_reverse = list(reversed(time_range_order))
        features_reversed = features_reversed[:, time_range_reverse, :, :]

    return torch.as_tensor(features_reversed)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct.item() / len(labels)


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_dataset(opt):
    if opt.dataset == 'NTU60CV':
        data_path = './dataset/ntu60/NTU60_CV.npz'
    elif opt.dataset == 'NTU60CS':
        data_path = './dataset/ntu60/NTU60_CS.npz'
    elif opt.dataset == 'NTU120CSet':
        data_path = './dataset/ntu120/NTU120_CSet.npz'
    elif opt.dataset == 'NTU120CSub':
        data_path = './dataset/ntu120/NTU120_CSub.npz'

    with np.load(data_path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return x_train, y_train, x_test, y_test


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Model is saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
