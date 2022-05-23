from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='none')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--origin', action='store_true', default=False,
                    help='Keep the original implementation as the paper.')
parser.add_argument('--repeat', type=int, default=50,
                    help='number of experiments')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='Show training process')
parser.add_argument('--split', type=str, default='equal',
                    help='Data split method')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj: sparse tensor, symmetric normalized Laplication
# A = D^(-1/2)*A*D^(1/2)
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset, origin=args.origin, split=args.split)

if args.cuda:
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(num_epoch, patience=25, verbose=False):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf")
    best_epoch = -1
    for epoch in range(num_epoch):
        t = time.time()
        optimizer.zero_grad()
        outputs = model(features, adj)
        # nll_loss: negative log likelihood loss
        loss_train = F.nll_loss(outputs[idx_train], labels[idx_train])
        acc_train = accuracy(outputs[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            outputs = model(features, adj)

        loss_val = F.nll_loss(outputs[idx_val], labels[idx_val])
        acc_val = accuracy(outputs[idx_val], labels[idx_val])
        if verbose:
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))

        # early stop
        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch
        if epoch == best_epoch + patience:
            break

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss: {:.4f}".format(loss_test.item()),
          "accuracy: {:.4f}".format(acc_test.item()))
    return acc_test.item()


if __name__ == "__main__":
    results = []
    for i in range(args.repeat):
        # model
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout,
                    origin=args.origin)
        if args.cuda:
            model.cuda()

        print("----- %d / %d runs -----" % (i+1, args.repeat))
        # Train model
        t_total = time.time()
        train(args.epochs, verbose=args.verbose)
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Test
        results.append(test())
    print("%d runs, mean: %g, var: %g" % (args.repeat, np.mean(results), np.std(results)))