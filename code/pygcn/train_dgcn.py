from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import assment_result
import os
from collections import Counter
import pandas as pd

from utils import accuracy, binary_accuracy, get_vttdata, get_afldata, get_degree_feature_list, check_and_creat_dir
from models import DGCN

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
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--origin', action='store_true', default=True,
                    help='Keep the original implementation as the paper.')
parser.add_argument('--test_only', action="store_true", default=False,
                    help='Test on existing model')
parser.add_argument('--repeat', type=int, default=1,
                    help='number of experiments')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='Show training process')
parser.add_argument('--split', type=str, default='random',
                    help='Data split method')
parser.add_argument('--rho', type=float, default=0.1,
                    help='Adj matrix corruption rate')
parser.add_argument('--corruption', type=str, default='node_shuffle',
                    help='Corruption method')

def get_data(dice, node_num, edges_path):
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    # adj: sparse tensor, symmetric normalized Laplication
    # A = D^(-1/2)*A*D^(1/2)
    print("prepared for loading data!")
    adj, features, graph = get_afldata(dice, node_num, edges_path)
    print("Load have done!")
    idx_train, idx_val, idx_test = get_vttdata(node_num)

    if args.cuda:
        features = features.cuda()
        adj = adj.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    return adj, features, idx_train, idx_val, idx_test, args, graph

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def train(num_epoch, time_step,last_embedding,patience=30, verbose=False):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf")
    best_epoch = -1
    for epoch in range(num_epoch):
        t = time.time()
        optimizer.zero_grad()
        if time_step >= 1:
            outputs, labels= model(features, adj, last_embedding)
        else:
            outputs, labels = model(features, adj)
        if args.cuda:
            labels = labels.cuda()
        loss_train = F.binary_cross_entropy_with_logits(outputs, labels)
        acc_train = binary_accuracy(outputs, labels)
        loss_train.backward()
        optimizer.step()

        loss = loss_train.item()
        accuracy = acc_train.item()
        if verbose:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss),
                  'acc_train: {:.4f}'.format(accuracy),
                  'time: {:.4f}s'.format(time.time() - t))

        # early stop
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
        if epoch == best_epoch + patience:
            break


def test(verbose=False):
    with torch.set_grad_enabled(False):
        model.eval()
        last_embedding = 0
        outputs, weight= model(features, adj, last_embedding)
        outputs_numpy = outputs.data.cpu().numpy()

    return outputs, weight, outputs_numpy


if __name__ == "__main__":
    dice = 0.0
    time_weight_list=[0]
    embedding_list = [0]
    NMI_list = []
    ################dataset input. Please input the right name of datasets and the corresponding number of nodes.

    # dataset = "cellphone"  ## An example dataset for node clustering
    # node_num = 400

    dataset = "bitcoinotc"   ## An example dataset for link prediction
    node_num = 6005

    ################dataset input

    isone_hot = True  ## If the link prediction experiment is chosen, please set isone_hot = True, otherwise isone_hot = False
    islabel = False   ## If the node clustering experiment is chosen, please set islabel = True, otherwise islabel = False
    method = "DGCN"
    base_data_path = "..\\..\\data\\"
    edges_base_path = "\\edges"
    label_base_path = "\\labels"
    edges_data_path = base_data_path + dataset + edges_base_path
    file_num = len(os.listdir(edges_data_path))
    if isone_hot:
        x_list, max_degree = get_degree_feature_list(edges_data_path,node_num)
    print("file_num:{}".format(file_num))
    for t in range(file_num):
        print("The {}th snapshot".format(t+1))
        time_edges_path = edges_data_path + "\\edges_t" + str(t+1) + ".txt"
        time_features_path = edges_data_path + "\\features_t" + str(t + 1) + ".txt"
        adj, features, idx_train, idx_val, idx_test, args, graph = get_data(dice, node_num, time_edges_path)
        print("Get data done!")
        if islabel:
            label_data_path = base_data_path + dataset + label_base_path + "\\label_" + str(t+1) +".txt"
            original_cluster = np.loadtxt(label_data_path, dtype=int)
        if isone_hot:
            print("the feature is generated by one-hot code!")
            features = x_list[t]
            features = sp.coo_matrix(features, shape=(node_num, max_degree))
            features = normalize(features)
            features = torch.FloatTensor(np.array(features.todense()))
        for i in range(args.repeat):
            # model
            model = DGCN(num_feat=features.shape[1],
                        num_hid=args.hidden,
                        time_step=t,
                        graph = graph,
                        time_weight=time_weight_list[-1],
                        dropout=args.dropout,
                        rho=args.rho,
                        corruption=args.corruption)

            print("----- %d / %d runs -----" % (i+1, args.repeat))
            # Train model
            t_total = time.time()
            if args.test_only:
                model = torch.load("model")
            else:
                train(args.epochs, t,embedding_list[-1], verbose=args.verbose)
                print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
                #torch.save(model, "model")

            # Test
            outputs, weight, outputs_numpy = test(verbose=args.verbose)
            save_emb_path = "../embeddings//" + dataset + "//" + method + "/embedding_t" + str(t) + ".txt"
            check_and_creat_dir(save_emb_path)
            np.savetxt(save_emb_path, outputs_numpy, fmt="%f")
            time_weight_list.append(weight)
            embedding_list.append(outputs)
            if islabel:
                k = len(Counter(original_cluster))
                NMI = assment_result.assement_result(original_cluster, outputs_numpy, k)
                print("NMI value is：{}".format(NMI))
                NMI_list.append(NMI)

    if islabel:
        ave_NMI = np.mean(NMI_list)
        print('--------------------------------------------')
        print("The average NMI is:{}".format(ave_NMI))
    else:
        print("“islabel = False”. Please execute 'links_prediction.py' for link prediction experiment！")
