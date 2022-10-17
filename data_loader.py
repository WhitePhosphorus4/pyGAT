import numpy as np
import scipy.sparse as sp
import torch
import random
from utils import *

def load_cora_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_txt_data(path='./Points/', data_name='SPP', num_points=None, each_class_num=50, val_num=1000, test_num=5000):
    '''load points data'''
    print('Loading {} dataset....'.format(data_name))
    all = np.load(path+data_name+'_adj.npy')[:, :]
    if num_points is None:
        num_points = all.shape[0]
    ADJ = sp.coo_matrix(all[:num_points, :num_points], dtype=np.float32)
    Feature = np.load(path+data_name+'_features.npy')[:num_points, :]

    # ADJ = torch.FloatTensor(ADJ)
    ADJ = sparse_mx_to_torch_sparse_tensor(ADJ)
    features = torch.FloatTensor(Feature[:, :6])
    labels = torch.LongTensor(Feature[:, -1])
    unique, count = np.unique(labels, return_counts=True)
    print('The number of total each class is {}'.format(dict(zip(unique,count))))

    # fix each class num
    cou = [0 for i in range(len(unique))]
    idx_train = []
    for i in range(num_points):
        cou[labels[i]-1] += 1
        if cou[labels[i]-1] <= each_class_num:
            idx_train.append(i)
    r = [i for i in range(num_points) if i not in idx_train]
    idx_val = random.sample(list(r), int(val_num))
    r = [i for i in r if i not in idx_val]
    idx_test = random.sample(list(r), int(test_num))
    # idx_val = r[:int(0.5*len(r))]
    # idx_test = r[int(0.5*len(r)):]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    unique, count = np.unique(labels[idx_train], return_counts=True)
    print('Train Dataset : The number of train each class is {}'.format(dict(zip(unique, count))))
    unique, count = np.unique(labels[idx_val], return_counts=True)
    print('Val Dataset : The number of train each class is {}'.format(dict(zip(unique, count))))
    unique, count = np.unique(labels[idx_test], return_counts=True)
    print('Test Dataset : The number of train each class is {}'.format(dict(zip(unique, count))))

    print('Loading {} dataset Done.'.format(data_name))
    return ADJ, features, labels, idx_train, idx_val, idx_test




if __name__ == '__main__':
    load_txt_data()
