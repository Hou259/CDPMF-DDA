import math
import random
import numpy as np
import sys
import scipy.io as scio
from sklearn.model_selection import KFold
import pandas as pd
from my_parser import parse_args
import torch

sys.path.append(".")


args = parse_args()
device = 'cuda:' + args.cuda

d = args.d
l = args.gnn_layer
temp = args.temp
epoch_no = args.epoch
dropout = args.dropout
lr = args.lr
seed = args.seed
# dataset = args.dataset
batch_size = args.batch_size
n_splits = args.n_splits
disease_TopK = args.disease_TopK
drug_TopK = args.drug_TopK

def load_mat(filepath):
    mat = scio.loadmat(filepath)
    drug_sim = mat["drug"].astype(np.float)
    disease_sim = mat["disease"].astype(np.float)
    drug_name = mat["Wrname"].reshape(-1)
    drug_num = len(drug_name)
    disease_name = mat["Wdname"].reshape(-1)
    disease_num = len(disease_name)
    interactions = mat["didr"].T
    return drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions

def load_lrssl(filepath, reduce=True):
    """ C drug:658, disease:409 association:2520 (False 2353)
        PREDICT drug:593, disease:313 association:1933 (Fdataset)
        LRSSL drug: 763, disease:681, association:3051
    """
    drug_chemical = pd.read_csv(filepath + "lrssl_simmat_dc_chemical.txt", sep="\t", index_col=0)
    drug_dataset = pd.read_csv(filepath + "lrssl_simmat_dc_domain.txt", sep="\t", index_col=0)
    drug_go = pd.read_csv(filepath + "lrssl_simmat_dc_go.txt", sep="\t", index_col=0)
    disease_sim = pd.read_csv(filepath + "lrssl_simmat_dg.txt", sep="\t", index_col=0)
    if reduce:
        drug_sim = (drug_chemical+drug_dataset+drug_go)/3
    else:
        drug_sim = drug_chemical
    drug_disease = pd.read_csv(filepath + "lrssl_admat_dgc.txt", sep="\t", index_col=0)
    drug_disease = drug_disease.T
    rr = drug_sim.to_numpy(dtype=np.float32)
    rd = drug_disease.to_numpy(dtype=np.float32)
    dd = disease_sim.to_numpy(dtype=np.float32)
    rname = drug_sim.columns.to_numpy()
    dname = disease_sim.columns.to_numpy()
    drug_sim = rr.astype(np.float)
    disease_sim = dd.astype(np.float)
    drug_name = rname.reshape(-1)
    drug_num = len(drug_name)
    disease_name = dname.reshape(-1)
    disease_num = len(disease_name)
    interactions = rd.T
    return drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions

def load_Lagcn(filepath):
    """drug:598, disease:269 association:18416
    """
    disease_sim = pd.read_csv(filepath + "dis_sim.csv", header=None).to_numpy(dtype=np.float32)
    interactions = pd.read_csv(filepath + "drug_dis.csv", header=None).to_numpy(dtype=np.float32)
    drug_sim = pd.read_csv(filepath + "drug_sim.csv", header=None).to_numpy(dtype=np.float32)
    disease_name = np.arange(disease_sim.shape[0])
    drug_name = np.arange(drug_sim.shape[0])
    disease_num = disease_sim.shape[0]
    drug_num = drug_sim.shape[1]
    return drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions

def get_train_adj(adj, train_mask, truth_label):
    for i in range(train_mask.shape[0]):
        for j in range(train_mask.shape[1]):
            if train_mask[i][j]:
                adj[i][j] = truth_label[i][j]
    return adj

def data_preparation(args):
    main_dir = sys.path[0]
    path = main_dir+"/data/" + args.dataset + "/"
    assert args.dataset in ["Cdataset", "Fdataset", "Lagcn", "LRSSL"]
    if args.dataset in ['Fdataset', 'Cdataset']:
        drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions = load_mat(
                                                                                                path + args.dataset + ".mat")
    elif args.dataset == "LRSSL":
        drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions = load_lrssl(path)
    else:
        drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions = load_Lagcn(path)
    return drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions

def kf(args, data):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    kfold = KFold(n_splits=args.n_splits, shuffle=True)

    pos_row, pos_col = np.nonzero(data)
    neg_row, neg_col = np.nonzero(1 - data)

    assert len(pos_row) + len(neg_row) == np.prod(data.shape)
    train_data, test_data = [], []

    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row),
                                                                            kfold.split(neg_row)):
        train_mask = np.zeros_like(data, dtype="bool")
        test_mask = np.zeros_like(data, dtype="bool")
        train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
        train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
        test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
        test_neg_edge = np.stack([neg_row[test_neg_idx], neg_col[test_neg_idx]])

        train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
        test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)

        train_mask[train_edge[0], train_edge[1]] = True
        test_mask[test_edge[0], test_edge[1]] = True
        train_data.append(train_mask)
        test_data.append(test_mask)

    pos_num = data.sum()
    neg_num = np.prod(data.shape) - pos_num
    pos_weight = neg_num / pos_num
    return train_data, test_data, pos_weight


class BatchManager(object):

    def __init__(self, data, batch_size, type):
        drug_input, disease_input, labels = [], [], []
        mask, label = data
        disease_drug_Adj = np.zeros((label.shape[0], label.shape[1]))
        disease_drug_Adj = get_train_adj(disease_drug_Adj, mask, label)

        self.train_adj = disease_drug_Adj

        if type == "train":
            train_mask, truth_label = data
            for i in range(len(train_mask)):
                for j in range(len(train_mask[0])):
                    if train_mask[i][j]:
                        drug_input.append(i)
                        disease_input.append(j)
                        labels.append(disease_drug_Adj[i][j])
                    else:
                        drug_input.append(i)
                        disease_input.append(j)
                        labels.append(0)
            num_batch = int(math.ceil(len(drug_input) / batch_size))
            self.batch_data = list()

            for i in range(num_batch):
                input_drug = drug_input[i * batch_size: (i + 1) * batch_size]
                input_disease = disease_input[i * batch_size: (i + 1) * batch_size]
                label = labels[i * batch_size: (i + 1) * batch_size]
                self.batch_data.append([input_drug, input_disease, label])

        elif type == "test":
            test_mask, truth_label = data
            for i in range(len(test_mask)):
                for j in range(len(test_mask[0])):
                    if test_mask[i][j]:
                        drug_input.append(i)
                        disease_input.append(j)
                        labels.append(truth_label[i][j])
            num_batch = int(math.ceil(len(drug_input) / batch_size))
            self.batch_data = list()
            for i in range(num_batch):
                input_drug = drug_input[i * batch_size: (i + 1) * batch_size]
                input_disease = disease_input[i * batch_size: (i + 1) * batch_size]
                label = labels[i * batch_size: (i + 1) * batch_size]
                self.batch_data.append([input_drug, input_disease, label])

        self.len_data = len(self.batch_data)

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]

def data_split(train_mask, test_mask, original_interactions):

    train_data = (train_mask, original_interactions)
    test_data = (test_mask, original_interactions)

    train_manager = BatchManager(train_data, batch_size, "train")
    test_manager = BatchManager(test_data, batch_size, "test")

    return train_manager, test_manager


