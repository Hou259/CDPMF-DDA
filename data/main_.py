from dataloader import *
import numpy as np
import torch
from model import DREMVCL
from utils import *
from sklearn import metrics
from my_parser import parse_args
from torch.optim.lr_scheduler import CyclicLR
import sys


sys.path.append(".")
args = parse_args()
device = 'cuda:' + args.cuda

d = args.d
l = args.gnn_layer
temp = args.temp
epoch_no = args.epoch
dropout = args.dropout
lr = args.lr
dataset = args.dataset
batch_size = args.batch_size
n_splits = args.n_splits
disease_TopK = args.disease_TopK
drug_TopK = args.drug_TopK
weight_decay = args.weight_decay
wd = args.wd
wr = args.wr

drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, original_interactions = data_preparation(args)
all_train_mask, all_test_mask, pos_weight = kf(args, original_interactions)

W, H, recMatrix = MF(original_interactions, drug_sim, disease_sim, 25)
recMatrix = torch.tensor(recMatrix)
W1 = W.T @ W
W1 = get_drug_sim_Matrix2(W1, drug_TopK)
W1 = torch.tensor(W1).double()
H1 = H.T @ H
H1 = get_disease_sim_Matrix2(H1,disease_TopK)
H1 = torch.tensor(H1).double()
print('MF done.')
all_scores, all_labels = [], []
drug_sim = get_drug_sim_Matrix(drug_sim, drug_TopK)
drug_train_mask, drug_test_mask, pos_weight1 = kf(args, drug_sim)

disease_sim = get_disease_sim_Matrix(disease_sim, disease_TopK)
disease_train_mask, disease_test_mask, pos_weight2 = kf(args, disease_sim)

def train():
    for fold_num in range(len(all_train_mask)):
        train_manager,_ = data_split(all_train_mask[fold_num], all_test_mask[fold_num],
                                                     original_interactions)
        train_adj = train_manager.train_adj
        train_adj = torch.tensor(train_adj)

        model = DREMVCL(drug_num, disease_num, d, train_adj, recMatrix, l, temp, dropout, batch_size, device, pos_weight, wr, wd)
        model.cuda(torch.device(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = CyclicLR(optimizer,
                                base_lr=0.1 * args.lr,
                                max_lr=args.lr,
                                step_size_up=20,
                                mode="exp_range",
                                gamma=0.995,
                                cycle_momentum=False)

        drug_manager, _ = data_split(drug_train_mask[fold_num], drug_test_mask[fold_num],
                                     drug_sim)
        drug_adj = drug_manager.train_adj
        drug_adj = torch.tensor(drug_adj)

        model1 = DREMVCL(drug_num, drug_num, d, drug_adj, W1, l, temp, dropout, batch_size, device, pos_weight1, wr, wd)
        model1.cuda(torch.device(device))
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler1 = CyclicLR(optimizer1,
                                base_lr=0.1 * args.lr,
                                max_lr=args.lr,
                                step_size_up=20,
                                mode="exp_range",
                                gamma=0.995,
                                cycle_momentum=False)
        disease_manager, _ = data_split(disease_train_mask[fold_num], disease_test_mask[fold_num],
                                      disease_sim)
        disease_adj = disease_manager.train_adj
        disease_adj = torch.tensor(disease_adj)

        model2 = DREMVCL(disease_num, disease_num, d, disease_adj, H1, l, temp, dropout, batch_size, device, pos_weight2, wr, wd)
        model2.cuda(torch.device(device))
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler2 = CyclicLR(optimizer,
                                base_lr=0.1 * args.lr,
                                max_lr=args.lr,
                                step_size_up=20,
                                mode="exp_range",
                                gamma=0.995,
                                cycle_momentum=False)

        for epoch in range(int(epoch_no)):
            model.train()
            # model1.train()
            # model2.train()
            # loss_dr,loss_rr,loss_dd = [],[],[]
            for batch in train_manager.iter_batch(shuffle=True):
                loss, scores = model.forward(batch, True)
                # loss_dr.append(loss.item())
                model.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            model.eval()
            # print(loss)
            for batch in drug_manager.iter_batch(shuffle=True):
                loss_rr, scores = model1.forward(batch, True)
                # loss_rr.append(loss.item())
                model1.zero_grad()
                loss_rr.backward()
                optimizer1.step()
                lr_scheduler1.step()
            model1.eval()
            # print(loss_rr)
            for batch in disease_manager.iter_batch(shuffle=True):
                loss_dd, scores = model2.forward(batch, True)
                # loss_dd.append(loss.item())
                model2.zero_grad()
                loss_dd.backward()
                optimizer2.step()
                lr_scheduler2.step()
            model2.eval()
            # print("loss_dr,loss_rr,loss_dd",loss_dr,loss_rr,loss_dd)

def cross_validation():
    avg_auroc, avg_aupr = [], []
    all_scores, all_labels = [], []
    for fold_num in range(len(all_train_mask)):
        print(f'---------------This is {fold_num + 1}-th fold validation.---------------')
        train_manager, test_manager = data_split(all_train_mask[fold_num], all_test_mask[fold_num],
                                                 original_interactions)
        train_adj = train_manager.train_adj
        train_adj = torch.tensor(train_adj)

        model = DREMVCL(drug_num, disease_num, d, train_adj, recMatrix, l, temp, dropout, batch_size, device,
                        pos_weight, wr, wd)
        model.cuda(torch.device(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = CyclicLR(optimizer,
                                base_lr=0.1 * args.lr,
                                max_lr=args.lr,
                                step_size_up=20,
                                mode="exp_range",
                                gamma=0.995,
                                cycle_momentum=False)
        for epoch in range(int(epoch_no)):
            model.train()
            for batch in train_manager.iter_batch(shuffle=True):
                loss, scores = model.forward(batch, True)
                model.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            model.eval()
            scores, labels = [], []
            for batch in test_manager.iter_batch():
                score, label = model.predict(batch)
                scores.append(score.detach().cpu().numpy())
                labels.append(label)
            scores = np.concatenate(scores)
            labels = np.concatenate(labels)
            aupr = metrics.average_precision_score(y_true=labels, y_score=scores)
            auroc = metrics.roc_auc_score(y_true=labels, y_score=scores)
            print(f'Epoch: {epoch + 1}, auroc: {auroc}, aupr: {aupr}')
            if (epoch + 1) == epoch_no:
                all_scores.append(scores)
                all_labels.append(labels)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    aupr = metrics.average_precision_score(y_true=all_labels, y_score=all_scores)
    auroc = metrics.roc_auc_score(y_true=all_labels, y_score=all_scores)
    avg_auroc.append(auroc)
    avg_aupr.append(aupr)
    print(f'------------------------------------------------------------------------')
    print(f"auroc：{auroc:.5f}")
    print(f"auroc：{aupr:.5f}")

if __name__ == "__main__":
    train()
    # cross_validation()




