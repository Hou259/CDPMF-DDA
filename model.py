import torch
import torch.nn as nn
from utils import sparse_dropout
import torch.nn.functional as F

class DREMVCL(nn.Module):
    def __init__(self, drug_num, disease_num, d, adj_norm,rec,l, temp, dropout, batch_user, device , pos_weight, wr, wd):
        super(DREMVCL,self).__init__()
        self.E_r_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(drug_num, d)))
        self.E_d_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(disease_num,d)))
        self.adj_norm = adj_norm
        self.dr = adj_norm
        self.l = l

        self.E_r_list = [None] * (l+1)
        self.E_d_list = [None] * (l+1)

        self.E_r_list[0] = self.E_r_0
        self.E_d_list[0] = self.E_d_0
        self.Z_r_list = [None] * (l+1)
        self.Z_d_list = [None] * (l+1)
        self.G_r_list = [None] * (l+1)
        self.G_d_list = [None] * (l+1)
        self.G_r_list[0] = self.E_r_0
        self.G_d_list[0] = self.E_d_0
        self.temp = temp
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user
        self.rec = rec
        self.E_r = None
        self.E_d = None
        self.device = device
        self.pos_weight = pos_weight
        self.wr = wr
        self.wd = wd

    def forward(self, batch, is_training):
        drugs, diseases, labels = batch
        for layer in range(1,self.l+1):
            self.E_r_list[layer - 1] = self.E_r_list[layer - 1].double()
            self.E_d_list[layer - 1] = self.E_d_list[layer - 1].double()

            self.adj_norm = self.adj_norm.double().cuda()
            self.E_r_list[layer - 1] = self.E_r_list[layer - 1].cuda()
            self.E_d_list[layer - 1] = self.E_d_list[layer - 1].cuda()

            self.Z_r_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout), self.E_d_list[layer-1])).cuda(torch.device(self.device))
            self.Z_d_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout).transpose(0,1), self.E_r_list[layer-1])).cuda(torch.device(self.device))

            self.rec = self.rec.cuda(torch.device(self.device))
            self.G_r_list[layer] = self.rec @ self.E_d_list[layer - 1]
            self.G_d_list[layer] = self.rec.T @ self.E_r_list[layer - 1]

            self.E_r_list[layer] = self.Z_r_list[layer]
            self.E_d_list[layer] = self.Z_d_list[layer]

        # self.G_r_list = [G_r.to(self.device) for G_r in self.G_r_list]
        self.G_r = sum(self.G_r_list)
        # self.G_d_list = [G_d.to(self.device) for G_d in self.G_d_list]
        self.G_d = sum(self.G_d_list)

        self.E_r = sum(self.E_r_list)
        self.E_d = sum(self.E_d_list)

        G_r_norm = self.G_r
        E_r_norm = self.E_r
        G_d_norm = self.G_d
        E_d_norm = self.E_d

        drugEmbedding = (self.wr * E_r_norm + (1 - self.wr) * G_r_norm)
        diseaseEmbedding = (self.wd * E_d_norm + (1 - self.wd) * G_d_norm)
        batch_drug_all_embeddings = drugEmbedding[drugs]
        batch_disease_all_embeddings = diseaseEmbedding[diseases]
        scores = torch.mul(batch_drug_all_embeddings, batch_disease_all_embeddings).sum(dim=1).cuda(torch.device(self.device))
        ssl_loss_drug = self.ssl_loss(E_r_norm, G_r_norm, drugs)
        ssl_loss_disease= self.ssl_loss(E_d_norm, G_d_norm, diseases)
        ssl_loss = 0.05 * ssl_loss_disease + 0.05 * ssl_loss_drug
        labels = torch.FloatTensor(labels).double().cuda(torch.device(self.device))
        BCE_loss = self.bce_loss_fn(torch.sigmoid(scores), labels)
        loss = BCE_loss + 0.3 * ssl_loss

        return loss, torch.sigmoid(scores)
    def bce_loss_fn(self, predict, label):
        predict = predict.reshape(-1)
        label = label.reshape(-1)
        weight = self.pos_weight * label + 1 - label
        loss = F.binary_cross_entropy(input=predict, target=label, weight=weight)
        return loss
    def ssl_loss(self, data1, data2, index):
        index = torch.unique(torch.LongTensor(index))
        embeddings1 = data1[index]
        embeddings2 = data2[index]
        norm_embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        norm_embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        pos_score = torch.sum(torch.mul(norm_embeddings1, norm_embeddings2), dim=1)
        all_score = torch.mm(norm_embeddings1, norm_embeddings2.T)
        pos_score = torch.exp(pos_score / 0.05)
        all_score = torch.sum(torch.exp(all_score / 0.05), dim=1)
        ssl_loss = (-torch.sum(torch.log(pos_score / ((all_score)))) / (len(index)))
        return ssl_loss
    def predict(self, batch):
        _, _, labels = batch
        _, scores = self.forward(batch,False)
        return scores, labels
















