import torch
import torch.nn as nn
import numpy as np
def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)

def MF(Y, M, D, r):
    m = M.shape[1]
    n = D.shape[1]
    B = np.where(Y != 0,1,0)
    W = np.random.rand(m, r)
    np.random.seed(0)
    H = np.random.rand(n, r)
    W = W.T
    H = H.T
    parm1 = 1
    parm2 = 1
    parm3 = 0.1
    parm4 = 0.1
    k = 1
    while k < 1000:
        W = W * ((H.dot(Y.T * B.T) + 2 * parm3 * W.dot(M)) /
                  (H.dot((H.T @ W) * B.T) + parm1 * W + 2 * parm3 * W.dot(W.T).dot(W)))
        H = H * ((W.dot(Y * B) + 2 * parm4 * H.dot(D)) /
                 (W.dot((W.T @ H) * B) + parm2 * H + 2 * parm4 * H.dot(H.T).dot(H)))
        k += 1
    score = W.T @ H
    return W, H, score

def get_drug_sim_Matrix(drug_similarity, drug_drug_topk):
    drug_drug_Matrix = np.zeros((drug_similarity.shape[0], drug_similarity.shape[1]), np.float32)
    drug_sim_Matrix = drug_similarity
    for drug_num0 in range(drug_similarity.shape[0]):
        drug_sim = {}
        for drug_num1 in range(drug_similarity.shape[1]):
            if drug_num0 == drug_num1:
                continue
            drug_sim[drug_num1] = drug_sim_Matrix[drug_num0][drug_num1]
        sorted_drug_list = sorted(drug_sim.items(), key=lambda d: d[1], reverse=True)
        for i in range(min(drug_drug_topk, len(sorted_drug_list))):
            drug_num1 = sorted_drug_list[i][0]
            drug_drug_Matrix[drug_num0][drug_num1] = 1
    return drug_drug_Matrix

def get_drug_sim_Matrix2(drug_similarity, drug_drug_topk):
    drug_drug_Matrix = np.zeros((drug_similarity.shape[0], drug_similarity.shape[1]), np.float32)
    drug_sim_Matrix = drug_similarity
    for drug_num0 in range(drug_similarity.shape[0]):
        drug_sim = {}
        for drug_num1 in range(drug_similarity.shape[1]):
            if drug_num0 == drug_num1:
                continue
            drug_sim[drug_num1] = drug_sim_Matrix[drug_num0][drug_num1]
        sorted_drug_list = sorted(drug_sim.items(), key=lambda d: d[1], reverse=True)
        for i in range(min(drug_drug_topk, len(sorted_drug_list))):
            drug_num1 = sorted_drug_list[i][0]
            drug_drug_Matrix[drug_num0][drug_num1] = drug_sim_Matrix[drug_num0][drug_num1]
    return drug_drug_Matrix

def get_disease_sim_Matrix(disease_similarity, disease_disease_topk):
    disease_disease_Matrix = np.zeros((disease_similarity.shape[0], disease_similarity.shape[1]), np.float32)
    disease_sim_Matrix = disease_similarity
    for disease_num0 in range(disease_similarity.shape[0]):
        disease_sim = {}
        for disease_num1 in range(disease_similarity.shape[1]):
            if disease_num0 == disease_num1:
                continue
            disease_sim[disease_num1] = disease_sim_Matrix[disease_num0][disease_num1]
        sorted_disease_list = sorted(disease_sim.items(), key=lambda d: d[1], reverse=True)
        for i in range(min(disease_disease_topk, len(sorted_disease_list))):
            disease_num1 = sorted_disease_list[i][0]
            disease_disease_Matrix[disease_num0][disease_num1] = 1
    return disease_disease_Matrix

def get_disease_sim_Matrix2(disease_similarity, disease_disease_topk):
    disease_disease_Matrix = np.zeros((disease_similarity.shape[0], disease_similarity.shape[1]), np.float32)
    disease_sim_Matrix = disease_similarity
    for disease_num0 in range(disease_similarity.shape[0]):
        disease_sim = {}
        for disease_num1 in range(disease_similarity.shape[1]):
            if disease_num0 == disease_num1:
                continue
            disease_sim[disease_num1] = disease_sim_Matrix[disease_num0][disease_num1]
        sorted_disease_list = sorted(disease_sim.items(), key=lambda d: d[1], reverse=True)
        for i in range(min(disease_disease_topk, len(sorted_disease_list))):
            disease_num1 = sorted_disease_list[i][0]
            disease_disease_Matrix[disease_num0][disease_num1] = disease_sim_Matrix[disease_num0][disease_num1]

    return disease_disease_Matrix









