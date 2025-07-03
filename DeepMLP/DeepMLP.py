# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : wangbing
# @File        : DeepMLP.py
# @Time     : 2025/7/1 11:34
# @Emal     : wangbing587@163.com
# @Desc     :


############################utils.py#################################
import pandas as pd
import numpy as np
import torch
import gseapy as gp
import networkx as nx
from sklearn.preprocessing import LabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import warnings

warnings.filterwarnings("ignore")


def expr_gsva(data, gsva_file='./database/KEGG_human.gmt'):
    df = data.T.apply(lambda x: (x - x.mean(0)) / x.std(0)).T
    es = gp.gsva(data=df,
                 gene_sets=gsva_file,
                 outdir=None)
    kegg = es.res2d.pivot(index='Term', columns='Name', values='ES').T.apply(lambda x: (x - x.mean(0)) / x.std(0)).T
    return df.map(float), kegg.map(float)


def PPINet(dn, dt, ppi):
    # 计算正常组 ppi_n
    ppisub = ppi[(ppi['GeneSymbol_A'].isin(dn.index)) & (ppi['GeneSymbol_B'].isin(dn.index))]
    ppisub['Value'] = 1.0
    G = nx.from_pandas_edgelist(ppisub, 'GeneSymbol_A', 'GeneSymbol_B', edge_attr='Value')
    G.add_nodes_from(dn.index)
    ppi_n = nx.to_pandas_adjacency(G, nodelist=dn.index)
    ppi_n.index.name = None

    # 计算差异性 PCC
    pcc_D = np.corrcoef(dt.values) - np.corrcoef(dn.values)
    M = pcc_D.mean()
    S = pcc_D.std()
    for t in np.linspace(2, 4, 21):
        t_up = M + t * S
        t_down = M - t * S

        # 动态调整蛋白质相互作用网络
        ppi_t = np.where(pcc_D < t_down, -1, np.where(pcc_D > t_up, 1, 0))
        ppi_t = ppi_n.values + ppi_t
        ppi_t = np.where(ppi_t >= 1, 1, 0)
        ppi_t = pd.DataFrame(data=ppi_t, columns=ppi_n.index, index=ppi_n.index)

        np.fill_diagonal(ppi_t.values, 1)
        pec = (ppi_n - ppi_t).abs().sum().sum() / ppi_n.sum().sum()
        if pec < 0.1:

            return ppi_n, ppi_t

    return ppi_n, ppi_t


def Net2Coo(ppi_net):
    ppi_net.index.name = None
    dp = ppi_net.stack().reset_index()
    dp = dp[dp[0] != 0]
    gene_map = {idx: i for i, idx in enumerate(ppi_net.index)}
    GeneA = dp['level_0'].map(gene_map).tolist()
    GeneB = dp['level_1'].map(gene_map).tolist()
    indices = np.array(list(zip(GeneA, GeneB))).T
    ppi_coo = torch.sparse.FloatTensor(indices=torch.tensor(indices, dtype=torch.int64),
                                       values=torch.ones(indices.shape[1], dtype=torch.float32),
                                       size=torch.Size((ppi_net.shape[0], ppi_net.shape[0]))).coalesce()
    return ppi_coo


def MultiLabelBinarizer(data, geo):
    # from sklearn.preprocessing import LabelBinarizer
    binarizer = LabelBinarizer()
    gocc = binarizer.fit_transform(geo['PSL'])
    gocc = pd.DataFrame(gocc, columns=np.sort(geo['PSL'].unique()))
    gocc['GeneSymbol'] = geo['GeneSymbol'].tolist()
    gocc = gocc.groupby('GeneSymbol').max()
    labels = pd.merge(pd.DataFrame(index=data.index), gocc, left_index=True, right_index=True, how='left')
    labels = labels.fillna(0)
    return labels


def data_split(labels, n_splits=5):
    # from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    gene_index = pd.DataFrame({'GeneIndex': np.arange(labels.shape[0]),
                               'GeneSymbol': labels.index})
    gene_index['CV'] = 'apply_set'
    y_sep = labels.loc[labels[labels.sum(1) != 0].index]

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(mskf.split(y_sep, y_sep)):
        val_idx = y_sep.iloc[val_index].index
        gene_index.loc[gene_index['GeneSymbol'].isin(val_idx), 'CV'] = f'CV_{fold + 1}'
    return gene_index


#############################model.py####################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np


class FeatureAttentionModule(nn.Module):
    def __init__(self, protein_dim, pathway_dim, hidden_dim=100):
        super(FeatureAttentionModule, self).__init__()
        # 蛋白质特征提取
        self.protein_fc = nn.Linear(protein_dim, hidden_dim)
        # 通路特征提取
        self.pathway_fc = nn.Linear(pathway_dim, hidden_dim)
        # 动态注意力机制
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, protein_data, pathway_data):
        # 特征提取
        protein_features = F.relu(self.protein_fc(protein_data))
        pathway_features = F.relu(self.pathway_fc(pathway_data))

        # 动态注意力机制
        Q = self.W_Q(protein_features)  # 查询矩阵
        K = self.W_K(pathway_features)  # 键矩阵
        V = self.W_V(pathway_features)  # 值矩阵

        # 计算注意力权重
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)

        # 加权求和
        Z = torch.matmul(attn_weights, V)
        return torch.cat((Z,protein_features),dim=1)


class GATLocalizationModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, dropout=0.5):
        super(GATLocalizationModule, self).__init__()
        self.hidden_dim = hidden_dim
        # 图注意力网络
        self.gat1 = GATConv(input_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        # 蛋白质定位预测
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 100),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(100, 50),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(50, 2))

    def forward(self, x, edge_index):
        # 图注意力网络
        x = F. leaky_relu(self.gat1(x, edge_index))
        x =  F. leaky_relu(self.gat2(x, edge_index))
        x = self.fc(x)
        return x


class DeepMLP(nn.Module):
    def __init__(self, protein_dim, pathway_dim, hidden_dim1=100, hidden_dim2=200):
        super(DeepMLP, self).__init__()
        # 特征提取与动态注意力机制模块
        self.feature_attention = FeatureAttentionModule(protein_dim, pathway_dim, hidden_dim1)
        # 图注意力网络与蛋白质定位预测模块
        self.gat_localization = GATLocalizationModule(hidden_dim2, hidden_dim2)

    def forward(self, protein_data, pathway_data, edge_index):
        # 特征提取
        integrated_features = self.feature_attention(protein_data, pathway_data)
        # 图注意力网络与蛋白质定位预测
        outputs = self.gat_localization(integrated_features, edge_index)
        return outputs


######################loss_function.py###################################
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        ce_loss = F.cross_entropy(outputs, targets, weight=self.alpha, reduction='none')
        focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()
        return focal_loss


############################datautils.py#################################

class DataProcess:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def main(self, data, an_col, geo, ppi, gsva_file='KEGG_human.gmt'):
        self.labels = MultiLabelBinarizer(data, geo)
        self.gene_index = data_split(self.labels, self.n_splits)

        dn = data[an_col[an_col['Type'] == 'Normal']['Sample_id']]
        dt = data[an_col[an_col['Type'] == 'Tumor']['Sample_id']]

        self.ppi_n, self.ppi_t = PPINet(dn, dt, ppi)
        self.edge_n, self.edge_t = Net2Coo(self.ppi_n), Net2Coo(self.ppi_t)
        self.x_n, self.k_n = expr_gsva(dn, gsva_file=gsva_file)
        self.x_t, self.k_t = expr_gsva(dt, gsva_file=gsva_file)


##########################main_normal.py#######################################

from sklearn.metrics import roc_auc_score


class PathLoc:
    def __init__(self, epochs=500, patience=50, n_splits=5):
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_splits = n_splits

    def main(self, DP):
        x_n = torch.tensor(DP.x_n.values).float().to(self.device)
        k_n = torch.tensor(DP.k_n.values).float().to(self.device)

        x_t = torch.tensor(DP.x_t.values).float().to(self.device)
        k_t = torch.tensor(DP.k_t.values).float().to(self.device)

        edge_n = DP.edge_n.to(self.device)
        edge_t = DP.edge_t.to(self.device)

        CV_idx = DP.gene_index.loc[DP.gene_index['CV'] != 'apply_set', 'GeneIndex'].values

        self.yN_prob = pd.DataFrame(index=DP.x_n.index)
        self.yT_prob = pd.DataFrame(index=DP.x_t.index)

        aucN = []
        aucT = []

        for marker in DP.labels.columns:
            y = torch.tensor(DP.labels[marker].values).long().to(self.device)
            weights = self.compute_class_weights(DP.labels.iloc[CV_idx][marker].map(int).values)

            self.yN_prob[marker], aucs_n = self.cross_validation(x_n, k_n, edge_n, y, DP.gene_index, weights)

            aucN.append(aucs_n)

            self.yT_prob[marker], aucs_t = self.cross_validation(x_t, k_t, edge_t, y, DP.gene_index, weights)

            aucT.append(aucs_t)
            # print(f'{marker}: Normal = {aucs_n} | Tumor = {aucs_t}')


        self.auc = pd.DataFrame({'Normal': aucN, 'Tumor': aucT}, index=DP.labels.columns)

    def cross_validation(self, x, k, edge, y, gene_index, weights):
        apply_idx = gene_index[gene_index['CV'] == 'apply_set']['GeneIndex'].values

        aucs = 0.0
        y_prob = np.zeros(y.shape[0])

        for i in range(self.n_splits):
            f = f'CV_{i + 1}'
            train_idx = gene_index[~gene_index['CV'].isin([f, 'apply_set'])]['GeneIndex'].values
            train_mask = torch.tensor(train_idx).long().to(self.device)

            val_idx = gene_index[gene_index['CV'] == f]['GeneIndex'].values
            val_mask = torch.tensor(val_idx).long().to(self.device)

            model = DeepMLP(x.shape[1], k.shape[1]).to(self.device)
            loss_func = FocalLoss(alpha=weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            val_auc_best = 0.0
            patience_counter = 0

            for epoch in range(self.epochs):
                self.train(model, loss_func, optimizer, x, k, edge, y, train_mask)
                outputs = self.test(model, x, k, edge)
                val_auc = roc_auc_score(y[val_mask].detach().cpu(), outputs[val_mask, 1].detach().cpu())

                if val_auc > val_auc_best:
                    outputs_best = outputs
                    val_auc_best = val_auc
                    patience_counter += 1
                else:
                    patience_counter = 0

                if patience_counter >= self.patience:
                    break

            outputs_prob = F.softmax(outputs_best.detach().cpu(), dim=1).numpy()[:, 1]
            y_prob[val_idx] = outputs_prob[val_idx]
            y_prob[apply_idx] += outputs_prob[apply_idx] / self.n_splits
            aucs += val_auc_best / self.n_splits

        return y_prob, aucs

    def train(self, model, loss_func, optimizer, x, k, edge, y, train_mask):
        model.train()
        outputs = model(x, k, edge)
        loss = loss_func(outputs[train_mask], y[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test(self, model, x, k, edge):
        model.eval()
        with torch.no_grad():
            outputs = model(x, k, edge)
        return outputs

    def compute_class_weights(self, target):
        weights = 1.0 / np.bincount(target)
        weights = weights / weights.sum()
        weights = (torch.tensor(weights).float().to(self.device))
        return weights


############################################PSLcalc##################################################

def FDRcalc(sorted_labels):
    R = np.cumsum(np.ones_like(sorted_labels))  # 预测为正类的样本数
    V = np.cumsum(sorted_labels == 0)  # 假阳性数
    FDR = V / R
    return FDR


def PSLcalc(label_r, y_prob, t=0.05):
    label_th = label_r.copy()
    label_th[:] = 0
    ths = []
    ms = []
    ns = []
    fdr = pd.DataFrame(index=label_r.index)
    markers = label_r.columns
    for marker in markers:
        a = label_r[marker].loc[y_prob[marker].sort_values(ascending=False).index]
        FDR = FDRcalc(a.values)
        f = a.copy()
        f[:] = FDR
        f = f.loc[label_r.index]
        fdr[marker] = f.loc[label_r.index].tolist()
        m = FDR[FDR > 0].min()
        if m >= t:
            threshold_idx = np.where(FDR <= m)[0][-1]
            n = m
        else:
            threshold_idx = np.where(FDR <= t)[0][-1]
            n = t

        ms.append(m)
        ns.append(threshold_idx)
        ths.append(n)
        label_th.loc[a.iloc[:(threshold_idx + 1)].index, marker] = 1
    label_th = label_th.astype(int)
    stat = pd.DataFrame({'PSL': markers, 'MinFDR': ms, 'Threshold': ths, 'Count': ns})
    return fdr, label_th, stat


#####################################DiffPSL#####################################

def DiffPSL(label_r, yN_prob, yT_prob, bN, bT, p):
    yN_prob1 = yN_prob.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0).apply(lambda x: x / x.sum(), axis=1)
    yT_prob1 = yT_prob.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0).apply(lambda x: x / x.sum(), axis=1)
    diff = (yT_prob1.round(3) - yN_prob1.round(3)) / (yN_prob1.round(3))
    diff[np.isnan(diff)] = 0.0
    diff[np.isinf(diff)] = 0.0
    diffT = diff * bT
    diffN = diff * bN

    result = pd.DataFrame({'DiffMaxPSL': diffT.columns[diffT.values.argmax(1)],
                           'DiffMinPSL': diffN.columns[diffN.values.argmin(1)],
                           'DiffMax': diffT.max(1), 'DiffMin': diffN.min(1)})

    result['DiffMaxRank'] = result['DiffMax'].rank(ascending=False).astype(int)
    result['DiffMinRank'] = result['DiffMin'].rank(ascending=True).astype(int)

    result['DiffMaxTop'] = np.where(result['DiffMaxRank'] <= int(result.shape[0] * p), 'YES', 'NO')
    result['DiffMinTop'] = np.where(result['DiffMinRank'] <= int(result.shape[0] * p), 'YES', 'NO')
    result.loc[result['DiffMaxPSL'] == result['DiffMinPSL'], 'DiffMinTop'] = 'NO'
    result.loc[result['DiffMaxPSL'] == result['DiffMinPSL'], 'DiffMaxTop'] = 'NO'

    result['DiffAllTop'] = 'NO'
    result.loc[(result['DiffMaxTop'] == 'YES') | (result['DiffMinTop'] == 'YES'), 'DiffAllTop'] = 'YES'
    return result


##############################################################################

import os
import argparse


def main(datafile, an_col, p, n_splits, epochs):
    geo = pd.read_csv('./database/GOA_PSL_human.csv')
    ppi = pd.read_csv('./database/BioGRID_PPI_human.csv')
    data = pd.read_csv(datafile, index_col=0)
    an_col = pd.read_csv(an_col)

    DP = DataProcess(n_splits=n_splits)
    DP.main(data, an_col, geo, ppi, gsva_file='./database/KEGG_human.gmt')

    model = PathLoc(epochs=epochs)
    model.main(DP)

    os.makedirs('output', exist_ok=True)


    DP.ppi_n.to_csv('./output/PPI_N.csv')
    DP.ppi_t.to_csv('./output/PPI_T.csv')
    DP.gene_index.to_csv('./output/gene_index.csv', index=None)
    DP.labels.to_csv('./output/label.csv')
    model.auc.to_csv('./output/auc.csv')
    model.yN_prob.to_csv('./output/Normal_prob.csv')
    model.yT_prob.to_csv('./output/Tumor_prob.csv')

    label_r = pd.DataFrame(data=np.where(DP.ppi_n @ DP.labels >= 1, 1, 0), columns=DP.labels.columns, index=DP.labels.index)

    aN, bN, cN = PSLcalc(label_r, model.yN_prob)
    aN.to_csv('./output/Normal_fdr.csv')
    bN.to_csv('./output/Normal_pred.csv')
    cN.to_csv('./output/Normal_threshold.csv', index=None)

    aT, bT, cT = PSLcalc(label_r, model.yT_prob)
    aT.to_csv('./output/Tumor_fdr.csv')
    bT.to_csv('./output/Tumor_pred.csv')
    cT.to_csv('./output/Tumor_threshold.csv', index=None)

    result = DiffPSL(label_r, model.yN_prob, model.yT_prob, bN, bT, p)

    result.to_csv('./output/DiffPSL.csv')



if __name__ == '__main__':
    import time
    parser = argparse.ArgumentParser(
        description="DeepMLP: A proteomics-driven deep learning framework for identifying mis-localized proteins across pan-cancer."
    )

    parser.add_argument("-f",
                        "--file",
                        dest='f',
                        type=str,
                        help="Path to the input data file. The first column must contain unique GeneSymbols.")

    parser.add_argument("-a",
                        "--an_col",
                        dest='a',
                        type=str,
                        help="Path to the sample annotation file. It must contain columns: ['Sample_id', 'Type'].")

    parser.add_argument("-p",
                        "--proportion",
                        dest='p',
                        type=float,
                        help="Proportion threshold for selecting loss-type and gain-type MLPs. Default is 0.03.",
                        default=0.03)

    parser.add_argument("-n",
                        "--n_splits",
                        dest='n',
                        type=int,
                        help="Number of folds for cross-validation. Default is 5.",
                        default=5)

    parser.add_argument("-e",
                        "--epochs",
                        dest='e',
                        type=int,
                        help="Number of training epochs. Default is 500.",
                        default=500)


    args = parser.parse_args()

    datafile = args.f
    an_col = args.a
    n_splits = args.n
    epochs = args.e
    p = args.p

    t0 = time.time()
    main(datafile, an_col,p, n_splits, epochs)
    print('using time: {} m {}s'.format(int((time.time() - t0) // 60), (time.time() - t0) % 60))
