'''
 -*- coding: utf-8 -*-
 @Author  : donghu
 @Time    : 2021/10/19 8:40
 @File    : 
 @Software: PyCharm
 来自17
 '''
import math
import random

import networkx as nx
import numpy as np
import scipy.sparse as ssp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from torch.nn import ModuleList, Conv1d, MaxPool1d, Linear, BCEWithLogitsLoss
from torch_geometric.nn import GCNConv, global_sort_pool
import torch.nn.functional as F

from preprocessing import mask_test_edges
from utils import read_data, sample_neg, links2subgraphs, gen_features, generate_node2vec_embeddings, node_label, \
    EarlyStopping
from numpy import *

seed = 1997
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


node_num = 151  # Number of nodes in the dynamic network 动态网络中节点总数
name_pre = "Enron-employees\\m_enron_employees"  # Prefix name of the data file 数据文件的前缀名称
max_thres = 2000  # Threshold of the maximum edge weight 最大边权值的阈值，这里不需要

# 从目标测试集合进行提取节点对
cur_adj8 = read_data(name_pre, 8, node_num, max_thres)
A_8 = ssp.csr_matrix(cur_adj8)
A_8[np.arange(node_num), np.arange(node_num)] = 0  # 移除自环
A_8.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x
np.random.seed(1997)  # make sure train-test split is consistent between notebooks
# train_pos, train_neg, test_pos, test_neg = sample_neg(A_8)
# mask_pos = test_pos
# l = len(test_pos[0])
# val_pos = (test_pos[0][:l//3], test_pos[1][:l//3])
# test_pos = (test_pos[0][l//3:], test_pos[1][l//3:])
# val_neg = (test_neg[0][:l//3], test_neg[1][:l//3])
# test_neg = (test_neg[0][l//3:], test_neg[1][l//3:])


adj_train, train_pos, train_neg, val_pos, val_neg, test_pos, test_neg \
    = mask_test_edges(A_8, test_frac=.2, val_frac=.1, prevent_disconnect=True, verbose=False)

# 从子图1-7挨个提取子图

only_predict = False
hop = 1
max_nodes_per_hop = None
node_information = None
no_parallel = True
use_embedding = True
use_attribute = False
attributes = None
max_labels = 0

adj = []
A = []
graph_A = []
graph_A_features = []
gnnGraph_A_train = []
gnnGraph_A_test = []
gnnGraph_A_val = []
features_A = []
for i in range(0, 7):
    # print("------------------------------------")
    adj.append(read_data(name_pre, i+1, node_num, max_thres))
    A_csr = ssp.csr_matrix(adj[i])
    A_csr[np.arange(node_num), np.arange(node_num)] = 0
    # for link in val_pos:
    #     A_csr[link[0], link[1]] = 0
    #     A_csr[link[1], link[0]] = 0
    # for link in test_pos:
    #     A_csr[link[0], link[1]] = 0
    #     A_csr[link[1], link[0]] = 0
    A_csr.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x
    node_information = generate_node2vec_embeddings(A_csr, emd_size=128, negative_injection=False, train_neg=train_neg)

    # print(A_csr.sum())
    A.append(A_csr)
    train_graphs, test_graphs, val_graphs, max_n_label, max_num_nodes = links2subgraphs(
        A_csr,
        train_pos,
        train_neg,
        test_pos,
        test_neg,
        val_pos,
        val_neg,
        hop,
        max_nodes_per_hop,
        node_information,
        no_parallel
    )
    max_labels = max(max_n_label, max_labels)
    gnnGraph_A_train.append(train_graphs)
    gnnGraph_A_test.append(test_graphs)
    gnnGraph_A_val.append(val_graphs)

print(gnnGraph_A_train[0][100].nodes)
print(gnnGraph_A_train[6][100].nodes)

print("最大标签数：")
print(max_n_label)
print(train_graphs[100].nodes)
print(gnnGraph_A_train[6][100].nodes)

# 得到总的图
adj_all = np.mat(np.zeros((node_num, node_num)))
for i in range(0, 7):
    cur_adj = read_data(name_pre, i+1, node_num, max_thres)
    adj_all += cur_adj

A_all = ssp.csr_matrix(adj_all)
A_all[np.arange(node_num), np.arange(node_num)] = 0  # 移除自环

A_all.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x

# 把总的边权重矩阵转化为类似于one-hot向量形式
A_weighted = A_all
A_weighted = np.sum(adj_all, axis=1)
A_weighted = A_weighted.flatten()
A_weighted = A_weighted/np.linalg.norm(A_weighted)
A_weighted = A_weighted.tolist()[-1]
A_weighted = np.diag(A_weighted)

# 总图的节点嵌入
node_information = generate_node2vec_embeddings(A_all, emd_size=128, negative_injection=False, train_neg=train_neg)

train_graphs, test_graphs, val_graphs, max_n_label, max_num_nodes = links2subgraphs(
        A_all,
        train_pos,
        train_neg,
        test_pos,
        test_neg,
        val_pos,
        val_neg,
        hop,
        max_nodes_per_hop,
        node_information,
        no_parallel
    )
print('# train: %d, # test: %d, # val: %d' % (len(train_graphs), len(test_graphs), len(val_graphs)))
gnnGraph_A_train.append(train_graphs)
gnnGraph_A_test.append(test_graphs)
gnnGraph_A_val.append(val_graphs)
max_labels = max(max_n_label, max_labels)

randnum = random.randint(0, 100)
for i in range(0, 8):
    random.seed(randnum)
    random.shuffle(gnnGraph_A_train[i])
    random.shuffle(gnnGraph_A_test[i])
    random.shuffle(gnnGraph_A_val[i])

# 判断要预测的子图顺序对应的目标链接在第八个快照中的存在与否
train_labels = []
test_labels = []
val_labels = []
for data in gnnGraph_A_train[0]:
    i, j = data.link
    if cur_adj8[i, j] == 0:
        train_labels.append(0)
    else:
        train_labels.append(1)
for data in gnnGraph_A_test[0]:
    i, j = data.link
    if cur_adj8[i, j] == 0:
        test_labels.append(0)
    else:
        test_labels.append(1)
for data in gnnGraph_A_val[0]:
    i, j = data.link
    if cur_adj8[i, j] == 0:
        val_labels.append(0)
    else:
        val_labels.append(1)
print(test_labels)
print(sum(train_labels))
print(sum(val_labels))
print(sum(test_labels))


t_list = []
for graph in gnnGraph_A_train:
    t = 0.6
    if t < 1:  # Transform percentile to number.
        num_nodes = sorted([data.num_nodes for data in graph])
        t = num_nodes[int(math.ceil(t * len(num_nodes))) - 1]
        t = max(10, t)
    t = int(t)
    t_list.append(t)
print(t_list)
t = int(mean(t_list))
print(t)

class DGCNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, GNN=GCNConv, k=0.6):
        super(DGCNN, self).__init__()

        if k < 1:  # Transform percentile to number.
            num_nodes = sorted([data.num_nodes for data in gnnGraph_A_train[-2]])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)
        print("k=")
        print(self.k)

        self.convs = ModuleList()
        self.convs.append(GNN(128+max_labels+1, hidden_channels))
        # self.convs.append(GNN(128, hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        # self.lin = Linear(dense_dim, 128)

        # self.lstm = torch.nn.LSTM(input_size=dense_dim, hidden_size=dense_dim, num_layers=2)  # 增加LSTM
        #
        # self.lin1 = Linear(dense_dim*2, 128)
        #
        # self.lin2 = Linear(128, 1)

        self.w = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w.data.fill_(0.5)
        # self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.w1.data.fill_(0.3)
        self.lin1 = Linear((dense_dim+65) * 2, 128)
        # self.lin2 = Linear(512, 256)
        # self.lin3 = Linear(256, 128)
        self.lin4 = Linear(128, 1)

    def forward(self, x, edge_index, batch):
        input_lstm = []
        for i in range(0, 7):
            xs = [x[i]]
            for conv in self.convs:
                xs += [torch.tanh(conv(xs[-1], edge_index[i]))]
            x[i] = torch.cat(xs[1:], dim=-1)
            emb_edge = (x[i][0] * x[i][1]).reshape(1, 65)
            # Global pooling.
            x[i] = global_sort_pool(x[i], batch, self.k)
            x[i] = x[i].unsqueeze(1)  # [num_graphs, 1, k * hidden]
            x[i] = F.relu(self.conv1(x[i]))
            x[i] = self.maxpool1d(x[i])
            x[i] = F.relu(self.conv2(x[i]))
            x[i] = x[i].view(x[i].size(0), -1)  # [num_graphs, dense_dim]
            # x[i] = F.relu(self.lin(x[i]))
            x[i] = torch.cat((emb_edge, x[i]), 1)
            input_lstm.append(x[i])
        # for t in range(6, 0, -1):
        #     c = t
        #     while c != 0:
        #         input_lstm[t] = input_lstm[t] + pow((1 - self.w), (7 - c)) * input_lstm[c - 1]
        #         c = c - 1
        c = 6
        while c != 0:
            input_lstm[6] = input_lstm[6] + pow((1 - self.w), (6 - c)) * input_lstm[c - 1]
            c = c - 1
        # inputs = torch.cat((input_lstm[0], input_lstm[1], input_lstm[2], input_lstm[3],
        #                     input_lstm[4], input_lstm[5], input_lstm[6]), dim=0)
        # inputs = torch.unsqueeze(inputs, 1)
        # # print(inputs.size())
        # output, (hn, cn) = self.lstm(inputs)
        # outputs = 0
        # for t in range(0, len(output)):
        #     outputs += pow((1-self.w1), (6-t))*output[t]

        # 对整个快照再进行一个处理
        xs = [x[7]]
        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index[7]))]
        x[7] = torch.cat(xs[1:], dim=-1)
        emb_edge = (x[7][0] * x[7][1]).reshape(1, 65)
        # Global pooling.
        x[7] = global_sort_pool(x[7], batch, self.k)
        x[7] = x[7].unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x[7] = F.relu(self.conv1(x[7]))
        x[7] = self.maxpool1d(x[7])
        x[7] = F.relu(self.conv2(x[7]))
        x[7] = x[7].view(x[7].size(0), -1)  # [num_graphs, dense_dim]
        x[7] = torch.cat((emb_edge, x[7]), 1)
        # x[7] = F.relu(self.lin(x[7]))

        # input_all = torch.cat((x[7], output[-1]), dim=0)
        # input_all = torch.cat((input_all[0], input_all[1]), 0)

        inputs = torch.cat((input_lstm[6], x[7]), dim=1)

        # MLP.
        x = F.relu(self.lin1(inputs))
        # x = F.relu(self.lin3(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin4(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DGCNN(hidden_channels=32, num_layers=2).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
print(model)


def train():
    model.train()
    total_loss = 0
    len_train = len(gnnGraph_A_train[0])

    for i in range(0, len_train-10, 10):
        optimizer.zero_grad()
        logits_batch = torch.tensor([1])
        for c in range(0, 10):
            x = []
            edge_index = []
            for j in range(0, 7):
                curAdj = torch.LongTensor(2, len(gnnGraph_A_train[j][i+c].edge_pairs) // 2)
                for t in range(0, len(gnnGraph_A_train[j][i+c].edge_pairs), 2):
                    curAdj[0][t // 2] = int(gnnGraph_A_train[j][i+c].edge_pairs[t])  # Index of the source node 源节点索引
                    curAdj[1][t // 2] = int(gnnGraph_A_train[j][i+c].edge_pairs[t + 1])  # Index of the target node 目的节点索引

                tensor_features = torch.tensor(gnnGraph_A_train[j][i+c].node_features)

                tag = gnnGraph_A_train[j][i+c].node_tags
                node_tag = torch.LongTensor(tag).view(-1, 1)
                node_tags = torch.zeros(gnnGraph_A_train[j][i+c].num_nodes, max_labels + 1)
                node_tags.scatter_(1, node_tag, 1)
                tensor_features = torch.cat([node_tags.type_as(tensor_features), tensor_features], 1)

                # label = gnnGraph_A_train[j][i+c].label
                # labels = []
                # for t in tag:
                #     labels.append(label)
                # label = torch.LongTensor(labels).view(-1, 1)
                # tensor_features = torch.cat([tensor_features, label], 1)

                x.append(tensor_features)
                edge_index.append(curAdj)

            tensor_features_all = torch.tensor(gnnGraph_A_train[7][i+c].node_features)

            tag_all = gnnGraph_A_train[7][i+c].node_tags
            node_tag_all = torch.LongTensor(tag_all).view(-1, 1)
            node_tags_all = torch.zeros(gnnGraph_A_train[7][i+c].num_nodes, max_labels + 1)
            node_tags_all.scatter_(1, node_tag_all, 1)
            tensor_features_all = torch.cat([node_tags_all.type_as(tensor_features_all), tensor_features_all], 1)

            # label = gnnGraph_A_train[7][i + c].label
            # labels = []
            # for t in tag_all:
            #     labels.append(label)
            # label = torch.LongTensor(labels).view(-1, 1)
            # tensor_features_all = torch.cat([tensor_features_all, label], 1)

            # tensor_features_all = torch.cat([tensor_features_all, torch.LongTensor(0)], 1)

            curAdj_all = torch.LongTensor(2, len(gnnGraph_A_train[7][i+c].edge_pairs) // 2)
            for t in range(0, len(gnnGraph_A_train[7][i+c].edge_pairs), 2):
                curAdj_all[0][t // 2] = int(gnnGraph_A_train[7][i+c].edge_pairs[t])  # Index of the source node 源节点索引
                curAdj_all[1][t // 2] = int(gnnGraph_A_train[7][i+c].edge_pairs[t + 1])  # Index of the target node 目的节点索引

            x.append(tensor_features_all)
            edge_index.append(curAdj_all)

            logits = model(x, edge_index, None)
            logits_batch = torch.cat((logits_batch, logits[-1]), 0)
        loss = BCEWithLogitsLoss()(logits_batch[1:].reshape(1, 10), torch.tensor([train_labels[i:i+10]], dtype=torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*10
    return total_loss / len_train

@torch.no_grad()
def test(dataloader, labels_true):

    model.eval()

    y_pred, y_true = [], []

    len_test = len(dataloader[0])

    total_loss = 0

    for i in range(0, len_test):
        x = []
        edge_index = []
        for j in range(0, 7):
            curAdj = torch.LongTensor(2, len(dataloader[j][i].edge_pairs) // 2)
            for t in range(0, len(dataloader[j][i].edge_pairs), 2):
                curAdj[0][t // 2] = int(dataloader[j][i].edge_pairs[t])  # Index of the source node 源节点索引
                curAdj[1][t // 2] = int(dataloader[j][i].edge_pairs[t + 1])  # Index of the target node 目的节点索引

            tensor_features = torch.tensor(dataloader[j][i].node_features)

            tag = dataloader[j][i].node_tags
            node_tag = torch.LongTensor(tag).view(-1, 1)
            node_tags = torch.zeros(dataloader[j][i].num_nodes, max_labels + 1)
            node_tags.scatter_(1, node_tag, 1)
            tensor_features = torch.cat([node_tags.type_as(tensor_features), tensor_features], 1)

            # label = dataloader[j][i].label
            # labels = []
            # for t in tag:
            #     labels.append(label)
            # label = torch.LongTensor(labels).view(-1, 1)
            # tensor_features = torch.cat([tensor_features, label], 1)

            # label = gnnGraph_A_train[j][i].label
            # label = torch.LongTensor(label).view(-1, 1)
            # tensor_features = torch.cat([tensor_features, label], 1)

            # p, q = dataloader[j][i].link
            # l = A_all[p, q] / 7
            # tensor_features = tensor_features * l
            # label = gnnGraph_A_train[j][i].label
            # if label == 0:
            #     tensor_features = tensor_features * 0.1*j
            # else:
            #     tensor_features = tensor_features*j

            x.append(tensor_features)
            edge_index.append(curAdj)

        tensor_features_all = torch.tensor(dataloader[7][i].node_features)

        tag_all = dataloader[7][i].node_tags
        node_tag_all = torch.LongTensor(tag_all).view(-1, 1)
        node_tags_all = torch.zeros(dataloader[7][i].num_nodes, max_labels + 1)
        node_tags_all.scatter_(1, node_tag_all, 1)
        tensor_features_all = torch.cat([node_tags_all.type_as(tensor_features_all), tensor_features_all], 1)

        # label = dataloader[7][i].label
        # labels = []
        # for t in tag_all:
        #     labels.append(label)
        # label = torch.LongTensor(labels).view(-1, 1)
        # tensor_features_all = torch.cat([tensor_features_all, label], 1)

        # tensor_features_all = torch.cat([tensor_features_all, torch.LongTensor(0)], 1)

        # p, q = dataloader[7][i].link
        # l = A_all[p, q] / 7
        # tensor_features_all = tensor_features_all * l
        # label = dataloader[7][i].label
        # if label == 0:
        #     tensor_features_all = tensor_features_all * 0.1

        curAdj_all = torch.LongTensor(2, len(dataloader[7][i].edge_pairs) // 2)
        for t in range(0, len(dataloader[7][i].edge_pairs), 2):
            curAdj_all[0][t // 2] = int(dataloader[7][i].edge_pairs[t])  # Index of the source node 源节点索引
            curAdj_all[1][t // 2] = int(dataloader[7][i].edge_pairs[t + 1])  # Index of the target node 目的节点索引

        x.append(tensor_features_all)
        edge_index.append(curAdj_all)

        logits = model(x, edge_index, None)
        y_pred.append(logits.view(-1).cpu())
        y = torch.tensor([labels_true[i]], dtype=torch.float)
        y_true.append(y.view(-1).cpu().to(torch.float))

        loss = BCEWithLogitsLoss()(logits.view(-1),
                                   torch.tensor([labels_true[i]], dtype=torch.float))
        total_loss += loss

    # return roc_auc _score(torch.cat(y_true), torch.cat(y_pred))
    return roc_auc_score(torch.cat(y_true), torch.cat(y_pred)), average_precision_score(torch.cat(y_true), torch.cat(y_pred)), mean_absolute_error(torch.cat(y_true), torch.cat(y_pred)), total_loss/len_test


stopper = EarlyStopping()

best_test_auc = test_auc = 0
for epoch in range(1, 101):
    loss = train()
    val_auc, val_ap, val_rk, val_loss = test(gnnGraph_A_val, val_labels)

    early_stop = stopper.step(val_loss, val_auc, model, "23Enron")

    if test_auc > best_test_auc:
        best_test_auc = test_auc
        # test_auc, test_ap = test(gnnGraph_A_test, test_labels)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, val_loss:{val_loss:.4f}, val_auc: {val_auc:.4f}')
    if early_stop:
        break

print("----------------")
stopper.load_checkpoint(model, "23Enron")
test_auc, test_ap, test_rk, test_loss = test(gnnGraph_A_test, test_labels)
print(f'Test_auc: {test_auc:.4f}, Test_ap: {test_ap:.4f},Test_rk: {test_rk:.4f}, Test_loss: {test_loss:.4f}')
val_auc, val_ap, val_rk, val_loss = test(gnnGraph_A_val, val_labels)
print(f'val_auc: {val_auc:.4f}, val_ap: {val_ap:.4f},val_rk: {val_rk:.4f}, val_loss: {val_loss:.4f}')


# best_val_auc = test_auc = 0
# best_val_ap = test_ap = 0
# best_val_loss = best_val_loss_auc = best_val_loss_ap = 1
# for epoch in range(1, 31):
#     loss = train()
#     test_auc, test_ap = test()
#     if test_auc > best_val_auc:
#         best_val_auc = test_auc
#     if test_ap > best_val_ap:
#         best_val_ap = test_ap
#         # torch.save(model.state_dict(), "model.pt")
#
#     if loss < best_val_loss:
#         best_val_loss = loss
#         best_val_loss_auc = test_auc
#         best_val_loss_ap = test_ap
#         # test_auc = test(test_loader)
#     print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, test_auc: {test_auc:.4f}, test_ap: {test_ap:.4f},')
# print("上面结果中最好的auc为：")
# print(f'best_test_auc: {best_val_auc:.4f},')
# print("上面结果中最好的loss情况为为：")
# print(f'best_test_loss: {best_val_loss:.4f}, best_val_loss_auc: {best_val_loss_auc:.4f}, best_val_loss_ap: {best_val_loss_ap:.4f}')

# m_state_dict = torch.load('model.pt')
# new_model = DGCNN(32, 2).to(device)
# new_model.load_state_dict(m_state_dict)
# auc = []
# ap = []
# for i in range(0, 20):
#     test_auc, test_ap = test(model)
#     auc.append(test_auc)
#     ap.append(test_ap)
# print(mean(auc))
# print(mean(ap))
