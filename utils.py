import math
import random
import datetime

import networkx as nx
import scipy.sparse as ssp

import numpy as np
import torch
from gensim.models import Word2Vec
from sklearn import metrics
from tqdm import tqdm

from networkx import neighbors
import util
import node2vec

def read_data(name_pre, time_index, node_num, max_thres):
    '''
    Function to read the network snapshot of specific time slice
    读取特定时间片下的网络快照的函数
    :param name_pre: the name prefix of the data file 数据文件名称前缀
    :param time_index: index of time slice 时间片索引
    :param node_num: number of nodes in the dynamic network 动态网络中节点总数
    :param max_thres: threshold of the maximum edge weight 最大边权值的阈值
    :return: adjacency matrix of the specific time slice 指定时间片的邻接矩阵
    '''
	# 文件名： edge_list_0.txt
    print('Read network snapshot #%d'%(time_index))
    #Initialize the adjacency matrix 初始化邻接矩阵
    curAdj = np.mat(np.zeros((node_num, node_num)))
    #Read the network snapshot of current time slice 读取当前时间片的网络快照
    f = open('%s_%d.csv'%(name_pre, time_index))
    # f = open('.\\data\\Enron-employees\\m_enron_employees_8.csv')
    line = f.readline()
    while line:
        seq = line.split(',')
        #print(seq)
        src = int(seq[0]) - 1  #Index of the source node 源节点索引
        tar = int(seq[1]) - 1  #Index of the target node 目的节点索引

        #Update the adjacency matrix 更新邻接矩阵
        curAdj[src, tar] = 1
        curAdj[tar, src] = 1
        line = f.readline()
    f.close()
    return curAdj


def sample_neg(net, test_ratio=0.3, train_pos=None, test_pos=None, max_train_num=None,
               all_unknown_as_negative=False):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None and test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None and train_pos is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num = len(train_pos[0]) if train_pos else 0
    test_num = len(test_pos[0]) if test_pos else 0
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    if not all_unknown_as_negative:
        # sample a portion unknown links as train_negs and test_negs (no overlap)
        while len(neg[0]) < train_num + test_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg  = (neg[0][:train_num], neg[1][:train_num])
        test_neg = (neg[0][train_num:], neg[1][train_num:])
    else:
        # regard all unknown links as test_negs, sample a portion from them as train_negs
        while len(neg[0]) < train_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg  = (neg[0], neg[1])
        test_neg_i, test_neg_j, _ = ssp.find(ssp.triu(net==0, k=1))
        test_neg = (test_neg_i.tolist(), test_neg_j.tolist())
    return train_pos, train_neg, test_pos, test_neg

def AA(A, test_pos, test_neg):
    # Adamic-Adar score
    A_ = A / np.log(A.sum(axis=1))
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    return CalcAUC(sim, test_pos, test_neg)

def CN(A, test_pos, test_neg):
    # Common Neighbor score
    sim = A.dot(A)
    return CalcAUC(sim, test_pos, test_neg)

def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, val_pos, val_neg, h=1,
                    max_nodes_per_hop=None, node_information=None, no_parallel=False):
    # automatically select h from {1, 2}
    if h == 'auto':
        # split train into val_train and val_test
        _, _, val_test_pos, val_test_neg = sample_neg(A, 0.1)
        val_A = A.copy()
        val_A[val_test_pos[0], val_test_pos[1]] = 0
        val_A[val_test_pos[1], val_test_pos[0]] = 0
        val_auc_CN = CN(val_A, val_test_pos, val_test_neg)
        val_auc_AA = AA(val_A, val_test_pos, val_test_neg)
        print('\033[91mValidation AUC of AA is {}, CN is {}\033[0m'.format(
            val_auc_AA, val_auc_CN))
        if val_auc_AA <= val_auc_CN:
            h = 2
            print('\033[91mChoose h=2\033[0m')
        else:
            h = 1
            print('\033[91mChoose h=1\033[0m')

    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    max_num_nodes = {'value': 0}

    def helper(A, links, g_label):
        g_list = []
        if no_parallel:
            # for i, j in tqdm(zip(links[0], links[1])):
            for p in links:
                i = p[0]
                j = p[1]
                g, n_labels, n_features, nodes = subgraph_extraction_labeling(
                    (i, j), A, h, max_nodes_per_hop, node_information
                )
                g_label = int(A[i, j])
                max_num_nodes['value'] = max(max_num_nodes['value'], len(nodes))
                max_n_label['value'] = max(max(n_labels), max_n_label['value'])
                g_list.append(util.GNNGraph(g, g_label, n_labels, n_features, nodes, [i, j], node_labels=n_labels))
            return g_list
        # else:
        #     # the parallel extraction code
        #     start = time.time()
        #     pool = mp.Pool(mp.cpu_count())
        #     results = pool.map_async(
        #         parallel_worker,
        #         [((i, j), A, h, max_nodes_per_hop, node_information) for i, j in zip(links[0], links[1])]
        #     )
        #     remaining = results._number_left
        #     pbar = tqdm(total=remaining)
        #     while True:
        #         pbar.update(remaining - results._number_left)
        #         if results.ready(): break
        #         remaining = results._number_left
        #         time.sleep(1)
        #     results = results.get()
        #     pool.close()
        #     pbar.close()
        #     g_list = [util.GNNGraph(g, g_label, n_labels, n_features) for g, n_labels, n_features in results]
        #     max_n_label['value'] = max(
        #         max([max(n_labels) for _, n_labels, _ in results]), max_n_label['value']
        #     )
        #     end = time.time()
        #     print("Time eplased for subgraph extraction: {}s".format(end-start))
        #     return g_list

    print('Enclosing subgraph extraction begins...')
    train_graphs, test_graphs, val_graphs = None, None, None
    if train_pos and train_neg:
        train_graphs = helper(A, train_pos, 1) + helper(A, train_neg, 0)
    if test_pos and test_neg:
        test_graphs = helper(A, test_pos, 1) + helper(A, test_neg, 0)
    if val_pos and val_neg:
        val_graphs = helper(A, val_pos, 1) + helper(A, val_neg, 0)
    # else:
    #     if test_pos:
    #         test_graphs = helper(A, test_pos, 1)
    #     if train_pos:
    #         train_graphs = helper(A, train_pos, 1)
    return train_graphs, test_graphs, val_graphs, max_n_label['value'], max_num_nodes['value']

def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None,
                                 node_information=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes)
    # if len(nodes) > 70:
    #     nodes = nodes[:70]
    subgraph = A[nodes, :][:, nodes]
    # apply node-labeling
    labels = node_label(subgraph)
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph)
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)
    return g, labels.tolist(), features, nodes

def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    return labels

def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res

def generate_node2vec_embeddings(A, emd_size=128, negative_injection=False, train_neg=None):
    if negative_injection:
        row, col = train_neg
        A = A.copy()
        A[row, col] = 1  # inject negative train
        A[col, row] = 1  # inject negative train
    nx_G = nx.from_scipy_sparse_matrix(A)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]
    # model = Word2Vec(walks, size=emd_size, window=10, min_count=0, sg=1,
    #         workers=8, iter=1)
    model = Word2Vec(walks, vector_size=emd_size, window=10, min_count=0, sg=1,
                     workers=8, epochs=1)
    wv = model.wv
    embeddings = np.zeros([A.shape[0], emd_size], dtype='float32')
    sum_embeddings = 0
    empty_list = []
    for i in range(A.shape[0]):
        if str(i) in wv:
            embeddings[i] = wv.word_vec(str(i))
            sum_embeddings += embeddings[i]
        else:
            empty_list.append(i)
    mean_embedding = sum_embeddings / (A.shape[0] - len(empty_list))
    embeddings[empty_list] = mean_embedding
    return embeddings

def gen_features(A, use_embedding, use_attribute, attributes, train_neg):
    if use_embedding:
        embeddings = generate_node2vec_embeddings(A, 128, True, train_neg)
        node_information = embeddings
    if use_attribute and attributes is not None:
        if node_information is not None:
            node_information = np.concatenate([node_information, attributes], axis=1)
        else:
            node_information = attributes
    return node_information


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model, name):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model, name)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model, name)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model, name):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), 'results/{}'.format(name+self.filename))

    def load_checkpoint(self, model, name):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load('results/{}'.format(name+self.filename)))
# class EarlyStopping(object):
#     def __init__(self, patience=10):
#         dt = datetime.datetime.now()
#         self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(dt.date(), dt.hour, dt.minute, dt.second)
#         self.patience = patience
#         self.counter = 0
#         self.best_acc = None
#         self.best_loss = None
#         self.early_stop = False
#
#     def step(self, loss, model, name):
#         if self.best_loss is None:
#             # self.best_acc = acc
#             self.best_loss = loss
#             self.save_checkpoint(model, name)
#         elif loss >= self.best_loss:
#             self.counter += 1
#             # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             if loss < self.best_loss:
#                 self.save_checkpoint(model, name)
#             self.best_loss = np.min((loss, self.best_loss))
#             # self.best_acc = np.max((acc, self.best_acc))
#             self.counter = 0
#         return self.early_stop
#
#     def save_checkpoint(self, model, name):
#         """Saves model when validation loss decreases."""
#         torch.save(model.state_dict(), 'results/{}'.format(name+self.filename))
#
#     def load_checkpoint(self, model, name):
#         """Load the latest checkpoint."""
#         model.load_state_dict(torch.load('results/{}'.format(name+self.filename)))
