import numpy as np
import pandas as pd
import torch
import scipy.io
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix, csc_matrix
# from itertools import product
import random
from sklearn.model_selection import train_test_split
import time
from ANF import Affinity_Network_Fusion
from scipy.sparse.linalg import spsolve
from sklearn.metrics import pairwise_distances, roc_curve, auc
from scipy.sparse.csgraph import laplacian

# from tqdm import tqdm

device = torch.device('cpu')



#%% MHGCN (Comparison Model)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        try:
            input = input.float()
        except:
            pass
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class MHGCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(MHGCN, self).__init__()
        """
        # Multilayer Graph Convolution
        """
        self.gc1 = GraphConvolution(nfeat, out)
        self.gc2 = GraphConvolution(out, out)
        self.dropout = dropout

        """
        Set the trainable weight of adjacency matrix aggregation
        """

        # MIPS
        self.weight_b = torch.nn.Parameter(torch.FloatTensor(5, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)



    def forward(self, feature, A, use_relu=True):
        final_A = adj_matrix_weight_merge(A, self.weight_b)

        try:
            feature = torch.tensor(feature.astype(float).toarray())
        except:
            try:
                feature = torch.from_numpy(feature.toarray())
            except:
                pass

        # Output of single-layer GCN
        U1 = self.gc1(feature, final_A)
        # Output of two-layer GCN
        U2 = self.gc2(U1, final_A)

        # Average pooling
        return (U1+U2)/2


#%% MHGCN Preprocessing

def coototensor(A):
    """
    Convert a coo_matrix to a torch sparse tensor
    """

    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def adj_matrix_weight_merge(A, adj_weight):
    """
    Multiplex Relation Aggregation
    """

    # imdb
    a = coototensor(A[0].tocoo())
    b = coototensor(A[1].tocoo())
    c = coototensor(A[2].tocoo())
    d = coototensor(A[3].tocoo())
    e = coototensor(A[4].tocoo())
    A_t = torch.stack([a, b, c, d, e], dim=2).to_dense()

    temp = torch.matmul(A_t, adj_weight)
    temp = torch.squeeze(temp, 2)

    return temp + temp.transpose(0, 1)


def graph_construct(data, sigma = 1, neighbor = 15):
    dist = pairwise_distances(data, metric = 'euclidean')
    W_temp = np.exp(-(dist **2) / sigma)
    
    A = kneighbors_graph(data, neighbor, mode='connectivity', include_self = True)
    A = 0.5 * (A + A.T)
    A = A.toarray()
    W = W_temp * A
    L = laplacian(W, normed = True)
    
    return L

def graph_SSL(L, y, mu = 1):
    I = csc_matrix(np.eye(len(y)))
    f = spsolve((I + mu*L), y)
    return f

#%% Fast Graph Integration (Proposed Model)

def labeled_Laplacian(L, labeled):
    L_labeled = L[labeled][:, labeled]
    
    return L_labeled 

def softmax(alpha):
    return np.exp(alpha)/sum(np.exp(alpha))

def SGF(L, y, labeled):
    '''
    L have to be 3-d Array!!
    '''
    L_labeled = []
    num_graph = L.shape[0]
    y = y.copy()[labeled]
    
    
    for i in range(num_graph):
        L_labeled.append(labeled_Laplacian(L[i], labeled))
    
    L_labeled = np.array(L_labeled)
    
    trm = np.zeros((num_graph, num_graph))
    for i in range(num_graph - 1):
        for j in range(i + 1, num_graph):
            trm[i, j] = (L_labeled[i] @ L_labeled[j]).diagonal().sum()
            
    trm = trm + trm.T
    
    for i in range(num_graph):
        trm[i, i] = (L_labeled[i] @ L_labeled[i]).diagonal().sum()
    
    trm = csc_matrix(trm)
    
    smoothness = np.zeros((num_graph, 1))
    for i in range(num_graph):
        # smoothness[i] = num_node - ((y.T @ L[i] @ y) / num_node)
        smoothness[i] = len(labeled) - (y.T @ L_labeled[i] @ y)
    smoothness = csc_matrix(smoothness)
    
    # alpha = inv(trm) @ smoothness
    alpha = spsolve(trm, smoothness)
    # alpha = softmax(alpha.toarray())
    return alpha 

def graph_integration(L, alpha):
    
    num_graph = L.shape[0]
    num_node = L[0].shape[1]
    
    L_integration = np.zeros((num_node, num_node))
    for i in range(num_graph):
        L_integration = L_integration + alpha[i]*L[i]
        
    return csc_matrix(L_integration)


#%% Data Load

title = 'MIPS'

directory = 'C:/Users/YTH/Desktop/EXP_SGF/data/' # Change Directory

mat_file_name = 'protein_mod_v1.mat' # Chnage Dataset Name
mat_file = scipy.io.loadmat(directory + mat_file_name)

W_total_temp = mat_file['W_Total']
W_total = []

for i in range(W_total_temp.shape[1]):
    W_total.append(csc_matrix(W_total_temp[0][i]))
    
W_total = np.array(W_total)
    
L_total_temp = mat_file['L_Total']

L_total = []

for i in range(L_total_temp.shape[1]):
    L_total.append(csc_matrix(L_total_temp[0][i]))

L_total = np.array(L_total)

label = mat_file['y']
num_graph = L_total.shape[0]
num_class = label.shape[1]
# num_class = 1

del mat_file
del W_total_temp
del L_total_temp

#%% ANF (Comparison Model)

# Time_ANF = pd.DataFrame(index = range(1, 1 + 1), columns = range(num_class))

# alpha_ANF = 0.5

# start = time.time()
# W_ANF = Affinity_Network_Fusion(W_total, alpha_ANF)
# Time_ANF.iloc[0, :] = time.time() - start

# L_ANF = csc_matrix(laplacian(W_ANF, normed = True))

# Time_ANF.to_excel('C:/Users/YTH/Desktop/EXP_SGF/EXP_MHGCN/MIPS/' + '{}_ANF_Time.xlsx'.format(title))


#%% Preprocessing

A1 = W_total[0].tocoo()
A2 = W_total[1].tocoo()
A3 = W_total[2].tocoo()
A4 = W_total[3].tocoo()
A5 = W_total[4].tocoo()

num_node = A1.shape[0]

feature = coo_matrix(np.eye(num_node))

A_star = [A1, A2, A3, A4, A5]

true_edges = []
false_edges = []

A_full = coo_matrix(np.ones((num_node, num_node)) - np.eye(num_node))
all_possible_pairs = set(zip(A_full.row, A_full.col))

for A in A_star:
    
    positive_pairs = set(zip(A.row, A.col))
    true_edges.extend(positive_pairs)
    
    negative_pairs = all_possible_pairs - positive_pairs
    false_edges.extend(negative_pairs)
    
del W_total
del A_full
del all_possible_pairs

#%% Train MHGCN

NUM_POS = 30
NUM_NEG = 30

NUM_EPOCH = 100

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0005
NUM_REP = 30
# NUM_REP = 1

NHID = 384
OUT = 200


NEIGHBOR = 15

def train_MHGCN():
    
    mhgcn = MHGCN(nfeat = feature.shape[1], nhid = NHID, out = OUT, dropout = 0)
    
    for epoch in range(1, NUM_EPOCH + 1):
        
        print('epoch {}'.format(epoch))
        
        mhgcn.to(device)
        opt = torch.optim.Adam(mhgcn.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        emb = mhgcn(feature, A_star)
        
        emb_true_first = []
        emb_true_second = []
        emb_false_first = []
        emb_false_second = []
        
        pos_samples = random.sample(true_edges, k=NUM_POS)
        neg_samples = random.sample(false_edges, k=NUM_NEG)
        
        for edge in pos_samples:
            emb_true_first.append(emb[int(edge[0])])
            emb_true_second.append(emb[int(edge[1])])

        for edge in neg_samples:
            emb_false_first.append(emb[int(edge[0])])
            emb_false_second.append(emb[int(edge[1])])
        
        emb_true_first = torch.cat(emb_true_first).reshape(-1, OUT)
        emb_true_second = torch.cat(emb_true_second).reshape(-1, OUT)
        emb_false_first = torch.cat(emb_false_first).reshape(-1, OUT)
        emb_false_second = torch.cat(emb_false_second ).reshape(-1, OUT)
        
        T1 = emb_true_first @ emb_true_second.T
        T2 = -(emb_false_first @ emb_false_second.T)
        
        pos_out = torch.diag(T1)
        neg_out = torch.diag(T2)
        
        loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))
        loss = loss.requires_grad_()

        opt.zero_grad()
        loss.backward()
        opt.step()
        
    return mhgcn

#%%

ratio = 0.2

AUC_pro = pd.DataFrame(index = range(1, NUM_REP + 1), columns = range(num_class))
AUC_MHGCN = pd.DataFrame(index = range(1, NUM_REP + 1), columns = range(num_class))
AUC_ANF = pd.DataFrame(index = range(1, NUM_REP + 1), columns = range(num_class))

Time_pro = pd.DataFrame(index = range(1, NUM_REP + 1), columns = range(num_class))
Time_MHGCN = pd.DataFrame(index = range(1, NUM_REP + 1), columns = range(num_class))


#%% Comparison

for i in range(NUM_REP):
    start = time.time()
    mhgcn = train_MHGCN()
    mhgcn.eval()
    H = mhgcn(feature, A_star).detach().numpy()
    L_MHGCN = csc_matrix(graph_construct(data = H, sigma = 1, neighbor = NEIGHBOR))
    Time_MHGCN.iloc[i, :] = time.time() - start
    
    for j in range(num_class):
        
        target = label[:, j]
        print("Class: {}   /   Iteration: {}".format(j, i + 1))
        
        y = target.copy()
        labeled, unlabeled = train_test_split(range(len(y)), train_size = ratio,
                                              shuffle = True, stratify = y)
        y[unlabeled] = 0
        y_real = target[unlabeled]
    
        # Proposed Method
        print("Proposed Method Start!  Class: {} / Iteration: {}".format(j, i + 1))
        start = time.time()
        alpha_proposed = SGF(L_total, y, labeled)
        alpha_proposed = softmax(alpha_proposed)
        Time_pro.iloc[i, j] = time.time() - start
        
        L_proposed = graph_integration(L_total, alpha_proposed)
        f = graph_SSL(L_proposed, y, mu = 1)
        y_pred = f[unlabeled]
        fpr, tpr, thresholds = roc_curve(y_real, y_pred, pos_label=1)
        AUC_pro.iloc[i, j] = auc(fpr, tpr)
        
        # MHGCN
        print("MHGCN Start!  Class: {} / Iteration: {}".format(j, i + 1))
        f = graph_SSL(L_MHGCN, y, mu = 1)
        y_pred = f[unlabeled]
        fpr, tpr, thresholds = roc_curve(y_real, y_pred, pos_label=1)
        AUC_MHGCN.iloc[i, j] = auc(fpr, tpr)
        
        
        # ANF
        # print("ANF Method Start!  Class: {} / Interation: {}".format(j, i + 1))
        
        # f = graph_SSL(L_ANF, y, mu = 1)
        # y_pred = f[unlabeled]
        # fpr, tpr, thresholds = roc_curve(y_real, y_pred, pos_label=1)
        # AUC_ANF.iloc[i, j] = auc(fpr, tpr)
        
        
        
        
        
        directory = 'C:/Users/YTH/Desktop/EXP_SGF/EXP_MHGCN/{}/'.format(title)


        AUC_pro.to_excel(directory + '{}_proposed_AUC({}%, {}rep).xlsx'.format(title, ratio*100, NUM_REP))
        AUC_MHGCN.to_excel(directory + '{}_MHGCN_AUC({}%, {}rep).xlsx'.format(title, ratio*100, NUM_REP))
        # AUC_ANF.to_excel(directory + '{}_ANF_AUC({}%, {}rep).xlsx'.format(title, ratio*100, NUM_REP))

        Time_pro.to_excel(directory + '{}_proposed_Time({}%, {}rep).xlsx'.format(title, ratio*100, NUM_REP))
        Time_MHGCN.to_excel(directory + '{}_MHGCN_Time({}%, {}rep).xlsx'.format(title, ratio*100, NUM_REP))
            
    

#%%

AUC_pro.loc['Avg', :] = AUC_pro.mean()
AUC_pro.loc['Std', :] = AUC_pro.std() 

AUC_MHGCN.loc['Avg', :] = AUC_MHGCN.mean()
AUC_MHGCN.loc['Std', :] = AUC_MHGCN.std() 

# AUC_ANF.loc['Avg', :] = AUC_ANF.mean()
# AUC_ANF.loc['Std', :] = AUC_ANF.std() 

Time_pro.loc['Avg', :] = Time_pro.mean()
Time_pro.loc['Std', :] = Time_pro.std() 

Time_MHGCN.loc['Avg', :] = Time_MHGCN.mean()
Time_MHGCN.loc['Std', :] = Time_MHGCN.std() 

#%%


directory = 'C:/Users/YTH/Desktop/EXP_SGF/EXP_MHGCN/{}/'.format(title)


AUC_pro.to_excel(directory + '{}_proposed_AUC({}%, {}rep).xlsx'.format(title, ratio*100, NUM_REP))
AUC_MHGCN.to_excel(directory + '{}_MHGCN_AUC({}%, {}rep).xlsx'.format(title, ratio*100, NUM_REP))
# AUC_ANF.to_excel(directory + '{}_ANF_AUC({}%, {}rep).xlsx'.format(title, ratio*100, NUM_REP))

Time_pro.to_excel(directory + '{}_proposed_Time({}%, {}rep).xlsx'.format(title, ratio*100, NUM_REP))
Time_MHGCN.to_excel(directory + '{}_MHGCN_Time({}%, {}rep).xlsx'.format(title, ratio*100, NUM_REP))

