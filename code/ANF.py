import numpy as np
from scipy.sparse import csc_matrix

def degree_matrix(A):
    return np.diag(A.sum(axis = 1))


def normalized_weight_matrix(W, normalization = False):
    
    if normalization:
        W = W.copy()
        np.fill_diagonal(W, 0)
        D = degree_matrix(W)
        P = (np.linalg.inv(D) @ W) / 2
        np.fill_diagonal(P, 1/2)
    
    else: 
        D = degree_matrix(W)
        P = np.linalg.inv(D) @ W
    
    return csc_matrix(P)



def Affinity_Network_Fusion(S_li, alpha):
    
    num_graph = len(S_li)
    num_node = S_li[0].shape[0]
    
    W_total = [normalized_weight_matrix(S.toarray() + np.eye(num_node)) for S in S_li]
    
    W_sum = np.zeros((num_node, num_node))
    
    for W in W_total:
        W_sum = W_sum + W
    
    
    W_new = []
    
    for W in W_total:
        W_other = (W_sum - W) / (num_graph - 1)
        W_temp = alpha*W@W_other+ (1-alpha)*W_other@W
        W_new.append(W_temp)
        
    
    W_new_sum = np.zeros((num_node, num_node))
    
    for W in W_new:
        W_new_sum = W_new_sum + W
        
    return (W_new_sum / num_graph)
    
    
    