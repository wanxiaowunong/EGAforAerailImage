import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy import sparse
from sklearn.neighbors import kneighbors_graph
import torch.nn.functional as F

def gen_Node(feature_map):
    x = feature_map.data[0]  # jiangwei  -->c*h*w
    x = x.transpose(0, 1).transpose(1, 2)  # -->h*c*w  h*w*c
    N = x.reshape([-1, x.shape[2]])  # reshape -->|h*w|*c
    return N



'''calculate adj'''
def gen_adj(feature_map):
    x = feature_map.data[0]  # jiangwei  -->c*h*w
    x = x.transpose(0, 1).transpose(1, 2)  # -->h*c*w  h*w*c
    x = x.reshape([-1, x.shape[2]])  # reshape -->|h*w|*c
    x1 = x.t()

    similiar = torch.mm(x,x1)  # dot product
    I = torch.eye(similiar.shape[0]).cuda()  # define I
    similiar=similiar+I # self concat

    D = torch.pow(similiar.sum(1).float(), -0.5)
    D = torch.diag(D)  # degree
    adj = torch.matmul(torch.matmul(similiar, D).t(), D)  # L=D^-0.5 * A * D^-0.5


    return adj
