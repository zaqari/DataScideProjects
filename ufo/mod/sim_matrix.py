import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from CommClusters.mod.dataplot import *

class probFn(nn.Module):

    def __init__(self, sigma):
        super(probFn,self).__init__()
        self.dist = torch.distributions.HalfNormal(sigma, validate_args=False)
        self.cos = nn.CosineSimilarity(dim=-1)

    def p(self, x):
        return torch.exp(self.dist.log_prob(x))

    def cosE(self, x, y):
        return 1. - self.cos(x.unsqueeze(1), y)

    def PROB(self, x, y):
        return self.p(self.cosE(x,y))

def prob(x,sigma=.5):
    P = torch.distributions.HalfNormal(sigma, validate_args=False)
    return torch.exp(P.log_prob(x))

def matrix(x, sigma=.5):
    cos = nn.CosineSimilarity(dim=-1)
    """
    x: a list of inputs
    return: a similarity matrix of shape denom x num x denom
    """
    k = torch.FloatTensor([[prob(1. - cos(r.unsqueeze(1), j).mean(), sigma=sigma) for j in x] for r in x])
    denom = (k * torch.eye(len(k))).sum(dim=-1)
    return (k/denom)

def med_matrix(x, sigma=.5):
    cos = nn.CosineSimilarity(dim=-1)
    """
    x: a list of inputs
    return: a similarity matrix of shape denom x num x denom
    """
    k = torch.FloatTensor([[prob((1. - cos(r.unsqueeze(1), j)).mean(), sigma=sigma) for j in x] for r in x])
    denom = (k * torch.eye(len(k))).sum(dim=-1)
    return (k/denom)

def sel(terms,IDX):
    return (IDX == np.array(terms).reshape(-1,1)).sum(axis=0).astype(np.bool)






###########################################################################
########### Experimental Functions
###########################################################################
def gaus_matrix(x):
    n = [(j @ j.T).sum(dim=-1) for j in x]
    n = [torch.distributions.Normal(mu.unsqueeze(-1),mu.std()) for mu in n]
    X = [[(i @ j.T).sum(dim=-1) for i in x] for j in x]
    resp = torch.FloatTensor([[torch.exp(n[i].log_prob(v.unsqueeze(-1))).sum() for i,v in enumerate(Xi)] for Xi in X])
    return resp

def prob_(x,sigma=.5):
    P = torch.distributions.Normal(1, sigma, validate_args=False)
    return torch.exp(P.log_prob(x))/torch.exp(P.log_prob(torch.FloatTensor([1])))

def matrix_(x, sigma=.5):
    cos = nn.CosineSimilarity(dim=-1)
    """
    x: a list of inputs
    return: a similarity matrix of shape denom x num x denom
    """
    k = torch.FloatTensor([[prob(cos(r.unsqueeze(1), j).mean(), sigma=sigma) for j in x] for r in x])
    denom = (k * torch.eye(len(k))).sum(dim=-1)
    return (k/denom)