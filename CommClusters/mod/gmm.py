import torch
import torch.nn as nn
import numpy as np
from CommClusters.mod.sel import *
from sklearn.datasets import make_spd_matrix


class gMM():

    def __init__(self, K, dims, eps=5e-3):
        super(gMM, self).__init__()
        self.means = None
        self.vars = torch.cat([torch.FloatTensor(make_spd_matrix(dims)).unsqueeze(0) for _ in range(K)], dim=0)
        self.k = K
        self.eps = eps
        self.dims = dims
        self.pi = torch.FloatTensor([1/self.k for _ in range(self.k)])

    def random_initialize(self, x):
        idxs = np.random.choice(len(x), size=(self.k,), replace=False)
        self.means = x[idxs].unsqueeze(0)
        self.vars = self.covariance(x)

    def seed_initialize(self, x, mu):
        self.means = mu.unsqueeze(0)
        self.vars = self.covariance(x)

    def prob(self, x):
        N = torch.distributions.MultivariateNormal(self.means, self.vars)
        # return torch.exp(N.log_prob(x).T)
        return torch.exp(N.log_prob(x).T)

    def predict(self,x):
        P = self.pi.unsqueeze(-1) * self.prob(x)
        P = P / (P.sum(dim=0) + self.eps)
        return P

    def fit(self, x, epochs=1, mu=[]):
        if len(mu) > 0:
            self.seed_initialize(x, mu)
        else:
            self.random_initialize(x)

        for ep in range(epochs):

            # E-step
            l = self.predict(x.unsqueeze(1))

            # M-step
            # calculating r-matrix
            r = l * self.pi.unsqueeze(-1)
            r = r/(r.sum(dim=0) + self.eps)

            #Calculating means
            self.means = (r.unsqueeze(-1) * x).sum(dim=1) / (r.sum(dim=-1).sum(dim=-1).view(-1,1) + self.eps)
            self.means = self.means.unsqueeze(0)
            #calculating update to covariance matrices
            self.vars = self.covariance(x)

            #Soft updating weights/pi per training epoch
            self.pi = self.pi + (self.eps * (r.sum(dim=-1)/(r.sum() + self.eps)))
            self.pi = self.pi/self.pi.sum()
            # print(ep, self.pi, '\n===][===')

            #WORKING ON ALGO from https://colab.research.google.com/drive/1Eb-G95_dd3XJ-0hm2qDqdtqMugLkSYE8#scrollTo=DrsHNw9L5fHc

    def covariance(self, x):
        l = self.predict(x.unsqueeze(1)) * self.pi.unsqueeze(-1)
        l = (l/l.sum(dim=0)).unsqueeze(-1)
        E = (x.unsqueeze(1) - self.means).transpose(0,1)
        covar = (((l*E).transpose(-1,-2) @ E) / (l.sum(dim=1).unsqueeze(-1))) / (self.dims)
        return torch.clamp(covar, min=self.eps)


class dGMM():

    def __init__(self,K, dims, eps=5e-3, lr=5e-3):
        super(dGMM, self).__init__()
        self.k = K
        self.dims = dims
        self.eps = eps
        self.lr = lr

        self.means = None
        self.vars = None
        self.pi = torch.FloatTensor([1/self.k for _ in range(self.k)])


    #################################################################
    ##### initialize values from scratch
    #################################################################
    def random_initialize(self, x):
        idxs = np.random.choice(len(x), size=(self.k,), replace=False)
        self.means = x[idxs]
        E = (x.unsqueeze(1) - self.means) ** 2
        self.vars = torch.rand(size=(self.k,self.dims)) * E.mean(dim=0)

    def seed_initialize(self, x, mu):
        self.means = mu
        E = (x.unsqueeze(1) - self.means) ** 2
        self.vars = torch.rand(size=mu.shape) * E.mean(dim=0) #(1/(x.shape[0]-1)) * E.sum(dim=0)


    #################################################################
    ##### Probability and covariance calculations
    #################################################################
    def likelihood(self, x):
        N = torch.distributions.Normal(self.means, self.vars)
        return torch.exp(N.log_prob(x.unsqueeze(1)))

    def covariance(self, x):
        posterior = self.likelihood(x) * self.pi.view(1,-1,1)
        posterior = posterior / (posterior.sum(dim=0) + self.eps)

        #Updating covariance by normal means was leading to instability. Instead,
        # I implement a bastard GIBBS samper to update covariance over time.
        #(1) Calculate Error
        E = (x.unsqueeze(1) - self.means)

        #(2) Calculate the directionality of error vec
        DIR = torch.ones(size=E.shape)
        DIR = DIR * ((E < 0).float() * -1)

        #(2) Find update amount
        r = (E**2) * DIR

        #(3) Update covariance by update amount * lr
        self.vars = self.vars - (self.lr * (1/(r.shape[0]-1)) * r.sum(dim=0))


    #################################################################
    ##### Fit model and predict outputs
    #################################################################
    def fit(self, x, epochs=1, mu=[]):
        if len(mu) > 0:
            self.seed_initialize(x, mu)
        else:
            self.random_initialize(x)

        for ep in range(epochs):

            ####### E-STEP #######
            l = self.likelihood(x)

            ####### M-STEP #######
            r = l * self.pi.unsqueeze(-1)
            r = r/(r.sum(dim=-1).unsqueeze(-1) + self.eps)

            self.means = (r * x.unsqueeze(1)).sum(dim=0) / (r.sum(dim=0) + self.eps)
            self.covariance(x)

            self.pi = self.pi + (self.lr * r.sum(dim=0).sum(dim=-1) / (r.sum() + self.eps))
            self.pi = self.pi/self.pi.sum()

    def predict(self, x):
        l = self.likelihood(x).sum(dim=-1)
        return l #(l/l.sum(dim=0))


    #################################################################
    ##### Save and load previous model versions
    #################################################################
    def save_weights(self, file):
        torch.save({'k': self.k,
                    'dims': self.dims,
                    'eps' : self.eps,
                    'lr': self.lr,
                    'mu': self.means,
                    'covar': self.vars,
                    'pi': self.pi}, file)

    def load_weights(self, file):
        m = torch.load(file)
        self.k, self.dims, self.eps, self.lr = m['k'], m['dims'], m['eps'], m['lr']
        self.means, self.vars, self.pi = m['mu'], m['covar'], m['pi']















