import torch
import torch.nn as nn

class agent(nn.Module):

    def __init__(self,dims):
        super(agent,self).__init__()
        self.dims = dims
        self.nn1 = nn.Linear(dims,100, bias=False)
        self.nn2 = nn.Linear(100,dims)
        self.act = nn.Tanh()

    def forward(self, centroids, x):
        loc = self.act(self.nn2(self.nn1(centroids)))
        loc = loc * centroids
        dist = torch.distributions.MultivariateNormal(
            loc = loc.unsqueeze(1),
            covariance_matrix = torch.eye(self.dims) * centroids.var()
        )
        return torch.exp(dist.log_prob(x)).T


class agent2(nn.Module):

    def __init__(self,dims):
        super(agent2,self).__init__()
        self.dims = dims
        self.nn1 = nn.Linear(dims,100)
        self.nn2 = nn.Linear(100,dims)

        self.nn3 = nn.Linear(dims, 100)
        self.nn4 = nn.Linear(100, dims)
        self.act = nn.ReLU()

    def forward(self, centroids, x):
        loc = self.act(self.nn2(self.nn1(centroids)))
        dist = torch.distributions.MultivariateNormal(
            loc = loc.unsqueeze(1),
            covariance_matrix = torch.eye(self.dims) * centroids.var()
        )
        return torch.exp(dist.log_prob(self.act(self.nn4(self.nn3(x))))).T


class RL(nn.Module):

    def __init__(self, agent, loss_fn, classes, memory_size=10):
        super(RL,self).__init__()
        self.cat_loss = loss_fn
        self.diff_loss = nn.CosineEmbeddingLoss()
        self.mod = agent
        self.memory = torch.zeros(size=(classes,memory_size,self.mod.c.shape[-1]))

    def replay(self):
        return torch.cat([self.mod(mem).unsqueeze(0) for mem in self.memory], dim=0)

    def fit_(self, X, Y, replace=False):
        x = self.mod(X)
        # diverge = self.diff_loss()
