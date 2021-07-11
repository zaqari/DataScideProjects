import torch
import torch.nn as nn
import numpy as np
from CommClusters.mod.sel import *
from sklearn.datasets import make_spd_matrix


class GMM():

    def __init__(self, k, dims, eps=1e-3, lr=1e-3):
        super(GMM, self).__init__()
        self.k = k
        self.dims = dims
        self.eps = eps
        self.lr = lr
        self.training_runs = 0

        self.means = torch.rand(size=(k,dims))
        self.vars = torch.cat([torch.FloatTensor(make_spd_matrix(dims)).unsqueeze(0) for _ in range(k)], dim=0)
        self.pred_vars = None
        self.pi = torch.FloatTensor([1/k for _ in range(k)])

    def _init_random(self, x):
        idxs = np.random.choice(len(x), size=(self.k,), replace=False)
        self.means = x[idxs]

    def _init_seeded(self, mu):
        self.means = mu

    def likelihood(self, x):
        N = torch.distributions.MultivariateNormal(self.means, self.vars)
        return torch.exp(N.log_prob(x.unsqueeze(1)))

    def fit(self, x, mu=[], epochs=1):
        if len(mu) > 0:
            self._init_seeded(mu)
        else:
            self._init_random(x)

        for ep in range(epochs):
            try:
                #Likelihood = E-Step
                #calculate responsibility matrix
                l = self.likelihood(x) + self.eps

                r = (l * self.pi.view(1, -1))
                #Try this first
                # r = (r/(r.sum(dim=0) + self.eps))
                #Else try
                r = r / (r.sum(dim=-1) + self.eps).unsqueeze(-1)

                #update Means
                self.means = (r.unsqueeze(-1) * x.unsqueeze(1)).sum(dim=0) / (r.sum(dim=0) + self.eps).unsqueeze(-1)

                #Update covariance
                E = x.unsqueeze(1) - self.means
                self.pred_vars = self.vars.clone()
                self.vars = ((r.unsqueeze(-1) * E) + self.eps).transpose(0,1).transpose(-1,-2) @ (E + self.eps).transpose(0,1)
                self.vars = self.vars / (r.sum(dim=0) + self.eps).view(-1,1,1)

                #Update pi
                self.pi = (r.mean(dim=0) + self.eps)
                self.training_runs += 1

                if self.training_runs == epochs:
                    self.pred_vars = self.vars

            except RuntimeError:
                break
            except ValueError:
                break

    def predict(self, x):
        N = torch.distributions.MultivariateNormal(self.means, self.pred_vars)
        l = torch.exp(N.log_prob(x.unsqueeze(1)))
        return l/l.sum(dim=-1).unsqueeze(-1)

class gmm():

    def __init__(self, k, dims, eps=1e-3, lr=1e-3, covariance_scaling=1):
        super(gmm, self).__init__()
        self.k = k
        self.dims = dims
        self.eps = eps
        self.lr = lr
        self.training_runs = 0
        self.continuous_fitting_runs = 0

        self.means = torch.rand(size=(k,dims))
        self.vars = torch.cat([torch.FloatTensor(make_spd_matrix(dims)[0]**2).unsqueeze(0) for _ in range(k)], dim=0)
        print('original', self.vars)
        self.pred_vars = None
        self.pred_means = None
        self.pi = torch.FloatTensor([1/k for _ in range(k)])

    def _init_random(self, x):
        idxs = np.random.choice(len(x), size=(self.k,), replace=False)
        self.means = x[idxs]

    def _init_seeded(self, mu):
        self.means = mu + (torch.rand(size=(mu.shape))*2)

    def likelihood(self, x, mu, var):
        N = torch.distributions.MultivariateNormal(mu, scale_tril=torch.diag(var), validate_args=False)
        return torch.exp(N.log_prob(x))
        # return (N.log_prob(x))

    def EMstep(self, x, mu, var):
        #### (1) E-Step
        l = torch.cat([self.likelihood(x, mu[i], var[i]).unsqueeze(0) for i in range(self.k)], dim=0)

        #### (2) M-Step
        # Responsibility matrix
        r = (l) * (self.pi.view(-1, 1))
        r = r / (r.sum(dim=0) + self.eps)

        ## (2.1) Update means
        self.means = torch.cat(
            [((r[i].unsqueeze(-1) * x).sum(dim=0) / (r[i].sum() + self.eps)).unsqueeze(0) for i in range(self.k)],
            dim=0)
        # never allow loc to hit zero per pytorch's draconian rules.
        # mu += ((mu == 0.).float() * self.eps)

        ## (2.2) Update covariance
        # Error calculation
        E = torch.cat([(x - mu[i]).unsqueeze(0) for i in range(self.k)], dim=0)

        # Update covariance
        var = (r.unsqueeze(-1) * E) * E
        # never allow covariance to hit zero per pytorch's draconian rules.
        self.vars = torch.cat([(var[i].sum(dim=0) / (r[i].sum() + self.eps)).unsqueeze(0) for i in range(self.k)], dim=0)
        # Update priors on cluster inclusion
        # self.pi = r.mean(dim=-1)

        #### (null) Update training counter
        self.training_runs += 1

    def fit(self, x, mu=[], epochs=1):

        if len(mu) > 0:
            self._init_seeded(mu)
        else:
            self._init_random(x)

        for ep in range(epochs):
            try:
                self.EMstep(x, self.means, self.vars)


            except RuntimeError:
                print('RunTime ERR')
                self.vars = self.vars + torch.rand(size=self.vars.shape)
                if torch.isnan(self.means.sum()):
                    self.means = self.means.nan_to_num(nan=self.eps,posinf=self.eps, neging=-self.eps)
                self.EMstep(x, self.means, self.vars)
                self.continuous_fitting_runs +=1

            except ValueError:
                print('Value ERR')
                self.vars = self.vars + torch.rand(size=self.vars.shape)
                if torch.isnan(self.means.sum()):
                    self.means = self.means.nan_to_num(nan=self.eps, posinf=self.eps, neging=-self.eps)
                self.EMstep(x, self.means, self.vars)
                self.continuous_fitting_runs += 1

    def predict(self, x):
        return torch.cat([self.likelihood(x, self.pred_means[i], self.pred_vars[i]).view(1, -1) for i in range(self.k)], dim=0)


class gmm_():

    def __init__(self, k, dims, eps=1e-3, lr=1e-3, covariance_scaling=1):
        super(gmm_, self).__init__()
        self.k = k
        self.dims = dims
        self.eps = eps
        self.lr = lr
        self.training_runs = 0
        self.continuous_fitting_runs = 0

        self.means = torch.rand(size=(k,dims))
        self.vars = torch.cat([torch.FloatTensor(make_spd_matrix(dims)).unsqueeze(0) for _ in range(k)], dim=0) * covariance_scaling
        self.pred_vars = None
        self.pred_means = None
        self.pi = torch.FloatTensor([1/k for _ in range(k)])

    def _init_random(self, x):
        idxs = np.random.choice(len(x), size=(self.k,), replace=False)
        self.means = x[idxs]

    def _init_seeded(self, mu):
        self.means = mu

    def likelihood(self, x, mu, var):
        N = torch.distributions.MultivariateNormal(mu, var)
        return torch.exp(N.log_prob(x))
        # return (N.log_prob(x))

    def fit(self, x, mu=[], epochs=1):

        if len(mu) > 0:
            self._init_seeded(mu)
        else:
            self._init_random(x)

        for ep in range(epochs):
            try:
                #### (1) E-Step
                l = torch.cat([self.likelihood(x, self.means[i], self.vars[i]).view(1, -1) for i in range(self.k)], dim=0) + self.eps

                #### (2) M-Step
                r = (l * self.pi.view(-1,1))
                r = r/(r.sum(dim=-1)).unsqueeze(-1)

                ## (2.1) Update means
                self.pred_means = self.means.clone()
                self.means = (torch.cat([((r[i].unsqueeze(-1) * x).sum(dim=0) / (r[i].sum()+self.eps)).unsqueeze(0) for i in range(self.k)], dim=0))


                ## (2.2) Update covariance
                #Error calculation
                E = torch.cat([(x - self.means[i]).unsqueeze(0) for i in range(self.k)], dim=0)
                #Save previous covariances
                self.pred_vars = self.vars.clone()
                #Update covariance
                self.vars = torch.cat([(((r[i].unsqueeze(-1) * E[i]).T @ E[i]) / (r[i].sum()+self.eps)).unsqueeze(0) for i in range(self.k)], dim=0)

                #Update priors on cluster inclusion
                self.pi = torch.cat([r[i].mean().view(-1) for i in range(self.k)], dim=-1).view(-1)

                #### (null) Update training counter
                self.training_runs += 1


            except RuntimeError:
                print('ERR')
                self.vars = self.pred_vars
                self.means = self.pred_means
                #### (1) E-Step
                l = torch.cat([self.likelihood(x, self.means[i], self.vars[i]).view(1, -1) for i in range(self.k)], dim=0)
                #### (2) M-Step
                r = (l * self.pi.view(-1, 1))
                r = r / (r.sum(dim=0))
                self.means = torch.cat([((r[i].unsqueeze(-1) * x).sum(dim=0) / (r[i].sum() + self.eps)).unsqueeze(0) for i in range(self.k)], dim=0)
                self.continuous_fitting_runs +=1

            except ValueError:
                print('ERR')
                self.vars = self.pred_vars
                #### (1) E-Step
                l = torch.cat([self.likelihood(x, self.means[i], self.vars[i]).view(1, -1) for i in range(self.k)], dim=0)
                #### (2) M-Step
                r = (l * self.pi.view(-1, 1))
                r = r / (r.sum(dim=0))
                self.means = torch.cat([((r[i].unsqueeze(-1) * x).sum(dim=0) / (r[i].sum() + self.eps)).unsqueeze(0) for i in range(self.k)], dim=0)
                self.continuous_fitting_runs += 1

    def predict(self, x):
        return torch.cat([self.likelihood(x, self.pred_means[i], self.pred_vars[i]).view(1, -1) for i in range(self.k)], dim=0)