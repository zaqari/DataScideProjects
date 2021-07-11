from CommClusters.mod.sim_matrix import *

class bcl():

    def __init__(self, P, df):
        super(bcl,self).__init__()

        self.P, self.df = P, df
        self.labels_ = np.array([np.nan for _ in range(len(self.df))])

    def fit(self, x, Nseed):
        denoms = []

