import numpy as np
import pandas as pd
import pyjags
import os, sys

partial_path = '/home/zaq/d/AFC/'
data_path = partial_path+'data/'

if __name__ == '__main__':
        args = sys.argv
        for arg in args:
                if '-d=' in arg:
                        data_path+=arg.split('=')[-1]

df = pd.read_csv(data_path)

G = df['gen.bin'].values
D = df['cur'].values
C = df['cor.bin'].values
Y = df['n'].values

data = {
        'y': Y,
        'gen': G,
        'cur': D,
        'cor': C,
        'G': len(np.unique(G)),
        'C': len(np.unique(C)),
        'D': len(np.unique(D)),
        'COUNTRIES': len(G)
        }


#All factors are independent
m = """
model {
        for (country in 1:COUNTRIES){
                y[country] ~ (COR[cor[country]]*incidents) + (GEN[gen[country]]*incidents) + (CUR[cur[country]]*incidents)
        }
        
        for (genlevel in 1:G){
                GEN[genlevel] ~ dnorm(gmu,gsig)
                gmu[genlevel] ~ dunif(0,100)
                gsig[genlevel] ~ dunif(0,100)
        }
        
        for (currency in 1:D){
                CUR[currency] ~ dnorm(dmu,dsig)
                dmu[currency] ~ dunif(0,100)
                dsig[currency] ~ dunif(0,100)
        }
        
        for (corlevel in 1:C){
                COR[corlevel] ~ dnorm(cmu,csig)
                cmu[corlevel] ~ dunif(0,100)
                csig[corlevel] ~ dunif(0,100)
        }
        
        
        for (country in 1:COUNTRIES){
                ypred[country] = (COR[cor[country]]*incidents) + (GEN[gen[country]]*incidents) + (CUR[cur[country]]*incidents)
        }
        
        incidents ~ dunif(0,1000)
        
}
"""

# All factors are hierarchical/Decision Tree
h = """
model {
        for (country in 1:COUNTRIES){
                y[country] ~ f[cur[country],gen[country],cor[country]]*incidents
        }
        
        for (genlevel in 1:G){
                gsig[genlevel] ~ dunif(0,100)
        }
        
        for (corlevel in 1:C){
                csig[corlevel] ~ dunif(0,100)
        }
        
        for (currency in 1:D){
                CUR[currency] ~ dnorm(dmu,dsig)
                dmu[currency] ~ dunif(0,100)
                dsig[currency] ~ dunif(0,100)
                
                for (genlevel in 1:G){
                        GEN[currency, genlevel] ~ dnorm(CUR[currency],gsig[genlevel])
                        
                        for (corlevel in 1:C){
                                f[currency,genlevel,corlevel] ~ dnorm(GEN[currency,genlevel],csig[corlevel])
                        }
                        
                }
                
        }

        for (country in 1:COUNTRIES){
                ypred[country] ~ f[cur[country],gen[country],cor[country]]*incidents
        }

        incidents ~ dunif(0,1000)

}
"""

Nsamples = 5000
chains = 3
burn_in = 1000

varnames = ['COR', 'GEN', 'CUR', 'ypred']

model = pyjags.Model(m, data=data, chains=chains)

model.sample(burn_in, vars=[])

samples = model.sample(Nsamples, vars=varnames)
for i in varnames:
        x = samples[i].reshape(*samples[i].shape[:-2], samples[i].shape[-2]*samples[i].shape[-1])
        np.save(partial_path+i+'.npy', x, allow_pickle=False)

        print(i, x.shape)

#samples = {i: samples[i].reshape(*samples[i].shape[:-2], samples[i].shape[-2]*samples[i].shape[-1]) for i in varnames}
