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
print(len(df), list(df))


G = df['gen.bin'].values+1
R = df['col.bin'].values+1
C = df['cor.bin'].values+1
#N = df['n'].values.astype(int)
k = df['fatalities'].values.astype(int) + 1
N = k.max()

print(len(G))
data = {
        'N': N,
        'k': k,
        'gen': G, 'G': len(np.unique(G)),
        'rul': R, 'R': len(np.unique(R)),
        #'cor': C, 'C': len(np.unique(C)),
        'COUNTRIES': len(G)
        }





#####REDO MODEL BUT USING BINOMIAL TO ESTIMATE RATE USING N for N and FATAILITES for K
#dbin has two parameters--rate and N-number

mBin = """
model {
        for (country in 1:COUNTRIES){
                k[country] ~ dpois(theta[rul[country],cor[country],gen[country]])
        }
        
        for (r in 1:R){
                for (c in 1:C){
                        for (g in 1:G){
                                theta[r,c,g] ~ dunif(0,N)
                        }
                
                }
        
        }
        
}

"""

mBin2 = """
model {
        for (country in 1:COUNTRIES){
                k[country] ~ dpois(theta[rul[country],gen[country]])
        }

        for (r in 1:R){
                for (g in 1:G){
                        theta[r,g] ~ dunif(0,N)
                }
        }

}

"""

mBinH = """
model {
        for (country in 1:COUNTRIES){
                k[country] ~ dpois(theta[rul[country],gen[country]])
        }
        
        for (r in 1:R){
                RN[r] ~ dnorm(F, Rsigma[r])
                Rsigma[r] ~ dunif(0,100)
        }

        for (r in 1:R){
                for (g in 1:G){
                        theta[r,g] ~ dnorm(RN[r], Gsigma[r,g])
                        Gsigma[r,g] ~ dunif(0,100)
                }
        }
        
        sigma ~ dunif(0,100)
        fmu ~ dunif(0,N)
        F ~ dnorm(fmu, sigma)

}

"""



Nsamples = 1000
chains = 3
burn_in = 1000

# varnames = [#'f', 'GEN',
#             'CUR', 'ypred',
#             'gmu', 'cmu', 'dmu',
#             'incidents']
#varnames = ['cN', 'cmu', 'csig', 'ypred', 'incidents']
varnames = ['theta']

model = pyjags.Model(mBinH, data=data, chains=chains)

model.sample(burn_in, vars=[])

samples = model.sample(Nsamples, vars=varnames)
for i in varnames:
        x = samples[i].reshape(*samples[i].shape[:-2], samples[i].shape[-2]*samples[i].shape[-1])
        np.save(partial_path+i+'.npy', x, allow_pickle=False)

        print(i, x.shape)

#samples = {i: samples[i].reshape(*samples[i].shape[:-2], samples[i].shape[-2]*samples[i].shape[-1]) for i in varnames}
