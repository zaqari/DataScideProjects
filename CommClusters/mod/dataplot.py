import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_data_square(x,labels,pal=None,metric='euclidean'):
    dfi = pd.DataFrame(x,columns=labels)
    dfi['L'] = labels
    dfi = dfi.set_index('L')
    if pal:
        sns.clustermap(dfi,cmap=pal,metric=metric)
    else:
        sns.clustermap(dfi)
    plt.show()

def plot_data(x,column_names,row_names,pal=None,metric='euclidean',verbose=False):
    dfi = pd.DataFrame(x,columns=column_names)
    dfi['L'] = row_names
    dfi = dfi.set_index('L')

    g = None

    if pal:
        g = sns.clustermap(dfi,cmap=pal,metric=metric)

    else:
        g = sns.clustermap(dfi,metric=metric)

    plt.show()

    if verbose:
        return g,dfi


def sel(terms,IDX):
    return (IDX == np.array(terms).reshape(-1,1)).sum(axis=0).astype(np.bool)