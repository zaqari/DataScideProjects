import numpy as np

def sel(terms,IDX):
    return (IDX == np.array(terms).reshape(-1,1)).sum(axis=0).astype(np.bool)
